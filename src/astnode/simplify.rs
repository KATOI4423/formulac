//! # astnode/simplify.rs
//!
//! Constant folding and algebraic simplification of [`AstNode`] expressions.
//!
//! ## Responsibilities
//! - Evaluating constant subexpressions at compile time (constant folding)
//! - Flattening and combining like terms in additions and multiplications
//! - Simplifying powers: `x^0 → 1`, `x^1 → x`, `(x^a)^b → x^(a*b)`
//! - Rewriting subtraction and division in terms of addition and multiplication
//!
//! ## Entry point
//! The public API is [`AstNode::simplify`], which recursively applies
//! all of the above transformations.

use num_complex::Complex;
use num_traits::{
    One,
    Zero,
};
use std::ops::{
    AddAssign,
    MulAssign,
};
use std::rc::Rc;

use crate::astnode::AstNode;
use crate::core::{
    ComplexMath,
    Real,
};
use crate::functions::{
    FunctionCall,
    FunctionKind,
};
use crate::lexer::Span;
use crate::operators::{
    BinaryOperatorKind,
    UnaryOperatorKind,
};

/// One group of terms of an addition that share the same symbolic factors.
struct LikeTermGroup<T: Real> {
    /// Symbolic factors in canonical order (numeric factors removed).
    key: Vec<AstNode<T>>,
    /// Product of the numeric factors, summed over the group.
    coeff: Complex<T>,
    /// Only kept for terms that were split; a single-factor term is its own key.
    original: Option<AstNode<T>>,
    merged: bool,
}

impl<T: Real> AstNode<T> {
    /// Simplifies the AST by constant-folding and algebraic normalization.
    pub(crate) fn simplify(self) -> Self
    where
        Complex<T>: AddAssign + MulAssign,
    {
        match self {
            Self::UnaryOperator { kind, expr, span } => {
                Self::fold_unary(kind, span, Self::unwrap_rc(expr))
            }
            Self::BinaryOperator { kind, left, right, span } => {
                Self::fold_binary(kind, span, (*left).clone(), (*right).clone())
            }
            // pow/powi get their own folding path.
            Self::FunctionCall { kind: FunctionKind::Pow  | FunctionKind::Powi, mut args, .. } => {
                let base = Self::unwrap_rc(args.remove(0));
                let exp  = Self::unwrap_rc(args.remove(0));
                Self::fold_pow(base, exp)
            }
            Self::FunctionCall { kind, args, span } => {
                Self::fold_generic_fn(kind, span, args, |k, s, a| Self::FunctionCall { kind: k, args: a, span: s })
            }
            Self::UserFunctionCall { func, args, span } => {
                Self::fold_generic_fn(func, span, args, |f, s, a| Self::UserFunctionCall { func: f, args: a, span: s })
            }
            other => other,
        }
    }

    // ── fold helpers ─────────────────────────────────────────────────────────

    /// Evaluates a function call if all arguments are numeric literals.
    fn fold_generic_fn<F, G>(func: F, span: Span, args: Vec<Rc<Self>>, make_node: G) -> Self
    where
        F: FunctionCall<T>,
        G: FnOnce(F,Span, Vec<Rc<Self>>) -> Self,
        Complex<T>: AddAssign + MulAssign,
    {
        let args: Vec<_> = args.into_iter().map(|arg| {
            let ast = Self::unwrap_rc(arg);
            Rc::new(Self::simplify(ast))
        }).collect();
        let all_numbers  = args.iter().all(|a| matches!(**a, Self::Number { .. }));
        if all_numbers {
            let nums: Vec<Complex<T>> = args.iter()
                .map(|a|
                    match a.as_ref() {
                        Self::Number { value, .. } => value.clone(),
                        _ => unreachable!()
                    }
                )
                .collect();
            Self::Number { value: func.apply(nums), span }
        } else {
            make_node(func, span, args)
        }
    }

    fn fold_unary(kind: UnaryOperatorKind, span: Span, expr: Self) -> Self
    where
        Complex<T>: AddAssign + MulAssign,
    {
        let expr = expr.simplify();

        // Assume that `UnaryOperatorKind` takes only `Positive` and `Negative`.
        if kind == UnaryOperatorKind::Positive {
            return expr;
        }

        // else => kind == unaryOperatorKind::Negative
        match expr {
            Self::Number { value, span } => Self::Number { value: kind.apply(value), span },
            other => (-Self::one(span) * other).simplify(), // goto fold_mul(span, -1, other)
        }
    }

    fn fold_binary(kind: BinaryOperatorKind, span: Span, left: Self, right: Self) -> Self
    where
        Complex<T>: AddAssign + MulAssign,
    {
        let left  = left.simplify();
        let right = right.simplify();

        // Both sides are numbers → evaluate immediately.
        if let (Self::Number { value: l, span: ls }, Self::Number { value: r, .. }) = (&left, &right) {
            // Use powi instead of Pow::apply (powc) to avoid rounding errors.
            if kind == BinaryOperatorKind::Pow && Self::is_i32_compatible(r) {
                return Self::Number { value: l.powi(r.re.to_i32()), span: *ls }
            }
            return Self::Number { value: kind.apply(l.clone(), r.clone()), span: *ls };
        }

        match kind {
            BinaryOperatorKind::Add => Self::fold_add(span, left, right),
            BinaryOperatorKind::Sub => Self::fold_sub(span, left, right),
            BinaryOperatorKind::Mul => Self::fold_mul(span, left, right),
            BinaryOperatorKind::Div => Self::fold_div(span, left, right),
            BinaryOperatorKind::Pow => Self::fold_pow(left, right),
        }
    }

    // ── addition ─────────────────────────────────────────────────────────────

    /// `left + right` with flattening, constant-folding, and like-term combining.
    fn fold_add(span: Span, left: Self, right: Self) -> Self
    where
        Complex<T>: AddAssign,
    {
        // 1. Flatten nested additions into a single term list.
        let mut terms = Vec::new();
        Self::collect_add_terms(left,  &mut terms);
        Self::collect_add_terms(right, &mut terms);

        // 2. Separate numeric constants from symbolic terms.
        let mut const_sum = Complex::zero();
        let mut sym_terms = Vec::new();
        for t in terms {
            match t {
                Self::Number { value, .. } => const_sum += value,
                other           => sym_terms.push(other),
            }
        }
        if !const_sum.is_zero() {
            sym_terms.push(Self::Number { value: const_sum, span });
        }

        // 3. Combine like terms (e.g. x + x → 2*x).
        let terms = Self::combine_like_add_terms(sym_terms);

        // 4. Fold back into a left-associative chain.
        Self::chain_add(terms, span)
    }

    pub(crate) fn collect_add_terms(node: Self, out: &mut Vec<Self>) {
        match node {
            Self::BinaryOperator { kind: BinaryOperatorKind::Add, left, right, .. } => {
                Self::collect_add_terms(
                    Self::unwrap_rc(left),
                    out,
                );
                Self::collect_add_terms(
                    Self::unwrap_rc(right),
                    out,
                );
            }
            other => out.push(other),
        }
    }

    /// Splits a term into (symbolic factors in canonical order, numeric coefficient).
    ///
    /// `3*z*sin(z)`, `z*3*sin(z)` and `sin(z)*(3*z)` all give `([z, sin(z)], 3)`.
    fn split_coefficient(self) -> (Vec<Self>, Complex<T>) {
        let mut factors = Vec::new();
        Self::collect_mul_terms(self, &mut factors);

        let mut coeff = Complex::one();
        let mut key = Vec::new();
        for f in factors {
            match f {
                Self::Number { value, .. } => coeff = coeff * value,
                other => key.push(other),
            }
        }

        key.sort_by(Self::structural_cmp);
        (key, coeff)
    }

    fn same_factors(lhs: &[Self], rhs: &[Self]) -> bool {
        lhs.len() == rhs.len()
            && lhs.iter().zip(rhs)
                .all(|(l, r)| l.structurally_eq(&r))
    }

    /// Combines `a*X + b*X → (a+b)*X`, where `X` is any product of symbolic
    /// factors (in any order), and removes zero-coefficient terms.
    ///
    /// Only the *numeric* coefficient is merged. Terms that share a common
    /// symbolic factor but differ in a symbolic one, such as
    /// `a*(x+y) + b*(x+y)`, are deliberately **not** factored into
    /// `(a+b)*(x+y)`, for these reasons:
    ///
    /// - The result is not well defined. With `x*y + x*z + y*z` there are
    ///   several common factors to extract, and a greedy choice depends on term
    ///   order. For `x*y*sin(z) + x*sin(z)` it is unclear whether to extract
    ///   `x`, `sin(z)` or both.
    /// - It can break the shapes that later passes rely on. Factoring `x` out
    ///   of `y*x^2 + a1*x` hides the polynomial from `horner()`, and factoring
    ///   out of `A*sin(u) + B*cos(u)` removes the `sin`/`cos` pair that the
    ///   `SinCos` fusion in `compile()` looks for.
    /// - Floating-point behavior changes. `a*K + b*K` and `(a+b)*K` round
    ///   differently and have different overflow boundaries: with `a = b = 1e308`
    ///   and `K = 0.5` the former is finite while `a+b` overflows first in the
    ///   latter.
    ///
    /// A term that has no like partner is returned unchanged, so the shape of
    /// terms that were not merged does not depend on this function.
    fn combine_like_add_terms(terms: Vec<Self>) -> Vec<Self>
    where
        Complex<T>: AddAssign,
    {
        let mut groups: Vec<LikeTermGroup<T>> = Vec::new();

        for term in terms {
            let (key, coeff, original) = match &term {
                Self::BinaryOperator { kind: BinaryOperatorKind::Mul, .. } | Self::Number { .. } => {
                    let original = term.clone();
                    let (key, coeff) = term.split_coefficient();
                    (key, coeff, Some(original))
                },
                // A lone symbolic factor: no splitting and no clone needed.
                _ => (vec![term], Complex::one(), None),
            };

            match groups.iter_mut().find(|g| Self::same_factors(&g.key, &key)) {
                Some(g) => {
                    g.coeff += coeff;
                    g.merged = true;
                },
                None => groups.push(LikeTermGroup { key, coeff, original, merged: false }),
            }
        }

        groups.into_iter()
            .filter(|g| !g.coeff.is_zero())
            .map(|g| {
                if !g.merged {
                    // Unmerged: hand back the term as it was written
                    return g.original.unwrap_or_else(|| g.key.into_iter().next().unwrap());
                }
                let span = g.original.as_ref()
                    .map_or_else(|| g.key[0].span(), |o| o.span());
                if g.key.is_empty() {
                    return Self::Number { value: g.coeff, span };
                }
                // ((c * a) * b) ..., the same left-nested shape as `fold_mul` produces.
                let coeff = (!g.coeff.is_one()).then(|| Self::Number { value: g.coeff, span });
                coeff.into_iter().chain(g.key).reduce(Self::mul).unwrap()
            }).collect()
    }

    fn chain_add(terms: Vec<Self>, span: Span) -> Self {
        match terms.len() {
            0 => Self::zero(span),
            1 => terms.into_iter().next().unwrap(),
            _ => terms.into_iter().reduce(|acc, t| acc.add(t)).unwrap(),
        }
    }

    // ── subtraction ──────────────────────────────────────────────────────────

    /// Rewrites `left - right` as `left + (-1)*right` and re-folds.
    fn fold_sub(span: Span, left: Self, right: Self) -> Self
    where
        Complex<T>: AddAssign + MulAssign,
    {
        Self::fold_binary(
            BinaryOperatorKind::Add,
            span,
            left,
            Self::Number { value: -Complex::one(), span: right.span() } * right,
        )
    }

    // ── multiplication ───────────────────────────────────────────────────────

    /// `left * right` with flattening, constant-folding, and same-base power combining.
    fn fold_mul(span: Span, left: Self, right: Self) -> Self
    where
        Complex<T>: AddAssign + MulAssign,
    {
        // 1. Flatten nested multiplications.
        let mut factors = Vec::new();
        Self::collect_mul_terms(left,  &mut factors);
        Self::collect_mul_terms(right, &mut factors);

        // 2. Pull out numeric constants.
        let mut const_prod = Complex::one();
        let mut sym_factors = Vec::new();
        for f in factors {
            match f {
                Self::Number { value: z, .. } => const_prod *= z,
                other => sym_factors.push(other),
            }
        }

        if const_prod.is_zero() {
            return Self::zero(span);
        }
        if !const_prod.is_one() {
            sym_factors.insert(0, Self::Number { value: const_prod, span });
        }

        // 3. Combine x^a * x^b → x^(a+b).
        let factors = Self::combine_like_pow_terms(sym_factors);

        // 4. Fold back.
        Self::chain_mul(factors, span)
    }

    pub(crate) fn collect_mul_terms(node: Self, out: &mut Vec<Self>) {
        match node {
            Self::BinaryOperator { kind: BinaryOperatorKind::Mul, left, right, .. } => {
                let left = Self::unwrap_rc(left);
                let right = Self::unwrap_rc(right);
                Self::collect_mul_terms(left,  out);
                Self::collect_mul_terms(right, out);
            }
            other => out.push(other),
        }
    }

    /// Combines `x^a * x^b → x^(a+b)`, removes zero-exponent terms.
    fn combine_like_pow_terms(terms: Vec<Self>) -> Vec<Self>
    where
        Complex<T>: AddAssign,
    {
        let mut map: Vec<(Self, Complex<T>)> = Vec::new();

        for term in terms {
            let (base, exp) = match term {
                Self::BinaryOperator { kind: BinaryOperatorKind::Pow, left, right, .. } => {
                    let left = Self::unwrap_rc(left);
                    let right = Self::unwrap_rc(right);
                    match right {
                        Self::Number { value: e, .. } => (left, e),
                        r => {
                            (left.pow(r), Complex::one()
                        )},
                    }
                }
                Self::FunctionCall { kind: FunctionKind::Pow | FunctionKind::Powi, ref args, .. } => {
                    let base = Self::unwrap_rc(args[0].clone());
                    match args[1].as_ref() {
                        Self::Number { value: e, .. } => (base, e.clone()),
                        _ => (term, Complex::one()),
                    }
                }
                other => (other, Complex::one()),
            };
            match map.iter_mut().find(|(b, _)| b.structurally_eq(&base)) {
                Some((_, e)) => *e += exp,
                None         => map.push((base, exp)),
            }
        }

        map.into_iter()
            .filter(|(_, e)| !(*e).is_zero())
            .map(|(base, exp)| {
                if exp.is_one() { base }
                else if Self::is_i32_compatible(&exp) { base.powi(exp.re.to_i32()) }
                else {
                    let span= base.span();
                    base.pow(Self::Number { value: exp, span })
                }
            })
            .collect()
    }

    fn chain_mul(factors: Vec<Self>, span: Span) -> Self
    where
        Complex<T>: AddAssign + MulAssign,
    {
        match factors.len() {
            0 => Self::one(span),
            1 => factors.into_iter().next().unwrap().simplify(),
            _ => factors.into_iter().reduce(|acc, f| acc.mul(f)).unwrap(),
        }
    }

    // ── division ─────────────────────────────────────────────────────────────

    /// Rewrites `left / right` as `left * right^-1` and re-folds.
    fn fold_div(span: Span, left: Self, right: Self) -> Self
    where
        Complex<T>: AddAssign + MulAssign,
    {
        Self::fold_mul(span, left, right.powi(-1).simplify())
    }

    // ── power ────────────────────────────────────────────────────────────────

    fn fold_pow(base: Self, exp: Self) -> Self
    where
        Complex<T>: AddAssign + MulAssign,
    {
        let mut base = base.simplify();
        let mut exp  = exp.simplify();

        // (x^a)^b → x^(a*b) is valid only when b is an i32-compatible integer
        loop {
            if !matches!(&exp, Self::Number { value, .. } if Self::is_i32_compatible(value)) {
                break;
            }

            match base {
                Self::FunctionCall { kind: FunctionKind::Pow | FunctionKind::Powi, mut args, .. } => {
                    let inner_base = Self::unwrap_rc(args.remove(0));
                    let inner_exp  = Self::unwrap_rc(args.remove(0));
                    exp  = inner_exp.mul(exp).simplify();
                    base = inner_base.simplify();
                }
                other => { base = other; break; }
            }
        }

        // x^1 → x, x^0 → 1
        if let Self::Number { value: e, span: s } = &exp {
            if (*e).is_one()  { return base; }
            if (*e).is_zero() { return Self::one(*s); }
        }

        match (base, exp) {
            (Self::Number { value: b, span: s }, _) if b.is_one() => Self::one(s),
            (Self::Number { value: b, span: s }, Self::Number { value: e, .. })
                if b.is_zero() && e.re > T::zero()
                => Self::zero(s),
            (Self::Number { value: b, span: s }, Self::Number { value: e, .. })
                =>
            {
                // Use powi instead of powc to avoid rounding errors.
                if Self::is_i32_compatible(&e) {
                    Self::Number { value: b.powi(e.re.to_i32()), span: s }
                } else {
                    Self::Number { value: b.powc(e), span: s }
                }
            },
            (b, Self::Number { value: e, .. }) if Self::is_i32_compatible(&e)
                => b.powi(e.re.to_i32()),
            (b, e) => {
                b.pow(e)
            },
        }
    }
}

#[cfg(test)]
mod fold_tests {
    use super::*;
    use crate::functions::UserFn;
    use approx::assert_abs_diff_eq;

    macro_rules! assert_astnode_eq {
        ($left:expr, $right:expr) => {{
            fn inner<T: Real>(left: &AstNode<T>, right: &AstNode<T>) {
                let epsilon = 1.0e-12;
                match (left, right) {
                    (AstNode::Number { value: lv, span: ls }, AstNode::Number { value: rv, span: rs }) => {
                        assert!((lv.re.clone() - rv.re.clone()).abs() < T::from_f64(epsilon));
                        assert!((lv.im.clone() - rv.im.clone()).abs() < T::from_f64(epsilon));
                        assert_eq!(ls, rs);
                    }
                    (AstNode::Argument { index: li, span: ls }, AstNode::Argument { index: ri, span: rs }) => {
                        assert_eq!(li, ri);
                        assert_eq!(ls, rs);
                    }
                    (AstNode::UnaryOperator { kind: lk, expr: le, span: ls }, AstNode::UnaryOperator { kind: rk, expr: re, span: rs }) => {
                        assert_eq!(lk, rk);
                        inner(le, re);
                        assert_eq!(ls, rs);
                    }
                    (AstNode::BinaryOperator { kind: lk, left: ll, right: lr, span: ls },
                    AstNode::BinaryOperator { kind: rk, left: rl, right: rr, span: rs }) => {
                        assert_eq!(lk, rk);
                        inner(ll, rl);
                        inner(lr, rr);
                        assert_eq!(ls, rs);
                    }
                    (AstNode::FunctionCall { kind: lk, args: la, span: ls },
                    AstNode::FunctionCall { kind: rk, args: ra, span: rs }) => {
                        assert_eq!(lk, rk);
                        assert_eq!(la.len(), ra.len());
                        for (a, b) in la.iter().zip(ra.iter()) {
                            inner(a, b);
                        }
                        assert_eq!(ls, rs);
                    }
                    (l, r) => panic!("AST nodes differ: left = {:?}, right = {:?}", l, r),
                }
            }
            inner(&$left, &$right);
        }};
    }

    fn c(re: f64, im: f64) -> Complex<f64> {
        Complex::new(re, im)
    }

    fn x_at(start: usize) -> AstNode<f64> {
        AstNode::Argument { index: 0, span: Span::from(start..start + 1) }
    }

    fn num_at(v: f64, start: usize) -> AstNode<f64> {
        AstNode::Number { value: Complex::new(v, 0.0), span: Span::from(start..start + 1) }
    }

    /// (leading numeric coefficient, number of symbolic factors) of a product chain.
    fn coeff_and_factors(node: AstNode<f64>) -> (Option<f64>, usize) {
        let mut f = Vec::new();
        AstNode::collect_mul_terms(node, &mut f);
        match f.first() {
            Some(AstNode::Number { value, .. }) => (Some(value.re), f.len() - 1),
            _ => (None, f.len()),
        }
    }

    #[test]
    fn fold_add_constants() {
        let left = AstNode::Number { value: Complex::from(2.0), span: Span::from(0..1) };
        let right = AstNode::Number { value: Complex::from(3.0), span: Span::from(1..2) };
        let result = AstNode::fold_add(left.span(), left, right);
        assert_astnode_eq!(result, AstNode::Number { value: Complex::from(5.0), span: Span::from(0..1) });
    }

    #[test]
    fn fold_add_like_terms() {
        let x1 = AstNode::<f64>::Argument { index: 0, span: Span::from(0..1) };
        let x2 = AstNode::<f64>::Argument { index: 0, span: Span::from(2..3) };
        let result = AstNode::fold_add(Span::from(1..2), x1.clone(), x2.clone());
        match result {
            AstNode::BinaryOperator { kind, left, right, .. } => {
                assert_eq!(kind, BinaryOperatorKind::Mul);
                match (&*left, &*right) {
                    (AstNode::Number { value, .. }, AstNode::Argument { index, .. })
                    | (AstNode::Argument { index, .. }, AstNode::Number { value, .. }) => {
                        assert_abs_diff_eq!(value.re, 2.0, epsilon = 1.0e-12);
                        assert_eq!(*index, 0);
                    }
                    _ => panic!("Expected 2 * x or x * 2"),
                }
            }
            _ => panic!("Expected multiplication result"),
        }
    }

    #[test]
    fn fold_add_mixed_terms() {
        // 3*x + 4*x + y → 7*x + y
        let x1 = AstNode::Argument { index: 0, span: Span::from(2..3) };
        let x2 = AstNode::Argument { index: 0, span: Span::from(8..9) };
        let y = AstNode::Argument { index: 1, span: Span::from(12..13)};
        let term1 = AstNode::Number { value: Complex::new(3.0, 0.0), span: Span::from(0..1) }.mul(x1.clone());
        let term2 = AstNode::Number { value: Complex::new(4.0, 0.0), span: Span::from(6..7) }.mul(x2);
        let left = AstNode::fold_add(Span::from(4..5), term1, term2); // 3x + 4x → 7x
        let result = AstNode::fold_add(Span::from(10..11), left, y.clone());
        match result {
            AstNode::BinaryOperator { kind, left, right, .. } => {
                assert_eq!(kind, BinaryOperatorKind::Add);
                match Rc::try_unwrap(left).unwrap() {
                    AstNode::BinaryOperator { kind, left, right, .. } => {
                        assert_eq!(kind, BinaryOperatorKind::Mul);
                        match *left {
                            AstNode::Number { value, .. } => assert!((value.re - 7.0) < 1.0e-12),
                            _ => panic!("Expected 7.0"),
                        };
                        match *right {
                            AstNode::Argument { index, .. } => assert_eq!(index, 0),
                            _ => panic!("Expected 0"),
                        }
                    },
                    _ => panic!("Expected BinaryOperator Mul"),
                };
                match Rc::try_unwrap(right).unwrap() {
                    AstNode::Argument { index, .. } => assert_eq!(index, 1),
                    _ => panic!("Expected 1"),
                }
            }
            _ => panic!("Expected BinaryOperator Add"),
        }
    }

    #[test]
    fn fold_add_removed_same_terms() {
        // x + 2 - x = 2
        let x = AstNode::Argument { index: 0, span: Span::from(0..1) };
        let n = AstNode::Number { value: Complex::from(2.0), span: Span::from(2..3) };
        let result = AstNode::fold_sub(
            Span::from(3..4),
            AstNode::fold_add(Span::from(1..2), x.clone(), n.clone()),
            x,
        );
        // The result should simplify to just the constant 2.0
        // The span of the result may be derived from the fold process
        match result {
            AstNode::Number { value, .. } => {
                assert_abs_diff_eq!(value.re, 2.0, epsilon = 1.0e-12);
            }
            _ => panic!("Expected simplified to Number"),
        }

        // -x + x = 0
        let x = AstNode::Argument { index: 0, span: Span::from(0..1) };
        let result = -x.clone() + x;
        assert_eq!(result.simplify(), AstNode::<f64>::zero(Span::from(0..1)));
    }

    #[test]
    fn fold_add_zero_terms() {
        // x + 0 → x
        let x = AstNode::Argument { index: 0, span: Span::from(0..1) };
        let zero = AstNode::Number { value: Complex::from(0.0), span: Span::from(2..3) };
        let result = AstNode::fold_add(Span::from(1..2), x.clone(), zero);
        assert_astnode_eq!(result, x);
    }

    #[test]
    fn like_terms_with_two_symbolic_factors_cancel() {
        // 3*x*sin(x) - 3*x*sin(x) = 0
        let t1 = num_at(3.0, 0).mul(x_at(2)).mul(x_at(4).sin());
        let t2 = num_at(3.0, 10).mul(x_at(12)).mul(x_at(14).sin());
        let r = (t1 - t2).simplify();
        assert!(matches!(r, AstNode::Number { value, .. } if value.is_zero()), "{r:?}");
    }

    #[test]
    fn like_terms_with_three_symbolic_factors_merge() {
        // exp(x)*sin(x)*x + 2*exp(x)*sin(x)*x = 3*(...)
        let t1 = x_at(0).exp().mul(x_at(10).sin()).mul(x_at(20));
        let t2 = num_at(2.0, 30).mul(x_at(32).exp()).mul(x_at(40).sin()).mul(x_at(50));
        let r = (t1 + t2).simplify();
        assert_eq!(coeff_and_factors(r), (Some(3.0), 3));
    }

    #[test]
    fn like_terms_ignore_factor_order() {
        // x*sin(x) + sin(x)*x = 2*x*sin(x)
        let r = (x_at(0).mul(x_at(2).sin()) + x_at(10).sin().mul(x_at(14))).simplify();
        assert_eq!(coeff_and_factors(r), (Some(2.0), 2));
    }

    #[test]
    fn unmerged_term_keeps_its_shape() {
        // sin(x)*x has no partner: factor order is left as written.
        let t = x_at(0).sin().mul(x_at(8)) + num_at(1.0, 20).mul(x_at(22).cos());
        let r = t.simplify();
        let AstNode::BinaryOperator { left, .. } = &r else { panic!("{r:?}") };
        assert!(matches!(left.as_ref(), AstNode::BinaryOperator { left: l, .. }
            if matches!(l.as_ref(), AstNode::FunctionCall { kind: FunctionKind::Sin, .. })));
    }

    #[test]
    fn fold_sub_basic() {
        // x - y → x + (-1) * y
        let x = AstNode::Argument { index: 0, span: Span::from(0..1) };
        let y = AstNode::<f64>::Argument { index: 1, span: Span::from(2..3) };
        let result = AstNode::fold_sub(Span::from(1..2), x.clone(), y.clone());
        match result {
            AstNode::BinaryOperator { kind, left, right, .. } => {
                assert_eq!(kind, BinaryOperatorKind::Add);
                match (&*left, &*right) {
                    (AstNode::Argument { index: xi, .. }, AstNode::BinaryOperator { kind: mk, left: ml, right: mr, .. }) => {
                        assert_eq!(xi, &0);
                        assert_eq!(mk, &BinaryOperatorKind::Mul);
                        match (&**ml, &**mr) {
                            (AstNode::Number { value, .. }, AstNode::Argument { index: yi, .. })
                            | (AstNode::Argument { index: yi, .. }, AstNode::Number { value, .. }) => {
                                assert_eq!(yi, &1);
                                assert_abs_diff_eq!(value.re, -1.0, epsilon = 1.0e-12);
                            }
                            _ => panic!("Expected -1 * y structure"),
                        }
                    }
                    _ => panic!("Expected x + (-1 * y)"),
                }
            }
            _ => panic!("Expected Add node"),
        }
    }

    #[test]
    fn fold_sub_with_constants() {
        // 5 - 3 → 2
        let left = AstNode::Number { value: Complex::new(5.0, 0.0), span: Span::from(0..1) };
        let right = AstNode::Number { value: Complex::new(3.0, 0.0), span: Span::from(4..5) };
        let result = AstNode::fold_sub(Span::from(2..3), left, right);
        match result {
            AstNode::Number { value, .. } => {
                assert_abs_diff_eq!(value.re, 2.0, epsilon = 1.0e-12);
            }
            _ => panic!("Expected Number"),
        }
    }

    #[test]
    fn fold_mul_constant_folding() {
        let expr = AstNode::fold_mul(
            Span::from(2..3),
            AstNode::Number { value: Complex::from(2.0), span: Span::from(0..1) },
            AstNode::Number { value: Complex::from(3.0), span: Span::from(4..5) }
        );
        match expr {
            AstNode::Number { value, .. } => {
                assert_abs_diff_eq!(value.re, 6.0, epsilon = 1.0e-12);
            }
            _ => panic!("Expected Number"),
        }
    }

    #[test]
    fn fold_mul_with_zero() {
        let expr = AstNode::fold_mul(
            Span::from(2..3),
            AstNode::Number { value: Complex::<f64>::zero(), span: Span::from(0..1) },
            AstNode::Argument { index: 0, span: Span::from(4..5) }
        );
        match expr {
            AstNode::Number { value, .. } => {
                assert!(value.is_zero());
            }
            _ => panic!("Expected Number"),
        }
    }

    #[test]
    fn fold_mul_with_one() {
        let expr = AstNode::fold_mul(
            Span::from(2..3),
            AstNode::Number { value: Complex::<f64>::one(), span: Span::from(0..1) },
            AstNode::Argument { index: 0, span: Span::from(4..5) }
        );
        assert_astnode_eq!(expr, AstNode::Argument { index: 0, span: Span::from(4..5) });
    }

    #[test]
    fn fold_div_to_mul_pow_neg1() {
        let expr = AstNode::fold_div(
            Span::from(2..3),
            AstNode::<f64>::Argument { index: 0, span: Span::from(0..1) },
            AstNode::<f64>::Argument { index: 1, span: Span::from(4..5) });
        // should become x * y^-1
        let expected = AstNode::<f64>::fold_mul(
            Span::from(2..3),
            AstNode::<f64>::Argument { index: 0, span: Span::from(0..1) },
            AstNode::<f64>::Argument { index: 1, span: Span::from(4..5) }.powi(-1));
        assert_astnode_eq!(expr, expected);
    }

    #[test]
    fn combine_same_base_powers() {
        // Use the same span for both arguments to ensure they're recognized as the same base
        let x = AstNode::Argument { index: 0, span: Span::from(0..1) };
        let expr = AstNode::fold_mul(
            Span::from(10..11),
            x.clone().pow(AstNode::Number { value: Complex::from(2.0), span: Span::from(4..7) }),
            x.clone().pow(AstNode::Number { value: Complex::from(3.5), span: Span::from(17..20) })
        );
        // x^2 * x^3.5 = x^5.5
        fn extract_exponent<T: Real>(node: &AstNode<T>) -> Option<T> {
            match node {
                AstNode::FunctionCall { kind, args, .. } if matches!(kind, FunctionKind::Pow | FunctionKind::Powi) => {
                    match args.get(1)?.as_ref() {
                        AstNode::Number { value, .. } => Some(value.re.clone()),
                        _ => None,
                    }
                }
                _ => None,
            }
        }

        if let Some(exp) = extract_exponent(&expr) {
            assert_abs_diff_eq!(exp, 5.5, epsilon = 1.0e-10);
        } else {
            panic!("Could not extract exponent from expression: {:?}", expr);
        }
    }

    #[test]
    fn combine_same_base_powers_to_powi() {
        // Use the same span for both arguments to ensure they're recognized as the same base
        let x = AstNode::Argument { index: 0, span: Span::from(0..1) };
        let expr = AstNode::fold_mul(
            Span::from(10..11),
            x.clone().pow(AstNode::Number { value: Complex::from(2.0), span: Span::from(4..7) }),
            x.clone().pow(AstNode::Number { value: Complex::from(3.0), span: Span::from(17..20) })
        );
        // x^2 * x^3 = x^5 which may be converted to powi(5)
        fn extract_exponent<T: Real>(node: &AstNode<T>) -> Option<T> {
            match node {
                AstNode::FunctionCall { kind, args, .. } if matches!(kind, FunctionKind::Pow | FunctionKind::Powi) => {
                    match args.get(1)?.as_ref() {
                        AstNode::Number { value, .. } => Some(value.re.clone()),
                        _ => None,
                    }
                }
                _ => None,
            }
        }

        if let Some(exp) = extract_exponent(&expr) {
            assert_abs_diff_eq!(exp, 5.0, epsilon = 1.0e-10);
        } else {
            panic!("Could not extract exponent from expression: {:?}", expr);
        }
    }

    #[test]
    fn fold_pow_test() {
        let span = Span::from(0..1);
        let cases = [
            // base, inner_exp, outer_exp, expected
            ( c(-1.0, 0.0), c(2.0, 0.0), c(0.5, 0.0), c(1.0, 0.0) ),
            ( c(-1.0, 0.0), c(0.5, 0.0), c(2.0, 0.0), c(-1.0, 0.0) ),
            ( c(-2.0, 1.0), c(2.0, 0.0), c(0.5, 0.0), c(2.0, -1.0) ),
            ( c(-1.0, 0.0), c(2.0, 0.0), c(-1.0, 0.0), c(1.0, 0.0) ),
        ];

        for (z, a, b, expected) in cases {
            let result = AstNode::Number { value: z, span }
                .pow(AstNode::Number { value: a, span })
                .pow(AstNode::Number { value: b, span })
                .simplify();
            assert_astnode_eq!(result, AstNode::Number { value: expected, span });
        }
    }

    #[test]
    fn fold_nested_pow_test() {
        let span = Span::from(0..1);
        let x = AstNode::Argument { index: 0, span };

        let result = x.clone().powi(2).powi(3)
            .pow(AstNode::Number { value: c(0.5, 0.0), span })
            .simplify();
        let expected = x.clone().powi(6).pow(AstNode::Number { value: c(0.5, 0.0), span });
        assert_eq!(result, expected);

        let result = x.clone().powi(2)
            .pow(AstNode::Number { value: c(0.5, 0.0), span })
            .powi(3).simplify();
        let expected = x.clone().powi(2)
            .pow(AstNode::Number { value: c(1.5, 0.0), span });
        assert_eq!(result, expected);
    }

    #[test]
    fn fold_pow_keeps_non_integer_outer_exponent() {
        let span = Span::from(0..1);
        let x = AstNode::Argument { index: 0, span };
        let result = x.clone()
            .pow(AstNode::Number {
                value: c(2.0, 0.0),
                span,
            })
            .pow(AstNode::Number {
                value: c(0.5, 0.0),
                span,
            }).simplify();
        let expected = x.powi(2)
            .pow(AstNode::Number { value: c(0.5, 0.0), span });

        // Ensure that the simplified AST preserves the nested power.
        assert_eq!(result, expected);
    }

    #[test]
    fn combine_same_base_ignores_spans() {
        // x * x with different spans -> powi(x, 2)
        let x1 = AstNode::<f64>::Argument { index: 0, span: Span::from(0..1) };
        let x2 = AstNode::<f64>::Argument { index: 0, span: Span::from(4..5) };
        let expr = AstNode::fold_mul(Span::from(2..3), x1, x2);
        match &expr {
            AstNode::FunctionCall { kind: FunctionKind::Powi, args, .. } => {
                assert!(matches!(args[0].as_ref(), AstNode::Argument { index: 0, .. }));
                assert!(matches!(args[1].as_ref(), AstNode::Number { value, .. } if value.re == 2.0));
            }
            other => panic!("Expected powi(x, 2), got {:?}", other),
        }
    }

    #[test]
    fn exp_times_exp_becomes_powi() {
        let e1 = AstNode::<f64>::Argument { index: 0, span: Span::from(4..5) }.exp();
        let e2 = AstNode::<f64>::Argument { index: 0, span: Span::from(12..13) }.exp();
        let expr = AstNode::fold_mul(Span::from(8..9), e1, e2);
        assert!(matches!(&expr, AstNode::FunctionCall { kind: FunctionKind::Powi, .. }));
    }

    #[test]
    fn nested_mul_flattening() {
        let expr = AstNode::fold_mul(
            Span::from(10..11),
            AstNode::fold_mul(
                Span::from(4..5),
                AstNode::Argument { index: 0, span: Span::from(0..1) },
                AstNode::Number { value: Complex::from(2.0), span: Span::from(6..7) }
            ),
            AstNode::Number { value: Complex::from(3.0), span: Span::from(12..13) }
        );
        // (x * 2) * 3 => 6 * x. Check structure rather than exact span.
        match expr {
            AstNode::BinaryOperator { kind, .. } => {
                assert_eq!(kind, BinaryOperatorKind::Mul);
            }
            _ => panic!("Expected BinaryOperator Mul"),
        }
    }

    #[test]
    fn simplify_number() {
        let node = AstNode::Number { value: Complex::new(3.0, 0.0), span: Span::from(0..1) };
        assert_astnode_eq!(node.clone().simplify(), node);
    }

    #[test]
    fn simplify_unary_operator() {
        let node = -AstNode::Number { value: Complex::new(2.0, 0.0), span: Span::from(2..3) };
        let simplified = node.simplify();
        assert_astnode_eq!(simplified, AstNode::Number { value: Complex::new(-2.0, 0.0), span: Span::from(2..3) });
    }

    #[test]
    fn simplify_binary_operator_full() {
        let node = AstNode::Number { value: Complex::new(2.0, 0.0), span: Span::from(0..1) } + AstNode::Number { value: Complex::new(3.0, 0.0), span: Span::from(4..5) };
        let simplified = node.simplify();
        assert_astnode_eq!(simplified, AstNode::Number { value: Complex::new(5.0, 0.0), span: Span::from(0..1) });
    }

    #[test]
    fn simplify_binary_operator_partial() {
        let node = AstNode::Argument { index: 0, span: Span::from(0..1) } + AstNode::Number { value: Complex::new(3.0, 0.0), span: Span::from(4..5) };
        let simplified = node.simplify();
        // The structure should remain similar (cannot simplify further)
        match simplified {
            AstNode::BinaryOperator { kind, .. } => {
                assert_eq!(kind, BinaryOperatorKind::Add);
            }
            _ => panic!("Expected BinaryOperator"),
        }
    }

    #[test]
    fn simplify_binary_operator_chain() {
        // x + 2 + 3 -> x + 5
        let node
            = AstNode::Argument { index: 0, span: Span::from(0..1) }
            + AstNode::Number { value: Complex::new(2.0, 0.0), span: Span::from(4..5) }
            + AstNode::Number { value: Complex::new(3.0, 0.0), span: Span::from(8..9) };
        let simplified = node.simplify();
        // Should be x + 5
        match simplified {
            AstNode::BinaryOperator { kind, .. } => {
                assert_eq!(kind, BinaryOperatorKind::Add);
            }
            _ => panic!("Expected BinaryOperator Add"),
        }

        // x * 2 * 3 * 4 -> 24 * x
        let node = AstNode::Argument { index: 0, span: Span::from(0..1) } * AstNode::Number { value: Complex::new(2.0, 0.0), span: Span::from(4..5) }
            * AstNode::Number { value: Complex::new(3.0, 0.0), span: Span::from(8..9) } * AstNode::Number { value: Complex::new(4.0, 0.0), span: Span::from(12..13) };
        let simplified = node.simplify();
        // Should be 24 * x
        match simplified {
            AstNode::BinaryOperator { kind, .. } => {
                assert_eq!(kind, BinaryOperatorKind::Mul);
            }
            _ => panic!("Expected BinaryOperator Mul"),
        }

        // 2 * x + 3 -> not changed
        let node = AstNode::Number { value: Complex::new(2.0, 0.0), span: Span::from(0..1) } * AstNode::Argument { index: 0, span: Span::from(4..5) } + AstNode::Number { value: Complex::new(3.0, 0.0), span: Span::from(8..9) };
        let simplified = node.clone().simplify();
        // Should still be addition
        match simplified {
            AstNode::BinaryOperator { kind, .. } => {
                assert_eq!(kind, BinaryOperatorKind::Add);
            }
            _ => panic!("Expected BinaryOperator Add"),
        }
    }

    #[test]
    fn simplify_function_call_full() {
        let node = AstNode::Number { value: Complex::new(2.0, 0.0), span: Span::from(0..1) }.pow(AstNode::Number { value: Complex::new(3.0, 0.0), span: Span::from(4..5) });
        let simplified = node.simplify();
        assert_astnode_eq!(simplified, AstNode::Number { value: Complex::new(8.0, 0.0), span: Span::from(0..1) });

        let node = AstNode::Number { value: Complex::new(2.0, 0.0), span: Span::from(0..1) }.exp();
        let simplified = node.clone().simplify();
        assert_astnode_eq!(simplified, AstNode::Number { value: Complex::from(2.0).exp(), span: Span::from(0..1) });
    }

    #[test]
    fn simplify_function_call_partial() {
        let node = AstNode::<f64>::Argument { index: 0, span: Span::from(0..1) }.pow(AstNode::Argument { index: 1, span: Span::from(4..5) }).simplify();
        assert_astnode_eq!(
            node,
            AstNode::Argument { index: 0, span: Span::from(0..1) }.pow(AstNode::Argument { index: 1, span: Span::from(4..5) })
        );
    }

    #[test]
    fn simplify_pow_to_powi() {
        let node = AstNode::Argument { index: 0, span: Span::from(0..1) }.pow(AstNode::Number { value: Complex::from(3.0), span: Span::from(4..5) }).simplify();
        assert_astnode_eq!(node, AstNode::Argument { index: 0, span: Span::from(0..1) }.powi(3));
    }

    #[test]
    fn simplify_pow_constant_exponents() {
        let span = Span::from(0..1);
        let cases = [
            //  base, exponent, expected
            ( c(2.0, 0.0), c(3.0, 0.0), c(8.0, 0.0) ),
            ( c(10.0, 0.0), c(2.0, 0.0), c(100.0, 0.0) ),
            ( c(-2.0, 0.0), c(2.0, 0.0), c(4.0, 0.0) ),
            ( c(1.0, 2.0), c(2.0, 0.0), c(-3.0, 4.0) ),
            ( c(2.0, 0.0), c(-3.0, 0.0), c(0.125, 0.0) ),
            ( c(3.0, 0.0), c(2.5, 0.0), c(3.0, 0.0).powc(c(2.5, 0.0)) ),
            ( c(2.0, 0.0), c(0.0, 2.0), c(2.0, 0.0).powc(c(0.0, 2.0)) ),
        ];

        for (base, exp, expected) in cases {
            let expected = AstNode::Number { value: expected, span };
            let base = AstNode::Number { value: base, span };
            let exp = AstNode::Number { value: exp, span };

            // simplify for BinaryOperatorKind::Pow
            let result = AstNode::BinaryOperator {
                kind: BinaryOperatorKind::Pow,
                left: Rc::new(base.clone()),
                right: Rc::new(exp.clone()),
                span,
            }.simplify();
            assert_eq!(result, expected);

            // simplify for FunctionKind::Pow
            let result = base.pow(exp).simplify();
            assert_eq!(result, expected);
        }
    }

    // Dummy user-defined function
    fn sum_func(args: [Complex<f64>; 2]) -> Complex<f64> {
        args[0] + args[1]
    }

    #[test]
    fn simplify_user_function_call_with_numbers() {
        let func = UserFn::new("sum", sum_func);

        let node = AstNode::UserFunctionCall {
            func,
            args: vec![
                Rc::new(AstNode::Number { value: Complex::from(1.0), span: Span::from(0..1) }),
                Rc::new(AstNode::Number { value: Complex::from(2.0), span: Span::from(4..5) }),
            ],
            span: Span::from(0..6),
        }.simplify();

        match node {
            AstNode::Number { value: val, span: _ } => assert_abs_diff_eq!(val.re, 3.0, epsilon=1e-12),
            _ => panic!("Expected simplified to Number"),
        }
    }

    #[test]
    fn simplify_user_function_call_with_no_numbers() {
        let func = UserFn::new("sum", sum_func);

        let node = AstNode::UserFunctionCall {
            func,
            args: vec![
                Rc::new(AstNode::Number { value: Complex::ONE, span: Span::from(0..1) }),
                Rc::new(AstNode::Argument { index: 0, span: Span::from(4..5) }),
            ],
            span: Span::from(0..6),
        };

        let simplified = node.clone().simplify();

        assert_eq!(simplified, node);
    }
}
