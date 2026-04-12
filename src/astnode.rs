//! # astnode.rs
//!
//! Parses mathematical expressions into an Abstract Syntax Tree (AST)
//! and compiles them into executable tokens.
//!
//! Supports real/complex numbers, constants, unary/binary operators,
//! built-in functions, user-defined functions, and symbolic differentiation.

pub mod core;
pub mod diff;
pub mod parser;

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

use crate::core::{
    ComplexMath,
    Real,
};
use crate::functions::{
    FunctionArgs,
    FunctionCall,
    FunctionKind,
};
use crate::lexer::Span;
use crate::operators::{
    BinaryOperatorKind,
};
use crate::token::Token;

pub(crate) type AstNode<T> = core::AstNode<T>;

// ─── simplify ───────────────────────────────────────────────────────────────

impl<T: Real> AstNode<T> {
    /// Simplifies the AST by constant-folding and algebraic normalization.
    pub fn simplify(self) -> Self
    where
        Complex<T>: AddAssign + MulAssign,
    {
        match self {
            Self::UnaryOperator { kind, expr, span } => {
                let expr = Self::unwrap_rc(expr).simplify();
                match expr {
                    Self::Number { value, span } => Self::Number { value: kind.apply(value), span },
                    other => Self::UnaryOperator { kind, expr: Rc::new(other), span },
                }
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
            Self::Number { value: func.apply(FunctionArgs::from(nums)), span }
        } else {
            make_node(func, span, args)
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

    fn collect_add_terms(node: Self, out: &mut Vec<Self>) {
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

    /// Combines `a*x + b*x → (a+b)*x`, removes zero-coefficient terms.
    fn combine_like_add_terms(terms: Vec<Self>) -> Vec<Self>
    where
        Complex<T>: AddAssign,
    {
        fn structurally_equal<T: Real>(a: &AstNode<T>, b: &AstNode<T>) -> bool {
            match (a, b) {
                (AstNode::Number { value: av, .. }, AstNode::Number { value: bv, .. }) => av == bv,
                (AstNode::Argument { index: ai, .. }, AstNode::Argument { index: bi, .. }) => ai == bi,
                (AstNode::UnaryOperator { kind: ak, expr: ae, .. }, AstNode::UnaryOperator { kind: bk, expr: be, .. }) => {
                    ak == bk && structurally_equal(ae, be)
                }
                (AstNode::BinaryOperator { kind: ak, left: al, right: ar, .. }, AstNode::BinaryOperator { kind: bk, left: bl, right: br, .. }) => {
                    ak == bk && structurally_equal(al, bl) && structurally_equal(ar, br)
                }
                (AstNode::FunctionCall { kind: ak, args: aa, .. }, AstNode::FunctionCall { kind: bk, args: ba, .. }) => {
                    ak == bk && aa.len() == ba.len() && aa.iter().zip(ba.iter()).all(|(x, y)| structurally_equal(x, y))
                }
                (AstNode::UserFunctionCall { func: af, args: aa, .. }, AstNode::UserFunctionCall { func: bf, args: ba, .. }) => {
                    af.name() == bf.name() && aa.len() == ba.len() && aa.iter().zip(ba.iter()).all(|(x, y)| structurally_equal(x, y))
                }
                (AstNode::Derivative { expr: ae, var: av, order: ao, .. }, AstNode::Derivative { expr: be, var: bv, order: bo, .. }) => {
                    av == bv && ao == bo && structurally_equal(ae, be)
                }
                _ => false,
            }
        }

        // Map: (variable node) → accumulated coefficient
        let mut map: Vec<(Self, Complex<T>)> = Vec::new();

        for term in terms {
            let (var, coeff) = match term {
                Self::BinaryOperator { kind: BinaryOperatorKind::Mul, left, right, .. } => {
                    let left = Self::unwrap_rc(left);
                    let right = Self::unwrap_rc(right);
                    match (left, right) {
                        (Self::Number { value: z, .. }, v) => (v, z),
                        (v, Self::Number { value: z, .. }) => (v, z),
                        (l, r)               => (l.mul(r), Complex::one()),
                    }
                }
                other => (other, Complex::one()),
            };
            match map.iter_mut().find(|(v, _)| structurally_equal(v, &var)) {
                Some((_, c)) => *c += coeff,
                None         => map.push((var, coeff)),
            }
        }

        map.into_iter()
            .filter(|(_, c)| !(*c).is_zero())
            .map(|(var, coeff)| {
                if coeff.is_one() { var }
                else { Self::Number { value: coeff, span: var.span() }.mul(var) }
            })
            .collect()
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

    fn collect_mul_terms(node: Self, out: &mut Vec<Self>) {
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
            match map.iter_mut().find(|(b, _)| *b == base) {
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

        // (x^a)^b → x^(a*b)
        loop {
            match base {
                Self::FunctionCall { kind: FunctionKind::Pow | FunctionKind::Powi, mut args, .. } => {
                    let inner_base = Self::unwrap_rc(args.remove(0))
                        ;
                    let inner_exp  = Self::unwrap_rc(args.remove(0))
                        ;
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
                => Self::Number { value: b.powc(e), span: s },
            (b, Self::Number { value: e, .. }) if Self::is_i32_compatible(&e)
                => b.powi(e.re.to_i32()),
            (b, e) => {
                b.pow(e)
            },
        }
    }
}

// ─── compile ────────────────────────────────────────────────────────────────

impl<T: Real> AstNode<T> {
    /// Compiles the AST into a flat sequence of postfix `Token`s.
    pub fn compile(&self) -> Vec<Token<T>> {
        let mut out = Vec::new();
        self.compile_into(&mut out);
        out
    }

    fn compile_into(&self, out: &mut Vec<Token<T>>) {
        match self {
            Self::Number { value, span } => out.push(Token::Number { value: value.clone(), span: *span }),
            Self::Argument { index, span } => out.push(Token::Argument { index: *index, span: *span }),
            Self::UnaryOperator { kind, expr, span } => {
                expr.compile_into(out);
                out.push(Token::UnaryOperator { kind: *kind, span: *span });
            }
            Self::BinaryOperator { kind, left, right, span } => {
                left.compile_into(out);
                right.compile_into(out);
                out.push(Token::BinaryOperator { kind: *kind, span: *span });
            }
            Self::FunctionCall { kind, args, span } => {
                for arg in args { arg.compile_into(out); }
                out.push(Token::Function { kind: *kind, span: *span });
            }
            Self::UserFunctionCall { func, args, span } => {
                for arg in args { arg.compile_into(out); }
                out.push(Token::UserFunction { func: func.clone(), span: *span });
            }
            Self::Derivative { .. } => {
                unreachable!("Derivative nodes must be resolved before compile()")
            }
        }
    }
}


#[cfg(test)]
mod astnode_tests {
    use super::*;
    use crate::functions::UserFn;
    use crate::operators::UnaryOperatorKind;
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

    #[test]
    fn test_fold_add_constants() {
        let left = AstNode::Number { value: Complex::from(2.0), span: Span::from(0..1) };
        let right = AstNode::Number { value: Complex::from(3.0), span: Span::from(1..2) };
        let result = AstNode::fold_add(left.span(), left, right);
        assert_astnode_eq!(result, AstNode::Number { value: Complex::from(5.0), span: Span::from(0..1) });
    }

    #[test]
    fn test_fold_add_like_terms() {
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
    fn test_fold_add_mixed_terms() {
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
    fn test_fold_add_removed_same_terms() {
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
    }

    #[test]
    fn test_fold_add_zero_terms() {
        // x + 0 → x
        let x = AstNode::Argument { index: 0, span: Span::from(0..1) };
        let zero = AstNode::Number { value: Complex::from(0.0), span: Span::from(2..3) };
        let result = AstNode::fold_add(Span::from(1..2), x.clone(), zero);
        assert_astnode_eq!(result, x);
    }

    #[test]
    fn test_fold_sub_basic() {
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
    fn test_fold_sub_with_constants() {
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
    fn test_mul_constant_folding() {
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
    fn test_mul_with_zero() {
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
    fn test_mul_with_one() {
        let expr = AstNode::fold_mul(
            Span::from(2..3),
            AstNode::Number { value: Complex::<f64>::one(), span: Span::from(0..1) },
            AstNode::Argument { index: 0, span: Span::from(4..5) }
        );
        assert_astnode_eq!(expr, AstNode::Argument { index: 0, span: Span::from(4..5) });
    }

    #[test]
    fn test_div_to_mul_pow_neg1() {
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
    fn test_combine_same_base_powers() {
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
    fn test_combine_same_base_powers_to_powi() {
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
    fn test_nested_mul_flattening() {
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
    fn test_simplify_number() {
        let node = AstNode::Number { value: Complex::new(3.0, 0.0), span: Span::from(0..1) };
        assert_astnode_eq!(node.clone().simplify(), node);
    }

    #[test]
    fn test_simplify_unary_operator() {
        let node = -AstNode::Number { value: Complex::new(2.0, 0.0), span: Span::from(2..3) };
        let simplified = node.simplify();
        assert_astnode_eq!(simplified, AstNode::Number { value: Complex::new(-2.0, 0.0), span: Span::from(2..3) });
    }

    #[test]
    fn test_simplify_binary_operator_full() {
        let node = AstNode::Number { value: Complex::new(2.0, 0.0), span: Span::from(0..1) } + AstNode::Number { value: Complex::new(3.0, 0.0), span: Span::from(4..5) };
        let simplified = node.simplify();
        assert_astnode_eq!(simplified, AstNode::Number { value: Complex::new(5.0, 0.0), span: Span::from(0..1) });
    }

    #[test]
    fn test_simplify_binary_operator_partial() {
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
    fn test_simplify_binary_operator_chain() {
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
    fn test_simplify_function_call_full() {
        let node = AstNode::Number { value: Complex::new(2.0, 0.0), span: Span::from(0..1) }.pow(AstNode::Number { value: Complex::new(3.0, 0.0), span: Span::from(4..5) });
        let simplified = node.simplify();
        assert_astnode_eq!(simplified, AstNode::Number { value: Complex::new(8.0, 0.0), span: Span::from(0..1) });

        let node = AstNode::Number { value: Complex::new(2.0, 0.0), span: Span::from(0..1) }.exp();
        let simplified = node.clone().simplify();
        assert_astnode_eq!(simplified, AstNode::Number { value: Complex::from(2.0).exp(), span: Span::from(0..1) });
    }

    #[test]
    fn test_simplify_function_call_partial() {
        let node = AstNode::<f64>::Argument { index: 0, span: Span::from(0..1) }.pow(AstNode::Argument { index: 1, span: Span::from(4..5) }).simplify();
        assert_astnode_eq!(
            node,
            AstNode::Argument { index: 0, span: Span::from(0..1) }.pow(AstNode::Argument { index: 1, span: Span::from(4..5) })
        );
    }

    #[test]
    fn test_simplify_pow_to_powi() {
        let node = AstNode::Argument { index: 0, span: Span::from(0..1) }.pow(AstNode::Number { value: Complex::from(3.0), span: Span::from(4..5) }).simplify();
        assert_astnode_eq!(node, AstNode::Argument { index: 0, span: Span::from(0..1) }.powi(3));
    }


    // Dummy user-defined function
    fn sum_func(args: [Complex<f64>; 2]) -> Complex<f64> {
        args[0] + args[1]
    }

    #[test]
    fn test_simplify_user_function_call_with_numbers() {
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
    fn test_simplify_user_function_call_with_no_numbers() {
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

    #[test]
    fn test_compile_number() {
        let ast = AstNode::Number { value: Complex::new(1.0, 0.0), span: Span::from(0..1) };
        let tokens = ast.compile();
        assert_eq!(tokens, vec![Token::Number { value: Complex::new(1.0, 0.0), span: Span::from(0..1) }]);
    }

    #[test]
    fn test_compile_argument() {
        let ast = AstNode::<f64>::Argument { index: 1, span: Span::from(0..1) };
        let tokens = ast.compile();
        assert_eq!(tokens, vec![Token::Argument { index: 1, span: Span::from(0..1) }]);
    }

    #[test]
    fn test_compile_unary_operator() {
        let ast = -AstNode::Number { value: Complex::new(1.0, 0.0), span: Span::from(2..3) };
        let tokens = ast.compile();
        assert_eq!(
            tokens,
            vec![Token::Number { value: Complex::new(1.0, 0.0), span: Span::from(2..3) }, Token::UnaryOperator { kind: UnaryOperatorKind::Negative, span: Span::from(2..3) }]
        );
    }

    #[test]
    fn test_compile_binary_operator() {
        let ast = AstNode::Number { value: Complex::new(1.0, 0.0), span: Span::from(0..1) } + AstNode::Argument { index: 1, span: Span::from(4..5) };
        let tokens = ast.compile();
        assert_eq!(
            tokens,
            vec![
                Token::Number { value: Complex::new(1.0, 0.0), span: Span::from(0..1) },
                Token::Argument { index: 1, span: Span::from(4..5) },
                Token::BinaryOperator { kind: BinaryOperatorKind::Add, span: Span::from(0..1) },
            ]
        );
    }

    #[test]
    fn test_compile_function_single_argument() {
        let ast = AstNode::Number { value: Complex::new(1.0, 0.0), span: Span::from(0..1) }.sin();
        let tokens = ast.compile();
        assert_eq!(tokens, vec![Token::Number { value: Complex::new(1.0, 0.0), span: Span::from(0..1) }, Token::Function { kind: FunctionKind::Sin, span: Span::from(0..1) }]);
    }

    #[test]
    fn test_compile_function_multi_arguments() {
        let ast = AstNode::Number { value: Complex::new(2.0, 0.0), span: Span::from(0..1) }.pow(AstNode::Argument { index: 0, span: Span::from(4..5) });
        let tokens = ast.compile();
        assert_eq!(
            tokens,
            vec![
                Token::Number { value: Complex::new(2.0, 0.0), span: Span::from(0..1) },
                Token::Argument { index: 0, span: Span::from(4..5) },
                Token::Function { kind: FunctionKind::Pow, span: Span::from(0..1) },
            ]
        );
    }

    #[test]
    fn test_compile_nested_expression() {
        // cos(1 + 2)
        let ast = (AstNode::Number { value: Complex::new(1.0, 0.0), span: Span::from(0..1) } + AstNode::Number { value: Complex::new(2.0, 0.0), span: Span::from(4..5) }).cos();
        let tokens = ast.compile();
        assert_eq!(
            tokens,
            vec![
                Token::Number { value: Complex::new(1.0, 0.0), span: Span::from(0..1) },
                Token::Number { value: Complex::new(2.0, 0.0), span: Span::from(4..5) },
                Token::BinaryOperator { kind: BinaryOperatorKind::Add, span: Span::from(0..1) },
                Token::Function { kind: FunctionKind::Cos, span: Span::from(0..1) },
            ]
        );
    }
}
