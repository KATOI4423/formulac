//! # astnode/horner.rs
//!
//! Rewrites polynomial sums into Horner form right before compilation.
//!
//! ## Responsibilities
//! - Detecting sums of terms `c * x^k` (`x` is an argument, `k` a positive integer,
//!   `c` free of `x`) inside a flattened addition chain
//! - Rebuilding them as `(((an * x + a{n-1}) * x + ...) * x + a0)`
//! - Leaving every other term untouched
//!
//! This pass runs after `simplify()` and is not part of it, so that simplification
//! stays idempotent and like-term merging is not disturbed.

use std::collections::BTreeMap;
use std::rc::Rc;

use num_traits::One;

use crate::astnode::AstNode;
use crate::core::Real;
use crate::functions::FunctionKind;
use crate::lexer::Span;
use crate::operators::BinaryOperatorKind;

/// Minimum nunber of terms with `k >= 1` required to apply the transformation.
const MIN_POLY_TERMS: usize = 2;

/// Minimum highest degree required to apply the transformation.
const MIN_DEGREE: i32 = 2;

impl<T: Real> AstNode<T> {
    /// Rewrites every eligible polynomial inside the tree into Horner form.
    pub(crate) fn horner(self) -> Self {
        match self {
            Self::BinaryOperator { kind: BinaryOperatorKind::Add, .. } => self.horner_add(),
            Self::BinaryOperator { kind, left, right, span } => Self::BinaryOperator {
                kind,
                left: Rc::new(Self::unwrap_rc(left).horner()),
                right: Rc::new(Self::unwrap_rc(right).horner()),
                span,
            },
            Self::UnaryOperator { kind, expr, span } => Self::UnaryOperator {
                kind,
                expr: Rc::new(Self::unwrap_rc(expr).horner()),
                span,
            },
            Self::FunctionCall { kind, args, span } => Self::FunctionCall {
                kind,
                args: Self::horner_args(args),
                span,
            },
            Self::UserFunctionCall { func, args, span } => Self::UserFunctionCall {
                func,
                args: Self::horner_args(args),
                span,
            },
            other => other,
        }
    }

    fn horner_args(args: Vec<Rc<Self>>) -> Vec<Rc<Self>> {
        args.into_iter()
            .map(|a| Rc::new(Self::unwrap_rc(a).horner()))
            .collect()
    }

    /// Handles a whole addition chain at once.
    fn horner_add(self) -> Self {
        let span = self.span();
        let mut terms = Vec::new();
        Self::collect_add_terms(self, &mut terms);

        // Process the terms (coefficients, nested sums) before looking at the chain itself.
        let terms: Vec<Self> = terms.into_iter().map(Self::horner).collect();

        let Some(var) = Self::pick_variable(&terms) else {
            return Self::rebuild_add(terms);
        };

        let mut poly: BTreeMap<i32, Self> = BTreeMap::new();
        let mut others = Vec::new();
        for term in terms {
            match Self::split_term(&term, var) {
                Some((coeff, k)) => {
                    let merged = match poly.remove(&k) {
                        Some(prev) => prev.add(coeff),
                        None => coeff,
                    };
                    poly.insert(k, merged);
                },
                None => others.push(term),
            }
        }

        let head = Self::build_horner(poly, var, span);
        others.into_iter().fold(head, Self::add)
    }


    fn rebuild_add(terms: Vec<Self>) -> Self {
        terms.into_iter()
            .reduce(Self::add)
            .expect("an addition chain has at least two terms")
    }

    /// Builds `( ( (c_n * x^g + c_m) * x^g' + ...) * x&k_min)` from `dgree -> coefficient`.
    fn build_horner(poly: BTreeMap<i32, Self>, var: usize, span: Span) -> Self {
        let x = Self::Argument { index: var, span };
        let mut it = poly.into_iter().rev(); // highest degree first
        let (mut prev_k, mut acc) = it.next().expect("polynomial has at least one term");
        for (k, coeff) in it {
            acc = Self::mul_x_pow(acc, &x, prev_k - k).add(coeff);
            prev_k = k;
        }
        if prev_k > 0 {
            acc = Self::mul_x_pow(acc, &x, prev_k);
        }
        acc
    }

    /// `acc * x^gap`, without emitting `1 * ...` or `powi(x, 1)`.
    fn mul_x_pow(acc: Self, x: &Self, gap: i32) -> Self {
        let xp = if gap == 1 { x.clone() } else { x.clone().powi(gap) };
        if matches!(&acc, Self::Number { value, .. } if value.is_one()) {
            xp
        } else {
            acc.mul(xp)
        }
    }

    // -- analysis -------------------------------------------------------------

    /// Returns `k` if `node` is `x` (k = 1) or `powi(x, k)` with `k >= 1` for argument `var`.
    fn power_of(node: &Self, var: usize) -> Option<i32> {
        match node {
            Self::Argument { index, .. } if *index == var => Some(1),
            Self::FunctionCall { kind: FunctionKind::Powi, args, .. } => {
                match (args[0].as_ref(), args[1].as_ref()) {
                    (Self::Argument { index, .. }, Self::Number { value, .. })
                        if *index == var && Self::is_i32_compatible(value) =>
                    {
                        let n = value.re.to_i32();
                        (n >= 1).then_some(n)
                    },
                    _ => None,
                }
            },
            _ => None,
        }
    }

    /// Collects the argument indices that appear as `x` / `powi(x, n)` factors of a term.
    fn collect_power_vars(node: &Self, out: &mut Vec<usize>) {
        match node {
            Self::Argument { index, .. } => out.push(*index),
            Self::FunctionCall { kind: FunctionKind::Powi, args, .. }
                if let Self::Argument { index, .. } = args[0].as_ref() =>
            {
                out.push(*index)
            },
            Self::BinaryOperator { kind: BinaryOperatorKind::Mul, left, right, .. } => {
                Self::collect_power_vars(left, out);
                Self::collect_power_vars(right, out);
            },
            _ => {},
        }
    }

    fn contains_arg(&self, var: usize) -> bool {
        match self {
            Self::Number { .. } => false,
            Self::Argument { index, .. } => *index == var,
            Self::UnaryOperator { expr, .. } | Self::Derivative { expr, .. }
                => expr.contains_arg(var),
            Self::BinaryOperator { left, right, .. }
                => left.contains_arg(var) || right.contains_arg(var),
            Self::FunctionCall { args, .. } | Self::UserFunctionCall { args, .. }
                => args.iter().any(|a| a.contains_arg(var)),
        }
    }

    /// Splits `term` into `(coefficient, degree)` with respect to `var`.
    ///
    /// Returns `None` if the term depends on `var` in any other way.
    fn split_term(term: &Self, var: usize) -> Option<(Self, i32)> {
        if let Some(k) = Self::power_of(term, var) {
            // x^k or powi(x, k) => (1, k)
            return Some((Self::one(term.span()), k));
        }
        if !term.contains_arg(var) {
            // term * x^0 => (term, 0)
            return Some((term.clone(), 0));
        }

        let Self::BinaryOperator { kind: BinaryOperatorKind::Mul, .. } = term else {
            return None;
        };
        let mut factors = Vec::new();
        Self::collect_mul_terms(term.clone(), &mut factors);

        let pos = factors.iter().position(|f| Self::power_of(f, var).is_some())?;
        let k = Self::power_of(&factors[pos], var)?;
        factors.remove(pos);

        if factors.iter().any(|f| f.contains_arg(var)) {
            return None; // e.g. x * sin(x), or x * x^2 that was not merged
        }

        let coeff = factors.into_iter().reduce(Self::mul)?;
        Some((coeff, k))
    }

    /// Picks the argument with the most `k >= 1` terms, if the cahin is worth rewriting.
    fn pick_variable(terms: &[Self]) -> Option<usize> {
        let mut candidates = Vec::new();
        for t in terms {
            Self::collect_power_vars(t, &mut candidates);
        }
        candidates.sort_unstable();
        candidates.dedup();

        candidates.into_iter().map(|var| {
            let (count, max_k) = terms.iter()
                .filter_map(|term| Self::split_term(term, var))
                .fold((0, 0), |(coef, m), (_, k)| {
                    if k >= 1 { (coef + 1, m.max(k)) } else { (coef, m) }
                });
            (var, count, max_k)
        })
        .filter(|&(_, count, max_k)| count >= MIN_POLY_TERMS && max_k >= MIN_DEGREE)
        .max_by_key(|&(_, count, max_k)| (count, max_k))
        .map(|(var, ..)| var)
    }
}
