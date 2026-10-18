//! # astnode/core.rs
//!
//! Defines the [`AstNode`] enum and provides helper methods for constructing
//! and inspecting AST nodes.
//!
//! This module is the foundation of the AST layer. Other submodules
//! (parsing, simplification, differentiation, compilation) operate on
//! the types defined here.

use num_complex::Complex;
use num_traits::{
    One, Zero,
};
use std::rc::Rc;

use crate::core::Real;
use crate::functions::{
    FunctionKind,
    UserFn,
};
use crate::lexer::Span;
use crate::operators::{
    BinaryOperatorKind,
    UnaryOperatorKind,
};

/// Abstract Syntax Tree node representing a mathematical expression.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum AstNode<T: Real> {
    /// Numeric literal.
    Number {
        value: Complex<T>,
        span: Span, // the location of value
    },

    /// Function argument by index.
    Argument {
        index: usize,
        span: Span, // the location of index
    },

    /// Unary operator applied to an expression.
    UnaryOperator {
        kind: UnaryOperatorKind,
        expr: Rc<AstNode<T>>,
        span: Span, // the location of operator
    },

    /// Binary operator applied to left and right expressions.
    BinaryOperator {
        kind: BinaryOperatorKind,
        left: Rc<AstNode<T>>,
        right: Rc<AstNode<T>>,
        span: Span, // the location of operator
    },

    /// Derivative node: `diff(expr, var, order)`.
    Derivative {
        expr: Rc<AstNode<T>>,
        var: usize,
        order: usize,
        span: Span, // the location of `diff`
    },

    /// Built-in function call.
    FunctionCall {
        kind: FunctionKind,
        args: Vec<Rc<AstNode<T>>>,
        span: Span, // the location of function
    },

    /// User-defined function call.
    UserFunctionCall {
        func: UserFn<T>,
        args: Vec<Rc<AstNode<T>>>,
        span: Span, // the location of function
    },
}

// ─── AstNode builder helpers ─────────────────────────────────────────────────
impl<T: Real> AstNode<T> {
    pub(crate) fn span(&self) -> Span
    {
        match self {
            Self::Argument { span, .. }
            | Self::BinaryOperator { span, .. }
            | Self::Derivative { span, .. }
            | Self::FunctionCall { span, .. }
            | Self::Number { span, .. }
            | Self::UnaryOperator { span, .. }
            | Self::UserFunctionCall { span, .. }
            => *span
        }
    }

    pub(crate) fn unwrap_rc(rc: Rc<Self>) -> Self {
        Rc::try_unwrap(rc).unwrap_or_else(|rc| (*rc).clone())
    }

    pub(crate) fn is_i32_compatible(z: &Complex<T>) -> bool {
        z.im.is_zero() && z.re.is_i32_compatible()
    }

    pub(crate) fn zero(span: Span) -> Self { Self::Number { value: Complex::zero(), span } }
    pub(crate) fn one(span: Span)  -> Self { Self::Number { value: Complex::one(), span } }

    pub(crate) fn is_neg_one(&self) -> bool {
        matches!(self, Self::Number { value, .. } if value.im.is_zero() && value.re == -T::one())
    }

    pub(crate) fn add(self, rhs: Self) -> Self {
        let span = self.span();
        Self::BinaryOperator { kind: BinaryOperatorKind::Add, left: Rc::new(self), right: Rc::new(rhs), span }
    }

    pub(crate) fn sub(self, rhs: Self) -> Self {
        let span = self.span();
        Self::BinaryOperator { kind: BinaryOperatorKind::Sub, left: Rc::new(self), right: Rc::new(rhs), span }
    }

    pub(crate) fn mul(self, rhs: Self) -> Self {
        let span = self.span();
        Self::BinaryOperator { kind: BinaryOperatorKind::Mul, left: Rc::new(self), right: Rc::new(rhs), span }
    }

    pub(crate) fn div(self, rhs: Self) -> Self {
        let span = self.span();
        Self::BinaryOperator { kind: BinaryOperatorKind::Div, left: Rc::new(self), right: Rc::new(rhs), span }
    }

    pub(crate) fn negative(self) -> Self {
        let span = self.span();
        Self::UnaryOperator { kind: UnaryOperatorKind::Negative, expr: Rc::new(self), span }
    }

    pub(crate) fn sin(self)  -> Self {
        let span = self.span();
        Self::FunctionCall { kind: FunctionKind::Sin,  args: vec![Rc::new(self)], span }
    }

    pub(crate) fn cos(self)  -> Self {
        let span = self.span();
        Self::FunctionCall { kind: FunctionKind::Cos,  args: vec![Rc::new(self)], span }
    }

    pub(crate) fn sinh(self) -> Self {
        let span = self.span();
        Self::FunctionCall { kind: FunctionKind::Sinh, args: vec![Rc::new(self)], span }
    }

    pub(crate) fn cosh(self) -> Self {
        let span = self.span();
        Self::FunctionCall { kind: FunctionKind::Cosh, args: vec![Rc::new(self)], span }
    }

    pub(crate) fn exp(self)  -> Self {
        let span = self.span();
        Self::FunctionCall { kind: FunctionKind::Exp,  args: vec![Rc::new(self)], span }
    }

    pub(crate) fn sqrt(self) -> Self {
        let span = self.span();
        Self::FunctionCall { kind: FunctionKind::Sqrt, args: vec![Rc::new(self)], span }
    }

    pub(crate) fn pow(self, exp: Self) -> Self
    {
        let span = self.span();
        Self::FunctionCall { kind: FunctionKind::Pow, args: vec![Rc::new(self), Rc::new(exp)], span }
    }

    pub(crate) fn powi(self, n: i32) -> Self
    {
        let span = self.span();
        Self::FunctionCall {
            kind: FunctionKind::Powi,
            args: vec![
                Rc::new(self),
                Rc::new(Self::Number { value: Complex::from(T::from_f64(n as f64)), span }),
            ],
            span,
        }
    }

    /// Compares two trees by structure only, ignoring every `span`.
    ///
    /// `PartialEq` on [`AstNode`] also compares spans, so two occurrences of the
    /// same subexpression at different positions are never equal under it.
    /// User functions are compared by name only (the closure cannot be compared).
    pub(crate) fn structurally_eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Number { value: a, .. }, Self::Number { value: b, .. }) => a == b,
            (Self::Argument { index: a, .. }, Self::Argument { index: b, .. }) => a == b,
            (Self::UnaryOperator { kind: ak, expr: ae, .. }, Self::UnaryOperator { kind: bk, expr: be, .. })
                => ak == bk && ae.structurally_eq(be),
            (Self::BinaryOperator { kind: ak, left: al, right: ar, .. }, Self::BinaryOperator { kind: bk, left: bl, right: br, .. })
                => ak == bk && al.structurally_eq(bl) && ar.structurally_eq(br),
            (Self::FunctionCall { kind: ak, args: aa, .. }, Self::FunctionCall { kind: bk, args: ba, .. })
                => ak == bk && Self::args_structually_eq(aa, ba),
            (Self::UserFunctionCall { func: af, args: aa, .. }, Self::UserFunctionCall { func: bf, args: ba, .. })
                => af == bf && Self::args_structually_eq(aa, ba),
            (Self::Derivative { expr: ae, var: av, order: ao, .. }, Self::Derivative { expr: be, var: bv, order: bo, .. })
                => av == bv && ao == bo && ae.structurally_eq(be),
            _ => false,
        }
    }

    fn args_structually_eq(a: &[Rc<Self>], b: &[Rc<Self>]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.structurally_eq(y))
    }
}

impl<T: Real> std::ops::Add for AstNode<T> { type Output = Self; fn add(self, rhs: Self) -> Self { self.add(rhs) } }
impl<T: Real> std::ops::Sub for AstNode<T> { type Output = Self; fn sub(self, rhs: Self) -> Self { self.sub(rhs) } }
impl<T: Real> std::ops::Mul for AstNode<T> { type Output = Self; fn mul(self, rhs: Self) -> Self { self.mul(rhs) } }
impl<T: Real> std::ops::Div for AstNode<T> { type Output = Self; fn div(self, rhs: Self) -> Self { self.div(rhs) } }
impl<T: Real> std::ops::BitXor for AstNode<T> { type Output = Self; fn bitxor(self, rhs: Self) -> Self { self.pow(rhs) } }
impl<T: Real> std::ops::Neg for AstNode<T> { type Output = Self; fn neg(self) -> Self { self.negative() } }

#[cfg(test)]
mod structurally_eq_tests {
    use super::*;

    fn x(span: Span) -> AstNode<f64> { AstNode::Argument { index: 0, span } }

    #[test]
    fn ignores_spans() {
        let a = x(Span::from(0..1)).sin();
        let b = x(Span::from(7..8)).sin();
        assert_ne!(a, b); // PartialEq includes spans
        assert!(a.structurally_eq(&b));
    }

    #[test]
    fn distinguishes_structure() {
        let s = Span::from(0..1);
        assert!(!x(s).sin().structurally_eq(&x(s).cos()));
        assert!(!x(s).structurally_eq(&AstNode::Argument { index: 1, span: s }));
        assert!(!x(s).powi(2).structurally_eq(&x(s).powi(3)));
        assert!(!(x(s) + x(s)).structurally_eq(&(x(s) * x(s))));
    }
}
