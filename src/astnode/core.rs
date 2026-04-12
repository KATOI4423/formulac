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
}

impl<T: Real> std::ops::Add for AstNode<T> { type Output = Self; fn add(self, rhs: Self) -> Self { self.add(rhs) } }
impl<T: Real> std::ops::Sub for AstNode<T> { type Output = Self; fn sub(self, rhs: Self) -> Self { self.sub(rhs) } }
impl<T: Real> std::ops::Mul for AstNode<T> { type Output = Self; fn mul(self, rhs: Self) -> Self { self.mul(rhs) } }
impl<T: Real> std::ops::Div for AstNode<T> { type Output = Self; fn div(self, rhs: Self) -> Self { self.div(rhs) } }
impl<T: Real> std::ops::BitXor for AstNode<T> { type Output = Self; fn bitxor(self, rhs: Self) -> Self { self.pow(rhs) } }
impl<T: Real> std::ops::Neg for AstNode<T> { type Output = Self; fn neg(self) -> Self { self.negative() } }
