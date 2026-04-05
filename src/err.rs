//! err.rs
//!
//!

use thiserror::Error;

use crate::lexer::Span;

#[derive(Debug, Error, PartialEq)]
pub enum ParseError {
    /// Unknown lexeme found.
    #[error("Unknown: {str} at {span}")]
    UnknownToken { str: String, span: Span },

    /// Internal Error
    #[error("Internal Error: {reason}")]
    InternalError { reason: String },

    /// The return value is wrong
    #[error("Return value parsed is wrong: {0}")]
    WrongReturn(String),

    /// Invalid formula use
    #[error("Invalid formula: {reason}")]
    InvalidFormula{ reason: String },

    /// Missing function arguments
    #[error("Missing function arguments for {func}")]
    MissingArgs { func: String },

    /// Missing right operand for binary operator
    #[error("Missing right operand for {operator}")]
    MissingRightOperator { operator: String },

    /// Missing left operand for binary operator
    #[error("Missing left operand for {operator}")]
    MissingLeftOperator { operator: String },

    /// Derivative undefined for X_i
    #[error("Undefined derivative of {func} for {idx}")]
    DerivativeUndefined { func: String, idx: usize },

    /// Invalid derivation use
    #[error("Invalid derivative at {span}: {reason}")]
    InvalidDerivative { span: Span, reason: String },

    /// The order of a derivative must be an integer
    #[error("Invalid derivative order {order} at {span}")]
    InvalidDerivativeOrder { span: Span, order: String },

    /// The argument index of function, derivate is out of range
    #[error("Argument Index for {func} is out of range: {idx}")]
    OutOfRange { func: String, idx: usize },
}