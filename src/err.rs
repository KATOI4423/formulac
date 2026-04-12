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
    #[error("Internal Error at {span}: {reason}")]
    InternalError { reason: String, span: Span },

    /// The return value is wrong
    #[error("Return value parsed is wrong: {0}")]
    WrongReturn(String),

    /// Invalid formula use
    #[error("Invalid formula at {span}: {reason}")]
    InvalidFormula{ reason: String, span: Span },

    /// Missing function arguments
    #[error("Missing function arguments for {func} at {span}")]
    MissingArgs { func: String, span: Span },

    /// Missing right operand for binary operator
    #[error("Missing right operand for {operator} at {span}")]
    MissingRightOperator { operator: String, span: Span },

    /// Missing left operand for binary operator
    #[error("Missing left operand for {operator} at {span}")]
    MissingLeftOperator { operator: String, span: Span },

    /// Derivative undefined for X_i
    #[error("Undefined derivative of {func} for {idx} at {span}")]
    DerivativeUndefined { func: String, idx: usize, span: Span },

    /// Invalid derivation use
    #[error("Invalid derivative at {span}: {reason}")]
    InvalidDerivative { span: Span, reason: String },

    /// The order of a derivative must be an integer
    #[error("Invalid derivative order {order} at {span}")]
    InvalidDerivativeOrder { span: Span, order: String },

    /// The argument index of function, derivate is out of range
    #[error("Argument Index for {func} at {span} is out of range: {idx}")]
    OutOfRange { func: String, idx: usize, span: Span },
}

#[derive(Debug, Error, PartialEq)]
pub enum InitializeError
{
    #[error("Mismatched number of derivatives ({number}): expected {expected}")]
    DerivativesNumberMismatched { expected: usize, number: usize }
}