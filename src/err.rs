//! err.rs
//!
//!

use thiserror::Error;

use crate::lexer::Span;

/// Errors that can occur while parsing, simplifying, or differentiating a formula.
#[derive(Debug, Error, PartialEq)]
pub enum ParseError {
    /// Unknown lexeme found.
    #[error("Unknown: {str} at {span}")]
    UnknownToken {
        /// The unrecognized source text.
        str: String,
        /// The source span containing the unrecognized text.
        span: Span,
    },

    /// Internal Error
    #[error("Internal Error at {span}: {reason}")]
    InternalError {
        /// A description of the internal failure.
        reason: String,
        /// The source span associated with the failure.
        span: Span,
    },

    /// The return value is wrong
    #[error("Return value parsed is wrong: {0}")]
    WrongReturn(String),

    /// Invalid formula use
    #[error("Invalid formula at {span}: {reason}")]
    InvalidFormula {
        /// A description of why the formula is invalid.
        reason: String,
        /// The source span where the invalid construct was detected.
        span: Span,
    },

    /// Missing function arguments
    #[error("Missing function arguments for {func} at {span}")]
    MissingArgs {
        /// The function for which arguments are missing.
        func: String,
        /// The source span of the function call.
        span: Span,
    },

    /// Missing right operand for binary operator
    #[error("Missing right operand for {operator} at {span}")]
    MissingRightOperator {
        /// The binary operator missing its right operand.
        operator: String,
        /// The source span of the operator.
        span: Span,
    },

    /// Missing left operand for binary operator
    #[error("Missing left operand for {operator} at {span}")]
    MissingLeftOperator {
        /// The binary operator missing its left operand.
        operator: String,
        /// The source span of the operator.
        span: Span,
    },

    /// Derivative undefined for X_i
    #[error("Undefined derivative of {func} for {idx} at {span}")]
    DerivativeUndefined {
        /// The function whose derivative is unavailable.
        func: String,
        /// The zero-based argument index for which the derivative is requested.
        idx: usize,
        /// The source span associated with the function.
        span: Span,
    },

    /// Invalid derivation use
    #[error("Invalid derivative at {span}: {reason}")]
    InvalidDerivative {
        /// The source span associated with the invalid derivative expression.
        span: Span,
        /// A description of why the derivative request is invalid.
        reason: String,
    },

    /// The order of a derivative must be an integer
    #[error("Invalid derivative order {order} at {span}")]
    InvalidDerivativeOrder {
        /// The source span containing the invalid order.
        span: Span,
        /// The source text supplied as the derivative order.
        order: String,
    },

    /// The argument index of function, derivate is out of range
    #[error("Argument Index for {func} at {span} is out of range: {idx}")]
    OutOfRange {
        /// The function whose argument index is out of range.
        func: String,
        /// The zero-based argument index that was requested.
        idx: usize,
        /// The source span associated with the invalid access.
        span: Span,
    },
}

/// Errors that can occur while initializing a user-defined function.
#[derive(Debug, Error, PartialEq)]
pub enum InitializeError
{
    #[error("Mismatched number of derivatives ({number}): expected {expected}")]
    /// The number of supplied derivative functions does not match the
    /// function's arity.
    DerivativesNumberMismatched {
        /// The number of derivatives required by the function.
        expected: usize,
        /// The number of derivatives that was supplied.
        number: usize,
    }
}