//! # token.rs
//!
//! This module defines the `Token` type produced by the lexer-to-parser bridge.
//! Tokens are classified from `Lexeme`s and consumed by the AST builder.

use std::collections::HashMap;
use std::str::FromStr;
use num_complex::Complex;

use crate::constants::Constants;
use crate::core::Real;
use crate::err::ParseError;
use crate::functions::{
    FunctionKind,
    UserFn,
};
use crate::lexer::{
    Lexeme,
    IMAGINARY_UNIT,
    Span,
};
use crate::operators::{
    OperatorKind,
    BinaryOperatorKind,
    UnaryOperatorKind,
};

pub const DIFFERENTIAL_OPERATOR_STR: &str = "diff";
pub(crate) type UserFnTable<T> = HashMap<String, UserFn<T>>;

/// Represents a parsed token in a mathematical expression.
///
/// Tokens are produced from `Lexeme`s and consumed by the AST builder.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Token<T: Real> {
    /// Resolved numeric value (literal, constant, or imaginary).
    Number {
        value: Complex<T>,
        span: Span,
    },

    /// Function argument by position index.
    Argument {
        index: usize,
        span: Span,
    },

    /// Ambiguous operator (resolved into unary or binary during parsing).
    Operator {
        kind: OperatorKind,
        span: Span,
    },

    /// Resolved unary operator.
    UnaryOperator {
        kind: UnaryOperatorKind,
        span: Span,
    },

    /// Resolved binary operator.
    BinaryOperator {
        kind: BinaryOperatorKind,
        span: Span,
    },

    /// Differential operator `diff`.
    DiffOperator {
        span: Span,
    },

    /// Built-in function (e.g., `sin`, `cos`).
    Function {
        kind: FunctionKind,
        span: Span,
    },

    /// User-defined function.
    UserFunction {
        func: UserFn<T>,
        span: Span,
    },

    /// Left parenthesis `(`.
    LParen {
        span: Span,
    },

    /// Right parenthesis `)`.
    RParen {
        span: Span,
    },

    /// Comma `,` used as argument separator.
    Comma {
        span: Span,
    },
}

impl<T: Real> Token<T> {
    /// Attempts to parse a string as a real number.
    fn parse_real(s: &str) -> Option<Complex<T>>
    where
        T: FromStr,
    {
        s.parse::<T>().ok().map(|v| Complex::new(v, T::zero()))
    }

    /// Attempts to parse a string as an imaginary number (e.g. `"3i"`, `"i"`).
    fn parse_imaginary(s: &str) -> Option<Complex<T>>
    where
        T: FromStr,
    {
        let num_part = s.strip_suffix(IMAGINARY_UNIT)?;
        if num_part.is_empty() {
            return Some(Complex::new(T::zero(), T::one()));
        }
        num_part.parse::<T>().ok().map(|v| Complex::new(T::zero(), v))
    }

    /// Classifies a `Lexeme` into a `Token`.
    ///
    /// Resolution order:
    /// 1. Numeric literal or constant
    /// 2. Differential operator
    /// 3. Function argument
    /// 4. Operator symbol (unary/binary disambiguation deferred to parser)
    /// 5. Built-in function
    /// 6. User-defined function
    /// 7. Parentheses and comma
    pub fn try_from(
        lexeme: &Lexeme,
        args: &[&str],
        constants: &Constants<T>,
        users: &UserFnTable<T>,
    ) -> Result<Self, ParseError>
    where
        T: FromStr,
    {
        let text = lexeme.text();
        let span = lexeme.span();

        // 1. Number or constant
        if let Some(value) = Self::parse_real(text)
            .or_else(|| Self::parse_imaginary(text))
            .or_else(|| constants.get(text))
        {
            return Ok(Token::Number { value, span });
        }

        // 2. Differential operator
        if text == DIFFERENTIAL_OPERATOR_STR {
            return Ok(Token::DiffOperator { span });
        }

        // 3. Function argument
        if let Some(index) = args.iter().position(|&a| a == text) {
            return Ok(Token::Argument { index, span });
        }

        // 4. Operator (unary/binary disambiguation deferred to AstNode parser)
        if let Ok(kind) = OperatorKind::from_str(text)
        {
            return Ok(Token::Operator { kind, span });
        }

        // 5. Built-in function
        if let Ok(kind) = FunctionKind::from_str(text) {
            return Ok(Token::Function { kind, span });
        }

        // 6. User-defined function
        if let Some(func) = users.get(text) {
            return Ok(Token::UserFunction { func: func.clone(), span });
        }

        // 7. Structural tokens
        match text {
            "(" => Ok(Token::LParen { span }),
            ")" => Ok(Token::RParen { span }),
            "," => Ok(Token::Comma { span }),
            _   => Err(ParseError::UnknownToken { str: lexeme.text().to_string(), span }),
        }
    }

    pub fn span(&self) -> Span
    {
        match self {
            Self::Number { span, .. }
            | Self::Argument { span, .. }
            | Self::Operator { span, .. }
            | Self::UnaryOperator { span, .. }
            | Self::BinaryOperator { span, .. }
            | Self::Function { span, .. }
            | Self::UserFunction { span, .. }
            | Self::DiffOperator { span }
            | Self::LParen { span }
            | Self::RParen { span }
            | Self::Comma { span }
            => *span,
        }
    }
}

#[cfg(test)]
mod token_tests {
    use super::*;

    #[test]
    fn test_number_token() {
        let lex = Lexeme::new("3.14", 0..4);
        let constants = Constants::new();
        let users = UserFnTable::new();
        let args: [&str; 0] = [];
        let token = Token::try_from(&lex, &args, &constants, &users).unwrap();
        match token {
            Token::Number { value, span: _ } => assert_eq!(value, Complex::new(3.14, 0.0)),
            _ => panic!("Expected Number token"),
        }
    }

    #[test]
    fn test_imaginary_number() {
        let lex = Lexeme::new("2i", 0..2);
        let constants = Constants::new();
        let users = UserFnTable::new();
        let args: [&str; 0] = [];
        let token = Token::try_from(&lex, &args, &constants, &users).unwrap();
        match token {
            Token::Number { value, span: _ } => assert_eq!(value, Complex::new(0.0, 2.0)),
            _ => panic!("Expected Number token"),
        }

        let lex = Lexeme::new("i", 0..2);
        let token = Token::try_from(&lex, &args, &constants, &users).unwrap();
        match token {
            Token::Number { value, span: _ } => assert_eq!(value, Complex::new(0.0, 1.0)),
            _ => panic!("Expected Number token"),
        }
    }

    #[test]
    fn test_constant_token() {
        let lex = Lexeme::new("PI", 0..2);
        let constants = Constants::default();
        let users = UserFnTable::new();
        let args: [&str; 0] = [];
        let token = Token::try_from(&lex, &args, &constants, &users).unwrap();
        match token {
            Token::Number { value, span: _ } => assert_eq!(value, Complex::new(std::f64::consts::PI, 0.0)),
            _ => panic!("Expected Number token"),
        }
    }

    #[test]
    fn test_argument_token() {
        let lex = Lexeme::new("arg0", 0..4);
        let constants = Constants::<f64>::new();
        let args = ["arg0"];
        let users = UserFnTable::new();
        let token = Token::try_from(&lex, &args, &constants, &users).unwrap();
        match token {
            Token::Argument { index, span: _ } => assert_eq!(index, 0),
            _ => panic!("Expected Argument token"),
        }
    }

    #[test]
    fn test_operator_token() {
        let lex = Lexeme::new("+", 0..1);
        let constants = Constants::<f64>::new();
        let args: [&str; 0] = [];
        let users = UserFnTable::new();
        let token = Token::try_from(&lex, &args, &constants, &users).unwrap();
        match token {
            Token::Operator { kind: _, span: _ } => {}, // OK
            _ => panic!("Expected Operator token"),
        }
    }

    #[test]
    fn test_function_token() {
        let lex = Lexeme::new("sin", 0..3);
        let constants = Constants::<f64>::new();
        let users = UserFnTable::new();
        let args: [&str; 0] = [];
        let token = Token::try_from(&lex, &args, &constants, &users).unwrap();
        match token {
            Token::Function { kind, span: _} => assert_eq!(kind, FunctionKind::Sin),
            _ => panic!("Expected Function token"),
        }
    }

    #[test]
    fn test_parentheses_and_comma() {
        let lex_l = Lexeme::new("(", 0..1);
        let lex_r = Lexeme::new(")", 0..1);
        let lex_c = Lexeme::new(",", 0..1);
        let constants = Constants::<f64>::new();
        let users = UserFnTable::new();
        let args: [&str; 0] = [];

        assert!(matches!(Token::try_from(&lex_l, &args, &constants, &users).unwrap(), Token::LParen { span: _ }));
        assert!(matches!(Token::try_from(&lex_r, &args, &constants, &users).unwrap(), Token::RParen { span: _ }));
        assert!(matches!(Token::try_from(&lex_c, &args, &constants, &users).unwrap(), Token::Comma { span: _ }));
    }

    #[test]
    fn test_unknown_string() {
        let lex = Lexeme::new("unknown", 0..7);
        let constants = Constants::<f64>::new();
        let users = UserFnTable::new();
        let args: [&str; 0] = [];
        let res = Token::try_from(&lex, &args, &constants, &users);
        assert!(res.is_err());
    }
}
