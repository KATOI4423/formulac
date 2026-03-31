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
};
use crate::operators::{
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
    Number(Complex<T>),

    /// Function argument by position index.
    Argument(usize),

    /// Ambiguous operator lexeme (resolved into unary or binary during parsing).
    Operator(Lexeme),

    /// Resolved unary operator.
    UnaryOperator(UnaryOperatorKind),

    /// Resolved binary operator.
    BinaryOperator(BinaryOperatorKind),

    /// Differential operator `diff`.
    DiffOperator(Lexeme),

    /// Built-in function (e.g., `sin`, `cos`).
    Function(FunctionKind),

    /// User-defined function.
    UserFunction(UserFn<T>),

    /// Left parenthesis `(`.
    LParen(Lexeme),

    /// Right parenthesis `)`.
    RParen(Lexeme),

    /// Comma `,` used as argument separator.
    Comma(Lexeme),
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

        // 1. Number or constant
        if let Some(val) = Self::parse_real(text)
            .or_else(|| Self::parse_imaginary(text))
            .or_else(|| constants.get(text).cloned())
        {
            return Ok(Token::Number(val));
        }

        // 2. Differential operator
        if text == DIFFERENTIAL_OPERATOR_STR {
            return Ok(Token::DiffOperator(lexeme.clone()));
        }

        // 3. Function argument
        if let Some(pos) = args.iter().position(|&a| a == text) {
            return Ok(Token::Argument(pos));
        }

        // 4. Operator (unary/binary disambiguation deferred to AstNode parser)
        if UnaryOperatorKind::try_from(lexeme.clone()).is_ok()
            || BinaryOperatorKind::try_from(lexeme.clone()).is_ok()
        {
            return Ok(Token::Operator(lexeme.clone()));
        }

        // 5. Built-in function
        if let Ok(func_kind) = FunctionKind::try_from(lexeme.clone()) {
            return Ok(Token::Function(func_kind));
        }

        // 6. User-defined function
        if let Some(user_func) = users.get(text) {
            return Ok(Token::UserFunction(user_func.clone()));
        }

        // 7. Structural tokens
        match text {
            "(" => Ok(Token::LParen(lexeme.clone())),
            ")" => Ok(Token::RParen(lexeme.clone())),
            "," => Ok(Token::Comma(lexeme.clone())),
            _   => Err(ParseError::UnknownToken(lexeme.clone())),
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
            Token::Number(val) => assert_eq!(val, Complex::new(3.14, 0.0)),
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
            Token::Number(val) => assert_eq!(val, Complex::new(0.0, 2.0)),
            _ => panic!("Expected Number token"),
        }

        let lex = Lexeme::new("i", 0..2);
        let token = Token::try_from(&lex, &args, &constants, &users).unwrap();
        match token {
            Token::Number(val) => assert_eq!(val, Complex::new(0.0, 1.0)),
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
            Token::Number(val) => assert_eq!(val, Complex::new(std::f64::consts::PI, 0.0)),
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
            Token::Argument(pos) => assert_eq!(pos, 0),
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
            Token::Operator(_) => {}, // OK
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
            Token::Function(f) => assert_eq!(f, FunctionKind::Sin),
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

        assert!(matches!(Token::try_from(&lex_l, &args, &constants, &users).unwrap(), Token::LParen(_)));
        assert!(matches!(Token::try_from(&lex_r, &args, &constants, &users).unwrap(), Token::RParen(_)));
        assert!(matches!(Token::try_from(&lex_c, &args, &constants, &users).unwrap(), Token::Comma(_)));
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
