//! # operators.rs
//!
//! This module defines unary and binary operators used in mathematical expressions.
//! It provides enums for operator kinds, precedence information, and application logic.

use num_complex::Complex;

use crate::core::{
    ComplexMath,
    Real,
};
use crate::err::ParseError;
use crate::lexer::Lexeme;

#[doc(hidden)]
/// Internal macro to define all unary operators.
macro_rules! unary_operator_kind {
    ($($name:ident => { symbol: $symbol:expr, apply: $apply:expr }),* $(,)?) => {
        /// Represents a unary operator in a mathematical expression.
        #[derive(Debug, Clone, Copy, PartialEq)]
        pub enum UnaryOperatorKind {
            $($name),*
        }

        impl TryFrom<Lexeme> for UnaryOperatorKind {
            type Error = ParseError;
            fn try_from(s: Lexeme) -> Result<Self, Self::Error> {
                match s.text() {
                    $( $symbol => Ok(Self::$name), )*
                    _ => Err(ParseError::UnknownToken(s)),
                }
            }
        }

        impl UnaryOperatorKind {
            /// Applies the unary operator to a complex number.
            pub fn apply<T: Real>(&self, x: Complex<T>) -> Complex<T> {
                match self {
                    $( Self::$name => $apply(x), )*
                }
            }

            /// Returns a list of all supported unary operator symbols.
            pub fn symbols() -> &'static [&'static str] {
                &[$($symbol), *]
            }
        }

        impl std::fmt::Display for UnaryOperatorKind {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    $( Self::$name => write!(f, $symbol), )*
                }
            }
        }
    };
}

unary_operator_kind! {
    Positive => { symbol: "+", apply: |x| x },
    Negative => { symbol: "-", apply: |x: Complex<_>| -x },
}

#[doc(hidden)]
/// Internal macro to define all binary operators.
macro_rules! binary_operators {
    ($($name:ident => {
        symbol: $symbol:expr,
        precedence: $prec:expr,
        left_assoc: $assoc:expr,
        apply: $apply:expr
    }),* $(,)?) => {
        /// Represents a binary operator in a mathematical expression.
        #[derive(Debug, Clone, Copy, PartialEq)]
        pub enum BinaryOperatorKind {
            $($name),*
        }

        impl TryFrom<Lexeme> for BinaryOperatorKind {
            type Error = ParseError;
            fn try_from(s: Lexeme) -> Result<Self, Self::Error> {
                match s.text() {
                    $( $symbol => Ok(Self::$name), )*
                    _ => Err(ParseError::UnknownToken(s.clone())),
                }
            }
        }

        impl BinaryOperatorKind {
            #[inline]
            pub fn precedence(&self) -> u8 {
                match self {
                    $( Self::$name => $prec, )*
                }
            }

            #[inline]
            pub fn is_left_assoc(&self) -> bool {
                match self {
                    $( Self::$name => $assoc, )*
                }
            }

            /// Applies the operator to two complex numbers.
            #[inline]
            pub fn apply<T: Real>(&self, l: Complex<T>, r: Complex<T>) -> Complex<T> {
                match self {
                    $(Self::$name => $apply(l, r),)*
                }
            }

            /// Returns a list of all supported binary operator symbols.
            pub fn symbols() -> &'static [&'static str] {
                &[$($symbol), *]
            }
        }

        impl std::fmt::Display for BinaryOperatorKind {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    $( Self::$name => write!(f, $symbol), )*
                }
            }
        }
    };
}

binary_operators! {
    Add => { symbol: "+", precedence: 0, left_assoc: true,  apply: |l, r| l + r },
    Sub => { symbol: "-", precedence: 0, left_assoc: true,  apply: |l, r| l - r },
    Mul => { symbol: "*", precedence: 1, left_assoc: true,  apply: |l, r| l * r },
    Div => { symbol: "/", precedence: 1, left_assoc: true,  apply: |l, r| l / r },
    Pow => { symbol: "^", precedence: 2, left_assoc: false, apply: |l: Complex<T>, r: Complex<T>| l.powc(r) },
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    // Helper
    fn c(re: f64, im:f64) -> Complex<f64> {
        Complex::new(re, im)
    }

    fn eq(a: Complex<f64>, b: Complex<f64>) -> bool {
        (a - b).norm() < 1.0e-10
    }

    // ── UnaryOperatorKind ─────────────────────────────────────
    mod unary {
        use super::*;

        #[test]
        fn from_valid_symbols() {
            assert_eq!(UnaryOperatorKind::try_from(Lexeme::new("+", 0..1)), Ok(UnaryOperatorKind::Positive));
            assert_eq!(UnaryOperatorKind::try_from(Lexeme::new("-", 0..1)), Ok(UnaryOperatorKind::Negative));
        }

        #[test]
        fn from_invalid_symbol() {
            assert!(UnaryOperatorKind::try_from(Lexeme::new("*", 0..1)).is_err());
            assert!(UnaryOperatorKind::try_from(Lexeme::new("", 0..1)).is_err());
            assert!(UnaryOperatorKind::try_from(Lexeme::new("++", 0..2)).is_err());
        }

        #[test]
        fn apply_positive_is_identity() {
            let cases = [c(0.0, 0.0), c(3.0, 0.0), c(-2.0, 5.0)];
            for x in cases {
                assert_eq!(UnaryOperatorKind::Positive.apply(x), x);
            }
        }

        #[test]
        fn apply_negative_negates() {
            assert_eq!(UnaryOperatorKind::Negative.apply(c(3.0, 4.0)), c(-3.0, -4.0));
            assert_eq!(UnaryOperatorKind::Negative.apply(c(0.0, 0.0)), c(0.0, 0.0));
        }

        #[test]
        fn symbols_contains_all() {
            let syms = UnaryOperatorKind::symbols();
            assert!(syms.contains(&"+"));
            assert!(syms.contains(&"-"));
        }

        #[test]
        fn display() {
            assert_eq!(UnaryOperatorKind::Positive.to_string(), "+");
            assert_eq!(UnaryOperatorKind::Negative.to_string(), "-");
        }
    }

    // ── BinaryOperatorKind ────────────────────────────────────
    mod binary {
        use super::*;

        // -- TryFrom --
        #[test]
        fn from_valid_symbols() {
            assert_eq!(BinaryOperatorKind::try_from(Lexeme::new("+", 0..1)), Ok(BinaryOperatorKind::Add));
            assert_eq!(BinaryOperatorKind::try_from(Lexeme::new("-", 0..1)), Ok(BinaryOperatorKind::Sub));
            assert_eq!(BinaryOperatorKind::try_from(Lexeme::new("*", 0..1)), Ok(BinaryOperatorKind::Mul));
            assert_eq!(BinaryOperatorKind::try_from(Lexeme::new("/", 0..1)), Ok(BinaryOperatorKind::Div));
            assert_eq!(BinaryOperatorKind::try_from(Lexeme::new("^", 0..1)), Ok(BinaryOperatorKind::Pow));
        }

        #[test]
        fn from_invalid_symbol() {
            assert!(BinaryOperatorKind::try_from(Lexeme::new("", 0..1)).is_err());
            assert!(BinaryOperatorKind::try_from(Lexeme::new("**", 0..2)).is_err());
            assert!(BinaryOperatorKind::try_from(Lexeme::new("!", 0..1)).is_err());
        }

        // -- precedence --
        #[test]
        fn precedence_ordering() {
            assert_eq!(BinaryOperatorKind::Add.precedence(), BinaryOperatorKind::Sub.precedence());
            assert!(BinaryOperatorKind::Mul.precedence() > BinaryOperatorKind::Add.precedence());
            assert!(BinaryOperatorKind::Div.precedence() > BinaryOperatorKind::Sub.precedence());
            assert!(BinaryOperatorKind::Pow.precedence() > BinaryOperatorKind::Mul.precedence());
        }

        // -- is_left_assoc --
        #[test]
        fn associativity() {
            assert!(BinaryOperatorKind::Add.is_left_assoc());
            assert!(BinaryOperatorKind::Sub.is_left_assoc());
            assert!(BinaryOperatorKind::Mul.is_left_assoc());
            assert!(BinaryOperatorKind::Div.is_left_assoc());
            assert!( ! BinaryOperatorKind::Pow.is_left_assoc() ); // right association
        }

        // -- apply: Real --
        #[test]
        fn apply_real() {
            let (a, b) = (c(6.0, 0.0), c(2.0, 0.0));
            assert_eq!(BinaryOperatorKind::Add.apply(a, b), c(8.0,  0.0));
            assert_eq!(BinaryOperatorKind::Sub.apply(a, b), c(4.0,  0.0));
            assert_eq!(BinaryOperatorKind::Mul.apply(a, b), c(12.0, 0.0));
            assert_eq!(BinaryOperatorKind::Div.apply(a, b), c(3.0,  0.0));
        }

        #[test]
        fn apply_pow_real() {
            assert!(eq(
                BinaryOperatorKind::Pow.apply(c(2.0, 0.0), c(10.0, 0.0)),
                c(1024.0, 0.0),
            ));
        }

        // -- apply: Complex --
        #[test]
        fn apply_add_complex() {
            // (1+2i) + (3+4i) = 4+6i
            assert_eq!(
                BinaryOperatorKind::Add.apply(c(1.0, 2.0), c(3.0, 4.0)),
                c(4.0, 6.0),
            );
        }

        #[test]
        fn apply_mul_complex() {
            // (1+i)(1-i) = 2
            assert!(eq(
                BinaryOperatorKind::Mul.apply(c(1.0, 1.0), c(1.0, -1.0)),
                c(2.0, 0.0),
            ));
        }

        #[test]
        fn apply_pow_complex() {
            // i^2 = -1
            assert!(eq(
                BinaryOperatorKind::Pow.apply(c(0.0, 1.0), c(2.0, 0.0)),
                c(-1.0, 0.0),
            ));
        }

        // -- apply --
        #[test]
        fn div_by_zero_does_not_panic() {
            let result = BinaryOperatorKind::Div.apply(c(1.0, 0.0), c(0.0, 0.0));
            assert!(result.re.is_infinite() || result.re.is_nan());
        }

        // -- symbols --
        #[test]
        fn symbols_contains_all() {
            let syms = BinaryOperatorKind::symbols();
            for s in ["+", "-", "*", "/", "^"] {
                assert!(syms.contains(&s), "missing symbol: {s}");
            }
        }

        // -- Display --
        #[test]
        fn display() {
            assert_eq!(BinaryOperatorKind::Add.to_string(), "+");
            assert_eq!(BinaryOperatorKind::Sub.to_string(), "-");
            assert_eq!(BinaryOperatorKind::Mul.to_string(), "*");
            assert_eq!(BinaryOperatorKind::Div.to_string(), "/");
            assert_eq!(BinaryOperatorKind::Pow.to_string(), "^");
        }
    }
}
