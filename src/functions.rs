//! # functions.rs
//!
//! This module defines built-in mathematical functions used in expressions.

use num_complex::{Complex, ComplexFloat};
use std::str::FromStr;

/// Error type for parsing function strings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ParseFunctionError {
    Unknown,
}

/// Typed arguments passed to a built-in function.
///
/// Enforces at the type level that unary and bunary functions
/// recieve the correct number of arguments.
#[derive(Clone, Debug, PartialEq)]
pub enum FunctionArgs {
    Unary(Complex<f64>),
    Binary(Complex<f64>, Complex<f64>),
}

/// A trait representing a callable mathematical function.
///
/// This trait is implemented by types that can be called with a fixed number
/// of arguments and return a `Complex<f64>` result. It is used in the AST
/// for evaluating both built-in functions (like `sin`, `cos`, `pow`) and
/// user-defined functions.
///
/// # Methods
///
/// - `apply(&self, args: &[Complex<f64>]) -> Complex<f64>`
///   Evaluates the function with the given arguments. The length of `args`
///   must match the function's arity.
///
/// - `arity(&self) -> usize`
///   Returns the number of arguments the function expects.
pub trait FunctionCall {
    /// Evaluates the function with the given arguments.
    fn apply(&self, arg: FunctionArgs) -> Complex<f64>;

    /// Returns the number of arguments this function expects.
    fn arity(&self) -> usize;
}

#[doc(hidden)]
/// Internal macro to define all functions
macro_rules! functions {
    ($( $variant:ident => {
        name:  $name:expr,
        arity: $kind:ident,
        apply: |$( $arg:ident ),+| $body:expr
    }, )*) => {
        /// Represents a built-in mathematical function.
        #[derive(Debug, Clone, Copy, PartialEq)]
        pub(crate) enum FunctionKind {
            $( $variant, )*
        }

        impl FromStr for FunctionKind {
            type Err = ParseFunctionError;
            fn from_str(s: &str) -> Result<Self, Self::Err> {
                match s {
                    $( $name => Ok(Self::$variant), )*
                    _ => Err(ParseFunctionError::Unknown),
                }
            }
        }

        impl FunctionKind {
            /// Returns all supported function name strings.
            pub fn symbols() -> &'static [&'static str] {
                &[$( $name, )*]
            }
        }

        impl FunctionCall for FunctionKind {
            fn arity(&self) -> usize {
                match self {
                    $( Self::$variant => functions!(@arity $kind), )*
                }
            }

            fn apply(&self, args: FunctionArgs) -> Complex<f64> {
                match self {
                    $( Self::$variant => {
                        functions!(@destructure $kind, args, $( $arg ),+);
                        $body
                    }, )*
                }
            }
        }

        impl std::fmt::Display for FunctionKind {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    $( Self::$variant => write!(f, $name), )*
                }
            }
        }
    };

    (@arity Unary) => { 1 };
    (@destructure Unary, $args:expr, $x:ident) => {
        let FunctionArgs::Unary($x) = $args else {
            unreachable!("arity mismatch: expected Unary")
        };
    };

    (@arity Binary) => { 2 };
    (@destructure Binary, $args: expr, $x:ident, $y:ident) => {
        let FunctionArgs::Binary($x, $y) = $args else {
            unreachable!("arity mismatch: expected Binary")
        };
    };
}

functions! {
    Sin   => { name: "sin",   arity: Unary, apply: |x| x.sin() },
    Cos   => { name: "cos",   arity: Unary, apply: |x| x.cos() },
    Tan   => { name: "tan",   arity: Unary, apply: |x| x.tan() },
    Asin  => { name: "asin",  arity: Unary, apply: |x| x.asin() },
    Acos  => { name: "acos",  arity: Unary, apply: |x| x.acos() },
    Atan  => { name: "atan",  arity: Unary, apply: |x| x.atan() },
    Sinh  => { name: "sinh",  arity: Unary, apply: |x| x.sinh() },
    Cosh  => { name: "cosh",  arity: Unary, apply: |x| x.cosh() },
    Tanh  => { name: "tanh",  arity: Unary, apply: |x| x.tanh() },
    Asinh => { name: "asinh", arity: Unary, apply: |x| x.asinh() },
    Acosh => { name: "acosh", arity: Unary, apply: |x| x.acosh() },
    Atanh => { name: "atanh", arity: Unary, apply: |x| x.atanh() },
    Exp   => { name: "exp",   arity: Unary, apply: |x| x.exp() },
    Ln    => { name: "ln",    arity: Unary, apply: |x| x.ln() },
    Log10 => { name: "log10", arity: Unary, apply: |x| x.log10() },
    Sqrt  => { name: "sqrt",  arity: Unary, apply: |x| x.sqrt() },
    Abs   => { name: "abs",   arity: Unary, apply: |x| Complex::from(x.abs()) },
    Conj  => { name: "conj",  arity: Unary, apply: |x| x.conj() },
    Pow   => { name: "pow",   arity: Binary, apply: |x, y| x.powc(y) },
    Powi  => { name: "powi",  arity: Binary, apply: |x, y| x.powi(y.re as i32) },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(re: f64, im: f64) -> Complex<f64> { Complex::new(re, im) }
    fn eq(a: Complex<f64>, b: Complex<f64>) -> bool { (a - b).norm() < 1e-10 }

    #[test]
    fn from_str_valid() {
        assert_eq!("sin".parse::<FunctionKind>(),  Ok(FunctionKind::Sin));
        assert_eq!("cos".parse::<FunctionKind>(),  Ok(FunctionKind::Cos));
        assert_eq!("pow".parse::<FunctionKind>(),  Ok(FunctionKind::Pow));
        assert_eq!("powi".parse::<FunctionKind>(), Ok(FunctionKind::Powi));
    }

    #[test]
    fn from_str_invalid() {
        assert_eq!("SIN".parse::<FunctionKind>(), Err(ParseFunctionError::Unknown));
        assert_eq!("".parse::<FunctionKind>(),    Err(ParseFunctionError::Unknown));
        assert_eq!("log".parse::<FunctionKind>(), Err(ParseFunctionError::Unknown));
    }

    #[test]
    fn arity_unary() {
        for f in [FunctionKind::Sin, FunctionKind::Cos, FunctionKind::Exp,
                  FunctionKind::Ln,  FunctionKind::Sqrt, FunctionKind::Abs] {
            assert_eq!(f.arity(), 1);
        }
    }

    #[test]
    fn arity_binary() {
        assert_eq!(FunctionKind::Pow.arity(),  2);
        assert_eq!(FunctionKind::Powi.arity(), 2);
    }

    #[test]
    fn apply_sin_cos() {
        assert!(eq(FunctionKind::Sin.apply(FunctionArgs::Unary(c(0.0, 0.0))), c(0.0, 0.0)));
        assert!(eq(FunctionKind::Cos.apply(FunctionArgs::Unary(c(0.0, 0.0))), c(1.0, 0.0)));
    }

    #[test]
    fn apply_exp_ln_roundtrip() {
        let x = c(1.0, 1.0);
        let exp_x = FunctionKind::Exp.apply(FunctionArgs::Unary(x));
        assert!(eq(FunctionKind::Ln.apply(FunctionArgs::Unary(exp_x)), x));
    }

    #[test]
    fn apply_abs_is_real() {
        assert!(eq(
            FunctionKind::Abs.apply(FunctionArgs::Unary(c(3.0, 4.0))),
            c(5.0, 0.0),
        ));
    }

    #[test]
    fn apply_pow_binary() {
        assert!(eq(
            FunctionKind::Pow.apply(FunctionArgs::Binary(c(2.0, 0.0), c(8.0, 0.0))),
            c(256.0, 0.0),
        ));
    }

    #[test]
    fn apply_powi_integer_exp() {
        assert!(eq(
            FunctionKind::Powi.apply(FunctionArgs::Binary(c(3.0, 0.0), c(3.0, 0.0))),
            c(27.0, 0.0),
        ));
    }

    #[test]
    fn display() {
        assert_eq!(FunctionKind::Sin.to_string(),   "sin");
        assert_eq!(FunctionKind::Log10.to_string(), "log10");
        assert_eq!(FunctionKind::Pow.to_string(),   "pow");
    }
}
