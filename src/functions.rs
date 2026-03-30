//! # functions.rs
//!
//! This module defines built-in mathematical functions used in expressions,
//! and custom functions at runtime, which can be used in expression parsing and evaluation.

use num_complex::Complex;
use std::sync::Arc;

use crate::core::{
    ComplexMath,
    Real,
};
use crate::err::ParseError;
use crate::lexer::Lexeme;

/// Typed arguments passed to a built-in function.
///
/// Enforces at the type level that unary and binary functions
/// receive the correct number of arguments.
#[derive(Clone, Debug, PartialEq)]
pub enum FunctionArgs<T: Real>
{
    Unary(Complex<T>),
    Binary(Complex<T>, Complex<T>),
}

impl<T: Real> FunctionArgs<T>
{
    /// Constructs `FunctionArgs` from a slice, based on length.
    pub(crate) fn from(args: impl IntoIterator<Item = Complex<T>>) -> Self {
        let mut args = args.into_iter();
        match (args.next(), args.next()) {
            (Some(a), None) => Self::Unary(a),
            (Some(a), Some(b)) => Self::Binary(a, b),
            _ => unreachable!("unsupported arity"),
        }
    }
}

pub trait FromFunctionArgs<T: Real, const N: usize>
{
    fn from_args(args: FunctionArgs<T>) -> Self;
}

impl<T: Real> FromFunctionArgs<T, 1> for [Complex<T>; 1]
{
    fn from_args(args: FunctionArgs<T>) -> Self
    {
        let FunctionArgs::<T>::Unary(x) = args else { unreachable!("arity mismatch") };
        [x]
    }
}

impl<T: Real> FromFunctionArgs<T, 2> for [Complex<T>; 2]
{
    fn from_args(args: FunctionArgs<T>) -> Self
    {
        let FunctionArgs::<T>::Binary(x, y) = args else { unreachable!("arity mismatch") };
        [x, y]
    }
}

/// A trait representing a callable mathematical function.
///
/// This trait is implemented by types that can be called with a fixed number
/// of arguments and return a `Complex<T>` result. It is used in the AST
/// for evaluating both built-in functions (like `sin`, `cos`, `pow`) and
/// user-defined functions.
///
/// # Methods
///
/// - `apply(&self, args: FunctionArgs<T>) -> Complex<T>`
///   Evaluates the function with the given arguments. The length of `args`
///   must match the function's arity.
///
/// - `arity(&self) -> usize`
///   Returns the number of arguments the function expects.
pub trait FunctionCall<T: Real>: Apply<T> + Arity {}

impl<T: Real, U> FunctionCall<T> for U
where
    U: Apply<T> + Arity,
{}

pub trait Apply<T: Real>
{
    /// Evaluates the function with the given arguments.
    fn apply(&self, arg: FunctionArgs<T>) -> Complex<T>;
}

pub trait Arity
{
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
        pub enum FunctionKind {
            $( $variant, )*
        }

        impl TryFrom<Lexeme> for FunctionKind {
            type Error = ParseError;
            fn try_from(s: Lexeme) -> Result<Self, Self::Error> {
                match s.text() {
                    $( $name => Ok(Self::$variant), )*
                    _ => Err(ParseError::UnknownToken(s)),
                }
            }
        }

        impl FunctionKind {
            /// Returns all supported function name strings.
            pub fn symbols() -> &'static [&'static str] {
                &[$( $name, )*]
            }
        }

        impl Arity for FunctionKind
        {
            fn arity(&self) -> usize {
                match self {
                    $( Self::$variant => functions!(@arity $kind), )*
                }
            }
        }

        impl<T: Real> Apply<T> for FunctionKind
        {
            fn apply(&self, args: FunctionArgs<T>) -> Complex<T> {
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
    Powi  => { name: "powi",  arity: Binary, apply: |x, y| x.powi(y.re.to_i32()) },
}

#[cfg(test)]
mod function_tests {
    use super::*;

    fn c(re: f64, im: f64) -> Complex<f64> { Complex::new(re, im) }
    fn eq(a: Complex<f64>, b: Complex<f64>) -> bool { (a - b).norm() < 1e-10 }

    #[test]
    fn try_from_valid() {
        assert_eq!(FunctionKind::try_from(Lexeme::new("sin", 0..4)), Ok(FunctionKind::Sin));
        assert_eq!(FunctionKind::try_from(Lexeme::new("cos", 0..4)), Ok(FunctionKind::Cos));
        assert_eq!(FunctionKind::try_from(Lexeme::new("pow", 0..4)), Ok(FunctionKind::Pow));
        assert_eq!(FunctionKind::try_from(Lexeme::new("powi", 0..5)), Ok(FunctionKind::Powi));
    }

    #[test]
    fn try_from_invalid() {
        assert!(FunctionKind::try_from(Lexeme::new("SIN", 0..4)).is_err());
        assert!(FunctionKind::try_from(Lexeme::new("", 0..1)).is_err());
        assert!(FunctionKind::try_from(Lexeme::new("log", 0..4)).is_err());
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

/// Tye closure type for user-defined custom functions.
type FuncType<T> = dyn Fn(FunctionArgs<T>) -> Complex<T> + Send + Sync;

#[derive(Clone)]
pub struct UserFn<T: Real>
{
    func: Arc<FuncType<T>>,
    deriv: Vec<UserFn<T>>,
    arity: usize,
    name: String,
}

impl<T: Real> UserFn<T> {
    /// Creates a new `UserFn`.
    ///
    /// # Arguments
    ///
    /// * `name`  - The name of the function.
    /// * `func`  - A closure that receives `[Complex<f64>; N]` and returns `Complex<f64>`.
    pub fn new<F, S, const N: usize>(name: S, func: F) -> Self
    where
        F: Fn([Complex<T>; N]) -> Complex<T> + Send + Sync + 'static,
        S: Into<String>,
        [Complex<T>; N]: FromFunctionArgs<T, N>,
    {
        Self {
            func: Arc::new(move |args| {
                let arr = <[Complex<T>; N]>::from_args(args);
                func(arr)
            }),
            deriv: Vec::new(),
            arity: N,
            name: name.into(),
        }
    }

    /// Attaches derivative functions, one per argument.
    ///
    /// The length of `diffs` must equal `self.arity`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use num_complex::Complex;
    /// use formulac::functions::UserFn;
    ///
    /// let df = UserFn::new(
    ///     "square_deriv",
    ///     |[x]| Complex::new(2.0, 0.0) * x,
    /// );
    /// let f = UserFn::new(
    ///     "square",
    ///     |[x]| x * x,
    /// ).with_derivative([df]);
    /// ```
    pub fn with_derivative(mut self, diffs: impl IntoIterator<Item = Self>) -> Self {
        let diffs: Vec<Self> = diffs.into_iter().collect();
        debug_assert_eq!(diffs.len(), self.arity);
        self.deriv = diffs;
        self
    }

    /// Returns the function name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the analytically registered derivative for argument `var`,
    /// or `None` if not registered.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use num_complex::Complex;
    /// use formulac::functions::UserFn;
    ///
    /// let df = UserFn::new(
    ///     "deriv",
    ///     |[x]| Complex::new(2.0, 0.0) * x,
    /// );
    /// let f = UserFn::new(
    ///     "square",
    ///     |[x]| x * x,
    /// ).with_derivative(vec![df]);
    ///
    /// assert!(f.derivative(0).is_some());
    /// assert!(f.derivative(1).is_none()); // out of range
    /// ```
    pub fn derivative(&self, var: usize) -> Option<&Self> {
        self.deriv.get(var)
    }
}

impl<T: Real> Arity for UserFn<T>
{
    fn arity(&self) -> usize {
        self.arity
    }
}

impl<T: Real> Apply<T> for UserFn<T> {
    fn apply(&self, args: FunctionArgs<T>) -> Complex<T> {
        (self.func)(args)
    }
}

impl<T: Real> std::fmt::Debug for UserFn<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UserFn")
            .field("name",  &self.name)
            .field("arity", &self.arity)
            .finish_non_exhaustive()
    }
}

impl<T: Real> PartialEq for UserFn<T> {
    /// Equality is based on `name` and `arity` only (closure cannot be compared).
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.arity == other.arity
    }
}

#[cfg(test)]
mod userfn_tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn c(re: f64, im: f64) -> Complex<f64> { Complex::new(re, im) }

    #[test]
    fn apply_unary() {
        let f = UserFn::new(
            "inc",
            |[x] : [Complex<f64>; 1]| x + Complex::ONE,
        );
        assert_eq!(f.apply(FunctionArgs::Unary(Complex::ZERO)), Complex::ONE);
    }

    #[test]
    fn apply_binary() {
        let f = UserFn::new(
            "add",
            |[x, y]| x + y,
        );
        assert_eq!(
            f.apply(FunctionArgs::Binary(c(1.0, 0.0), c(2.0, 0.0))),
            c(3.0, 0.0),
        );
    }

    #[test]
    fn partial_eq() {
        let f1 = UserFn::new("f", |[x] : [Complex<f64>; 1]| x);
        let f2 = UserFn::new("f", |[x] : [Complex<f64>; 1]| x + x);
        let f3 = UserFn::new("g", |[x] : [Complex<f64>; 1]| x);
        let f4 = UserFn::new("f", |[x, y] : [Complex<f64>; 2]| x + y);
        assert_eq!(f1, f2);
        assert_ne!(f1, f3);
        assert_ne!(f1, f4);
    }

    #[test]
    fn without_derivative() {
        let f = UserFn::new("f", |[x] : [Complex<f64>; 1]| x * x);
        assert!(f.derivative(0).is_none());
    }

    #[test]
    fn with_analytic_derivative() {
        let df = UserFn::new(
            "square_deriv",
            |[x]| c(2.0, 0.0) * x,
        );
        let f = UserFn::new(
            "square",
            |[x]| x * x,
        ).with_derivative(vec![df]);

        let deriv = f.derivative(0).expect("should exist");
        let result = deriv.apply(FunctionArgs::Unary(c(4.0, 0.0)));
        assert_abs_diff_eq!(result.re, 8.0, epsilon = 1e-12);
        assert_abs_diff_eq!(result.im, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn debug_contains_name_and_arity() {
        let f = UserFn::new(
            "mul",
            |[x] : [Complex<f64>; 1]| x * x,
        );
        let s = format!("{:?}", f);
        assert!(s.contains("mul"));
        assert!(s.contains("arity"));
    }
}
