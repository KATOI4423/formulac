//! # functions.rs
//!
//! This module defines built-in mathematical functions used in expressions,
//! and custom functions at runtime, which can be used in expression parsing and evaluation.

use num_complex::{Complex, ComplexFloat};
use std::str::FromStr;
use std::sync::Arc;

/// Error type for parsing function strings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ParseFunctionError {
    Unknown,
}

/// Typed arguments passed to a built-in function.
///
/// Enforces at the type level that unary and binary functions
/// receive the correct number of arguments.
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
/// - `apply(&self, args: FunctionArgs) -> Complex<f64>`
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
mod function_tests {
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

/// Tye closure type for user-defined custom functions.
type FuncType = dyn Fn(FunctionArgs) -> Complex<f64> + Send + Sync;

#[derive(Clone)]
pub struct UserFn {
    func: Arc<FuncType>,
    deriv: Vec<UserFn>,
    arity: usize,
    name: String,
}

impl UserFn {
    /// Creates a new `UserFn`.
    ///
    /// # Arguments
    ///
    /// * `name`  - The name of the function.
    /// * `func`  - A closure that receives [`FunctionArgs`] and returns `Complex<f64>`.
    /// * `arity` - The number of arguments the function expects.
    pub fn new<F, S>(name: S, func: F, arity: usize) -> Self
    where
        F: Fn(FunctionArgs) -> Complex<f64> + Send + Sync + 'static,
        S: Into<String>,
    {
        Self {
            func: Arc::new(func),
            deriv: Vec::new(),
            arity,
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
    /// use formulac::functions::{
    ///     UserFn,
    ///     FunctionArgs,
    ///     FunctionCall,
    /// };
    ///
    /// let df = UserFn::new(
    ///     "square_deriv",
    ///     |args| {
    ///         let FunctionArgs::Unary(x) = args else { unreachable!() };
    ///         Complex::new(2.0, 0.0) * x
    ///     },
    ///     1,
    /// );
    /// let f = UserFn::new(
    ///     "square",
    ///     |args| {
    ///         let FunctionArgs::Unary(x) = args else { unreachable!() };
    ///         x * x
    ///     },
    ///     1,
    /// ).with_derivative(vec![df]);
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
    /// use formulac::functions::{
    ///     UserFn,
    ///     FunctionArgs,
    ///     FunctionCall,
    /// };
    ///
    /// let df = UserFn::new(
    ///     "deriv",
    ///     |args| {
    ///         let FunctionArgs::Unary(x) = args else { unreachable!() };
    ///         Complex::new(2.0, 0.0) * x
    ///     },
    ///     1,
    /// );
    /// let f = UserFn::new(
    ///     "square",
    ///     |args| {
    ///         let FunctionArgs::Unary(x) = args else { unreachable!() };
    ///         x * x
    ///     },
    ///     1,
    /// ).with_derivative(vec![df]);
    ///
    /// assert!(f.derivative(0).is_some());
    /// assert!(f.derivative(1).is_none()); // out of range
    /// ```
    pub fn derivative(&self, var: usize) -> Option<&Self> {
        self.deriv.get(var)
    }

    /// Returns a new `UserFn` that numerically differentiates
    /// this function with respect to argument `var` using the central difference method.
    ///
    /// Falls back to this when no analytic derivative is registered.
    /// Returns `None` if `var >= arity`.
    ///
    /// The step size `dh` is chosen as `args[var] * 1e-6` to scale with the input.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use num_complex::Complex;
    /// use formulac::functions::{
    ///     UserFn,
    ///     FunctionArgs,
    ///     FunctionCall,
    /// };
    ///
    /// // f(x) = x^3, f'(x) ≈ 3x^2
    /// let f = UserFn::new(
    ///     "cube",
    ///     |args| {
    ///         let FunctionArgs::Unary(x) = args else { unreachable!() };
    ///         x * x * x
    ///     },
    ///     1,
    /// );
    ///
    /// let ndf = f.numeric_deriv(0).unwrap();
    /// let result = ndf.apply(FunctionArgs::Unary(Complex::new(2.0, 0.0)));
    /// // f'(2) ≈ 12.0
    /// assert!((result.re - 12.0).abs() < 1e-6);
    /// ```
    pub fn numeric_deriv(&self, var: usize) -> Option<Self> {
        if var >= self.arity {
            return None;
        }

        let func  = self.func.clone();
        let arity = self.arity;
        let name  = format!("{}_num_diff_arg{}", self.name, var);

        Some(UserFn::new(
            &name,
            move |args| {
                // FunctionArgs を &[Complex<f64>] に展開して中央差分を計算する
                let mut argv: Vec<Complex<f64>> = match args {
                    FunctionArgs::Unary(x)       => vec![x],
                    FunctionArgs::Binary(x, y)   => vec![x, y],
                };
                let dh = argv[var] * 1.0e-6;
                argv[var] += dh;
                let plus  = func(Self::to_args(arity, &argv));
                argv[var] -= dh * 2.0;
                let minus = func(Self::to_args(arity, &argv));
                (plus - minus) / (dh * 2.0)
            },
            arity,
        ))
    }

    /// Helper to convert a `Vec<Complex<f64>>` back to `FunctionArgs` by arity.
    fn to_args(arity: usize, v: &[Complex<f64>]) -> FunctionArgs {
        match arity {
            1 => FunctionArgs::Unary(v[0]),
            2 => FunctionArgs::Binary(v[0], v[1]),
            _ => unreachable!("unsupported arity: {}", arity),
        }
    }
}

impl FunctionCall for UserFn {
    fn apply(&self, args: FunctionArgs) -> Complex<f64> {
        (self.func)(args)
    }

    fn arity(&self) -> usize {
        self.arity
    }
}

impl std::fmt::Debug for UserFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UserFn")
            .field("name",  &self.name)
            .field("arity", &self.arity)
            .finish_non_exhaustive()
    }
}

impl PartialEq for UserFn {
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
            |args| {
                let FunctionArgs::Unary(x) = args else { unreachable!() };
                x + Complex::ONE
            },
            1,
        );
        assert_eq!(f.apply(FunctionArgs::Unary(Complex::ZERO)), Complex::ONE);
    }

    #[test]
    fn apply_binary() {
        let f = UserFn::new(
            "add",
            |args| {
                let FunctionArgs::Binary(x, y) = args else { unreachable!() };
                x + y
            },
            2,
        );
        assert_eq!(
            f.apply(FunctionArgs::Binary(c(1.0, 0.0), c(2.0, 0.0))),
            c(3.0, 0.0),
        );
    }

    #[test]
    fn arity() {
        let f = UserFn::new("f", |args| {
            let FunctionArgs::Binary(x, y) = args else { unreachable!() };
            x + y
        }, 2);
        assert_eq!(f.arity(), 2);
    }

    #[test]
    fn partial_eq() {
        let f1 = UserFn::new("f", |args| { let FunctionArgs::Unary(x) = args else { unreachable!() }; x }, 1);
        let f2 = UserFn::new("f", |args| { let FunctionArgs::Unary(x) = args else { unreachable!() }; x + x }, 1);
        let f3 = UserFn::new("g", |args| { let FunctionArgs::Unary(x) = args else { unreachable!() }; x }, 1);
        let f4 = UserFn::new("f", |args| { let FunctionArgs::Binary(x, y) = args else { unreachable!() }; x + y }, 2);
        assert_eq!(f1, f2);
        assert_ne!(f1, f3);
        assert_ne!(f1, f4);
    }

    #[test]
    fn without_derivative() {
        let f = UserFn::new("f", |args| { let FunctionArgs::Unary(x) = args else { unreachable!() }; x * x }, 1);
        assert!(f.derivative(0).is_none());
    }

    #[test]
    fn with_analytic_derivative() {
        let df = UserFn::new(
            "square_deriv",
            |args| {
                let FunctionArgs::Unary(x) = args else { unreachable!() };
                c(2.0, 0.0) * x
            },
            1,
        );
        let f = UserFn::new(
            "square",
            |args| { let FunctionArgs::Unary(x) = args else { unreachable!() }; x * x },
            1,
        ).with_derivative(vec![df]);

        let deriv = f.derivative(0).expect("should exist");
        let result = deriv.apply(FunctionArgs::Unary(c(4.0, 0.0)));
        assert_abs_diff_eq!(result.re, 8.0, epsilon = 1e-12);
        assert_abs_diff_eq!(result.im, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn numeric_deriv_cube() {
        // f(x) = x^3, f'(x) = 3x^2, f'(2) = 12
        let f = UserFn::new(
            "cube",
            |args| { let FunctionArgs::Unary(x) = args else { unreachable!() }; x * x * x },
            1,
        );
        let ndf = f.numeric_deriv(0).unwrap();
        let result = ndf.apply(FunctionArgs::Unary(c(2.0, 0.0)));
        assert_abs_diff_eq!(result.re, 12.0, epsilon = 1e-5);
        assert_abs_diff_eq!(result.im,  0.0, epsilon = 1e-5);
    }

    #[test]
    fn numeric_deriv_out_of_range() {
        let f = UserFn::new(
            "f",
            |args| { let FunctionArgs::Unary(x) = args else { unreachable!() }; x },
            1,
        );
        assert!(f.numeric_deriv(1).is_none());
    }

    #[test]
    fn debug_contains_name_and_arity() {
        let f = UserFn::new(
            "mul",
            |args| { let FunctionArgs::Unary(x) = args else { unreachable!() }; x * x },
            1,
        );
        let s = format!("{:?}", f);
        assert!(s.contains("mul"));
        assert!(s.contains("arity"));
    }
}
