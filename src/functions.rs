//! # functions.rs
//!
//! This module defines built-in mathematical functions used in expressions,
//! and custom functions at runtime, which can be used in expression parsing and evaluation.

use num_complex::Complex;
use std::str::FromStr;
use std::sync::Arc;
use std::vec::Drain;

use crate::core::{
    ComplexMath,
    Real,
};
use crate::err::InitializeError;

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

/// Provides evaluation of a mathematical function.
///
/// Implementors receive their arguments as a vector of complex values and
/// return the resulting complex value. The expected number of arguments is
/// described separately by [`Arity::arity`].
///
/// Implementations are used by the expression evaluator for both built-in
/// and user-defined functions.
pub trait Apply<T: Real>
{
    /// Evaluates the function with the given arguments.
    fn apply(&self, arg: Vec<Complex<T>>) -> Complex<T>;
}

/// Describes the number of arguments accepted by a mathematical function.
pub trait Arity
{
    /// Returns the number of arguments this function expects.
    fn arity(&self) -> usize;
}

macro_rules! count_args {
    () => { 0usize };
    ($head:ident $(, $tail:ident)*) => { 1usize + count_args!($($tail),*)}
}

#[doc(hidden)]
/// Internal macro to define all functions
macro_rules! functions {
    ($( $variant:ident => {
        name:  $name:expr,
        apply: |$( $arg:ident ),+| $body:expr
    }, )*) => {
        /// Represents a built-in mathematical function.
        #[derive(Debug, Clone, Copy, PartialEq)]
        pub enum FunctionKind {
            $(
                #[doc = "A built-in mathematical function."]
                $variant,
            )*
        }

        impl FromStr for FunctionKind {
            type Err = (); // unknown only
            fn from_str(s: &str) -> Result<Self, Self::Err>
            {
                match s {
                    $( $name => Ok(Self::$variant), )*
                    _ => Err(())
                }
            }
        }

        impl FunctionKind {
            /// Returns all supported function name strings.
            pub fn symbols() -> &'static [&'static str] {
                &[$( $name, )*]
            }

            #[inline(always)]
            fn pop_array<T: Real, const N: usize>(stack: &mut Vec<Complex<T>>) -> [Complex<T>; N] {
                let base = stack.len() - N;
                let mut it = stack.drain(base..); // move in the same order
                std::array::from_fn(|_| it.next().unwrap()) // No alloc. No reverse.
            }

            #[inline]
            pub(crate) fn apply_stack<T: Real>(&self, stack: &mut Vec<Complex<T>>) {
                match self {
                    $( Self::$variant => {
                        const N: usize = count_args!($($arg),+);
                        let [$($arg),+] = Self::pop_array::<T, N>(stack);
                        stack.push($body);
                    } )*
                }
            }
        }

        impl Arity for FunctionKind
        {
            fn arity(&self) -> usize {
                match self {
                    $( Self::$variant => count_args!($($arg),+), )*
                }
            }
        }

        impl<T: Real> Apply<T> for FunctionKind
        {
            fn apply(&self, args: Vec<Complex<T>>) -> Complex<T> {
                match self {
                    $( Self::$variant => {
                        let mut it = args.into_iter();
                        $( let $arg = it.next().unwrap(); )+
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
}

functions! {
    Sin   => { name: "sin",     apply: |x| x.sin() },
    Cos   => { name: "cos",     apply: |x| x.cos() },
    Tan   => { name: "tan",     apply: |x| x.tan() },
    Asin  => { name: "asin",    apply: |x| x.asin() },
    Acos  => { name: "acos",    apply: |x| x.acos() },
    Atan  => { name: "atan",    apply: |x| x.atan() },
    Sinh  => { name: "sinh",    apply: |x| x.sinh() },
    Cosh  => { name: "cosh",    apply: |x| x.cosh() },
    Tanh  => { name: "tanh",    apply: |x| x.tanh() },
    Asinh => { name: "asinh",   apply: |x| x.asinh() },
    Acosh => { name: "acosh",   apply: |x| x.acosh() },
    Atanh => { name: "atanh",   apply: |x| x.atanh() },
    Exp   => { name: "exp",     apply: |x| x.exp() },
    Ln    => { name: "ln",      apply: |x| x.ln() },
    Log10 => { name: "log10",   apply: |x| x.log10() },
    Sqrt  => { name: "sqrt",    apply: |x| x.sqrt() },
    Abs   => { name: "abs",     apply: |x| x.abs() },
    Conj  => { name: "conj",    apply: |x| x.conj() },
    Pow   => { name: "pow",     apply: |x, y| x.powc(y) },
    Powi  => { name: "powi",    apply: |x, y| x.powi(y.re.to_i32()) },
}

#[cfg(test)]
mod function_tests {
    use super::*;

    fn c(re: f64, im: f64) -> Complex<f64> { Complex::new(re, im) }
    fn eq(a: Complex<f64>, b: Complex<f64>) -> bool { (a - b).norm() < 1e-10 }

    #[test]
    fn from_str_valid() {
        assert_eq!(FunctionKind::from_str("sin"), Ok(FunctionKind::Sin));
        assert_eq!(FunctionKind::from_str("cos"), Ok(FunctionKind::Cos));
        assert_eq!(FunctionKind::from_str("pow"), Ok(FunctionKind::Pow));
        assert_eq!(FunctionKind::from_str("powi"), Ok(FunctionKind::Powi));
    }

    #[test]
    fn from_str_invalid() {
        assert!(FunctionKind::from_str("SIN").is_err());
        assert!(FunctionKind::from_str("").is_err());
        assert!(FunctionKind::from_str("log").is_err());
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
        assert!(eq(FunctionKind::Sin.apply(vec![c(0.0, 0.0)]), c(0.0, 0.0)));
        assert!(eq(FunctionKind::Cos.apply(vec![c(0.0, 0.0)]), c(1.0, 0.0)));
    }

    #[test]
    fn apply_exp_ln_roundtrip() {
        let x = c(1.0, 1.0);
        let exp_x = FunctionKind::Exp.apply(vec![x]);
        assert!(eq(FunctionKind::Ln.apply(vec![exp_x]), x));
    }

    #[test]
    fn apply_abs_is_real() {
        assert!(eq(
            FunctionKind::Abs.apply(vec![c(3.0, 4.0)]),
            c(5.0, 0.0),
        ));
    }

    #[test]
    fn apply_pow_binary() {
        assert!(eq(
            FunctionKind::Pow.apply(vec![c(2.0, 0.0), c(8.0, 0.0)]),
            c(256.0, 0.0),
        ));
    }

    #[test]
    fn apply_powi_integer_exp() {
        assert!(eq(
            FunctionKind::Powi.apply(vec![c(3.0, 0.0), c(4.0, 0.0)]),
            c(81.0, 0.0),
        ));
    }

    #[test]
    fn display() {
        assert_eq!(FunctionKind::Sin.to_string(),   "sin");
        assert_eq!(FunctionKind::Log10.to_string(), "log10");
        assert_eq!(FunctionKind::Pow.to_string(),   "pow");
    }
}

/// Closure type for user-defined custom functions.
type FuncType<T> = dyn Fn(&mut Drain<'_, Complex<T>>) -> Complex<T> + Send + Sync;

/// A user-defined mathematical function.
///
/// A `UserFn` stores a named function, its arity, and optionally one analytic
/// derivative function for each argument. User-defined functions can be
/// registered with [`crate::builder::Builder::with_user_functions`] and then
/// called from formulas by name.
///
/// The function and all registered derivatives are required to be thread-safe
/// so that a compiled formula can be evaluated concurrently.
#[derive(Clone)]
pub struct UserFn<T: Real>
{
    inner: Arc<UserFnInner<T>>,
}

#[derive(Clone)]
struct UserFnInner<T: Real>
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
    /// * `func`  - A closure that receives an array of `N` complex values and returns a complex value.
    pub fn new<F, S, const N: usize>(name: S, func: F) -> Self
    where
        F: Fn([Complex<T>; N]) -> Complex<T> + Send + Sync + 'static,
        S: Into<String>,
    {
        let inner = UserFnInner {
            func: Arc::new(move |it| {
                let arr = std::array::from_fn(|_| it.next().expect("arity mismatch"));
                func(arr)
            }),
            deriv: Vec::new(),
            arity: N,
            name: name.into(),
        };

        Self {
            inner: Arc::new(inner),
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
    pub fn with_derivative(mut self, diffs: impl IntoIterator<Item = Self>) -> Result<Self, InitializeError> {
        let diffs: Vec<Self> = diffs.into_iter().collect();
        if diffs.len() != self.inner.arity {
            return Err(InitializeError::DerivativesNumberMismatched {
                expected: self.inner.arity, number: diffs.len()
            });
        }
        Arc::make_mut(&mut self.inner).deriv = diffs;

        Ok(self)
    }

    /// Returns the function name.
    pub fn name(&self) -> &str {
        &self.inner.name
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
    /// ).with_derivative(vec![df])
    /// .expect("Mistake derivative count");
    ///
    /// assert!(f.derivative(0).is_some());
    /// assert!(f.derivative(1).is_none()); // out of range
    /// ```
    pub fn derivative(&self, var: usize) -> Option<&Self> {
        self.inner.deriv.get(var)
    }

    pub(crate) fn apply_stack(&self, stack: &mut Vec<Complex<T>>) {
        let base = stack.len() - self.inner.arity;
        let mut drain = stack.drain(base..);
        let result = (self.inner.func)(&mut drain);
        drop(drain);
        stack.push(result)
    }
}

impl<T: Real> Arity for UserFn<T>
{
    fn arity(&self) -> usize {
        self.inner.arity
    }
}

impl<T: Real> Apply<T> for UserFn<T> {
    fn apply(&self, mut args: Vec<Complex<T>>) -> Complex<T> {
        (self.inner.func)(&mut args.drain(..))
    }
}

impl<T: Real> std::fmt::Debug for UserFn<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UserFn")
            .field("name",  &self.inner.name)
            .field("arity", &self.inner.arity)
            .finish_non_exhaustive()
    }
}

impl<T: Real> PartialEq for UserFn<T> {
    /// Equality is based on `name` and `arity` only (closure cannot be compared).
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner) ||
        self.inner.name == other.inner.name && self.inner.arity == other.inner.arity
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
        assert_eq!(f.apply(vec![Complex::ZERO]), Complex::ONE);
    }

    #[test]
    fn apply_binary() {
        let f = UserFn::new(
            "add",
            |[x, y]| x + y,
        );
        assert_eq!(
            f.apply(vec![c(1.0, 0.0), c(2.0, 0.0)]),
            c(3.0, 0.0),
        );
    }

    #[test]
    fn apply_ternary() {
        let f = UserFn::new(
            "sum",
            |[x, y, z]| x + y + z,
        );
        assert_eq!(
            f.apply(vec![c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0)]),
            c(6.0, 0.0),
        );

        let f = UserFn::new(
            "poly",
            |[x, y, z]|
                c(100.0, 0.0) * x + c(10.0, 0.0) * y + z
        );
        assert_eq!(
            f.apply(vec![c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0)]),
            c(123.0, 0.0),
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
        ).with_derivative(vec![df])
        .unwrap();

        let deriv = f.derivative(0).expect("should exist");
        let result = deriv.apply(vec![c(4.0, 0.0)]);
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
