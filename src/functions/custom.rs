//! # custom.rs
//!
//! User-defined function support with numerical differentiation.
//!
//! This module provides `CustomFunc<T, S>` for creating and managing user-defined
//! functions with arbitrary arity. It supports both analytical and numerical
//! derivative computation, making it suitable for symbolic differentiation and
//! automatic differentiation applications.

use crate::functions::core::{
    ComplexBackend,
};

use num_traits::identities::{
    One,
};
use std::io::{
    ErrorKind,
};
use std::ops::{
    AddAssign,
    Sub, SubAssign,
    Mul,
    Div,
};
use std::sync::Arc;
use smallvec::{
    SmallVec,
};

type Error = Box<dyn std::error::Error + Send + Sync + 'static>;

const ARITY_THRESH: usize = 4;

/// User-defined function with derivative support.
///
/// `CustomFunc<T, S>` represents a function with arbitrary arity that can compute
/// numerical derivatives. The generic parameters allow flexibility in complex number
/// representation (T) and scalar type (S) for multi-precision support.
///
/// Derivatives are stored efficiently using `SmallVec`, which inlines up to
/// `ARITY_THRESH` derivatives before heap allocation.
///
/// # Type Parameters
///
/// * `T` - Complex number type implementing `ComplexBackend<S>`
/// * `S` - Scalar type (e.g., `f64` for standard precision)
#[derive(Clone)] // The Debug for CuctomFunc can't be derived automatically because it includes a closure.
pub(crate) struct CustomFunc<T, S>
where
    T: ComplexBackend<S> + Send + Sync + 'static,
    S: Clone + Send + Sync + 'static,
{
    func: Arc<dyn Fn(&[T]) -> T + Send + Sync>,
    deriv: SmallVec<[Arc<CustomFunc<T, S>>; ARITY_THRESH]>,
    arity: usize,
}

impl<T, S> CustomFunc<T, S>
where
    T: ComplexBackend<S> + Send + Sync + 'static,
    S: Clone + Send + Sync + 'static
{
    /// Creates a new custom function.
    ///
    /// # Arguments
    ///
    /// * `func` - A closure implementing the function logic
    /// * `arity` - Number of arguments the function expects
    pub fn new<F>(func: F, arity: usize) -> Self
    where
        F: Fn(&[T]) -> T + Send + Sync + 'static,
    {
        Self {
            func: Arc::new(func),
            deriv: SmallVec::new(),
            arity,
        }
    }

    /// Helper function for creating errors.
    fn err<V>(kind: ErrorKind, msg: String) -> Result<V, Error>
    {
        Err(Box::new(std::io::Error::new(kind, msg)))
    }

    /// Sets analytical derivative functions.
    ///
    /// Associates explicit derivative functions with this function for each argument.
    /// The number of derivatives must match the function's arity.
    ///
    /// # Arguments
    ///
    /// * `derivatives` - Iterator of derivative functions, one for each argument
    ///
    /// # Errors
    ///
    /// Returns error if the number of derivatives does not match arity.
    pub fn with_derivatives<I>(mut self, derivatives: I) -> Result<Self, Error>
    where
        I: IntoIterator<Item = CustomFunc<T, S>>,
    {
        let deriv : SmallVec<[Arc<CustomFunc<T, S>>; ARITY_THRESH]>
            = derivatives.into_iter().map(Arc::new).collect();

        if deriv.len() != self.arity {
            Self::err(
                ErrorKind::InvalidInput,
                format!("expected {} derivative functions, got {}", self.arity, deriv.len())
            )?;
        }

        self.deriv = deriv;
        Ok(self)
    }

    /// Computes numerical derivative via central difference method.
    ///
    /// Approximates the partial derivative with respect to argument at index `idx`
    /// using the central difference formula:
    /// `(f(x+h) - f(x-h)) / (2h)`
    ///
    /// The step size `h` is automatically scaled based on the argument magnitude
    /// to balance truncation and rounding errors.
    ///
    /// # Arguments
    ///
    /// * `idx` - Index of the argument to differentiate with respect to
    ///
    /// # Errors
    ///
    /// Returns error if `idx >= arity`.
    pub fn numeric_deriv(&self, idx: usize) -> Result<Self, Error>
    where
        T: Clone
         + AddAssign
         + Sub<Output = T> + SubAssign
         + Mul<Output = T>
         + Div<Output = T>,
        S: PartialOrd + One,
    {
        if idx >= self.arity {
            Self::err(
                ErrorKind::InvalidInput,
                format!("idx '{}' must be smaller than the arity '{}'", idx, self.arity)
            )?;
        };

        let func = self.func.clone();
        let arity = self.arity;

        Ok(CustomFunc::new(
            move |args: &[T]| {
                debug_assert_eq!(
                    args.len(), arity,
                    "expected {} derivative functions, got {}", arity, args.len()
                );

                let eps = T::eps().sqrt();
                let two = T::two();
                let abs = args[idx].abs();
                let scale = if abs > S::one() { abs } else { S::one() };
                let h = eps * T::from(scale);
                let mut args_plus = args.to_vec();
                args_plus[idx] += h.clone();
                let mut args_minus = args.to_vec();
                args_minus[idx] -= h.clone();
                (func(&args_plus) - func(&args_minus)) / (h * two)
            },
            arity,
        ))
    }

    /// Retrieves a reference to the derivative function.
    ///
    /// Returns the pre-computed analytical derivative for the given argument index,
    /// or `None` if no derivative was registered.
    ///
    /// # Arguments
    ///
    /// * `idx` - Index of the argument
    pub fn derivative(&self, idx: usize) -> Option<&Self> {
        self.deriv.get(idx).map(Arc::as_ref)
    }

    /// Evaluates the function with the given arguments.
    ///
    /// # Arguments
    ///
    /// * `args` - Slice of argument values
    ///
    /// # Errors
    ///
    /// Returns error if the number of arguments does not match arity.
    pub fn apply(&self, args: &[T]) -> Result<T, Error>
    {
        if args.len() != self.arity {
            Self::err(
                ErrorKind::InvalidInput,
                format!("expected {} arguments, got {}", self.arity, args.len())
            )?;
        }

        Ok((self.func)(args))
    }

    /// Returns the number of arguments the function expects.
    pub fn arity(&self) -> usize {
        self.arity
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn test_create_function() {
        let func = CustomFunc::new(|args: &[Complex<f64>]| args[0].clone(), 1);
        assert_eq!(func.arity(), 1);
    }

    #[test]
    fn test_apply_success() {
        let func = CustomFunc::new(|args: &[Complex<f64>]| args[0].clone() + args[1].clone(), 2);
        let args = vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)];
        let result = func.apply(&args).unwrap();
        assert_eq!(result, Complex::new(3.0, 0.0));
    }

    #[test]
    fn test_apply_arity_mismatch() {
        let func = CustomFunc::new(|args: &[Complex<f64>]| args[0].clone(), 2);
        let args = vec![Complex::new(1.0, 0.0)];
        let result = func.apply(&args);
        assert!(result.is_err());
    }

    #[test]
    fn test_with_derivatives_success() {
        let func = CustomFunc::new(|args: &[Complex<f64>]| args[0].clone() * args[0].clone(), 1);
        let deriv = CustomFunc::new(|args: &[Complex<f64>]| args[0].clone() + args[0].clone(), 1);
        let result = func.with_derivatives(vec![deriv]);
        assert!(result.is_ok());
        let func_with_deriv = result.unwrap();
        assert!(func_with_deriv.derivative(0).is_some());
    }

    #[test]
    fn test_with_derivatives_arity_mismatch() {
        let func = CustomFunc::new(|args: &[Complex<f64>]| args[0].clone(), 1);
        let deriv1 = CustomFunc::new(|args: &[Complex<f64>]| args[0].clone(), 1);
        let deriv2 = CustomFunc::new(|args: &[Complex<f64>]| args[0].clone(), 1);
        let result = func.with_derivatives(vec![deriv1, deriv2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_derivative_access() {
        let func = CustomFunc::new(|args: &[Complex<f64>]| args[0].clone(), 2);
        let deriv0 = CustomFunc::new(|args: &[Complex<f64>]| args[0].clone(), 2);
        let deriv1 = CustomFunc::new(|args: &[Complex<f64>]| args[1].clone(), 2);
        let func_with_deriv = func
            .with_derivatives(vec![deriv0, deriv1])
            .unwrap();

        assert!(func_with_deriv.derivative(0).is_some());
        assert!(func_with_deriv.derivative(1).is_some());
        assert!(func_with_deriv.derivative(2).is_none());
    }

    #[test]
    fn test_numeric_deriv_simple() {
        // f(x) = x^2
        let func = CustomFunc::new(|args: &[Complex<f64>]| {
            let x = args[0];
            x * x
        }, 1);

        let deriv = func.numeric_deriv(0).unwrap();

        // At x = 2.0, df/dx should be approximately 4.0
        let args = vec![Complex::new(2.0, 0.0)];
        let result = deriv.apply(&args).unwrap();

        // Numerical derivative should be close to 4.0
        assert!((result.re - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_numeric_deriv_index_out_of_bounds() {
        let func = CustomFunc::new(|args: &[Complex<f64>]| args[0].clone(), 1);
        let result = func.numeric_deriv(1);
        assert!(result.is_err());
    }

    #[test]
    fn test_numeric_deriv_complex() {
        // f(z) = z + conj(z) (real part is 2*Re(z))
        let func = CustomFunc::new(|args: &[Complex<f64>]| {
            let result = args[0] + args[0].conj();
            println!("f({}) = {}", args[0], result);
            result
        }, 1);

        let deriv = func.numeric_deriv(0).unwrap();

        // At z = 1 + i, df/dz should be approximately 2
        let args = vec![Complex::new(1.0, 1.0)];
        println!("Computing derivative at z = {}", args[0]);
        let result = deriv.apply(&args).unwrap();
        println!("Derivative result: {}", result);

        assert!((result.re - 2.0).abs() < 1e-4, "expected 2.0, got {}", result.re);
    }

    #[test]
    fn test_apply_with_complex_result() {
        let func = CustomFunc::new(|args: &[Complex<f64>]| {
            args[0].sin()
        }, 1);

        let args = vec![Complex::new(0.0, 0.0)];
        let result = func.apply(&args).unwrap();
        assert_eq!(result, Complex::new(0.0, 0.0));
    }

    #[test]
    fn test_clone_preserves_function() {
        let func = CustomFunc::new(|args: &[Complex<f64>]| args[0] * args[0], 1);
        let func_cloned = func.clone();

        let args = vec![Complex::new(3.0, 0.0)];
        let result1 = func.apply(&args).unwrap();
        let result2 = func_cloned.apply(&args).unwrap();

        assert_eq!(result1, result2);
    }
}
