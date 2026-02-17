//! # core.rs
//!
//! Core trait definitions for complex number operations.
//!
//! This module defines the `ComplexBackend` trait which abstracts mathematical
//! operations on complex numbers, enabling different implementations to be used
//! interchangeably.

use num_traits::float::{
    Float,
    FloatConst,
    FloatCore,
};

/// Trait for complex number mathematical operations.
///
/// Provides a unified interface for performing mathematical operations on complex numbers.
/// Implementations can use different backends (e.g., `num_complex::Complex`, custom types).
pub(crate) trait ComplexBackend<S>:
    Clone + Send + Sync + 'static
where
    S: Clone + Send + Sync + 'static
{
    fn zero() -> Self;
    fn one() -> Self;
    fn two() -> Self;
    fn i() -> Self;

    fn sin(&self) -> Self;
    fn cos(&self) -> Self;
    fn tan(&self) -> Self;

    fn asin(&self) -> Self;
    fn acos(&self) -> Self;
    fn atan(&self) -> Self;

    fn sinh(&self) -> Self;
    fn cosh(&self) -> Self;
    fn tanh(&self) -> Self;

    fn asinh(&self) -> Self;
    fn acosh(&self) -> Self;
    fn atanh(&self) -> Self;

    fn exp(&self) -> Self;
    fn ln(&self) -> Self;
    fn log10(&self) -> Self;

    fn pow(&self, rhs: &Self) -> Self;
    fn powi(&self, n: i32) -> Self;

    fn sqrt(&self) -> Self;
    fn abs(&self) -> S;
    fn conj(&self) -> Self;

    fn re(&self) -> S;
    fn im(&self) -> S;

    fn from(v: S) -> Self;
    fn eps() -> Self;
}

use num_complex::{
    Complex, ComplexFloat,
};

/// Implementation of `ComplexBackend` for `num_complex::Complex`.
impl<T> ComplexBackend<T> for Complex<T>
where
    T: Float + FloatConst + FloatCore + Send + Sync + 'static,
{
    fn zero() -> Self { <Complex::<T> as num_traits::Zero>::zero() }
    fn one() -> Self { <Complex::<T> as num_traits::One>::one() }
    fn two() -> Self { <Complex::<T> as num_traits::One>::one() + <Complex::<T> as num_traits::One>::one() }
    fn i() -> Self { Complex::i() }

    fn sin(&self) -> Self { ComplexFloat::sin(*self) }
    fn cos(&self) -> Self { ComplexFloat::cos(*self) }
    fn tan(&self) -> Self { ComplexFloat::tan(*self) }

    fn asin(&self) -> Self { ComplexFloat::asin(*self) }
    fn acos(&self) -> Self { ComplexFloat::acos(*self) }
    fn atan(&self) -> Self { ComplexFloat::atan(*self) }

    fn sinh(&self) -> Self { ComplexFloat::sinh(*self) }
    fn cosh(&self) -> Self { ComplexFloat::cosh(*self) }
    fn tanh(&self) -> Self { ComplexFloat::tanh(*self) }

    fn asinh(&self) -> Self { ComplexFloat::asinh(*self) }
    fn acosh(&self) -> Self { ComplexFloat::acosh(*self) }
    fn atanh(&self) -> Self { ComplexFloat::atanh(*self) }

    fn exp(&self) -> Self { ComplexFloat::exp(*self) }
    fn ln(&self) -> Self { ComplexFloat::ln(*self) }
    fn log10(&self) -> Self { ComplexFloat::log10(*self) }

    fn pow(&self, rhs: &Self) -> Self { ComplexFloat::powc(*self, *rhs) }
    fn powi(&self, n: i32) -> Self { ComplexFloat::powi(*self, n) }

    fn sqrt(&self) -> Self { ComplexFloat::sqrt(*self) }
    fn abs(&self) -> T { ComplexFloat::abs(*self) }
    fn conj(&self) -> Self { ComplexFloat::conj(*self) }

    fn re(&self) -> T { self.re }
    fn im(&self) -> T { self.im }

    fn from(v: T) -> Self { <Complex<T> as From<T>>::from(v) }
    fn eps() -> Self { <Complex<T> as From<T>>::from(<T as Float>::epsilon()) }
}
