//! core.rs

use num_complex::Complex;
use num_traits::{
    Num,
    Zero,
};
use std::f64;

/// Defines the real-number operations required by the formula engine.
///
/// The trait abstracts over the scalar type used for the real and imaginary
/// components of [`num_complex::Complex`] values. The crate provides an
/// implementation for `f64`; additional implementations can be supplied by
/// downstream users when their scalar type satisfies the required operations.
///
/// In addition to arithmetic, the trait exposes mathematical constants and
/// elementary functions used by the expression evaluator.
pub trait Real: Num + std::ops::Neg<Output = Self>
    + Clone
    + PartialEq + PartialOrd
    + std::fmt::Debug
{
    // Basic
    /// Creates a value from an [`f64`].
    fn from_f64(v: f64) -> Self;

    /// Converts the value to an [`i32`], using the implementation-defined conversion semantics.
    fn to_i32(&self) -> i32;

    /// Returns whether the value can be represented exactly as an [`i32`].
    fn is_i32_compatible(&self) -> bool {
        self.clone().fract().is_zero()
            && *self >= Self::from_f64(i32::MIN as f64)
            && *self <= Self::from_f64(i32::MAX as f64)
    }
    /// Returns the fractional part of the value.
    fn fract(self) -> Self;

    /// Returns the value with its fractional part removed.
    fn trunc(self) -> Self;

    // Constants
    /// Returns Euler's number, *e*.
    fn e() -> Self;

    /// Returns 1 / π.
    fn frac_1_pi() -> Self;

    /// Returns 1 / √2.
    fn frac_1_sqrt_2() -> Self;

    /// Returns 2 / π.
    fn frac_2_pi() -> Self;

    /// Returns 2 / √π.
    fn frac_2_sqrt_pi() -> Self;

    /// Returns π / 2.
    fn frac_pi_2() -> Self;

    /// Returns π / 3.
    fn frac_pi_3() -> Self;

    /// Returns π / 4.
    fn frac_pi_4() -> Self;

    /// Returns π / 6.
    fn frac_pi_6() -> Self;

    /// Returns π / 8.
    fn frac_pi_8() -> Self;

    /// Returns ln(2).
    fn ln_2() -> Self;

    /// Returns ln(10).
    fn ln_10() -> Self;

    /// Returns log₂(10).
    fn log2_10() -> Self;

    /// Returns log₂(e).
    fn log2_e() -> Self;

    /// Returns log₁₀(2).
    fn log10_2() -> Self;

    /// Returns log₁₀(e).
    fn log10_e() -> Self;

    /// Returns π.
    fn pi() -> Self;

    /// Returns √2.
    fn sqrt_2() -> Self;

    /// Returns τ, equal to 2π.
    fn tau() -> Self;

    // Trigonometric functions
    /// Returns the sine of the value.
    fn sin(self) -> Self;

    /// Returns the cosine of the value.
    fn cos(self) -> Self;

    /// Returns the tangent of the value.
    fn tan(self) -> Self;

    /// Returns the inverse sine of the value.
    fn asin(self) -> Self;

    /// Returns the inverse cosine of the value.
    fn acos(self) -> Self;

    /// Returns the inverse tangent of the value.
    fn atan(self) -> Self;

    /// Returns the four-quadrant inverse tangent of `self / other`.
    fn atan2(self, other: Self) -> Self;

    /// Returns the sine and cosine of the value as `(sin, cos)`.
    fn sin_cos(self) -> (Self, Self);

    // Hyperbolic functions
    /// Returns the hyperbolic sine of the value.
    fn sinh(self) -> Self;

    /// Returns the hyperbolic cosine of the value.
    fn cosh(self) -> Self;

    /// Returns the hyperbolic tangent of the value.
    fn tanh(self) -> Self;

    /// Returns the inverse hyperbolic sine of the value.
    fn asinh(self) -> Self;

    /// Returns the inverse hyperbolic cosine of the value.
    fn acosh(self) -> Self;

    /// Returns the inverse hyperbolic tangent of the value.
    fn atanh(self) -> Self;

    // Exponential and Logarithmic
    /// Returns the exponential function of the value.
    fn exp(self) -> Self;

    /// Returns the natural logarithm of the value.
    fn ln(self) -> Self;

    /// Returns the base-10 logarithm of the value.
    fn log10(self) -> Self;

    // Others
    /// Returns the square root of the value.
    fn sqrt(self) -> Self;

    /// Returns the absolute value of the value.
    fn abs(self) -> Self;

    /// Returns the Euclidean norm of the pair `(self, other)`.
    fn hypot(self, other: Self) -> Self;

    // Power
    /// Raises the value to a real-valued power.
    fn pow(self, rhs: Self) -> Self;

    /// Raises the value to an integer power.
    fn powi(self, n: i32) -> Self;
}

impl Real for f64 {
    fn from_f64(v: f64) -> Self { v }
    fn to_i32(&self) -> i32
    {
        if !self.is_finite() {
            return 0;
        }

        let truncated = self.trunc();
        if truncated > i32::MAX as Self {
            i32::MAX
        } else if truncated < i32::MIN as Self {
            i32::MIN
        } else {
            truncated as i32
        }
    }
    fn is_i32_compatible(&self) -> bool {
        const MAX: f64 = i32::MAX as f64;
        const MIN: f64 = i32::MIN as f64;
        self.fract().is_zero() && MIN <= *self && *self <= MAX
    }
    fn fract(self) -> Self { self.fract() }
    fn trunc(self) -> Self { self.trunc() }

    fn e() -> Self { f64::consts::E }
    fn frac_1_pi() -> Self { f64::consts::FRAC_1_PI }
    fn frac_1_sqrt_2() -> Self { f64::consts::FRAC_1_SQRT_2 }
    fn frac_2_pi() -> Self { f64::consts::FRAC_2_PI }
    fn frac_2_sqrt_pi() -> Self { f64::consts::FRAC_2_SQRT_PI }
    fn frac_pi_2() -> Self { f64::consts::FRAC_PI_2 }
    fn frac_pi_3() -> Self { f64::consts::FRAC_PI_3 }
    fn frac_pi_4() -> Self { f64::consts::FRAC_PI_4 }
    fn frac_pi_6() -> Self { f64::consts::FRAC_PI_6 }
    fn frac_pi_8() -> Self { f64::consts::FRAC_PI_8 }
    fn ln_2() -> Self { f64::consts::LN_2 }
    fn ln_10() -> Self { f64::consts::LN_10 }
    fn log2_10() -> Self { f64::consts::LOG2_10 }
    fn log2_e() -> Self { f64::consts::LOG2_E }
    fn log10_2() -> Self { f64::consts::LOG10_2 }
    fn log10_e() -> Self { f64::consts::LOG10_E }
    fn pi() -> Self { f64::consts::PI }
    fn sqrt_2() -> Self { f64::consts::SQRT_2 }
    fn tau() -> Self { f64::consts::TAU }

    fn sin(self) -> Self { self.sin() }
    fn cos(self) -> Self { self.cos() }
    fn tan(self) -> Self { self.tan() }
    fn asin(self) -> Self { self.asin() }
    fn acos(self) -> Self { self.acos() }
    fn atan(self) -> Self { self.atan() }
    fn atan2(self, other: Self) -> Self { self.atan2(other) }
    fn sin_cos(self) -> (Self, Self) { self.sin_cos() }

    fn sinh(self) -> Self { self.sinh() }
    fn cosh(self) -> Self { self.cosh() }
    fn tanh(self) -> Self { self.tanh() }
    fn asinh(self) -> Self { self.asinh() }
    fn acosh(self) -> Self { self.acosh() }
    fn atanh(self) -> Self { self.atanh() }

    fn exp(self) -> Self { self.exp() }
    fn ln(self) -> Self { self.ln() }
    fn log10(self) -> Self { self.log10() }

    fn sqrt(self) -> Self { self.sqrt() }
    fn abs(self) -> Self { self.abs() }
    fn hypot(self, other: Self) -> Self { self.hypot(other) }

    fn pow(self, rhs: Self) -> Self { self.powf(rhs) }
    fn powi(self, n: i32) -> Self { self.powi(n) }
}

/// Provides elementary complex-valued mathematical operations.
///
/// This trait supplies the operations needed to evaluate built-in functions
/// on [`num_complex::Complex`] values. The crate implements it for
/// `Complex<T>` where `T` implements [`Real`].
pub trait ComplexMath {
    // Trigonometric functions
    /// Returns the complex sine.
    fn sin(self) -> Self;
    /// Returns the complex cosine.
    fn cos(self) -> Self;
    /// Returns the complex tangent.
    fn tan(self) -> Self;
    /// Returns the complex inverse sine.
    fn asin(self) -> Self;
    /// Returns the complex inverse cosine.
    fn acos(self) -> Self;
    /// Returns the complex inverse tangent.
    fn atan(self) -> Self;

    // Hyperbolic functions
    /// Returns the complex hyperbolic sine.
    fn sinh(self) -> Self;

    /// Returns the hyperbolic cosine of the value.
    fn cosh(self) -> Self;

    /// Returns the hyperbolic tangent of the value.
    fn tanh(self) -> Self;

    /// Returns the inverse hyperbolic sine of the value.
    fn asinh(self) -> Self;

    /// Returns the inverse hyperbolic cosine of the value.
    fn acosh(self) -> Self;

    /// Returns the inverse hyperbolic tangent of the value.
    fn atanh(self) -> Self;

    // Exponential and Logarithmic
    /// Returns the exponential function of the value.
    fn exp(self) -> Self;

    /// Returns the natural logarithm of the value.
    fn ln(self) -> Self;

    /// Returns the base-10 logarithm of the value.
    fn log10(self) -> Self;

    // Others
    /// Returns the square root of the value.
    fn sqrt(self) -> Self;
    /// Returns the magnitude as a real complex value.
    fn abs(self) -> Self;
    /// Returns the complex conjugate.
    fn conj(self) -> Self;

    // Power
    /// Raises the complex value to a complex power.
    fn powc(self, rhs: Self) -> Self;
    /// Raises the complex value to an integer power.
    fn powi(self, n: i32) -> Self;
}

impl<T: Real> ComplexMath for Complex<T> {
    fn sin(self) -> Self
    {
        // sin(a + bi) = sin(a) cosh(b) + i cos(a) sinh(b)
        let (a, b) = (self.re, self.im);
        let (sin_a, cos_a) = a.sin_cos();
        Self {
            re: sin_a * b.clone().cosh(),
            im: cos_a * b.sinh(),
        }
    }

    fn cos(self) -> Self
    {
        // cos(a + bi) = cos(a) cosh(b) - i sin(a) sinh(b)
        let (a, b) = (self.re, self.im);
        let (sin_a, cos_a) = a.sin_cos();
        Self {
            re: cos_a * b.clone().cosh(),
            im: -(sin_a * b.sinh()),
        }
    }

    fn tan(self) -> Self {
        // tan(a + bi) = (sin(2a) + i sinh(2b)) / (cos(2b) + cosh(2b))
        let (a2, b2) = (self.re * T::from_f64(2.0), self.im * T::from_f64(2.0));
        let (sin_2a, cos_2a) = a2.sin_cos();
        let (sinh_2b, cosh_2b) = (b2.clone().sinh(), b2.cosh());

        Complex::new(sin_2a, sinh_2b) / (cos_2a + cosh_2b)
    }

    fn asin(self) -> Self {
        // asin(z) = -i ln(iz + sqrt(1 - z^2))
        let z = self;
        let z2 = z.clone() * z.clone(); // z^2
        let iz = Complex::new(-z.im, z.re);
        let i = Complex::new(T::zero(), T::one());
        let one = Complex::new(T::one(), T::zero());

        -i * (iz + (one - z2).sqrt()).ln()
    }

    fn acos(self) -> Self {
        // acos(z) = -i ln(z + i sqrt(1 - z^2))
        let z = self;
        let z2 = z.clone() * z.clone(); // z^2
        let i = Complex::new(T::zero(), T::one());
        let one = Complex::new(T::one(), T::zero());

        -(i.clone()) * (z + i * (one - z2).sqrt()).ln()
    }

    fn atan(self) -> Self {
        // atan(z) = (i/2) ln((1 - iz)/(1 + iz))
        let i = Complex::new(T::zero(), T::one());
        let one = Complex::new(T::one(), T::zero());

        let iz = i * self;

        let num = one.clone() - iz.clone();
        let den = one + iz;

        let half_i = Complex::new(T::zero(), T::one() * T::from_f64(0.5));

        half_i * (num / den).ln()
    }

    fn sinh(self) -> Self {
        // sinh(z) = (exp(z) - exp(-z)) / 2 = sinh(x) cos(y) + i cosh(x) sin(y)
        let (x, y) = (self.re, self.im);
        let (sin_y, cos_y) = y.sin_cos();
        Self {
            re: x.clone().sinh() * cos_y,
            im: x.cosh() * sin_y,
        }
    }

    fn cosh(self) -> Self {
        // cosh(z) = (exp(z) + exp(-z)) / 2 = cosh(x) cos(y) + i sinh(x) sin(y)
        let (x, y) = (self.re, self.im);
        let (sin_y, cos_y) = y.sin_cos();
        Self {
            re: x.clone().cosh() * cos_y,
            im: x.sinh() * sin_y,
        }
    }

    fn tanh(self) -> Self {
        // tanh(z) = (exp(2z) - 1) / (exp(2z) + 1)
        let e2 = (self.clone() + self).exp();
        let one = T::one();
        (e2.clone() - one.clone()) / (e2 + one)
      }

    fn asinh(self) -> Self {
        // asinh(z) = ln(z + sqrt(z^2 + 1))
        let one = Complex::new(T::one(), T::zero());
        (self.clone() + (self.clone() * self + one).sqrt()).ln()
    }

    fn acosh(self) -> Self {
        // acosh(z) = ln(z + sqrt(z-1) * sqrt(z+1))
        let one = Complex::new(T::one(), T::zero());
        (self.clone() + (self.clone() - one.clone()).sqrt() * (self + one).sqrt()).ln()
    }

    fn atanh(self) -> Self {
        // atanh(z) = (1/2) ln((1+z)/(1-z))
        let one = Complex::new(T::one(), T::zero());

        ((one.clone() + self.clone()) / (one - self)).ln() * T::from_f64(0.5)
    }

    fn exp(self) -> Self
    {
        // exp(a + bi) = exp(a) * (cos(b) + i sin(b))
        let re = self.re;
        let im = self.im;

        let exp = re.exp();

        Self {
            re: exp.clone() * im.clone().cos(),
            im: exp * im.sin(),
        }
    }

    fn ln(self) -> Self
    {
        // ln(z) = ln|z| + i arg(z)
        let r = self.re.clone().hypot(self.im.clone());
        let theta = self.im.atan2(self.re);

        Self {
            re: r.ln(),
            im: theta,
        }
    }

    fn log10(self) -> Self
    {
        // log10(z) = ln(z) * log10(e)
        self.ln() * T::log10_e()
    }

    fn sqrt(self) -> Self {
        // sqrt(z) = sqrt((|z| + re)/2) + i * sign(im) * sqrt((|z| - re)/2)
        let r = self.re.clone().hypot(self.im.clone());
        let half = T::from_f64(0.5);

        let (re, im) = if self.re > T::zero() {
            let u = ((r + self.re) * half.clone()).sqrt();
            (u.clone(), self.im * half / u)
        } else {
            let v = ((r - self.re) * half.clone()).sqrt();
            (self.im * half / v.clone(), v)
        };

        Complex::new(re, im)
    }

    fn abs(self) -> Self { Complex::new(self.re.hypot(self.im), T::zero()) }

    fn conj(self) -> Self { Complex::new(self.re, -self.im) }

    fn powc(self, rhs: Self) -> Self
    {
        // z ^ w = exp(ln(z) * w)
        (rhs * self.ln()).exp()
    }

    fn powi(self, n: i32) -> Self {
        if n == 0 {
            return Complex::new(T::one(), T::zero());
        }

        if n < 0 {
            return Complex::new(T::one(), T::zero()) / self.powi(-n);
        }

        let mut result = Complex::new(T::one(), T::zero());
        let mut base = self;
        let mut exp = n;

        while exp > 1 {
            if exp & 1 == 1 {
                result = result * base.clone();
            }
            base = base.clone() * base;
            exp >>=1;
        }

        result * base
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    const EPS: f64 = 1e-10;

    fn approx_eq(a: Complex<f64>, b: Complex<f64>) {
        assert!(
            (a.re - b.re).abs() < EPS,
            "re mismatch: {} vs {}",
            a.re,
            b.re
        );
        assert!(
            (a.im - b.im).abs() < EPS,
            "im mismatch: {} vs {}",
            a.im,
            b.im
        );
    }

    #[test]
    fn test_sin_cos_identity() {
        let z = Complex::new(1.2, -0.7);

        let sin = z.clone().sin();
        let cos = z.clone().cos();

        let lhs = sin.clone() * sin + cos.clone() * cos;
        let rhs = Complex::new(1.0, 0.0);

        approx_eq(lhs, rhs);
    }

    #[test]
    fn test_exp_ln_identity() {
        let z = Complex::new(0.5, -1.3);

        let result = z.clone().ln().exp();

        approx_eq(result, z);
    }

    #[test]
    fn test_exp_i_pi() {
        let pi = std::f64::consts::PI;
        let z = Complex::new(0.0, pi);

        let result = z.exp();

        approx_eq(result, Complex::new(-1.0, 0.0));
    }

    #[test]
    fn test_sin_i() {
        let z = Complex::new(0.0, 1.0);

        let result = z.sin();

        // sin(i) = i sinh(1)
        let expected = Complex::new(0.0, 1.0_f64.sinh());

        approx_eq(result, expected);
    }

    #[test]
    fn test_real_consistency() {
        let x = 0.7;
        let z = Complex::new(x, 0.0);

        approx_eq(Complex::new(x.sin(), 0.0), z.clone().sin());
        approx_eq(Complex::new(x.cos(), 0.0), z.clone().cos());
        approx_eq(Complex::new(x.exp(), 0.0), z.clone().exp());
        approx_eq(Complex::new(x.ln(), 0.0), z.clone().ln());
    }

    #[test]
    fn test_sqrt() {
        let z = Complex::new(3.0, 4.0);

        let sqrt = z.clone().sqrt();
        let back = sqrt.clone() * sqrt;

        approx_eq(back, z);
    }

    #[test]
    fn test_powc() {
        let z = Complex::new(1.2, 0.7);
        let w = Complex::new(-0.3, 0.5);

        let result = z.clone().powc(w.clone());

        // 検証：exp(w ln z)
        let expected = (w * z.ln()).exp();

        approx_eq(result, expected);
    }

    #[test]
    fn test_powi() {
        let z = Complex::new(1.1, -0.4);

        let result = z.clone().powi(5);

        let expected = z.clone() * z.clone() * z.clone() * z.clone() * z;

        approx_eq(result, expected);
    }

    #[test]
    fn test_tan_identity() {
        let z = Complex::new(0.8, -0.3);

        let tan = z.clone().tan();
        let expected = z.clone().sin() / z.cos();

        approx_eq(tan, expected);
    }

    #[test]
    fn test_log10() {
        let z = Complex::new(1.3, 0.4);

        let result = z.clone().log10();

        let expected = z.ln() / std::f64::consts::LN_10;

        approx_eq(result, expected);
    }
}
