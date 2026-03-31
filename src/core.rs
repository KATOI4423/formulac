//! core.rs

use num_complex::Complex;
use num_traits::Num;
use std::{f64, i32};

pub trait Real: Num + std::ops::Neg<Output = Self>
    + Clone
    + PartialEq + PartialOrd
    + std::fmt::Debug
{
    // Basic
    fn from_f64(v: f64) -> Self;
    fn to_i32(&self) -> i32;
    fn fract(self) -> Self;
    fn trunc(self) -> Self;

    // Constants
    fn e() -> Self;
    fn frac_1_pi() -> Self;
    fn frac_1_sqrt_2() -> Self;
    fn frac_2_pi() -> Self;
    fn frac_2_sqrt_pi() -> Self;
    fn frac_pi_2() -> Self;
    fn frac_pi_3() -> Self;
    fn frac_pi_4() -> Self;
    fn frac_pi_6() -> Self;
    fn frac_pi_8() -> Self;
    fn ln_2() -> Self;
    fn ln_10() -> Self;
    fn log2_10() -> Self;
    fn log2_e() -> Self;
    fn log10_2() -> Self;
    fn log10_e() -> Self;
    fn pi() -> Self;
    fn sqrt_2() -> Self;
    fn tau() -> Self;

    // Trigonometric functions
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
    fn asin(self) -> Self;
    fn acos(self) -> Self;
    fn atan(self) -> Self;
    fn atan2(self, other: Self) -> Self;

    // Hyperbolic functions
    fn sinh(self) -> Self;
    fn cosh(self) -> Self;
    fn tanh(self) -> Self;
    fn asinh(self) -> Self;
    fn acosh(self) -> Self;
    fn atanh(self) -> Self;

    // Exponential and Logarithmic
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn log10(self) -> Self;

    // Others
    fn sqrt(self) -> Self;
    fn abs(self) -> Self;
    fn hypot(self, other: Self) -> Self;

    // Power
    fn pow(self, rhs: Self) -> Self;
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

pub trait ComplexMath {
    // Trigonometric functions
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
    fn asin(self) -> Self;
    fn acos(self) -> Self;
    fn atan(self) -> Self;

    // Hyperbolic functions
    fn sinh(self) -> Self;
    fn cosh(self) -> Self;
    fn tanh(self) -> Self;
    fn asinh(self) -> Self;
    fn acosh(self) -> Self;
    fn atanh(self) -> Self;

    // Exponential and Logarithmic
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn log10(self) -> Self;

    // Others
    fn sqrt(self) -> Self;
    fn abs(self) -> Self;
    fn conj(self) -> Self;

    // Power
    fn powc(self, rhs: Self) -> Self;
    fn powi(self, n: i32) -> Self;
}

impl<T: Real> ComplexMath for Complex<T> {
    fn sin(self) -> Self
    {
        // sin(a + bi) = sin(a) cosh(b) + i cos(a) sinh(b)
        let re = self.re; let im = self.im;
        Self {
            re: re.clone().sin() * im.clone().cosh(),
            im: re.cos() * im.sinh(),
        }
    }

    fn cos(self) -> Self
    {
        // cos(a + bi) = cos(a) cosh(b) - i sin(a) sinh(b)
        let re = self.re; let im = self.im;
        Self {
            re: re.clone().cos() * im.clone().cosh(),
            im: -(re.sin() * im.sinh()),
        }
    }

    fn tan(self) -> Self { self.clone().sin() / self.cos() }

    fn asin(self) -> Self {
        // asin(z) = -i ln(iz + sqrt(1 - z^2))
        let i = Complex::new(T::zero(), T::one());
        let one = Complex::new(T::one(), T::zero());

        let iz = i.clone() * self.clone();
        let sqrt = (one - self.clone() * self).sqrt();

        -(i) * (iz + sqrt).ln()
    }

    fn acos(self) -> Self {
        // acos(z) = -i ln(z + i sqrt(1 - z^2))
        let i = Complex::new(T::zero(), T::one());
        let one = Complex::new(T::one(), T::zero());

        let sqrt = (one - self.clone() * self.clone()).sqrt();

        -(i.clone()) * (self + i * sqrt).ln()
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
        // sinh(z) = (exp(z) - exp(-z)) / 2
        (self.clone().exp() - (-self).exp()) * T::from_f64(0.5)
    }

    fn cosh(self) -> Self {
        // cosh(z) = (exp(z) + exp(-z)) / 2
        (self.clone().exp() + (-self).exp()) * T::from_f64(0.5)
    }

    fn tanh(self) -> Self { self.clone().sinh() / self.cosh() }

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
        // log10(z) = ln(z) / log10(e)
        self.ln() / T::log10_e()
    }

    fn sqrt(self) -> Self {
        // sqrt(z) = sqrt((|z| + re)/2) + i * sign(im) * sqrt((|z| - re)/2)
        let r = self.re.clone().hypot(self.im.clone());
        let half = T::from_f64(0.5);

        let re = ((r.clone() + self.re.clone()) * half.clone()).sqrt();
        let im = ((r - self.re) * half).sqrt();

        let im = if self.im >= T::zero() { im } else { -im };

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

        while exp > 0 {
            if exp % 2 == 1 {
                result = result * base.clone();
            }
            base = base.clone() * base;
            exp /= 2;
        }

        result
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
