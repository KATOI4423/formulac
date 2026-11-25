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

    /// Returns whether the sign bit of the value is set.
    fn is_sign_negative(&self) -> bool { *self < Self::zero() }

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

    /// Returns `exp(x - 1)`. Override it for accuracy near one.
    fn exp_m1(self) -> Self {
        self.exp() - Self::one()
    }

    /// Returns the natural logarithm of the value.
    fn ln(self) -> Self;

    /// Returns `ln(1 + x)` for too small x. Override it for accuracy near zero.
    fn ln_1p(self) -> Self { (Self::one() + self).ln() }

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
    fn is_sign_negative(&self) -> bool { f64::is_sign_negative(*self) }

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
    fn exp_m1(self) -> Self { self.exp_m1() }
    fn ln(self) -> Self { self.ln() }
    fn ln_1p(self) -> Self { self.ln_1p() }
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
    /// Returns the sine and cosine of the value as `(sin, cos)`.
    fn sin_cos(self) -> (Self, Self) where Self: Sized + Clone {
        (self.clone().sin(), self.cos())
    }
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

/// Raises `base` to a positive integer power by squaring, without redundant clones.
fn powu<T: Real>(mut base: Complex<T>, mut exp: u32) -> Complex<T> {
    debug_assert!(exp > 0);
    // Skip the low zero bits so that `result` never starts from 1.
    while exp & 1 == 0 {
        base = base.clone() * base;
        exp >>= 1;
    }
    if exp == 1 {
        return base;
    }
    let mut result = base.clone();
    exp >>= 1;
    loop {
        base = base.clone() * base;
        if exp & 1 == 1 {
            if exp == 1 {
                return result * base; // Last multiplication: no clone needed.
            }
            result = result * base.clone();
        }
        exp >>= 1;
    }
}

/// Computes the inverse hyperbolic sine for the first quadrant.
///
/// For moderate inputs, use a cancellation-resistant formula based on
/// the distances from z to the branch points ±i.
///
/// For large inputs, scale the input by s = max(|x|, |y|) to avoid
/// overflow in intermediate calculations. The real part is computed
/// without explicitly constructing A = (|z - i| + |z + i|) / 2,
/// which may overflow even when the final result is representable.
fn asinh_inner<T: Real>(z: Complex<T>) -> Complex<T> {
    let (x, y) = (z.re, z.im);
    let one = T::one();
    let two = T::from_f64(2.0);

    let scale = if x > y { x.clone() } else { y.clone() };

    if scale > two {
        // Normalize the input to avoid overflow.
        //   X = x / s, Y = y / s, t = 1 / s
        //   where s = max(x, y).
        let xs = x / scale.clone();
        let ys = y / scale.clone();
        let t = one.clone() / scale.clone();

        // Im = arg(z + sqrt(1 + z^2))
        //
        // Dividing the argument of sqrt by s gives:
        //   w = sqrt(t^2 + (X + iY)^2)
        //   Im = atan2(Y + Im(w), X + Re(w))
        let im = {
            let re = (xs.clone() - ys.clone())
                * (xs.clone() + ys.clone())
                + t.clone() * t.clone();

            let im = two.clone() * xs.clone() * ys.clone();
            let w = Complex::new(re, im).sqrt();
            (ys.clone() + w.im).atan2(xs.clone() + w.re)
        };

        // Re = acosh(A)
        //
        // Instead of constructing A = s * A_scaled directly,
        // compute:
        //   acosh(A) = ln(A) + ln(1 + sqrt(1 - 1/A^2))
        //
        // This avoids overflow when A exceeds the representable range.
        let re = {
            let p = xs.clone().hypot(t.clone() - ys.clone());
            let q = xs.clone().hypot(t + ys);

            let a_scaled = (p + q) * T::from_f64(0.5);

            // 1 / A = 1 / (s * A_scaled)
            let inv_a = (one.clone() / scale.clone()) / a_scaled.clone();

            scale.ln()
                + a_scaled.ln()
                + (one - inv_a.clone() * inv_a).sqrt().ln_1p()
        };

        return Complex::new(re, im);
    }

    // Moderate inputs:
    // Re = acosh(A), computed through m = A - 1 to avoid cancellation
    // when A is close to 1.
    //
    // Im = arg(z + sqrt(1 + z^2))
    let im = {
        let s = Complex::new(
            (one.clone() - y.clone()) * (one.clone() + y.clone())
                + x.clone() * x.clone(),
            two.clone() * x.clone() * y.clone(),
        )
        .sqrt();

        (y.clone() + s.im).atan2(x.clone() + s.re)
    };

    let re = {
        let d = one.clone() - y.clone();
        let e = one.clone() + y;

        let p = x.clone().hypot(d.clone());
        let q = x.clone().hypot(e.clone());

        if d >= T::zero() {
            // m = x^2 * k. Keep x outside of sqrt(m * (m + 2)) so that a tiny x
            // does not underflow through x^2.
            let den = p + d;
            let k1 = if den.is_zero() { T::zero() } else { one.clone() / den };
            let k = (k1 + one.clone() / (q + e)) * T::from_f64(0.5);
            let m = x.clone() * x.clone() * k.clone();
            let sq = x * (k * (m.clone() + two)).sqrt();
            (m + sq).ln_1p()
        } else {
            let m = (p - d + x.clone() * x.clone() / (q + e)) * T::from_f64(0.5);
            (m.clone() + (m.clone() * (m + two)).sqrt()).ln_1p()
        }
    };

    Complex::new(re, im)
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
        // tan(a + bi) = -i tanh(-b + ai)
        let (a, b) = (self.re, self.im);
        let tanh = Self::new(-b, a).tanh();
        Self::new(tanh.im, -tanh.re)
    }

    fn asin(self) -> Self {
        // Handle the real axis explicitly to preserve signed zero.
        if self.im.is_zero() {
            let w = ComplexMath::asinh(Self::new(
                T::zero(),
                self.re.clone().abs(),
            ));

            let re = if self.re.is_sign_negative() { -w.im } else { w.im };
            let im = if self.im.is_sign_negative() { -w.re } else { w.re };
            return Self::new(re, im);
        }

        // asin(z) = -i asinh(i*z)
        let w = ComplexMath::asinh(Self::new(-self.im, self.re));
        Self::new(w.im, -w.re)
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
        // tanh(x + iy) = { sinh(2x) + i sin(2y) } / { cosh(2x) + cos(2y) }
        //              ↓ t = exp(-2|x|)
        //              = { sign(x) (1 - t^2) + i 2t sin(2y) } / { 1 + 2t cos(2y) + t^2 }
        //              ↓ d = 1 - t = -exp_m1(-2|x|)
        //              = { sign(x) d * (1 + t) + i 4t sin(y) cos(y) } / { d^2 + 4t cos^2(y) }
        let (x, y) = (self.re, self.im);
        let (sin_y, cos_y) = y.sin_cos();

        let mx2 = -T::from_f64(2.0) * x.clone().abs();
        let t = mx2.clone().exp();
        let d = -(mx2.exp_m1());
        let t4 = T::from_f64(4.0) * t.clone();
        let denom = d.clone() * d.clone() + t4.clone() * cos_y.clone() * cos_y.clone();
        let re = {
            let re = d * (T::one() + t);
            if x < T::zero() { -re } else { re }
        };
        Complex::new(
            re / denom.clone(),
            t4 * sin_y * cos_y / denom,
        )
      }

    fn asinh(self) -> Self {
        // asinh(-z) = -asinh(z)
        if self.re.is_sign_negative() {
            return -ComplexMath::asinh(-self);
        }
        // asinh(conj(z)) = conj(asinh(z))
        if self.im.is_sign_negative() {
            return ComplexMath::asinh(self.conj()).conj()
        }

        // Here Re >= 0 && Im >= 0
        asinh_inner(self)
    }

    fn acosh(self) -> Self {
        // acosh(z) = ln(z + sqrt(z-1) * sqrt(z+1))
        let one = Complex::new(T::one(), T::zero());
        (self.clone() + (self.clone() - one.clone()).sqrt() * (self + one).sqrt()).ln()
    }

    fn atanh(self) -> Self {
        // use `atanh(-z) = -atanh(z)` to avoid `ln_p1(x)` for x ~ -1.
        if self.re.is_sign_negative() {
            return -ComplexMath::atanh(-self);
        }

        // Re(atanh(z)) = 1/4 * ln( 1 + 4x / ( (1-x)^2 + y^2 ) )
        // Im(atanh(z)) = 1/2 * atan2(2y, 1 - x^2 - y^2)
        let (x, y) = (self.re, self.im);

        let re = {
            let x1 = T::one() - x.clone();

            // Scale the denominator to avoid overflow.
            let scale = {
                let abs_x1 = x1.clone().abs();
                let abs_y = y.clone().abs();

                let s = if abs_x1 > abs_y { abs_x1 } else { abs_y };
                if s > T::one() { s } else { T::one() }
            };

            let xs = x1 / scale.clone();
            let ys = y.clone() / scale.clone();
            let denom = xs.clone() * xs + ys.clone() * ys;

            let d = T::from_f64(4.0) * (x.clone() / scale.clone()) / (scale * denom);

            d.ln_1p() * T::from_f64(0.25)
        };

        let im = {
            // atan2 does not change when both arguments are divided by the same
            // positive value. Divide them by s^2 so that neither 2y nor x^2 + y^2
            // overflows for huge inputs. For |z| <= 1, s = 1 and nothing is rounded.
            let scale = {
                let abs_y = y.clone().abs();
                let s = if x > abs_y { x.clone() } else { abs_y };
                if s > T::one() { s } else { T::one() }
            };
            let ys = y / scale.clone();

            let a = T::from_f64(2.0) * ys.clone() / scale.clone();
            // (1 - x)(1 + x) avoids cancellation in 1 - x^2 near |x| = 1.
            let b = ((T::one() - x.clone()) / scale.clone()) * ((T::one() + x) / scale)
                - ys.clone() * ys;

            a.atan2(b) * T::from_f64(0.5)
        };

        Complex::new(re, im)
    }

    fn sin_cos(self) -> (Self, Self) {
        // Compute all shared real and hyperbolic terms only once.
        // sin(a + bi) = sin(a) cosh(b) + i cos(a) sinh(b)
        // cos(a + bi) = cos(a) cosh(b) - i sin(a) sinh(b)
        let (a, b) = (self.re, self.im);
        let (sin_a, cos_a) = a.sin_cos();
        let (sinh_b, cosh_b) = (b.clone().sinh(), b.cosh());

        let sin = Self {
            re: sin_a.clone() * cosh_b.clone(),
            im: cos_a.clone() * sinh_b.clone(),
        };
        let cos = Self {
            re: cos_a * cosh_b,
            im: -sin_a * sinh_b,
        };

        (sin, cos)
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
        if self.im.is_zero() {
            if self.re > T::zero() {
                return Complex::new(self.re.sqrt(), T::zero());
            } else if self.re < T::zero() {
                return Complex::new(T::zero(), self.re.abs().sqrt());
            } else if self.is_zero() {
                return Complex::zero();
            }
            // Exceptional values (NaN) are handled below.
        }

        //  ________   _____________               _____________
        // √ a + bi = √ (r + a) / 2   + i sign(b) √ (r - a) / 2
        let r = self.re.clone().hypot(self.im.clone());
        let half = T::from_f64(0.5);

        let (re, im) = if self.re > T::zero() {
            let u = ((r * half.clone()) + self.re * half.clone()).sqrt();
            (u.clone(), self.im * half / u)
        } else {
            let v = ((r * half.clone()) - (self.re * half.clone())).sqrt();
            let v = if self.im >= T::zero() { v } else { -v };
            (self.im * half / v.clone(), v)
        };

        Complex::new(re, im)
    }

    fn abs(self) -> Self { Complex::new(self.re.hypot(self.im), T::zero()) }

    fn conj(self) -> Self { Complex::new(self.re, -self.im) }

    fn powc(self, rhs: Self) -> Self
    {
        // z ^ w = exp(ln(z) * w)
        if self.is_zero() && rhs.re > T::zero() {
            return Self::zero();
        }
        (rhs * self.ln()).exp()
    }

    fn powi(self, n: i32) -> Self {
        match n {
            0 => Complex::new(T::one(), T::zero()),
            1 => self,
            -1 => self.inv(),
            _ if n > 0 => powu(self, n as u32),
            _ => powu(self.inv(), n.unsigned_abs()), // Also correct for i32::MIN.
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::core::ComplexMath;
    use num_complex::Complex;
    use num_traits::{Signed, Zero};

    const EPS: f64 = 1e-10;

    fn c(re: f64, im: f64) -> Complex<f64> {
        Complex::new(re, im)
    }

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

    fn assert_rel(a: Complex<f64>, b: Complex<f64>) {
        for (a, b, fmt) in [(a.re, b.re, "re"), (a.im, b.im, "im")] {
            let err = (a - b).abs();
            if !a.is_zero() {
                assert!(err <= EPS * a.abs().max(f64::MIN_POSITIVE), "{fmt}: {a:e} vs {b:e}")
            } else if ! b.is_zero() {
                assert!(err <= EPS * b.abs().max(f64::MIN_POSITIVE), "{fmt}: {a:e} vs {b:e}")
            } else {
                assert!(true); // a == b
            };
        }
    }

    #[test]
    fn test_sin_cos_identity() {
        let z = Complex::new(1.2, -0.7);

        let sin = ComplexMath::sin(z.clone());
        let cos = ComplexMath::cos(z.clone());

        let lhs = sin.clone() * sin + cos.clone() * cos;
        let rhs = Complex::new(1.0, 0.0);

        approx_eq(lhs, rhs);
    }

    #[test]
    fn test_exp_ln_identity() {
        let z = Complex::new(0.5, -1.3);

        let result = ComplexMath::exp(ComplexMath::ln(z.clone()));

        approx_eq(result, z);
    }

    #[test]
    fn test_exp_i_pi() {
        let pi = std::f64::consts::PI;
        let z = Complex::new(0.0, pi);

        let result = ComplexMath::exp(z);

        approx_eq(result, Complex::new(-1.0, 0.0));
    }

    #[test]
    fn test_sin_i() {
        let z = Complex::new(0.0, 1.0);

        let result = ComplexMath::sin(z);

        // sin(i) = i sinh(1)
        let expected = Complex::new(0.0, 1.0_f64.sinh());

        approx_eq(result, expected);
    }

    #[test]
    fn test_real_consistency() {
        let x = 0.7_f64;
        let z = Complex::new(x, 0.0);

        approx_eq(Complex::new(x.sin(), 0.0), ComplexMath::sin(z.clone()));
        approx_eq(Complex::new(x.cos(), 0.0), ComplexMath::cos(z.clone()));
        approx_eq(Complex::new(x.exp(), 0.0), ComplexMath::exp(z.clone()));
        approx_eq(Complex::new(x.ln(), 0.0), ComplexMath::ln(z.clone()));
    }

    #[test]
    fn test_sqrt() {
        let z = Complex::new(3.0, 4.0);

        let sqrt = ComplexMath::sqrt(z.clone());
        let back = sqrt.clone() * sqrt;

        approx_eq(back, z);
    }

    #[test]
    fn test_sqrt_branch() {
        // (z, sqrt(z))
        let list = [
            (Complex::new(3.0, 4.0), Complex::new(2.0, 1.0)),
            (Complex::new(3.0, -4.0), Complex::new(2.0, -1.0)),
            (Complex::new(-3.0, 4.0), Complex::new(1.0, 2.0)),
            (Complex::new(-3.0, -4.0), Complex::new(1.0, -2.0)),
            (Complex::new(0.0, 2.0), Complex::new(1.0, 1.0)),
            (Complex::new(0.0, -2.0), Complex::new(1.0, -1.0)),
            (Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)),
            (Complex::new(-1.0, 0.0), Complex::new(0.0, 1.0)),
            (Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)),
        ];

        for (z, expected) in list {
            approx_eq(ComplexMath::sqrt(z), expected);
        }
    }

    #[test]
    fn test_sqrt_accuracy() {
        // finite test cases
        for (z, expected) in [
            ( c(1e308, 1e308), c(1.098684113467809966e154, 4.5508986056222734e153) ),
            ( c(-1e308, 1e308), c(4.5508986056222734e153, 1.098684113467809966e154) ),
            ( c(1e308, 0.0), c(1e154, 0.0) ),
            ( c(-1e308, 0.0), c(0.0, 1e154) ),
            ( c(-1e308, -0.0), c(0.0, 1e154) ),
            ( c(-4.0, -0.0), c(0.0, 2.0) ), // sqrt(-4.0 - 0.0i) -> +2.0i
            ( c(-4.0, 0.0), c(0.0, 2.0) ), // sqrt(-4.0 + 0.0i) -> +2.0i
            ( c(1e308, 1e-308), c(1e154, 0.0) ), // Im(sqrt(z)) = 5.0...E-463
            ( c(1e-308, 1e308), c(7.071067811865475244e153, 7.071067811865475244e153) ),
            ( c(1e-308, 1e-308), c(1.098684113467809966e-154, 4.5508986056222734e-155)),
            ( c(-1e-308, 1e-308), c(4.5508986056222734e-155, 1.098684113467809966e-154) ),

        ] {
            let result = ComplexMath::sqrt(z);
            assert_rel(result, expected);
        }

        // NaN test cases
        for z in [
            c(f64::NAN, 0.0),
            c(0.0, f64::NAN),
            c(f64::NAN, f64::NAN),
        ] {
            let result = ComplexMath::sqrt(z);
            assert!(result.re.is_nan() && result.im.is_nan(), "sqrt({z}) -> {result}");
        }

        // infinite test cases
        let result = ComplexMath::sqrt(c(f64::INFINITY, 0.0));
        assert!(result.re.is_infinite() && result.re.is_positive() && result.im.is_zero());
        let result = ComplexMath::sqrt(c(f64::NEG_INFINITY, 0.0));
        assert!(result.re.is_zero() && result.im.is_infinite() && result.im.is_positive());
        let result = ComplexMath::sqrt(c(f64::INFINITY, 1.0));
        assert!(result.re.is_infinite() && result.re.is_positive() && result.im.is_zero());
        let result = ComplexMath::sqrt(c(f64::NEG_INFINITY, 1.0));
        assert!(result.re.is_zero() && result.im.is_infinite() && result.im.is_positive());
    }

    #[test]
    fn test_powc() {
        let z = Complex::new(1.2, 0.7);
        let w = Complex::new(-0.3, 0.5);

        let result = ComplexMath::powc(z.clone(), w.clone());

        let expected = (w * z.ln()).exp();

        approx_eq(result, expected);
    }

    #[test]
    fn test_powc_accuracy() {
        // finite test cases
        for (z, w, expected) in [
            ( c(0.0, 0.0), c(2.5, 0.0), c(0.0, 0.0) ),
            ( c(0.0, 0.0), c(2.5, 3.0), c(0.0, 0.0) ),
            ( c(0.0, 0.0), c(1.0, 3.0), c(0.0, 0.0) ),
            ( c(0.0, 0.0), c(0.5, -3.0), c(0.0, 0.0) ),
            ( c(1.0, 0.0), c(2.5, 3.0), c(1.0, 0.0) ),
            ( c(0.0, 1.0), c(2.5, 3.0), c(-6.35214599841284e-3, -6.35214599841284e-3) ),
        ] {
            let result = ComplexMath::powc(z, w);
            assert_rel(result, expected);
        }

        // infinite/NaN test cases
        for (z, w) in [
            ( c(0.0, 0.0), c(0.0, 3.0) ),
            ( c(0.0, 0.0), c(0.0, 0.0) ),
            ( c(0.0, 0.0), c(-2.5, 0.0) ),
            ( c(0.0, 0.0), c(-2.5, 3.0) ),
        ] {
            let result = ComplexMath::powc(z, w);
            assert!(!result.re.is_finite() || !result.im.is_finite(), "({z})^({w}) -> {result}");
        }
    }

    #[test]
    fn test_powi() {
        let z = Complex::new(1.1, -0.4);

        for n in -8..12 {
            let result = ComplexMath::powi(z.clone(), n);
            let expected = num_complex::ComplexFloat::powi(z, n);
            approx_eq(result, expected);
        }

        {
            let result = ComplexMath::powi(z.clone(), i32::MIN);
            let expected = num_complex::ComplexFloat::powi(z, i32::MIN);
            approx_eq(result, expected);
        }
    }

    #[test]
    fn test_tan_identity() {
        let z = Complex::new(0.8, -0.3);

        let tan = ComplexMath::tan(z.clone());
        let expected = ComplexMath::sin(z.clone()) / ComplexMath::cos(z);

        approx_eq(tan, expected);
    }

    #[test]
    fn test_log10() {
        let z = Complex::new(1.3, 0.4);

        let result = ComplexMath::log10(z.clone());

        let expected = ComplexMath::ln(z) / std::f64::consts::LN_10;

        approx_eq(result, expected);
    }

    #[test]
    fn test_asinh_accuracy() {
        let cases = [
            // z, expected
            // Large inputs
            ( c(-1.0e4, 0.0), c(-9.903487555036128, 0.0) ),
            ( c(-1.0e8, 0.0), c(-19.113827924512311, 0.0) ),
            ( c(1.0e8, 1.0e8), c(19.460401514792283, 0.7853981633974483) ),
            ( c(-1.0e8, 1.0e8), c(-19.460401514792283, 0.7853981633974483) ),
            ( c(1.0e8, -1.0e8), c(19.460401514792283, -0.7853981633974483) ),
            ( c(0.0, 1.0e8), c(19.113827924512311, 1.5707963267948966) ),
            ( c(1.0e200, 0.0), c(461.2101657793691, 0.0) ),
            ( c(0.0, 1.0e200), c(461.2101657793691, std::f64::consts::FRAC_PI_2), ),
            ( c(1.0e200, 1.0e200), c(461.55673936964905, 0.7853981633974483) ),
            ( c(1.0e308, 1.0e308),  c(710.235929413006, 0.7853981633974483) ),
            // small |z| (cancellation in ln(1 + eps))
            ( c(1.0e-10, 0.0), c(1.0e-10, 0.0) ),
            ( c(0.0, 1.0e-10), c(0.0, 1.0e-10) ),
            ( c(1.0e-10, 1.0e-10), c(1.0e-10, 1.0e-10) ),
            ( c(1.0e-5, 0.5), c(1.1547005383279316e-5, 0.52359877555980886) ),
            // imaginary axis, |y| < 1 and |y| > 1
            ( c(0.0, 0.5), c(0.0, 0.5235987755982989) ),
            ( c(0.0, 2.0), c(1.3169578969248167, 1.5707963267948966) ),
            // near the branch points +-i
            ( c(1.0e-3, 1.0), c(0.0316254112399638, 1.5391761860173468) ),
            ( c(1.0e-8, 1.0000001), c(0.00044777087109671771, 1.5707739939449679) ),
            ( c(1.0e-8, 0.9999999), c(2.2332851059580247e-5, 1.5703485559166216) ),
            // all quadrants
            ( c(3.0, 4.0), c(2.2999140408792696, 0.91761685335147866) ),
            ( c(-3.0, 4.0), c(-2.2999140408792696, 0.91761685335147866) ),
            ( c(3.0, -4.0), c(2.2999140408792696, -0.91761685335147866) ),
            ( c(-3.0, -4.0), c(-2.2999140408792696, -0.91761685335147866) ),
            ( c(0.5, -0.5), c(0.53063753095251783, -0.45227844715119068) ),
        ];

        for (z, expected) in cases {
            assert_rel(ComplexMath::asinh(z), expected);
        }
    }

    #[test]
    fn test_asinh_tiny_real_part() {
        // x^2 underflows for |x| < ~1.5e-154; the real part must not become 0.
        let cases = [
            // z, expected
            ( c(1.0e-170, 0.0), c(1.0e-170, 0.0) ),
            ( c(-1.0e-170, 0.0), c(-1.0e-170, 0.0) ),
            ( c(1.0e-170, 1.0e-170), c(1.0e-170, 1.0e-170) ),
            ( c(1.0e-170, 0.5), c(1.1547005383792515e-170, 0.5235987755982989) ),
            ( c(1.0e-200, 0.3), c(1.0482848367219182e-200, 0.3046926540153975) ),
            ( c(1.0e-300, 0.999), c(2.236627204212921e-299, 1.526071239626163) ),
            ( c(5.0e-324, 0.0), c(5.0e-324, 0.0) ),
        ];
        for (z, expected) in cases {
            assert_rel(ComplexMath::asinh(z), expected);
        }

        // asin(z) = -i asinh(iz) shares the same path.
        let cases = [
            ( c(0.0, 1.0e-170), c(0.0, 1.0e-170) ),
            ( c(1.0e-170, 1.0e-170), c(1.0e-170, 1.0e-170) ),
            ( c(0.3, 1.0e-200), c(0.3046926540153975, 1.0482848367219182e-200) ),
        ];
        for (z, expected) in cases {
            assert_rel(ComplexMath::asin(z), expected);
        }
    }

    #[test]
    fn test_asin_accuracy() {
        let cases = [
            // z, expected
            // real axis: x < -1 and x > 1 are cuts, Im >= 0 (continuous from above, as C99)
            ( c(-1.0e8, 0.0), c(-1.5707963267948966, 19.113827924512311) ),
            ( c(1.0e8, 0.0), c(1.5707963267948966, 19.113827924512311) ),
            ( c(-2.0, 0.0), c(-1.5707963267948966, 1.3169578969248167) ),
            ( c(2.0, 0.0), c(1.5707963267948966, 1.3169578969248167) ),
            // real axis, |x| <= 1
            ( c(0.5, 0.0), c(0.5235987755982989, 0.0) ),
            ( c(-0.5, 0.0), c(-0.5235987755982989, 0.0) ),
            ( c(1.0e-10, 0.0), c(1.0e-10, 0.0) ),
            // imaginary axis
            ( c(0.0, 1.0e-10), c(0.0, 1.0e-10) ),
            ( c(0.0, 1.0e8), c(0.0, 19.113827924512311) ),
            ( c(0.0, -1.0e8), c(0.0, -19.113827924512311) ),
            // just off the cuts: the sign of Im follows the sign of Im z
            ( c(-1.0e8, 1.0e-3), c(-1.5707963267848966, 19.113827924512311) ),
            ( c(1.0e8, -1.0e-3), c(1.5707963267848966, -19.113827924512311) ),
            ( c(2.0, 1.0e-8), c(1.5707963210213939, 1.3169578969248167) ),
            ( c(-2.0, 1.0e-8), c(-1.5707963210213939, 1.3169578969248167) ),
            ( c(2.0, -1.0e-8), c(1.5707963210213939, -1.3169578969248167) ),
            // small |z|
            ( c(1.0e-10, 1.0e-10), c(1.0e-10, 1.0e-10) ),
            // all quadrants
            ( c(0.5, 0.5), c(0.45227844715119068, 0.53063753095251783) ),
            ( c(3.0, 4.0), c(0.63398386563917672, 2.3055090312434769) ),
            ( c(-3.0, 4.0), c(-0.63398386563917672, 2.3055090312434769) ),
            ( c(3.0, -4.0), c(0.63398386563917672, -2.3055090312434769) ),
            ( c(-3.0, -4.0), c(-0.63398386563917672, -2.3055090312434769) ),
            // Exactly at the branch points
            (c(0.0, 1.0), c(0.0, 0.8813735870195430)),
            (c(0.0, -1.0), c(0.0, -0.8813735870195430)),
            // Extremely small values
            (c(1.0e-200, 0.0), c(1.0e-200, 0.0)),
            // Signed zero on branch cuts
            (c(2.0, 0.0), c(std::f64::consts::FRAC_PI_2, 1.3169578969248167)),
            (c(2.0, -0.0), c(std::f64::consts::FRAC_PI_2, -1.3169578969248167)),
        ];

        for (z, expected) in cases {
            assert_rel(ComplexMath::asin(z), expected);
        }
    }

    #[test]
    fn test_tan_accuracy() {
        let cases = [
            // target, expected
            ( c(0.3, 400.0), c(0.0, 1.0) ),
            ( c(0.3, -400.0), c(0.0, -1.0) ),
            ( c(1e-10, 0.0), c(1e-10, 0.0) ),
            ( c(0.0, 1e-10), c(0.0, 1e-10) ),
            ( c(std::f64::consts::FRAC_1_SQRT_2, 0.0), c(0.854510432009601, 0.0) ),
            ( c(1.0, 2.0), c(0.033812826079896690, 1.01479361614663) ),
            ( c(100.0, 25.0), c(-3.36874405845758474e-22, 0.999999999999999999) ),
        ];

        for (z, expected) in cases {
            assert_rel(ComplexMath::tan(z), expected);
        }
    }

    #[test]
    fn test_tanh_accuracy() {
        let cases = [
            // target, expected
            ( c(1.0e-10, 0.0), c(1.0e-10, 0.0) ),
            ( c(20.0, 0.0), c(0.9999999999999999915, 0.0) ),
            ( c(21.0, 0.0), c(0.99999999999999999885, 0.0) ),
            ( c(400.0, 0.0), c(1.0, 0.0) ),
            ( c(-1e-10, 0.0), c(-1.0e-10, 0.0) ),
            ( c(-21.0, 0.0), c(-0.99999999999999999885, 0.0) ),
            ( c(-400.0, 0.0), c(-1.0, 0.0) ),
            ( c(1.0, 2.0), c(1.16673625724091988, -0.2434582011857252527) ),
            ( c(1.0, -2.0), c(1.16673625724091988, 0.2434582011857252527) ),
            ( c(0.0, 1.0), c(0.0, 1.5574077246549023) ),
            ( c(0.0, std::f64::consts::FRAC_PI_2), c(0.0, 1.633123935319537e16) ),
            (
                c(1e-5, std::f64::consts::FRAC_PI_2),
                c(
                    1e5 + 3.333e-6,
                    6.123233995736766e-17 /* π/2 - FRAC_PI_2 */ / (1e-5_f64.sinh() * 1e-5_f64.sinh()) /* for rounding-error */
                )
            ),
            ( c(25.0, 1.0), c(1.0, 3.507614547488030451e-22) ),
            (
                c(1.0, std::f64::consts::PI),
                c(
                    1.0_f64.tanh(),
                    -1.2246467991473532e-16 /* π - PI */ / (1.0_f64.cosh() * 1.0_f64.cosh())) /* for rounding-error */
            ),
        ];

        for (z, expected) in cases {
            assert_rel(ComplexMath::tanh(z), expected);
        }
    }

    #[test]
    fn test_atanh_accuracy() {
        let cases = [
            // target, expected
            // Pure imaginary values
            ( c(0.0, 0.5), c(0.0, 0.4636476090008061) ),
            ( c(0.0, 2.0), c(0.0, 1.1071487177940904) ),
            ( c(0.0, -0.5), c(0.0, -0.4636476090008061) ),
            ( c(0.0, -2.0), c(0.0, -1.1071487177940904) ),

            // General complex values
            ( c(0.5, 0.5), c(0.40235947810852507, 0.5535743588970452) ),
            ( c(0.5, 1.0), c(0.2388778612568591, 0.847575660670829) ),
            ( c(2.0, 1.0), c(0.40235947810852507, 1.3389725222944935) ),

            // Negative real and positive imaginary parts
            ( c(-0.5, 0.5), c(-0.40235947810852507, 0.5535743588970452) ),

            // Positive branch cut
            ( c(2.0, 0.0), c(0.5493061443340548, std::f64::consts::FRAC_PI_2) ),
            ( c(2.0, -0.0), c(0.5493061443340548, -std::f64::consts::FRAC_PI_2) ),

            // Negative branch cut
            ( c(-2.0, 0.0), c(-0.5493061443340548, std::f64::consts::FRAC_PI_2) ),
            ( c(-2.0, -0.0), c(-0.5493061443340548, -std::f64::consts::FRAC_PI_2) ),

            // Inside the branch cut interval
            ( c(0.5, 0.0), c(0.5493061443340548, 0.0) ),
            ( c(0.5, -0.0), c(0.5493061443340548, -0.0) ),

            // Approaching the positive branch point from inside
            ( c(0.999999, 0.0), c(7.254328619247669, 0.0) ),

            // Approaching the positive branch point from above
            ( c(1.0, 1.0e-8), c(9.556913962256155, 0.7853981658974483) ),

            // Approaching the negative branch point from inside
            ( c(-0.999999, 0.0), c(-7.254328619247669, 0.0) ),
            ( c(-0.999999, 1.0e-10), c(-7.2543286167620472, 5.00000248333458343e-5) ),
            ( c(-0.999999, -1.0e-10), c(-7.2543286167620472, -5.00000248333458343e-5) ),

            // small |z|
            ( c(1.0e-10, 0.0), c(1.0e-10, 0.0) ),
            ( c(1.0e-10, 1.0e-10), c(1.0e-10, 1.0e-10) ),

            // large |z|
            ( c(1.0e-200, 0.0), c(1.0e-200, 0.0) ),
            ( c(1.0e+200, 0.0), c(1.0e-200, std::f64::consts::FRAC_PI_2) ),
            ( c(1.0e+200, 1.0e+200), c(5.0e-201, std::f64::consts::FRAC_PI_2) ),
        ];

        for (z, expected) in cases {
            assert_rel(ComplexMath::atanh(z), expected);
        }
    }

    #[test]
    fn test_atanh_imag_near_branch_points() {
        // 1 - x^2 - y^2 cancelled near x = +-1 (relative error up to 8e-11),
        // which the 1e-10 tolerance of assert_rel cannot detect.
        let cases = [
            // z, expected
            ( c(1.000000010318334, 3.3480469705374203e-9), c(9.51621969283969, 1.4139175933520196) ),
            ( c(0.9999999999, 1.0e-12), c(11.85947401510907, 0.0049998329299221645) ),
            ( c(-0.9999999, -1.0e-9), c(-8.4055963922722, -0.004999833595964148) ),
        ];
        for (z, expected) in cases {
            let r = ComplexMath::atanh(z);
            for (got, exp, part) in [(r.re, expected.re, "re"), (r.im, expected.im, "im")] {
                assert!((got - exp).abs() <= 1.0e-14 * exp.abs(), "atanh({z}).{part}: {got:e} vs {exp:e}");
            }
        }
    }

    #[test]
    fn test_atanh_imag_does_not_overflow() {
        // 2y and x^2 + y^2 overflowed for |y| > ~9e307, and atan2(-inf, -inf)
        // gave 3*pi/8 instead of pi/2.
        let cases = [
            // z, expected imaginary part
            ( c(1.0e308, 1.0e308), std::f64::consts::FRAC_PI_2 ),
            ( c(-1.0e308, 1.0e308), std::f64::consts::FRAC_PI_2 ),
            ( c(0.0, 1.0e308), std::f64::consts::FRAC_PI_2 ),
            ( c(1.0e300, -1.0e308), -std::f64::consts::FRAC_PI_2 ),
            ( c(1.7e308, 1.7e308), std::f64::consts::FRAC_PI_2 ),
            ( c(1.0e308, 1.0), std::f64::consts::FRAC_PI_2 ),
        ];
        for (z, expected) in cases {
            let r = ComplexMath::atanh(z);
            assert!((r.im - expected).abs() <= 1.0e-15, "atanh({z}).im: {:e}", r.im);
            // The real part is about x / |z|^2, below 1e-300 for these inputs.
            assert!(r.re.is_finite() && r.re.abs() < 1.0e-300, "atanh({z}).re: {:e}", r.re);
            assert_eq!(r.re.is_sign_negative(), z.re.is_sign_negative(), "atanh({z}).re: {:e}", r.re);
        }
    }
}
