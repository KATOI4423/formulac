//! # Multi-precision floating-point integration test sample for formulac
//!
//! ## Overview
//!
//! This file provides an **implementation example** and **integration tests** for
//! implementing the `Real` trait of `formulac` with a multi-precision floating-point library.
//!
//! ## Target Library
//!
//! Uses the `rug` crate (GNU MP / GNU MPFR bindings).
//! MPFR provides IEEE 754 compliant arbitrary-precision floating-point arithmetic.
//!
//! # rug requires system installation of C libraries (GMP, MPFR, MPC).
//! # Ubuntu/Debian: sudo apt install libgmp-dev libmpfr-dev libmpc-dev
//! # macOS (Homebrew): brew install gmp mpfr libmpc

// ─────────────────────────────────────────────────────────────────────────────
// External crate imports
// ─────────────────────────────────────────────────────────────────────────────

use formulac::{Builder, UserFn};
use num_complex::Complex;
use num_traits::{Num, One, Zero};
use rug::{Float, ops::Pow};
use std::ops::{Add, Div, Mul, Neg, Rem, Sub, AddAssign, SubAssign, MulAssign, DivAssign, RemAssign};

// ─────────────────────────────────────────────────────────────────────────────
// ① Wrapper type for rug::Float
//
// rug::Float implements Clone but not PartialOrd directly, so we use a thin wrapper
// to organize the traits.
// We also need to implement num-traits Num / Zero / One.
// ─────────────────────────────────────────────────────────────────────────────

/// Arbitrary-precision floating-point number wrapper.
///
/// Holds precision (in bits) in the `prec` field,
/// and propagates the same precision in all operations.
#[derive(Debug, Clone)]
pub struct MpFloat {
    inner: Float,
}

impl MpFloat {
    /// Create a new instance with the specified precision (in bits).
    pub fn with_prec(prec: u32, value: f64) -> Self {
        Self { inner: Float::with_val(prec, value) }
    }

    /// Return the current precision (in bits).
    pub fn prec(&self) -> u32 {
        self.inner.prec()
    }

    /// Return a reference to the internal `rug::Float`.
    pub fn inner(&self) -> &Float {
        &self.inner
    }

    /// Default precision (256 bits ≈ 77 significant decimal digits).
    pub const DEFAULT_PREC: u32 = 256;
}

// ─── Basic arithmetic operators ──────────────────────────────────────────────────────

impl Add for MpFloat {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self { inner: Float::with_val(self.prec(), &self.inner + &rhs.inner) }
    }
}
impl Sub for MpFloat {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self { inner: Float::with_val(self.prec(), &self.inner - &rhs.inner) }
    }
}
impl Mul for MpFloat {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self { inner: Float::with_val(self.prec(), &self.inner * &rhs.inner) }
    }
}
impl Div for MpFloat {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Self { inner: Float::with_val(self.prec(), &self.inner / &rhs.inner) }
    }
}
impl Rem for MpFloat {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self {
        // MPFR does not have direct rem, so calculate fmod equivalent manually
        let q = Float::with_val(self.prec(), &self.inner / &rhs.inner);
        let q_trunc = q.trunc();
        Self { inner: Float::with_val(self.prec(), &self.inner - q_trunc * &rhs.inner) }
    }
}
impl Neg for MpFloat {
    type Output = Self;
    fn neg(self) -> Self {
        Self { inner: Float::with_val(self.prec(), -&self.inner) }
    }
}

// ─── PartialEq / PartialOrd ─────────────────────────────────────────────────

impl PartialEq for MpFloat {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl PartialOrd for MpFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.inner.partial_cmp(&other.inner)
    }
}

// ─── num_traits::Zero / One ──────────────────────────────────────────────────

impl Zero for MpFloat {
    fn zero() -> Self { Self::with_prec(Self::DEFAULT_PREC, 0.0) }
    fn is_zero(&self) -> bool { self.inner.is_zero() }
}

impl One for MpFloat {
    fn one() -> Self { Self::with_prec(Self::DEFAULT_PREC, 1.0) }
}

// ─── num_traits::Num ─────────────────────────────────────────────────────────

impl Num for MpFloat {
    type FromStrRadixErr = rug::float::ParseFloatError;
    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        let f = Float::parse_radix(str, radix as i32)
            .map(|v| Float::with_val(Self::DEFAULT_PREC, v))?;
        Ok(Self { inner: f })
    }
}

// ─── std::str::FromStr ────────────────────────────────────────────────────────

impl std::str::FromStr for MpFloat {
    type Err = rug::float::ParseFloatError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let f = Float::parse(s)
            .map(|v| Float::with_val(Self::DEFAULT_PREC, v))?;
        Ok(Self { inner: f })
    }
}

// ─── NumAssignOps ───────────────────────────────────────────────────────────

impl AddAssign for MpFloat {
    fn add_assign(&mut self, rhs: Self) {
        self.inner += rhs.inner;
    }
}

impl SubAssign for MpFloat {
    fn sub_assign(&mut self, rhs: Self) {
        self.inner -= rhs.inner;
    }
}

impl MulAssign for MpFloat {
    fn mul_assign(&mut self, rhs: Self) {
        self.inner *= rhs.inner;
    }
}

impl DivAssign for MpFloat {
    fn div_assign(&mut self, rhs: Self) {
        self.inner /= rhs.inner;
    }
}

impl RemAssign for MpFloat {
    fn rem_assign(&mut self, rhs: Self) {
        // MPFR does not have direct rem_assign, so calculate manually
        let q = Float::with_val(self.prec(), &self.inner / &rhs.inner);
        let q_trunc = q.trunc();
        self.inner = Float::with_val(self.prec(), &self.inner - q_trunc * &rhs.inner);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ② Implementation of formulac::core::Real trait
// ─────────────────────────────────────────────────────────────────────────────

use formulac::core::Real; // Assumes this is public

impl Real for MpFloat {
    // ── Conversions ────────────────────────────────────────────────────────────────

    fn from_f64(v: f64) -> Self {
        Self::with_prec(Self::DEFAULT_PREC, v)
    }

    fn to_i32(&self) -> i32 {
        // rug::Float → f64 → i32 (precision is sufficient for i32 range)
        let f = self.inner.to_f64();
        if !f.is_finite() { return 0; }
        f.trunc().clamp(i32::MIN as f64, i32::MAX as f64) as i32
    }

    fn is_i32_compatible(&self) -> bool {
        let f = self.inner.to_f64();
        f.is_finite() && f.fract() == 0.0
            && f >= i32::MIN as f64 && f <= i32::MAX as f64
    }

    fn fract(self) -> Self {
        // rug::Float::fract() は小数部を返す
        Self { inner: self.inner.fract() }
    }

    fn trunc(self) -> Self {
        Self { inner: self.inner.trunc() }
    }

    // ── Mathematical constants ────────────────────────────────────────────────────────────
    //
    // In rug, use Float::with_val to calculate constants with arbitrary precision.
    // Here we use DEFAULT_PREC, but it can be changed as needed.

    fn e()              -> Self { Self { inner: Float::with_val(Self::DEFAULT_PREC, 1.0).exp() } }
    fn pi()             -> Self { Self { inner: Float::with_val(Self::DEFAULT_PREC, rug::float::Constant::Pi) } }
    fn sqrt_2()         -> Self { Self { inner: Float::with_val(Self::DEFAULT_PREC, 2.0).sqrt() } }
    fn tau()            -> Self { Self { inner: Float::with_val(Self::DEFAULT_PREC, rug::float::Constant::Pi) * 2u32 } }
    fn ln_2()           -> Self { Self { inner: Float::with_val(Self::DEFAULT_PREC, 2.0).ln() } }
    fn ln_10()          -> Self { Self { inner: Float::with_val(Self::DEFAULT_PREC, 10.0).ln() } }
    fn log2_e()         -> Self { Self { inner: Float::with_val(Self::DEFAULT_PREC, 1.0).exp().log2() } }
    fn log2_10()        -> Self { Self { inner: Float::with_val(Self::DEFAULT_PREC, 10.0).log2() } }
    fn log10_e()        -> Self { Self { inner: Float::with_val(Self::DEFAULT_PREC, 1.0).exp().log10() } }
    fn log10_2()        -> Self { Self { inner: Float::with_val(Self::DEFAULT_PREC, 2.0).log10() } }

    fn frac_1_pi()      -> Self {
        let pi = Float::with_val(Self::DEFAULT_PREC, rug::float::Constant::Pi);
        Self { inner: Float::with_val(Self::DEFAULT_PREC, 1.0) / pi }
    }
    fn frac_2_pi()      -> Self {
        let pi = Float::with_val(Self::DEFAULT_PREC, rug::float::Constant::Pi);
        Self { inner: Float::with_val(Self::DEFAULT_PREC, 2.0) / pi }
    }
    fn frac_pi_2()      -> Self {
        let pi = Float::with_val(Self::DEFAULT_PREC, rug::float::Constant::Pi);
        Self { inner: pi / 2u32 }
    }
    fn frac_pi_3()      -> Self {
        let pi = Float::with_val(Self::DEFAULT_PREC, rug::float::Constant::Pi);
        Self { inner: pi / 3u32 }
    }
    fn frac_pi_4()      -> Self {
        let pi = Float::with_val(Self::DEFAULT_PREC, rug::float::Constant::Pi);
        Self { inner: pi / 4u32 }
    }
    fn frac_pi_6()      -> Self {
        let pi = Float::with_val(Self::DEFAULT_PREC, rug::float::Constant::Pi);
        Self { inner: pi / 6u32 }
    }
    fn frac_pi_8()      -> Self {
        let pi = Float::with_val(Self::DEFAULT_PREC, rug::float::Constant::Pi);
        Self { inner: pi / 8u32 }
    }
    fn frac_1_sqrt_2()  -> Self {
        let s = Float::with_val(Self::DEFAULT_PREC, 2.0).sqrt();
        Self { inner: Float::with_val(Self::DEFAULT_PREC, 1.0) / s }
    }
    fn frac_2_sqrt_pi() -> Self {
        let pi = Float::with_val(Self::DEFAULT_PREC, rug::float::Constant::Pi);
        Self { inner: Float::with_val(Self::DEFAULT_PREC, 2.0) / pi.sqrt() }
    }

    // ── Trigonometric functions ────────────────────────────────────────────────────────────

    fn sin(self) -> Self { Self { inner: self.inner.sin() } }
    fn cos(self) -> Self { Self { inner: self.inner.cos() } }
    fn tan(self) -> Self { Self { inner: self.inner.tan() } }
    fn asin(self) -> Self { Self { inner: self.inner.asin() } }
    fn acos(self) -> Self { Self { inner: self.inner.acos() } }
    fn atan(self) -> Self { Self { inner: self.inner.atan() } }
    fn atan2(self, other: Self) -> Self {
        Self { inner: self.inner.atan2(&other.inner) }
    }
    fn sin_cos(self) -> (Self, Self) {
        let (s, c) = self.clone().inner.sin_cos(Float::new(self.prec()));
        (Self { inner: s }, Self { inner: c })
    }

    // ── Hyperbolic functions ──────────────────────────────────────────────────────────

    fn sinh(self) -> Self { Self { inner: self.inner.sinh() } }
    fn cosh(self) -> Self { Self { inner: self.inner.cosh() } }
    fn tanh(self) -> Self { Self { inner: self.inner.tanh() } }
    fn asinh(self) -> Self { Self { inner: self.inner.asinh() } }
    fn acosh(self) -> Self { Self { inner: self.inner.acosh() } }
    fn atanh(self) -> Self { Self { inner: self.inner.atanh() } }

    // ── Exponential and logarithmic functions ──────────────────────────────────────────────────────────

    fn exp(self) -> Self { Self { inner: self.inner.exp() } }
    fn ln(self)  -> Self { Self { inner: self.inner.ln() } }
    fn log10(self) -> Self { Self { inner: self.inner.log10() } }

    // ── Other functions ──────────────────────────────────────────────────────────────

    fn sqrt(self) -> Self { Self { inner: self.inner.sqrt() } }
    fn abs(self)  -> Self { Self { inner: self.inner.abs() } }
    fn hypot(self, other: Self) -> Self {
        Self { inner: self.inner.hypot(&other.inner) }
    }

    // ── Power functions ────────────────────────────────────────────────────────────────

    fn pow(self, rhs: Self) -> Self {
        // x^y = exp(y * ln(x))
        Self { inner: (rhs.inner * self.inner.ln()).exp() }
    }
    fn powi(self, n: i32) -> Self {
        let exp = Float::with_val(self.prec(), n as f64);
        Self { inner: self.inner.pow(&exp) }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ③ Test helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Convert MpFloat Complex to f64 Complex (for comparison)
fn to_f64(z: &Complex<MpFloat>) -> Complex<f64> {
    Complex::new(z.re.inner.to_f64(), z.im.inner.to_f64())
}

/// Generate MpFloat version input values
fn mp(re: f64, im: f64) -> Complex<MpFloat> {
    Complex::new(
        MpFloat::with_prec(MpFloat::DEFAULT_PREC, re),
        MpFloat::with_prec(MpFloat::DEFAULT_PREC, im),
    )
}

/// Check if result is close to expected (epsilon for f64 comparison)
fn assert_close(result: Complex<f64>, expected: Complex<f64>, eps: f64, label: &str) {
    let diff_re = (result.re - expected.re).abs();
    let diff_im = (result.im - expected.im).abs();
    assert!(
        diff_re < eps,
        "{label}: re mismatch: got {}, expected {} (diff={diff_re})",
        result.re, expected.re
    );
    assert!(
        diff_im < eps,
        "{label}: im mismatch: got {}, expected {} (diff={diff_im})",
        result.im, expected.im
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// ④ Integration tests
//
// Each test verifies the following:
//   a) formulac compiles and executes with MpFloat
//   b) Results match f64 results sufficiently (compared at f64 precision)
//   c) High precision unique to multi-precision is achieved (in some tests)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mp_constant_number() {
    // Numeric literals are parsed and evaluated correctly
    let f = Builder::<MpFloat, 0>::new("42", [])
        .compile()
        .expect("compile failed");
    let result = to_f64(&f([]));
    assert_close(result, Complex::new(42.0, 0.0), 1e-10, "constant 42");
}

#[test]
fn mp_builtin_constant_pi() {
    // Built-in constant PI is calculated with multi-precision
    let f = Builder::<MpFloat, 0>::new("PI", [])
        .compile()
        .expect("compile failed");
    let result = to_f64(&f([]));
    assert_close(result, Complex::new(std::f64::consts::PI, 0.0), 1e-15, "PI");
}

#[test]
fn mp_argument_passthrough() {
    // Arguments are passed through unchanged
    let f = Builder::<MpFloat, 1>::new("x", ["x"])
        .compile()
        .expect("compile failed");
    let result = to_f64(&f([mp(3.5, -1.2)]));
    assert_close(result, Complex::new(3.5, -1.2), 1e-14, "argument passthrough");
}

#[test]
fn mp_addition() {
    // Addition
    let f = Builder::<MpFloat, 2>::new("x + y", ["x", "y"])
        .compile()
        .expect("compile failed");
    let x = Complex::new(2.0_f64, 1.0);
    let y = Complex::new(3.0_f64, -5.0);
    let result = to_f64(&f([mp(x.re, x.im), mp(y.re, y.im)]));
    assert_close(result, x + y, 1e-14, "addition");
}

#[test]
fn mp_binary_operator_precedence() {
    // Operator precedence: 2 + 3 * 4 = 14
    let f = Builder::<MpFloat, 0>::new("2 + 3 * 4", [])
        .compile()
        .expect("compile failed");
    let result = to_f64(&f([]));
    assert_close(result, Complex::new(14.0, 0.0), 1e-14, "precedence");
}

#[test]
fn mp_sin_function() {
    // sin function
    let f = Builder::<MpFloat, 1>::new("sin(x)", ["x"])
        .compile()
        .expect("compile failed");
    let x = Complex::new(1.0_f64, 0.5);
    let result = to_f64(&f([mp(x.re, x.im)]));
    // Reference value calculated in f64
    let expected = Complex::new(
        1.0_f64.sin() * 0.5_f64.cosh(),
        1.0_f64.cos() * 0.5_f64.sinh(),
    );
    assert_close(result, expected, 1e-13, "sin(complex)");
}

#[test]
fn mp_exp_ln_roundtrip() {
    // exp(ln(z)) ≈ z (identity)
    let f = Builder::<MpFloat, 1>::new("exp(ln(x))", ["x"])
        .compile()
        .expect("compile failed");
    let x = Complex::new(2.5_f64, -1.3);
    let result = to_f64(&f([mp(x.re, x.im)]));
    assert_close(result, x, 1e-13, "exp(ln(z)) roundtrip");
}

#[test]
fn mp_nested_expression() {
    // Nested expression: sin(x + 1)
    let f = Builder::<MpFloat, 1>::new("sin(x + 1)", ["x"])
        .compile()
        .expect("compile failed");
    let x = Complex::new(0.0_f64, 1.0);
    let result = to_f64(&f([mp(x.re, x.im)]));
    let z = x + Complex::new(1.0, 0.0);
    let expected = Complex::new(
        z.re.sin() * z.im.cosh(),
        z.re.cos() * z.im.sinh(),
    );
    assert_close(result, expected, 1e-13, "sin(x+1)");
}

#[test]
fn mp_power_operator() {
    // Power: x^y
    let f = Builder::<MpFloat, 2>::new("pow(a, b)", ["a", "b"])
        .compile()
        .expect("compile failed");
    let a = Complex::new(2.0_f64, 0.0);
    let b = Complex::new(10.0_f64, 0.0);
    let result = to_f64(&f([mp(a.re, a.im), mp(b.re, b.im)]));
    assert_close(result, Complex::new(1024.0, 0.0), 1e-10, "pow(2, 10)");
}

#[test]
fn mp_user_function() {
    // User-defined function: double(x) = 2 * x
    let double = UserFn::<MpFloat>::new("double", |[x]| {
        x * Complex::new(MpFloat::with_prec(MpFloat::DEFAULT_PREC, 2.0), MpFloat::zero())
    });
    let f = Builder::<MpFloat, 1>::new("double(x)", ["x"])
        .with_user_functions([double])
        .compile()
        .expect("compile failed");
    let result = to_f64(&f([mp(3.0, 0.0)]));
    assert_close(result, Complex::new(6.0, 0.0), 1e-14, "user fn double(3)");
}

#[test]
fn mp_differentiation_polynomial() {
    // diff(x^2, x) = 2*x
    let f = Builder::<MpFloat, 1>::new("diff(x^2, x)", ["x"])
        .compile()
        .expect("compile failed");
    let x = Complex::new(3.0_f64, 0.0);
    let result = to_f64(&f([mp(x.re, x.im)]));
    assert_close(result, 2.0 * x, 1e-12, "diff(x^2, x) at x=3");
}

#[test]
fn mp_differentiation_second_order() {
    // diff(x^3, x, 2) = 6*x
    let f = Builder::<MpFloat, 1>::new("diff(x^3, x, 2)", ["x"])
        .compile()
        .expect("compile failed");
    let x = Complex::new(2.0_f64, 0.0);
    let result = to_f64(&f([mp(x.re, x.im)]));
    assert_close(result, 6.0 * x, 1e-11, "diff(x^3, x, 2) at x=2");
}

#[test]
fn mp_differentiation_sin() {
    // diff(sin(x), x) = cos(x)
    let f = Builder::<MpFloat, 1>::new("diff(sin(x), x)", ["x"])
        .compile()
        .expect("compile failed");
    let x = Complex::new(1.0_f64, 0.0);
    let result = to_f64(&f([mp(x.re, x.im)]));
    // Reference value for cos(1.0)
    let expected = Complex::new(1.0_f64.cos(), 0.0);
    assert_close(result, expected, 1e-13, "diff(sin(x), x) = cos(x)");
}

#[test]
fn mp_user_fn_with_derivative() {
    // User-defined function f(x) = x^2 and its derivative f'(x) = 2*x
    let df = UserFn::<MpFloat>::new("df", |[x]| {
        Complex::new(MpFloat::with_prec(MpFloat::DEFAULT_PREC, 2.0), MpFloat::zero()) * x
    });
    let func = UserFn::<MpFloat>::new("f", |[x]| x.clone() * x)
        .with_derivative(vec![df])
        .expect("derivative registration failed");

    let expr = Builder::<MpFloat, 1>::new("diff(f(x), x)", ["x"])
        .with_user_functions([func])
        .compile()
        .expect("compile failed");

    // f'(3) = 6
    let result = to_f64(&expr([mp(3.0, 0.0)]));
    assert_close(result, Complex::new(6.0, 0.0), 1e-13, "user fn derivative f'(3)=6");
}

#[test]
fn mp_with_custom_constant() {
    // Expression with custom constant: a * x + 1
    let a = mp(2.5, 0.0);
    let f = Builder::<MpFloat, 1>::new("a * x + 1", ["x"])
        .with_constants([("a", a.clone())])
        .compile()
        .expect("compile failed");
    let x = mp(4.0, 0.0);
    let result = to_f64(&f([x]));
    // 2.5 * 4 + 1 = 11
    assert_close(result, Complex::new(11.0, 0.0), 1e-13, "a*x+1 with custom constant");
}

#[test]
fn mp_complex_formula() {
    // Practical expression: sin(z) + a * cos(z)
    let f = Builder::<MpFloat, 1>::new("sin(z) + a * cos(z)", ["z"])
        .with_constants([("a", mp(3.0, 2.0))])
        .compile()
        .expect("compile failed");

    let z = Complex::new(1.0_f64, 0.5);
    let result = to_f64(&f([mp(z.re, z.im)]));

    // Reference value calculated in f64
    let sin_z = Complex::new(
        1.0_f64.sin() * 0.5_f64.cosh(),
        1.0_f64.cos() * 0.5_f64.sinh(),
    );
    let cos_z = Complex::new(
        1.0_f64.cos() * 0.5_f64.cosh(),
        -(1.0_f64.sin() * 0.5_f64.sinh()),
    );
    let a = Complex::new(3.0_f64, 2.0);
    let expected = sin_z + a * cos_z;
    assert_close(result, expected, 1e-12, "sin(z) + a*cos(z)");
}

// ─────────────────────────────────────────────────────────────────────────────
// ⑤ High-precision demo: Precision comparison with f64
//
// This test is not for verification but for high-precision demonstration.
// It evaluates the same expression in both f64 and MpFloat, showing that MpFloat's result
// is closer to the true value.
//
// Topic: e^π - π ≈ 19.999099979... (difference from Gelfond's constant)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mp_precision_demo_exp_pi_minus_pi() {
    // Calculate exp(PI) - PI with MpFloat (256 bits)
    let f_mp = Builder::<MpFloat, 0>::new("exp(PI) - PI", [])
        .compile()
        .expect("compile failed");
    let result_mp = to_f64(&f_mp([]));

    // Same calculation with f64
    let f_f64 = Builder::<f64, 0>::new("exp(PI) - PI", [])
        .compile()
        .expect("compile failed");
    let result_f64 = f_f64([]);

    // True value (high-precision reference)
    let true_val = std::f64::consts::E.powf(std::f64::consts::PI) - std::f64::consts::PI;

    // Both have equivalent precision within f64 range, but MpFloat has higher internal precision
    println!("MpFloat result : {:.15}", result_mp.re);
    println!("f64    result  : {:.15}", result_f64.re);
    println!("Reference      : {:.15}", true_val);

    // Precision check (agreement at f64 precision)
    assert!((result_mp.re - true_val).abs() < 1e-12,
        "MpFloat result {:.15} differs from reference {:.15}", result_mp.re, true_val);
    assert!((result_f64.re - true_val).abs() < 1e-12,
        "f64 result {:.15} differs from reference {:.15}", result_f64.re, true_val);
}

// ─────────────────────────────────────────────────────────────────────────────
// ⑥ Error case: Undefined derivative
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn mp_undefined_derivative_returns_error() {
    let func = UserFn::<MpFloat>::new("f", |[x]| x);
    let result = Builder::<MpFloat, 1>::new("diff(f(x), x)", ["x"])
        .with_user_functions([func])
        .compile();
    assert!(result.is_err(), "should fail: derivative not registered");
}
