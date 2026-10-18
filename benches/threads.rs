//! benches/threads.rs
//!
//! Multi-thread benchmarks for the compiled closure (plan item R1).
//!
//! Goal: find out whether, per evaluation, the cost is dominated by
//!   (a) heap allocation (stack `Vec`, argument `Vec`s), or
//!   (b) `Complex<T>::clone()` of arguments / constants,
//! for `T = f64` and `T = MpFloat` (rug, 256 bit), with 1 / 4 / 8 threads.
//!
//! Groups
//! - `mt/parts/<type>/<part>/<threads>` : individual cost components
//!     alloc_stack_cap4 : `Vec::<Complex<T>>::with_capacity(4)` + drop
//!     clone_complex    : one `Complex<T>::clone()` + drop
//!     clone_args2      : `[Complex<T>; 2]::clone()` + drop
//! - `mt/eval/<type>/<formula>/<threads>` : the compiled closure itself
//!
//! How to read the numbers
//! - Each thread runs `iters` evaluations; the reported time is wall time / iters,
//!   i.e. the latency of ONE evaluation on ONE thread while `threads` threads run.
//!   A perfectly scalable workload is flat across thread counts.
//!   Scaling ratio = time(8 threads) / time(1 thread); > 1 means contention
//!   (allocator, memory bandwidth) or fewer physical cores than threads.
//! - Attribution: compare `eval/<type>/id` with `parts` (alloc_stack + clone_args2
//!   + clone_complex). For `add`, `poly`, `wave` the remainder is arithmetic plus
//!   the argument `Vec`s of builtin function calls.
//!
//! Run:
//!   cargo bench --bench threads
//!   cargo bench --bench threads -- mt/eval/mpfloat
//!   cargo bench --bench threads -- --save-baseline before
//!
//! Record `nproc`, toolchain and CPU governor next to the results.

use criterion::{BatchSize,BenchmarkId,Criterion,criterion_group,criterion_main};
use formulac::builder::{Builder, Scratch};
use formulac::core::{ComplexMath, Real};
use num_complex::Complex;
use std::hint::black_box;
use std::ops::{AddAssign, MulAssign};
use std::str::FromStr;
use std::sync::Barrier;
use std::time::{Duration, Instant};

// Reuse the MpFloat wrapper from the integration test (no duplication).
// Its #[test] functions are compiled out in a bench build.
#[path = "../tests/mpfloat_integration.rs"]
#[allow(dead_code, unused_imports)]
mod mpfloat;
use mpfloat::MpFloat;

const THREADS: [usize; 3] = [1, 4, 8];

/// (name, formula). All use the same two arguments `x`, `y`.
const FORMULAS: &[(&str, &str)] = &[
    ("id", "x"),
    ("add", "x + y"),
    ("poly", "a0 + a1*x + a2*x^2 + a3*x^3"),
    ("poly5", "a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4 + a5*x^5"),
    ("poly10", "a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4 + a5*x^5 + a6*x^6 + a7*x^7 + a8*x^8 + a9*x^9 + a10*x^10"),
    ("wave", "A*sin(w*x + phy) + B*cos(w*x + phy)"),
     // Constant-exponent powi path (used to decide whether PowiConst is worth it).
    ("pow2", "x^2"),
    ("pow3", "x^3"),
    ("pow10", "x^10"),
    ("sub", "x - y"),
    ("sub_mul", "x - y * x"),
    ("sub_chain", "x - y - x * y"),
];

// ─────────────────────────────────────────────────────────────────────────────
// Constructors per numeric type
// ─────────────────────────────────────────────────────────────────────────────

fn mk_f64(re: f64, im: f64) -> Complex<f64> {
    Complex::new(re, im)
}

fn mk_mp(re: f64, im: f64) -> Complex<MpFloat> {
    Complex::new(
        MpFloat::with_prec(MpFloat::DEFAULT_PREC, re),
        MpFloat::with_prec(MpFloat::DEFAULT_PREC, im),
    )
}

/// Extra coefficients used only by the polynomial formulas above.
const EXTRA_COEFFS: [&str; 7] = ["a4", "a5", "a6", "a7", "a8", "a9", "a10"];

fn consts<T: Real>(mk: fn(f64, f64) -> Complex<T>) -> Vec<(&'static str, Complex<T>)> {
    let mut v = vec![
        ("a0", mk(1.0, 2.0)),
        ("a1", mk(-2.0, 3.5)),
        ("a2", mk(5.25, -0.22)),
        ("a3", mk(-0.03, 4.03)),
        ("A", mk(5.25, -0.22)),
        ("B", mk(-0.03, 4.03)),
        ("w", mk(0.25, 0.333)),
        ("phy", mk(-2.0, 3.5)),
    ];
    for (i, name) in EXTRA_COEFFS.iter().enumerate() {
        let k = (i + 4) as f64;
        v.push((*name, mk(1.0 + 0.3 * k, 0.5 - 0.1 * k)));
    }

    v
}

// ─────────────────────────────────────────────────────────────────────────────
// Thread harness
// ─────────────────────────────────────────────────────────────────────────────

/// Runs `run` on `threads` threads, `iters` iterations each, and returns the wall
/// time between "all threads ready" and "all threads done".
///
/// `setup` runs inside each thread BEFORE timing starts (so per-thread inputs are
/// allocated by that thread, and thread spawn cost is excluded).
fn run_threads<S, Setup, Run>(threads: usize, iters: u64, setup: Setup, run: Run) -> Duration
where
    Setup: Fn() -> S + Sync,
    Run: Fn(&mut S, u64) + Sync,
{
    let start = Barrier::new(threads + 1);
    let end = Barrier::new(threads + 1);
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                let mut state = setup();
                start.wait();
                run(&mut state, iters);
                end.wait();
            });
        }
        start.wait();
        let t0 = Instant::now();
        end.wait();
        t0.elapsed()
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// mt/parts : cost components
// ─────────────────────────────────────────────────────────────────────────────

fn bench_parts<T>(c: &mut Criterion, tname: &str, mk: fn(f64, f64) -> Complex<T>)
where
    T: Real + Send + Sync + 'static,
{
    let mut g = c.benchmark_group(format!("mt/parts/{tname}"));
    g.sample_size(20).measurement_time(Duration::from_secs(3));

    for &threads in &THREADS {
        g.bench_with_input(
            BenchmarkId::new("alloc_stack_cap4", threads),
            &threads,
            |b, &threads| {
                b.iter_custom(|iters| {
                    run_threads(
                        threads,
                        iters,
                        || (),
                        |_: &mut (), n: u64| {
                            for _ in 0..n {
                                let v: Vec<Complex<T>> = Vec::with_capacity(black_box(4));
                                drop(black_box(v));
                            }
                        },
                    )
                })
            },
        );

        g.bench_with_input(
            BenchmarkId::new("clone_complex", threads),
            &threads,
            |b, &threads| {
                b.iter_custom(|iters| {
                    run_threads(
                        threads,
                        iters,
                        || mk(0.7, 0.1),
                        |x: &mut Complex<T>, n: u64| {
                            for _ in 0..n {
                                drop(black_box(black_box(&*x).clone()));
                            }
                        },
                    )
                })
            },
        );

        g.bench_with_input(
            BenchmarkId::new("clone_args2", threads),
            &threads,
            |b, &threads| {
                b.iter_custom(|iters| {
                    run_threads(
                        threads,
                        iters,
                        || [mk(0.7, 0.1), mk(1.3, -0.4)],
                        |args: &mut [Complex<T>; 2], n: u64| {
                            for _ in 0..n {
                                drop(black_box(black_box(&*args).clone()));
                            }
                        },
                    )
                })
            },
        );
    }
    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// part/powi_* : cost components of the constant exponent
//   part/powi_exponent_clone : clone + drop of the exponent `Complex<T>`
//   part/powi_to_i32         : `y.re.to_i32()` only
//   part/powi_exponent_path  : clone + to_i32 + drop, i.e. the exponent-related
//                              cost of the generic `Function(Powi)` path
// ─────────────────────────────────────────────────────────────────────────────

/// Runs a single-threaded micro benchmark on a constant exponent `3 + 0i`.
///
/// `body` receives the exponent through `black_box`, so the optimizer cannot
/// fold the work away. Each (`group`, `tname`) pair becomes one benchmark ID.
fn powi_part<T>(
    c: &mut Criterion,
    group: &str,
    tname: &str,
    mk: fn(f64, f64) -> Complex<T>,
    body: impl Fn(&Complex<T>),
) where
    T: Real + Send + Sync + 'static,
{
    let exponent = mk(3.0, 0.0);
    let mut g = c.benchmark_group(group);
    g.bench_function(tname, |b| b.iter(|| body(black_box(&exponent))));
    g.finish();
}

/// Registers the `part/powi_*` benchmarks for `f64` and `MpFloat`.
fn bench_powi_parts(c: &mut Criterion) {
    // Clone + drop of the exponent.
    powi_part::<f64>(c, "part/powi_exponent_clone", "f64", mk_f64, |e| {
        let _ = black_box(e.clone());
    });
    powi_part::<MpFloat>(c, "part/powi_exponent_clone", "mpfloat", mk_mp, |e| {
        drop(black_box(e.clone()));
    });

    // `to_i32` only.
    powi_part::<f64>(c, "part/powi_to_i32", "f64", mk_f64, |e| {
        black_box(Real::to_i32(&e.re));
    });
    powi_part::<MpFloat>(c, "part/powi_to_i32", "mpfloat", mk_mp, |e| {
        black_box(Real::to_i32(&e.re));
    });

    // Clone + to_i32 + drop (exponent cost of the old path).
    powi_part::<f64>(c, "part/powi_exponent_path", "f64", mk_f64, |e| {
        let y = e.clone();
        let n = Real::to_i32(&y.re);
        let _ = black_box(y);
        black_box(n);
    });
    powi_part::<MpFloat>(c, "part/powi_exponent_path", "mpfloat", mk_mp, |e| {
        let y = e.clone();
        let n = Real::to_i32(&y.re);
        drop(black_box(y));
        black_box(n);
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// part/complex_powi : cost of `ComplexMath::powi` alone (no VM involved)
//   part/complex_powi/<type>/n=<exponent>
// ─────────────────────────────────────────────────────────────────────────────

/// Benchmarks `ComplexMath::powi` for several exponents.
///
/// The base is cloned in the setup closure, so the clone is not part of the
/// measured time. Only the multiplications / inversion (and dropping the result) are.
fn bench_complex_powi_for<T>(c: &mut Criterion, tname: &str, mk: fn(f64, f64) -> Complex<T>)
where
    T: Real + Send + Sync + 'static,
{
    let base = mk(0.7, 0.1);
    let mut g = c.benchmark_group(format!("part/complex_powi/{tname}"));
    for n in [2, 3, 10, -1, -3] {
        g.bench_function(format!("n={n}"), |b| {
            b.iter_batched(
                || black_box(&base).clone(),
                |x| ComplexMath::powi(x, black_box(n)),
                BatchSize::SmallInput,
            )
        });
    }
    g.finish();
}

/// Registers `part/complex_powi/*` for `f64` and `MpFloat`.
fn bench_complex_powi(c: &mut Criterion) {
    bench_complex_powi_for::<f64>(c, "f64", mk_f64);
    bench_complex_powi_for::<MpFloat>(c, "mpfloat", mk_mp);
}

// ─────────────────────────────────────────────────────────────────────────────
// mt/eval : compiled closure
// ─────────────────────────────────────────────────────────────────────────────

fn bench_eval<T>(c: &mut Criterion, tname: &str, mk: fn(f64, f64) -> Complex<T>)
where
    T: Real + FromStr + Send + Sync + 'static,
    Complex<T>: AddAssign + MulAssign,
{
    let mut g = c.benchmark_group(format!("mt/eval/{tname}"));
    g.sample_size(20).measurement_time(Duration::from_secs(3));

    for (name, formula) in FORMULAS {
        // compile once, outside the measured loop; the closure is shared by reference
        let expr = Builder::<T, 2>::new(formula, ["x", "y"])
            .with_constants(consts(mk))
            .compile()
            .expect("compile failed");

        for &threads in &THREADS {
            g.bench_with_input(
                BenchmarkId::new(*name, threads),
                &threads,
                |b, &threads| {
                    b.iter_custom(|iters| {
                        run_threads(
                            threads,
                            iters,
                            || ([mk(0.7, 0.1), mk(1.3, -0.4)], expr.new_scratch()),
                            |st: &mut ([Complex<T>; 2], Scratch<T>), n: u64| {
                                for _ in 0..n {
                                    drop(black_box(expr.eval_with_scratch(
                                        black_box(&st.0),
                                        &mut st.1,
                                    )));
                                }
                            },
                        )
                    })
                },
            );
        }
    }
    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// part/complex_{add,sub,mul,div} : cost of one complex operation (no VM involved)
//   part/complex_<op>/<type>
// ─────────────────────────────────────────────────────────────────────────────

/// Benchmarks a single complex operation on two operands.
///
/// The operands are cloned in the setup closure, so the clone is not part of the
/// measured time. Only the operation (and dropping the result) is.
fn bench_complex_ops_for<T>(c: &mut Criterion, tname: &str, mk: fn(f64, f64) -> Complex<T>)
where
    T: Real + Send + Sync + 'static,
{
    type Op<T> = fn(Complex<T>, Complex<T>) -> Complex<T>;
    let ops: [(&str, Op<T>); 4] = [
        ("add", |a, b| a + b),
        ("sub", |a, b| a - b),
        ("mul", |a, b| a * b),
        ("div", |a, b| a / b),
    ];
    let (x, y) = (mk(0.7, 0.1), mk(1.3, -0.4));

    for (name, op) in ops {
        let mut g = c.benchmark_group(format!("part/complex_{name}"));
        g.bench_function(tname, |b| {
            b.iter_batched(
                || (black_box(&x).clone(), black_box(&y).clone()),
                |(l, r)| op(l, r),
                BatchSize::SmallInput,
            )
        });
        g.finish();
    }
}

/// Registers `part/complex_{add,sub,mul,div}/*` for `f64` and `MpFloat`.
fn bench_complex_ops(c: &mut Criterion) {
    bench_complex_ops_for::<f64>(c, "f64", mk_f64);
    bench_complex_ops_for::<MpFloat>(c, "mpfloat", mk_mp);
}

// ─────────────────────────────────────────────────────────────────────────────
// part/complex_sin_cos : upper bound of sharing `sin(u)` and `cos(u)` (N10)
//   part/complex_sin_cos/<type>/separate : ComplexMath::sin(z) + ComplexMath::cos(z)
//   part/complex_sin_cos/<type>/shared   : sin_cos(a), cosh(b), sinh(b) called once each
// ─────────────────────────────────────────────────────────────────────────────

/// Computes `(sin(z), cos(z))` with one `sin_cos`, one `cosh` and one `sinh` call.
///
/// sin(a + bi) = sin(a) cosh(b) + i cos(a) sinh(b)
/// cos(a + bi) = cos(a) cosh(b) - i sin(a) sinh(b)
fn sin_cos_shared<T: Real>(z: Complex<T>) -> (Complex<T>, Complex<T>) {
    let (a, b) = (z.re, z.im);
    let (sin_a, cos_a) = a.sin_cos();
    let cosh_b = b.clone().cosh();
    let sinh_b = b.sinh();
    let sin = Complex::new(sin_a.clone() * cosh_b.clone(), cos_a.clone() * sinh_b.clone());
    let cos = Complex::new(cos_a * cosh_b, -(sin_a * sinh_b));
    (sin, cos)
}

/// Benchmarks the two ways of getting both `sin(z)` and `cos(z)`.
///
/// Both variants receive two clones of `z` from the setup closure, so the
/// measured time contains the same number of drops in either case.
fn bench_complex_sin_cos_for<T>(c: &mut Criterion, tname: &str, mk: fn(f64, f64) -> Complex<T>)
where
    T: Real + Send + Sync + 'static,
{
    let z = mk(0.7, 0.1);
    let mut g = c.benchmark_group(format!("part/complex_sin_cos/{tname}"));
    g.bench_function("separate", |b| {
        b.iter_batched(
            || (black_box(&z).clone(), black_box(&z).clone()),
            |(z1, z2)| (ComplexMath::sin(z1), ComplexMath::cos(z2)),
            BatchSize::SmallInput,
        )
    });
    g.bench_function("shared", |b| {
        b.iter_batched(
            || (black_box(&z).clone(), black_box(&z).clone()),
            |(z1, z2)| {
                drop(z2); // keep the number of drops equal to `separate`
                sin_cos_shared(z1)
            },
            BatchSize::SmallInput,
        )
    });
    g.finish();
}

/// Registers `part/complex_sin_cos/*` and checks that both variants agree (f64).
fn bench_complex_sin_cos(c: &mut Criterion) {
    let z = mk_f64(0.7, 0.1);
    let (s, co) = sin_cos_shared(z);
    let (es, eco) = (ComplexMath::sin(z), ComplexMath::cos(z));
    assert!((s - es).norm() < 1e-12 && (co - eco).norm() < 1e-12, "sin_cos_shared disagrees");

    bench_complex_sin_cos_for::<f64>(c, "f64", mk_f64);
    bench_complex_sin_cos_for::<MpFloat>(c, "mpfloat", mk_mp);
}

// ─────────────────────────────────────────────────────────────────────────────
// part/constant_push : clone + push + pop of a constant (approximates `Instruction::Constant`)
// ─────────────────────────────────────────────────────────────────────────────

/// Benchmarks cloning a value onto a pre-allocated stack and popping it again.
///
/// `Instruction` is crate-private, so the dispatch itself is not included.
fn bench_constant_push_for<T>(c: &mut Criterion, tname: &str, mk: fn(f64, f64) -> Complex<T>)
where
    T: Real + Send + Sync + 'static,
{
    let value = mk(0.7, 0.1);
    let mut stack: Vec<Complex<T>> = Vec::with_capacity(4);
    let mut g = c.benchmark_group("part/constant_push");
    g.bench_function(tname, |b| {
        b.iter(|| {
            stack.push(black_box(&value).clone());
            drop(black_box(stack.pop()));
        })
    });
    g.finish();
}

/// Registers `part/constant_push/*` for `f64` and `MpFloat`.
fn bench_constant_push(c: &mut Criterion) {
    bench_constant_push_for::<f64>(c, "f64", mk_f64);
    bench_constant_push_for::<MpFloat>(c, "mpfloat", mk_mp);
}

// ─────────────────────────────────────────────────────────────────────────────
// Registration
// ─────────────────────────────────────────────────────────────────────────────

fn bench_all(c: &mut Criterion) {
    bench_parts::<f64>(c, "f64", mk_f64);
    bench_eval::<f64>(c, "f64", mk_f64);
    bench_parts::<MpFloat>(c, "mpfloat", mk_mp);
    bench_eval::<MpFloat>(c, "mpfloat", mk_mp);
    bench_powi_parts(c);
    bench_complex_powi(c);
    bench_complex_ops(c);
    bench_complex_sin_cos(c);
    bench_constant_push(c);
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
