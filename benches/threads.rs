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
//!                        (the harness passes args by value, so every eval pays this)
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

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use formulac::builder::{Builder, Scratch};
use formulac::core::Real;
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
    ("wave", "A*sin(w*x + phy) + B*cos(w*x + phy)"),
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

fn consts<T: Real>(mk: fn(f64, f64) -> Complex<T>) -> Vec<(&'static str, Complex<T>)> {
    vec![
        ("a0", mk(1.0, 2.0)),
        ("a1", mk(-2.0, 3.5)),
        ("a2", mk(5.25, -0.22)),
        ("a3", mk(-0.03, 4.03)),
        ("A", mk(5.25, -0.22)),
        ("B", mk(-0.03, 4.03)),
        ("w", mk(0.25, 0.333)),
        ("phy", mk(-2.0, 3.5)),
    ]
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
                                    // args are passed by value, so one array clone per call
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
// Registration
// ─────────────────────────────────────────────────────────────────────────────

fn bench_all(c: &mut Criterion) {
    bench_parts::<f64>(c, "f64", mk_f64);
    bench_eval::<f64>(c, "f64", mk_f64);
    bench_parts::<MpFloat>(c, "mpfloat", mk_mp);
    bench_eval::<MpFloat>(c, "mpfloat", mk_mp);
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
