//! benches.rs
//!
//! Benchmarks for formulac, split into two kinds of groups:
//!
//! - `compile/*`  : cost of `Builder::compile()` (Builder construction is excluded)
//! - `eval/*`     : cost of calling the compiled closure (compile is excluded)
//! - `builder/*`  : cost of `Builder::new` + `with_constants` only
//!
//! Notes:
//! - `simplify()` folds constants and merges like terms, so every `eval/*` formula
//!   below is written so that it does NOT collapse into a constant at compile time
//!   (constants -> function arguments, distinct sub-expressions instead of `x+x+...`).
//! - Every input goes through `black_box` so the optimizer cannot fold it away.
//!
//! Baseline workflow:
//!   cargo bench --bench benches -- --save-baseline before
//!   cargo bench --bench benches -- --baseline before

use criterion::{
    criterion_group,
    criterion_main,
    BenchmarkId,
    Criterion,
};
use formulac::builder::Builder;
use formulac::UserFn;
use num_complex::{Complex, ComplexFloat};

use std::hint::black_box;

type C = Complex<f64>;

// ─────────────────────────────────────────────────────────────────────────────
// Formula generators
// ─────────────────────────────────────────────────────────────────────────────

/// `sin(sin(...sin(x)...))` — never folds.
fn nested_sin(n: usize) -> String {
    let mut f = "x".to_string();
    for _ in 0..n {
        f = format!("sin({f})");
    }
    f
}

/// `x+x+...+x` — collapses into `n*x` in simplify (use for compile/simplify cost only).
fn like_terms(n: usize) -> String {
    vec!["x"; n].join("+")
}

/// `sin(x+1.5)+sin(x+2.5)+...` — structurally distinct terms, never merged.
fn distinct_terms(n: usize) -> String {
    (1..=n)
        .map(|k| format!("sin(x+{k}.5)"))
        .collect::<Vec<_>>()
        .join("+")
}

/// `sin(x+1.5)^2*sin(x+2.5)^2*...` — distinct bases (compile only: the product underflows).
fn distinct_pows(n: usize) -> String {
    (1..=n)
        .map(|k| format!("sin(x+{k}.5)^2"))
        .collect::<Vec<_>>()
        .join("*")
}

/// Numeric literal with `n` digits. f64 overflows beyond ~308 digits.
fn long_literal(n: usize) -> String {
    "1234567890".repeat(n / 10 + 1)[..n].to_string()
}

/// `x0 + x1 + ... + x(n-1)`
fn sum_of_args(n: usize) -> String {
    (0..n).map(|i| format!("x{i}")).collect::<Vec<_>>().join(" + ")
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared case tables
// ─────────────────────────────────────────────────────────────────────────────

/// Constants shared by the practical formulas.
fn practical_consts() -> Vec<(&'static str, C)> {
    vec![
        ("a0", Complex::new(1.0, 2.0)),
        ("a1", Complex::new(-2.0, 3.5)),
        ("a2", Complex::new(5.25, -0.22)),
        ("a3", Complex::new(-0.03, 4.03)),
        ("a4", Complex::new(1.0, 0.0)),
        ("w", Complex::new(0.25, 0.333)),
        ("phy", Complex::new(-2.0, 3.5)),
        ("A", Complex::new(5.25, -0.22)),
        ("B", Complex::new(-0.03, 4.03)),
        ("lam", Complex::new(0.25, 0.333)),
    ]
}

/// (name, formula) with one argument `x`.
const SIMPLE_CASES: &[(&str, &str)] = &[
    ("simple", "x + 1"),
    ("arithmetic", "x*x + 2*x + 1"),
    ("functions", "sin(x) + cos(x) + exp(x)"),
    ("nested_ops", "((x + 1) * (x - 1)) / (x * x + 2)"),
];

const PRACTICAL_CASES: &[(&str, &str)] = &[
    ("polynomial", "a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4"),
    ("wave", "A*sin(w*x + phy) + B*cos(w*x + phy)"),
    ("exp_decay", "A*exp(-lam*x) + B"),
];

const DIFF_CASES: &[(&str, &str)] = &[
    ("x^2", "diff(x^2, x)"),
    ("sin(x)", "diff(sin(x), x)"),
    ("exp(x^2+3*x+1)", "diff(exp(x^2+3*x+1), x)"),
    ("sin(cos(x))", "diff(sin(cos(x)), x)"),
    ("x^10+x^5+x^2", "diff(x^10 + x^5 + x^2, x)"),
];

const INVALID_CASES: &[(&str, &str)] = &[
    ("unknown_func", "unknown_func(x)"),
    ("missing_rparen", "1 + (2 * 3"),
    ("double_star", "x ** 2"),
    ("unknown_lexeme", "1 + @"),
];

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// compile/<group>/<name> for one-argument formulas. Builder construction is outside the loop.
fn compile_cases(c: &mut Criterion, group: &str, cases: &[(&str, &str)], consts: &[(&str, C)]) {
    let mut g = c.benchmark_group(group);
    for (name, formula) in cases {
        let builder = Builder::<f64, 1>::new(formula, ["x"]).with_constants(consts.iter().cloned());
        g.bench_function(*name, |b| b.iter(|| black_box(builder.compile())));
    }
    g.finish();
}

/// eval/<group>/<name> for one-argument formulas. compile() is outside the loop.
fn eval_cases(c: &mut Criterion, group: &str, cases: &[(&str, &str)], consts: &[(&str, C)]) {
    let x = Complex::new(0.7, 0.1);
    let mut g = c.benchmark_group(group);
    for (name, formula) in cases {
        let expr = Builder::<f64, 1>::new(formula, ["x"])
            .with_constants(consts.iter().cloned())
            .compile()
            .unwrap();
        g.bench_function(*name, |b| b.iter(|| expr.eval(black_box([x]))));
    }
    g.finish();
}

/// compile/<group>/<n> for generated formulas of growing size.
fn compile_scaling(c: &mut Criterion, group: &str, sizes: &[usize], make: impl Fn(usize) -> String) {
    let mut g = c.benchmark_group(group);
    if sizes.iter().any(|&n| n >= 300) {
        g.sample_size(10);
    }
    for &n in sizes {
        let builder = Builder::<f64, 1>::new(&make(n), ["x"]);
        g.bench_function(BenchmarkId::from_parameter(n), |b| {
            b.iter(|| black_box(builder.compile()))
        });
    }
    g.finish();
}

/// eval/<group>/<n> for generated formulas of growing size.
fn eval_scaling(c: &mut Criterion, group: &str, sizes: &[usize], make: impl Fn(usize) -> String) {
    let x = Complex::new(0.7, 0.1);
    let mut g = c.benchmark_group(group);
    for &n in sizes {
        let expr = Builder::<f64, 1>::new(&make(n), ["x"]).compile().unwrap();
        g.bench_function(BenchmarkId::from_parameter(n), |b| {
            b.iter(|| expr.eval(black_box([x])))
        });
    }
    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// compile/*
// ─────────────────────────────────────────────────────────────────────────────

fn bench_compile_basic(c: &mut Criterion) {
    compile_cases(c, "compile/basic", SIMPLE_CASES, &[]);
    compile_cases(c, "compile/practical", PRACTICAL_CASES, &practical_consts());
    compile_cases(c, "compile/diff", DIFF_CASES, &[]);
}

fn bench_compile_scaling(c: &mut Criterion) {
    compile_scaling(c, "compile/deep_nesting", &[1, 10, 100, 1000], nested_sin);
    // Collapses to n*x: measures the like-term merge path in simplify() (S1).
    compile_scaling(c, "compile/like_terms", &[1, 10, 100, 1000], like_terms);
    // Distinct terms: measures the O(n^2) search in combine_like_add_terms (S1).
    compile_scaling(c, "compile/distinct_terms", &[10, 100, 1000], distinct_terms);
    // Distinct bases: measures combine_like_pow_terms (S1).
    compile_scaling(c, "compile/distinct_pows", &[10, 100, 300], distinct_pows);
    // f64 overflows around 308 digits, so stop at 300.
    compile_scaling(c, "compile/long_literal", &[1, 10, 100, 300], long_literal);
}

fn bench_compile_many_names(c: &mut Criterion) {
    // 100 constants (all folded away at compile time)
    let const_names: Vec<String> = (0..100).map(|i| format!("c{i}")).collect();
    let consts: Vec<(String, C)> = const_names.iter().map(|n| (n.clone(), Complex::new(1.0, 0.0))).collect();
    let formula = const_names.join(" + ");
    let builder = Builder::<f64, 0>::new(&formula, []).with_constants(consts.clone());
    c.bench_function("compile/constants_many/100", |b| b.iter(|| black_box(builder.compile())));

    // 100 arguments (linear lookup in Token::try_from, P1)
    let names: [String; 100] = std::array::from_fn(|i| format!("x{i}"));
    let refs: [&str; 100] = std::array::from_fn(|i| names[i].as_str());
    let builder = Builder::<f64, 100>::new(&sum_of_args(100), refs);
    c.bench_function("compile/args_many/100", |b| b.iter(|| black_box(builder.compile())));

    // Builder construction alone (this is what the old "compile with paren" benches measured)
    let formula = "(a+b)*(c-d)/(e+f)";
    let small: Vec<(&str, C)> = ["a", "b", "c", "d", "e", "f"].iter().map(|k| (*k, Complex::new(1.0, 0.0))).collect();
    c.bench_function("builder/new_with_constants/6", |b| {
        b.iter(|| black_box(Builder::<f64, 0>::new(formula, []).with_constants(small.clone())))
    });
    c.bench_function("builder/new_with_constants/100", |b| {
        b.iter(|| black_box(Builder::<f64, 0>::new(&formula, []).with_constants(consts.clone())))
    });
}

fn bench_compile_derivative_apis(c: &mut Criterion) {
    let builder = Builder::<f64, 2>::new("sin(x)*y + exp(x*y)", ["x", "y"]);
    let mut g = c.benchmark_group("compile/derivative_api");
    g.bench_function("compile_only", |b| b.iter(|| black_box(builder.compile())));
    g.bench_function("compile_with_derivative", |b| {
        b.iter(|| black_box(builder.compile_with_derivative("x")))
    });
    g.bench_function("compile_with_all_partials", |b| {
        b.iter(|| black_box(builder.compile_with_all_partials()))
    });
    g.finish();
}

fn bench_compile_invalid(c: &mut Criterion) {
    compile_cases(c, "compile/invalid", INVALID_CASES, &[]);
}

// ─────────────────────────────────────────────────────────────────────────────
// eval/*
// ─────────────────────────────────────────────────────────────────────────────

fn bench_eval_basic(c: &mut Criterion) {
    // Fixed overhead of the closure (stack allocation etc.): a single constant.
    let constant = Builder::<f64, 0>::new("42", []).compile().unwrap();
    c.bench_function("eval/constant", |b| b.iter(|| constant.eval(black_box([]))));

    eval_cases(c, "eval/basic", SIMPLE_CASES, &[]);
    eval_cases(c, "eval/practical", PRACTICAL_CASES, &practical_consts());
    eval_cases(c, "eval/diff", DIFF_CASES, &[]);
}

fn bench_eval_scaling(c: &mut Criterion) {
    eval_scaling(c, "eval/deep_nesting", &[1, 10, 100, 1000], nested_sin);
    eval_scaling(c, "eval/distinct_terms", &[1, 10, 100, 1000], distinct_terms);
}

fn bench_eval_args(c: &mut Criterion) {
    // 6 arguments instead of 6 constants, so nothing folds away.
    let args: [C; 6] = std::array::from_fn(|i| Complex::new(1.0 + i as f64 * 0.1, 0.2));
    let names = ["x0", "x1", "x2", "x3", "x4", "x5"];
    let mut g = c.benchmark_group("eval/args6");
    for (name, formula) in [
        ("with_paren", "(x0+x1)*(x2-x3)/(x4+x5)"),
        ("without_paren", "x0+x1*x2-x3/x4+x5"),
    ] {
        let expr = Builder::<f64, 6>::new(formula, names).compile().unwrap();
        g.bench_function(name, |b| b.iter(|| expr.eval(black_box(args))));
    }
    g.finish();

    // 100 arguments
    let names: [String; 100] = std::array::from_fn(|i| format!("x{i}"));
    let refs: [&str; 100] = std::array::from_fn(|i| names[i].as_str());
    let args: [C; 100] = std::array::from_fn(|i| Complex::new(1.0 + i as f64 * 0.01, 0.1));
    let expr = Builder::<f64, 100>::new(&sum_of_args(100), refs).compile().unwrap();
    c.bench_function("eval/args_many/100", |b| b.iter(|| expr.eval(black_box(args))));
}

fn bench_eval_user_function(c: &mut Criterion) {
    let x = Complex::new(0.7, 0.1);
    let mut g = c.benchmark_group("eval/user_function");

    let f = UserFn::<f64>::new("f", |[x]| x * x + Complex::new(1.0, 0.0));
    let expr = Builder::<f64, 1>::new("f(x)", ["x"]).with_user_functions([f.clone()]).compile().unwrap();
    g.bench_function("unary_x1", |b| b.iter(|| expr.eval(black_box([x]))));

    let expr = Builder::<f64, 1>::new("f(x) + f(x+1) + f(x+2)", ["x"])
        .with_user_functions([f])
        .compile()
        .unwrap();
    g.bench_function("unary_x3", |b| b.iter(|| expr.eval(black_box([x]))));

    let h = UserFn::<f64>::new("h", |[a, b]| a * b);
    let expr = Builder::<f64, 1>::new("h(x, x+1)", ["x"]).with_user_functions([h]).compile().unwrap();
    g.bench_function("binary", |b| b.iter(|| expr.eval(black_box([x]))));

    g.finish();
}

/// For each unary builtin, compare:
///   direct_num_complex : num_complex's own implementation
///   direct_formulac    : formulac::core::ComplexMath (same algorithm as the VM calls)
///   parsed             : compiled closure
/// `parsed - direct_formulac` is the VM overhead (what R1-R3 target).
macro_rules! bench_builtin_unary {
    ($c:expr, $($f:ident),* $(,)?) => {{
        let x = Complex::new(1.0_f64, 0.5);
        $(
            {
                let mut g = $c.benchmark_group(concat!("eval/builtin/", stringify!($f)));
                g.bench_function("direct_num_complex", |b| b.iter(|| black_box(x).$f()));
                g.bench_function("direct_formulac", |b| {
                    b.iter(|| formulac::core::ComplexMath::$f(black_box(x)))
                });
                let expr = Builder::<f64, 1>::new(concat!(stringify!($f), "(x)"), ["x"])
                    .compile()
                    .unwrap();
                g.bench_function("parsed", |b| b.iter(|| expr.eval(black_box([x]))));
                g.finish();
            }
        )*
    }};
}

fn bench_eval_builtin_unary(c: &mut Criterion) {
    bench_builtin_unary!(
        c,
        sin, cos, tan,
        asin, acos, atan,
        sinh, cosh, tanh,
        asinh, acosh, atanh,
        exp, ln, log10,
        sqrt, abs, conj,
    );
}

fn bench_eval_builtin_pow(c: &mut Criterion) {
    let x = Complex::new(1.0_f64, 0.5);
    let y = Complex::new(2.0_f64, -0.5);

    let mut g = c.benchmark_group("eval/builtin/pow");
    g.bench_function("direct_num_complex", |b| b.iter(|| black_box(x).powc(black_box(y))));
    g.bench_function("direct_formulac", |b| {
        b.iter(|| formulac::core::ComplexMath::powc(black_box(x), black_box(y)))
    });
    let expr = Builder::<f64, 2>::new("pow(x, y)", ["x", "y"]).compile().unwrap();
    g.bench_function("parsed", |b| b.iter(|| expr.eval(black_box([x, y]))));
    g.finish();

    let n = 3_i32;
    let yi = Complex::new(3.0_f64, 0.0);
    let mut g = c.benchmark_group("eval/builtin/powi");
    g.bench_function("direct_num_complex", |b| b.iter(|| black_box(x).powi(black_box(n))));
    g.bench_function("direct_formulac", |b| {
        b.iter(|| formulac::core::ComplexMath::powi(black_box(x), black_box(n)))
    });
    let expr = Builder::<f64, 2>::new("powi(x, y)", ["x", "y"]).compile().unwrap();
    g.bench_function("parsed", |b| b.iter(|| expr.eval(black_box([x, yi]))));
    g.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Registration
// ─────────────────────────────────────────────────────────────────────────────

criterion_group!(
    compile_benches,
    bench_compile_basic,
    bench_compile_scaling,
    bench_compile_many_names,
    bench_compile_derivative_apis,
    bench_compile_invalid,
);

criterion_group!(
    eval_benches,
    bench_eval_basic,
    bench_eval_scaling,
    bench_eval_args,
    bench_eval_user_function,
    bench_eval_builtin_unary,
    bench_eval_builtin_pow,
);

criterion_main!(compile_benches, eval_benches);
