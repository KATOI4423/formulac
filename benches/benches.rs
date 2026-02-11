//! benches.rs
use criterion::{criterion_group, criterion_main, Criterion};
use formulac::builder::Builder;
use num_complex::{Complex, ComplexFloat};
use paste::paste;

fn bench_analyze_liner(c: &mut Criterion) {
    let make_much_operand = |n: usize| (0..=n).map(|_| "x").collect::<Vec<_>>().join("+");
    for n in [1, 10, 100, 1000] {
        let formula = make_much_operand(n);
        c.bench_function(&format!("compile {} operands", n), |b| {
            b.iter(|| { let _ = Builder::new(&formula, &["x"]).compile(); } )
        });

        let expr = Builder::new(&formula, &["x"]).compile().unwrap();
        let x = Complex::ONE;
        c.bench_function(&format!("exec {} operands", n), |b| {
            b.iter(|| expr(&[x]))
        });
    }
}

fn bench_analyze_nested(c: &mut Criterion) {
    let make_much_nested = |n: usize| {
        let mut formula = "x".to_string();
        for _ in 0..n {
            formula = format!("sin({})", formula);
        }
        formula
    };
    for n in [1, 10, 100, 1000] {
        let formula = make_much_nested(n);
        c.bench_function(&format!("compile {} nested", n), |b| {
            b.iter(|| { let _ = Builder::new(&formula, &["x"]).compile(); })
        });

        let expr = Builder::new(&formula, &["x"]).compile().unwrap();
        let x = Complex::ONE;
        c.bench_function(&format!("exec {} nested", n), |b| {
            b.iter(|| expr(&[x]))
        });
    }
}

fn bench_analyze_literal(c: &mut Criterion) {
    let make_much_order = |n: usize| {
        let digits = "0123456789";
        digits.repeat((n + 9) / 10)[..n].to_string()
    };
    for n in [1, 10, 100, 1000] {
        let formula = make_much_order(n);
        c.bench_function(&format!("compile {} order literal", n), |b| {
            b.iter(|| { let _ =Builder::new(&formula, &["x"]).compile(); })
        });

        let expr = Builder::new(&formula, &["x"]).compile().unwrap();
        let x = Complex::ONE;
        c.bench_function(&format!("exec {} order literal", n), |b| {
            b.iter(|| expr(&[x]))
        });
    }
}

fn bench_analyze_paren(c: &mut Criterion) {
    let consts = ('a'..='f').into_iter()
        .map(|key| (key.to_string(), 1.0f64))
        .collect::<Vec<(String, f64)>>();

    let formula = "(a+b)*(c-d)/(e+f)";
    c.bench_function(&format!("compile with paren '{}'", formula), |b| {
        b.iter(|| Builder::new(&formula, &[]).with_constants(consts.clone()))
    });

    let expr = Builder::new(&formula, &["x"]).with_constants(consts.clone()).compile().unwrap();
    let x = Complex::ONE;
    c.bench_function(&format!("exec with paren '{}'", formula), |b| {
        b.iter(|| expr(&[x]))
    });

    let formula = "a+b*c-d/e+f";
    c.bench_function(&format!("compile without paren '{}'", formula), |b| {
        b.iter(|| Builder::new(&formula, &[]).with_constants(consts.clone()))
    });

    let expr = Builder::new(&formula, &["x"]).with_constants(consts.clone()).compile().unwrap();
    let x = Complex::ONE;
    c.bench_function(&format!("exec with paren '{}'", formula), |b| {
        b.iter(|| expr(&[x]))
    });
}

fn bench_analyze_many_vars(c: &mut Criterion) {
    let consts_names: Vec<String> = (1..=100).map(|i| format!("a{}", i)).collect();
    let consts_refs: Vec<&str> = consts_names.iter().map(|s| s.as_str()).collect();
    let consts: Vec<(&String, Complex<f64>)> = consts_names.iter().map(|name| (name, Complex::ONE)).collect();

    // a1 + a2 + ... + a100
    let formula = consts_names.join(" + ");

    c.bench_function("compile many vars (100)", |b| {
        b.iter(|| { let _ = Builder::new(&formula, &consts_refs).with_constants(consts.clone()).compile(); })
    });

    let expr = Builder::new(&formula, &consts_refs).with_constants(consts.clone()).compile().unwrap();
    c.bench_function("exec many vars (100)", |b| {
        b.iter(|| expr(&[]))
    });
}

fn bench_analyze_diff(c: &mut Criterion) {
    let formulas = [
        "diff(x^2, x)",
        "diff(sin(x), x)",
        "diff(exp(x^2+3*x+1), x)",
        "diff(sin(cos(x)), x)",
        "diff(x^10 + x^5 + x^2, x)",
    ];

    for formula in &formulas {
        c.bench_function(&format!("compile diff '{}'", formula), |b| {
            b.iter(|| { let _ = Builder::new(&formula, &["x"]).compile(); })
        });

        let expr = Builder::new(&formula, &["x"]).compile().unwrap();
        let x = Complex::new(0.7, 0.1);
        c.bench_function(&format!("exec diff '{}'", formula), |b| {
            b.iter(|| expr(&[x]))
        });
    }
}

fn bench_analyze_invalid(c: &mut Criterion) {
    let invalid_formulas = [
        "unknown_func(x)",      // unknown function
        "1 + (2 * 3",           // forget ')'
        "x ** 2",               // unknown operand '**'
        "1 + @",                // unknown lexeme '@'
    ];

    for formula in &invalid_formulas {
        c.bench_function(&format!("compile invalid: {}", formula), |b| {
            b.iter(|| {
                let _ = Builder::new(formula, &["x"]).compile();
            })
        });
    }
}

criterion_group!(bench_analyze,
    bench_analyze_liner,
    bench_analyze_nested,
    bench_analyze_literal,
    bench_analyze_paren,
    bench_analyze_many_vars,
    bench_analyze_diff,
    bench_analyze_invalid,
);

fn bench_practical_polynomial(c: &mut Criterion) {
    let consts = [
        ("a0", Complex::new(1.0, 2.0)),
        ("a1", Complex::new(-2.0, 3.5)),
        ("a2", Complex::new(5.25, -0.22)),
        ("a3", Complex::new(-0.03, 4.03)),
        ("a4", Complex::new(1.0, 0.0)),
    ];

    let formula = "a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4";
    c.bench_function(&format!("compile polynomial '{}'", formula), |b| {
        b.iter(|| { let _ = Builder::new(&formula, &["x"]).with_constants(consts.clone()).compile(); })
    });

    let expr = Builder::new(&formula, &["x"]).with_constants(consts.clone()).compile().unwrap();
    let x = Complex::new(0.05, 2.4);
    c.bench_function(&format!("exec polynomial '{}'", formula), |b| {
        b.iter(|| expr(&[x]))
    });
}

fn bench_practical_wafe_function(c: &mut Criterion) {
    let consts = [
        ("w", Complex::new(0.25, 0.333)),
        ("phy", Complex::new(-2.0, 3.5)),
        ("A", Complex::new(5.25, -0.22)),
        ("B", Complex::new(-0.03, 4.03)),
    ];

    let formula = "A*sin(w*t + phy) + B*cos(w*t + phy)";
    c.bench_function(&format!("compile wave function '{}'", formula), |b| {
        b.iter(|| { let _ = Builder::new(&formula, &["t"]).with_constants(consts.clone()).compile(); })
    });

    let expr = Builder::new(&formula, &["t"]).with_constants(consts.clone()).compile().unwrap();
    let t = Complex::new(0.3, 0.0);
    c.bench_function(&format!("exec wave function '{}'", formula), |b| {
        b.iter(|| expr(&[t]))
    });
}

fn bench_practical_exponential_decay(c: &mut Criterion) {
    let consts = [
        ("λ", Complex::new(0.25, 0.333)),
        ("A", Complex::new(3.28, -0.92)),
        ("B", Complex::new(-0.12, 8.03)),
    ];

    let formula = "A*exp(-λ*t) + B";
    c.bench_function(&format!("compile exponential decay '{}'", formula), |b| {
        b.iter(|| { let _ = Builder::new(&formula, &["t"]).with_constants(consts.clone()).compile(); })
    });

    let expr = Builder::new(&formula, &["t"]).with_constants(consts.clone()).compile().unwrap();
    let t = Complex::new(0.3, 0.0);
    c.bench_function(&format!("exec exponential decay '{}'", formula), |b| {
        b.iter(|| expr(&[t]))
    });
}

criterion_group!(bench_practical,
    bench_practical_polynomial,
    bench_practical_wafe_function,
    bench_practical_exponential_decay,
);

macro_rules! compares_one_arity_functions {
    ($( $variant: ident ),* $(,)? ) => {
        paste! {
            $(
                pub fn [<bench_compares_ $variant>](c: &mut Criterion) {
                    let x = Complex::new(1.0, 0.5);

                    c.bench_function(concat!("direct ", stringify!($variant), "(x)"), |b| {
                        b.iter(|| x.$variant())
                    });

                    let expr = Builder::new(concat!(stringify!($variant), "(x)"), &["x"]).compile().unwrap();
                    c.bench_function(concat!("parsed \"", stringify!($variant), "(x)\""), |b| {
                        b.iter(|| expr(&[x]))
                    });
                }
            )*
        }
    };
}

compares_one_arity_functions! {
    sin,    cos,    tan,
    asin,   acos,   atan,
    sinh,   cosh,   tanh,
    asinh,  acosh,  atanh,
    exp,    ln,     log10,
    sqrt,   abs,    conj,
}

pub fn bench_compares_pow(c: &mut Criterion) {
    let x = Complex::new(1.0, 0.5);
    let y = Complex::new(2.0, -0.5);

    c.bench_function("direct x.powc(y)", |b| {
        b.iter(|| x.powc(y))
    });

    let expr = Builder::new("pow(x, y)", &["x", "y"]).compile().unwrap();
    c.bench_function(r#"parsed "pow(x, y)""#, |b| {
        b.iter(|| expr(&[x, y]))
    });
}

pub fn bench_compares_powi(c: &mut Criterion) {
    let x = Complex::new(1.0, 0.5);
    let y = Complex::new(2.0, -0.5);

    c.bench_function("direct x.powi(y)", |b| {
        b.iter(|| x.powi(y.re() as i32))
    });

    let expr = Builder::new("powi(x, y)", &["x", "y"]).compile().unwrap();
    c.bench_function(r#"parsed "powi(x, y)""#, |b| {
        b.iter(|| expr(&[x, y]))
    });
}


criterion_group!(bench_compare,
    bench_compares_sin,     bench_compares_cos,     bench_compares_tan,
    bench_compares_asin,    bench_compares_acos,    bench_compares_atan,
    bench_compares_sinh,    bench_compares_cosh,    bench_compares_tanh,
    bench_compares_asinh,   bench_compares_acosh,   bench_compares_atanh,
    bench_compares_exp,     bench_compares_ln,      bench_compares_log10,
    bench_compares_sqrt,    bench_compares_abs,     bench_compares_conj,

    bench_compares_pow,     bench_compares_powi,
);

criterion_main!{
    bench_analyze,
    bench_practical,
    bench_compare,
}
