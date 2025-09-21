//! benches.rs
use criterion::{criterion_group, criterion_main, Criterion};
use formulac::{
    compile,
    variable::{Variables, UserDefinedTable},
};
use num_complex::{Complex, ComplexFloat};
use paste::paste;

fn bench_analyze_liner(c: &mut Criterion) {
    let vars = Variables::new();
    let users = UserDefinedTable::new();

    let make_much_operand = |n: usize| (0..=n).map(|_| "x").collect::<Vec<_>>().join("+");
    for n in [1, 10, 100, 1000] {
        let formula = make_much_operand(n);
        c.bench_function(&format!("compile {} operands", n), |b| {
            b.iter(|| compile(&formula, &["x"], &vars, &users))
        });

        let expr = compile(&formula, &["x"], &vars, &users).unwrap();
        let x = Complex::ONE;
        c.bench_function(&format!("exec {} operands", n), |b| {
            b.iter(|| expr(&[x]))
        });
    }
}

fn bench_analyze_nested(c: &mut Criterion) {
    let vars = Variables::new();
    let users = UserDefinedTable::new();

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
            b.iter(|| compile(&formula, &["x"], &vars, &users))
        });

        let expr = compile(&formula, &["x"], &vars, &users).unwrap();
        let x = Complex::ONE;
        c.bench_function(&format!("exec {} nested", n), |b| {
            b.iter(|| expr(&[x]))
        });
    }
}

fn bench_analyze_literal(c: &mut Criterion) {
    let vars = Variables::new();
    let users = UserDefinedTable::new();

    let make_much_order = |n: usize| {
        let digits = "0123456789";
        digits.repeat((n + 9) / 10)[..n].to_string()
    };
    for n in [1, 10, 100, 1000] {
        let formula = make_much_order(n);
        c.bench_function(&format!("compile {} order literal", n), |b| {
            b.iter(|| compile(&formula, &["x"], &vars, &users))
        });

        let expr = compile(&formula, &["x"], &vars, &users).unwrap();
        let x = Complex::ONE;
        c.bench_function(&format!("exec {} order literal", n), |b| {
            b.iter(|| expr(&[x]))
        });
    }
}

fn bench_analyze_paren(c: &mut Criterion) {
    let mut vars = Variables::new();
    for ch in 'a'..='f' {
        vars.insert(&[(&ch.to_string(), Complex::ONE)]);
    }

    let users = UserDefinedTable::new();

    let formula = "(a+b)*(c-d)/(e+f)";
    c.bench_function(&format!("compile with paren '{}'", formula), |b| {
        b.iter(|| compile(&formula, &[], &vars, &users))
    });

    let expr = compile(&formula, &["x"], &vars, &users).unwrap();
    let x = Complex::ONE;
    c.bench_function(&format!("exec with paren '{}'", formula), |b| {
        b.iter(|| expr(&[x]))
    });

    let formula = "a+b*c-d/e+f";
    c.bench_function(&format!("compile without paren '{}'", formula), |b| {
        b.iter(|| compile(&formula, &[], &vars, &users))
    });

    let expr = compile(&formula, &["x"], &vars, &users).unwrap();
    let x = Complex::ONE;
    c.bench_function(&format!("exec with paren '{}'", formula), |b| {
        b.iter(|| expr(&[x]))
    });
}

fn bench_analyze_many_vars(c: &mut Criterion) {
    let mut vars = Variables::new();
    let users = UserDefinedTable::new();

    let var_names: Vec<String> = (1..=100).map(|i| format!("a{}", i)).collect();
    let var_refs: Vec<&str> = var_names.iter().map(|s| s.as_str()).collect();
    for name in &var_names {
        vars.insert(&[(name, Complex::ONE)]);
    }

    // a1 + a2 + ... + a100
    let formula = var_names.join(" + ");

    c.bench_function("compile many vars (100)", |b| {
        b.iter(|| compile(&formula, &var_refs, &vars, &users))
    });

    let expr = compile(&formula, &[], &vars, &users).unwrap();
    c.bench_function("exec many vars (100)", |b| {
        b.iter(|| expr(&[]))
    });
}

fn bench_analyze_invalid(c: &mut Criterion) {
    let vars = Variables::new();
    let users = UserDefinedTable::new();

    let invalid_formulas = [
        "unknown_func(x)",      // unknown function
        "1 + (2 * 3",           // forget ')'
        "x ** 2",               // unknown operand '**'
        "1 + @",                // unknown lexeme '@'
    ];

    for formula in &invalid_formulas {
        c.bench_function(&format!("compile invalid: {}", formula), |b| {
            b.iter(|| {
                let _ = compile(formula, &["x"], &vars, &users);
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
    bench_analyze_invalid,
);

fn bench_practical_polynomial(c: &mut Criterion) {
    let mut vars = Variables::new();
    vars.insert(&[
        ("a0", Complex::new(1.0, 2.0)),
        ("a1", Complex::new(-2.0, 3.5)),
        ("a2", Complex::new(5.25, -0.22)),
        ("a3", Complex::new(-0.03, 4.03)),
        ("a4", Complex::new(1.0, 0.0)),
    ]);

    let users = UserDefinedTable::new();

    let formula = "a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4";
    c.bench_function(&format!("compile polynomial '{}'", formula), |b| {
        b.iter(|| compile(&formula, &[], &vars, &users))
    });

    let expr = compile(&formula, &["x"], &vars, &users).unwrap();
    let x = Complex::new(0.05, 2.4);
    c.bench_function(&format!("exec polynomial '{}'", formula), |b| {
        b.iter(|| expr(&[x]))
    });
}

fn bench_practical_wafe_function(c: &mut Criterion) {
    let mut vars = Variables::new();
    vars.insert(&[
        ("w", Complex::new(0.25, 0.333)),
        ("phy", Complex::new(-2.0, 3.5)),
        ("A", Complex::new(5.25, -0.22)),
        ("B", Complex::new(-0.03, 4.03)),
    ]);

    let users = UserDefinedTable::new();

    let formula = "A*sin(w*t + phy) + B*cos(w*t + phy)";
    c.bench_function(&format!("compile wave function '{}'", formula), |b| {
        b.iter(|| compile(&formula, &[], &vars, &users))
    });

    let expr = compile(&formula, &["t"], &vars, &users).unwrap();
    let x = Complex::new(0.3, 0.0);
    c.bench_function(&format!("exec wave function '{}'", formula), |b| {
        b.iter(|| expr(&[x]))
    });
}

fn bench_practical_exponential_decay(c: &mut Criterion) {
    let mut vars = Variables::new();
    vars.insert(&[
        ("λ", Complex::new(0.25, 0.333)),
        ("A", Complex::new(3.28, -0.92)),
        ("B", Complex::new(-0.12, 8.03)),
    ]);

    let users = UserDefinedTable::new();

    let formula = "A*exp(-λ*t) + B";
    c.bench_function(&format!("compile exponential decay '{}'", formula), |b| {
        b.iter(|| compile(&formula, &[], &vars, &users))
    });

    let expr = compile(&formula, &["t"], &vars, &users).unwrap();
    let x = Complex::new(0.3, 0.0);
    c.bench_function(&format!("exec exponential decay '{}'", formula), |b| {
        b.iter(|| expr(&[x]))
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

                    let vars = Variables::new();
                    let users = UserDefinedTable::new();
                    let expr = compile(concat!(stringify!($variant), "(x)"), &["x"], &vars, &users).unwrap();
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

    let vars = Variables::new();
    let users = UserDefinedTable::new();
    let expr = compile("pow(x, y)", &["x", "y"], &vars, &users).unwrap();
    c.bench_function("parsed \"pow(x, y)\"", |b| {
        b.iter(|| expr(&[x, y]))
    });
}

pub fn bench_compares_powi(c: &mut Criterion) {
    let x = Complex::new(1.0, 0.5);
    let y = Complex::new(2.0, -0.5);

    c.bench_function("direct x.powi(y)", |b| {
        b.iter(|| x.powi(y.re() as i32))
    });

    let vars = Variables::new();
    let users = UserDefinedTable::new();
    let expr = compile("powi(x, y)", &["x", "y"], &vars, &users).unwrap();
    c.bench_function("parsed \"powi(x, y)\"", |b| {
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
