# formulac

`formulac` is a Rust library for parsing, evaluating, and differentiating mathematical expressions.
It supports complex numbers, user-defined functions, and higher-order derivatives.
Ideal for symbolic computation, mathematical simulations, and evaluating formulas in Rust applications.

## Features

- **Complex number support** — Evaluate expressions involving real and imaginary components using `num_complex::Complex<f64>`.
- **Reverse Polish Notation (RPN)** — Converts infix expressions to RPN using the Shunting-Yard algorithm for efficient evaluation.
- **Built-in mathematical operators & functions** — Supports `+`, `-`, `*`, `/`, `^`, and standard functions like `sin`, `cos`, `exp`, `log`, and more.
  See `src/astnode.rs` or [API Overview](#core-types--api-overview) for the list of available functions, constants, and operator symbols.
- **Unary & Binary operators** — Unary operators (`+`, `-`) and binary operators (`+`, `-`, `*`, `/`, `^`) are represented as `UnaryOperatorKind` and `BinaryOperatorKind`.
- **Abstract Syntax Tree (AST)** — Expressions are parsed into `AstNode` structures, enabling inspection, simplification, and compilation into executable closures.
- **User-defined functions** — Easily register custom functions or constants at runtime using a simple API (`UserDefinedTable`).
- **Differentiation support** — Parse and evaluate differential expressions using the `diff` operator (e.g., `diff(sin(x), x)`).
- **Safe and dependency-light** — No use of unsafe Rust or heavyweight external parsers.

---

## Usage

Add to your `Cargo.toml`:

```bash
cargo add formulac
```

### Basic Example

```rust
use num_complex::Complex;
use formulac::{Builder, Variables, UserDefinedTable};

fn main() {
    let vars = Variables::from(&[("a", Complex::new(3.0, 2.0))]);

    let formula = "sin(z) + a * cos(z)";
    let expr = Builder::new(formula, &["z"])
        .with_variables(vars)
        .compile()
        .expect("Failed to compile formula");

    let result = expr(&[Complex::new(1.0, 2.0)]); // evaluates 'sin(1+2i) + (3+2i) * cos(1+2i)'
    println!("Result = {}", result);
}
```

### Registering a Custom Function

You can use `UserDefinedTable::register` (check the function conflict) or `UserDefinedTable::insert` (overwrite the conflict function)

```rust
use num_complex::Complex;
use formulac::{Builder, Variables, UserDefinedTable, UserDefinedFunction};

fn main() {
    // Define a function f(x) = x^2 + 1
    let func = UserDefinedFunction::new(
        "f",
        |args: &[Complex<f64>]| args[0] * args[0] + Complex::new(1.0, 0.0),
        1,
    );

    // Create user-defined function table
    let users = UserDefinedTable::default()
        .register(func).unwrap();

    let builder = Builder::new("f(3)", &[])
        .with_user_defined_functions(users.clone()); // use it again later

    let expr = builder.compile()
        .expect("Failed to compile formula with UserDefinedFunction");

    assert_eq!(expr(&[]), Complex::new(10.0, 0.0));

    let users = users.insert(UserDefinedFunction::new(
        "f", // it conflicts the above function.
        |args: &[Complex<f64>] args[0] + Complex::new(2.0, 1.0),
        1,
    ));

    let expr = builder.with_user_defined_functions(users)
        .compile().unwrap();

    assert_eq!(expr(&[]), Complex::new(5.0, 1.0));
}
```

---

## Differentiation Support

`formulac` can represent **derivative expressions** in the AST.
Built-in functions (e.g. `sin`, `cos`, `exp`, `log`, …) already have derivative rules,
but **user-defined functions require the user to explicitly register their derivative form**.

**Note on Differential Order:**

- Only integer-order derivatives are supported; fractional derivatives are not allowed.
- The maximum allowed order is `i8::MAX` (127). Attempting to compute a derivative higher than this will result in a runtime error.

### Differentiation Example

You can directly write differentiation expressions using the `diff` operator:

```rust
use num_complex::Complex;
use formulac::{Builder, Variables, UserDefinedTable};

fn main() {
    // Differentiate sin(x) with respect to x
    let formula = "diff(sin(x), x)";
    let expr = Builder::new(formula, &["x"])
        .compile()
        .expect("Failed to compile formula");

    let result = expr(&[Complex::new(1.0, 0.0)]); // evaluates cos(1)
    println!("Result = {}", result);
}
```

When computing derivatives of order 2 or higher, specify the order:

```rust
use num_complex::Complex;
use formulac::{Builder, Variables, UserDefinedTable};

fn main() {
    // Differentiate sin(x) with respect to x
    let formula = "diff(sin(x), x, 2)";
    let expr = Builder::new(formula, &["x"])
        .compile()
        .expect("Failed to compile formula");

    let result = expr(&[Complex::new(1.0, 0.0)]); // evaluates to -sin(1)
    println!("Result = {}", result);
}
```

### Example: User-defined function with derivative

You can define your own functions and provide derivatives for them. The derivative must be registered in the same order as the function arguments.

```rust
use num_complex::Complex;
use formulac::{Builder, Variables, UserDefinedTable, UserDefinedFunction};

fn main() {
    // Define f(x) = x^2, derivative f'(x) = 2x
    let deriv = UserDefinedFunction::new(
        "df",
        |args: &[Complex<f64>] Complex::new(2.0, 0.0) * args[0],
        1,
    );
    let func = UserDefinedFunction::new(
        "f",
        |args: &[Complex<f64>]| args[0] * args[0],
        1,
    ).with_derivative(vec![deriv]);

    let users = UserDefinedTable::default()
        .register(func).unwrap();

    let expr = Builder::new("diff(f(x), x)", &["x"])
        .with_user_defined_functions(users)
        .compile()
        .expect("Failed to compile formula with UserDefinedFunction");

    let result = expr(&[Complex::new(3.0, 0.0)]); // evaluates f'(3) = 6
    println!("Result: {}", result);
}
```

### User-defined function without derivative

You can also define your own functions without providing derivatives for them. In the case, the derivatives will be calculated automatically by numerical differentiation.

The step size is determined automatically used by the argument value:

$$
dh = x_i \times 10^{-6}
$$

Note: Numerical differentiation is an aproximation, so it may be less accurate than analytical derivatives.

```rust
use num_complex::Complex;
use formulac::{Builder, Variables, UserDefinedTable, UserDefinedFunction};

fn main() {
    // Define f(x) = x^2 without derivative
    let users = UserDefinedTable::default()
        .register(UserDefinedFunction::new(
           "f",
            |args: &[Complex<f64>]| args[0] * args[0],
            1,
        )).unwrap();

    let expr = Builder::new("diff(f(x), x)", &["x"])
        .with_user_defined_functions(users)
        .compile()
        .expect("Failed to compile formula with UserDefinedFunction");

    let result = expr(&[Complex::new(3.0, 0.0)]); // evaluates f'(3) ≈ 6
    println!("Result: {}", result);
}
```

### Example: Multi-variable functions (Partial derivatives)

For functions with multiple variables, you can register partial derivatives with respect to each argument. Use the same order as the function arguments.

```rust
use num_complex::Complex;
use formulac::{Builder, Variables, UserDefinedTable, UserDefinedFunction};

fn main() {
    // Define a partial derivative w.r.t x: ∂g/∂x = 2*x*y
    let deriv_x = UserDefinedFunction::new(
        "dg_dx",
        |args: &[Complex<f64>]| Complex::new(2.0, 0.0) * args[0] * args[1],
        2,
    );
    // Define a partial derivative w.r.t y: ∂g/∂y = x^2 + 3*y^2
    let deriv_y = UserDefinedFunction::new(
        "dg_dy",
        |args: &[Complex<f64>]| args[0]*args[0] + Complex::new(3.0, 0.0)*args[1]*args[1],
        2,
    );
    // Define g(x, y) = x^2 * y + y^3
    let func = UserDefinedFunction::new(
        "g",
        |args: &[Complex<f64>]| args[0]*args[0]*args[1] + args[1]*args[1]*args[1],
        2,
    ).with_derivative(vec![deriv_x, deriv_y]);

    let users = UserDefinedTable::default()
        .register(func).unwrap();

    let expr_dx = Builder::new("diff(g(x, y), x)", &["x", "y"])
        .with_user_defined_functions(users.clone()) // use it again later
        .compile()
        .unwrap();
    let result_dx = expr_dx(&[Complex::new(2.0, 0.0), Complex::new(3.0, 0.0)]);
    println!("∂g/∂x at (2, 3) = {}", result_dx); // 12

    let expr_dy = Builder::new("diff(g(x, y), y)", &["x", "y"])
        .with_user_defined_functions(users)
        .compile()
        .unwrap();
    let result_dy = expr_dy(&[Complex::new(2.0, 0.0), Complex::new(3.0, 0.0)]);
    println!("∂g/∂y at (2, 3) = {}", result_dy); // 31
}
```

## Core Types & API Overview

- **`compile`**
  Compiles a formula string into a Rust closure `Fn(&[Complex<f64>]) -> Complex<f64>` that evaluates the expression for given variable values.

- **`Variables`**
  A lookup table mapping variable names (strings) to `Complex<f64>` values, used during parsing.

- **`UserDefinedTable`**
  Allows registration of custom functions under user-defined names for use in expressions.

### Available mathematical constants

| String           | Value      | Description                               |
| ---------------- | ---------- | ----------------------------------------- |
| `E`              | `e`        | The base of natural logarithm (≈ 2.71828) |
| `FRAC_1_PI`      | `1 / π`    | Reciprocal of π                           |
| `FRAC_1_SQRT_2`  | `1 / √2`   | Reciprocal of square root of 2            |
| `FRAC_2_PI`      | `2 / π`    | 2 divided by π                            |
| `FRAC_2_SQRT_PI` | `2 / √π`   | 2 divided by square root of π             |
| `FRAC_PI_2`      | `π / 2`    | Half of π                                 |
| `FRAC_PI_3`      | `π / 3`    | One-third of π                            |
| `FRAC_PI_4`      | `π / 4`    | One-fourth of π                           |
| `FRAC_PI_6`      | `π / 6`    | One-sixth of π                            |
| `FRAC_PI_8`      | `π / 8`    | One-eighth of π                           |
| `LN_2`           | `ln(2)`    | Natural logarithm of 2                    |
| `LN_10`          | `ln(10)`   | Natural logarithm of 10                   |
| `LOG2_10`        | `log2(10)` | Base-2 logarithm of 10                    |
| `LOG2_E`         | `log2(e)`  | Base-2 logarithm of e                     |
| `LOG10_2`        | `log10(2)` | Base-10 logarithm of 2                    |
| `LOG10_E`        | `log10(e)` | Base-10 logarithm of e                    |
| `PI`             | `π`        | Ratio of circle circumference to diameter |
| `SQRT_2`         | `√2`       | Square root of 2                          |
| `TAU`            | `2 * π`    | Full circle in radians                    |

### Available unary operators

| String | Function   | Description       |
| ------ | ---------- | ----------------- |
| `+`    | `Positive` | Identity operator |
| `-`    | `Negative` | Negation operator |

### Available binary operators

| String | Function | Description    |
| ------ | -------- | -------------- |
| `+`    | `Add`    | Addition       |
| `-`    | `Sub`    | Subtraction    |
| `*`    | `Mul`    | Multiplication |
| `/`    | `Div`    | Division       |
| `^`    | `Pow`    | Power (x^y)    |

### Available Functions

| String  | Function     | Description                      |
| ------- | ------------ | -------------------------------- |
| `sin`   | `Sin(x)`     | Sine function                    |
| `cos`   | `Cos(x)`     | Cosine function                  |
| `tan`   | `Tan(x)`     | Tangent function                 |
| `asin`  | `Asin(x)`    | Arc sine                         |
| `acos`  | `Acos(x)`    | Arc cosine                       |
| `atan`  | `Atan(x)`    | Arc tangent                      |
| `sinh`  | `Sinh(x)`    | Hyperbolic sine                  |
| `cosh`  | `Cosh(x)`    | Hyperbolic cosine                |
| `tanh`  | `Tanh(x)`    | Hyperbolic tangent               |
| `asinh` | `Asinh(x)`   | Hyperbolic arcsine               |
| `acosh` | `Acosh(x)`   | Hyperbolic arccosine             |
| `atanh` | `Atanh(x)`   | Hyperbolic arctangent            |
| `exp`   | `Exp(x)`     | Exponential function e^x         |
| `ln`    | `Ln(x)`      | Natural logarithm                |
| `log10` | `Log10(x)`   | Base-10 logarithm                |
| `sqrt`  | `Sqrt(x)`    | Square root                      |
| `abs`   | `Abs(x)`     | Absolute value                   |
| `conj`  | `Conj(x)`    | Complex conjugate                |
| `pow`   | `Pow(x, y)`  | x raised to y (complex exponent) |
| `powi`  | `Powi(x, n)` | x raised to integer n            |

### Available differential operator

| String          | Function        | Description                                                          |
| --------------- | --------------- | -------------------------------------------------------------------- |
| `diff(f, x)`    | `diff(f, x)`    | First-order derivative of `f` with respect to `x`                    |
| `diff(f, x, n)` | `diff(f, x, n)` | n-th order derivative of `f` with respect to `x` (**max `i8::MAX`**) |

---

## Benchmarking

The `formulac` crate provides benchmarks using the Criterion crate to measure both compilation and execution performance.

**Note**:

- Criterion is a dev-dependency, so benchmarks are only available in development builds.
- :warning: Some benchmarks (e.g., 1000 nested operations) may take longer to run.
- Benchmarks require the `criterion` crate as a dev-dependency and are intended for development/testing purposes only.

### Benchmark Source

The benchmarks are located in `benches/benches.rs` and cover:

- Linear expressions (many operands like polynomials)
- Nested expressions (e.g., sin(sin(...)))
- Numeric literals
- Expressions with and without parentheses
- Many variable references (up to 100 variables)
- Differentiate expressions
- Invalid expressions (error cases)
- Practical expressions (polynomials, wave functions, exponential decay)
- Comparison of direct calls vs. parsed calls for standard functions (sin, cos, pow, etc.)

### Run Benchmarks

Run the benchmarks with:

``` bash
cargo bench
```

Both `compile` and `exec` times are measured. Criterion generates detailed statistics and plots in `target/criterion`.

### Viewing Results

Open the generated HTML report in a browser to view benchmark results and comparisons:

``` bash
open target/criterion/report/index.html
```

---

## License

Licensed under **MIT OR Apache-2.0** — choose the license that best suits your project.

---

## Contribution & Contact

Contributions, feature requests, and bug reports are welcome!
Please feel free to open issues or submit pull requests via the GitHub repository.
