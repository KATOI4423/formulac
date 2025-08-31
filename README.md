# formulac

A Rust crate for parsing and evaluating mathematical expressions with full complex-number support, extensible user-defined functions, and detailed AST representation.

## Features

- **Complex number support** — Evaluate expressions involving real and imaginary components using `num_complex::Complex<f64>`.
- **Reverse Polish Notation (RPN)** — Converts infix expressions to RPN using the Shunting-Yard algorithm for efficient evaluation.
- **Built-in mathematical operators & functions** — Supports `+`, `-`, `*`, `/`, `^`, and standard functions like `sin`, `cos`, `exp`, `log`, and more.
- **Unary & Binary operators** — Unary operators (`+`, `-`) and binary operators (`+`, `-`, `*`, `/`, `^`) are represented as `UnaryOperatorKind` and `BinaryOperatorKind`.
- **Abstract Syntax Tree (AST)** — Expressions are parsed into `AstNode` structures, enabling inspection, simplification, and compilation into executable closures.
- **User-defined functions** — Easily register custom functions or constants at runtime using a simple API (`UserDefinedTable`).
- **Safe and dependency-light** — No use of unsafe Rust or heavyweight external parsers.

---

## Usage

Add to your `Cargo.toml`:

```bash
cargo add formulac
```

or

```toml
[dependencies]
formulac = "0.4"
num-complex = "0.4"
```

### Basic Example

```rust
use num_complex::Complex;
use formulac::{compile, variable::Variables, variable::UserDefinedTable};

fn main() {
    let mut vars = Variables::new();
    vars.insert(&[("a", Complex::new(3.0, 2.0))]);

    let users = UserDefinedTable::new();
    let formula = "sin(z) + a * cos(z)";
    let expr = compile(formula, &["z"], &vars, &users)
        .expect("Failed to compile formula");

    let result = expr(&[Complex::new(1.0, 2.0)]); // calc 'sin(1+2i) + (3+2i) * cos(1+2i)'
    println!("Result = {}", result);
}
```

### Registering a Custom Function

```rust
use num_complex::Complex;
use formulac::{compile, variable::Variables, UserDefinedTable, UserDefinedFunction};

// Create user-defined function table
let mut users = UserDefinedTable::new();

// Define a function f(x) = x^2 + 1
let func = UserDefinedFuncion::new(
    "my_func",
    |args: &[Complex<f64>]| args[0] * args[0] + Complex::new(1.0, 0.0),
    1,
);
users.register("f", func);

let mut vars = Variables::new();
let expr = compile("f(3)", &[], &vars, &users).unwrap();
assert_eq!(expr(&[]), Complex::new(10.0, 0.0));
```

---

## Core Types & API Overview

- **`compile`**
  Compiles a formula string into a Rust closure `Fn(&[Complex<f64>]) -> Complex<f64>` that evaluates the expression for given variable values.

- **`Variables`**
  A lookup table mapping variable names (strings) to `Complex<f64>` values, used during parsing.

- **`UserDefinedTable`**
  Allows registration of custom functions under user-defined names for use in expressions.

---

## License

Licensed under **MIT OR Apache-2.0** — choose the license that best suits your project.

---

## Contribution & Contact

Contributions, feature requests, and bug reports are welcome!
Please feel free to open issues or submit pull requests via the GitHub repository.
