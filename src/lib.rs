//! # formulac
//!
//! `formulac` is a Rust library for parsing and evaluating mathematical
//! expressions with support for **complex numbers** and **extensible user-defined functions**.
//!
//! ## Overview
//! - Parse and evaluate expressions containing real and imaginary numbers.
//! - Use built-in operators, constants, and mathematical functions.
//! - Register your own constants and functions.
//! - Compile expressions into callable closures for repeated evaluation without re-parsing.
//!
//! Internally, expressions are first tokenized into lexeme,
//! then converted to an AST using the Shunting-Yard algorithm,
//! and finally compiled into Reverse Polish Notation (RPN) stack operations
//! for fast repeated execution.
//!
//! ## Feature Highlights
//! - **Complex number support** using [`num_complex::Complex<f64>`]
//! - **User-defined functions and constants**
//! - **Variables and arguments**
//! - **Operator precedence** and parentheses handling
//! - **Efficient compiled closures** avoiding repeated parsing
//!
//! ## Example
//! ```rust
//! use num_complex::Complex;
//! use formulac::{Builder, UserDefinedTable};
//!
//! let expr = Builder::new("sin(z) + a * cos(z)", &["z"])
//!     .with_constants([("a", Complex::new(3.0, 2.0))])
//!     .compile()
//!     .expect("Failed to compile formula");
//!
//! let result = expr(&[Complex::new(1.0, 2.0)]);
//! println!("Result = {}", result);
//! ```
//!
//! ## Example: Retrieving All Names
//! ```rust
//! use formulac::constants;
//!
//! // Constants
//! let constant_names: Vec<String> = constants::names();
//! println!("Constants: {:?}", constant_names);
//!
//! // Unary operators
//! //let unary_names: Vec<&'static str> = UnaryOperatorKind::names();
//! //println!("Unary Operators: {:?}", unary_names);
//!
//! // Binary operators
//! //let binary_names: Vec<&'static str> = BinaryOperatorKind::names();
//! //println!("Binary Operators: {:?}", binary_names);
//!
//! // Functions
//! //let function_names: Vec<&'static str> = FunctionKind::names();
//! //println!("Functions: {:?}", function_names);
//! ```
//!
//! ## When to Use
//! Use `formulac` when you need:
//! - Fast repeated evaluation of mathematical formulas
//! - Complex number support in expressions
//! - Runtime extensibility via custom functions or constants
//!
//! ## License
//! Licensed under either **MIT** or **Apache-2.0** at your option.

pub mod lexer;
pub mod astnode;
pub mod variable;
pub mod builder;
pub mod constants;

pub type UserDefinedFunction = variable::UserDefinedFunction;
pub type UserDefinedTable = variable::UserDefinedTable;
pub type Builder = builder::Builder;
