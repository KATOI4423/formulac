//! # formulac
//!
//! `formulac` is a Rust library for parsing and evaluating mathematical
//! expressions with support for **complex numbers** and **extensible user-defined functions**.
//!
//! ## Overview
//! - Parse and evaluate expressions containing real and imaginary numbers.
//! - Use built-in operators, constants, and mathematical functions.
//! - Register your own variables and functions.
//! - Compile expressions into callable closures for repeated evaluation without re-parsing.
//!
//! Internally, expressions are first tokenized into lexeme,
//! then converted to an AST using the Shunting-Yard algorithm,
//! and finally compiled into Reverse Polish Notation (RPN) stack operations
//! for fast repeated execution.
//!
//! ## Feature Highlights
//! - **Complex number support** using [`num_complex::Complex<f64>`]
//! - **User-defined functions and constants** via [`UserDefinedTable`]
//! - **Variables and arguments** managed by [`Variables`]
//! - **Operator precedence** and parentheses handling
//! - **Efficient compiled closures** avoiding repeated parsing
//!
//! ## Example
//! ```rust
//! use num_complex::Complex;
//! use formulac::{compile, Variables, UserDefinedTable};
//!
//! let mut vars = Variables::new();
//! vars.insert(&[("a", Complex::new(3.0, 2.0))]);
//!
//! let users = UserDefinedTable::new();
//! let expr = compile("sin(z) + a * cos(z)", &["z"], &vars, &users)
//!     .expect("Failed to compile formula");
//!
//! let result = expr(&[Complex::new(1.0, 2.0)]);
//! println!("Result = {}", result);
//! ```
//!
//! ## Example: Retrieving All Names
//! ```rust
//! use formulac::parser::{constant, UnaryOperatorKind, BinaryOperatorKind, FunctionKind};
//!
//! // Constants
//! let constant_names: Vec<&'static str> = constant::names();
//! println!("Constants: {:?}", constant_names);
//!
//! // Unary operators
//! let unary_names: Vec<&'static str> = UnaryOperatorKind::names();
//! println!("Unary Operators: {:?}", unary_names);
//!
//! // Binary operators
//! let binary_names: Vec<&'static str> = BinaryOperatorKind::names();
//! println!("Binary Operators: {:?}", binary_names);
//!
//! // Functions
//! let function_names: Vec<&'static str> = FunctionKind::names();
//! println!("Functions: {:?}", function_names);
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

mod lexer;
pub mod parser;
pub mod variable;

use num_complex::Complex;
use crate::{parser::Token};

pub type UserDefinedFunction = variable::UserDefinedFunction;
pub type UserDefinedTable = variable::UserDefinedTable;
pub type Variables = variable::Variables;

/// Compiles a mathematical expression into an executable closure.
///
/// This function parses a formula string into an abstract syntax tree (AST),
/// simplifies it, and then compiles it into a list of stack operations
/// (in Reverse Polish Notation). The result is returned as a closure that
/// can be called multiple times with different argument values without
/// re-parsing the formula.
///
/// # Parameters
/// - `formula`: A string slice containing the mathematical expression to compile.
/// - `arg_names`: A slice of argument names (`&str`) that the formula depends on.
///   The closure returned will expect argument values in the same order.
/// - `vars`: A [`Variables`] table mapping variable names
///   to constant values available in the formula.
/// - `users`: A [`UserDefinedTable`] containing
///   any user-defined functions or constants available in the formula.
///
/// # Returns
/// On success, returns a closure of type:
///
/// ```rust,ignore
/// Fn(&[Complex<f64>]) -> Complex<f64>
/// ```
///
/// - The closure takes a slice of complex argument values corresponding to `arg_names`.
/// - Returns `Complex<f64>` if evaluation succeeds.
/// - Panics if the number of arguments provided does not match `arg_names.len()`.
///   So check it with debug build.
///
/// On failure, returns an error string describing the parsing or compilation error.
///
/// # Example
/// ```rust
/// use num_complex::Complex;
/// use formulac::{compile, Variables, UserDefinedTable};
///
/// let mut vars = Variables::new();
/// vars.insert(&[("a", Complex::new(3.0, 2.0))]);
///
/// let users = UserDefinedTable::new();
/// let expr = compile("sin(z) + a * cos(z)", &["z"], &vars, &users)
///     .expect("Failed to compile formula");
///
/// let result = expr(&[Complex::new(1.0, 2.0)]);
/// println!("Result = {}", result);
/// ```
///
/// # Notes
/// - The formula string must be a valid expression using supported operators,
///   variables, and functions.
/// - Argument names are resolved in the order provided by `arg_names`.
/// - This function does not evaluate immediately; instead, it produces
///   a reusable compiled closure for efficient repeated evaluation.
pub fn compile(
    formula: &str,
    arg_names: &[&str],
    vars: &Variables,
    users: &UserDefinedTable
) -> Result<impl Fn(&[Complex<f64>]) -> Complex<f64>, String>
{
    let lexemes = lexer::from(formula);
    let tokens = parser::AstNode::from(&lexemes, arg_names, vars, users)?
        .simplify().compile();

    let expected_arity = arg_names.len();
    let func = move |arg_values: &[Complex<f64>]| {
        // check arity only debug build
        debug_assert_eq!(arg_values.len(), expected_arity);

        let mut stack: Vec<Complex<f64>> = Vec::new();
        for token in tokens.iter() {
            match token {
                Token::Number(val) => stack.push(*val),
                Token::Argument(idx) => stack.push(arg_values[*idx]),
                Token::UnaryOperator(oper) => {
                    let expr = stack.pop().unwrap();
                    stack.push(oper.apply(expr));
                },
                Token::BinaryOperator(oper) => {
                    let r = stack.pop().unwrap();
                    let l = stack.pop().unwrap();
                    stack.push(oper.apply(l, r));
                },
                Token::Function(func) => {
                    let n = func.arity();
                    let mut args: Vec<Complex<f64>> = Vec::with_capacity(n);
                    args.resize(n, Complex::new(0.0, 0.0));

                    for i in (0..n).rev() {
                        args[i] = stack.pop().unwrap();
                    }
                    stack.push(func.apply(&args));
                },
                Token::UserFunction(func) => {
                    let n = func.arity();
                    let mut args: Vec<Complex<f64>> = Vec::with_capacity(n);
                    args.resize(n, Complex::new(0.0, 0.0));

                    for i in (0..n).rev() {
                        args[i] = stack.pop().unwrap();
                    }
                    stack.push(func.apply(&args));
                },
                _ => unreachable!("Invalid tokens found: use compiled tokens"),
            }
        }

        stack.pop().unwrap()
    };

    Ok(func)
}

#[cfg(test)]
mod compile_test {
    use super::*;
    use num_complex::{Complex, ComplexFloat};
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_constant_number() {
        let vars = Variables::new();
        let users = UserDefinedTable::new();
        let f = compile("42", &[], &vars, &users).unwrap();
        let result = f(&[]);
        assert_eq!(result, Complex::new(42.0, 0.0));
    }

    #[test]
    fn test_constant_str() {
        let vars = Variables::new();
        let users = UserDefinedTable::new();
        let f = compile("PI", &[], &vars, &users).unwrap();
        let result = f(&[]);
        assert_eq!(result, Complex::from(std::f64::consts::PI));
    }

    #[test]
    fn test_argument() {
        let vars = Variables::new();
        let users = UserDefinedTable::new();
        let f = compile("x", &["x"], &vars, &users).unwrap();
        let result = f(&[Complex::new(3.0, 0.0)]);
        assert_eq!(result, Complex::new(3.0, 0.0));
    }

    #[test]
    fn test_addition() {
        let vars = Variables::new();
        let users = UserDefinedTable::new();
        let f = compile("x + y", &["x", "y"], &vars, &users).unwrap();
        let x = Complex::new(2.0, 1.0);
        let y = Complex::new(3.0, 5.0);
        let result = f(&[x, y]);
        assert_abs_diff_eq!(result.re(), (x + y).re(), epsilon=1.0e-12);
        assert_abs_diff_eq!(result.im(), (x + y).im(), epsilon=1.0e-12);
    }

    #[test]
    fn test_nested_expression() {
        let vars = Variables::new();
        let users = UserDefinedTable::new();
        let f = compile("sin(x + 1)", &["x"], &vars, &users).unwrap();
        let result = f(&[Complex::new(0.0, 1.0)]);
        let expected = Complex::new(1.0, 1.0).sin();
        assert_abs_diff_eq!(result.re(), expected.re(), epsilon=1.0e-12);
        assert_abs_diff_eq!(result.im(), expected.im(), epsilon=1.0e-12);
    }

    #[test]
    fn test_binary_operator_precedence() {
        let vars = Variables::new();
        let users = UserDefinedTable::new();
        let f = compile("2 + 3 * 4", &[], &vars, &users).unwrap();
        let result = f(&[]);
        let expected = Complex::from(2.0 + 3.0 * 4.0);
        assert_abs_diff_eq!(result.re(), expected.re(), epsilon=1.0e-12);
        assert_abs_diff_eq!(result.im(), expected.im(), epsilon=1.0e-12);
    }

    #[test]
    fn test_function_with_two_args() {
        let vars = Variables::new();
        let users = UserDefinedTable::new();
        let f = compile("pow(a, b)", &["a", "b"], &vars, &users).unwrap();
        let a = Complex::new(2.0, 1.0);
        let b = Complex::new(-2.0, 3.0);
        let result = f(&[a, b]);
        let expected = a.powc(b);
        assert_abs_diff_eq!(result.re(), expected.re(), epsilon=1.0e-12);
        assert_abs_diff_eq!(result.im(), expected.im(), epsilon=1.0e-12);
    }

    #[test]
    #[should_panic]
    fn test_too_less_args_length() {
        let vars = Variables::new();
        let users = UserDefinedTable::new();
        let f = compile("x + 1", &["x"], &vars, &users).unwrap();
        f(&[]);
        // Too much arguments
        f(&[Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)]);
    }

    #[test]
    #[should_panic]
    fn test_too_much_args_length() {
        let vars = Variables::new();
        let users = UserDefinedTable::new();
        let f = compile("x + 1", &["x"], &vars, &users).unwrap();
        f(&[Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)]);
    }

    #[test]
    fn test_variables() {
        let a = Complex::new(2.0, 1.0);
        let b = Complex::new(-4.0, 2.0);
        let x = Complex::new(1.0, 0.0);
        let mut vars = Variables::new();
        let users = UserDefinedTable::new();
        vars.insert(&[("a", a), ("b", b),]);

        let f = compile("a * x + b", &["x"], &vars, &users).unwrap();
        let result = f(&[x]);
        let expected = a * x + b;
        assert_abs_diff_eq!(result.re(), expected.re(), epsilon=1.0e-12);
        assert_abs_diff_eq!(result.im(), expected.im(), epsilon=1.0e-12);
    }
}
