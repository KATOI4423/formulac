//! # formulac
//!
//! `formulac` is a Rust library for parsing and evaluating mathematical expressions
//! with support for **complex numbers** and **extensible user-defined functions**.
//!
//! It allows you to:
//! - Parse and evaluate expressions containing real and imaginary numbers.
//! - Use built-in operators, constants, and mathematical functions.
//! - Register your own variables and functions.
//! - Work with expressions in a compiled, callable form for repeated evaluation.
//!
//! Internally, expressions are converted to Reverse Polish Notation (RPN),
//! then compiled into a sequence of stack operations for fast execution.
//!
//! ## Feature Overview
//! - **Complex number support** using [`num_complex::Complex`]
//! - **Custom functions** that can be registered at runtime
//! - **Variables and arguments** passed at evaluation time
//! - **Operator precedence** and parentheses handling
//! - **Efficient compiled execution** avoiding repeated parsing
//!

mod lexer;
mod parser;
pub mod variable;

use num_complex::Complex;
use crate::{parser::make_function_list, variable::{UserDefinedTable, Variables}};

pub fn compile(
    formula: &str,
    arg_names: &[&str],
    vars: &Variables,
    users: &UserDefinedTable
) -> Result<impl Fn(&[Complex<f64>]) -> Option<Complex<f64>>, String>
{
    let lexemes = lexer::from(formula);
    let ast = parser::AstNode::from(&lexemes, arg_names, vars, users)?;
    let tokens = ast.simplify().compile();
    let funcs = make_function_list(tokens);

    let func = move |arg_values: &[Complex<f64>]| {
        if arg_names.len() != arg_values.len() {
            return None;
        }
        let mut stack: Vec<Complex<f64>> = Vec::new();
        for func in funcs.iter() {
            func(&mut stack, arg_values);
        }
        stack.pop()
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
        assert_eq!(result, Some(Complex::new(42.0, 0.0)));
    }

    #[test]
    fn test_constant_str() {
        let vars = Variables::new();
        let users = UserDefinedTable::new();
        let f = compile("PI", &[], &vars, &users).unwrap();
        let result = f(&[]);
        assert_eq!(result, Some(Complex::from(std::f64::consts::PI)));
    }

    #[test]
    fn test_argument() {
        let vars = Variables::new();
        let users = UserDefinedTable::new();
        let f = compile("x", &["x"], &vars, &users).unwrap();
        let result = f(&[Complex::new(3.0, 0.0)]);
        assert_eq!(result, Some(Complex::new(3.0, 0.0)));
    }

    #[test]
    fn test_addition() {
        let vars = Variables::new();
        let users = UserDefinedTable::new();
        let f = compile("x + y", &["x", "y"], &vars, &users).unwrap();
        let x = Complex::new(2.0, 1.0);
        let y = Complex::new(3.0, 5.0);
        let result = f(&[x, y]).unwrap();
        assert_abs_diff_eq!(result.re(), (x + y).re(), epsilon=1.0e-12);
        assert_abs_diff_eq!(result.im(), (x + y).im(), epsilon=1.0e-12);
    }

    #[test]
    fn test_nested_expression() {
        let vars = Variables::new();
        let users = UserDefinedTable::new();
        let f = compile("sin(x + 1)", &["x"], &vars, &users).unwrap();
        let result = f(&[Complex::new(0.0, 1.0)]).unwrap();
        let expected = Complex::new(1.0, 1.0).sin();
        assert_abs_diff_eq!(result.re(), expected.re(), epsilon=1.0e-12);
        assert_abs_diff_eq!(result.im(), expected.im(), epsilon=1.0e-12);
    }

    #[test]
    fn test_binary_operator_precedence() {
        let vars = Variables::new();
        let users = UserDefinedTable::new();
        let f = compile("2 + 3 * 4", &[], &vars, &users).unwrap();
        let result = f(&[]).unwrap();
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
        let result = f(&[a, b]).unwrap();
        let expected = a.powc(b);
        assert_abs_diff_eq!(result.re(), expected.re(), epsilon=1.0e-12);
        assert_abs_diff_eq!(result.im(), expected.im(), epsilon=1.0e-12);
    }

    #[test]
    fn test_invalid_args_length() {
        let vars = Variables::new();
        let users = UserDefinedTable::new();
        let f = compile("x + 1", &["x"], &vars, &users).unwrap();
        // Too less arguments
        assert_eq!(f(&[]), None);
        // Too much arguments
        assert_eq!(f(&[Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)]), None);
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
        let result = f(&[x]).unwrap();
        let expected = a * x + b;
        assert_abs_diff_eq!(result.re(), expected.re(), epsilon=1.0e-12);
        assert_abs_diff_eq!(result.im(), expected.im(), epsilon=1.0e-12);
    }

}
