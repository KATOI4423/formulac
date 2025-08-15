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
//! ## Examples
//! ```rust
//! use approx::assert_abs_diff_eq;
//! use num_complex::Complex;
//! use formulac::variable::Variables;
//! use formulac::compile;
//!
//! // Define available variables
//! let mut vars = Variables::new();
//! vars.insert(&[("a", Complex::new(3.0, 4.0))]);
//!
//! // Compile an expression with arguments
//! let expr = compile("z + a * 2", &["z"], &vars).unwrap();
//!
//! // Evaluate the compiled expression with argument a = (1 + 2i)
//! let result = expr(&[Complex::new(1.0, 2.0)]);
//! assert_abs_diff_eq!(result.re, 7.0, epsilon = 1.0e-10);
//! assert_abs_diff_eq!(result.im, 10.0, epsilon = 1.0e-10);
//! ```
//!
//! ## How it works
//! 1. The formula string is parsed into tokens.
//! 2. Tokens are converted into Reverse Polish Notation (RPN) for unambiguous ordering.
//! 3. The RPN sequence is compiled into a closure operating on a stack of `Complex<f64>` values.
//! 4. The resulting closure can be called repeatedly with different arguments for fast evaluation.

mod token;
mod rpn;
pub mod variable;

use std::collections::VecDeque;

use num_complex::Complex;
use crate::token::{Token, Operator, Function};
use crate::rpn::make_rpn;
use crate::variable::Variables;

type Stack = VecDeque<Complex<f64>>;
type FuncList = VecDeque<Box<dyn Fn(&mut Stack, &[Complex<f64>])>>;

/// Compiles an operator token into a stack operation.
///
/// This function validates that there are at least two operands on the stack,
/// then appends a closure to the function list that will:
/// - Pop the top two operands from the stack.
/// - Apply the given operator to them.
/// - Push the result back onto the stack.
///
/// # Arguments
/// * `stack` - The current compilation-time stack used for validation.
/// * `func_list` - The list of compiled closures representing stack operations.
/// * `oper` - The operator to compile.
///
/// # Errors
/// Returns an error if the stack does not contain enough operands.
///
/// # Example
/// ```
/// // Called internally by `compile` when encountering a binary operator.
/// ```
fn compile_case_of_operator(stack: &mut Stack, func_list: &mut FuncList, oper: Operator) -> Result<(), String> {
    if stack.len() < 2 {
        return Err("Invalid formula: missing number of argument for operator".into());
    }
    func_list.push_back(Box::new(
        move |stack, _| {
            let arg2 = stack.pop_back().unwrap();
            let arg1 = stack.pop_back().unwrap();
            let args = [arg1, arg2];
            stack.push_back(oper.func(&args));
        }
    ));
    stack.pop_back(); stack.pop_back();
    stack.push_back(Complex { re: 0.0, im: 0.0 }); // temp value (the actual calculated value can only be obtained at runtime)
    return Ok(());
}

/// Compiles a function token into a stack operation.
///
/// This function ensures that the required number of arguments are present
/// on the stack, updates the variable count, and appends a closure to `func_list`
/// that will:
/// - Pop the required number of arguments from the stack.
/// - Reverse them into the correct order.
/// - Call the function with these arguments.
/// - Push the result back onto the stack.
///
/// # Arguments
/// * `stack` - The current compilation-time stack used for validation.
/// * `func_list` - The list of compiled closures representing stack operations.
/// * `variable_cnt` - The number of variables currently on the stack; updated after consuming arguments.
/// * `func` - The function to compile.
///
/// # Errors
/// Returns an error if there are not enough arguments on the stack.
///
/// # Example
/// ```
/// // Called internally by `compile` when encountering a user-defined or built-in function.
/// ```
fn compile_case_of_function(stack: &mut Stack, func_list: &mut FuncList, variable_cnt: &mut usize, func: Function) -> Result<(), String> {
    let stack_len = stack.len();
    let args_num = func.args_num() as usize;

    if stack_len < args_num {
        return Err("Invalid formula: missing number of argument for function".into());
    }
    if let Some(cnt) = variable_cnt.checked_sub(args_num) {
        *variable_cnt = cnt;
    } else {
        return Err("Invalid formula: missing number of argument for function".into());
    }
    func_list.push_back(Box::new(
        move |stack, _| {
            let mut args: Vec<Complex<f64>> = Vec::with_capacity(args_num);
            for _ in 0..args_num {
                args.push(stack.pop_back().unwrap());
            }
            args.reverse();
            stack.push_back(func.func(&args));
        }
    ));
    for _ in 0..args_num {
        stack.pop_back();
    }
    stack.push_back(Complex { re: 0.0, im: 0.0 }); // temp value (the actual calculated value can only be obtained at runtime)
    return Ok(());
}

/// Creates an executable function from a list of stack operations.
///
/// This function consumes a `FuncList` (list of closures) and returns a closure
/// that:
/// - Initializes an empty evaluation stack.
/// - Executes each compiled closure in order, passing in the evaluation stack
///   and the runtime arguments.
/// - Returns the final value remaining on the stack.
///
/// # Arguments
/// * `func_list` - The list of compiled stack operations.
///
/// # Returns
/// A closure of type `Fn(&[Complex<f64>]) -> Complex<f64>` that evaluates the expression.
///
/// # Panics
/// Panics if the evaluation stack is empty after executing all operations.
///
/// # Example
/// ```
/// // Used internally by `compile` to produce the final callable function.
/// ```
fn make_function(func_list: FuncList) -> impl Fn(&[Complex<f64>]) -> Complex<f64> {
    move |args: &[Complex<f64>]| {
        let mut stack: Stack = VecDeque::new();
        for f in func_list.iter() {
            f(&mut stack, args);
        }
        return stack.pop_back().expect("stack is empty");
    }
}

/// Compiles a formula string into an executable function.
///
/// This is the main entry point for turning a formula string into a compiled
/// closure that can be executed with different argument values.
/// It performs the following steps:
/// 1. Tokenizes and converts the formula into Reverse Polish Notation (RPN).
/// 2. Iterates over the RPN tokens and compiles each into a stack operation
///    using [`compile_case_of_operator`] and [`compile_case_of_function`].
/// 3. Wraps the sequence of operations into a callable function via [`make_function`].
///
/// # Arguments
/// * `formula` - The formula string to compile.
/// * `args` - A list of argument variable names used in the formula.
/// * `vars` - A reference to the variables table.
///
/// # Returns
/// On success, returns a closure of type `Fn(&[Complex<f64>]) -> Complex<f64>`
/// that can be called to evaluate the formula with given arguments.
///
/// # Errors
/// Returns an error if the formula is invalid, such as having missing operands
/// for an operator or function.
///
/// # Example
/// ```
/// use approx::assert_abs_diff_eq;
/// use num_complex::Complex;
/// use formulac::variable::Variables;
/// use formulac::compile;
///
/// let mut vars = Variables::new();
/// vars.insert(&[("c", Complex::new(0.5, 0.3))]);
///
/// let expr = compile("z^2 + c", &["z"], &vars).unwrap();
/// let result = expr(&[Complex::new(0.3, -0.1)]);
/// assert_abs_diff_eq!(result.re, 0.58, epsilon = 1.0e-10);
/// assert_abs_diff_eq!(result.im, 0.24, epsilon = 1.0e-10);
/// ```
pub fn compile<'a>(
    formula: &str,
    args: &[&str],
    vars: &'a Variables
) -> Result<impl Fn(&[Complex<f64>]) -> Complex<f64> + 'a, String>
{
    let rpn = make_rpn(formula, args, vars)?;
    let mut func_list: FuncList = VecDeque::new();
    let mut stack: Stack = VecDeque::new();
    let mut variable_cnt: usize = 0;

    for token in rpn {
        dbg!(format!("token: {:?}", token));
        dbg!(format!("stack: {:?}", stack));
        match token {
            Token::Operator(oper)
                => compile_case_of_operator(&mut stack, &mut func_list, oper)?,
            Token::Function(func)
                => compile_case_of_function(&mut stack, &mut func_list, &mut variable_cnt, func)?,
            Token::Variable(v) |
            Token::Constant(v) |
            Token::Real(v) |
            Token::Imaginary(v) => {
                stack.push_back(v);
                func_list.push_back(Box::new(
                    move |stack, _| stack.push_back(v)
                ));
            },
            Token::Argument(idx) => {
                stack.push_back(Complex { re: 0.0, im: 0.0 }); // push temp value (the actual argument used can only be obtained at runtime)
                func_list.push_back(Box::new(
                    move |stack, args| stack.push_back(args[idx])
                ));
            },
            Token::LParen | Token::RParen | Token::Comma
                => return Err(format!("make_rpn returns invalid token list: it includes {:?}", token)),
        }
        variable_cnt += 1;
    }

    Ok(make_function(func_list))
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;
    use approx::assert_abs_diff_eq;
    use crate::variable::Variables;

    #[test]
    fn test_compile_basic_operations() {
        let vars = Variables::new();

        // Addition
        let expr = compile("1 + 2", &[], &vars).unwrap();
        let result = expr(&[]);
        assert_abs_diff_eq!(result.re, 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(result.im, 0.0, epsilon = 1e-12);

        // Subtraction
        let expr = compile("5 - 2", &[], &vars).unwrap();
        let result = expr(&[]);
        assert_abs_diff_eq!(result.re, 3.0, epsilon = 1e-12);

        // Multiplication
        let expr = compile("3 * 4", &[], &vars).unwrap();
        let result = expr(&[]);
        assert_abs_diff_eq!(result.re, 12.0, epsilon = 1e-12);

        // Division
        let expr = compile("10 / 2", &[], &vars).unwrap();
        let result = expr(&[]);
        assert_abs_diff_eq!(result.re, 5.0, epsilon = 1e-12);
    }

    #[test]
    fn test_compile_complex_numbers() {
        let vars = Variables::new();

        // Real + imaginary
        let expr = compile("1 + 2i", &[], &vars).unwrap();
        let result = expr(&[]);
        assert_abs_diff_eq!(result.re, 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(result.im, 2.0, epsilon = 1e-12);

        // Imaginary squared
        let expr = compile("i * i", &[], &vars).unwrap();
        let result = expr(&[]);
        assert_abs_diff_eq!(result.re, -1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(result.im, 0.0, epsilon = 1e-12);
    }
    #[test]
    fn test_compile_with_variables_and_arguments() {
        let mut vars = Variables::new();
        vars.insert(&[("a", Complex::new(2.0, 3.0))]);

        let expr = compile("a + x", &["x"], &vars).unwrap();
        let result = expr(&[Complex::new(1.0, -1.0)]);
        assert_abs_diff_eq!(result.re, 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(result.im, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_compile_functions() {
        let vars = Variables::new();

        // Using built-in functions like sin, cos if supported
        let expr = compile("sin(0)", &[], &vars).unwrap();
        let result = expr(&[]);
        assert_abs_diff_eq!(result.re, 0.0, epsilon = 1e-12);

        let expr = compile("cos(0)", &[], &vars).unwrap();
        let result = expr(&[]);
        assert_abs_diff_eq!(result.re, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_compile_operator_precedence() {
        let vars = Variables::new();

        // Multiplication before addition
        let expr = compile("1 + 2 * 3", &[], &vars).unwrap();
        let result = expr(&[]);
        assert_abs_diff_eq!(result.re, 7.0, epsilon = 1e-12);

        // Parentheses override precedence
        let expr = compile("(1 + 2) * 3", &[], &vars).unwrap();
        let result = expr(&[]);
        assert_abs_diff_eq!(result.re, 9.0, epsilon = 1e-12);
    }

    #[test]
    fn test_compile_negative_numbers() {
        let vars = Variables::new();

        let expr = compile("-3 + 2", &[], &vars).unwrap();
        let result = expr(&[]);
        assert_abs_diff_eq!(result.re, -1.0, epsilon = 1e-12);

        let expr = compile("4 * -2", &[], &vars).unwrap();
        let result = expr(&[]);
        assert_abs_diff_eq!(result.re, -8.0, epsilon = 1e-12);
    }

    #[test]
    fn test_compile_errors() {
        let vars = Variables::new();

        // Missing operand
        let err = match compile("1 +", &[], &vars) {
            Ok(_) => panic!("Expected error, got Ok"),
            Err(e) => e,
        };
        assert!(err.contains("missing number of argument"));

        // Unknown variable
        let err = match compile("x + 1", &[], &vars) {
            Ok(_) => panic!("Expected error, got Ok"),
            Err(e) => e,
        };
        assert!(err.contains("Unknown string"));
    }
}
