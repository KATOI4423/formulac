//! # builder.rs
//!
//! This module provides structures and utilities for building function object.

use crate::lexer;
use crate::astnode;
use crate::variable;

use astnode::*;
use variable::*;
use num_complex::Complex;


pub struct Builder
{
    formula: String,
    args: Vec<String>,
    vars: Variables,
    usrs: UserDefinedTable,
}

impl Builder
{
    /// Creates a new `Builder` instance with the given formula and argument names.
    ///
    /// This is the starting point for building a compiled mathematical expression.
    /// You can chain methods like `with_variables` and `with_user_defined_functions`
    /// to configure the builder before calling `compile`.
    ///
    /// # Parameters
    /// - `formula`: A string slice containing the mathematical expression to compile.
    /// - `arg_names`: A slice of argument names (`&str`) that the formula depends on.
    ///   These will be used as placeholders for input values in the compiled closure.
    ///
    /// # Returns
    /// A new `Builder` instance with default (empty) variables and user-defined functions.
    ///
    /// # Examples
    /// ```rust
    /// use formulac::Builder;
    /// use num_complex::Complex;
    ///
    /// let builder = Builder::new("x + 1", &["x"]);
    /// let func = builder.compile()
    ///     .expect("Failed to compile 'x + 1'");
    /// println!("{} + 1 = {}", 3, func(&[Complex::new(3.0, 0.0)]));
    /// ```
    pub fn new(formula: &str, arg_names: &[&str]) -> Self
    {
        Self {
            formula: formula.to_string(),
            args: arg_names.to_vec().iter().map(|arg| arg.to_string()).collect(),
            vars: Variables::new(),
            usrs: UserDefinedTable::new(),
        }
    }

    /// Sets the variables for the builder.
    ///
    /// Variables are constants that can be referenced in the formula by name.
    /// This method allows you to provide a pre-configured `Variables` table.
    ///
    /// # Parameters
    /// - `variables`: A `Variables` instance containing named constants.
    ///
    /// # Returns
    /// The `Builder` instance with the updated variables, allowing method chaining.
    ///
    /// # Examples
    /// ```rust
    /// use formulac::{Builder, Variables};
    /// use num_complex::Complex;
    ///
    /// let vars = Variables::from([("a", Complex::new(1.0, 0.0))]);
    /// let builder = Builder::new("a + x", &["x"])
    ///     .with_variables(vars);
    /// ```
    pub fn with_variables(mut self, variables: Variables) -> Self
    {
        for (key, val) in variables.iter() {
            self.vars.insert((key.as_str(), *val));
        }
        self
    }

    /// Sets the user-defined functions for the builder.
    ///
    /// User-defined functions allow you to extend the formula parser with custom operations.
    /// This method allows you to provide a pre-configured `UserDefinedTable`.
    ///
    /// # Parameters
    /// - `user_defined_functions`: A `UserDefinedTable` instance containing custom functions.
    ///
    /// # Returns
    /// The `Builder` instance with the updated user-defined functions, allowing method chaining.
    ///
    /// # Examples
    /// ```rust
    /// use formulac::{Builder, UserDefinedTable, UserDefinedFunction};
    /// use num_complex::Complex;
    ///
    /// let users = UserDefinedTable::default()
    ///     .register(UserDefinedFunction::new(
    ///         "double", |args| args[0] * Complex::new(2.0, 0.0), 1
    ///     )).unwrap();
    ///
    /// let builder = Builder::new("double(x)", &["x"]).with_user_defined_functions(users);
    /// ```
    pub fn with_user_defined_functions(mut self, user_defined_functions: UserDefinedTable) -> Self
    {
        for func in user_defined_functions.iter() {
            self.usrs.add(func.clone());
        }
        self
    }

    /// Compiles a mathematical expression into an executable closure.
    ///
    /// This function parses a formula string into an abstract syntax tree (AST),
    /// simplifies it, and then compiles it into a list of stack operations
    /// (in Reverse Polish Notation). The result is returned as a closure that
    /// can be called multiple times with different argument values without
    /// re-parsing the formula.
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
    /// use formulac::{Builder, Variables, UserDefinedTable};
    ///
    /// let vars = Variables::from([("a", Complex::new(3.0, 2.0))]);
    ///
    /// let expr = Builder::new("sin(z) + a * cos(z)", &["z"])
    ///     .with_variables(vars)
    ///     .compile()
    ///     .expect("Failed to compile formula");
    ///
    /// let result = expr(&[Complex::new(1.0, 2.0)]);
    /// println!("Result = {}", result);
    /// ```
    ///
    /// # Notes
    /// - This function does not evaluate immediately; instead, it produces
    ///   a reusable compiled closure for efficient repeated evaluation.
    pub fn compile(&self) -> Result<impl Fn(&[Complex<f64>]) -> Complex<f64> + Send + Sync + 'static, String>
    {
        let lexemes = lexer::from(&self.formula);
        let args: Vec<&str> = self.args.iter().map(|arg| arg.as_str() ).collect();
        let tokens = astnode::AstNode::from(&lexemes, &args, &self.vars, &self.usrs)?
            .simplify().compile();

        let expected_arity = args.len();
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
}

#[cfg(test)]
mod compile_test {
    use crate::variable::UserDefinedFunction;

    use super::*;
    use num_complex::{Complex};
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_constant_number() {
        let f = Builder::new("42", &[])
            .compile().unwrap();
        let result = f(&[]);
        assert_eq!(result, Complex::new(42.0, 0.0));
    }

    #[test]
    fn test_constant_str() {
        let f = Builder::new("PI", &[])
            .compile().unwrap();
        let result = f(&[]);
        assert_eq!(result, Complex::from(std::f64::consts::PI));
    }

    #[test]
    fn test_argument() {
        let f = Builder::new("x", &["x"])
            .compile().unwrap();
        let result = f(&[Complex::new(3.0, 0.0)]);
        assert_eq!(result, Complex::new(3.0, 0.0));
    }

    #[test]
    fn test_addition() {
        let f = Builder::new("x + y", &["x", "y"])
            .compile().unwrap();
        let x = Complex::new(2.0, 1.0);
        let y = Complex::new(3.0, 5.0);
        let result = f(&[x, y]);
        assert_abs_diff_eq!(result.re, (x + y).re, epsilon=1.0e-12);
        assert_abs_diff_eq!(result.im, (x + y).im, epsilon=1.0e-12);
    }

    #[test]
    fn test_nested_expression() {
        let f = Builder::new("sin(x + 1)", &["x"])
            .compile().unwrap();
        let result = f(&[Complex::new(0.0, 1.0)]);
        let expected = Complex::new(1.0, 1.0).sin();
        assert_abs_diff_eq!(result.re, expected.re, epsilon=1.0e-12);
        assert_abs_diff_eq!(result.im, expected.im, epsilon=1.0e-12);
    }

    #[test]
    fn test_binary_operator_precedence() {
        let f = Builder::new("2 + 3 * 4", &[])
            .compile().unwrap();
        let result = f(&[]);
        let expected = Complex::from(2.0 + 3.0 * 4.0);
        assert_abs_diff_eq!(result.re, expected.re, epsilon=1.0e-12);
        assert_abs_diff_eq!(result.im, expected.im, epsilon=1.0e-12);
    }

    #[test]
    fn test_function_with_two_args() {
        let f = Builder::new("pow(a, b)", &["a", "b"])
            .compile().unwrap();
        let a = Complex::new(2.0, 1.0);
        let b = Complex::new(-2.0, 3.0);
        let result = f(&[a, b]);
        let expected = a.powc(b);
        assert_abs_diff_eq!(result.re, expected.re, epsilon=1.0e-12);
        assert_abs_diff_eq!(result.im, expected.im, epsilon=1.0e-12);
    }

    #[test]
    fn test_differentiate_without_order() {
        let f = Builder::new("diff(x^2, x)", &["x"])
            .compile().unwrap();
        let x = Complex::new(2.0, 1.0);
        let result = f(&[x]);
        let expected = 2.0 * x;
        assert_abs_diff_eq!(result.re, expected.re, epsilon=1.0e-12);
        assert_abs_diff_eq!(result.im, expected.im, epsilon=1.0e-12);
    }

    #[test]
    fn test_differentiate_with_order() {
        let f = Builder::new("diff(x^3, x, 2)", &["x"])
            .compile().unwrap();
        let x = Complex::new(2.0, 1.0);
        let result = f(&[x]);
        let expected = 6.0 * x;
        assert_abs_diff_eq!(result.re, expected.re, epsilon=1.0e-12);
        assert_abs_diff_eq!(result.im, expected.im, epsilon=1.0e-12);
    }

    #[test]
    fn test_differentiate_with_userdefinedfunction() {
        // Define df(x) = 2*x
        let deriv = UserDefinedFunction::new(
            "df",
            |args: &[Complex<f64>]| Complex::new(2.0, 0.0) * args[0],
            1
        );

        // Define f(x) = x^2
        let func = UserDefinedFunction::new(
            "f",
            |args: &[Complex<f64>]| args[0] * args[0],
            1,
        ).with_derivative(vec![deriv]);
        let users = UserDefinedTable::default()
            .register(func).unwrap();

        let expr = Builder::new("diff(f(x), x)", &["x"])
            .with_user_defined_functions(users)
            .compile().unwrap();

        let result = expr(&[Complex::new(3.0, 0.0)]); // evaluates f'(3) = 6
        assert_abs_diff_eq!(result.re, 6.0, epsilon=1.0e-12);
        assert_abs_diff_eq!(result.im, 0.0, epsilon=1.0e-12);
    }

    #[test]
    fn test_differentiate_with_partial_derivative() {
        // Define a partial derivative w.r.t x: ∂g/∂x = 2*x*y
        let dg_dx = UserDefinedFunction::new(
            "dgdx",
            |args: &[Complex<f64>]| Complex::new(2.0, 0.0) * args[0] * args[1],
            2,
        );
        // Define a partial derivative w.r.t y: ∂g/∂y = x^2 + 3*y^2
        let dg_dy = UserDefinedFunction::new(
            "dgdy",
            |args: &[Complex<f64>]| args[0]*args[0] + Complex::new(3.0, 0.0)*args[1]*args[1],
            2,
        );

        // Define g(x, y) = x^2 * y + y^3
        let func = UserDefinedFunction::new(
            "g",
            |args: &[Complex<f64>]| args[0]*args[0]*args[1] + args[1]*args[1]*args[1],
            2,
        ).with_derivative(vec![dg_dx, dg_dy]);

        let users = UserDefinedTable::default()
            .register(func).unwrap();

        let x = Complex::new(2.0, 0.0);
        let y = Complex::new(3.0, 0.0);

        let expr_dx = Builder::new("diff(g(x, y), x)", &["x", "y"])
            .with_user_defined_functions(users.clone())
            .compile()
            .unwrap();
        let result_dx = expr_dx(&[x, y]);
        let expect_dx = 2.0 * x * y;
        assert_abs_diff_eq!(result_dx.re, expect_dx.re, epsilon=1.0e-12);
        assert_abs_diff_eq!(result_dx.im, expect_dx.im, epsilon=1.0e-12);

        let expr_dy = Builder::new("diff(g(x, y), y)", &["x", "y"])
            .with_user_defined_functions(users)
            .compile().unwrap();
        let result_dy = expr_dy(&[Complex::new(2.0, 0.0), Complex::new(3.0, 0.0)]);
        let expect_dy = x * x + 3.0 * y * y;
        assert_abs_diff_eq!(result_dy.re, expect_dy.re, epsilon=1.0e-12);
        assert_abs_diff_eq!(result_dy.im, expect_dy.im, epsilon=1.0e-12);
    }

    #[test]
    fn test_differentiate_with_userdefinedfunction_numerical() {
        // Define f(x) = x^2 without specifying derivative (uses numerical differentiation)
        let func = UserDefinedFunction::new(
            "f",
            |args: &[Complex<f64>]| args[0] * args[0],
            1,
        );
        let users = UserDefinedTable::default()
            .register(func).unwrap();

        let expr = Builder::new("diff(f(x), x)", &["x"])
            .with_user_defined_functions(users)
            .compile().unwrap();

        let x = Complex::new(3.0, 0.0);
        let result = expr(&[x]); // evaluates numerical derivative of f at x=3
        let expected = 2.0 * x; // analytical derivative: 2x
        // Allow some tolerance due to numerical differentiation
        assert_abs_diff_eq!(result.re, expected.re, epsilon=1e-5);
        assert_abs_diff_eq!(result.im, expected.im, epsilon=1e-5);
    }

    #[test]
    fn test_differentiate_with_userdefinedfunction_numerical_complex() {
        // Define f(z) = z^2 (complex)
        let func = UserDefinedFunction::new(
            "f",
            |args: &[Complex<f64>]| args[0] * args[0],
            1,
        );
        let users = UserDefinedTable::default()
            .register(func).unwrap();

        let expr = Builder::new("diff(f(z), z)", &["z"])
            .with_user_defined_functions(users)
            .compile().unwrap();

        let z = Complex::new(1.0, 2.0);
        let result = expr(&[z]);
        let expected = 2.0 * z; // d/dz (z^2) = 2z
        assert_abs_diff_eq!(result.re, expected.re, epsilon=1e-4);
        assert_abs_diff_eq!(result.im, expected.im, epsilon=1e-4);
    }

    #[test]
    #[should_panic]
    fn test_too_less_args_length() {
        let f = Builder::new("x + 1", &["x"])
            .compile().unwrap();
        f(&[]);
        // Too much arguments
        f(&[Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)]);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic]
    fn test_too_much_args_length() {
        let f = Builder::new("x + 1", &["x"])
            .compile().unwrap();
        f(&[Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)]);
    }

    #[cfg(not(debug_assertions))]
    #[test]
    fn test_too_much_args_length() {
        let f = Builder::new("x + 1", &["x"], &vars, &users)
            .compile().unwrap();
        let result = f(&[Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)]);
        assert_abs_diff_eq!(result.re, 2.0, epsilon=1.0e-12);
        assert_abs_diff_eq!(result.im, 0.0, epsilon=1.0e-12);
    }

    #[test]
    fn test_structure_lifetime() {
        let a = Complex::new(1.0, 2.0);
        let x = Complex::new(2.0, -1.0);
        let f = {
            let usrs = UserDefinedTable::default()
                .register(UserDefinedFunction::new(
                    "f", |args| args[0].conj(), 1
                )).unwrap();
            let vars = Variables::from([("a", a.clone())]);
            Builder::new("f(x + a)", &["x"])
                .with_variables(vars)
                .with_user_defined_functions(usrs)
                .compile().unwrap()
        };

        let result = f(&[x]);
        let expected = (x + a).conj();
        assert_abs_diff_eq!(result.re, expected.re, epsilon=1.0e-12);
        assert_abs_diff_eq!(result.im, expected.im, epsilon=1.0e-12);
    }
}

#[cfg(test)]
mod issue_test {
    use super::*;
    use num_complex::{Complex};
    use approx::assert_abs_diff_eq;

    #[test]
    /// It appears as if parenthesis are not effecting function call precedence in the way
    /// that the example code would have me believe. I.e f(x) + y is being parsed as f(x +y)
    /// # This issue was reported at v0.5.0, and resolved in v0.5.1
    fn test_issue_1() {
        let z = Complex::new(1.0, 3.0);

        let expr_1 = Builder::new("sin(z) + z",&["z"])
            .compile().expect("failed to compile formula");
        let result_1 = expr_1(&[z]);
        let expect_1 = z.sin() + z;

        assert_abs_diff_eq!(result_1.re, expect_1.re, epsilon=1.0e-12);
        assert_abs_diff_eq!(result_1.im, expect_1.im, epsilon=1.0e-12);

        let expr_2 = Builder::new("sin(z + z)",&["z"])
            .compile().expect("failed to compile formula");
        let result_2 = expr_2(&[z]);
        let expect_2 = (z+z).sin();
        assert_abs_diff_eq!(result_2.re, expect_2.re, epsilon=1.0e-12);
        assert_abs_diff_eq!(result_2.im, expect_2.im, epsilon=1.0e-12);

        let expr_3 = Builder::new("(sin(z)) + z",&["z"])
            .compile().expect("failed to compile formula");
        let result_3 = expr_3(&[z]);
        let expect_3 = (z.sin()) + z;
        assert_abs_diff_eq!(result_3.re, expect_3.re, epsilon=1.0e-12);
        assert_abs_diff_eq!(result_3.im, expect_3.im, epsilon=1.0e-12);
    }
}
