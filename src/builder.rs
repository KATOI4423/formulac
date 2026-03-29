//! # builder.rs
//!
//! This module provides structures and utilities for building function object.

use num_complex::Complex;

use crate::astnode;
use crate::constants::Constants;
use crate::err::ParseError;
use crate::functions::{
    FunctionArgs,
    FunctionCall,
    UserFn,
};
use crate::lexer;
use crate::token::{
    Token,
    UserFnTable,
};


pub struct Builder<const N: usize>
{
    formula: String,
    args: [String; N],
    constants: Constants,
    usrs: UserFnTable,
}

impl<const N: usize> Builder<N>
{
    /// Creates a new `Builder` instance with the given formula and argument names.
    ///
    /// This is the starting point for building a compiled mathematical expression.
    /// You can chain methods like `with_constants` and `with_user_functions`
    /// to configure the builder before calling `compile`.
    ///
    /// # Parameters
    /// - `formula`: A string slice containing the mathematical expression to compile.
    /// - `arg_names`: A slice of argument names (`&str`) that the formula depends on.
    ///   These will be used as placeholders for input values in the compiled closure.
    ///
    /// # Returns
    /// A new `Builder` instance with default constants and user-defined functions.
    ///
    /// # Examples
    /// ```rust
    /// use formulac::builder::Builder;
    /// use num_complex::Complex;
    ///
    /// let builder = Builder::new("x + 1", ["x"]);
    /// let func = builder.compile()
    ///     .expect("Failed to compile 'x + 1'");
    /// println!("{} + 1 = {}", 3, func([Complex::new(3.0, 0.0)]));
    /// ```
    pub fn new(formula: &str, arg_names: [&str; N]) -> Self
    {
        Self {
            formula: formula.to_string(),
            args: arg_names.map(|arg| arg.to_string()),
            constants: Constants::default(),
            usrs: UserFnTable::new(),
        }
    }

    /// Sets the constants for the builder.
    ///
    /// Constants can be referenced in the formula by name.
    /// This method allows you to provide a pre-configured `Constants` table.
    ///
    /// # Parameters
    /// - `constants`: A `Constants` instance containing named constants.
    ///
    /// # Returns
    /// The `Builder` instance with the updated constants, allowing method chaining.
    ///
    /// # Examples
    /// ```rust
    /// use formulac::builder::Builder;
    /// use num_complex::Complex;
    ///
    /// let builder = Builder::new("a + x", ["x"])
    ///     .with_constants([
    ///         ("a", Complex::new(1.0, 0.0)),
    ///         ("b", Complex::new(-1.0, 2.5))
    ///     ]);
    /// ```
    pub fn with_constants<I, S, V>(mut self, constants: I) -> Self
    where
        I: IntoIterator<Item = (S, V)>,
        String: From<S>,
        Complex<f64>: From<V>,
    {
        for (key, value) in constants {
            self.constants.insert(key, value);
        }
        self
    }

    /// Sets the user-defined functions for the builder.
    ///
    /// User-defined functions allow you to extend the formula parser with custom operations.
    /// This method allows you to provide a pre-configured list of `UserFn`.
    ///
    /// # Parameters
    /// - `user_functions`: A list of `UserFn` instance containing custom functions.
    ///
    /// # Returns
    /// The `Builder` instance with the updated user-defined functions, allowing method chaining.
    ///
    /// # Examples
    /// ```rust
    /// use formulac::builder::Builder;
    /// use formulac::functions::{FunctionArgs, UserFn};
    /// use num_complex::Complex;
    ///
    /// let func = UserFn::new("double", |[x]| x * Complex::new(2.0, 0.0));
    ///
    /// let builder = Builder::new("double(x)", ["x"])
    ///     .with_user_functions([func]);
    /// ```
    pub fn with_user_functions<I>(mut self, user_functions: I) -> Self
    where
        I: IntoIterator<Item = UserFn>,
    {
        for func in user_functions.into_iter() {
            self.usrs.insert(func.name().into(), func);
        }
        self
    }

    fn build_tokens(&self) -> Result<Vec<Token>, ParseError>
    {
        let lexemes = lexer::from(&self.formula);
        let args: Vec<&str> = self.args.iter().map(|arg| arg.as_str()).collect();
        let tokens = astnode::AstNode::from(&lexemes, &args, &self.constants, &self.usrs)?
            .simplify()
            .compile();

        Ok(tokens)
    }

    fn build_executor(tokens: Vec<Token>)
        -> impl Fn([Complex<f64>; N]) -> Complex<f64> + Send + Sync + 'static
    {
        move |arg_values: [Complex<f64>; N]| {
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
                        for _ in 0..n {
                            args.push(stack.pop().unwrap())
                        }
                        args.reverse();
                        stack.push(func.apply(FunctionArgs::from(args)));
                    },
                    Token::UserFunction(func) => {
                        let n = func.arity();
                        let mut args: Vec<Complex<f64>> = Vec::with_capacity(n);
                        args.resize(n, Complex::new(0.0, 0.0));

                        for i in (0..n).rev() {
                            args[i] = stack.pop().unwrap();
                        }
                        stack.push(func.apply(FunctionArgs::from(args)));
                    },
                    _ => unreachable!("Invalid tokens found: use compiled tokens"),
                }
            }

            stack.pop().unwrap_or_else(|| unreachable!("empty stack at end"))
        }
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
    /// Fn([Complex<f64>]) -> Complex<f64>
    /// ```
    ///
    /// - The closure takes a slice of complex argument values corresponding to `arg_names`.
    /// - Returns `Complex<f64>` if evaluation succeeds.
    ///
    /// On failure, returns an error enum describing the parsing or compilation error.
    ///
    /// # Example
    /// ```rust
    /// use num_complex::Complex;
    /// use formulac::builder::Builder;
    ///
    /// let expr = Builder::new("sin(z) + a * cos(z)", ["z"])
    ///     .with_constants([("a", Complex::new(3.0, 2.0))])
    ///     .compile()
    ///     .expect("Failed to compile formula");
    ///
    /// let result = expr([Complex::new(1.0, 2.0)]);
    /// println!("Result = {}", result);
    /// ```
    ///
    /// # Notes
    /// - This function does not evaluate immediately; instead, it produces
    ///   a reusable compiled closure for efficient repeated evaluation.
    pub fn compile(&self) -> Result<impl Fn([Complex<f64>; N]) -> Complex<f64> + Send + Sync + 'static, ParseError>
    {
        let tokens = self.build_tokens()?;

        Ok(Self::build_executor(tokens))
    }
}

#[cfg(test)]
mod compile_test {
    use crate::functions::{
        UserFn,
    };

    use super::*;
    use num_complex::{Complex};
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_constant_number() {
        let f = Builder::new("42", [])
            .compile().unwrap();
        let result = f([]);
        assert_eq!(result, Complex::new(42.0, 0.0));
    }

    #[test]
    fn test_constant_str() {
        let f = Builder::new("PI", [])
            .compile().unwrap();
        let result = f([]);
        assert_eq!(result, Complex::from(std::f64::consts::PI));
    }

    #[test]
    fn test_argument() {
        let f = Builder::new("x", ["x"])
            .compile().unwrap();
        let result = f([Complex::new(3.0, 0.0)]);
        assert_eq!(result, Complex::new(3.0, 0.0));
    }

    #[test]
    fn test_addition() {
        let f = Builder::new("x + y", ["x", "y"])
            .compile().unwrap();
        let x = Complex::new(2.0, 1.0);
        let y = Complex::new(3.0, 5.0);
        let result = f([x, y]);
        assert_abs_diff_eq!(result.re, (x + y).re, epsilon=1.0e-12);
        assert_abs_diff_eq!(result.im, (x + y).im, epsilon=1.0e-12);
    }

    #[test]
    fn test_nested_expression() {
        let f = Builder::new("sin(x + 1)", ["x"])
            .compile().unwrap();
        let result = f([Complex::new(0.0, 1.0)]);
        let expected = Complex::new(1.0, 1.0).sin();
        assert_abs_diff_eq!(result.re, expected.re, epsilon=1.0e-12);
        assert_abs_diff_eq!(result.im, expected.im, epsilon=1.0e-12);
    }

    #[test]
    fn test_binary_operator_precedence() {
        let f = Builder::new("2 + 3 * 4", [])
            .compile().unwrap();
        let result = f([]);
        let expected = Complex::from(2.0 + 3.0 * 4.0);
        assert_abs_diff_eq!(result.re, expected.re, epsilon=1.0e-12);
        assert_abs_diff_eq!(result.im, expected.im, epsilon=1.0e-12);
    }

    #[test]
    fn test_function_with_two_args() {
        let f = Builder::new("pow(a, b)", ["a", "b"])
            .compile().unwrap();
        let a = Complex::new(2.0, 1.0);
        let b = Complex::new(-2.0, 3.0);
        let result = f([a, b]);
        let expected = a.powc(b);
        assert_abs_diff_eq!(result.re, expected.re, epsilon=1.0e-12);
        assert_abs_diff_eq!(result.im, expected.im, epsilon=1.0e-12);
    }

    #[test]
    fn test_differentiate_without_order() {
        let f = Builder::new("diff(x^2, x)", ["x"])
            .compile().unwrap();
        let x = Complex::new(2.0, 1.0);
        let result = f([x]);
        let expected = 2.0 * x;
        assert_abs_diff_eq!(result.re, expected.re, epsilon=1.0e-12);
        assert_abs_diff_eq!(result.im, expected.im, epsilon=1.0e-12);
    }

    #[test]
    fn test_differentiate_with_order() {
        let f = Builder::new("diff(x^3, x, 2)", ["x"])
            .compile().unwrap();
        let x = Complex::new(2.0, 1.0);
        let result = f([x]);
        let expected = 6.0 * x;
        assert_abs_diff_eq!(result.re, expected.re, epsilon=1.0e-12);
        assert_abs_diff_eq!(result.im, expected.im, epsilon=1.0e-12);
    }

    #[test]
    fn test_differentiate_with_userfn() {
        // Define df(x) = 2*x
        let deriv = UserFn::new("df", |[x]| Complex::new(2.0, 0.0) * x);

        // Define f(x) = x^2
        let func = UserFn::new("f", |[x]| x * x)
            .with_derivative(vec![deriv]);

        let expr = Builder::new("diff(f(x), x)", ["x"])
            .with_user_functions([func])
            .compile().unwrap();

        let result = expr([Complex::new(3.0, 0.0)]); // evaluates f'(3) = 6
        assert_abs_diff_eq!(result.re, 6.0, epsilon=1.0e-12);
        assert_abs_diff_eq!(result.im, 0.0, epsilon=1.0e-12);
    }

    #[test]
    fn test_differentiate_with_partial_derivative() {
        // Define a partial derivative w.r.t x: ∂g/∂x = 2*x*y
        let dg_dx = UserFn::new("dgdx", |[x, y]| Complex::new(2.0, 0.0) * x * y);
        // Define a partial derivative w.r.t y: ∂g/∂y = x^2 + 3*y^2
        let dg_dy = UserFn::new("dgdy", |[x, y]| x * x + Complex::new(3.0, 0.0) * y * y);

        // Define g(x, y) = x^2 * y + y^3
        let func = UserFn::new("g", |[x, y]| x * x * y + y * y * y)
            .with_derivative(vec![dg_dx, dg_dy]);

        let x = Complex::new(2.0, 0.0);
        let y = Complex::new(3.0, 0.0);

        let expr_dx = Builder::new("diff(g(x, y), x)", ["x", "y"])
            .with_user_functions([func.clone()])
            .compile()
            .unwrap();
        let result_dx = expr_dx([x, y]);
        let expect_dx = 2.0 * x * y;
        assert_abs_diff_eq!(result_dx.re, expect_dx.re, epsilon=1.0e-12);
        assert_abs_diff_eq!(result_dx.im, expect_dx.im, epsilon=1.0e-12);

        let expr_dy = Builder::new("diff(g(x, y), y)", ["x", "y"])
            .with_user_functions([func.clone()])
            .compile().unwrap();
        let result_dy = expr_dy([Complex::new(2.0, 0.0), Complex::new(3.0, 0.0)]);
        let expect_dy = x * x + 3.0 * y * y;
        assert_abs_diff_eq!(result_dy.re, expect_dy.re, epsilon=1.0e-12);
        assert_abs_diff_eq!(result_dy.im, expect_dy.im, epsilon=1.0e-12);
    }

    #[test]
    fn test_differentiate_undefined() {
        let func = UserFn::new("f", |[x]| x);
        assert!(Builder::new("diff(f(x), x)", ["x"]).with_user_functions([func]).compile().is_err());
    }

    #[test]
    fn test_structure_lifetime() {
        let a = Complex::new(1.0, 2.0);
        let x = Complex::new(2.0, -1.0);
        let f = {
            let constants = [("a", a.clone())];
            Builder::new("f(x + a)",  ["x"])
                .with_constants(constants)
                .with_user_functions([
                    UserFn::new("f", |[x]| x.conj()),
                ])
                .compile().unwrap()
        };

        let result = f([x]);
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

        let expr_1 = Builder::new("sin(z) + z", ["z"])
            .compile().expect("failed to compile formula");
        let result_1 = expr_1([z]);
        let expect_1 = z.sin() + z;

        assert_abs_diff_eq!(result_1.re, expect_1.re, epsilon=1.0e-12);
        assert_abs_diff_eq!(result_1.im, expect_1.im, epsilon=1.0e-12);

        let expr_2 = Builder::new("sin(z + z)", ["z"])
            .compile().expect("failed to compile formula");
        let result_2 = expr_2([z]);
        let expect_2 = (z+z).sin();
        assert_abs_diff_eq!(result_2.re, expect_2.re, epsilon=1.0e-12);
        assert_abs_diff_eq!(result_2.im, expect_2.im, epsilon=1.0e-12);

        let expr_3 = Builder::new("(sin(z)) + z", ["z"])
            .compile().expect("failed to compile formula");
        let result_3 = expr_3([z]);
        let expect_3 = (z.sin()) + z;
        assert_abs_diff_eq!(result_3.re, expect_3.re, epsilon=1.0e-12);
        assert_abs_diff_eq!(result_3.im, expect_3.im, epsilon=1.0e-12);
    }
}
