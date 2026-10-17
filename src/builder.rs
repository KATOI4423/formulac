//! # builder.rs
//!
//! This module provides structures and utilities for building function object.

use num_complex::Complex;
use std::ops::{
    AddAssign,
    MulAssign,
};
use std::str::FromStr;
use std::sync::Arc;

use crate::astnode::AstNode;
use crate::astnode::compile::{
    Instruction,
    Program,
};
use crate::core::Real;
use crate::constants::Constants;
use crate::err::ParseError;
use crate::functions::UserFn;
use crate::lexer;
use crate::token::{
    UserFnTable,
};

#[derive(Debug, Clone)]
/// Configures and compiles a mathematical expression.
///
/// A `Builder` stores the expression text, the names of its arguments,
/// user-provided constants, and user-defined functions. The configuration is
/// consumed by [`Builder::compile`] or one of the derivative compilation
/// methods to produce a reusable [`CompiledFormula`].
///
/// `N` is the number of arguments accepted by the resulting compiled formula.
/// Argument values are supplied to [`CompiledFormula::eval`] in the same order
/// as the `arg_names` passed to [`Builder::new`].
///
/// # Type Parameters
///
/// * `T` - The real scalar type used for the real and imaginary components of
///   complex values.
/// * `N` - The number of named arguments in the formula.
pub struct Builder<T: Real, const N: usize>
{
    formula: String,
    args: [String; N],
    constants: Constants<T>,
    usrs: UserFnTable<T>,
}

impl<T: Real, const N: usize> Builder<T, N>
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
    /// println!("{} + 1 = {}", 3, func.eval(&[Complex::new(3.0, 0.0)]));
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
    /// let builder = Builder::<f64, _>::new("a + x", ["x"])
    ///     .with_constants([
    ///         ("a", Complex::new(1.0, 0.0)),
    ///         ("b", Complex::new(-1.0, 2.5))
    ///     ]);
    /// ```
    pub fn with_constants<I, S, V>(mut self, constants: I) -> Self
    where
        I: IntoIterator<Item = (S, V)>,
        String: From<S>,
        Complex<T>: From<V>,
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
    /// use formulac::functions::UserFn;
    /// use num_complex::Complex;
    ///
    /// let func = UserFn::<f64>::new("double", |[x]| x * Complex::new(2.0, 0.0));
    ///
    /// let builder = Builder::<f64, _>::new("double(x)", ["x"])
    ///     .with_user_functions([func]);
    /// ```
    pub fn with_user_functions<I>(mut self, user_functions: I) -> Self
    where
        I: IntoIterator<Item = UserFn<T>>,
    {
        for func in user_functions.into_iter() {
            self.usrs.insert(func.name().into(), func);
        }
        self
    }

    fn build_tokens(&self) -> Result<Program<T>, ParseError>
    where
        T: FromStr,
        Complex<T>: AddAssign + MulAssign,
    {
        let tokens = self.build_astnode()?
            .compile();
        Ok(tokens)
    }

    fn build_astnode(&self) -> Result<AstNode<T>, ParseError>
    where
        T: FromStr,
        Complex<T>: AddAssign + MulAssign,
    {
        let lexemes = lexer::from(&self.formula);
        let args: Vec<&str> = self.args.iter().map(|arg| arg.as_str()).collect();
        let astnode = AstNode::from(&lexemes, &args, &self.constants, &self.usrs)?
            .simplify();
        Ok(astnode)
    }

    /// Compiles a mathematical expression into an executable structure.
    ///
    /// This function parses a formula string into an abstract syntax tree (AST),
    /// simplifies it, and then compiles it into a list of stack operations
    /// (in Reverse Polish Notation). The result is returned as a structure that
    /// can be called multiple times with different argument values without
    /// re-parsing the formula with `eval()`.
    ///
    /// # Returns
    /// On success, returns a structure: CompiledFormula:
    ///
    /// - The structure takes a slice of complex argument values corresponding to `arg_names`.
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
    /// let result = expr.eval(&[Complex::new(1.0, 2.0)]);
    /// println!("Result = {}", result);
    /// ```
    ///
    /// # Notes
    /// - This function does not evaluate immediately; instead, it produces
    ///   a reusable compiled structure for efficient repeated evaluation.
    pub fn compile(&self) -> Result<CompiledFormula<T, N>, ParseError>
    where
        T: FromStr + Send + Sync + 'static,
        Complex<T>: AddAssign + MulAssign,
    {
        let program = self.build_tokens()?;

        Ok(CompiledFormula::from_program(program))
    }

    fn get_argument_index(&self, variable: impl AsRef<str>) -> Option<usize>
    {
        self.args.iter().enumerate()
            .find(|(_, str)| *str == variable.as_ref())
            .map(|(i, _)| i)
    }

    /// Compiles the expression and its symbolic derivative with respect to a named argument.
    ///
    /// This returns a tuple containing:
    /// 1. the compiled original expression, and
    /// 2. the compiled derivative with respect to `variable`.
    ///
    /// The derivative is generated by differentiating the AST before compilation,
    /// so both structures are produced from the same parsed expression.
    ///
    /// # Parameters
    ///
    /// - `variable`: the argument name to differentiate by.
    ///
    /// # Returns
    ///
    /// Returns `Ok((expression, derivative))` on success.
    ///
    /// # Errors
    ///
    /// Returns `ParseError::InvalidDerivative` when the specified `variable`
    /// is not present in the builder's argument list.
    ///
    /// # Example
    ///
    /// ```rust
    /// use formulac::builder::Builder;
    /// use num_complex::Complex;
    ///
    /// let (f, df) = Builder::new("sin(x) + y", ["x", "y"])
    ///     .compile_with_derivative("x")
    ///     .unwrap();
    ///
    /// let x = Complex::new(1.0, -1.0);
    /// let y = Complex::new(2.0, 0.0);
    ///
    /// assert_eq!(f.eval(&[x, y]), x.sin() + y);
    /// assert_eq!(df.eval(&[x, y]), x.cos());
    /// ```
    pub fn compile_with_derivative(&self, variable: impl AsRef<str>)
    -> Result<
        (CompiledFormula<T, N>, CompiledFormula<T, N>),
        ParseError,
    >
    where
        T: FromStr + Send + Sync + 'static,
        Complex<T>: AddAssign + MulAssign,
    {
        let variable = variable.as_ref();
        let idx = self.get_argument_index(variable)
            .ok_or(ParseError::InvalidDerivative {
                span: lexer::Span::from(0..0), // dummy span
                reason: format!("Unknown argument string {}", variable),
            })?;

        let astnode = self.build_astnode()?;
        let program = astnode.clone().compile();
        let derive_program = astnode.differentiate(idx)?.simplify().compile();

        Ok((CompiledFormula::from_program(program), CompiledFormula::from_program(derive_program)))
    }

    /// Compiles the original expression together with all partial derivatives.
    ///
    /// The returned tuple contains:
    /// 1. the compiled original expression, and
    /// 2. a vector of compiled partial derivatives in the same order as `arg_names`.
    ///
    /// # Returns
    ///
    /// Returns `Ok((expression, partials))` on success.
    ///
    /// # Errors
    ///
    /// Returns `ParseError` if parsing, simplification, compilation,
    /// or differentiation fails for any argument.
    ///
    /// # Example
    ///
    /// ```rust
    /// use formulac::builder::Builder;
    /// use num_complex::Complex;
    ///
    /// let (f, partials) = Builder::new("x * y + z", ["x", "y", "z"])
    ///     .compile_with_all_partials()
    ///     .unwrap();
    ///
    /// let x = Complex::new(1.0, -1.0);
    /// let y = Complex::new(2.0, 0.0);
    /// let z = Complex::new(3.0, 2.0);
    ///
    /// let df_dx = &partials[0]; // ∂/∂x
    /// let df_dy = &partials[1]; // ∂/∂y
    /// let df_dz = &partials[2]; // ∂/∂z
    ///
    /// assert_eq!(f.eval(&[x, y, z]), x * y + z);
    /// assert_eq!(df_dx.eval(&[x, y, z]), y);
    /// assert_eq!(df_dy.eval(&[x, y, z]), x);
    /// assert_eq!(df_dz.eval(&[x, y, z]), Complex::new(1.0, 0.0));
    /// ```
    pub fn compile_with_all_partials(
        &self,
    ) -> Result<(
        CompiledFormula<T, N>,
        Vec<CompiledFormula<T, N>>,
    ), ParseError>
    where
        T: FromStr + Send + Sync + 'static,
        Complex<T>: AddAssign + MulAssign,
    {
        let astnode = self.build_astnode()?;
        let original_program = astnode.clone().compile();

        let partials = (0..N)
            .map(|idx| {
                let derived_program = astnode.clone().differentiate(idx)?.simplify().compile();
                Ok(CompiledFormula::from_program(derived_program))
            })
            .collect::<Result<Vec<_>, ParseError>>()?;

        Ok((CompiledFormula::from_program(original_program), partials))
    }
}

/// Reusable working buffer for [`CompiledFormula::eval_with_scratch`].
///
/// Evaluating a formula uses a stack of intermediate values. Passing a `Scratch`
/// lets repeated evaluations reuse a single allocation instead of allocating a new
/// stack on every call.
///
/// # Usage
/// - Use **one `Scratch` per thread.** It is mutable state and is taken as `&mut`,
///   so one instance cannot be used from two threads at once. It is [`Send`] when
///   `T: Send`, so moving it to another thread is fine.
/// - Its contents are discarded at the start of every evaluation. No reset is
///   needed, and one `Scratch` may be used with different formulas (it grows on
///   demand).
///
/// Create one with [`CompiledFormula::new_scratch`], which sizes it for that
/// formula, or with [`Scratch::new`].
#[derive(Debug, Clone)]
pub struct Scratch<T: Real> {
    stack: Vec<Complex<T>>,
}

impl<T: Real> Scratch<T> {
    /// Creates a scratch buffer with room for `capacity` intermediate values.
    ///
    /// A buffer that is too small is safe: it grows on demand during evaluation.
    /// Prefer [`CompiledFormula::new_scratch`] to get the right size automatically.
    pub(crate) fn new(size: usize) -> Self {
        Self {
            stack: Vec::with_capacity(size),
        }
    }
}

/// A compiled formula, ready for repeated evaluation.
///
/// Created by [`Builder::compile`], [`Builder::compile_with_derivative`] and
/// [`Builder::compile_with_all_partials`]. `N` is the number of arguments given to
/// [`Builder::new`].
///
/// # Evaluation
/// - [`eval`](Self::eval): the simplest form; allocates a working stack per call.
/// - [`eval_with_scratch`](Self::eval_with_scratch): reuses a [`Scratch`], so no
///   stack is allocated per call. Preferred for repeated or multi-threaded use.
///
/// # Sharing and threads
/// A `CompiledFormula` is immutable. Cloning is cheap: clones share the same
/// compiled program through an [`Arc`]. It is `Send + Sync` when `T` is, so one
/// instance can be shared by reference between threads. The only mutable state is
/// the [`Scratch`], of which each thread should own one.
///
/// # Examples
/// ```rust
/// use formulac::Builder;
/// use num_complex::Complex;
///
/// let f = Builder::<f64, 1>::new("x * x + 1", ["x"]).compile().unwrap();
///
/// std::thread::scope(|s| {
///     for t in 0..4 {
///         let f = &f;
///         s.spawn(move || {
///             let mut scratch = f.new_scratch(); // one per thread
///             let x = Complex::new(t as f64, 0.0);
///             assert_eq!(f.eval_with_scratch(&[x], &mut scratch), f.eval(&[x]));
///         });
///     }
/// });
/// ```
#[derive(Debug, Clone)]
pub struct CompiledFormula<T: Real, const N: usize> {
    program: Arc<Program<T>>,
}

impl<T: Real, const N: usize> CompiledFormula<T, N> {
    fn from_program(program: Program<T>) -> Self {
        Self {
            program: Arc::new(program),
        }
    }

    /// Creates a [`Scratch`] sized for this formula.
    ///
    /// Equivalent to `Scratch::new` with this formula's maximum stack depth, so the
    /// first call to [`eval_with_scratch`](Self::eval_with_scratch) does not allocate.
    /// Create one per thread.
    pub fn new_scratch(&self) -> Scratch<T> {
        Scratch::new(self.program.stack_size())
    }

    /// Evaluates the formula for the given arguments.
    ///
    /// A working stack is allocated on every call. When the same formula is evaluated
    /// many times, prefer [`eval_with_scratch`](Self::eval_with_scratch).
    ///
    /// # Parameters
    /// - `args`: argument values, in the order given to [`Builder::new`](crate::builder::Builder::new).
    ///
    /// # Returns
    /// The computed value. No error is returned: domain errors such as division by zero
    /// follow the semantics of `T` (`NaN` / `inf` for `f64`).
    ///
    /// # Panics
    /// Only if a user-defined function panics; the panic is propagated unchanged.
    ///
    /// # Examples
    /// ```rust
    /// use formulac::Builder;
    /// use num_complex::Complex;
    ///
    /// let f = Builder::<f64, 1>::new("x * x + 1", ["x"]).compile().unwrap();
    /// assert_eq!(f.eval(&[Complex::new(3.0, 0.0)]), Complex::new(10.0, 0.0));
    /// ```
    pub fn eval(&self, args: &[Complex<T>; N]) -> Complex<T> {
        self.eval_with_scratch(args, &mut Scratch::new(self.program.stack_size()))
    }

    /// Evaluates the formula, reusing `scratch` as the working stack.
    ///
    /// The result is the same as [`eval`](Self::eval), but no stack is allocated per call
    /// once `scratch` is large enough. This is the preferred form for repeated or
    /// multi-threaded evaluation.
    ///
    /// # Parameters
    /// - `args`: argument values, taken by value (see [`eval`](Self::eval)).
    /// - `scratch`: working buffer. Its previous contents are discarded at the start of
    ///   every call, so it needs no reset and may be reused across different formulas
    ///   (it grows if necessary). Its contents after the call are unspecified.
    ///
    /// # Threading
    /// A `CompiledFormula` can be shared between threads freely; a [`Scratch`] is mutable
    /// state, so use **one per thread**. `&mut` makes concurrent use of a single
    /// `Scratch` impossible at compile time.
    ///
    /// # Returns
    /// The computed value (see [`eval`](Self::eval) for error semantics).
    ///
    /// # Panics
    /// Only if a user-defined function panics. The `scratch` stays usable afterwards.
    ///
    /// # Examples
    /// ```rust
    /// use formulac::Builder;
    /// use num_complex::Complex;
    ///
    /// let f = Builder::<f64, 1>::new("x * x + 1", ["x"]).compile().unwrap();
    /// let mut scratch = f.new_scratch();
    /// for i in 0..3 {
    ///     let x = Complex::new(i as f64, 0.0);
    ///     assert_eq!(f.eval_with_scratch(&[x], &mut scratch), f.eval(&[x]));
    /// }
    /// ```
    pub fn eval_with_scratch(&self, args: &[Complex<T>; N], scratch: &mut Scratch<T>) -> Complex<T> {
        scratch.stack.reserve(self.program.stack_size());
        scratch.stack.clear();

        for instruction in self.program.code().iter() {
            match instruction {
                Instruction::Constant(value) => scratch.stack.push(value.clone()),
                Instruction::Argument(index) => scratch.stack.push(args[*index].clone()),
                Instruction::UnaryOperator(kind) => {
                    let expr = scratch.stack.pop().unwrap();
                    scratch.stack.push(kind.apply(expr));
                },
                Instruction::BinaryOperator(kind) => {
                    let r = scratch.stack.pop().unwrap();
                    let l = scratch.stack.pop().unwrap();
                    scratch.stack.push(kind.apply(l, r));
                },
                Instruction::Function(kind) => {
                    kind.apply_stack(&mut scratch.stack);
                },
                Instruction::PowiConst(n) => {
                    let x = scratch.stack.pop().unwrap();
                    scratch.stack.push(x.powi(*n));
                },
                Instruction::UserFunction(func) => {
                    func.apply_stack(&mut scratch.stack);
                },
            }
        }

        scratch.stack.pop().unwrap_or_else(|| unreachable!("empty stack at end"))
    }
}


#[cfg(test)]
mod compile_test {
    use crate::functions::{
        Arity,
        UserFn,
    };
    use crate::operators::BinaryOperatorKind;

    use super::*;
    use num_complex::{Complex};
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_constant_number() {
        let f = Builder::new("42", [])
            .compile().unwrap();
        let result = f.eval(&[]);
        assert_eq!(result, Complex::new(42.0, 0.0));
    }

    #[test]
    fn test_constant_str() {
        let f = Builder::new("PI", [])
            .compile().unwrap();
        let result = f.eval(&[]);
        assert_eq!(result, Complex::from(std::f64::consts::PI));
    }

    #[test]
    fn test_argument() {
        let f = Builder::new("x", ["x"])
            .compile().unwrap();
        let result = f.eval(&[Complex::new(3.0, 0.0)]);
        assert_eq!(result, Complex::new(3.0, 0.0));
    }

    #[test]
    fn test_addition() {
        let f = Builder::new("x + y", ["x", "y"])
            .compile().unwrap();
        let x = Complex::new(2.0, 1.0);
        let y = Complex::new(3.0, 5.0);
        let result = f.eval(&[x, y]);
        assert_abs_diff_eq!(result.re, (x + y).re, epsilon=1.0e-12);
        assert_abs_diff_eq!(result.im, (x + y).im, epsilon=1.0e-12);
    }

    #[test]
    fn test_nested_expression() {
        let f = Builder::new("sin(x + 1)", ["x"])
            .compile().unwrap();
        let result = f.eval(&[Complex::new(0.0, 1.0)]);
        let expected = Complex::new(1.0, 1.0).sin();
        assert_abs_diff_eq!(result.re, expected.re, epsilon=1.0e-12);
        assert_abs_diff_eq!(result.im, expected.im, epsilon=1.0e-12);
    }

    #[test]
    fn test_binary_operator_precedence() {
        let f = Builder::<f64, _>::new("2 + 3 * 4", [])
            .compile().unwrap();
        let result = f.eval(&[]);
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
        let result = f.eval(&[a, b]);
        let expected = a.powc(b);
        assert_abs_diff_eq!(result.re, expected.re, epsilon=1.0e-12);
        assert_abs_diff_eq!(result.im, expected.im, epsilon=1.0e-12);
    }

    #[test]
    fn test_differentiate_without_order() {
        let f = Builder::new("diff(x^2, x)", ["x"])
            .compile().unwrap();
        let x = Complex::new(2.0, 1.0);
        let result = f.eval(&[x]);
        let expected = 2.0 * x;
        assert_abs_diff_eq!(result.re, expected.re, epsilon=1.0e-12);
        assert_abs_diff_eq!(result.im, expected.im, epsilon=1.0e-12);
    }

    #[test]
    fn test_differentiate_with_order() {
        let f = Builder::new("diff(x^3, x, 2)", ["x"])
            .compile().unwrap();
        let x = Complex::new(2.0, 1.0);
        let result = f.eval(&[x]);
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
            .with_derivative(vec![deriv]).unwrap();

        let expr = Builder::new("diff(f(x), x)", ["x"])
            .with_user_functions([func])
            .compile().unwrap();

        let result = expr.eval(&[Complex::new(3.0, 0.0)]); // evaluates f'(3) = 6
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
            .with_derivative(vec![dg_dx, dg_dy]).unwrap();

        let x = Complex::new(2.0, 0.0);
        let y = Complex::new(3.0, 0.0);

        let expr_dx = Builder::new("diff(g(x, y), x)", ["x", "y"])
            .with_user_functions([func.clone()])
            .compile()
            .unwrap();
        let result_dx = expr_dx.eval(&[x, y]);
        let expect_dx = 2.0 * x * y;
        assert_abs_diff_eq!(result_dx.re, expect_dx.re, epsilon=1.0e-12);
        assert_abs_diff_eq!(result_dx.im, expect_dx.im, epsilon=1.0e-12);

        let expr_dy = Builder::new("diff(g(x, y), y)", ["x", "y"])
            .with_user_functions([func.clone()])
            .compile().unwrap();
        let result_dy = expr_dy.eval(&[Complex::new(2.0, 0.0), Complex::new(3.0, 0.0)]);
        let expect_dy = x * x + 3.0 * y * y;
        assert_abs_diff_eq!(result_dy.re, expect_dy.re, epsilon=1.0e-12);
        assert_abs_diff_eq!(result_dy.im, expect_dy.im, epsilon=1.0e-12);
    }

    #[test]
    fn test_differentiate_undefined() {
        let func = UserFn::<f64>::new("f", |[x]| x);
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

        let result = f.eval(&[x]);
        let expected = (x + a).conj();
        assert_abs_diff_eq!(result.re, expected.re, epsilon=1.0e-12);
        assert_abs_diff_eq!(result.im, expected.im, epsilon=1.0e-12);
    }

    #[test]
    fn test_different_args_num() {
        let x = Complex::new(1.0, -1.0);
        let y = Complex::new(2.0, 0.0);

        let func = Builder::new("f(x) + g(x, y)", ["x", "y"])
            .with_user_functions([
                UserFn::new("f", |[x]| x),
                UserFn::new("g", |[x, y]| x + y),
            ])
            .compile()
            .unwrap();

        let result = func.eval(&[x, y]);
        let expected = (x) + (x + y);
        assert_abs_diff_eq!(result.re, expected.re, epsilon=1.0e-12);
        assert_abs_diff_eq!(result.im, expected.im, epsilon=1.0e-12);
    }

    fn assert_send_sync<X: Send + Sync>() {}

    #[test]
    fn compiled_formula_and_scratch_are_thread_safe() {
        assert_send_sync::<CompiledFormula<f64, 1>>();
        assert_send_sync::<Scratch<f64>>();
    }

    fn simulate_max_depth<T: Real>(code: &[Instruction<T>]) -> usize {
        let (mut cur, mut max) = (0usize, 0usize);
        for t in code {
            match t {
                Instruction::Constant(..) | Instruction::Argument(..) => cur += 1,
                Instruction::UnaryOperator(..) => {}
                Instruction::BinaryOperator(..) => cur -= 1,
                Instruction::Function(kind) => cur = cur - kind.arity() + 1,
                Instruction::PowiConst(..) => {}, // pop 1, push 1 -> net 0
                Instruction::UserFunction(func) => cur = cur - func.arity() + 1,
            }
            max = max.max(cur);
        }
        assert_eq!(cur, 1, "final stack depth must be 1");
        max
    }

    #[test]
    fn stack_size_matches_simulation() {
        for f in ["x", "x + y", "x+y*x-y/(x+1)", "pow(x, y + sin(x*y))",
                  "sin(cos(sin(x)))", "diff(x^3 + sin(x*y), x)", "(x+y)*(x-y)*(x+1)*(y+2)",
                  "x^3", "x^3 + y^2 * x", "powi(x, y)", "powi(x + y, 3) * powi(x, y)",
        ] {
            let p = Builder::<f64, 2>::new(f, ["x", "y"]).build_tokens().unwrap();
            assert_eq!(p.stack_size(), simulate_max_depth(p.code()), "formula: {f}");
        }
    }

    #[test]
    fn powi_const_eval_matches() {
        let f = Builder::new("x^3 + powi(x, -2)", ["x"]).compile().unwrap();
        let x = Complex::new(1.2, -0.7);
        let ret = f.eval(&[x]);
        let expexted = x * x * x + (x * x).inv();
        assert_abs_diff_eq!(ret.re, expexted.re);
        assert_abs_diff_eq!(ret.im, expexted.im);
    }

    #[test]
    fn scratch_reuse_and_undersized_scratch() {
        let f = Builder::new("sin(x) * y + x", ["x", "y"]).compile().unwrap();
        let g = Builder::new("x + 1", ["x", "y"]).compile().unwrap();
        let a = [Complex::new(0.3, 0.4), Complex::new(-1.0, 2.0)];

        let mut s = Scratch::new(0);
        let expected = f.eval(&a);
        for _ in 0..10 {
            assert_eq!(f.eval_with_scratch(&a, &mut s), expected);
            assert_eq!(g.eval_with_scratch(&a, &mut s), g.eval(&a));
        }
    }

    #[test]
    fn shared_formula_with_per_thread_scratch() {
        let f = Builder::new("sin(x) + cos(x)*cos(x) + x*x", ["x"]).compile().unwrap();
        let xs: Vec<_> = (0..64).map(|i| Complex::new(i as f64 * 0.1, 0.5)).collect();
        let expected: Vec<_> = xs.iter().map(|&x| f.eval(&[x])).collect();

        std::thread::scope(|s| {
            for _ in 0..8 {
                s.spawn(|| {
                    let mut scratch = Scratch::new(0);
                    for (x, e) in xs.iter().zip(&expected) {
                        assert_eq!(f.eval_with_scratch(&[*x], &mut scratch), *e);
                    }
                });
            }
        });
    }

    #[test]
    fn scratch_is_usable_after_user_fn_panic() {
        let boom = UserFn::<f64>::new("boom", |[x]| { if x.re > 100.0 { panic!("boom") } x });
        let f = Builder::new("x + boom(x)", ["x"]).with_user_functions([boom]).compile().unwrap();
        let mut s = Scratch::new(0);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            f.eval_with_scratch(&[Complex::new(1000.0, 0.0)], &mut s)
        }));
        assert!(r.is_err());
        assert_eq!(f.eval_with_scratch(&[Complex::new(1.0, 0.0)], &mut s), Complex::new(2.0, 0.0));
    }

    #[test]
    fn sub_is_lowered_without_multiplication_by_minus_one() {
        let x = Complex::new(0.7, 0.1);
        let y = Complex::new(1.3, -0.4);
        let cases: [(&str, usize, Complex<f64>); 4] = [
            ("x - y", 1, x - y),
            ("1 - x", 1, Complex::new(1.0, 0.0) - x),
            ("x - y * x", 1, x - y * x),
            ("x - y - x * y", 2, x - y - x * y),
        ];
        for (formula, subs, expected) in cases {
            let p = Builder::<f64, 2>::new(formula, ["x", "y"]).build_tokens().unwrap();
            let count = p.code().iter()
                .filter(|i| matches!(i, Instruction::BinaryOperator(BinaryOperatorKind::Sub)))
                .count();
            assert_eq!(count, subs, "formula: {formula}");
            // No `* -1` must remain.
            assert!(!p.code().iter().any(|i| matches!(i, Instruction::Constant(c) if *c == Complex::new(-1.0, 0.0))));
            let f = Builder::<f64, 2>::new(formula, ["x", "y"]).compile().unwrap();
            let r = f.eval(&[x, y]);
            assert_abs_diff_eq!(r.re, expected.re, epsilon = 1.0e-12);
            assert_abs_diff_eq!(r.im, expected.im, epsilon = 1.0e-12);
        }
    }

    /// Compares a compiled 2-argument formula against a closed-form reference.
    fn assert_matches(formula: &str, reference: impl Fn(Complex<f64>, Complex<f64>) -> Complex<f64>) {
        let consts: Vec<(String, Complex<f64>)> = (0..=10)
            .map(|k| (format!("a{k}"), Complex::new(1.0 + 0.3 * k as f64, 0.5 - 0.1 * k as f64)))
            .collect();
        let f = Builder::<f64, 2>::new(formula, ["x", "y"])
            .with_constants(consts)
            .compile()
            .unwrap();
        for (x, y) in [
            (Complex::new(0.7, 0.1), Complex::new(-0.3, 0.9)),
            (Complex::new(-1.2, 0.8), Complex::new(0.5, -0.5)),
            (Complex::new(0.3, -0.6), Complex::new(1.0, 0.0)),
        ] {
            let r = f.eval(&[x, y]);
            let e = reference(x, y);
            assert_abs_diff_eq!(r.re, e.re, epsilon = 1.0e-9);
            assert_abs_diff_eq!(r.im, e.im, epsilon = 1.0e-9);
        }
    }

    #[test]
    fn unary_minus_binds_tighter_than_add_and_mul() {
        let one = Complex::new(1.0, 0.0);
        assert_matches("-x + y", |x, y| -x + y);
        assert_matches("-x * y + 1", move |x, y| -x * y + one);
        assert_matches("x - -y", |x, y| x - -y);
        assert_matches("-x^2 + y", |x, y| -(x * x) + y);      // -x^2 is -(x^2)
        assert_matches("x^-1 + y", |x, y| x.powi(-1) + y);
        assert_matches("2^-x * y", |x, y| Complex::new(2.0, 0.0).powc(-x) * y);
    }

    fn a(k: usize) -> Complex<f64> {
        Complex::new(1.0 + 0.3 * k as f64, 0.5 - 0.1 * k as f64)
    }

    #[test]
    fn horner_dense_polynomial_uses_n_multiplications() {
        let consts: Vec<(String, Complex<f64>)> =
            (0..=5).map(|k| (format!("a{k}"), a(k))).collect();
        let p = Builder::<f64, 1>::new("a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4 + a5*x^5", ["x"])
            .with_constants(consts)
            .build_tokens()
            .unwrap();
        let muls = p.code().iter()
            .filter(|i| matches!(i, Instruction::BinaryOperator(BinaryOperatorKind::Mul)))
            .count();
        assert_eq!(muls, 5);
        assert!(!p.code().iter().any(|i| matches!(i, Instruction::PowiConst(_))));
    }

    #[test]
    fn horner_dense_and_sparse_values() {
        assert_matches("a0 + a1*x + a2*x^2 + a3*x^3", |x, _| a(0) + a(1)*x + a(2)*x*x + a(3)*x*x*x);
        assert_matches("a0 + a2*x^2 + a5*x^5 + a10*x^10",
            |x, _| a(0) + a(2)*x.powi(2) + a(5)*x.powi(5) + a(10)*x.powi(10));
        assert_matches("a1*x + a3*x^3", |x, _| a(1)*x + a(3)*x.powi(3));       // no constant term
    }

    #[test]
    fn horner_coefficients_one_and_minus_one() {
        assert_matches("x^3 + x^2 + x + 1", |x, _| x.powi(3) + x.powi(2) + x + Complex::new(1.0, 0.0));
        assert_matches("1 - x + x^2 - x^3", |x, _| Complex::new(1.0, 0.0) - x + x.powi(2) - x.powi(3));
        assert_matches("-x^3 + a1*x", |x, _| -x.powi(3) + a(1)*x);
    }

    #[test]
    fn horner_leaves_non_polynomial_terms() {
        assert_matches("x^2 + x + sin(x)", |x, _| x.powi(2) + x + x.sin());
        assert_matches("x^-1 + x + x^2", |x, _| x.powi(-1) + x + x.powi(2));   // negative exponent
        assert_matches("x^2.5 + x + x^2", |x, _| x.powc(Complex::new(2.5, 0.0)) + x + x.powi(2));
    }

    #[test]
    fn horner_multi_variable_and_symbolic_coefficients() {
        assert_matches("x^2*y + x*y^2 + 1", |x, y| x*x*y + x*y*y + Complex::new(1.0, 0.0));
        assert_matches("y*x^2 + a1*x + a0", |x, y| y*x*x + a(1)*x + a(0));
        assert_matches("x^2*y + x^2*a1 + x", |x, y| x*x*y + x*x*a(1) + x);     // same degree, merged
    }

    #[test]
    fn horner_nested_and_derivative() {
        assert_matches("sin(x^2 + a1*x + a0)", |x, _| (x*x + a(1)*x + a(0)).sin());
        assert_matches("diff(x^10 + x^5 + x^2, x)",
            |x, _| Complex::new(10.0, 0.0)*x.powi(9) + Complex::new(5.0, 0.0)*x.powi(4) + Complex::new(2.0, 0.0)*x);
    }

    #[test]
    fn horner_keeps_stack_size_consistent() {
        for f in ["x^3 + y^2 * x", "a0 + a2*x^2 + a5*x^5", "1 - x + x^2 - x^3"] {
            let p = Builder::<f64, 2>::new(f, ["x", "y"])
                .with_constants((0..=5).map(|k| (format!("a{k}"), a(k))))
                .build_tokens().unwrap();
            assert_eq!(p.stack_size(), simulate_max_depth(p.code()), "formula: {f}");
        }
    }

    #[test]
    fn horner_at_zero() {
        let f = Builder::new("a0 + a2 * x^2 + a5 * x^5", ["x"])
            .with_constants((0..=5).map(|k| (format!("a{k}"), a(k))))
            .compile().unwrap();
        assert_eq!(f.eval(&[Complex::new(0.0, 0.0)]), a(0));
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
        let result_1 = expr_1.eval(&[z]);
        let expect_1 = z.sin() + z;

        assert_abs_diff_eq!(result_1.re, expect_1.re, epsilon=1.0e-12);
        assert_abs_diff_eq!(result_1.im, expect_1.im, epsilon=1.0e-12);

        let expr_2 = Builder::new("sin(z + z)", ["z"])
            .compile().expect("failed to compile formula");
        let result_2 = expr_2.eval(&[z]);
        let expect_2 = (z+z).sin();
        assert_abs_diff_eq!(result_2.re, expect_2.re, epsilon=1.0e-12);
        assert_abs_diff_eq!(result_2.im, expect_2.im, epsilon=1.0e-12);

        let expr_3 = Builder::new("(sin(z)) + z", ["z"])
            .compile().expect("failed to compile formula");
        let result_3 = expr_3.eval(&[z]);
        let expect_3 = (z.sin()) + z;
        assert_abs_diff_eq!(result_3.re, expect_3.re, epsilon=1.0e-12);
        assert_abs_diff_eq!(result_3.im, expect_3.im, epsilon=1.0e-12);
    }
}
