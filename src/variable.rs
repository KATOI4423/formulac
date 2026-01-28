//! # variables.rs
//!
//! This module provides structures and utilities for managing variables and user-defined functions
//! in mathematical expressions.
//!
//! ## Variables
//!
//! The `Variables` struct manages a collection of named variables.
//! Each variable is stored as a `Complex<f64>`, allowing both real and complex values.
//! It provides convenient methods to insert, retrieve, check existence, and clear variables.
//!
//! # Examples
//!
//! ```rust
//! use formulac::Variables;
//! use num_complex::Complex;
//!
//! // Register variables
//! let mut vars = Variables::from(&[("x", Complex::new(1.0, 0.0)), ("y", Complex::new(2.0, 3.0))]);
//!
//! // Check existence
//! assert!(vars.contains("x"));
//!
//! // Retrieve value
//! assert_eq!(*vars.get("y").unwrap(), Complex::new(2.0, 3.0));
//!
//! // Clear all variables
//! vars.clear();
//! assert!(!vars.contains("x"));
//! ```
//!
//! ## User-defined Functions
//!
//! The module also provides `UserDefinedFunction` and `UserDefinedTable` for dynamically
//! registering custom functions at runtime, which can be used in expression parsing and evaluation.
//!
//! # Examples
//!
//! ```rust
//! use num_complex::Complex;
//! use formulac::variable::FunctionCall;
//! use formulac::{UserDefinedTable, UserDefinedFunction};
//!
//! // Define and register a custom function
//! let func = UserDefinedFunction::new(
//!     "my_func",
//!     |args| args[0] + args[1],
//!     2,
//! );
//!
//! // Create a new user-defined table
//! let table = UserDefinedTable::default()
//!     .register(func).unwrap();
//!
//! // Retrieve and apply the function
//! if let Some(f) = table.get("my_func") {
//!     let result = f.apply(&[Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)]);
//!     assert_eq!(result, Complex::new(3.0, 0.0));
//! }
//! ```
//!
//! ## Derivatives of User-defined Functions
//!
//! A `UserDefinedFunction` can optionally be associated with a derivative function
//! by using [`with_derivative`](UserDefinedFunction::with_derivative).
//! The derivative must have the same arity as the original function
//! and will be returned by [`derivative`](UserDefinedFunction::derivative) if defined.
//!
//! # Examples
//!
//! ```rust
//! use num_complex::Complex;
//! use formulac::variable::FunctionCall;
//! use formulac::UserDefinedFunction;
//!
//! // Define f(x) = x^2 with derivative f'(x) = 2x
//! let deriv = UserDefinedFunction::new(
//!     "deriv",
//!     |args| Complex::new(2.0, 0.0) * args[0],
//!     1,
//! );
//! let func = UserDefinedFunction::new(
//!     "square",
//!     |args| args[0] * args[0],
//!     1,
//! ).with_derivative(vec![ deriv ]);
//!
//! // Apply its derivateive if available
//! if let Some(deriv) = func.derivative(0) {
//!     let dresult = deriv.apply(&[Complex::new(3.0, 0.0)]);
//!     assert_eq!(dresult, Complex::new(6.0, 0.0));
//! }
//! ```

use num_complex::Complex;
use std::collections::HashMap;
use std::sync::Arc;

/// A collection of named variables for expression evaluation.
///
/// `Variables` maintains a mapping from variable names (`String`) to values (`Complex<f64>`),
/// allowing expressions to reference these values by name during parsing or evaluation.
///
/// # Examples
///
/// ```rust
/// use formulac::Variables;
/// use num_complex::Complex;
///
/// let vars = Variables::from(&[("x", Complex::new(1.0, 0.0)), ("y", Complex::new(2.0, 3.0))]);
///
/// assert!(vars.contains("x"));
/// assert_eq!(*vars.get("y").unwrap(), Complex::new(2.0, 3.0));
/// ```
#[derive(Debug, PartialEq)]
pub struct Variables {
    table: HashMap<String, Complex<f64>>,
}

impl Variables {
    /// Creates an empty `Variables` table.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use formulac::Variables;
    ///
    /// let vars = Variables::new();
    /// assert!(vars.get("x").is_none());
    /// ```
    pub fn new() -> Self {
        Self {
            table: HashMap::new(),
        }
    }

    /// Constructs a `Variables` table from a slice of key-value pairs.
    ///
    /// Values can be any type convertible into `Complex<f64>`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use formulac::Variables;
    ///
    /// let vars = Variables::from(&[("a", 1.0), ("b", 2.0)]);
    /// assert!(vars.contains("a"));
    /// ```
    pub fn from<V>(items: &[(&str, V)]) -> Self
    where
        V: Clone,
        Complex<f64>: From<V>,
    {
        let mut vars = Self::new();
        for item in items.iter() {
            vars.insert(item.clone());
        }
        vars
    }

    /// Inserts multiple variables into the table.
    ///
    /// Each element of the slice is a tuple of variable name and value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use formulac::Variables;
    /// use num_complex::Complex;
    ///
    /// let mut vars = Variables::new();
    /// vars.insert(("x", Complex::new(1.0, 0.0)));
    /// ```
    pub fn insert<V>(&mut self, items: (&str, V))
    where
        V: Clone,
        Complex<f64>: From<V>,
    {
        self.table.insert(items.0.to_string(), Complex::from(items.1.clone()));
    }

    /// Checks if a variable with the given name exists in the table.
    ///
    /// Returns `true` if the variable exists, otherwise `false`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use formulac::Variables;
    ///
    /// let mut vars = Variables::default();
    /// vars.insert(("x", 1.0));
    /// assert!(vars.contains("x"));
    /// assert!(!vars.contains("y"));
    /// ```
    pub fn contains(&self, key: &str) -> bool {
        self.table.contains_key(key)
    }

    /// Retrieves a reference to the value of a variable by name.
    ///
    /// Returns `Some(&Complex<f64>)` if found, otherwise `None`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use formulac::Variables;
    /// use num_complex::Complex;
    ///
    /// let vars = formulac::vars!(("x", Complex::new(2.0, 3.0)));
    /// assert_eq!(*vars.get("x").unwrap(), Complex::new(2.0, 3.0));
    /// ```
    pub fn get(&self, key: &str) -> Option<&Complex<f64>> {
        self.table.get(key)
    }

    /// Clears all variables from the table.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use formulac::Variables;
    ///
    /// let mut vars = Variables::default();
    /// vars.insert(("x", 1.0));
    /// vars.clear();
    /// assert!(!vars.contains("x"));
    /// ```
    pub fn clear(&mut self) {
        self.table.clear();
    }
}

impl Default for Variables {
    fn default() -> Self {
        Self::new()
    }
}

/// Creates a Variables containing the arguments.
///
/// `vars!` allows Vecs to be defined with the same syntax as array expressions.
/// There are two forms of this macro:
///
/// - Create a Vec containing a given list of elements:
/// ```rust
/// use num_complex::Complex;
/// use formulac::{Variables, vars};
/// let vars = vars![
///     ("a", Complex::new(1.0, 0.0)),
///     ("b", Complex::new(3.2, 1.0))
/// ];
/// assert_eq!(vars.get("a"), Some(&Complex::new(1.0, 0.0)));
/// assert_eq!(vars.get("b"), Some(&Complex::new(3.2, 1.0)));
/// assert_eq!(vars.get("c"), None);
/// ```
#[macro_export]
macro_rules! vars {
    ( $( $x:expr), * ) => {
        {
            let mut vars = Variables::new();
            $( vars.insert($x); )*
            vars
        }
    };
}


/// The function type for user-defined functions.
type FuncType = dyn Fn(&[Complex<f64>]) -> Complex<f64> + Send + Sync;

/// Represents a user-defined function with a fixed arity.
///
/// `UserDefinedFunction` allows dynamic registration of custom functions
/// that can be invoked with an array of `Complex<f64>` arguments.
///
/// # Examples
///
/// ```rust
/// use num_complex::Complex;
/// use formulac::variable::FunctionCall;
/// use formulac::UserDefinedFunction;
///
/// // Define a function that adds 1 to the argument
/// let f = UserDefinedFunction::new(
///     "increment",
///     |args| args[0] + Complex::new(1.0, 0.0),
///     1,
/// );
///
/// // Apply the function
/// let result = f.apply(&[Complex::new(2.0, 0.0)]);
/// assert_eq!(result, Complex::new(3.0, 0.0));
///
/// // Check arity
/// assert_eq!(f.arity(), 1);
/// ```
#[derive(Clone)]
pub struct UserDefinedFunction {
    func: Arc<FuncType>,
    deriv: Vec<UserDefinedFunction>,
    arity: usize,
    name: String,
}

/// A trait representing a callable mathematical function.
///
/// This trait is implemented by types that can be called with a fixed number
/// of arguments and return a `Complex<f64>` result. It is used in the AST
/// for evaluating both built-in functions (like `sin`, `cos`, `pow`) and
/// user-defined functions.
///
/// # Methods
///
/// - `apply(&self, args: &[Complex<f64>]) -> Complex<f64>`
///   Evaluates the function with the given arguments. The length of `args`
///   must match the function's arity.
///
/// - `arity(&self) -> usize`
///   Returns the number of arguments the function expects.
pub trait FunctionCall {
    /// Evaluates the function with the given arguments.
    fn apply(&self, args: &[Complex<f64>]) -> Complex<f64>;

    /// Returns the number of arguments this function expects.
    fn arity(&self) -> usize;
}

impl UserDefinedFunction {
    /// Creates a new `UserDefinedFunction`.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the function.
    /// * `func` - A closure or function pointer that takes a slice of `Complex<f64>` and returns a `Complex<f64>`.
    /// * `arity` - The number of arguments the function expects.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use num_complex::Complex;
    /// use formulac::UserDefinedFunction;
    ///
    /// let func = UserDefinedFunction::new(
    ///     "my_func",
    ///     |args| args[0] + args[1],
    ///     2,
    /// );
    /// ```
    pub fn new<F>(name: & str, func: F, arity: usize) -> Self
    where
        F: Fn(&[Complex<f64>]) -> Complex<f64> + Send + Sync + 'static,
    {
        Self {
            func: Arc::new(func),
            deriv: Vec::new(),
            arity,
            name: name.to_string(),
        }
    }

    /// Sets the derivative function for this user-defined function.
    /// The derivative function should have the same signature as the main function.
    ///
    /// # Arguments
    /// * `diff` - A closure or function pointer that takes a slice of `Complex<f64>` and returns a `Complex<f64>`.
    ///
    /// # Examples
    /// ```rust
    /// use num_complex::Complex;
    /// use formulac::UserDefinedFunction;
    ///
    /// let deriv = UserDefinedFunction::new(
    ///     "deriv",
    ///     |args| Complex::new(2.0, 0.0) * args[0],
    ///     1,
    /// );
    ///
    /// let func = UserDefinedFunction::new(
    ///     "square",
    ///     |args| args[0] * args[0],
    ///     1,
    /// ).with_derivative(vec![ deriv ]);
    /// ```
    pub fn with_derivative(mut self, diffs: Vec<Self>) -> Self
    {
        debug_assert_eq!(diffs.len(), self.arity);
        self.deriv = diffs;
        self
    }

    /// Returns User-defined function name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns numerical derivative
    ///
    /// ## about dh
    /// dh is decided automatically by the value of variable: `args[var]` * 1.0e-6
    pub fn numeric_deriv(&self, var: usize) -> Option<Self>
    {
        if var > self.arity {
            return None;
        }

        let func = self.func.clone();
        let arity = self.arity;
        let name = format!("{}_num_diff_arg{}", self.name, var);
        Some(UserDefinedFunction::new(
            &name,
            move |args| {
                let dh = args[var] * 1.0e-6;
                let mut args_plus = args.to_vec();
                args_plus[var] += dh;
                let mut args_minus = args.to_vec();
                args_minus[var] -= dh;
                (func(&args_plus) - func(&args_minus)) / (dh * 2.0)
            },
            arity,
        ))
    }

    /// Returns a new `UserDefinedFunction` representing the derivative of this function, if defined.
    /// If no derivative function was set, returns `None`.
    ///
    /// # Examples
    /// ```rust
    /// use num_complex::Complex;
    /// use formulac::variable::FunctionCall;
    /// use formulac::UserDefinedFunction;
    ///
    /// let deriv = UserDefinedFunction::new(
    ///     "deriv",
    ///     |args| Complex::new(2.0, 0.0) * args[0],
    ///     1,
    /// );
    ///
    /// let func = UserDefinedFunction::new(
    ///     "square",
    ///     |args| args[0] * args[0],
    ///     1,
    /// ).with_derivative(vec![ deriv ]);
    ///
    /// if let Some(deriv) = func.derivative(0) {
    ///     let result = deriv.apply(&[Complex::new(3.0, 0.0)]);
    ///     assert_eq!(result, Complex::new(6.0, 0.0));
    /// }
    /// ```
    pub fn derivative(&self, var: usize) -> Option<&Self> {
        debug_assert!(var < self.arity, "var must be smaller than self.arity");
        self.deriv.get(var)
    }
}

impl FunctionCall for UserDefinedFunction {
    /// Applies the function to the given arguments.
    ///
    /// # Arguments
    ///
    /// * `args` - A slice of `Complex<f64>` representing the function's arguments.
    ///
    /// # Returns
    ///
    /// The result of the function as a `Complex<f64>`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use num_complex::Complex;
    /// use formulac::variable::FunctionCall;
    /// use formulac::UserDefinedFunction;
    ///
    /// let func = UserDefinedFunction::new(
    ///     "double",
    ///     |args| args[0] * Complex::new(2.0, 0.0),
    ///     1,
    /// );
    ///
    /// let result = func.apply(&[Complex::new(3.0, 0.0)]);
    /// assert_eq!(result, Complex::new(6.0, 0.0));
    /// ```
    fn apply(&self, args: &[Complex<f64>]) -> Complex<f64> {
        (self.func)(args)
    }

    /// Returns the arity (number of arguments) of the function.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use formulac::variable::FunctionCall;
    /// use formulac::UserDefinedFunction;
    /// use num_complex::Complex;
    ///
    /// let func = UserDefinedFunction::new("add", |args| args[0] + args[1], 2);
    /// assert_eq!(func.arity(), 2);
    /// ```
    fn arity(&self) -> usize {
        self.arity
    }
}

impl std::fmt::Debug for UserDefinedFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UserDefinedFunction")
            .field("name", &self.name)
            .field("arity", &self.arity)
            .finish_non_exhaustive()
    }
}

impl PartialEq for UserDefinedFunction {
    /// Checks equality based on `name` and `arity`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use formulac::UserDefinedFunction;
    /// use num_complex::Complex;
    ///
    /// let func1 = UserDefinedFunction::new("f", |args| args[0], 1);
    /// let func2 = UserDefinedFunction::new("f", |args| args[0] * Complex::new(2.0, 0.0), 1);
    /// let func3 = UserDefinedFunction::new("g", |args| args[0], 1);
    ///
    /// assert_eq!(func1, func2); // Same name and arity
    /// assert_ne!(func1, func3); // Different name
    /// ```
    fn eq(&self, other: &Self) -> bool {
        (self.arity == other.arity) && (self.name == other.name)
    }
}

/// A table that stores user-defined custom functions.
///
/// `UserDefinedTable` allows dynamic registration of user-defined functions,
/// enabling runtime extension of the formula parser with additional functionality.
///
/// # Examples
///
/// ```rust
/// use num_complex::Complex;
/// use formulac::{UserDefinedTable, UserDefinedFunction};
///
/// // Define a custom function
/// let my_func = UserDefinedFunction::new(
///     "my_func",
///     |args| (args[0] + Complex::new(1.0, 0.0)) / (args[0] - Complex::new(1.0, 0.0)),
///     1,
/// );
///
/// // Create a new user-defined table and register the custom function
/// let users = UserDefinedTable::default()
///     .register(my_func).unwrap();
///
/// // Retrieve and use the function
/// if let Some(func) = users.get("my_func") {
///     println!("Found function {:?}", func);
/// }
/// ```
///
/// # Notes
/// - It is the caller's responsibility to ensure that user-defined functions
///   do not conflict with built-in names.
#[derive(Clone)]
pub struct UserDefinedTable {
    table: HashMap<String, UserDefinedFunction>,
}

impl UserDefinedTable {
    /// Creates an empty `UserDefinedTable`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use formulac::UserDefinedTable;
    ///
    /// let table: UserDefinedTable = UserDefinedTable::new();
    /// assert!(table.get("any_func").is_none());
    /// ```
    pub fn new() -> Self {
        Self { table: HashMap::new() }
    }

    /// Adds a user-defined function to the table, overwriting any existing function with the same name.
    ///
    /// This method allows chaining multiple additions without consuming self.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use num_complex::Complex;
    /// use formulac::{UserDefinedTable, UserDefinedFunction};
    ///
    /// let mut table = UserDefinedTable::default();
    /// let func1 = UserDefinedFunction::new("f", |args| args[0] + Complex::ONE, 1);
    /// let func2 = UserDefinedFunction::new("g", |args| args[0] * Complex::new(2.0, 0.0), 1);
    ///
    /// table.add(func1);
    /// table.add(func2); // allows chaining multiple additions
    ///
    /// assert!(table.get("f").is_some());
    /// assert!(table.get("g").is_some());
    /// ```
    pub fn add(&mut self, func: UserDefinedFunction)
    {
        self.table.insert(func.name().to_string(), func);
    }

    /// Registers a new user-defined function under the given name.
    ///
    /// If a function with the same name already exists, it returns error.
    ///
    /// # Results
    ///  - Ok(Self): the func was registered successfully.
    ///  - Err(String): the func was already registered
    ///
    /// # Examples
    ///
    /// ```rust
    /// use num_complex::Complex;
    /// use formulac::{UserDefinedTable, UserDefinedFunction};
    ///
    /// let func = UserDefinedFunction::new(
    ///     "double",
    ///     |args| args[0] * Complex::new(2.0, 0.0),
    ///     1,
    /// );
    /// let table = UserDefinedTable::default()
    ///     .register(func).unwrap();
    /// assert!(table.get("double").is_some());
    /// ```
    pub fn register(mut self, func: UserDefinedFunction) -> Result<Self, String> {
        if self.table.contains_key(func.name()) {
            return Err(format!("{} is already registered", func.name()));
        }
        self.table.insert(func.name().to_string(), func);
        Ok(self)
    }

    /// Inserts a user-defined function into the table, overwriting any existing function with the same name.
    ///
    /// This method allows updating or replacing functions without error, unlike `register`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use num_complex::Complex;
    /// use formulac::{UserDefinedTable, UserDefinedFunction};
    /// use formulac::variable::FunctionCall; // to use f.apply()
    ///
    /// let func1 = UserDefinedFunction::new("my_func", |args| args[0] + Complex::new(1.0, 0.0), 1);
    /// let func2 = UserDefinedFunction::new("my_func", |args| args[0] * Complex::new(2.0, 0.0), 1);
    ///
    /// let table = UserDefinedTable::default()
    ///     .insert(func1)
    ///     .insert(func2); // !! overwrite func1 !!
    ///
    /// if let Some(f) = table.get("my_func") {
    ///     let result = f.apply(&[Complex::new(3.0, 0.0)]);
    ///     assert_eq!(result, Complex::new(6.0, 0.0)); // the result of func2
    /// }
    /// ```
    pub fn insert(mut self, func: UserDefinedFunction) -> Self {
        self.table.insert(func.name().to_string(), func);
        self
    }

    /// Retrieves a user-defined function by its name.
    ///
    /// Returns `Some(&UserDefinedFunction)` if a function with the given name exists,
    /// or `None` otherwise.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use num_complex::Complex;
    /// use formulac::{UserDefinedTable, UserDefinedFunction};
    /// use formulac::variable::FunctionCall;
    ///
    /// let func = UserDefinedFunction::new("square", |args| args[0] * args[0], 1);
    /// let table = UserDefinedTable::default()
    ///     .register(func).unwrap();
    ///
    /// if let Some(f) = table.get("square") {
    ///     let result = f.apply(&[Complex::new(3.0, 0.0)]);
    ///     assert_eq!(result, Complex::new(9.0, 0.0));
    /// }
    /// ```
    pub fn get(&self, name: &str) -> Option<&UserDefinedFunction> {
        self.table.get(name)
    }

    /// Removes a function from the table,
    /// returning the value at the function name if the function was previously in the table.
    ///
    /// # Example
    /// ```rust
    /// use formulac::{UserDefinedTable, UserDefinedFunction};
    ///
    /// let func = UserDefinedFunction::new("func", |args| args[0] * args[0], 1);
    /// let mut table = UserDefinedTable::default()
    ///     .insert(func.clone());
    /// assert_eq!(table.remove("func"), Some(func));
    /// assert_eq!(table.remove("func"), None);
    /// ```
    pub fn remove(&mut self, name: &str) -> Option<UserDefinedFunction>
    {
        self.table.remove(name)
    }

    /// Clears all user-defined functions from the table.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use num_complex::Complex;
    /// use formulac::{UserDefinedTable, UserDefinedFunction};
    ///
    /// let func = UserDefinedFunction::new("identity", |args| args[0], 1);
    /// let mut table = UserDefinedTable::default()
    ///     .register(func).unwrap();
    ///
    /// table.clear();
    /// assert!(table.get("identity").is_none());
    /// ```
    pub fn clear(&mut self) {
        self.table.clear();
    }
}

impl Default for UserDefinedTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod variables_tests {
    use super::*;

    #[test]
    fn test_insert_and_get() {
        let mut vars = Variables::new();
        vars.insert(("a", Complex::new(1.0, 0.0)));
        vars.insert(("b", Complex::new(2.0, 3.0)));

        assert_eq!(vars.get("a"), Some(&Complex::new(1.0, 0.0)));
        assert_eq!(vars.get("b"), Some(&Complex::new(2.0, 3.0)));
        assert_eq!(vars.get("c"), None);
    }

    #[test]
    fn test_contains() {
        let mut vars = Variables::new();
        vars.insert(("x", Complex::new(5.0, 0.0)));

        assert!(vars.contains("x"));
        assert!(!vars.contains("y"));
    }

    #[test]
    fn test_clear() {
        let mut vars = Variables::new();
        vars.insert(("foo", Complex::new(1.0, 0.0)));
        assert!(vars.contains("foo"));

        vars.clear();
        assert!(!vars.contains("foo"));
        assert_eq!(vars.get("foo"), None);
    }

    #[test]
    fn test_from_slice() {
        let items = &[("p", 3.0), ("q", 4.5)];
        let vars = Variables::from(items);

        assert_eq!(vars.get("p"), Some(&Complex::new(3.0, 0.0)));
        assert_eq!(vars.get("q"), Some(&Complex::new(4.5, 0.0)));
    }

    #[test]
    fn test_default() {
        let vars = Variables::default();
        assert!(!vars.contains("anything"));
    }

    #[test]
    fn test_variable_macro() {
        let vars = vars![("a", 1.0), ("b", 2.0)];
        let mut expect = Variables::new();
        expect.insert(("a", 1.0));
        expect.insert(("b", 2.0));

        assert_eq!(vars, expect);
    }
}

#[cfg(test)]
mod user_defined_function_tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use num_complex::{Complex};

    #[test]
    fn test_apply() {
        let func = UserDefinedFunction::new(
            "inc",
            |args: &[Complex<f64>]| args[0] + Complex::ONE,
            1,
        );

        let result = func.apply(&[Complex::ZERO]);
        assert_eq!(result, Complex::ONE);
    }

    #[test]
    fn test_arity() {
        let func = UserDefinedFunction::new(
            "sum",
            |args: &[Complex<f64>]| args[0] + args[1],
            2,
        );

        assert_eq!(func.arity(), 2);
    }

    #[test]
    fn test_debug() {
        let func = UserDefinedFunction::new(
            "mul",
            |args: &[Complex<f64>]| args[0] * args[0],
            1
        );
        let debug_str = format!("{:?}", func);
        assert!(debug_str.contains("name"));
        assert!(debug_str.contains("mul"));
        assert!(debug_str.contains("arity"));
        assert!(debug_str.contains("1"));
    }

    #[test]
    fn test_partial_eq() {
        let f1 = UserDefinedFunction::new("f", |args: &[Complex<f64>]| args[0], 1);
        let f2 = UserDefinedFunction::new("f", |args: &[Complex<f64>]| args[0] + args[0], 1);
        let f3 = UserDefinedFunction::new("g", |args: &[Complex<f64>]| args[0], 1);
        let f4 = UserDefinedFunction::new("f", |args: &[Complex<f64>]| args[0], 2);

        assert_eq!(f1, f2);
        assert_ne!(f1, f3);
        assert_ne!(f1, f4);
    }

    #[test]
    fn test_without_derivative() {
        let f = UserDefinedFunction::new("square", |args| args[0] * args[0], 1);
        assert!(f.derivative(0).is_none());
    }

    #[test]
    fn test_with_derivative() {
        let df = UserDefinedFunction::new("diff", |args: &[Complex<f64>]| Complex::new(2.0, 0.0) * args[0], 1);
        let f = UserDefinedFunction::new("square", |args| args[0] * args[0], 1)
            .with_derivative(vec![df]);

        let deriv = f.derivative(0).expect("derivative should exist");
        let deriv_result = deriv.apply(&[Complex::new(4.0, 0.0)]);
        assert_abs_diff_eq!(deriv_result.re, 8.0, epsilon=1.0e-12);
        assert_abs_diff_eq!(deriv_result.im, 0.0, epsilon=1.0e-12);

        assert!(format!("{:?}", deriv).contains("diff"));
    }
}

#[cfg(test)]
mod user_defined_table_tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use num_complex::{Complex, ComplexFloat};

    #[test]
    fn test_register_and_get() {
        let func = UserDefinedFunction::new(
            "my_func",
            |args| args[0] + Complex::new(1.0, 0.0),
            1,
        );
        let table = UserDefinedTable::default()
            .register(func).unwrap();

        let retrieved = table.get("my_func");
        assert!(retrieved.is_some());

        let result = retrieved.unwrap().apply(&[Complex::new(2.0, 0.0)]);
        assert_abs_diff_eq!(result.re(), 3.0, epsilon=1.0e-12);
        assert_abs_diff_eq!(result.im(), 0.0, epsilon=1.0e-12);
    }

    #[test]
    fn test_insert() {
        let func1 = UserDefinedFunction::new(
            "my_func",
            |args| args[0] + Complex::new(1.0, 0.0),
            1,
        );
        let func2 = UserDefinedFunction::new(
            "my_func",
            |args| args[0] - Complex::new(1.0, 0.0),
            1,
        );
        let table = UserDefinedTable::default()
            .insert(func1)
            .insert(func2);

        let retrieved = table.get("my_func");
        assert!(retrieved.is_some());

        let result = retrieved.unwrap().apply(&[Complex::new(2.0, 0.0)]);
        assert_abs_diff_eq!(result.re(), 1.0, epsilon=1.0e-12);
        assert_abs_diff_eq!(result.im(), 0.0, epsilon=1.0e-12);
    }

    #[test]
    fn test_name_conflict() {
        let func1 = UserDefinedFunction::new(
            "func",
            |args| args[0] + Complex::new(1.0, 0.0),
            1,
        );
        let func2 = UserDefinedFunction::new(
            "func",
            |_args| Complex::ZERO,
            0,
        );

        let table = UserDefinedTable::default()
            .register(func1).unwrap()
            .register(func2);

        assert!(table.is_err());
    }

    #[test]
    fn test_clear() {
        let func = UserDefinedFunction::new(
            "func",
            |args| args[0],
            1,
        );

        let mut table = UserDefinedTable::default()
            .register(func).unwrap();
        assert!(table.get("func").is_some());

        table.clear();
        assert!(table.get("func").is_none());
    }

    #[test]
    fn test_default() {
        let table = UserDefinedTable::default();
        assert!(table.get("nonexistent").is_none());
    }
}
