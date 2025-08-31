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
//! use formulac::variable::Variables;
//! use num_complex::Complex;
//!
//! let mut vars = Variables::default();
//!
//! // Insert variables
//! vars.insert(&[("x", Complex::new(1.0, 0.0)), ("y", Complex::new(2.0, 3.0))]);
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
//! use formulac::variable::{UserDefinedTable, UserDefinedFunction};
//!
//! // Create a new user-defined table
//! let mut table = UserDefinedTable::default();
//!
//! // Define and register a custom function
//! let func = UserDefinedFunction::new(
//!     "my_func",
//!     |args| args[0] + args[1],
//!     2,
//! );
//! table.register("my_func", func);
//!
//! // Retrieve and apply the function
//! if let Some(f) = table.get("my_func") {
//!     let result = f.apply(&[Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)]);
//!     assert_eq!(result, Complex::new(3.0, 0.0));
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
/// use formulac::variable::Variables;
/// use num_complex::Complex;
///
/// let mut vars = Variables::default();
/// vars.insert(&[("x", Complex::new(1.0, 0.0)), ("y", Complex::new(2.0, 3.0))]);
///
/// assert!(vars.contains("x"));
/// assert_eq!(*vars.get("y").unwrap(), Complex::new(2.0, 3.0));
/// ```
#[derive(Debug)]
pub struct Variables {
    table: HashMap<String, Complex<f64>>,
}

impl Variables {
    /// Creates an empty `Variables` table.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use formulac::variable::Variables;
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
    /// use formulac::variable::Variables;
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
        vars.insert(items);
        vars
    }

    /// Inserts multiple variables into the table.
    ///
    /// Each element of the slice is a tuple of variable name and value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use formulac::variable::Variables;
    /// use num_complex::Complex;
    ///
    /// let mut vars = Variables::default();
    /// vars.insert(&[("x", Complex::new(1.0, 0.0)), ("y", Complex::new(2.0, 3.0))]);
    /// ```
    pub fn insert<V>(&mut self, items: &[(&str, V)])
    where
        V: Clone,
        Complex<f64>: From<V>,
    {
        for (key, val) in items {
            self.table.insert(key.to_string(), Complex::from(val.clone()));
        }
    }

    /// Checks if a variable with the given name exists in the table.
    ///
    /// Returns `true` if the variable exists, otherwise `false`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use formulac::variable::Variables;
    ///
    /// let mut vars = Variables::default();
    /// vars.insert(&[("x", 1.0)]);
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
    /// use formulac::variable::Variables;
    /// use num_complex::Complex;
    ///
    /// let mut vars = Variables::default();
    /// vars.insert(&[("x", Complex::new(2.0, 3.0))]);
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
    /// use formulac::variable::Variables;
    ///
    /// let mut vars = Variables::default();
    /// vars.insert(&[("x", 1.0)]);
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


/// Represents a user-defined function with a fixed arity.
///
/// `UserDefinedFunction` allows dynamic registration of custom functions
/// that can be invoked with an array of `Complex<f64>` arguments.
///
/// # Examples
///
/// ```rust
/// use num_complex::Complex;
/// use formulac::variable::UserDefinedFunction;
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
pub struct UserDefinedFunction<'a> {
    func: Arc<dyn Fn(&[Complex<f64>]) -> Complex<f64> + Send + Sync>,
    arity: usize,
    name: &'a str,
}

impl<'a> UserDefinedFunction<'a> {
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
    /// use formulac::variable::UserDefinedFunction;
    ///
    /// let func = UserDefinedFunction::new(
    ///     "my_func",
    ///     |args| args[0] + args[1],
    ///     2,
    /// );
    /// ```
    pub fn new<F>(name: &'a str, func: F, arity: usize) -> Self
    where
        F: Fn(&[Complex<f64>]) -> Complex<f64> + Send + Sync + 'static,
    {
        Self {
            func: Arc::new(func),
            arity,
            name,
        }
    }

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
    /// use formulac::variable::UserDefinedFunction;
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
    pub fn apply(&self, args: &[Complex<f64>]) -> Complex<f64> {
        (self.func)(args)
    }

    /// Returns the arity (number of arguments) of the function.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use formulac::variable::UserDefinedFunction;
    /// use num_complex::Complex;
    ///
    /// let func = UserDefinedFunction::new("add", |args| args[0] + args[1], 2);
    /// assert_eq!(func.arity(), 2);
    /// ```
    pub fn arity(&self) -> usize {
        self.arity
    }
}

impl<'a> std::fmt::Debug for UserDefinedFunction<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UserDefinedFunction")
            .field("name", &self.name)
            .field("arity", &self.arity)
            .finish_non_exhaustive()
    }
}

impl<'a> PartialEq for UserDefinedFunction<'a> {
    /// Checks equality based on `name` and `arity`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use formulac::variable::UserDefinedFunction;
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
/// use formulac::variable::{UserDefinedTable, UserDefinedFunction};
///
/// // Create a new user-defined table
/// let mut users = UserDefinedTable::default();
///
/// // Define a custom function
/// let my_func = UserDefinedFunction::new(
///     "my_func",
///     |args| (args[0] + Complex::new(1.0, 0.0)) / (args[0] - Complex::new(1.0, 0.0)),
///     1,
/// );
///
/// // Register the custom function
/// users.register("my_func", my_func);
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
pub struct UserDefinedTable<'a> {
    table: HashMap<String, UserDefinedFunction<'a>>,
}

impl<'a> UserDefinedTable<'a> {
    /// Creates an empty `UserDefinedTable`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use formulac::variable::UserDefinedTable;
    ///
    /// let table: UserDefinedTable = UserDefinedTable::new();
    /// assert!(table.get("any_func").is_none());
    /// ```
    pub fn new() -> Self {
        Self { table: HashMap::new() }
    }

    /// Registers a new user-defined function under the given name.
    ///
    /// If a function with the same name already exists, it is replaced.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use num_complex::Complex;
    /// use formulac::variable::{UserDefinedTable, UserDefinedFunction};
    ///
    /// let mut table = UserDefinedTable::new();
    /// let func = UserDefinedFunction::new(
    ///     "double",
    ///     |args| args[0] * Complex::new(2.0, 0.0),
    ///     1,
    /// );
    /// table.register("double", func);
    /// assert!(table.get("double").is_some());
    /// ```
    pub fn register(&mut self, name: &str, func: UserDefinedFunction<'a>) {
        self.table.insert(name.to_string(), func);
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
    /// use formulac::variable::{UserDefinedTable, UserDefinedFunction};
    ///
    /// let mut table = UserDefinedTable::new();
    /// let func = UserDefinedFunction::new("square", |args| args[0] * args[0], 1);
    /// table.register("square", func);
    ///
    /// if let Some(f) = table.get("square") {
    ///     let result = f.apply(&[Complex::new(3.0, 0.0)]);
    ///     assert_eq!(result, Complex::new(9.0, 0.0));
    /// }
    /// ```
    pub fn get(&self, name: &str) -> Option<&UserDefinedFunction<'a>> {
        self.table.get(name)
    }

    /// Clears all user-defined functions from the table.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use num_complex::Complex;
    /// use formulac::variable::{UserDefinedTable, UserDefinedFunction};
    ///
    /// let mut table = UserDefinedTable::new();
    /// let func = UserDefinedFunction::new("identity", |args| args[0], 1);
    /// table.register("identity", func);
    ///
    /// table.clear();
    /// assert!(table.get("identity").is_none());
    /// ```
    pub fn clear(&mut self) {
        self.table.clear();
    }
}

impl<'a> Default for UserDefinedTable<'a> {
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
        vars.insert(&[("a", Complex::new(1.0, 0.0)), ("b", Complex::new(2.0, 3.0))]);

        assert_eq!(vars.get("a"), Some(&Complex::new(1.0, 0.0)));
        assert_eq!(vars.get("b"), Some(&Complex::new(2.0, 3.0)));
        assert_eq!(vars.get("c"), None);
    }

    #[test]
    fn test_contains() {
        let mut vars = Variables::new();
        vars.insert(&[("x", Complex::new(5.0, 0.0))]);

        assert!(vars.contains("x"));
        assert!(!vars.contains("y"));
    }

    #[test]
    fn test_clear() {
        let mut vars = Variables::new();
        vars.insert(&[("foo", Complex::new(1.0, 0.0))]);
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
}

#[cfg(test)]
mod user_defined_function_tests {
    use super::*;
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
}

#[cfg(test)]
mod user_defined_table_tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use num_complex::{Complex, ComplexFloat};

    #[test]
    fn test_register_and_get() {
        let mut table = UserDefinedTable::new();

        let func = UserDefinedFunction::new(
            "my_func",
            |args| args[0] + Complex::new(1.0, 0.0),
            1,
        );
        table.register("my_func", func);

        let retrieved = table.get("my_func");
        assert!(retrieved.is_some());

        let result = retrieved.unwrap().apply(&[Complex::new(2.0, 0.0)]);
        assert_abs_diff_eq!(result.re(), 3.0, epsilon=1.0e-12);
        assert_abs_diff_eq!(result.im(), 0.0, epsilon=1.0e-12);
    }

    #[test]
    fn test_overwrite() {
        let mut table = UserDefinedTable::new();

        let func1 = UserDefinedFunction::new(
            "func1",
            |args| args[0] + Complex::new(1.0, 0.0),
            1,
        );
        let func2 = UserDefinedFunction::new(
            "func2",
            |_args| Complex::ZERO,
            0,
        );

        table.register("func", func1);
        table.register("func", func2);

        let retrieved = table.get("func").unwrap();
        let result = retrieved.apply(&[Complex::ONE]);
        assert_eq!(result, Complex::ZERO);
    }

    #[test]
    fn test_clear() {
        let mut table = UserDefinedTable::new();

        let func = UserDefinedFunction::new(
            "func",
            |args| args[0],
            1,
        );

        table.register("f", func);
        assert!(table.get("f").is_some());

        table.clear();
        assert!(table.get("f").is_none());
    }

    #[test]
    fn test_default() {
        let table = UserDefinedTable::default();
        assert!(table.get("nonexistent").is_none());
    }
}
