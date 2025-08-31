//! # variables.rs
//!
//! This module provides the `Variables` struct, which manages a table of named
//! variables used in mathematical expressions.
//!
//! Each variable is stored as a `Complex<f64>`, allowing
//! both real and complex values to be represented. The module provides convenient
//! methods to insert, retrieve, check, and clear variables.

use num_complex::Complex;
use std::collections::HashMap;
use std::sync::Arc;

/// A collection of named variables for expression evaluation.
///
/// `Variables` stores a mapping from variable names (`String`) to values (`Complex<f64>`),
/// allowing expressions to reference these values by name during parsing or evaluation.
///
/// # Examples
///
/// ```
/// use formulac::variable::Variables;
/// use num_complex::Complex;
///
/// let mut vars = Variables::default();
/// vars.insert(&[("x", Complex::new(1.0, 0.0)), ("y", Complex::new(2.0, 3.0))]);
///
/// assert!(vars.contains("x"));
/// assert_eq!(*vars.get("y").unwrap(), Complex::new(2.0, 3.0).into());
/// ```
#[derive(Debug)]
pub struct Variables {
    table: HashMap<String, Complex<f64>>
}

impl Variables {
    /// Creates a new empty `Variables` table.
    pub fn new() -> Self {
        Self {
            table: HashMap::new(),
        }
    }

    /// Constructs a `Variables` table from a slice of key-value pairs.
    ///
    /// Values can be any type convertible into `f64` or `Complex<f64>`.
    ///
    /// # Examples
    ///
    /// ```
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
    /// ```
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
    pub fn contains(&self, key: &str) -> bool {
        self.table.contains_key(key)
    }

    /// Retrieves a reference to the value of a variable by name.
    ///
    /// Returns `Some(&Complex<f64>)` if found, otherwise `None`.
    pub fn get(&self, key: &str) -> Option<&Complex<f64>> {
        self.table.get(key)
    }

    /// Clears all variables from the table.
    pub fn clear(&mut self) {
        self.table.clear();
    }
}

impl Default for Variables {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone)]
pub struct UserDefinedFunction<'a> {
    func: Arc<dyn Fn(&[Complex<f64>]) -> Complex<f64> + Send + Sync>,
    arity: usize,
    name: &'a str,
}

impl<'a> UserDefinedFunction<'a> {
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

    pub fn apply(&self, args: &[Complex<f64>]) -> Complex<f64> {
        (self.func)(args)
    }

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
    fn eq(&self, other: &Self) -> bool {
        (self.arity == other.arity) && (self.name == other.name)
    }
}

/// A table that stores user-defined custom functions such.
///
/// `UserDefinedTable` allows dynamic registration.
/// This enables extending the formula parser at runtime with additional
/// functionality defined by the user.
///
/// # Examples
///
/// ```
/// use num_complex::Complex;
/// use formulac::variable::UserDefinedTable;
/// use formulac::variable::UserDefinedFunction;
///
/// // Create a new user-defined table
/// let mut users = UserDefinedTable::default();
/// let my_func = UserDefinedFunction::new(
///     "my_func",
///     | args | (args[0] + Complex::new(1.0, 0.0)) / (args[0] - Complex::new(1.0, 0.0)),
///     1,
/// );
///
/// // Register a custom constant
/// users.register("my_func", my_func);
///
/// // Retrieve the token
/// if let Some(func) = users.get("my_func") {
///     println!("Found function {:?}", func);
/// }
/// ```
///
/// # Notes
/// - It is the caller's responsibility to ensure that user-defined functions,
///   do not conflict with built-in names.
#[derive(Clone)]
pub struct UserDefinedTable<'a> {
    table: HashMap<String, UserDefinedFunction<'a>>,
}

impl<'a> UserDefinedTable<'a> {
    /// Creates an empty `UserDefinedTable`.
    pub fn new() -> Self {
        Self { table: HashMap::new(), }
    }

    /// Registers a new token under the given name.
    ///
    /// If the name already exists, the previous function is replaced.
    pub fn register(&mut self, name: &str, func: UserDefinedFunction<'a>)
    {
        self.table.insert(name.to_string(), func);
    }

    /// Retrieves a custom function by its name.
    ///
    /// Returns `Some(&UserDefinedFunction)` if the name exists, or `None` otherwise.
    pub fn get(&self, name: &str)
        -> Option<&UserDefinedFunction<'a>> {
        self.table.get(name)
    }

    /// Clear its table.
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
mod userdefinedfunction_tests {
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
mod userdefinedtable_tests {
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
