//! # variables.rs
//!
//! This module provides the `Variables` struct, which manages a table of named
//! variables used in mathematical expressions.
//!
//! Each variable is stored as a `Complex<f64>` (typically `Complex<f64>`), allowing
//! both real and complex values to be represented. The module provides convenient
//! methods to insert, retrieve, check, and clear variables.

use num_complex::Complex;
use std::collections::HashMap;
use crate::token::Token;

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
/// let mut vars = Variables::new();
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
        return vars;
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
    /// let mut vars = Variables::new();
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

/// A table that stores user-defined tokens such as custom functions or constants.
///
/// `UserDefinedTable` allows dynamic registration and retrieval of tokens
/// that are not part of the built-in operators, functions, or constants.
/// This enables extending the formula parser at runtime with additional
/// functionality defined by the user.
///
/// # Examples
///
/// ```
/// use formulac::token::Token;
/// use formulac::variable::UserDefinedTable;
///
/// // Create a new user-defined table
/// let mut users = UserDefinedTable::new();
///
/// // Register a custom constant
/// users.register("my_const", Token::Number(num_complex::Complex::new(42.0, 0.0)));
///
/// // Retrieve the token
/// if let Some(token) = users.get("my_const") {
///     println!("Found token: {:?}", token);
/// }
/// ```
///
/// # Notes
///
/// - It is the caller's responsibility to ensure that user-defined tokens
///   do not conflict with built-in names.
/// - Only certain token types are valid for registration (`Token::Function`,
///   `Token::Operator`, and `Token::Constant`).
#[derive(Clone)]
pub struct UserDefinedTable {
    table: HashMap<String, Token>,
}

impl UserDefinedTable {
    /// Creates an empty `UserDefinedTable`.
    pub fn new() -> Self {
        Self { table: HashMap::new(), }
    }

    /// Registers a new token under the given name.
    ///
    /// If the name already exists, the previous token is replaced and returned.
    pub fn register(&mut self, name: &str, token: Token) -> Option<Token> {
        self.table.insert(name.to_string(), token)
    }

    /// Retrieves a token by its name.
    ///
    /// Returns `Some(&Token)` if the name exists, or `None` otherwise.
    pub fn get(&self, name: &str) -> Option<&Token> {
        self.table.get(name)
    }

    /// Clear its table.
    pub fn clear(&mut self) {
        self.table.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_real() {
        let mut vars = Variables::new();
        vars.insert(&[("a", 1.0)]);
        assert_eq!(*vars.get("a").unwrap(), Complex::from(1.0));
    }

    #[test]
    fn test_insert_complex() {
        let mut vars = Variables::new();
        vars.insert(&[("b", Complex::new(1.0, 2.0))]);
        assert_eq!(*vars.get("b").unwrap(), Complex::new(1.0, 2.0));
    }

    #[test]
    fn test_from() {
        let vars = Variables::from(&[("1", 1.0), ("2", 2.0)]);
        assert_eq!(*vars.get("1").unwrap(), Complex::from(1.0));
        assert_eq!(*vars.get("2").unwrap(), Complex::from(2.0));
    }

    #[test]
    fn test_contains() {
        let mut vars = Variables::new();
        assert_eq!(vars.contains("test"), false);

        vars.insert(&[("test", 1.0)]);
        assert_eq!(vars.contains("test"), true);
    }

    #[test]
    fn test_clear() {
        let mut vars = Variables::from(&[("1", 1.0)]);
        assert_eq!(*vars.get("1").unwrap(), Complex::from(1.0));

        vars.clear();
        assert_eq!(vars.get("1").is_none(), true);
    }
}
