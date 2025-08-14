//! # variables.rs
//!
//! This module provides the `Variables` struct, which manages a table of named
//! variables used in mathematical expressions.
//!
//! Each variable is stored as a `FuncReturn` (typically `Complex<f64>`), allowing
//! both real and complex values to be represented. The module provides convenient
//! methods to insert, retrieve, check, and clear variables.

use std::collections::HashMap;
use crate::token::FuncReturn;

/// A collection of named variables for expression evaluation.
///
/// `Variables` stores a mapping from variable names (`String`) to values (`FuncReturn`),
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
    table: HashMap<String, FuncReturn>
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
    /// Values can be any type convertible into `FuncReturn` (e.g., `f64` or `Complex<f64>`).
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
        FuncReturn: From<V>,
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
        FuncReturn: From<V>,
    {
        for (key, val) in items {
            self.table.insert(key.to_string(), FuncReturn::from(val.clone()));
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
    /// Returns `Some(&FuncReturn)` if found, otherwise `None`.
    pub fn get(&self, key: &str) -> Option<&FuncReturn> {
        self.table.get(key)
    }

    /// Clears all variables from the table.
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
        assert_eq!(*vars.get("a").unwrap(), FuncReturn::from(1.0));
    }

    #[test]
    fn test_insert_complex() {
        let mut vars = Variables::new();
        vars.insert(&[("b", FuncReturn::new(1.0, 2.0))]);
        assert_eq!(*vars.get("b").unwrap(), FuncReturn::new(1.0, 2.0));
    }

    #[test]
    fn test_from() {
        let vars = Variables::from(&[("1", 1.0), ("2", 2.0)]);
        assert_eq!(*vars.get("1").unwrap(), FuncReturn::from(1.0));
        assert_eq!(*vars.get("2").unwrap(), FuncReturn::from(2.0));
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
        assert_eq!(*vars.get("1").unwrap(), FuncReturn::from(1.0));

        vars.clear();
        assert_eq!(vars.get("1").is_none(), true);
    }
}
