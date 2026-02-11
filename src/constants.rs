//! # constants.rs
//!
//! Mathematical constants used in formula parsing.
//!
//! This module provides a centralized store of predefined mathematical constants
//! (E, PI, SQRT_2, etc.) that can be referenced in expressions.

use num_complex::Complex;
use std::collections::HashMap;

/// A collection of named mathematical constants.
///
/// `Constants` maintains a mapping from constant names (`String`) to values (`Complex<f64>`),
/// allowing expressions to reference these values by name during parsing or evaluation.
#[derive(Debug, PartialEq)]
pub(crate) struct Constants
{
    map: HashMap<String, Complex<f64>>,
}

impl Constants
{
    /// Creates an empty `Constants` table.
    pub fn new() -> Self { Self { map: HashMap::new() } }

    /// Constructs a `Constants` table from an iterator of key-value pairs.
    pub fn from<I, S, V>(items: I) -> Self
    where
        I: IntoIterator<Item = (S, V)>,
        String: From<S>,
        Complex<f64>: From<V>,
    {
        let mut consts = Self::new();
        for (key, val) in items {
            consts.insert(key, val);
        }
        consts
    }

    /// Inserts a constant into the table.
    pub fn insert<S, V>(&mut self, key: S, value: V)
    where
        String: From<S>,
        Complex<f64>: From<V>,
    {
        self.map.insert(String::from(key), Complex::from(value));
    }

    /// Checks if a constant with the given name exists in the table.
    pub fn contains<S>(&self, key: S) -> bool
    where
        String: From<S>,
    {
        self.map.contains_key(String::from(key).as_str())
    }

    /// Retrieves a reference to the value of a constant by name.
    pub fn get<S>(&self, key: S) -> Option<&Complex<f64>>
    where
        String: From<S>,
    {
        self.map.get(String::from(key).as_str())
    }

    /// Clears all constants from the table.
    pub fn clear(&mut self) { self.map.clear(); }

    /// Returns the number of elements in the table.
    pub fn len(&self) -> usize { self.map.len() }

    /// Returns true if the table contains no elements.
    pub fn is_empty(&self) -> bool { self.map.is_empty() }

    /// Returns a list of supported mathematical constant names.
    pub fn keys(&self) -> impl Iterator<Item = &str>
    {
        self.map.keys().map(|s| s.as_str())
    }

    /// Returns an iterator over the constants.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &Complex<f64>)>
    {
        self.map.iter()
    }
}

impl<S, V> FromIterator<(S, V)> for Constants
where
    String: From<S>,
    Complex<f64>: From<V>,
{
    fn from_iter<T: IntoIterator<Item = (S, V)>>(iter: T) -> Self
    {
        Self::from(iter)
    }
}

impl Default for Constants
{
    fn default() -> Self
    {
        Self::from([
            ("E", std::f64::consts::E),
            ("FRAC_1_PI", std::f64::consts::FRAC_1_PI),
            ("FRAC_1_SQRT_2", std::f64::consts::FRAC_1_SQRT_2),
            ("FRAC_2_PI", std::f64::consts::FRAC_2_PI),
            ("FRAC_2_SQRT_PI", std::f64::consts::FRAC_2_SQRT_PI),
            ("FRAC_PI_2", std::f64::consts::FRAC_PI_2),
            ("FRAC_PI_3", std::f64::consts::FRAC_PI_3),
            ("FRAC_PI_4", std::f64::consts::FRAC_PI_4),
            ("FRAC_PI_6", std::f64::consts::FRAC_PI_6),
            ("FRAC_PI_8", std::f64::consts::FRAC_PI_8),
            ("LN_2", std::f64::consts::LN_2),
            ("LN_10", std::f64::consts::LN_10),
            ("LOG2_10", std::f64::consts::LOG2_10),
            ("LOG2_E", std::f64::consts::LOG2_E),
            ("LOG10_2", std::f64::consts::LOG10_2),
            ("LOG10_E", std::f64::consts::LOG10_E),
            ("PI", std::f64::consts::PI),
            ("SQRT_2", std::f64::consts::SQRT_2),
            ("TAU", std::f64::consts::TAU),
        ])
    }
}

/// Returns a list of strings that can be specified by default.
pub fn names() -> Vec<String>
{
    Constants::default().iter()
        .map(|(key, _value)| key.to_owned())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn test_new_insert_contains_get() {
        let mut c = Constants::new();
        assert!(c.is_empty());

        c.insert("PI", std::f64::consts::PI);
        assert!(c.contains("PI"));

        let pi = c.get("PI").expect("PI should be present");
        assert_eq!(pi.re, std::f64::consts::PI);
        assert_eq!(pi.im, 0.0);
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn test_from_and_from_iter() {
        let arr = [("PI", std::f64::consts::PI), ("E", std::f64::consts::E)];
        let c = Constants::from(arr);
        assert!(c.contains("PI"));
        assert!(c.contains("E"));

        let vec = vec![
            ("A", Complex::new(1.0, 0.5)),
            ("B", Complex::from(2.0f64)),
        ];
        let c2: Constants = vec.into_iter().collect();
        assert!(c2.contains("A"));
        assert!(c2.contains("B"));
    }

    #[test]
    fn test_keys_iter_and_names_len() {
        let consts = Constants::default();
        let keys: Vec<&str> = consts.keys().collect();
        assert!(keys.contains(&"PI"));
        assert!(keys.contains(&"E"));
        // expected number of default constants (matches Default implementation)
        assert_eq!(keys.len(), 19);
    }

    #[test]
    fn test_iter_len_clear_is_empty() {
        let mut c = Constants::new();
        c.insert("X", 3.14);
        c.insert("Y", 2.71);
        assert_eq!(c.len(), 2);

        // iter yields (&String, &Complex<f64>)
        let mut found = vec![];
        for (k, v) in c.iter() {
            found.push((k.clone(), v.clone()));
        }
        assert!(found.iter().any(|(k, _)| k == "X"));
        assert!(found.iter().any(|(k, _)| k == "Y"));

        c.clear();
        assert!(c.is_empty());
    }

    #[test]
    fn test_owned_keys_conversion() {
        let consts = Constants::default();
        let owned: Vec<String> = consts.keys().map(|s| s.to_string()).collect();
        assert!(owned.contains(&"PI".to_string()));
    }
}
