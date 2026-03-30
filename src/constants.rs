//! # constants.rs
//!
//! Mathematical constants used in formula parsing.
//!
//! This module provides a centralized store of predefined mathematical constants
//! (E, PI, SQRT_2, etc.) that can be referenced in expressions.

use num_complex::Complex;
use std::collections::HashMap;

use crate::core::Real;

/// A collection of named mathematical constants.
///
/// `Constants` maintains a mapping from constant names (`String`) to values (`Complex<T>`),
/// allowing expressions to reference these values by name during parsing or evaluation.
#[derive(Debug, PartialEq)]
pub struct Constants<T: Real>
{
    map: HashMap<String, Complex<T>>,
}

impl<T: Real> Constants<T>
{
    /// Creates an empty `Constants` table.
    pub fn new() -> Self { Self { map: HashMap::new() } }

    /// Creates a `Constants` table with builtin-constants
    pub fn with_builtins() -> Self { Self::default() }

    /// Constructs a `Constants` table from an iterator of key-value pairs.
    pub fn from<I, S, V>(items: I) -> Self
    where
        I: IntoIterator<Item = (S, V)>,
        S: Into<String>,
        Complex<T>: From<V>,
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
        S: Into<String>,
        Complex<T>: From<V>,
    {
        self.map.insert(key.into(), Complex::from(value));
    }

    /// Checks if a constant with the given name exists in the table.
    pub fn contains<S>(&self, key: S) -> bool
    where
        S: AsRef<str>,
    {
        self.map.contains_key(key.as_ref())
    }

    /// Retrieves a reference to the value of a constant by name.
    pub fn get<S>(&self, key: S) -> Option<&Complex<T>>
    where
        S: AsRef<str>,
    {
        self.map.get(key.as_ref())
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

    /// Returns a list of strings that can be specified by default.
    pub fn symbols() -> &'static [&'static str]
    {
        BUILTIN_CONSTANT_NAMES
    }

    /// Returns an iterator over the constants.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &Complex<T>)>
    {
        self.map.iter()
    }
}

impl<S, V, R: Real> FromIterator<(S, V)> for Constants<R>
where
    S: Into<String>,
    Complex<R>: From<V>,
{
    fn from_iter<T: IntoIterator<Item = (S, V)>>(iter: T) -> Self
    {
        Self::from(iter)
    }
}

macro_rules! define_builtin_constants {
    ($( $name:expr => $func:ident ),* $(,)? ) => {
        pub const BUILTIN_CONSTANT_NAMES: &'static [&'static str] = &[
            $( $name, )*
        ];

        impl<T: Real> Constants<T> {
            /// Helper to build builtin constants
            fn resolve_builtin(name: &str) -> Option<T>
            {
                match name {
                    $( $name => Some(T::$func()), )*
                    _ => None,
                }
            }
        }
    };
}

define_builtin_constants! {
    "E" => e,
    "FRAC_1_PI" => frac_1_pi,
    "FRAC_1_SQRT_2" => frac_1_sqrt_2,
    "FRAC_2_PI" => frac_2_pi,
    "FRAC_2_SQRT_PI" => frac_2_sqrt_pi,
    "FRAC_PI_2" => frac_pi_2,
    "FRAC_PI_3" => frac_pi_3,
    "FRAC_PI_4" => frac_pi_4,
    "FRAC_PI_6" => frac_pi_6,
    "FRAC_PI_8" => frac_pi_8,
    "LN_2" => ln_2,
    "LN_10" => ln_10,
    "LOG2_10" => log2_10,
    "LOG2_E" => log2_e,
    "LOG10_2" => log10_2,
    "LOG10_E" => log10_e,
    "PI" => pi,
    "SQRT_2" => sqrt_2,
    "TAU" => tau,
}

impl<T: Real> Default for Constants<T>
{
    fn default() -> Self
    {
        BUILTIN_CONSTANT_NAMES
            .iter()
            .filter_map(|&name| {
                Self::resolve_builtin(name).map(|v| (name, v))
            })
            .collect()
    }
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
        let c2: Constants<f64> = vec.into_iter().collect();
        assert!(c2.contains("A"));
        assert!(c2.contains("B"));
    }

    #[test]
    fn test_keys_iter_and_names_len() {
        let consts = Constants::<f64>::default();
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
        let consts = Constants::<f64>::default();
        let owned: Vec<String> = consts.keys().map(|s| s.to_string()).collect();
        assert!(owned.contains(&"PI".to_string()));
    }
}
