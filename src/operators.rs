//! # operators.rs

use crate::functions::core::ComplexBackend;

use std::ops::{
    Neg,
    Add,
    Sub,
    Mul,
    Div,
};


#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ParseOperatorError {
    UnknownOperator,
}

macro_rules! unary_operator_kinds {
    ($( $name: ident => { symbol: $symbol:expr, apply: $apply:expr } ), + $(,)? ) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub(crate) enum UnaryOperatorKind {
            $($name), *
        }

        impl UnaryOperatorKind {
            pub fn apply<T, S>(&self, x: T) -> T
            where
                T: ComplexBackend<S> + Neg<Output = T>,
                S: Clone + Send + Sync + 'static
            {
                match self {
                    $( Self::$name => $apply(x), )*
                }
            }

            pub fn names() -> &'static [&'static str]
            {
                &[ $( $symbol ),+ ]
            }
        }

        impl std::str::FromStr for UnaryOperatorKind {
            type Err = ParseOperatorError;


            fn from_str(s: &str) -> Result<Self, Self::Err>
            {
                match s {
                    $(
                        $symbol => Ok(Self::$name),
                    )+
                    _ => Err(ParseOperatorError::UnknownOperator),
                }
            }
        }
    };
}

unary_operator_kinds! {
    Pos => { symbol: "+", apply: |x: T| x },
    Neg => { symbol: "-", apply: |x: T| -x },
}

pub fn unary_operator_names() -> &'static[&'static str]
{
    UnaryOperatorKind::names()
}


#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BinaryOperatorInfo {
    pub precedence: u8,
    pub is_left_assoc: bool,
}

macro_rules! binary_operator_kinds {
    ($( $name:ident => {
        symbol: $symbol:expr,
        precedence: $prec:expr,
        left_assoc: $assoc:expr,
        apply: $apply:expr
    }), + $(,)?) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub(crate) enum BinaryOperatorKind {
            $( $name ),*
        }

        impl BinaryOperatorKind {
            pub fn info(&self) -> BinaryOperatorInfo
            {
                match self {
                    $(
                        Self::$name => BinaryOperatorInfo{ precedence: $prec, is_left_assoc: $assoc }
                    ), +
                }
            }

            pub fn apply<T, S>(&self, l: T, r: T) -> T
            where
                T: ComplexBackend<S>
                    + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
                S: Clone + Send + Sync + 'static,
            {
                match self {
                    $(
                        Self::$name => $apply(l, r)
                    ), +
                }
            }

            pub fn names() -> &'static[&'static str]
            {
                &[ $( $symbol ), + ]
            }
        }

        impl std::str::FromStr for BinaryOperatorKind {
            type Err = ParseOperatorError;

            fn from_str(s: &str) -> Result<Self, Self::Err>
            {
                match s {
                    $(
                        $symbol => Ok(Self::$name),
                    ) +
                    _ => Err(Self::Err::UnknownOperator),
                }
            }
        }
    };
}

binary_operator_kinds! {
    Add => { symbol: "+", precedence: 0, left_assoc: true,  apply: |l: T, r: T| l + r },
    Sub => { symbol: "-", precedence: 0, left_assoc: true,  apply: |l: T, r: T| l - r },
    Mul => { symbol: "*", precedence: 1, left_assoc: true,  apply: |l: T, r: T| l * r },
    Div => { symbol: "/", precedence: 1, left_assoc: true,  apply: |l: T, r: T| l / r },
    Pow => { symbol: "^", precedence: 2, left_assoc: true,  apply: |l: T, r: T| l.pow(&r) },
}

pub fn binary_opeator_names() -> &'static [&'static str]
{
    BinaryOperatorKind::names()
}


#[cfg(test)]
mod unary_operator_kind_tests {
    use super::*;

    use num_complex::Complex;
    use std::str::FromStr;

    #[test]
    fn test_from_str() {
        assert_eq!(UnaryOperatorKind::from_str("+"), Ok(UnaryOperatorKind::Pos));
        assert_eq!(UnaryOperatorKind::from_str("-"), Ok(UnaryOperatorKind::Neg));
        assert!(UnaryOperatorKind::from_str("*").is_err());
        assert!(UnaryOperatorKind::from_str("").is_err());
        assert!(UnaryOperatorKind::from_str("x").is_err());
    }

    #[test]
    fn test_apply() {
        let x = Complex::new(2.0, -1.0);
        assert_eq!(UnaryOperatorKind::Pos.apply(x), x);
        assert_eq!(UnaryOperatorKind::Neg.apply(x), -x);
    }
}

#[cfg(test)]
mod binary_operator_kinds_tests {
    use super::*;

    use std::str::FromStr;

    #[test]
    fn test_binary_operator_kind_from() {
        assert_eq!(BinaryOperatorKind::from_str("+"), Ok(BinaryOperatorKind::Add));
        assert_eq!(BinaryOperatorKind::from_str("-"), Ok(BinaryOperatorKind::Sub));
        assert_eq!(BinaryOperatorKind::from_str("*"), Ok(BinaryOperatorKind::Mul));
        assert_eq!(BinaryOperatorKind::from_str("/"), Ok(BinaryOperatorKind::Div));
        assert_eq!(BinaryOperatorKind::from_str("^"), Ok(BinaryOperatorKind::Pow));

        assert!(BinaryOperatorKind::from_str("").is_err());
        assert!(BinaryOperatorKind::from_str("x").is_err());
        assert!(BinaryOperatorKind::from_str("%").is_err());
    }
}
