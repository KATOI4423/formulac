//! # operators.rs

use crate::functions::core::ComplexBackend;

use std::ops::Neg;

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
