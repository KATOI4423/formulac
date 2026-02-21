//! # buildin.rs
//!
//! Standard mathematical functions for formula evaluation.
//!
//! This module defines the built-in mathematical functions available in formulas,
//! including trigonometric, hyperbolic, exponential, and logarithmic functions.

use crate::core::{
    ComplexBackend,
};

use std::marker::PhantomData;

/// Error type for parsing standard function names.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ParseStdFuncError {
    /// The function name is not recognized.
    UnknownFunction,
}

macro_rules! define_functions {
    ( $( $name:ident => $imp:expr), + $(,)? ) => {
        /// Enumeration of available standard functions.
        #[allow(non_camel_case_types)] // To use ident as string to compare them.
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub(crate) enum FuncKind {
            $( $name ), +
        }

        impl FuncKind {
            /// Returns a list of available function names.
            pub(crate) fn available_names() -> &'static [&'static str]
            {
                &[ $( stringify!($name) ),+ ]
            }
        }

        impl std::str::FromStr for FuncKind {
            type Err = ParseStdFuncError;

            /// Parses a string into a FuncKind variant.
            fn from_str(s: &str) -> Result<Self, Self::Err>
            {
                match s {
                    $(
                        stringify!($name) => Ok(Self::$name),
                    )+
                    _ => Err(ParseStdFuncError::UnknownFunction),
                }
            }
        }

        impl<T, S> From<FuncKind> for FunctionImpl<T, S>
        where
            T: ComplexBackend<S>,
            S: Clone + Send + Sync + 'static,
        {
            /// Converts a FuncKind into its implementation.
            fn from(kind: FuncKind) -> Self
            {
                match kind {
                    $(
                        FuncKind::$name => $imp,
                    )+
                }
            }
        }
    };
}

define_functions!(
    sin     => FunctionImpl::Unary(|x: &T| x.sin()),
    cos     => FunctionImpl::Unary(|x: &T| x.cos()),
    tan     => FunctionImpl::Unary(|x: &T| x.tan()),
    asin    => FunctionImpl::Unary(|x: &T| x.asin()),
    acos    => FunctionImpl::Unary(|x: &T| x.acos()),
    atan    => FunctionImpl::Unary(|x: &T| x.atan()),
    sinh    => FunctionImpl::Unary(|x: &T| x.sinh()),
    cosh    => FunctionImpl::Unary(|x: &T| x.cosh()),
    tanh    => FunctionImpl::Unary(|x: &T| x.tanh()),
    asinh   => FunctionImpl::Unary(|x: &T| x.asinh()),
    acosh   => FunctionImpl::Unary(|x: &T| x.acosh()),
    atanh   => FunctionImpl::Unary(|x: &T| x.atanh()),
    exp     => FunctionImpl::Unary(|x: &T| x.exp()),
    ln      => FunctionImpl::Unary(|x: &T| x.ln()),
    log10   => FunctionImpl::Unary(|x: &T| x.log10()),
    sqrt    => FunctionImpl::Unary(|x: &T| x.sqrt()),
    abs     => FunctionImpl::Unary(|x: &T| T::from(x.abs())),
    conj    => FunctionImpl::Unary(|x: &T| x.conj()),
    pow     => FunctionImpl::Binary(|l: &T, r: &T| l.pow(r)),
    powi    => FunctionImpl::Powi(|x: &T, n: i32| x.powi(n)),
);

/// Function implementation variants.
///
/// Represents different types of mathematical function implementations:
///  - unary functions that take one complex argument
///  - binary functions that take two complex arguments
///  - integer power functions
#[derive(Clone, Debug)]
pub(crate) enum FunctionImpl<T, S>
where
    T: ComplexBackend<S>,
    S: Clone + Send + Sync + 'static,
{
    /// Unary function taking a single complex number.
    Unary(fn(&T) -> T),
    /// Binary function taking two complex numbers.
    Binary(fn(&T, &T) -> T),
    /// Integer power function.
    Powi(fn(&T, i32) -> T),
    /// Marker to hold the type parameter S.
    #[doc(hidden)]
    _Phantom(PhantomData<S>),
}
