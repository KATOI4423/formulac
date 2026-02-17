//! Top-level functions module.
//!
//! This module groups function-related implementations used by the expression
//! system. It provides:
//!
//! - `buildin`: built-in mathematical functions (registry and concrete
//!   implementations).
//! - `core`: the numerical/complex backend abstraction (`ComplexBackend`) that
//!   defines required operations for complex/scalar types.
//! - `custom`: user-defined functions with support for registering analytic
//!   derivatives and for numeric differentiation of closures.
//!
//! The module also exposes a small utility `names()` which returns a static
//! slice of available built-in function names; this is convenient for the
//! parser, completion lists, or error messages.
//!
//! Note: all submodules are `pub(crate)` because the functions API is an
//! internal implementation detail of the crate.
pub(crate) mod buildin;
pub(crate) mod core;
pub(crate) mod custom;

/// Return the available built-in function names.
///
/// The slice is static and intended for use by the parser, error messages,
/// or autocompletion UI.
pub fn names() -> &'static [&'static str]
{
    buildin::FuncKind::available_names()
}
