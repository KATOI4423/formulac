//! # astnode
//!
//! Defines the [`AstNode`] enum and organizes the AST pipeline into submodules.
//!
//! ## Submodules
//! - [`core`]: [`AstNode`] enum definition and node construction helpers
//! - [`parser`]: Shunting-Yard parsing from [`Lexeme`]s into an [`AstNode`] tree
//! - [`simplify`]: Constant folding and algebraic simplification
//! - [`diff`]: Symbolic differentiation
//! - [`compile`]: Compilation into a flat postfix [`Token`] sequence

pub(crate) mod compile;
pub(crate) mod core;
pub(crate) mod diff;
pub(crate) mod parser;
pub(crate) mod simplify;

pub(crate) type AstNode<T> = core::AstNode<T>;
