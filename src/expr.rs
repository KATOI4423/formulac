//! # expr.rs
//!
//! Read-only access to the expression tree of a parsed formula.
//!
//! [`Builder::parse`](crate::Builder::parse) returns an [`Expr`], an owned
//! expression tree. Its nodes are inspected through [`ExprRef`], a cheap,
//! copyable reference to one node, and [`ExprKind`], which tells what the node is.
//!
//! The internal AST layout is not exposed, so the parser and simplifier can
//! evolve without breaking code that walks the tree.
//!
//! # Example: listing the top-level terms of a sum
//! ```rust
//! use formulac::Builder;
//! use formulac::expr::{ExprKind, ExprRef};
//! use formulac::operators::BinaryOperatorKind;
//!
//! fn collect_terms<'a>(e: ExprRef<'a, f64, 1>, out: &mut Vec<ExprRef<'a, f64, 1>>) {
//!     match e.kind() {
//!         ExprKind::BinaryOperator { kind: BinaryOperatorKind::Add, left, right } => {
//!             collect_terms(left, out);
//!             collect_terms(right, out);
//!         }
//!         _ => out.push(e),
//!     }
//! }
//!
//! let expr = Builder::<f64, 1>::new("z^2 + sin(z) + 3", ["z"]).parse().unwrap();
//! let mut terms = Vec::new();
//! collect_terms(expr.view(), &mut terms);
//! assert_eq!(terms.len(), 3);
//! ```

use num_complex::Complex;
use std::rc::Rc;

use crate::astnode::AstNode;
use crate::core::Real;
use crate::functions::{
    FunctionKind,
    UserFn,
};
use crate::operators::{
    BinaryOperatorKind,
    UnaryOperatorKind,
};

/// An owned expression tree produced by [`Builder::parse`](crate::Builder::parse).
///
/// `N` is the number of arguments of the formula, as in [`Builder`](crate::Builder).
/// The argument names given to the builder are kept with the tree.
///
/// Cloning is cheap: subtrees are shared. An `Expr` is not `Send` or `Sync`;
/// use it on the thread that created it.
#[derive(Debug, Clone)]
pub struct Expr<T: Real, const N: usize> {
    root: AstNode<T>,
    args: Rc<[String; N]>,
}

impl<T: Real, const N: usize> Expr<T, N> {
    pub(crate) fn new(root: AstNode<T>, args: [String; N]) -> Self {
        Self { root, args: Rc::new(args) }
    }

    /// Returns a reference to the root node.
    pub fn view(&self) -> ExprRef<'_, T, N> {
        ExprRef { node: &self.root, args: &self.args }
    }

    /// Returns what the root node is. Shorthand for `self.view().kind()`.
    pub fn kind(&self) -> ExprKind<'_, T, N> {
        self.view().kind()
    }
}

/// A reference to one node of an [`Expr`].
///
/// It is `Copy`, so it can be passed around freely while walking the tree.
#[derive(Debug, Clone, Copy)]
pub struct ExprRef<'a, T: Real, const N: usize> {
    node: &'a AstNode<T>,
    args: &'a Rc<[String; N]>,
}

impl<'a, T: Real, const N: usize> ExprRef<'a, T, N> {
    fn child(&self, node: &'a AstNode<T>) -> Self {
        Self { node, args: self.args }
    }

    fn children(&self, nodes: &'a [Rc<AstNode<T>>]) -> Vec<Self> {
        nodes.iter().map(|n| self.child(n)).collect()
    }

    /// Returns what this node is, with references to its children.
    pub fn kind(&self) -> ExprKind<'a, T, N> {
        match self.node {
            AstNode::Number { value, .. } => ExprKind::Number { value },
            AstNode::Argument { index, .. } => ExprKind::Argument {
                index: *index,
                name: self.args[*index].as_str(),
            },
            AstNode::UnaryOperator { kind, expr, .. } => ExprKind::UnaryOperator {
                kind: *kind,
                expr: self.child(expr),
            },
            AstNode::BinaryOperator { kind, left, right, .. } => ExprKind::BinaryOperator {
                kind: *kind,
                left: self.child(left),
                right: self.child(right),
            },
            AstNode::FunctionCall { kind, args, .. } => ExprKind::FunctionCall {
                kind: *kind,
                args: self.children(args),
            },
            AstNode::UserFunctionCall { func, args, .. } => ExprKind::UserFunctionCall {
                func,
                args: self.children(args),
            },
            // The parser resolves `diff(...)` immediately, so an `Expr` never holds one.
            AstNode::Derivative { .. } => unreachable!("Derivative nodes are resolved during parsing"),
        }
    }

    /// Copies the subtree rooted at this node into a new [`Expr`].
    ///
    /// The copy is cheap: the nodes below are shared, not duplicated.
    pub fn to_expr(&self) -> Expr<T, N> {
        Expr { root: self.node.clone(), args: Rc::clone(self.args) }
    }
}

/// What a node of an expression tree is.
///
/// Returned by [`ExprRef::kind`] and [`Expr::kind`]. Child nodes are given as
/// [`ExprRef`]s. Named constants such as `PI` appear as [`ExprKind::Number`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ExprKind<'a, T: Real, const N: usize> {
    /// A numeric literal or a named constant.
    Number {
        /// The value.
        value: &'a Complex<T>,
    },

    /// An argument of the formula.
    Argument {
        /// Position of the argument in the names given to [`Builder::new`](crate::Builder::new).
        index: usize,
        /// Name of the argument.
        name: &'a str,
    },

    /// A unary operator such as `-x`.
    UnaryOperator {
        /// The operator.
        kind: UnaryOperatorKind,
        /// The operand.
        expr: ExprRef<'a, T, N>,
    },

    /// A binary operator such as `x + y`.
    BinaryOperator {
        /// The operator.
        kind: BinaryOperatorKind,
        /// The left operand.
        left: ExprRef<'a, T, N>,
        /// The right operand.
        right: ExprRef<'a, T, N>,
    },

    /// A call to a built-in function such as `sin(x)`.
    FunctionCall {
        /// The function.
        kind: FunctionKind,
        /// The arguments, in order.
        args: Vec<ExprRef<'a, T, N>>,
    },

    /// A call to a user-defined function.
    UserFunctionCall {
        /// The function.
        func: &'a UserFn<T>,
        /// The arguments, in order.
        args: Vec<ExprRef<'a, T, N>>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Builder;

    fn parse(formula: &str) -> Expr<f64, 2> {
        Builder::new(formula, ["x", "y"]).parse().unwrap()
    }

    #[test]
    fn number() {
        let e = parse("2.5");
        assert!(matches!(e.kind(), ExprKind::Number { value } if *value == Complex::new(2.5, 0.0)));
    }

    #[test]
    fn named_constant_is_number() {
        let e = parse("PI");
        assert!(matches!(e.kind(), ExprKind::Number { value } if *value == Complex::new(std::f64::consts::PI, 0.0)));
    }

    #[test]
    fn argument_has_index_and_name() {
        let e = parse("y");
        assert!(matches!(e.kind(), ExprKind::Argument { index: 1, name: "y" }));
    }

    #[test]
    fn unary_minus_after_binary_minus() {
        // `2 - -x` is `2 - (-x)`
        let e = parse("2 - -x");
        let ExprKind::BinaryOperator { kind: BinaryOperatorKind::Sub, left, right } = e.kind() else {
            panic!("expected Sub");
        };
        assert!(matches!(left.kind(), ExprKind::Number { value } if value.re == 2.0));
        let ExprKind::UnaryOperator { kind: UnaryOperatorKind::Negative, expr } = right.kind() else {
            panic!("expected Negative");
        };
        assert!(matches!(expr.kind(), ExprKind::Argument { index: 0, .. }));
    }

    #[test]
    fn parse_keeps_raw_structure() {
        // Not simplified: the constant power and the division are kept as written.
        let e = parse("2^3 / x");
        let ExprKind::BinaryOperator { kind: BinaryOperatorKind::Div, left, .. } = e.kind() else {
            panic!("expected Div");
        };
        assert!(matches!(left.kind(), ExprKind::BinaryOperator { kind: BinaryOperatorKind::Pow, .. }));
    }

    #[test]
    fn function_call_args_in_order() {
        let e = parse("pow(x, y)");
        let ExprKind::FunctionCall { kind: FunctionKind::Pow, args } = e.kind() else {
            panic!("expected pow");
        };
        assert_eq!(args.len(), 2);
        assert!(matches!(args[0].kind(), ExprKind::Argument { name: "x", .. }));
        assert!(matches!(args[1].kind(), ExprKind::Argument { name: "y", .. }));
    }

    #[test]
    fn user_function_call() {
        let double = UserFn::<f64>::new("double", |[v]| v * 2.0);
        let e = Builder::new("double(x)", ["x"])
            .with_user_functions([double])
            .parse()
            .unwrap();
        let ExprKind::UserFunctionCall { func, args } = e.kind() else {
            panic!("expected user function");
        };
        assert_eq!(func.name(), "double");
        assert_eq!(args.len(), 1);
    }

    #[test]
    fn diff_is_resolved() {
        // diff(x^2, x) is differentiated during parsing; no derivative node remains.
        let e = parse("diff(x^2, x)");
        fn walk(e: ExprRef<'_, f64, 2>) {
            match e.kind() {
                ExprKind::UnaryOperator { expr, .. } => walk(expr),
                ExprKind::BinaryOperator { left, right, .. } => { walk(left); walk(right); }
                ExprKind::FunctionCall { args, .. } | ExprKind::UserFunctionCall { args, .. } => args.into_iter().for_each(walk),
                _ => (),
            }
        }
        walk(e.view()); // must not panic
    }

    #[test]
    fn to_expr_copies_subtree() {
        let e = parse("sin(x) + y");
        let ExprKind::BinaryOperator { left, .. } = e.kind() else { panic!("expected Add") };
        let sub = left.to_expr();
        assert!(matches!(sub.kind(), ExprKind::FunctionCall { kind: FunctionKind::Sin, .. }));
    }

    #[test]
    fn parse_error_is_reported() {
        assert!(Builder::<f64, 1>::new("x +", ["x"]).parse().is_err());
        assert!(Builder::<f64, 1>::new("unknown(x)", ["x"]).parse().is_err());
    }
}
