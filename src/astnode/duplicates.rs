//! # astnode/duplicates.rs
//!
//! Test-only analysis: how often does the same subexpression occur
//! more than once, and how often do `sin(u)` and `cos(u)` share the same `u`?
//!
//! Run the report with:
//!   cargo test --lib duplicates::tests::report -- --nocapture

#[cfg(test)]
use crate::astnode::AstNode;
#[cfg(test)]
use crate::core::Real;
#[cfg(test)]
use crate::functions::FunctionKind;

#[cfg(test)]
impl<T: Real> AstNode<T> {
    /// Pushes every non-leaf node (pre-order) onto `out`.
    fn collect_inner_nodes<'a>(&'a self, out: &mut Vec<&'a Self>) {
        match self {
            Self::Number { .. } | Self::Argument { .. } => (),
            Self::UnaryOperator { expr, .. } | Self::Derivative { expr, .. } => {
                out.push(self);
                expr.collect_inner_nodes(out);
            }
            Self::BinaryOperator { left, right, .. } => {
                out.push(self);
                left.collect_inner_nodes(out);
                right.collect_inner_nodes(out);
            }
            Self::FunctionCall { args, .. } | Self::UserFunctionCall { args, .. } => {
                out.push(self);
                args.iter().for_each(|a| a.collect_inner_nodes(out));
            }
        }
    }

    /// Number of nodes in the tree, leaves included.
    pub(crate) fn node_count(&self) -> usize {
        match self {
            Self::Number { .. } | Self::Argument { .. } => 1,
            Self::UnaryOperator { expr, .. } | Self::Derivative { expr, .. } => 1 + expr.node_count(),
            Self::BinaryOperator { left, right, .. } => 1 + left.node_count() + right.node_count(),
            Self::FunctionCall { args, .. } | Self::UserFunctionCall { args, .. } => {
                1 + args.iter().map(|a| a.node_count()).sum::<usize>()
            }
        }
    }

    /// Returns every non-leaf subtree that occurs at least twice, with its number of occurrences.
    ///
    /// Occurrences nested inside another duplicate are counted as well, so the result
    /// is an upper bound on what common-subexpression sharing could remove.
    pub(crate) fn duplicate_subtrees(&self) -> Vec<(&Self, usize)> {
        let mut nodes = Vec::new();
        self.collect_inner_nodes(&mut nodes);

        let mut groups: Vec<(&Self, usize)> = Vec::new();
        for node in nodes {
            match groups.iter_mut().find(|(g, _)| g.structurally_eq(node)) {
                Some((_, count)) => *count += 1,
                None => groups.push((node, 1)),
            }
        }
        groups.retain(|(_, count)| *count >= 2);
        groups
    }

    fn arg_of(&self, wanted: FunctionKind) -> Option<&Self> {
        match self {
            Self::FunctionCall { kind, args, .. } if *kind == wanted => Some(args[0].as_ref()),
            _ => None,
        }
    }

    /// Counts `sin(u)` nodes that have a `cos(u)` with a structurally equal `u` somewhere in the tree.
    pub(crate) fn sin_cos_pairs(&self) -> usize {

        let mut nodes = Vec::new();
        self.collect_inner_nodes(&mut nodes);

        let coses: Vec<&Self> = nodes.iter().filter_map(|n| n.arg_of(FunctionKind::Cos)).collect();
        nodes.iter()
            .filter_map(|n| n.arg_of(FunctionKind::Sin))
            .filter(|u| coses.iter().any(|c| c.structurally_eq(u)))
            .count()
    }
}

#[cfg(test)]
mod tests {
    use num_complex::Complex;

    use super::*;
    use crate::constants::Constants;
    use crate::lexer;
    use crate::token::UserFnTable;

    fn parse(formula: &str) -> AstNode<f64> {
        let mut constants = Constants::<f64>::new();
        for (name, re, im) in [
            ("a0", 1.0, 2.0), ("a1", -2.0, 3.5), ("a2", 5.25, -0.22), ("a3", -0.03, 4.03),
            ("a4", 1.0, 0.0), ("w", 0.25, 0.333), ("phy", -2.0, 3.5), ("A", 5.25, -0.22),
            ("B", -0.03, 4.03), ("lam", 0.25, 0.333),
        ] {
            constants.insert(name, Complex::new(re, im));
        }
        let lexemes = lexer::from(formula);
        AstNode::from(&lexemes, &["x"], &constants, &UserFnTable::new())
            .unwrap()
            .simplify()
    }

    #[test]
    fn wave_shares_sin_cos_argument() {
        let ast = parse("A*sin(w*x + phy) + B*cos(w*x + phy)");
        assert_eq!(ast.sin_cos_pairs(), 1);
        // `w*x + phy` occurs twice.
        assert!(ast.duplicate_subtrees().iter().any(|(_, n)| *n == 2));
    }

    #[test]
    fn no_pair_when_arguments_differ() {
        assert_eq!(parse("sin(x) + cos(x + 1)").sin_cos_pairs(), 0);
        assert_eq!(parse("A*exp(-lam*x) + B").sin_cos_pairs(), 0);
    }

    /// Prints a table for the formulas used in the benchmarks.
    #[test]
    fn report() {
        let formulas = [
            "x + 1",
            "x*x + 2*x + 1",
            "sin(x) + cos(x) + exp(x)",
            "((x + 1) * (x - 1)) / (x * x + 2)",
            "a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4",
            "A*sin(w*x + phy) + B*cos(w*x + phy)",
            "A*exp(-lam*x) + B",
            "diff(x^2, x)",
            "diff(sin(x), x)",
            "diff(exp(x^2+3*x+1), x)",
            "diff(sin(cos(x)), x)",
            "diff(x^10 + x^5 + x^2, x)",
        ];
        println!("{:<44} {:>6} {:>10} {:>10} {:>9}", "formula", "nodes", "dup groups", "dup nodes", "sin/cos");
        for f in formulas {
            let ast = parse(f);
            let dups = ast.duplicate_subtrees();
            // Nodes that sharing could remove: (occurrences - 1) * size, upper bound.
            let removable: usize = dups.iter().map(|(n, c)| (c - 1) * n.node_count()).sum();
            println!("{:<44} {:>6} {:>10} {:>10} {:>9}", f, ast.node_count(), dups.len(), removable, ast.sin_cos_pairs());
        }
    }
}
