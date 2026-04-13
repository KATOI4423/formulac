//! # astnode/compile.rs
//!
//! Compiles an [`AstNode`] tree into a flat sequence of postfix [`Token`]s
//! (Reverse Polish Notation) for stack-based evaluation.
//!
//! ## Entry point
//! [`AstNode::compile`] performs a depth-first traversal of the AST
//! and emits tokens in evaluation order.
//! The resulting token sequence is consumed by the executor in [`builder`].

use crate::astnode::AstNode;
use crate::core::Real;
use crate::token::Token;

impl<T: Real> AstNode<T> {
    /// Compiles the AST into a flat sequence of postfix `Token`s.
    pub(crate) fn compile(&self) -> Vec<Token<T>> {
        let mut out = Vec::new();
        self.compile_into(&mut out);
        out
    }

    fn compile_into(&self, out: &mut Vec<Token<T>>) {
        match self {
            Self::Number { value, span } => out.push(Token::Number { value: value.clone(), span: *span }),
            Self::Argument { index, span } => out.push(Token::Argument { index: *index, span: *span }),
            Self::UnaryOperator { kind, expr, span } => {
                expr.compile_into(out);
                out.push(Token::UnaryOperator { kind: *kind, span: *span });
            }
            Self::BinaryOperator { kind, left, right, span } => {
                left.compile_into(out);
                right.compile_into(out);
                out.push(Token::BinaryOperator { kind: *kind, span: *span });
            }
            Self::FunctionCall { kind, args, span } => {
                for arg in args { arg.compile_into(out); }
                out.push(Token::Function { kind: *kind, span: *span });
            }
            Self::UserFunctionCall { func, args, span } => {
                for arg in args { arg.compile_into(out); }
                out.push(Token::UserFunction { func: func.clone(), span: *span });
            }
            Self::Derivative { .. } => {
                unreachable!("Derivative nodes must be resolved before compile()")
            }
        }
    }
}

#[cfg(test)]
mod astnode_tests {
    use super::*;
    use crate::functions::FunctionKind;
    use crate::operators::{
        BinaryOperatorKind,
        UnaryOperatorKind,
    };
    use crate::lexer::Span;
    use num_complex::Complex;

    #[test]
    fn test_compile_number() {
        let ast = AstNode::Number { value: Complex::new(1.0, 0.0), span: Span::from(0..1) };
        let tokens = ast.compile();
        assert_eq!(tokens, vec![Token::Number { value: Complex::new(1.0, 0.0), span: Span::from(0..1) }]);
    }

    #[test]
    fn test_compile_argument() {
        let ast = AstNode::<f64>::Argument { index: 1, span: Span::from(0..1) };
        let tokens = ast.compile();
        assert_eq!(tokens, vec![Token::Argument { index: 1, span: Span::from(0..1) }]);
    }

    #[test]
    fn test_compile_unary_operator() {
        let ast = -AstNode::Number { value: Complex::new(1.0, 0.0), span: Span::from(2..3) };
        let tokens = ast.compile();
        assert_eq!(
            tokens,
            vec![Token::Number { value: Complex::new(1.0, 0.0), span: Span::from(2..3) }, Token::UnaryOperator { kind: UnaryOperatorKind::Negative, span: Span::from(2..3) }]
        );
    }

    #[test]
    fn test_compile_binary_operator() {
        let ast = AstNode::Number { value: Complex::new(1.0, 0.0), span: Span::from(0..1) } + AstNode::Argument { index: 1, span: Span::from(4..5) };
        let tokens = ast.compile();
        assert_eq!(
            tokens,
            vec![
                Token::Number { value: Complex::new(1.0, 0.0), span: Span::from(0..1) },
                Token::Argument { index: 1, span: Span::from(4..5) },
                Token::BinaryOperator { kind: BinaryOperatorKind::Add, span: Span::from(0..1) },
            ]
        );
    }

    #[test]
    fn test_compile_function_single_argument() {
        let ast = AstNode::Number { value: Complex::new(1.0, 0.0), span: Span::from(0..1) }.sin();
        let tokens = ast.compile();
        assert_eq!(tokens, vec![Token::Number { value: Complex::new(1.0, 0.0), span: Span::from(0..1) }, Token::Function { kind: FunctionKind::Sin, span: Span::from(0..1) }]);
    }

    #[test]
    fn test_compile_function_multi_arguments() {
        let ast = AstNode::Number { value: Complex::new(2.0, 0.0), span: Span::from(0..1) }.pow(AstNode::Argument { index: 0, span: Span::from(4..5) });
        let tokens = ast.compile();
        assert_eq!(
            tokens,
            vec![
                Token::Number { value: Complex::new(2.0, 0.0), span: Span::from(0..1) },
                Token::Argument { index: 0, span: Span::from(4..5) },
                Token::Function { kind: FunctionKind::Pow, span: Span::from(0..1) },
            ]
        );
    }

    #[test]
    fn test_compile_nested_expression() {
        // cos(1 + 2)
        let ast = (AstNode::Number { value: Complex::new(1.0, 0.0), span: Span::from(0..1) } + AstNode::Number { value: Complex::new(2.0, 0.0), span: Span::from(4..5) }).cos();
        let tokens = ast.compile();
        assert_eq!(
            tokens,
            vec![
                Token::Number { value: Complex::new(1.0, 0.0), span: Span::from(0..1) },
                Token::Number { value: Complex::new(2.0, 0.0), span: Span::from(4..5) },
                Token::BinaryOperator { kind: BinaryOperatorKind::Add, span: Span::from(0..1) },
                Token::Function { kind: FunctionKind::Cos, span: Span::from(0..1) },
            ]
        );
    }
}
