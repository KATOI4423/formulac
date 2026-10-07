//! # astnode/compile.rs
//!
//! Compiles an [`AstNode`] tree into a flat sequence of postfix [`Instruction`]s
//! (Reverse Polish Notation) for stack-based evaluation.
//!
//! ## Entry point
//! [`AstNode::compile`] performs a depth-first traversal of the AST
//! and emits instructions in evaluation order.
//! The resulting instruction sequence is consumed by the executor in [`builder`].

use num_complex::Complex;

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

/// A single executable instruction of the stack machine.
///
/// Unlike [`Token`](crate::token::Token), which exists only for the parser,
/// an `Instruction` carries no source span and no parser-only variants, so
/// the evaluator can `match` it exhaustively.
///
/// Instructions are emitted in postfix order by [`AstNode::compile`].
/// Each one consumes values from the evaluation stack and pushes exactly one
/// result, except the two leaf instructions which only push.
///
/// | Instruction                              | Pops      | Pushes | Net stack effect |
/// |------------------------------------------|-----------|--------|------------------|
/// | [`Constant`](Self::Constant)             | 0         | 1      | `+1`             |
/// | [`Argument`](Self::Argument)             | 0         | 1      | `+1`             |
/// | [`UnaryOperator`](Self::UnaryOperator)   | 1         | 1      | `0`              |
/// | [`BinaryOperator`](Self::BinaryOperator) | 2         | 1      | `-1`             |
/// | [`Function`](Self::Function)             | `arity`   | 1      | `1 - arity`      |
/// | [`UserFunction`](Self::UserFunction)     | `arity`   | 1      | `1 - arity`      |
///
/// A valid program leaves exactly one value on the stack, and its maximum
/// depth equals [`Program::stack_size`].
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Instruction<T: Real> {
    /// Pushes a clone of a numeric literal.
    Constant(Complex<T>),

    /// Pushes a clone of the evaluation argument at this position.
    ///
    /// The index is resolved from the argument names when the formula is
    /// parsed, so it is always within the `N` arguments of the formula.
    Argument(usize),

    /// Pops one value, applies the unary operator, and pushes the result.
    UnaryOperator(UnaryOperatorKind),

    /// Pops the right operand, then the left operand (the right one is on top),
    /// applies the binary operator, and pushes the result.
    BinaryOperator(BinaryOperatorKind),

    /// Pops `arity` values and pushes the result of a built-in function.
    ///
    /// The arguments are consumed in their original order: the first
    /// argument is the deepest of the popped values, so `pow(x, y)` has `y`
    /// on top of the stack.
    Function(FunctionKind),

    /// Pops `arity` values and pushes the result of a user-defined function.
    ///
    /// The argument order is the same as for [`Function`](Self::Function).
    UserFunction(UserFn<T>),
}

/// Analyzed executable (immutable and shareable)
#[derive(Debug, Clone)]
pub(crate) struct Program<T: Real> {
    code: Vec<Instruction<T>>,
    stack_size: usize,
}

impl<T: Real> Program<T> {
    pub(crate) fn new(code: Vec<Instruction<T>>, stack_size: usize) -> Self {
        Self {
            code,
            stack_size,
        }
    }

    pub(crate) fn code(&self) -> &[Instruction<T>] {
        &self.code
    }

    pub(crate) fn stack_size(&self) -> usize {
        self.stack_size
    }
}

impl<T: Real> AstNode<T> {
    /// Compiles the AST into a flat sequence of postfix `Token`s.
    pub(crate) fn compile(&self) -> Program<T> {
        let mut code = Vec::new();
        self.compile_into(&mut code);
        let stack_size = self.required_stack_size();
        Program::new(code, stack_size)
    }

    fn compile_into(&self, out: &mut Vec<Instruction<T>>) {
        match self {
            Self::Number { value, .. } => out.push(Instruction::Constant(value.clone())),
            Self::Argument { index, .. } => out.push(Instruction::Argument(*index)),
            Self::UnaryOperator { kind, expr, .. } => {
                expr.compile_into(out);
                out.push(Instruction::UnaryOperator(*kind));
            }
            Self::BinaryOperator { kind, left, right, .. } => {
                left.compile_into(out);
                right.compile_into(out);
                out.push(Instruction::BinaryOperator(*kind));
            }
            Self::FunctionCall { kind, args, .. } => {
                args.iter().for_each(|arg| arg.compile_into(out));
                out.push(Instruction::Function(*kind));
            }
            Self::UserFunctionCall { func, args, .. } => {
                args.iter().for_each(|arg| arg.compile_into(out));
                out.push(Instruction::UserFunction(func.clone()));
            }
            Self::Derivative { .. } => {
                unreachable!("Derivative nodes must be resolved before compile()")
            }
        }
    }

    /// Returns the minimum stack capacity required to evaluate this node.
    ///
    /// ## Note:
    /// Every AST node is assumed to consume the values produced by its children
    /// and leave exactly one value on the stack.
    pub(crate) fn required_stack_size(&self) -> usize {
        match self {
            Self::Number { .. } | Self::Argument { .. } => 1,
            Self::UnaryOperator { expr, .. } => expr.required_stack_size(),
            Self::BinaryOperator { left, right, .. } => {
                let lhs = left.required_stack_size();
                let rhs = 1 + right.required_stack_size(); // One value (rhs result of lhs) is already on the stack.
                lhs.max(rhs)
            },
            Self::Derivative { .. } => unreachable!("Derivative nodes must be resolved before required_stack_size()"),
            Self::FunctionCall { args, .. }
            | Self::UserFunctionCall { args, .. } => {
                args.iter().enumerate().map(|(index, arg)| {
                    index + arg.required_stack_size()
                })
                .max()
                .unwrap_or(1) // respond to no argument function, such as pi()
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
        let program = ast.compile();
        assert_eq!(program.code, vec![Instruction::Constant(Complex::new(1.0, 0.0))]);
        assert_eq!(program.stack_size, 1);
    }

    #[test]
    fn test_compile_argument() {
        let ast = AstNode::<f64>::Argument { index: 1, span: Span::from(0..1) };
        let program = ast.compile();
        assert_eq!(program.code, vec![Instruction::Argument(1)]);
        assert_eq!(program.stack_size, 1);
    }

    #[test]
    fn test_compile_unary_operator() {
        let ast = -AstNode::Number { value: Complex::new(1.0, 0.0), span: Span::from(2..3) };
        let program = ast.compile();
        assert_eq!(
            program.code,
            vec![Instruction::Constant(Complex::new(1.0, 0.0)), Instruction::UnaryOperator(UnaryOperatorKind::Negative)],
        );
        assert_eq!(program.stack_size, 1);
    }

    #[test]
    fn test_compile_binary_operator() {
        let ast = AstNode::Number { value: Complex::new(1.0, 0.0), span: Span::from(0..1) } + AstNode::Argument { index: 1, span: Span::from(4..5) };
        let program = ast.compile();
        assert_eq!(
            program.code,
            vec![
                Instruction::Constant(Complex::new(1.0, 0.0)),
                Instruction::Argument(1),
                Instruction::BinaryOperator(BinaryOperatorKind::Add),
            ]
        );
        assert_eq!(program.stack_size, 2);
    }

    #[test]
    fn test_compile_function_single_argument() {
        let ast = AstNode::Number { value: Complex::new(1.0, 0.0), span: Span::from(0..1) }.sin();
        let program = ast.compile();
        assert_eq!(program.code, vec![Instruction::Constant(Complex::new(1.0, 0.0)), Instruction::Function(FunctionKind::Sin)]);
        assert_eq!(program.stack_size, 1);
    }

    #[test]
    fn test_compile_function_multi_arguments() {
        let ast = AstNode::Number { value: Complex::new(2.0, 0.0), span: Span::from(0..1) }.pow(AstNode::Argument { index: 0, span: Span::from(4..5) });
        let program = ast.compile();
        assert_eq!(
            program.code,
            vec![
                Instruction::Constant(Complex::new(2.0, 0.0)),
                Instruction::Argument(0),
                Instruction::Function(FunctionKind::Pow),
            ]
        );
        assert_eq!(program.stack_size, 2);
    }

    #[test]
    fn test_compile_nested_expression() {
        // cos(1 + 2)
        let ast = (AstNode::Number { value: Complex::new(1.0, 0.0), span: Span::from(0..1) } + AstNode::Number { value: Complex::new(2.0, 0.0), span: Span::from(4..5) }).cos();
        let program = ast.compile();
        assert_eq!(
            program.code,
            vec![
                Instruction::Constant(Complex::new(1.0, 0.0)),
                Instruction::Constant(Complex::new(2.0, 0.0)),
                Instruction::BinaryOperator(BinaryOperatorKind::Add),
                Instruction::Function(FunctionKind::Cos),
            ]
        );
        assert_eq!(program.stack_size, 2);
    }

    #[test]
    fn test_required_stack_size_function_with_nested_arguments() {
        let ast = AstNode::Number {
            value: Complex::new(1.0, 0.0),
            span: Span::from(0..1),
        }
        .pow(
            AstNode::Number {
                value: Complex::new(2.0, 0.0),
                span: Span::from(3..4),
            } + AstNode::Number {
                value: Complex::new(3.0, 0.0),
                span: Span::from(7..8),
            } * AstNode::Number {
                value: Complex::new(4.0, 0.0),
                span: Span::from(11..12),
            },
        );

        let program = ast.compile();

        assert_eq!(program.stack_size, 4);
    }
}
