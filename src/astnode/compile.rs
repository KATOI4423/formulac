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
    Arity,
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
/// | [`PowiConst`](Self::PowiConst)           | 1         | 1      | `0`              |
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

    /// Pops one value, raises it to the compile-time integer power, pushes the result.
    PowiConst(i32),

    /// Pops `arity` values and pushes the result of a user-defined function.
    ///
    /// The argument order is the same as for [`Function`](Self::Function).
    UserFunction(UserFn<T>),
}

impl<T: Real> Instruction<T> {
    /// Returns the stack depth after executing this instruction.
    pub(crate) fn stack_size_after(&self, current_stack_size: usize) -> usize {
        match self {
            Self::Constant(..) | Self::Argument(..) => current_stack_size + 1,
            Self::UnaryOperator(..) => current_stack_size,
            Self::BinaryOperator(..) => current_stack_size - 1,
            // Add first, so a zero-arity function does not underflow
            Self::Function(kind) => current_stack_size + 1 - kind.arity(),
            Self::UserFunction(kind) => current_stack_size + 1 - kind.arity(),
            Self::PowiConst(..) => current_stack_size,
        }
    }
}

/// Analyzed executable (immutable and shareable)
#[derive(Debug, Clone)]
pub(crate) struct Program<T: Real> {
    code: Vec<Instruction<T>>,
    stack_size: usize,
}

impl<T: Real> Program<T> {
    pub(crate) fn new(code: Vec<Instruction<T>>) -> Self {
        let stack_size = Self::calc_stack_size(&code);

        Self {
            code,
            stack_size,
        }
    }

    /// Calculates the minimum stack capacity required to evaluate this code.
    fn calc_stack_size(code: &[Instruction<T>]) -> usize {
        let (mut cur, mut max) = (0, 0);
        for instruction in code {
            cur = instruction.stack_size_after(cur);
            max = max.max(cur);
        }

        debug_assert_eq!(cur, 1, "Final stack depth must be 1");
        max
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
        Program::new(code)
    }

    fn get_const_powi_exponent(&self) -> Option<i32> {
        let Self::FunctionCall { kind: FunctionKind::Powi, args, .. } = self else {
            return None;
        };
        let Self::Number { value, .. } = args[1].as_ref() else {
            return None;
        };
        if Self::is_i32_compatible(value) {
            Some(value.re.to_i32())
        } else {
            None
        }
    }

    /// Returns true if the leftmost factor of a product chain is the constant -1.
    fn has_leading_neg_one(&self) -> bool {
        match self {
            Self::BinaryOperator { kind: BinaryOperatorKind::Mul, left, .. } => {
                left.is_neg_one() || left.has_leading_neg_one()
            }
            _ => false,
        }
    }

    /// Compiles a product chain whose leading factor is -1, without that factor.
    fn compile_without_leading_neg_one(&self, out: &mut Vec<Instruction<T>>) {
        let Self::BinaryOperator { left, right, .. } = self else { unreachable!() };
        if left.is_neg_one() {
            right.compile_into(out);
        } else {
            left.compile_without_leading_neg_one(out);
            right.compile_into(out);
            out.push(Instruction::BinaryOperator(BinaryOperatorKind::Mul));
        }
    }

    fn compile_into(&self, out: &mut Vec<Instruction<T>>) {
        match self {
            Self::Number { value, .. } => out.push(Instruction::Constant(value.clone())),
            Self::Argument { index, .. } => out.push(Instruction::Argument(*index)),
            Self::UnaryOperator { kind: UnaryOperatorKind::Positive, expr, .. } => {
                expr.compile_into(out);
            }
            Self::UnaryOperator { kind, expr, .. } => {
                expr.compile_into(out);
                out.push(Instruction::UnaryOperator(*kind));
            }
            // a + (-1)*b ... => a - b ...
            Self::BinaryOperator { kind: BinaryOperatorKind::Add, left, right, .. }
                if right.has_leading_neg_one() =>
            {
                left.compile_into(out);
                right.compile_without_leading_neg_one(out);
                out.push(Instruction::BinaryOperator(BinaryOperatorKind::Sub));
            },
            // (-1)*a ... + b => b - a ...  (a constant is placed last by simplify, e.g. `1 - x`)
            Self::BinaryOperator { kind: BinaryOperatorKind::Add, left, right, .. }
                if left.has_leading_neg_one() =>
            {
                right.compile_into(out);
                left.compile_without_leading_neg_one(out);
                out.push(Instruction::BinaryOperator(BinaryOperatorKind::Sub));
            },
            Self::BinaryOperator { kind, left, right, .. } => {
                left.compile_into(out);
                right.compile_into(out);
                out.push(Instruction::BinaryOperator(*kind));
            }
            Self::FunctionCall { kind: FunctionKind::Powi, args, .. }
                if let Some(n) = self.get_const_powi_exponent() =>
            {
                args[0].compile_into(out);
                out.push(Instruction::PowiConst(n)); // Convert only once during compilation.
            },
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
}

#[cfg(test)]
mod stack_size_tests {
    use super::*;
    use crate::operators::{BinaryOperatorKind, UnaryOperatorKind};

    fn arg() -> Instruction<f64> {
        Instruction::Argument(0)
    }

    fn constant() -> Instruction<f64> {
        Instruction::Constant(Complex::ONE)
    }

    fn assert_stack_size(code: &[Instruction<f64>], expected: usize) {
        let prog = Program::new(Vec::from(code));
        assert_eq!(prog.stack_size(), expected);
    }

    #[test]
    fn test_calc_stack_size_single_value() {
        assert_stack_size(&[constant()], 1);
        assert_stack_size(&[arg()], 1);
    }

    #[test]
    fn test_calc_stack_size_unary_operator() {
        assert_stack_size(
            &[
                arg(),
                Instruction::UnaryOperator(UnaryOperatorKind::Positive),
            ],
            1,
        );
    }

    #[test]
    fn test_calc_stack_size_binary_operator() {
        assert_stack_size(
            &[
                arg(),
                arg(),
                Instruction::BinaryOperator(BinaryOperatorKind::Add),
            ],
            2,
        );
    }

    #[test]
    fn test_calc_stack_size_left_associative_expression() {
        // (a + b) * (c + d)
        assert_stack_size(
            &[
                arg(),
                arg(),
                Instruction::BinaryOperator(BinaryOperatorKind::Add),
                arg(),
                arg(),
                Instruction::BinaryOperator(BinaryOperatorKind::Add),
                Instruction::BinaryOperator(BinaryOperatorKind::Mul),
            ],
            3,
        );
    }

    #[test]
    fn test_calc_stack_size_right_nested_expression() {
        // a + (b * (c + d))
        assert_stack_size(
            &[
                arg(),
                arg(),
                arg(),
                arg(),
                Instruction::BinaryOperator(BinaryOperatorKind::Add),
                Instruction::BinaryOperator(BinaryOperatorKind::Mul),
                Instruction::BinaryOperator(BinaryOperatorKind::Add),
            ],
            4,
        );
    }

    #[test]
    fn test_calc_stack_size_function_call() {
        // sin(a)
        assert_stack_size(
            &[
                arg(),
                Instruction::Function(FunctionKind::Sin),
            ],
            1,
        );
    }

    #[test]
    fn test_calc_stack_size_nested_function_calls() {
        // sin(cos(a))
        assert_stack_size(
            &[
                arg(),
                Instruction::Function(FunctionKind::Cos),
                Instruction::Function(FunctionKind::Sin),
            ],
            1,
        );
    }

    #[test]
    fn test_calc_stack_size_powi_const() {
        // powi(a, 3) is compiled to a single argument followed by PowiConst.
        assert_stack_size(
            &[
                arg(),
                Instruction::PowiConst(3),
            ],
            1,
        );
    }

    #[test]
    fn test_calc_stack_size_optimized_subtraction() {
        // a + (-1) * b => a - b
        assert_stack_size(
            &[
                arg(),
                arg(),
                Instruction::BinaryOperator(BinaryOperatorKind::Sub),
            ],
            2,
        );
    }

    #[test]
    fn test_calc_stack_size_unary() {
        // (+x)
        assert_stack_size(
            &[
                arg(),
                Instruction::UnaryOperator(UnaryOperatorKind::Positive),
            ],
            1,
        );

        // (-x)
        assert_stack_size(
            &[
                arg(),
                Instruction::UnaryOperator(UnaryOperatorKind::Negative),
            ],
            1,
        );

        // x + (+y)
        assert_stack_size(
            &[
                arg(),
                constant(),
                Instruction::UnaryOperator(UnaryOperatorKind::Positive),
                Instruction::BinaryOperator(BinaryOperatorKind::Add),
            ],
            2,
        );
    }

    #[test]
    fn test_stack_size_arg0_userfn() {
        let func = UserFn::new("f", |[]| { Complex::<f64>::ONE });
        // 1 + f()
        assert_stack_size(
            &[
                constant(),
                Instruction::UserFunction(func),
                Instruction::BinaryOperator(BinaryOperatorKind::Add),
            ],
            2,
        );
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

    #[test]
    fn test_instruction_large_enum_variant() {
        assert!(size_of::<Instruction<f64>>() <= 32);
    }
}
