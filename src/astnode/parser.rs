//! # astnode/parser.rs
//!
//! Converts a slice of [`Lexeme`]s into an [`AstNode`] tree using the
//! Shunting-Yard algorithm.
//!
//! ## Responsibilities
//! - Tokenizing lexemes and classifying operators as unary or binary
//! - Managing the operator stack (precedence, associativity)
//! - Resolving function calls, user-defined functions, and `diff` expressions
//! - Returning a single root [`AstNode`] or a [`ParseError`]
//!
//! Symbolic differentiation invoked by `diff(...)` is delegated to
//! [`AstNode::differentiate`], which is implemented separately.

use std::rc::Rc;
use std::str::FromStr;

use crate::astnode::AstNode;
use crate::constants::Constants;
use crate::core::Real;
use crate::functions::Arity;
use crate::err::ParseError;
use crate::lexer::{
    Lexeme,
    Span,
};
use crate::operators::{
    BinaryOperatorKind,
    OperatorKind,
    UnaryOperatorKind,
};
use crate::token::{
    Token,
    UserFnTable,
};

impl<T: Real> AstNode<T> {
    /// Parses a slice of `Lexeme`s into an AST using a shunting-yard algorithm.
    pub fn from(
        lexemes:   &[Lexeme],
        args:      &[&str],
        constants: &Constants<T>,
        users:     &UserFnTable<T>,
    ) -> Result<Self, ParseError>
    where
        T: FromStr,
    {
        let (mut output, mut ops)
            = Self::process_tokens(lexemes, args, constants, users)?;
        Self::finalize_parsing(&mut output, &mut ops)
    }

    // ── shunting-yard helpers ────────────────────────────────────────────────

    fn process_tokens(
        lexemes: &[Lexeme],
        args: &[&str],
        constants: &Constants<T>,
        users: &UserFnTable<T>,
    ) -> Result<(Vec<Self>, Vec<Token<T>>), ParseError>
    where
        T: FromStr,
    {
        let mut output: Vec<Self>  = Vec::new();
        let mut ops:    Vec<Token<T>> = Vec::new();

        // Tracks whether the previous token produced a value,
        // used to distinguish unary from binary operators.
        let mut prev_is_value = false;

        for lexeme in lexemes {
            let token = Token::try_from(lexeme, args, constants, users)?;
            match token {
                Token::Number { value, span } => {
                    output.push(Self::Number { value, span });
                    prev_is_value = true;
                }
                Token::Argument { index, span } => {
                    output.push(Self::Argument { index, span });
                    prev_is_value = true;
                }
                Token::Operator { kind, span } => {
                    if prev_is_value {
                        Self::push_binary_op(&mut output, &mut ops, kind, span)?;
                    } else {
                        Self::push_unary_op(&mut ops, kind, span)?;
                    }
                    prev_is_value = false;
                }
                // Functions and diff are pushed onto the op stack;
                // they are resolved when their closing ')' is encountered.
                Token::DiffOperator { .. } | Token::Function { .. } | Token::UserFunction { .. } => {
                    ops.push(token);
                    prev_is_value = false;
                }
                // '(' resets prev_is_value so the next token
                // (e.g. `-` in `(-x)`) is treated as unary.
                Token::LParen { .. } => {
                    ops.push(token);
                    prev_is_value = false;
                }
                Token::RParen { .. } => {
                    Self::flush_until_lparen(&mut output, &mut ops, lexeme)?;
                    prev_is_value = true;
                }
                Token::Comma { .. } => {
                    Self::flush_until_lparen_keep(&mut output, &mut ops, lexeme)?;
                    prev_is_value = false;
                }
                _ => return Err(ParseError::InvalidFormula {
                    reason: format!("unexpected token from '{}'", lexeme.text()),
                    span: lexeme.span(),
                }),
            }
        }

        Ok((output, ops))
    }

    /// Pushes a unary operator onto the op stack.
    fn push_unary_op(ops: &mut Vec<Token<T>>, kind: OperatorKind, span: Span) -> Result<(), ParseError> {
        let kind = UnaryOperatorKind::try_from(kind)
            .map_err(|_| ParseError::InvalidFormula {
                reason: format!("unknown unary operator '{}'", kind),
                span,
            })?;
        ops.push(Token::UnaryOperator { kind, span });
        Ok(())
    }

    /// Pops higher-precedence binary operators from the stack, then pushes the new one.
    fn push_binary_op(
        output: &mut Vec<Self>,
        ops:    &mut Vec<Token<T>>,
        kind:   OperatorKind,
        span:   Span,
    ) -> Result<(), ParseError> {
        let oper = BinaryOperatorKind::try_from(kind)
            .map_err(|_| ParseError::InvalidFormula {
                reason: format!("unknown binary operator '{}'", kind),
                span,
            })?;

        // Shunting-yard precedence rule.
        while let Some(Token::BinaryOperator { kind: top, span }) = ops.last() {
            let should_pop = if oper.is_left_assoc() {
                top.precedence() >= oper.precedence()
            } else {
                top.precedence() > oper.precedence()
            };
            if !should_pop { break; }
            Self::apply_binary(output, *top, *span)?;
            ops.pop();
        }
        ops.push(Token::BinaryOperator { kind: oper, span });
        Ok(())
    }

    /// Drains the entire op stack at end-of-input.
    fn flush_all(
        output: &mut Vec<Self>,
        ops:    &mut Vec<Token<T>>,
    ) -> Result<(), ParseError> {
        while let Some(token) = ops.pop() {
            match token {
                Token::LParen { span } | Token::RParen { span } => {
                    return Err(ParseError::InvalidFormula {
                        reason: "mismatched parentheses".into(),
                        span
                    });
                }
                t => Self::apply_token(output, t)?,
            }
        }
        Ok(())
    }

    /// Pops and applies operators until a `(` is found, then resolves any
    /// pending function/diff call sitting just below the `(`.
    fn flush_until_lparen(
        output: &mut Vec<Self>,
        ops:    &mut Vec<Token<T>>,
        lex:    &Lexeme,
    ) -> Result<(), ParseError> {
        loop {
            match ops.pop() {
                Some(Token::LParen { .. }) => break,
                Some(t) => Self::apply_token(output, t)?,
                None => return Err(ParseError::InvalidFormula {
                    reason: "mismatched ')'".into(),
                    span: lex.span()
                }),
            }
        }
        // Check for a function/diff call sitting just below the '('.
        if let Some(top) = ops.pop() {
            match top {
                Token::Function { kind, span } => Self::apply_fn(output, kind.arity(), span, |args| Self::FunctionCall { kind, args, span })?,
                Token::UserFunction { func, span } => Self::apply_fn(output, func.arity(), span, |args| Self::UserFunctionCall { func, args, span })?,
                Token::DiffOperator { span } => Self::apply_diff(output, span)?,
                other => ops.push(other), // not a call; put it back
            }
        }
        Ok(())
    }

    /// Like `flush_until_lparen`, but leaves the `(` on the stack (used for `,`).
    fn flush_until_lparen_keep(
        output: &mut Vec<Self>,
        ops:    &mut Vec<Token<T>>,
        lex:    &Lexeme,
    ) -> Result<(), ParseError> {
        loop {
            match ops.last() {
                Some(Token::LParen { .. }) => return Ok(()),
                Some(_) => {
                    let t = ops.pop().unwrap();
                    Self::apply_token(output, t)?;
                }
                None => return Err(ParseError::InvalidFormula {
                    reason: "mismatched ','".into(),
                    span: lex.span(),
                }),
            }
        }
    }

    // ── token application ────────────────────────────────────────────────────

    /// Dispatches a single token to the appropriate `apply_*` function.
    fn apply_token(output: &mut Vec<Self>, token: Token<T>) -> Result<(), ParseError> {
        let span = token.span();
        match token {
            Token::UnaryOperator { kind, span }  => Self::apply_unary(output, kind, span),
            Token::BinaryOperator { kind, span } => Self::apply_binary(output, kind, span),
            Token::Function { kind, span } => Self::apply_fn(output, kind.arity(), span, |args| Self::FunctionCall { kind, args, span }),
            Token::UserFunction { func, span } => Self::apply_fn(output, func.arity(), span, |args| Self::UserFunctionCall { func, args, span }),
            Token::DiffOperator { span }  => Self::apply_diff(output, span),
            other => Err(ParseError::InvalidFormula {
                reason: format!("unexpected token in operator stack: {:?}", other),
                span,
            }),
        }
    }

    fn apply_unary(output: &mut Vec<Self>, op: UnaryOperatorKind, span: Span) -> Result<(), ParseError> {
        let expr = output.pop().ok_or(ParseError::InternalError {
            reason: format!("missing operand for unary '{}'", op),
            span
        })?;
        output.push(Self::UnaryOperator { kind: op, expr: Rc::new(expr), span });
        Ok(())
    }

    fn apply_binary(output: &mut Vec<Self>, op: BinaryOperatorKind, span: Span) -> Result<(), ParseError> {
        let right = output.pop().ok_or(ParseError::MissingRightOperator { operator: op.to_string(), span })?;
        let left  = output.pop().ok_or(ParseError::MissingLeftOperator  { operator: op.to_string(), span })?;
        output.push(Self::BinaryOperator { kind: op, left: Rc::new(left), right: Rc::new(right), span });
        Ok(())
    }

    /// Generic function-application helper shared by built-in and user-defined functions.
    ///
    /// Pops `arity` nodes from `output` in order and calls `make_node` to build the AST node.
    fn apply_fn<F>(
        output:    &mut Vec<Self>,
        arity:     usize,
        span: Span,
        make_node: F,
    ) -> Result<(), ParseError>
    where
        F: FnOnce(Vec<Rc<Self>>) -> Self,
    {
        if output.len() < arity {
            return Err(ParseError::MissingArgs { func: format!("<arity {}>", arity), span });
        }
        let start = output.len() - arity;
        let args  = output.drain(start..).map(Rc::new).collect();
        output.push(make_node(args));
        Ok(())
    }

    fn parse_diff_args(output: &mut Vec<Self>, span: Span) -> Result<(AstNode<T>, usize, usize), ParseError>
    {
        let top = output.pop().ok_or(ParseError::InvalidDerivative {
            span,
            reason: "missing argument (expected variable or order)".into(),
        })?;

        let (var_idx, order) = match top {
            // diff(f, x, n) — explicit order
            Self::Number { value: z, .. } => {
                if !z.im.is_zero() || !z.re.clone().fract().is_zero() {
                    return Err(ParseError::InvalidDerivativeOrder { span, order: format!("{:?}", z) });
                }
                let order = z.re.clone().to_i32();
                if order > i8::MAX as i32 {
                    return Err(ParseError::InvalidDerivativeOrder { span, order: format!("{:?}", z) });
                }
                let var = match output.pop() {
                    Some(Self::Argument { index, .. }) => index,
                    Some(other) => return Err(ParseError::InvalidDerivative {
                        span,
                        reason: format!("expected Argument before order, got {:?}", other),
                    }),
                    None => return Err(ParseError::InvalidDerivative {
                        span,
                        reason: "missing variable before order".into(),
                    }),
                };
                (var, order)
            }
            // diff(f, x) — default order 1
            Self::Argument { index, .. } => (index, 1),

            other => return Err(ParseError::InvalidDerivative {
                span,
                reason: format!("expected Argument or Number, got {:?}", other),
            }),
        };

        let expr = output.pop().ok_or(ParseError::InvalidDerivative {
            span,
            reason: "missing expression to differentiate".into(),
        })?;

        Ok((expr, var_idx, order as usize))
    }

    fn apply_diff(output: &mut Vec<Self>, span: Span) -> Result<(), ParseError> {
        let (mut expr, var, order) = Self::parse_diff_args(output, span)?;

        for _ in 0..order {
            expr = expr.differentiate(var)?;
        }
        output.push(expr);
        Ok(())
    }

    fn finalize_parsing(output: &mut Vec<Self>, ops: &mut Vec<Token<T>>) -> Result<Self, ParseError> {
        // Drain the remaining operator stack.
        Self::flush_all(output, ops)?;

        match output.len() {
            1 => Ok(output.pop().unwrap()),
            0 => Err(ParseError::WrongReturn("no AST node produced".into())),
            _ => Err(ParseError::WrongReturn("too many AST nodes remaining".into())),
        }
    }
}

#[cfg(test)]
mod astnode_tests {
    use std::collections::HashMap;
    use num_complex::Complex;

    use super::*;
    use crate::lexer;
    use crate::functions::{
        FunctionKind,
        UserFn,
    };

    type UserFnTable<T> = HashMap<String, UserFn<T>>;

    macro_rules! assert_astnode_eq {
        ($left:expr, $right:expr) => {{
            fn inner<T: Real>(left: &AstNode<T>, right: &AstNode<T>) {
                let epsilon = 1.0e-12;
                match (left, right) {
                    (AstNode::Number { value: lv, span: ls }, AstNode::Number { value: rv, span: rs }) => {
                        assert!((lv.re.clone() - rv.re.clone()).abs() < T::from_f64(epsilon));
                        assert!((lv.im.clone() - rv.im.clone()).abs() < T::from_f64(epsilon));
                        assert_eq!(ls, rs);
                    }
                    (AstNode::Argument { index: li, span: ls }, AstNode::Argument { index: ri, span: rs }) => {
                        assert_eq!(li, ri);
                        assert_eq!(ls, rs);
                    }
                    (AstNode::UnaryOperator { kind: lk, expr: le, span: ls }, AstNode::UnaryOperator { kind: rk, expr: re, span: rs }) => {
                        assert_eq!(lk, rk);
                        inner(le, re);
                        assert_eq!(ls, rs);
                    }
                    (AstNode::BinaryOperator { kind: lk, left: ll, right: lr, span: ls },
                    AstNode::BinaryOperator { kind: rk, left: rl, right: rr, span: rs }) => {
                        assert_eq!(lk, rk);
                        inner(ll, rl);
                        inner(lr, rr);
                        assert_eq!(ls, rs);
                    }
                    (AstNode::FunctionCall { kind: lk, args: la, span: ls },
                    AstNode::FunctionCall { kind: rk, args: ra, span: rs }) => {
                        assert_eq!(lk, rk);
                        assert_eq!(la.len(), ra.len());
                        for (a, b) in la.iter().zip(ra.iter()) {
                            inner(a, b);
                        }
                        assert_eq!(ls, rs);
                    }
                    (l, r) => panic!("AST nodes differ: left = {:?}, right = {:?}", l, r),
                }
            }
            inner(&$left, &$right);
        }};
    }

    #[test]
    fn test_single_number_astnode() {
        let lexemes = lexer::from("42");
        let ast = AstNode::from(&lexemes, &[], &Constants::new(), &UserFnTable::new()).unwrap();
        assert_astnode_eq!(ast, AstNode::Number { value: Complex::new(42.0, 0.0), span: Span::from(0..2) })
    }

    #[test]
    fn test_unary_operator_negative_astnode() {
        let lexemes = lexer::from("- 3");
        let ast = AstNode::from(&lexemes, &[], &Constants::new(), &UserFnTable::new()).unwrap();
        assert_astnode_eq!(ast, AstNode::UnaryOperator {
            kind: UnaryOperatorKind::Negative,
            expr: Rc::new(AstNode::Number { value: Complex::new(3.0, 0.0), span: Span::from(2..3) }),
            span: Span::from(0..1),
        });
    }

    #[test]
    fn test_binary_operator_precedence_astnode() {
        let lexemes = lexer::from("2 + 3 * 4");
        let ast = AstNode::from(&lexemes, &[], &Constants::new(), &UserFnTable::new()).unwrap();
        // expected: (2 + (3 * 4))
        assert_astnode_eq!(ast, AstNode::BinaryOperator {
            kind: BinaryOperatorKind::Add,
            left: Rc::new(AstNode::Number { value: Complex::from(2.0), span: Span::from(0..1) }),
            right: Rc::new(AstNode::BinaryOperator {
                kind: BinaryOperatorKind::Mul,
                left: Rc::new(AstNode::Number { value: Complex::from(3.0), span: Span::from(4..5) }),
                right: Rc::new(AstNode::Number { value: Complex::from(4.0), span: Span::from(8..9) }),
                span: Span::from(6..7)
            }),
            span: Span::from(2..3)
        });
    }

    #[test]
    fn test_parentheses_override_precedence_astnode() {
        let lexemes = lexer::from("( 2 + 3 ) * 4");
        let ast = AstNode::from(&lexemes, &[], &Constants::new(), &UserFnTable::new()).unwrap();
        // expected: ((2 + 3) * 4)
        match ast {
            AstNode::BinaryOperator { kind, left, right, .. } => {
                assert_eq!(kind, BinaryOperatorKind::Mul);
                assert_eq!(*right, AstNode::Number { value: Complex::new(4.0, 0.0), span: Span::from(12..13) });
                match *left {
                    AstNode::BinaryOperator { kind, .. } => assert_eq!(kind, BinaryOperatorKind::Add),
                    _ => panic!("Expected Add inside parentheses"),
                }
            }
            _ => panic!("Expected Mul node"),
        }
    }

    #[test]
    fn test_function_single_arg_astnode() {
        let lexemes = lexer::from("sin ( 0 )");
        let ast = AstNode::from(&lexemes, &[], &Constants::new(), &UserFnTable::new()).unwrap();
        assert_astnode_eq!(ast, AstNode::FunctionCall {
            kind: FunctionKind::Sin,
            args: vec![Rc::new(AstNode::Number { value: Complex::from(0.0), span: Span::from(6..7) })],
            span: Span::from(0..3),
        });
    }

    #[test]
    fn test_function_multiple_args_astnode() {
        let lexemes = lexer::from("pow ( 2 , 3 )");
        let ast = AstNode::from(&lexemes, &[], &Constants::new(), &UserFnTable::new()).unwrap();
        assert_astnode_eq!(ast, AstNode::FunctionCall {
            kind: FunctionKind::Pow,
            args: vec![
                Rc::new(AstNode::Number { value: Complex::from(2.0), span: Span::from(6..7) }),
                Rc::new(AstNode::Number { value: Complex::from(3.0), span: Span::from(10..11) }),
            ],
            span: Span::from(0..3),
        });

        let lexemes = lexer::from("pow ( sin(x) , 3 )");
        let ast = AstNode::from(&lexemes, &["x"], &Constants::new(), &UserFnTable::new()).unwrap();
        assert_astnode_eq!(ast, AstNode::FunctionCall {
            kind: FunctionKind::Pow,
            args: vec![
                Rc::new(AstNode::FunctionCall {
                    kind: FunctionKind::Sin,
                    args: vec![Rc::new(AstNode::Argument { index: 0, span: Span::from(10..11) })],
                    span: Span::from(6..9),
                }),
                Rc::new(AstNode::Number { value: Complex::from(3.0), span: Span::from(15..16) }),
            ],
            span: Span::from(0..3),
        });
    }

    #[test]
    fn test_imaginary_number_astnode() {
        let lexemes = lexer::from("5i");
        let ast = AstNode::from(&lexemes, &[], &Constants::new(), &UserFnTable::new()).unwrap();
        assert_eq!(ast, AstNode::Number { value: Complex::new(0.0, 5.0), span: Span::from(0..2) });
    }

    #[test]
    fn test_unknown_token_astnode_error() {
        let lexemes = lexer::from("@");
        let res = AstNode::from(&lexemes, &[], &Constants::<f64>::new(), &UserFnTable::new());
        assert!(res.is_err());
    }
}
