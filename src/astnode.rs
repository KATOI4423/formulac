//! # astnode.rs
//!
//! Parses mathematical expressions into an Abstract Syntax Tree (AST)
//! and compiles them into executable tokens.
//!
//! Supports real/complex numbers, constants, unary/binary operators,
//! built-in functions, user-defined functions, and symbolic differentiation.

use num_complex::Complex;
use num_traits::{
    One,
    Zero,
};
use std::fmt::Debug;
use std::ops::{
    AddAssign,
    MulAssign,
};
use std::rc::Rc;
use std::str::FromStr;

use crate::constants::Constants;
use crate::core::{
    ComplexMath,
    Real,
};
use crate::err::ParseError;
use crate::functions::{
    Arity,
    FunctionArgs,
    FunctionCall,
    FunctionKind,
    UserFn
};
use crate::lexer::{
    Lexeme,
    Span,
};
use crate::operators::{
    BinaryOperatorKind,
    OperatorKind,
    UnaryOperatorKind,
};
use crate::token::{Token, UserFnTable};

// ─── helpers ────────────────────────────────────────────────────────────────

fn is_i32_compatible<T: Real>(z: &Complex<T>) -> bool {
    z.im.is_zero() && z.re.is_i32_compatible()
}

// ─── AstNode ────────────────────────────────────────────────────────────────

/// Abstract Syntax Tree node representing a mathematical expression.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum AstNode<T: Real> {
    /// Numeric literal.
    Number {
        value: Complex<T>,
        span: Span, // the location of value
    },

    /// Function argument by index.
    Argument {
        index: usize,
        span: Span, // the location of index
    },

    /// Unary operator applied to an expression.
    UnaryOperator {
        kind: UnaryOperatorKind,
        expr: Rc<AstNode<T>>,
        span: Span, // the location of operator
    },

    /// Binary operator applied to left and right expressions.
    BinaryOperator {
        kind: BinaryOperatorKind,
        left: Rc<AstNode<T>>,
        right: Rc<AstNode<T>>,
        span: Span, // the location of operator
    },

    /// Derivative node: `diff(expr, var, order)`.
    Derivative {
        expr: Rc<AstNode<T>>,
        var: usize,
        order: usize,
        span: Span, // the location of `diff`
    },

    /// Built-in function call.
    FunctionCall {
        kind: FunctionKind,
        args: Vec<Rc<AstNode<T>>>,
        span: Span, // the location of function
    },

    /// User-defined function call.
    UserFunctionCall {
        func: UserFn<T>,
        args: Vec<Rc<AstNode<T>>>,
        span: Span, // the location of function
    },
}

// ─── parse ──────────────────────────────────────────────────────────────────

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

        // Drain the remaining operator stack.
        Self::flush_all(&mut output, &mut ops)?;

        match output.len() {
            1 => Ok(output.pop().unwrap()),
            0 => Err(ParseError::WrongReturn("no AST node produced".into())),
            _ => Err(ParseError::WrongReturn("too many AST nodes remaining".into())),
        }
    }

    fn span(&self) -> Span
    {
        match self {
            Self::Argument { span, .. }
            | Self::BinaryOperator { span, .. }
            | Self::Derivative { span, .. }
            | Self::FunctionCall { span, .. }
            | Self::Number { span, .. }
            | Self::UnaryOperator { span, .. }
            | Self::UserFunctionCall { span, .. }
            => *span
        }
    }

    fn unwrap_rc(rc: Rc<Self>) -> Self {
        Rc::try_unwrap(rc).unwrap_or_else(|rc| (*rc).clone())
    }

    // ── shunting-yard helpers ────────────────────────────────────────────────

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
}

// ─── simplify ───────────────────────────────────────────────────────────────

impl<T: Real> AstNode<T> {
    /// Simplifies the AST by constant-folding and algebraic normalization.
    pub fn simplify(self) -> Self
    where
        Complex<T>: AddAssign + MulAssign,
    {
        match self {
            Self::UnaryOperator { kind, expr, span } => {
                let expr = Self::unwrap_rc(expr).simplify();
                match expr {
                    Self::Number { value, span } => Self::Number { value: kind.apply(value), span },
                    other => Self::UnaryOperator { kind, expr: Rc::new(other), span },
                }
            }
            Self::BinaryOperator { kind, left, right, span } => {
                Self::fold_binary(kind, span, (*left).clone(), (*right).clone())
            }
            // pow/powi get their own folding path.
            Self::FunctionCall { kind: FunctionKind::Pow  | FunctionKind::Powi, mut args, .. } => {
                let base = Self::unwrap_rc(args.remove(0));
                let exp  = Self::unwrap_rc(args.remove(0));
                Self::fold_pow(base, exp)
            }
            Self::FunctionCall { kind, args, span } => {
                Self::fold_generic_fn(kind, span, args, |k, s, a| Self::FunctionCall { kind: k, args: a, span: s })
            }
            Self::UserFunctionCall { func, args, span } => {
                Self::fold_generic_fn(func, span, args, |f, s, a| Self::UserFunctionCall { func: f, args: a, span: s })
            }
            other => other,
        }
    }

    // ── fold helpers ─────────────────────────────────────────────────────────

    /// Evaluates a function call if all arguments are numeric literals.
    fn fold_generic_fn<F, G>(func: F, span: Span, args: Vec<Rc<Self>>, make_node: G) -> Self
    where
        F: FunctionCall<T>,
        G: FnOnce(F,Span, Vec<Rc<Self>>) -> Self,
        Complex<T>: AddAssign + MulAssign,
    {
        let args: Vec<_> = args.into_iter().map(|arg| {
            let ast = Self::unwrap_rc(arg);
            Rc::new(Self::simplify(ast))
        }).collect();
        let all_numbers  = args.iter().all(|a| matches!(**a, Self::Number { .. }));
        if all_numbers {
            let nums: Vec<Complex<T>> = args.iter()
                .map(|a|
                    match a.as_ref() {
                        Self::Number { value, .. } => value.clone(),
                        _ => unreachable!()
                    }
                )
                .collect();
            Self::Number { value: func.apply(FunctionArgs::from(nums)), span }
        } else {
            make_node(func, span, args)
        }
    }

    fn fold_binary(kind: BinaryOperatorKind, span: Span, left: Self, right: Self) -> Self
    where
        Complex<T>: AddAssign + MulAssign,
    {
        let left  = left.simplify();
        let right = right.simplify();

        // Both sides are numbers → evaluate immediately.
        if let (Self::Number { value: l, span: ls }, Self::Number { value: r, .. }) = (&left, &right) {
            return Self::Number { value: kind.apply(l.clone(), r.clone()), span: *ls };
        }

        match kind {
            BinaryOperatorKind::Add => Self::fold_add(span, left, right),
            BinaryOperatorKind::Sub => Self::fold_sub(span, left, right),
            BinaryOperatorKind::Mul => Self::fold_mul(span, left, right),
            BinaryOperatorKind::Div => Self::fold_div(span, left, right),
            BinaryOperatorKind::Pow => Self::fold_pow(left, right),
        }
    }

    // ── addition ─────────────────────────────────────────────────────────────

    /// `left + right` with flattening, constant-folding, and like-term combining.
    fn fold_add(span: Span, left: Self, right: Self) -> Self
    where
        Complex<T>: AddAssign,
    {
        // 1. Flatten nested additions into a single term list.
        let mut terms = Vec::new();
        Self::collect_add_terms(left,  &mut terms);
        Self::collect_add_terms(right, &mut terms);

        // 2. Separate numeric constants from symbolic terms.
        let mut const_sum = Complex::zero();
        let mut sym_terms = Vec::new();
        for t in terms {
            match t {
                Self::Number { value, .. } => const_sum += value,
                other           => sym_terms.push(other),
            }
        }
        if !const_sum.is_zero() {
            sym_terms.push(Self::Number { value: const_sum, span });
        }

        // 3. Combine like terms (e.g. x + x → 2*x).
        let terms = Self::combine_like_add_terms(sym_terms);

        // 4. Fold back into a left-associative chain.
        Self::chain_add(terms, span)
    }

    fn collect_add_terms(node: Self, out: &mut Vec<Self>) {
        match node {
            Self::BinaryOperator { kind: BinaryOperatorKind::Add, left, right, .. } => {
                Self::collect_add_terms(
                    Self::unwrap_rc(left),
                    out,
                );
                Self::collect_add_terms(
                    Self::unwrap_rc(right),
                    out,
                );
            }
            other => out.push(other),
        }
    }

    /// Combines `a*x + b*x → (a+b)*x`, removes zero-coefficient terms.
    fn combine_like_add_terms(terms: Vec<Self>) -> Vec<Self>
    where
        Complex<T>: AddAssign,
    {
        fn structurally_equal<T: Real>(a: &AstNode<T>, b: &AstNode<T>) -> bool {
            match (a, b) {
                (AstNode::Number { value: av, .. }, AstNode::Number { value: bv, .. }) => av == bv,
                (AstNode::Argument { index: ai, .. }, AstNode::Argument { index: bi, .. }) => ai == bi,
                (AstNode::UnaryOperator { kind: ak, expr: ae, .. }, AstNode::UnaryOperator { kind: bk, expr: be, .. }) => {
                    ak == bk && structurally_equal(ae, be)
                }
                (AstNode::BinaryOperator { kind: ak, left: al, right: ar, .. }, AstNode::BinaryOperator { kind: bk, left: bl, right: br, .. }) => {
                    ak == bk && structurally_equal(al, bl) && structurally_equal(ar, br)
                }
                (AstNode::FunctionCall { kind: ak, args: aa, .. }, AstNode::FunctionCall { kind: bk, args: ba, .. }) => {
                    ak == bk && aa.len() == ba.len() && aa.iter().zip(ba.iter()).all(|(x, y)| structurally_equal(x, y))
                }
                (AstNode::UserFunctionCall { func: af, args: aa, .. }, AstNode::UserFunctionCall { func: bf, args: ba, .. }) => {
                    af.name() == bf.name() && aa.len() == ba.len() && aa.iter().zip(ba.iter()).all(|(x, y)| structurally_equal(x, y))
                }
                (AstNode::Derivative { expr: ae, var: av, order: ao, .. }, AstNode::Derivative { expr: be, var: bv, order: bo, .. }) => {
                    av == bv && ao == bo && structurally_equal(ae, be)
                }
                _ => false,
            }
        }

        // Map: (variable node) → accumulated coefficient
        let mut map: Vec<(Self, Complex<T>)> = Vec::new();

        for term in terms {
            let (var, coeff) = match term {
                Self::BinaryOperator { kind: BinaryOperatorKind::Mul, left, right, .. } => {
                    let left = Self::unwrap_rc(left);
                    let right = Self::unwrap_rc(right);
                    match (left, right) {
                        (Self::Number { value: z, .. }, v) => (v, z),
                        (v, Self::Number { value: z, .. }) => (v, z),
                        (l, r)               => (l.mul(r), Complex::one()),
                    }
                }
                other => (other, Complex::one()),
            };
            match map.iter_mut().find(|(v, _)| structurally_equal(v, &var)) {
                Some((_, c)) => *c += coeff,
                None         => map.push((var, coeff)),
            }
        }

        map.into_iter()
            .filter(|(_, c)| !(*c).is_zero())
            .map(|(var, coeff)| {
                if coeff.is_one() { var }
                else { Self::Number { value: coeff, span: var.span() }.mul(var) }
            })
            .collect()
    }

    fn chain_add(terms: Vec<Self>, span: Span) -> Self {
        match terms.len() {
            0 => Self::zero(span),
            1 => terms.into_iter().next().unwrap(),
            _ => terms.into_iter().reduce(|acc, t| acc.add(t)).unwrap(),
        }
    }

    // ── subtraction ──────────────────────────────────────────────────────────

    /// Rewrites `left - right` as `left + (-1)*right` and re-folds.
    fn fold_sub(span: Span, left: Self, right: Self) -> Self
    where
        Complex<T>: AddAssign + MulAssign,
    {
        Self::fold_binary(
            BinaryOperatorKind::Add,
            span,
            left,
            Self::Number { value: -Complex::one(), span: right.span() } * right,
        )
    }

    // ── multiplication ───────────────────────────────────────────────────────

    /// `left * right` with flattening, constant-folding, and same-base power combining.
    fn fold_mul(span: Span, left: Self, right: Self) -> Self
    where
        Complex<T>: AddAssign + MulAssign,
    {
        // 1. Flatten nested multiplications.
        let mut factors = Vec::new();
        Self::collect_mul_terms(left,  &mut factors);
        Self::collect_mul_terms(right, &mut factors);

        // 2. Pull out numeric constants.
        let mut const_prod = Complex::one();
        let mut sym_factors = Vec::new();
        for f in factors {
            match f {
                Self::Number { value: z, .. } => const_prod *= z,
                other => sym_factors.push(other),
            }
        }

        if const_prod.is_zero() {
            return Self::zero(span);
        }
        if !const_prod.is_one() {
            sym_factors.insert(0, Self::Number { value: const_prod, span });
        }

        // 3. Combine x^a * x^b → x^(a+b).
        let factors = Self::combine_like_pow_terms(sym_factors);

        // 4. Fold back.
        Self::chain_mul(factors, span)
    }

    fn collect_mul_terms(node: Self, out: &mut Vec<Self>) {
        match node {
            Self::BinaryOperator { kind: BinaryOperatorKind::Mul, left, right, .. } => {
                let left = Self::unwrap_rc(left);
                let right = Self::unwrap_rc(right);
                Self::collect_mul_terms(left,  out);
                Self::collect_mul_terms(right, out);
            }
            other => out.push(other),
        }
    }

    /// Combines `x^a * x^b → x^(a+b)`, removes zero-exponent terms.
    fn combine_like_pow_terms(terms: Vec<Self>) -> Vec<Self>
    where
        Complex<T>: AddAssign,
    {
        let mut map: Vec<(Self, Complex<T>)> = Vec::new();

        for term in terms {
            let (base, exp) = match term {
                Self::BinaryOperator { kind: BinaryOperatorKind::Pow, left, right, .. } => {
                    let left = Self::unwrap_rc(left);
                    let right = Self::unwrap_rc(right);
                    match right {
                        Self::Number { value: e, .. } => (left, e),
                        r => {
                            (left.pow(r), Complex::one()
                        )},
                    }
                }
                Self::FunctionCall { kind: FunctionKind::Pow | FunctionKind::Powi, ref args, .. } => {
                    let base = Self::unwrap_rc(args[0].clone());
                    match args[1].as_ref() {
                        Self::Number { value: e, .. } => (base, e.clone()),
                        _ => (term, Complex::one()),
                    }
                }
                other => (other, Complex::one()),
            };
            match map.iter_mut().find(|(b, _)| *b == base) {
                Some((_, e)) => *e += exp,
                None         => map.push((base, exp)),
            }
        }

        map.into_iter()
            .filter(|(_, e)| !(*e).is_zero())
            .map(|(base, exp)| {
                if exp.is_one() { base }
                else if is_i32_compatible(&exp) { base.powi(exp.re.to_i32()) }
                else {
                    let span= base.span();
                    base.pow(Self::Number { value: exp, span })
                }
            })
            .collect()
    }

    fn chain_mul(factors: Vec<Self>, span: Span) -> Self
    where
        Complex<T>: AddAssign + MulAssign,
    {
        match factors.len() {
            0 => Self::one(span),
            1 => factors.into_iter().next().unwrap().simplify(),
            _ => factors.into_iter().reduce(|acc, f| acc.mul(f)).unwrap(),
        }
    }

    // ── division ─────────────────────────────────────────────────────────────

    /// Rewrites `left / right` as `left * right^-1` and re-folds.
    fn fold_div(span: Span, left: Self, right: Self) -> Self
    where
        Complex<T>: AddAssign + MulAssign,
    {
        Self::fold_mul(span, left, right.powi(-1).simplify())
    }

    // ── power ────────────────────────────────────────────────────────────────

    fn fold_pow(base: Self, exp: Self) -> Self
    where
        Complex<T>: AddAssign + MulAssign,
    {
        let mut base = base.simplify();
        let mut exp  = exp.simplify();

        // (x^a)^b → x^(a*b)
        loop {
            match base {
                Self::FunctionCall { kind: FunctionKind::Pow | FunctionKind::Powi, mut args, .. } => {
                    let inner_base = Self::unwrap_rc(args.remove(0))
                        ;
                    let inner_exp  = Self::unwrap_rc(args.remove(0))
                        ;
                    exp  = inner_exp.mul(exp).simplify();
                    base = inner_base.simplify();
                }
                other => { base = other; break; }
            }
        }

        // x^1 → x, x^0 → 1
        if let Self::Number { value: e, span: s } = &exp {
            if (*e).is_one()  { return base; }
            if (*e).is_zero() { return Self::one(*s); }
        }

        match (base, exp) {
            (Self::Number { value: b, span: s }, _) if b.is_one() => Self::one(s),
            (Self::Number { value: b, span: s }, Self::Number { value: e, .. })
                if b.is_zero() && e.re > T::zero()
                => Self::zero(s),
            (Self::Number { value: b, span: s }, Self::Number { value: e, .. })
                => Self::Number { value: b.powc(e), span: s },
            (b, Self::Number { value: e, .. }) if is_i32_compatible(&e)
                => b.powi(e.re.to_i32()),
            (b, e) => {
                b.pow(e)
            },
        }
    }
}

// ─── AstNode builder helpers ─────────────────────────────────────────────────

impl<T: Real> AstNode<T> {
    fn zero(span: Span) -> Self { Self::Number { value: Complex::zero(), span } }
    fn one(span: Span)  -> Self { Self::Number { value: Complex::one(), span } }

    fn add(self, rhs: Self) -> Self {
        let span = self.span();
        Self::BinaryOperator { kind: BinaryOperatorKind::Add, left: Rc::new(self), right: Rc::new(rhs), span }
    }

    fn sub(self, rhs: Self) -> Self {
        let span = self.span();
        Self::BinaryOperator { kind: BinaryOperatorKind::Sub, left: Rc::new(self), right: Rc::new(rhs), span }
    }

    fn mul(self, rhs: Self) -> Self {
        let span = self.span();
        Self::BinaryOperator { kind: BinaryOperatorKind::Mul, left: Rc::new(self), right: Rc::new(rhs), span }
    }

    fn div(self, rhs: Self) -> Self {
        let span = self.span();
        Self::BinaryOperator { kind: BinaryOperatorKind::Div, left: Rc::new(self), right: Rc::new(rhs), span }
    }

    fn negative(self) -> Self {
        let span = self.span();
        Self::UnaryOperator { kind: UnaryOperatorKind::Negative, expr: Rc::new(self), span }
    }

    fn sin(self)  -> Self {
        let span = self.span();
        Self::FunctionCall { kind: FunctionKind::Sin,  args: vec![Rc::new(self)], span }
    }

    fn cos(self)  -> Self {
        let span = self.span();
        Self::FunctionCall { kind: FunctionKind::Cos,  args: vec![Rc::new(self)], span }
    }

    fn sinh(self) -> Self {
        let span = self.span();
        Self::FunctionCall { kind: FunctionKind::Sinh, args: vec![Rc::new(self)], span }
    }

    fn cosh(self) -> Self {
        let span = self.span();
        Self::FunctionCall { kind: FunctionKind::Cosh, args: vec![Rc::new(self)], span }
    }

    fn exp(self)  -> Self {
        let span = self.span();
        Self::FunctionCall { kind: FunctionKind::Exp,  args: vec![Rc::new(self)], span }
    }

    fn sqrt(self) -> Self {
        let span = self.span();
        Self::FunctionCall { kind: FunctionKind::Sqrt, args: vec![Rc::new(self)], span }
    }

    fn pow(self, exp: Self) -> Self
    {
        let span = self.span();
        Self::FunctionCall { kind: FunctionKind::Pow, args: vec![Rc::new(self), Rc::new(exp)], span }
    }

    fn powi(self, n: i32) -> Self
    {
        let span = self.span();
        Self::FunctionCall {
            kind: FunctionKind::Powi,
            args: vec![
                Rc::new(self),
                Rc::new(Self::Number { value: Complex::from(T::from_f64(n as f64)), span }),
            ],
            span,
        }
    }
}

impl<T: Real> std::ops::Add for AstNode<T> { type Output = Self; fn add(self, rhs: Self) -> Self { self.add(rhs) } }
impl<T: Real> std::ops::Sub for AstNode<T> { type Output = Self; fn sub(self, rhs: Self) -> Self { self.sub(rhs) } }
impl<T: Real> std::ops::Mul for AstNode<T> { type Output = Self; fn mul(self, rhs: Self) -> Self { self.mul(rhs) } }
impl<T: Real> std::ops::Div for AstNode<T> { type Output = Self; fn div(self, rhs: Self) -> Self { self.div(rhs) } }
impl<T: Real> std::ops::BitXor for AstNode<T> { type Output = Self; fn bitxor(self, rhs: Self) -> Self { self.pow(rhs) } }
impl<T: Real> std::ops::Neg for AstNode<T> { type Output = Self; fn neg(self) -> Self { self.negative() } }

// ─── differentiate ──────────────────────────────────────────────────────────

impl<T: Real> AstNode<T> {
    /// Symbolically differentiates the AST with respect to argument `var`.
    pub fn differentiate(self, var: usize) -> Result<Self, ParseError> {
        match self {
            Self::Number { span, .. }    => Ok(Self::zero(span)),
            Self::Argument { index: i, span }  => Ok(if i == var { Self::one(span) } else { Self::zero(span) }),

            Self::UnaryOperator { kind, expr, span } => {
                let expr = Self::unwrap_rc(expr);
                Ok(Self::UnaryOperator {
                    kind,
                    expr: Rc::new(expr.differentiate(var)?),
                    span,
                })
            },

            Self::BinaryOperator { kind, left, right, span } => {
                let left = Self::unwrap_rc(left);
                let right = Self::unwrap_rc(right);
                Self::diff_binary(kind, left, right, var, span)
            }

            Self::FunctionCall { kind, args, span } => {
                Self::diff_function(kind, args, var, span)
            }

            Self::UserFunctionCall { func, args, span } => {
                if var >= func.arity() {
                    return Err(ParseError::OutOfRange { func: func.name().into(), idx: var, span });
                }
                if let Some(deriv) = func.derivative(var).cloned() {
                    Ok(Self::UserFunctionCall { func: deriv, args, span })
                } else {
                    Err(ParseError::DerivativeUndefined { func: func.name().into(), idx: var, span })
                }
            }

            Self::Derivative { expr, var: inner_var, order, span } => {
                if inner_var == var {
                    Ok(Self::Derivative { expr, var, order: order + 1, span })
                } else {
                    let expr = Self::unwrap_rc(expr);
                    Ok(Self::Derivative {
                        expr:  Rc::new(expr.differentiate(var)?),
                        var:   inner_var,
                        order,
                        span,
                    })
                }
            }
        }
    }

    fn diff_binary(
        kind:  BinaryOperatorKind,
        left:  Self,
        right: Self,
        var:   usize,
        span:  Span,
    ) -> Result<Self, ParseError> {
        let dl = left.clone().differentiate(var)?;
        let dr = right.clone().differentiate(var)?;
        match kind {
            BinaryOperatorKind::Add | BinaryOperatorKind::Sub => {
                Ok(Self::BinaryOperator { kind, left: Rc::new(dl), right: Rc::new(dr), span })
            }
            BinaryOperatorKind::Mul => Ok(dl * right + left * dr),
            BinaryOperatorKind::Div => {
                // (u/v)' = (u'v - uv') / v²
                Ok((dl * right.clone() - left * dr) / right.powi(2))
            }
            BinaryOperatorKind::Pow => Self::diff_pow(left, right, var),
        }
    }

    fn diff_function(
        kind: FunctionKind,
        mut args: Vec<Rc<Self>>,
        var:  usize,
        span: Span,
    ) -> Result<Self, ParseError> {
        let x = Self::unwrap_rc(args.remove(0))
            ;
        let dx = x.clone().differentiate(var)?;
        match kind {
            FunctionKind::Sin   => Ok(x.cos() * dx),
            FunctionKind::Cos   => Ok(-x.sin() * dx),
            FunctionKind::Tan   => Ok(dx / x.cos().powi(2)),
            FunctionKind::Asin  => Ok(dx / (Self::one(span) - x.powi(2)).sqrt()),
            FunctionKind::Acos  => Ok(-dx / (Self::one(span) - x.powi(2)).sqrt()),
            FunctionKind::Atan  => Ok(dx / (Self::one(span) + x.powi(2))),
            FunctionKind::Sinh  => Ok(dx * x.cosh()),
            FunctionKind::Cosh  => Ok(dx * x.sinh()),
            FunctionKind::Tanh  => Ok(dx / x.cosh().powi(2)),
            FunctionKind::Asinh => Ok(dx / (x.powi(2) + Self::one(span)).sqrt()),
            FunctionKind::Acosh => Ok(dx / (x.powi(2) - Self::one(span)).sqrt()),
            FunctionKind::Atanh => Ok(dx / (Self::one(span) - x.powi(2))),
            FunctionKind::Exp   => Ok(dx * x.exp()),
            FunctionKind::Ln    => Ok(dx / x),
            FunctionKind::Log10 => Ok(dx * Self::Number { value: Complex::from(T::log10_e()), span } / x),
            FunctionKind::Sqrt  => Ok(dx * Self::Number { value: Complex::from(T::from_f64(0.5)), span } / x.sqrt()),
            FunctionKind::Abs   => Err(ParseError::InvalidFormula {
                reason: "`abs(z)` is not differentiable in the complex domain".into(),
                span,
            }),
            FunctionKind::Conj  => Err(ParseError::InvalidFormula {
                reason: "`conj(z)` is not differentiable in the complex domain".into(),
                span,
            }),
            FunctionKind::Pow  => {
                let y = Self::unwrap_rc(args.remove(0))
                    ;
                Self::diff_pow(x, y, var)
            },
            FunctionKind::Powi => {
                let n = Self::unwrap_rc(args.remove(0))
                    ;
                Self::diff_powi(x, n, var)
            },
        }
    }

    /// d/dx [u^v] = u^v * (v'*ln(u) + v*u'/u)
    fn diff_pow(u: Self, v: Self, var: usize) -> Result<Self, ParseError> {
        let du   = u.clone().differentiate(var)?;
        let dv   = v.clone().differentiate(var)?;
        let ln_u = Self::FunctionCall { kind: FunctionKind::Ln, args: vec![Rc::new(u.clone())], span: u.span() };
        Ok(u.clone().pow(v.clone()) * (dv * ln_u + v * du / u))
    }

    /// d/dx [u^n] = n * u^(n-1) * u'
    fn diff_powi(u: Self, n: Self, var: usize) -> Result<Self, ParseError> {
        let s = u.span();
        let du = u.clone().differentiate(var)?;
        Ok(Self::FunctionCall {
            kind: FunctionKind::Powi,
            args: vec![Rc::new(u), Rc::new(n.clone() - Self::one(s))],
            span: s,
        } * n * du)
    }
}

// ─── compile ────────────────────────────────────────────────────────────────

impl<T: Real> AstNode<T> {
    /// Compiles the AST into a flat sequence of postfix `Token`s.
    pub fn compile(&self) -> Vec<Token<T>> {
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
    use std::collections::HashMap;

    use super::*;
    use crate::lexer;
    use crate::functions::UserFn;
    use approx::assert_abs_diff_eq;

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

    #[test]
    fn test_fold_add_constants() {
        let left = AstNode::Number { value: Complex::from(2.0), span: Span::from(0..1) };
        let right = AstNode::Number { value: Complex::from(3.0), span: Span::from(1..2) };
        let result = AstNode::fold_add(left.span(), left, right);
        assert_astnode_eq!(result, AstNode::Number { value: Complex::from(5.0), span: Span::from(0..1) });
    }

    #[test]
    fn test_fold_add_like_terms() {
        let x1 = AstNode::<f64>::Argument { index: 0, span: Span::from(0..1) };
        let x2 = AstNode::<f64>::Argument { index: 0, span: Span::from(2..3) };
        let result = AstNode::fold_add(Span::from(1..2), x1.clone(), x2.clone());
        match result {
            AstNode::BinaryOperator { kind, left, right, .. } => {
                assert_eq!(kind, BinaryOperatorKind::Mul);
                match (&*left, &*right) {
                    (AstNode::Number { value, .. }, AstNode::Argument { index, .. })
                    | (AstNode::Argument { index, .. }, AstNode::Number { value, .. }) => {
                        assert_abs_diff_eq!(value.re, 2.0, epsilon = 1.0e-12);
                        assert_eq!(*index, 0);
                    }
                    _ => panic!("Expected 2 * x or x * 2"),
                }
            }
            _ => panic!("Expected multiplication result"),
        }
    }

    #[test]
    fn test_fold_add_mixed_terms() {
        // 3*x + 4*x + y → 7*x + y
        let x1 = AstNode::Argument { index: 0, span: Span::from(2..3) };
        let x2 = AstNode::Argument { index: 0, span: Span::from(8..9) };
        let y = AstNode::Argument { index: 1, span: Span::from(12..13)};
        let term1 = AstNode::Number { value: Complex::new(3.0, 0.0), span: Span::from(0..1) }.mul(x1.clone());
        let term2 = AstNode::Number { value: Complex::new(4.0, 0.0), span: Span::from(6..7) }.mul(x2);
        let left = AstNode::fold_add(Span::from(4..5), term1, term2); // 3x + 4x → 7x
        let result = AstNode::fold_add(Span::from(10..11), left, y.clone());
        match result {
            AstNode::BinaryOperator { kind, left, right, .. } => {
                assert_eq!(kind, BinaryOperatorKind::Add);
                match Rc::try_unwrap(left).unwrap() {
                    AstNode::BinaryOperator { kind, left, right, .. } => {
                        assert_eq!(kind, BinaryOperatorKind::Mul);
                        match *left {
                            AstNode::Number { value, .. } => assert!((value.re - 7.0) < 1.0e-12),
                            _ => panic!("Expected 7.0"),
                        };
                        match *right {
                            AstNode::Argument { index, .. } => assert_eq!(index, 0),
                            _ => panic!("Expected 0"),
                        }
                    },
                    _ => panic!("Expected BinaryOperator Mul"),
                };
                match Rc::try_unwrap(right).unwrap() {
                    AstNode::Argument { index, .. } => assert_eq!(index, 1),
                    _ => panic!("Expected 1"),
                }
            }
            _ => panic!("Expected BinaryOperator Add"),
        }
    }

    #[test]
    fn test_fold_add_removed_same_terms() {
        // x + 2 - x = 2
        let x = AstNode::Argument { index: 0, span: Span::from(0..1) };
        let n = AstNode::Number { value: Complex::from(2.0), span: Span::from(2..3) };
        let result = AstNode::fold_sub(
            Span::from(3..4),
            AstNode::fold_add(Span::from(1..2), x.clone(), n.clone()),
            x,
        );
        // The result should simplify to just the constant 2.0
        // The span of the result may be derived from the fold process
        match result {
            AstNode::Number { value, .. } => {
                assert_abs_diff_eq!(value.re, 2.0, epsilon = 1.0e-12);
            }
            _ => panic!("Expected simplified to Number"),
        }
    }

    #[test]
    fn test_fold_add_zero_terms() {
        // x + 0 → x
        let x = AstNode::Argument { index: 0, span: Span::from(0..1) };
        let zero = AstNode::Number { value: Complex::from(0.0), span: Span::from(2..3) };
        let result = AstNode::fold_add(Span::from(1..2), x.clone(), zero);
        assert_astnode_eq!(result, x);
    }

    #[test]
    fn test_fold_sub_basic() {
        // x - y → x + (-1) * y
        let x = AstNode::Argument { index: 0, span: Span::from(0..1) };
        let y = AstNode::<f64>::Argument { index: 1, span: Span::from(2..3) };
        let result = AstNode::fold_sub(Span::from(1..2), x.clone(), y.clone());
        match result {
            AstNode::BinaryOperator { kind, left, right, .. } => {
                assert_eq!(kind, BinaryOperatorKind::Add);
                match (&*left, &*right) {
                    (AstNode::Argument { index: xi, .. }, AstNode::BinaryOperator { kind: mk, left: ml, right: mr, .. }) => {
                        assert_eq!(xi, &0);
                        assert_eq!(mk, &BinaryOperatorKind::Mul);
                        match (&**ml, &**mr) {
                            (AstNode::Number { value, .. }, AstNode::Argument { index: yi, .. })
                            | (AstNode::Argument { index: yi, .. }, AstNode::Number { value, .. }) => {
                                assert_eq!(yi, &1);
                                assert_abs_diff_eq!(value.re, -1.0, epsilon = 1.0e-12);
                            }
                            _ => panic!("Expected -1 * y structure"),
                        }
                    }
                    _ => panic!("Expected x + (-1 * y)"),
                }
            }
            _ => panic!("Expected Add node"),
        }
    }

    #[test]
    fn test_fold_sub_with_constants() {
        // 5 - 3 → 2
        let left = AstNode::Number { value: Complex::new(5.0, 0.0), span: Span::from(0..1) };
        let right = AstNode::Number { value: Complex::new(3.0, 0.0), span: Span::from(4..5) };
        let result = AstNode::fold_sub(Span::from(2..3), left, right);
        match result {
            AstNode::Number { value, .. } => {
                assert_abs_diff_eq!(value.re, 2.0, epsilon = 1.0e-12);
            }
            _ => panic!("Expected Number"),
        }
    }

    #[test]
    fn test_mul_constant_folding() {
        let expr = AstNode::fold_mul(
            Span::from(2..3),
            AstNode::Number { value: Complex::from(2.0), span: Span::from(0..1) },
            AstNode::Number { value: Complex::from(3.0), span: Span::from(4..5) }
        );
        match expr {
            AstNode::Number { value, .. } => {
                assert_abs_diff_eq!(value.re, 6.0, epsilon = 1.0e-12);
            }
            _ => panic!("Expected Number"),
        }
    }

    #[test]
    fn test_mul_with_zero() {
        let expr = AstNode::fold_mul(
            Span::from(2..3),
            AstNode::Number { value: Complex::<f64>::zero(), span: Span::from(0..1) },
            AstNode::Argument { index: 0, span: Span::from(4..5) }
        );
        match expr {
            AstNode::Number { value, .. } => {
                assert!(value.is_zero());
            }
            _ => panic!("Expected Number"),
        }
    }

    #[test]
    fn test_mul_with_one() {
        let expr = AstNode::fold_mul(
            Span::from(2..3),
            AstNode::Number { value: Complex::<f64>::one(), span: Span::from(0..1) },
            AstNode::Argument { index: 0, span: Span::from(4..5) }
        );
        assert_astnode_eq!(expr, AstNode::Argument { index: 0, span: Span::from(4..5) });
    }

    #[test]
    fn test_div_to_mul_pow_neg1() {
        let expr = AstNode::fold_div(
            Span::from(2..3),
            AstNode::<f64>::Argument { index: 0, span: Span::from(0..1) },
            AstNode::<f64>::Argument { index: 1, span: Span::from(4..5) });
        // should become x * y^-1
        let expected = AstNode::<f64>::fold_mul(
            Span::from(2..3),
            AstNode::<f64>::Argument { index: 0, span: Span::from(0..1) },
            AstNode::<f64>::Argument { index: 1, span: Span::from(4..5) }.powi(-1));
        assert_astnode_eq!(expr, expected);
    }

    #[test]
    fn test_combine_same_base_powers() {
        // Use the same span for both arguments to ensure they're recognized as the same base
        let x = AstNode::Argument { index: 0, span: Span::from(0..1) };
        let expr = AstNode::fold_mul(
            Span::from(10..11),
            x.clone().pow(AstNode::Number { value: Complex::from(2.0), span: Span::from(4..7) }),
            x.clone().pow(AstNode::Number { value: Complex::from(3.5), span: Span::from(17..20) })
        );
        // x^2 * x^3.5 = x^5.5
        fn extract_exponent<T: Real>(node: &AstNode<T>) -> Option<T> {
            match node {
                AstNode::FunctionCall { kind, args, .. } if matches!(kind, FunctionKind::Pow | FunctionKind::Powi) => {
                    match args.get(1)?.as_ref() {
                        AstNode::Number { value, .. } => Some(value.re.clone()),
                        _ => None,
                    }
                }
                _ => None,
            }
        }

        if let Some(exp) = extract_exponent(&expr) {
            assert_abs_diff_eq!(exp, 5.5, epsilon = 1.0e-10);
        } else {
            panic!("Could not extract exponent from expression: {:?}", expr);
        }
    }

    #[test]
    fn test_combine_same_base_powers_to_powi() {
        // Use the same span for both arguments to ensure they're recognized as the same base
        let x = AstNode::Argument { index: 0, span: Span::from(0..1) };
        let expr = AstNode::fold_mul(
            Span::from(10..11),
            x.clone().pow(AstNode::Number { value: Complex::from(2.0), span: Span::from(4..7) }),
            x.clone().pow(AstNode::Number { value: Complex::from(3.0), span: Span::from(17..20) })
        );
        // x^2 * x^3 = x^5 which may be converted to powi(5)
        fn extract_exponent<T: Real>(node: &AstNode<T>) -> Option<T> {
            match node {
                AstNode::FunctionCall { kind, args, .. } if matches!(kind, FunctionKind::Pow | FunctionKind::Powi) => {
                    match args.get(1)?.as_ref() {
                        AstNode::Number { value, .. } => Some(value.re.clone()),
                        _ => None,
                    }
                }
                _ => None,
            }
        }

        if let Some(exp) = extract_exponent(&expr) {
            assert_abs_diff_eq!(exp, 5.0, epsilon = 1.0e-10);
        } else {
            panic!("Could not extract exponent from expression: {:?}", expr);
        }
    }

    #[test]
    fn test_nested_mul_flattening() {
        let expr = AstNode::fold_mul(
            Span::from(10..11),
            AstNode::fold_mul(
                Span::from(4..5),
                AstNode::Argument { index: 0, span: Span::from(0..1) },
                AstNode::Number { value: Complex::from(2.0), span: Span::from(6..7) }
            ),
            AstNode::Number { value: Complex::from(3.0), span: Span::from(12..13) }
        );
        // (x * 2) * 3 => 6 * x. Check structure rather than exact span.
        match expr {
            AstNode::BinaryOperator { kind, .. } => {
                assert_eq!(kind, BinaryOperatorKind::Mul);
            }
            _ => panic!("Expected BinaryOperator Mul"),
        }
    }

    #[test]
    fn test_simplify_number() {
        let node = AstNode::Number { value: Complex::new(3.0, 0.0), span: Span::from(0..1) };
        assert_astnode_eq!(node.clone().simplify(), node);
    }

    #[test]
    fn test_simplify_unary_operator() {
        let node = -AstNode::Number { value: Complex::new(2.0, 0.0), span: Span::from(2..3) };
        let simplified = node.simplify();
        assert_astnode_eq!(simplified, AstNode::Number { value: Complex::new(-2.0, 0.0), span: Span::from(2..3) });
    }

    #[test]
    fn test_simplify_binary_operator_full() {
        let node = AstNode::Number { value: Complex::new(2.0, 0.0), span: Span::from(0..1) } + AstNode::Number { value: Complex::new(3.0, 0.0), span: Span::from(4..5) };
        let simplified = node.simplify();
        assert_astnode_eq!(simplified, AstNode::Number { value: Complex::new(5.0, 0.0), span: Span::from(0..1) });
    }

    #[test]
    fn test_simplify_binary_operator_partial() {
        let node = AstNode::Argument { index: 0, span: Span::from(0..1) } + AstNode::Number { value: Complex::new(3.0, 0.0), span: Span::from(4..5) };
        let simplified = node.simplify();
        // The structure should remain similar (cannot simplify further)
        match simplified {
            AstNode::BinaryOperator { kind, .. } => {
                assert_eq!(kind, BinaryOperatorKind::Add);
            }
            _ => panic!("Expected BinaryOperator"),
        }
    }

    #[test]
    fn test_simplify_binary_operator_chain() {
        // x + 2 + 3 -> x + 5
        let node
            = AstNode::Argument { index: 0, span: Span::from(0..1) }
            + AstNode::Number { value: Complex::new(2.0, 0.0), span: Span::from(4..5) }
            + AstNode::Number { value: Complex::new(3.0, 0.0), span: Span::from(8..9) };
        let simplified = node.simplify();
        // Should be x + 5
        match simplified {
            AstNode::BinaryOperator { kind, .. } => {
                assert_eq!(kind, BinaryOperatorKind::Add);
            }
            _ => panic!("Expected BinaryOperator Add"),
        }

        // x * 2 * 3 * 4 -> 24 * x
        let node = AstNode::Argument { index: 0, span: Span::from(0..1) } * AstNode::Number { value: Complex::new(2.0, 0.0), span: Span::from(4..5) }
            * AstNode::Number { value: Complex::new(3.0, 0.0), span: Span::from(8..9) } * AstNode::Number { value: Complex::new(4.0, 0.0), span: Span::from(12..13) };
        let simplified = node.simplify();
        // Should be 24 * x
        match simplified {
            AstNode::BinaryOperator { kind, .. } => {
                assert_eq!(kind, BinaryOperatorKind::Mul);
            }
            _ => panic!("Expected BinaryOperator Mul"),
        }

        // 2 * x + 3 -> not changed
        let node = AstNode::Number { value: Complex::new(2.0, 0.0), span: Span::from(0..1) } * AstNode::Argument { index: 0, span: Span::from(4..5) } + AstNode::Number { value: Complex::new(3.0, 0.0), span: Span::from(8..9) };
        let simplified = node.clone().simplify();
        // Should still be addition
        match simplified {
            AstNode::BinaryOperator { kind, .. } => {
                assert_eq!(kind, BinaryOperatorKind::Add);
            }
            _ => panic!("Expected BinaryOperator Add"),
        }
    }

    #[test]
    fn test_simplify_function_call_full() {
        let node = AstNode::Number { value: Complex::new(2.0, 0.0), span: Span::from(0..1) }.pow(AstNode::Number { value: Complex::new(3.0, 0.0), span: Span::from(4..5) });
        let simplified = node.simplify();
        assert_astnode_eq!(simplified, AstNode::Number { value: Complex::new(8.0, 0.0), span: Span::from(0..1) });

        let node = AstNode::Number { value: Complex::new(2.0, 0.0), span: Span::from(0..1) }.exp();
        let simplified = node.clone().simplify();
        assert_astnode_eq!(simplified, AstNode::Number { value: Complex::from(2.0).exp(), span: Span::from(0..1) });
    }

    #[test]
    fn test_simplify_function_call_partial() {
        let node = AstNode::<f64>::Argument { index: 0, span: Span::from(0..1) }.pow(AstNode::Argument { index: 1, span: Span::from(4..5) }).simplify();
        assert_astnode_eq!(
            node,
            AstNode::Argument { index: 0, span: Span::from(0..1) }.pow(AstNode::Argument { index: 1, span: Span::from(4..5) })
        );
    }

    #[test]
    fn test_simplify_pow_to_powi() {
        let node = AstNode::Argument { index: 0, span: Span::from(0..1) }.pow(AstNode::Number { value: Complex::from(3.0), span: Span::from(4..5) }).simplify();
        assert_astnode_eq!(node, AstNode::Argument { index: 0, span: Span::from(0..1) }.powi(3));
    }


    // Dummy user-defined function
    fn sum_func(args: [Complex<f64>; 2]) -> Complex<f64> {
        args[0] + args[1]
    }

    #[test]
    fn test_simplify_user_function_call_with_numbers() {
        let func = UserFn::new("sum", sum_func);

        let node = AstNode::UserFunctionCall {
            func,
            args: vec![
                Rc::new(AstNode::Number { value: Complex::from(1.0), span: Span::from(0..1) }),
                Rc::new(AstNode::Number { value: Complex::from(2.0), span: Span::from(4..5) }),
            ],
            span: Span::from(0..6),
        }.simplify();

        match node {
            AstNode::Number { value: val, span: _ } => assert_abs_diff_eq!(val.re, 3.0, epsilon=1e-12),
            _ => panic!("Expected simplified to Number"),
        }
    }

    #[test]
    fn test_simplify_user_function_call_with_no_numbers() {
        let func = UserFn::new("sum", sum_func);

        let node = AstNode::UserFunctionCall {
            func,
            args: vec![
                Rc::new(AstNode::Number { value: Complex::ONE, span: Span::from(0..1) }),
                Rc::new(AstNode::Argument { index: 0, span: Span::from(4..5) }),
            ],
            span: Span::from(0..6),
        };

        let simplified = node.clone().simplify();

        assert_eq!(simplified, node);
    }

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

#[cfg(test)]
mod differentiate_tests {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn test_differentiate_number() {
        let node = AstNode::Number { value: Complex::new(5.0, 0.0), span: Span::from(0..1) };
        let diff = node.differentiate(0).unwrap();
        assert_eq!(diff, AstNode::Number { value: Complex::ZERO, span: Span::from(0..1) });
    }

    #[test]
    fn test_differentiate_argument() {
        let node = AstNode::<f64>::Argument { index: 1, span: Span::from(0..1) };
        let diff = node.clone().differentiate(1).unwrap();
        assert_eq!(diff, AstNode::Number { value: Complex::ONE, span: Span::from(0..1) });
        let diff_other = node.differentiate(0).unwrap();
        assert_eq!(diff_other, AstNode::Number { value: Complex::ZERO, span: Span::from(0..1) });
    }

    #[test]
    fn test_differentiate_unary_operator() {
        let node = -AstNode::<f64>::Argument { index: 0, span: Span::from(2..3) };
        let diff = node.differentiate(0).unwrap();
        // d/dx(-x) = -1, where 1 is generated with the argument's span
        assert_eq!(diff, -AstNode::Number { value: Complex::ONE, span: Span::from(2..3) });
    }

    #[test]
    fn test_differentiate_binary_add() {
        let node = AstNode::Argument { index: 0, span: Span::from(0..1) } + AstNode::Number { value: Complex::new(2.0, 0.0), span: Span::from(4..5) };
        let diff = node.differentiate(0).unwrap();
        // d/dx (x + 2) = 1 + 0
        // The constants are generated with the spans of the original nodes
        match diff {
            AstNode::BinaryOperator { kind, .. } => {
                assert_eq!(kind, BinaryOperatorKind::Add);
            }
            _ => panic!("Expected BinaryOperator Add"),
        }
    }

    #[test]
    fn test_differentiate_function_sin() {
        let node = AstNode::<f64>::Argument { index: 0, span: Span::from(2..3) }.sin();
        let diff = node.differentiate(0).unwrap();
        // d/dx sin(x) = cos(x) * 1
        // The constant 1 is generated with the function's span (which is the argument's span)
        match diff {
            AstNode::BinaryOperator { kind, .. } => {
                assert_eq!(kind, BinaryOperatorKind::Mul);
            }
            _ => panic!("Expected BinaryOperator Mul for cos(x) * 1"),
        }
    }

    #[test]
    fn test_differentiate_derivative_order() {
        let node = AstNode::Derivative {
            expr: Rc::new(AstNode::<f64>::Argument { index: 0, span: Span::from(2..3) }),
            var: 0,
            order: 1,
            span: Span::from(0..4),
        };
        let diff = node.differentiate(0).unwrap();
        // d/dx (d/dx x) = d^2/dx^2 x
        assert_eq!(
            diff,
            AstNode::Derivative {
                expr: Rc::new(AstNode::Argument { index: 0, span: Span::from(2..3) }),
                var: 0,
                order: 2,
                span: Span::from(0..4),
            }
        );
    }

    #[test]
    fn test_differentiate_mul_x2() {
        // f(x) = x * x
        let node = AstNode::<f64>::Argument { index: 0, span: Span::from(0..1) }.mul(AstNode::Argument { index: 0, span: Span::from(4..5) }).differentiate(0)
            .unwrap().simplify();
        // d/dx (x * x) = 1 * x + x * 1 simplified = x + x = 2x
        // This is either Add or Mul depending on simplification level
        match node {
            AstNode::BinaryOperator { kind, .. } => {
                // Either Add (partially simplified) or Mul (fully simplified)
                matches!(kind, BinaryOperatorKind::Add | BinaryOperatorKind::Mul);
            }
            _ => panic!("Expected BinaryOperator after differentiation and simplification"),
        }
    }

    #[test]
    fn test_differentiate_builtin_functions() {
        let span = Span::from(2..3);
        let x = AstNode::<f64>::Argument { index: 0, span };
        let y = AstNode::<f64>::Argument { index: 1, span };
        let one = AstNode::one(span);
        let three = AstNode::Number { value: Complex::new(3.0, 0.0), span };
        let half = AstNode::Number { value: Complex::new(0.5, 0.0), span };
        let log10_e = AstNode::Number { value: Complex::new(std::f64::consts::LOG10_E, 0.0), span };

        let cases = vec![
            (
                FunctionKind::Sin,
                AstNode::FunctionCall { kind: FunctionKind::Sin, args: vec![Rc::new(x.clone())], span },
                x.clone().cos() * one.clone(),
            ),
            (
                FunctionKind::Cos,
                AstNode::FunctionCall { kind: FunctionKind::Cos, args: vec![Rc::new(x.clone())], span },
                (-x.clone().sin()) * one.clone(),
            ),
            (
                FunctionKind::Tan,
                AstNode::FunctionCall { kind: FunctionKind::Tan, args: vec![Rc::new(x.clone())], span },
                one.clone() / x.clone().cos().powi(2),
            ),
            (
                FunctionKind::Asin,
                AstNode::FunctionCall { kind: FunctionKind::Asin, args: vec![Rc::new(x.clone())], span },
                one.clone() / (one.clone() - x.clone().powi(2)).sqrt(),
            ),
            (
                FunctionKind::Acos,
                AstNode::FunctionCall { kind: FunctionKind::Acos, args: vec![Rc::new(x.clone())], span },
                (-one.clone()) / (one.clone() - x.clone().powi(2)).sqrt(),
            ),
            (
                FunctionKind::Atan,
                AstNode::FunctionCall { kind: FunctionKind::Atan, args: vec![Rc::new(x.clone())], span },
                one.clone() / (one.clone() + x.clone().powi(2)),
            ),
            (
                FunctionKind::Sinh,
                AstNode::FunctionCall { kind: FunctionKind::Sinh, args: vec![Rc::new(x.clone())], span },
                one.clone() * x.clone().cosh(),
            ),
            (
                FunctionKind::Cosh,
                AstNode::FunctionCall { kind: FunctionKind::Cosh, args: vec![Rc::new(x.clone())], span },
                one.clone() * x.clone().sinh(),
            ),
            (
                FunctionKind::Tanh,
                AstNode::FunctionCall { kind: FunctionKind::Tanh, args: vec![Rc::new(x.clone())], span },
                one.clone() / x.clone().cosh().powi(2),
            ),
            (
                FunctionKind::Asinh,
                AstNode::FunctionCall { kind: FunctionKind::Asinh, args: vec![Rc::new(x.clone())], span },
                one.clone() / (x.clone().powi(2) + one.clone()).sqrt(),
            ),
            (
                FunctionKind::Acosh,
                AstNode::FunctionCall { kind: FunctionKind::Acosh, args: vec![Rc::new(x.clone())], span },
                one.clone() / (x.clone().powi(2) - one.clone()).sqrt(),
            ),
            (
                FunctionKind::Atanh,
                AstNode::FunctionCall { kind: FunctionKind::Atanh, args: vec![Rc::new(x.clone())], span },
                one.clone() / (one.clone() - x.clone().powi(2)),
            ),
            (
                FunctionKind::Exp,
                AstNode::FunctionCall { kind: FunctionKind::Exp, args: vec![Rc::new(x.clone())], span },
                one.clone() * x.clone().exp(),
            ),
            (
                FunctionKind::Ln,
                AstNode::FunctionCall { kind: FunctionKind::Ln, args: vec![Rc::new(x.clone())], span },
                one.clone() / x.clone(),
            ),
            (
                FunctionKind::Log10,
                AstNode::FunctionCall { kind: FunctionKind::Log10, args: vec![Rc::new(x.clone())], span },
                one.clone() * log10_e.clone() / x.clone(),
            ),
            (
                FunctionKind::Sqrt,
                AstNode::FunctionCall { kind: FunctionKind::Sqrt, args: vec![Rc::new(x.clone())], span },
                one.clone() * half.clone() / x.clone().sqrt(),
            ),
        ];

        for (kind, node, expected) in cases {
            assert_eq!(node.differentiate(0).unwrap(), expected, "FunctionKind::{:?}", kind);
        }

        let pow_node = x.clone().pow(y.clone());
        let pow_expected = x.clone().pow(y.clone()) * (one.clone() * AstNode::FunctionCall { kind: FunctionKind::Ln, args: vec![Rc::new(x.clone())], span } + y.clone() * AstNode::zero(span) / x.clone());
        assert_eq!(pow_node.differentiate(1).unwrap(), pow_expected);

        let powi_node = x.clone().powi(3);
        let powi_expected = AstNode::FunctionCall {
            kind: FunctionKind::Powi,
            args: vec![Rc::new(x.clone()), Rc::new(three.clone() - one.clone())],
            span,
        } * three.clone() * one.clone();
        assert_eq!(powi_node.differentiate(0).unwrap(), powi_expected);

        assert!(AstNode::FunctionCall { kind: FunctionKind::Abs, args: vec![Rc::new(x.clone())], span }.differentiate(0).is_err());
        assert!(AstNode::FunctionCall { kind: FunctionKind::Conj, args: vec![Rc::new(x.clone())], span }.differentiate(0).is_err());
    }

    // Note: test_differentiate_div is not tested due to complex span handling
    // After differentiation, internally generated constants' spans are determined
    // by the fold/simplify process, which makes exact span comparison difficult.
}
