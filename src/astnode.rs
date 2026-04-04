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
use crate::lexer::Lexeme;
use crate::operators::{BinaryOperatorKind, UnaryOperatorKind};
use crate::token::{Token, UserFnTable};

// ─── helpers ────────────────────────────────────────────────────────────────

fn is_i32_compatible<T: Real>(z: &Complex<T>) -> bool {
    z.im.is_zero() && z.re.clone().fract().is_zero()
        && (T::from_f64(i32::MIN as f64)..=T::from_f64(i32::MAX as f64)).contains(&z.re)
}

// ─── AstNode ────────────────────────────────────────────────────────────────

/// Abstract Syntax Tree node representing a mathematical expression.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum AstNode<T: Real> {
    /// Numeric literal.
    Number(Complex<T>),

    /// Function argument by index.
    Argument(usize),

    /// Unary operator applied to an expression.
    UnaryOperator {
        kind: UnaryOperatorKind,
        expr: Rc<AstNode<T>>,
    },

    /// Binary operator applied to left and right expressions.
    BinaryOperator {
        kind: BinaryOperatorKind,
        left: Rc<AstNode<T>>,
        right: Rc<AstNode<T>>,
    },

    /// Derivative node: `diff(expr, var, order)`.
    Derivative {
        expr: Rc<AstNode<T>>,
        var: usize,
        order: usize,
    },

    /// Built-in function call.
    FunctionCall {
        kind: FunctionKind,
        args: Vec<Rc<AstNode<T>>>,
    },

    /// User-defined function call.
    UserFunctionCall {
        func: UserFn<T>,
        args: Vec<Rc<AstNode<T>>>,
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
                Token::Number(val) => {
                    output.push(Self::Number(val));
                    prev_is_value = true;
                }
                Token::Argument(pos) => {
                    output.push(Self::Argument(pos));
                    prev_is_value = true;
                }
                Token::Operator(lex) => {
                    if prev_is_value {
                        Self::push_binary_op(&mut output, &mut ops, lex)?;
                    } else {
                        Self::push_unary_op(&mut ops, lex)?;
                    }
                    prev_is_value = false;
                }
                // Functions and diff are pushed onto the op stack;
                // they are resolved when their closing ')' is encountered.
                Token::DiffOperator(_) | Token::Function(_) | Token::UserFunction(_) => {
                    ops.push(token);
                    prev_is_value = false;
                }
                // '(' resets prev_is_value so the next token
                // (e.g. `-` in `(-x)`) is treated as unary.
                Token::LParen(_) => {
                    ops.push(token);
                    prev_is_value = false;
                }
                Token::RParen(_) => {
                    Self::flush_until_lparen(&mut output, &mut ops, lexeme)?;
                    prev_is_value = true;
                }
                Token::Comma(_) => {
                    Self::flush_until_lparen_keep(&mut output, &mut ops, lexeme)?;
                    prev_is_value = false;
                }
                _ => return Err(ParseError::InternalError {
                    reason: format!("unexpected token from '{}'", lexeme.text()),
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

    // ── shunting-yard helpers ────────────────────────────────────────────────

    /// Pushes a unary operator onto the op stack.
    fn push_unary_op(ops: &mut Vec<Token<T>>, lex: Lexeme) -> Result<(), ParseError> {
        let kind = UnaryOperatorKind::try_from(lex.clone())
            .map_err(|_| ParseError::InvalidFormula {
                reason: format!("unknown unary operator '{}'", lex.text()),
            })?;
        ops.push(Token::UnaryOperator(kind));
        Ok(())
    }

    /// Pops higher-precedence binary operators from the stack, then pushes the new one.
    fn push_binary_op(
        output: &mut Vec<Self>,
        ops:    &mut Vec<Token<T>>,
        lex:    Lexeme,
    ) -> Result<(), ParseError> {
        let oper = BinaryOperatorKind::try_from(lex.clone())
            .map_err(|_| ParseError::InvalidFormula {
                reason: format!("unknown binary operator '{}'", lex.text()),
            })?;

        // Shunting-yard precedence rule.
        while let Some(Token::BinaryOperator(top)) = ops.last() {
            let should_pop = if oper.is_left_assoc() {
                top.precedence() >= oper.precedence()
            } else {
                top.precedence() > oper.precedence()
            };
            if !should_pop { break; }
            Self::apply_binary(output, *top)?;
            ops.pop();
        }
        ops.push(Token::BinaryOperator(oper));
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
                Some(Token::LParen(_)) => break,
                Some(t) => Self::apply_token(output, t)?,
                None => return Err(ParseError::InvalidFormula {
                    reason: format!("mismatched ')' at {}..{}", lex.start(), lex.end()),
                }),
            }
        }
        // Check for a function/diff call sitting just below the '('.
        if let Some(top) = ops.pop() {
            match top {
                Token::Function(f)     => Self::apply_fn(output, f.arity(), |args| Self::FunctionCall { kind: f, args })?,
                Token::UserFunction(f) => Self::apply_fn(output, f.arity(), |args| Self::UserFunctionCall { func: f, args })?,
                Token::DiffOperator(l) => Self::apply_diff(output, l)?,
                other                  => ops.push(other), // not a call; put it back
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
                Some(Token::LParen(_)) => return Ok(()),
                Some(_) => {
                    let t = ops.pop().unwrap();
                    Self::apply_token(output, t)?;
                }
                None => return Err(ParseError::InvalidFormula {
                    reason: format!("mismatched ',' at {}..{}", lex.start(), lex.end()),
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
                Token::LParen(_) | Token::RParen(_) => {
                    return Err(ParseError::InvalidFormula {
                        reason: "mismatched parentheses".into(),
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
        match token {
            Token::UnaryOperator(op)  => Self::apply_unary(output, op),
            Token::BinaryOperator(op) => Self::apply_binary(output, op),
            Token::Function(f)        => Self::apply_fn(output, f.arity(), |args| Self::FunctionCall { kind: f, args }),
            Token::UserFunction(f)    => Self::apply_fn(output, f.arity(), |args| Self::UserFunctionCall { func: f, args }),
            Token::DiffOperator(lex)  => Self::apply_diff(output, lex),
            other => Err(ParseError::InvalidFormula {
                reason: format!("unexpected token in operator stack: {:?}", other),
            }),
        }
    }

    fn apply_unary(output: &mut Vec<Self>, op: UnaryOperatorKind) -> Result<(), ParseError> {
        let expr = output.pop().ok_or(ParseError::InternalError {
            reason: format!("missing operand for unary '{}'", op),
        })?;
        output.push(Self::UnaryOperator { kind: op, expr: Rc::new(expr) });
        Ok(())
    }

    fn apply_binary(output: &mut Vec<Self>, op: BinaryOperatorKind) -> Result<(), ParseError> {
        let right = output.pop().ok_or(ParseError::MissingRightOperator { operator: op.to_string() })?;
        let left  = output.pop().ok_or(ParseError::MissingLeftOperator  { operator: op.to_string() })?;
        output.push(Self::BinaryOperator { kind: op, left: Rc::new(left), right: Rc::new(right) });
        Ok(())
    }

    /// Generic function-application helper shared by built-in and user-defined functions.
    ///
    /// Pops `arity` nodes from `output` in order and calls `make_node` to build the AST node.
    fn apply_fn<F>(
        output:    &mut Vec<Self>,
        arity:     usize,
        make_node: F,
    ) -> Result<(), ParseError>
    where
        F: FnOnce(Vec<Rc<Self>>) -> Self,
    {
        if output.len() < arity {
            return Err(ParseError::MissingArgs { func: format!("<arity {}>", arity) });
        }
        let start = output.len() - arity;
        let args  = output.drain(start..).map(Rc::new).collect();
        output.push(make_node(args));
        Ok(())
    }

    fn parse_diff_args(output: &mut Vec<Self>, lexeme: &Lexeme) -> Result<(AstNode<T>, usize, usize), ParseError>
    {
        let top = output.pop().ok_or(ParseError::InvalidDerivative {
            lexeme: lexeme.clone(),
            reason: "missing argument (expected variable or order)".into(),
        })?;

        let (var_idx, order) = match top {
            // diff(f, x, n) — explicit order
            Self::Number(z) => {
                if !z.im.is_zero() || !z.re.clone().fract().is_zero() {
                    return Err(ParseError::InvalidDerivativeOrder { target: lexeme.clone(), order: format!("{:?}", z) });
                }
                let order = z.re.clone().to_i32();
                if order > i8::MAX as i32 {
                    return Err(ParseError::InvalidDerivativeOrder { target: lexeme.clone(), order: format!("{:?}", z) });
                }
                let var = match output.pop() {
                    Some(Self::Argument(idx)) => idx,
                    Some(other) => return Err(ParseError::InvalidDerivative {
                        lexeme: lexeme.clone(),
                        reason: format!("expected Argument before order, got {:?}", other),
                    }),
                    None => return Err(ParseError::InvalidDerivative {
                        lexeme: lexeme.clone(),
                        reason: "missing variable before order".into(),
                    }),
                };
                (var, order)
            }
            // diff(f, x) — default order 1
            Self::Argument(idx) => (idx, 1),

            other => return Err(ParseError::InvalidDerivative {
                lexeme: lexeme.clone(),
                reason: format!("expected Argument or Number, got {:?}", other),
            }),
        };

        let expr = output.pop().ok_or(ParseError::InvalidDerivative {
            lexeme: lexeme.clone(),
            reason: "missing expression to differentiate".into(),
        })?;

        Ok((expr, var_idx, order as usize))
    }

    fn apply_diff(output: &mut Vec<Self>, lexeme: Lexeme) -> Result<(), ParseError> {
        let (mut expr, var, order) = Self::parse_diff_args(output, &lexeme)?;

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
            Self::UnaryOperator { kind, expr } => {
                let expr = Rc::try_unwrap(expr)
                    .unwrap_or_else(|rc| (*rc).clone())
                    .simplify();
                match expr {
                    Self::Number(v) => Self::Number(kind.apply(v)),
                    other => Self::UnaryOperator { kind, expr: Rc::new(other) },
                }
            }
            Self::BinaryOperator { kind, left, right } => {
                Self::fold_binary(kind, (*left).clone(), (*right).clone())
            }
            // pow/powi get their own folding path.
            Self::FunctionCall { kind: FunctionKind::Pow  | FunctionKind::Powi, mut args } => {
                let base = Rc::try_unwrap(args.remove(0))
                    .unwrap_or_else(|rc| (*rc).clone());
                let exp  = Rc::try_unwrap(args.remove(0))
                    .unwrap_or_else(|rc| (*rc).clone());
                Self::fold_pow(base, exp)
            }
            Self::FunctionCall { kind, args } => {
                Self::fold_generic_fn(kind, args, |k, a| Self::FunctionCall { kind: k, args: a })
            }
            Self::UserFunctionCall { func, args } => {
                Self::fold_generic_fn(func, args, |f, a| Self::UserFunctionCall { func: f, args: a })
            }
            other => other,
        }
    }

    // ── fold helpers ─────────────────────────────────────────────────────────

    /// Evaluates a function call if all arguments are numeric literals.
    fn fold_generic_fn<F, G>(func: F, args: Vec<Rc<Self>>, make_node: G) -> Self
    where
        F: FunctionCall<T>,
        G: FnOnce(F, Vec<Rc<Self>>) -> Self,
        Complex<T>: AddAssign + MulAssign,
    {
        let args: Vec<_> = args.into_iter().map(|arg| {
            let ast = Rc::try_unwrap(arg)
                .unwrap_or_else(|rc| (*rc).clone());
            Rc::new(Self::simplify(ast))
        }).collect();
        let all_numbers  = args.iter().all(|a| matches!(**a, Self::Number(_)));
        if all_numbers {
            let nums: Vec<Complex<T>> = args.iter()
                .map(|a|
                    match a.as_ref() {
                        Self::Number(v) => v.clone(),
                        _ => unreachable!()
                    }
                )
                .collect();
            Self::Number(func.apply(FunctionArgs::from(nums)))
        } else {
            make_node(func, args)
        }
    }

    fn fold_binary(kind: BinaryOperatorKind, left: Self, right: Self) -> Self
    where
        Complex<T>: AddAssign + MulAssign,
    {
        let left  = left.simplify();
        let right = right.simplify();

        // Both sides are numbers → evaluate immediately.
        if let (Self::Number(l), Self::Number(r)) = (&left, &right) {
            return Self::Number(kind.apply(l.clone(), r.clone()));
        }

        match kind {
            BinaryOperatorKind::Add => Self::fold_add(left, right),
            BinaryOperatorKind::Sub => Self::fold_sub(left, right),
            BinaryOperatorKind::Mul => Self::fold_mul(left, right),
            BinaryOperatorKind::Div => Self::fold_div(left, right),
            BinaryOperatorKind::Pow => Self::fold_pow(left, right),
        }
    }

    // ── addition ─────────────────────────────────────────────────────────────

    /// `left + right` with flattening, constant-folding, and like-term combining.
    fn fold_add(left: Self, right: Self) -> Self
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
                Self::Number(z) => const_sum += z,
                other           => sym_terms.push(other),
            }
        }
        if !const_sum.is_zero() {
            sym_terms.push(Self::Number(const_sum));
        }

        // 3. Combine like terms (e.g. x + x → 2*x).
        let terms = Self::combine_like_add_terms(sym_terms);

        // 4. Fold back into a left-associative chain.
        Self::chain_add(terms)
    }

    fn collect_add_terms(node: Self, out: &mut Vec<Self>) {
        match node {
            Self::BinaryOperator { kind: BinaryOperatorKind::Add, left, right } => {
                Self::collect_add_terms(
                    Rc::try_unwrap(left).unwrap_or_else(|rc| (*rc).clone()),
                    out,
                );
                Self::collect_add_terms(
                    Rc::try_unwrap(right).unwrap_or_else(|rc| (*rc).clone()),
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
        // Map: (variable node) → accumulated coefficient
        let mut map: Vec<(Self, Complex<T>)> = Vec::new();

        for term in terms {
            let (var, coeff) = match term {
                Self::BinaryOperator { kind: BinaryOperatorKind::Mul, left, right } => {
                    let left = Rc::try_unwrap(left).unwrap_or_else(|rc| (*rc).clone());
                    let right = Rc::try_unwrap(right).unwrap_or_else(|rc| (*rc).clone());
                    match (left, right) {
                        (Self::Number(z), v) => (v, z),
                        (v, Self::Number(z)) => (v, z),
                        (l, r)               => (l.mul(r), Complex::one()),
                    }
                }
                other => (other, Complex::one()),
            };
            match map.iter_mut().find(|(v, _)| *v == var) {
                Some((_, c)) => *c += coeff,
                None         => map.push((var, coeff)),
            }
        }

        map.into_iter()
            .filter(|(_, c)| !(*c).is_zero())
            .map(|(var, coeff)| {
                if coeff.is_one() { var }
                else { Self::Number(coeff).mul(var) }
            })
            .collect()
    }

    fn chain_add(terms: Vec<Self>) -> Self {
        match terms.len() {
            0 => Self::zero(),
            1 => terms.into_iter().next().unwrap(),
            _ => terms.into_iter().reduce(|acc, t| acc.add(t)).unwrap(),
        }
    }

    // ── subtraction ──────────────────────────────────────────────────────────

    /// Rewrites `left - right` as `left + (-1)*right` and re-folds.
    fn fold_sub(left: Self, right: Self) -> Self
    where
        Complex<T>: AddAssign + MulAssign,
    {
        Self::fold_binary(
            BinaryOperatorKind::Add,
            left,
            Self::Number(-Complex::one()).mul(right),
        )
    }

    // ── multiplication ───────────────────────────────────────────────────────

    /// `left * right` with flattening, constant-folding, and same-base power combining.
    fn fold_mul(left: Self, right: Self) -> Self
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
                Self::Number(z) => const_prod *= z,
                other           => sym_factors.push(other),
            }
        }

        if const_prod.is_zero() {
            return Self::zero();
        }
        if !const_prod.is_one() {
            sym_factors.insert(0, Self::Number(const_prod));
        }

        // 3. Combine x^a * x^b → x^(a+b).
        let factors = Self::combine_like_pow_terms(sym_factors);

        // 4. Fold back.
        Self::chain_mul(factors)
    }

    fn collect_mul_terms(node: Self, out: &mut Vec<Self>) {
        match node {
            Self::BinaryOperator { kind: BinaryOperatorKind::Mul, left, right } => {
                let left = Rc::try_unwrap(left).unwrap_or_else(|rc| (*rc).clone());
                let right = Rc::try_unwrap(right).unwrap_or_else(|rc| (*rc).clone());
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
                Self::BinaryOperator { kind: BinaryOperatorKind::Pow, left, right } => {
                    let left = Rc::try_unwrap(left).unwrap_or_else(|rc| (*rc).clone());
                    let right = Rc::try_unwrap(right).unwrap_or_else(|rc| (*rc).clone());
                    match right {
                        Self::Number(e) => (left, e),
                        r               => (left.pow(r), Complex::one()),
                    }
                }
                Self::FunctionCall { kind: FunctionKind::Pow | FunctionKind::Powi, ref args } => {
                    let base = Rc::try_unwrap(args[0].clone()).unwrap_or_else(|rc| (*rc).clone());
                    match args[1].as_ref() {
                        Self::Number(e) => (base, e.clone()),
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
                else { base.pow(Self::Number(exp)) }
            })
            .collect()
    }

    fn chain_mul(factors: Vec<Self>) -> Self
    where
        Complex<T>: AddAssign + MulAssign,
    {
        match factors.len() {
            0 => Self::one(),
            1 => factors.into_iter().next().unwrap().simplify(),
            _ => factors.into_iter().reduce(|acc, f| acc.mul(f)).unwrap(),
        }
    }

    // ── division ─────────────────────────────────────────────────────────────

    /// Rewrites `left / right` as `left * right^-1` and re-folds.
    fn fold_div(left: Self, right: Self) -> Self
    where
        Complex<T>: AddAssign + MulAssign,
    {
        Self::fold_mul(left, right.powi(-1).simplify())
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
                Self::FunctionCall { kind: FunctionKind::Pow | FunctionKind::Powi, mut args } => {
                    let inner_base = Rc::try_unwrap(args.remove(0))
                        .unwrap_or_else(|rc| (*rc).clone());
                    let inner_exp  = Rc::try_unwrap(args.remove(0))
                        .unwrap_or_else(|rc| (*rc).clone());
                    exp  = inner_exp.mul(exp).simplify();
                    base = inner_base.simplify();
                }
                other => { base = other; break; }
            }
        }

        // x^1 → x, x^0 → 1
        if let Self::Number(e) = &exp {
            if (*e).is_one()  { return base; }
            if (*e).is_zero() { return Self::one(); }
        }

        match (base, exp) {
            (Self::Number(b), _) if b.is_one() => Self::one(),
            (Self::Number(b), Self::Number(e)) if b.is_zero() && e.re > T::zero() => Self::zero(),
            (Self::Number(b), Self::Number(e)) => Self::Number(b.powc(e)),
            (b, Self::Number(e)) if is_i32_compatible(&e) => b.powi(e.re.to_i32()),
            (b, e) => b.pow(e),
        }
    }
}

// ─── AstNode builder helpers ─────────────────────────────────────────────────

impl<T: Real> AstNode<T> {
    fn zero() -> Self { Self::Number(Complex::zero()) }
    fn one()  -> Self { Self::Number(Complex::one()) }

    fn add(self, rhs: Self) -> Self { Self::BinaryOperator { kind: BinaryOperatorKind::Add, left: Rc::new(self), right: Rc::new(rhs) } }
    fn sub(self, rhs: Self) -> Self { Self::BinaryOperator { kind: BinaryOperatorKind::Sub, left: Rc::new(self), right: Rc::new(rhs) } }
    fn mul(self, rhs: Self) -> Self { Self::BinaryOperator { kind: BinaryOperatorKind::Mul, left: Rc::new(self), right: Rc::new(rhs) } }
    fn div(self, rhs: Self) -> Self { Self::BinaryOperator { kind: BinaryOperatorKind::Div, left: Rc::new(self), right: Rc::new(rhs) } }

    fn negative(self) -> Self { Self::UnaryOperator { kind: UnaryOperatorKind::Negative, expr: Rc::new(self) } }

    fn sin(self)  -> Self { Self::FunctionCall { kind: FunctionKind::Sin,  args: vec![Rc::new(self)] } }
    fn cos(self)  -> Self { Self::FunctionCall { kind: FunctionKind::Cos,  args: vec![Rc::new(self)] } }
    fn sinh(self) -> Self { Self::FunctionCall { kind: FunctionKind::Sinh, args: vec![Rc::new(self)] } }
    fn cosh(self) -> Self { Self::FunctionCall { kind: FunctionKind::Cosh, args: vec![Rc::new(self)] } }
    fn exp(self)  -> Self { Self::FunctionCall { kind: FunctionKind::Exp,  args: vec![Rc::new(self)] } }
    fn sqrt(self) -> Self { Self::FunctionCall { kind: FunctionKind::Sqrt, args: vec![Rc::new(self)] } }

    fn pow(self, exp: Self) -> Self {
        Self::FunctionCall { kind: FunctionKind::Pow, args: vec![Rc::new(self), Rc::new(exp)] }
    }
    fn powi(self, n: i32) -> Self {
        Self::FunctionCall { kind: FunctionKind::Powi, args: vec![Rc::new(self), Rc::new(Self::Number(Complex::from(T::from_f64(n as f64))))] }
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
            Self::Number(_)    => Ok(Self::zero()),
            Self::Argument(i)  => Ok(if i == var { Self::one() } else { Self::zero() }),

            Self::UnaryOperator { kind, expr } => {
                let expr = Rc::try_unwrap(expr).unwrap_or_else(|rc| (*rc).clone());
                Ok(Self::UnaryOperator {
                    kind,
                    expr: Rc::new(expr.differentiate(var)?),
                })
            },

            Self::BinaryOperator { kind, left, right } => {
                let left = Rc::try_unwrap(left).unwrap_or_else(|rc| (*rc).clone());
                let right = Rc::try_unwrap(right).unwrap_or_else(|rc| (*rc).clone());
                Self::diff_binary(kind, left, right, var)
            }

            Self::FunctionCall { kind, args } => {
                Self::diff_function(kind, args, var)
            }

            Self::UserFunctionCall { func, args } => {
                if var >= func.arity() {
                    return Err(ParseError::OutOfRange { func: func.name().into(), idx: var });
                }
                if let Some(deriv) = func.derivative(var).cloned() {
                    Ok(Self::UserFunctionCall { func: deriv, args })
                } else {
                    Err(ParseError::DerivativeUndefined { func: func.name().into(), idx: var })
                }
            }

            Self::Derivative { expr, var: inner_var, order } => {
                if inner_var == var {
                    Ok(Self::Derivative { expr, var, order: order + 1 })
                } else {
                    let expr = Rc::try_unwrap(expr).unwrap_or_else(|rc| (*rc).clone());
                    Ok(Self::Derivative {
                        expr:  Rc::new(expr.differentiate(var)?),
                        var:   inner_var,
                        order,
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
    ) -> Result<Self, ParseError> {
        let dl = left.clone().differentiate(var)?;
        let dr = right.clone().differentiate(var)?;
        match kind {
            BinaryOperatorKind::Add | BinaryOperatorKind::Sub => {
                Ok(Self::BinaryOperator { kind, left: Rc::new(dl), right: Rc::new(dr) })
            }
            BinaryOperatorKind::Mul => Ok(dl.mul(right).add(left.mul(dr))),
            BinaryOperatorKind::Div => {
                // (u/v)' = (u'v - uv') / v²
                Ok(dl.mul(right.clone()).sub(left.mul(dr)).div(right.powi(2)))
            }
            BinaryOperatorKind::Pow => Self::diff_pow(left, right, var),
        }
    }

    fn diff_function(
        kind: FunctionKind,
        mut args: Vec<Rc<Self>>,
        var:  usize,
    ) -> Result<Self, ParseError> {
        let x = Rc::try_unwrap(args.remove(0))
            .unwrap_or_else(|rc| (*rc).clone());
        let dx = x.clone().differentiate(var)?;
        match kind {
            FunctionKind::Sin   => Ok(x.cos().mul(dx)),
            FunctionKind::Cos   => Ok(x.sin().negative().mul(dx)),
            FunctionKind::Tan   => Ok(dx.div(x.cos().powi(2))),
            FunctionKind::Asin  => Ok(dx.div(Self::one().sub(x.powi(2)))),
            FunctionKind::Acos  => Ok(dx.negative().div(Self::one().sub(x.powi(2)))),
            FunctionKind::Atan  => Ok(dx.div(Self::one().add(x.powi(2)))),
            FunctionKind::Sinh  => Ok(dx.mul(x.cosh())),
            FunctionKind::Cosh  => Ok(dx.mul(x.sinh())),
            FunctionKind::Tanh  => Ok(dx.div(x.cosh().powi(2))),
            FunctionKind::Asinh => Ok(dx.div(x.powi(2).add(Self::one()).sqrt())),
            FunctionKind::Acosh => Ok(dx.div(x.powi(2).sub(Self::one()).sqrt())),
            FunctionKind::Atanh => Ok(dx.div(Self::one().sub(x.powi(2)))),
            FunctionKind::Exp   => Ok(dx.mul(x.exp())),
            FunctionKind::Ln    => Ok(dx.div(x)),
            FunctionKind::Log10 => Ok(dx.mul(Self::Number(Complex::from(T::log10_e()))).div(x)),
            FunctionKind::Sqrt  => Ok(dx.mul(Self::Number(Complex::from(T::from_f64(0.5)))).div(x.sqrt())),
            FunctionKind::Abs   => Ok(x.clone().div(x.abs()).mul(dx)),
            FunctionKind::Conj  => Err(ParseError::InvalidFormula {
                reason: "`conj(z)` is not differentiable in the complex domain".into(),
            }),
            FunctionKind::Pow  => {
                let y = Rc::try_unwrap(args.remove(0))
                    .unwrap_or_else(|rc| (*rc).clone());
                Self::diff_pow(x, y, var)
            },
            FunctionKind::Powi => {
                let n = Rc::try_unwrap(args.remove(0))
                    .unwrap_or_else(|rc| (*rc).clone());
                Self::diff_powi(x, n, var)
            },
        }
    }

    /// d/dx [u^v] = u^v * (v'*ln(u) + v*u'/u)
    fn diff_pow(u: Self, v: Self, var: usize) -> Result<Self, ParseError> {
        let du   = u.clone().differentiate(var)?;
        let dv   = v.clone().differentiate(var)?;
        let ln_u = Self::FunctionCall { kind: FunctionKind::Ln, args: vec![Rc::new(u.clone())] };
        Ok(u.clone().pow(v.clone()) * (dv * ln_u + v * du / u))
    }

    /// d/dx [u^n] = n * u^(n-1) * u'
    fn diff_powi(u: Self, n: Self, var: usize) -> Result<Self, ParseError> {
        let du = u.clone().differentiate(var)?;
        Ok(Self::FunctionCall {
            kind: FunctionKind::Powi,
            args: vec![Rc::new(u), Rc::new(n.clone() - Self::one())],
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
            Self::Number(v)    => out.push(Token::Number(v.clone())),
            Self::Argument(i)  => out.push(Token::Argument(*i)),
            Self::UnaryOperator { kind, expr } => {
                expr.compile_into(out);
                out.push(Token::UnaryOperator(*kind));
            }
            Self::BinaryOperator { kind, left, right } => {
                left.compile_into(out);
                right.compile_into(out);
                out.push(Token::BinaryOperator(*kind));
            }
            Self::FunctionCall { kind, args } => {
                for arg in args { arg.compile_into(out); }
                out.push(Token::Function(*kind));
            }
            Self::UserFunctionCall { func, args } => {
                for arg in args { arg.compile_into(out); }
                out.push(Token::UserFunction(func.clone()));
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
                    (AstNode::Number(l), AstNode::Number(r)) => {
                        assert!((l.re.clone() - r.re.clone()).abs() < T::from_f64(epsilon));
                        assert!((l.im.clone() - r.im.clone()).abs() < T::from_f64(epsilon));
                    }
                    (AstNode::Argument(l), AstNode::Argument(r)) => {
                        assert_eq!(l, r);
                    }
                    (AstNode::UnaryOperator { kind: lk, expr: le }, AstNode::UnaryOperator { kind: rk, expr: re }) => {
                        assert_eq!(lk, rk);
                        inner(le, re);
                    }
                    (AstNode::BinaryOperator { kind: lk, left: ll, right: lr },
                    AstNode::BinaryOperator { kind: rk, left: rl, right: rr }) => {
                        assert_eq!(lk, rk);
                        inner(ll, rl);
                        inner(lr, rr);
                    }
                    (AstNode::FunctionCall { kind: lk, args: la },
                    AstNode::FunctionCall { kind: rk, args: ra }) => {
                        assert_eq!(lk, rk);
                        assert_eq!(la.len(), ra.len());
                        for (a, b) in la.iter().zip(ra.iter()) {
                            inner(a, b);
                        }
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
        match ast {
            AstNode::Number(val) => assert_eq!(val, Complex::new(42.0, 0.0)),
            _ => panic!("Expected Number AST node"),
        }
    }

    #[test]
    fn test_unary_operator_negative_astnode() {
        let lexemes = lexer::from("- 3");
        let ast = AstNode::from(&lexemes, &[], &Constants::new(), &UserFnTable::new()).unwrap();
        match ast {
            AstNode::UnaryOperator { kind, expr } => {
                assert_eq!(kind, UnaryOperatorKind::Negative);
                match *expr {
                    AstNode::Number(val) => assert_eq!(val, Complex::new(3.0, 0.0)),
                    _ => panic!("Expected Number child"),
                }
            }
            _ => panic!("Expected UnaryOp AST node"),
        }
    }

    #[test]
    fn test_binary_operator_precedence_astnode() {
        let lexemes = lexer::from("2 + 3 * 4");
        let ast = AstNode::from(&lexemes, &[], &Constants::new(), &UserFnTable::new()).unwrap();
        // expected: (2 + (3 * 4))
        match ast {
            AstNode::BinaryOperator { kind, left, right } => {
                assert_eq!(kind, BinaryOperatorKind::Add);
                let right = Rc::try_unwrap(right).unwrap();
                match right {
                    AstNode::BinaryOperator { kind, left, right } => {
                        assert_eq!(kind, BinaryOperatorKind::Mul);
                        assert_eq!(*left, AstNode::Number(Complex::new(3.0, 0.0)));
                        assert_eq!(*right, AstNode::Number(Complex::new(4.0, 0.0)));
                    }
                    _ => panic!("Expected Mul node"),
                }
                assert_eq!(*left, AstNode::Number(Complex::new(2.0, 0.0)));
            }
            _ => panic!("Expected Add node"),
        }
    }

    #[test]
    fn test_parentheses_override_precedence_astnode() {
        let lexemes = lexer::from("( 2 + 3 ) * 4");
        let ast = AstNode::from(&lexemes, &[], &Constants::new(), &UserFnTable::new()).unwrap();
        // expected: ((2 + 3) * 4)
        match ast {
            AstNode::BinaryOperator { kind, left, right } => {
                assert_eq!(kind, BinaryOperatorKind::Mul);
                assert_eq!(*right, AstNode::Number(Complex::new(4.0, 0.0)));
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
        match ast {
            AstNode::FunctionCall { kind, args } => {
                assert_eq!(kind, FunctionKind::Sin);
                assert_eq!(args.len(), 1);
                assert_eq!(Rc::try_unwrap(args[0].clone()).unwrap_or_else(|rc| (*rc).clone()), AstNode::Number(Complex::new(0.0, 0.0)));
            }
            _ => panic!("Expected Function node"),
        }
    }

    #[test]
    fn test_function_multiple_args_astnode() {
        let lexemes = lexer::from("pow ( 2 , 3 )");
        let ast = AstNode::from(&lexemes, &[], &Constants::new(), &UserFnTable::new()).unwrap();
        match ast {
            AstNode::FunctionCall { kind, args } => {
                assert_eq!(kind, FunctionKind::Pow);
                assert_eq!(args.len(), 2);
                assert_eq!(Rc::try_unwrap(args[0].clone()).unwrap_or_else(|rc| (*rc).clone()), AstNode::Number(Complex::new(2.0, 0.0)));
                assert_eq!(Rc::try_unwrap(args[1].clone()).unwrap_or_else(|rc| (*rc).clone()), AstNode::Number(Complex::new(3.0, 0.0)));
            }
            _ => panic!("Expected Function node"),
        }

        let lexemes = lexer::from("pow ( sin(x) , 3 )");
        let ast = AstNode::from(&lexemes, &["x"], &Constants::new(), &UserFnTable::new()).unwrap();
        match ast {
            AstNode::FunctionCall { kind, args } => {
                assert_eq!(kind, FunctionKind::Pow);
                assert_eq!(args.len(), 2);
                let AstNode::FunctionCall { kind: k, args: a } = Rc::try_unwrap(args[0].clone()).unwrap_or_else(|rc| (*rc).clone()) else { unreachable!() };
                assert_eq!(k, FunctionKind::Sin);
                assert_eq!(a.len(), 1);
                assert_eq!(Rc::try_unwrap(a[0].clone()).unwrap_or_else(|rc| (*rc).clone()), AstNode::Argument(0));
                assert_eq!(Rc::try_unwrap(args[1].clone()).unwrap_or_else(|rc| (*rc).clone()), AstNode::Number(Complex::new(3.0, 0.0)));
            }
            _ => panic!("Expected Function node"),
        }
    }

    #[test]
    fn test_imaginary_number_astnode() {
        let lexemes = lexer::from("5i");
        let ast = AstNode::from(&lexemes, &[], &Constants::new(), &UserFnTable::new()).unwrap();
        assert_eq!(ast, AstNode::Number(Complex::new(0.0, 5.0)));
    }

    #[test]
    fn test_unknown_token_astnode_error() {
        let lexemes = lexer::from("@");
        let res = AstNode::from(&lexemes, &[], &Constants::<f64>::new(), &UserFnTable::new());
        assert!(res.is_err());
    }

    #[test]
    fn test_fold_add_constants() {
        let left = AstNode::Number(Complex::from(2.0));
        let right = AstNode::Number(Complex::from(3.0));
        let result = AstNode::fold_add(left, right);
        assert_astnode_eq!(result, AstNode::Number(Complex::from(5.0)));
    }

    #[test]
    fn test_fold_add_like_terms() {
        let x = AstNode::Argument(0);
        let result = AstNode::fold_add(x.clone(), x.clone());
        assert_eq!(result, AstNode::Number(Complex::new(2.0, 0.0)).mul(x));
    }

    #[test]
    fn test_fold_add_mixed_terms() {
        // 3*x + 4*x + y → 7*x + y
        let x = AstNode::Argument(0);
        let y = AstNode::Argument(1);
        let term1 = AstNode::Number(Complex::new(3.0, 0.0)).mul(x.clone());
        let term2 = AstNode::Number(Complex::new(4.0, 0.0)).mul(x.clone());
        let left = AstNode::fold_add(term1, term2); // 3x + 4x → 7x
        let result = AstNode::fold_add(left, y.clone());
        let expected = AstNode::fold_add(AstNode::Number(Complex::new(7.0, 0.0)).mul(x), y);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_fold_add_removed_same_terms() {
        // x + 2 - x → x
        let x = AstNode::Argument(0);
        let n = AstNode::Number(Complex::from(2.0));
        let result = AstNode::fold_sub(
            AstNode::fold_add(x.clone(), n.clone()),
            x
        );
        assert_astnode_eq!(result, n);
    }

    #[test]
    fn test_fold_add_zero_terms() {
        // x + 0 → x
        let x = AstNode::<f64>::Argument(0);
        let zero = AstNode::Number(Complex::<f64>::zero());
        let result = AstNode::fold_add(x.clone(), zero);
        assert_astnode_eq!(result, x);
    }

    #[test]
    fn test_fold_sub_basic() {
        // x - y → x + (-1) * y
        let x = AstNode::<f64>::Argument(0);
        let y = AstNode::<f64>::Argument(1);
        let result = AstNode::fold_sub(x.clone(), y.clone());
        let expected = AstNode::fold_add(x, AstNode::Number(-Complex::ONE).mul(y));
        assert_astnode_eq!(result, expected);
    }

    #[test]
    fn test_fold_sub_with_constants() {
        // 5 - 3 → 2
        let left = AstNode::Number(Complex::new(5.0, 0.0));
        let right = AstNode::Number(Complex::new(3.0, 0.0));
        let result = AstNode::fold_sub(left, right);
        assert_astnode_eq!(result, AstNode::Number(Complex::new(2.0, 0.0)));
    }

    #[test]
    fn test_mul_constant_folding() {
        let expr = AstNode::fold_mul(
            AstNode::Number(Complex::from(2.0)),
            AstNode::Number(Complex::from(3.0)));
        assert_astnode_eq!(expr, AstNode::Number(Complex::from(6.0)));
    }

    #[test]
    fn test_mul_with_zero() {
        let expr = AstNode::fold_mul(
            AstNode::Number(Complex::<f64>::zero()),
            AstNode::Argument(0));
        assert_astnode_eq!(expr, AstNode::Number(Complex::zero()));
    }

    #[test]
    fn test_mul_with_one() {
        let expr = AstNode::fold_mul(
            AstNode::Number(Complex::<f64>::one()),
            AstNode::Argument(0));
        assert_astnode_eq!(expr, AstNode::Argument(0));
    }

    #[test]
    fn test_div_to_mul_pow_neg1() {
        let expr = AstNode::fold_div(
            AstNode::<f64>::Argument(0),
            AstNode::<f64>::Argument(1));
        // should become x * y^-1
        let expected = AstNode::<f64>::fold_mul(
            AstNode::<f64>::Argument(0),
            AstNode::<f64>::Argument(1).powi(-1));
        assert_astnode_eq!(expr, expected);
    }

    #[test]
    fn test_combine_same_base_powers() {
        let expr = AstNode::fold_mul(
            AstNode::Argument(0).pow(AstNode::Number(Complex::from(2.0))),
            AstNode::Argument(0).pow(AstNode::Number(Complex::from(3.5))));
        assert_astnode_eq!(expr, AstNode::Argument(0).pow(AstNode::Number(Complex::from(5.5))));
    }

    #[test]
    fn test_combine_same_base_powers_to_powi() {
        let expr = AstNode::fold_mul(
            AstNode::Argument(0).pow(AstNode::Number(Complex::from(2.0))),
            AstNode::Argument(0).pow(AstNode::Number(Complex::from(3.0))));
        assert_astnode_eq!(expr, AstNode::Argument(0).powi(5));
    }

    #[test]
    fn test_nested_mul_flattening() {
        let expr = AstNode::fold_mul(
            AstNode::fold_mul(
                AstNode::Argument(0),
                AstNode::Number(Complex::from(2.0))),
            AstNode::Number(Complex::from(3.0))
        );
        // (x * 2) * 3 => 6 * x
        let expected = AstNode::fold_mul(
            AstNode::Number(Complex::from(6.0)),
            AstNode::Argument(0));
        assert_eq!(expr, expected);
    }

    #[test]
    fn test_simplify_number() {
        let node = AstNode::Number(Complex::new(3.0, 0.0));
        assert_astnode_eq!(node.clone().simplify(), node);
    }

    #[test]
    fn test_simplify_unary_operator() {
        let node = -AstNode::Number(Complex::new(2.0, 0.0));
        let simplified = node.simplify();
        assert_astnode_eq!(simplified, AstNode::Number(Complex::new(-2.0, 0.0)));
    }

    #[test]
    fn test_simplify_binary_operator_full() {
        let node = AstNode::Number(Complex::new(2.0, 0.0)) + AstNode::Number(Complex::new(3.0, 0.0));
        let simplified = node.simplify();
        assert_astnode_eq!(simplified, AstNode::Number(Complex::new(5.0, 0.0)));
    }

    #[test]
    fn test_simplify_binary_operator_partial() {
        let node = AstNode::Argument(0) + AstNode::Number(Complex::new(3.0, 0.0));
        let simplified = node.clone().simplify();
        assert_astnode_eq!(simplified, node);
    }

    #[test]
    fn test_simplify_binary_operator_chain() {
        // x + 2 + 3 -> x + 5
        let node
            = AstNode::Argument(0)
            + AstNode::Number(Complex::new(2.0, 0.0))
            + AstNode::Number(Complex::new(3.0, 0.0));
        let simplified = node.simplify();
        assert_astnode_eq!(
            simplified,
            AstNode::Argument(0) + AstNode::Number(Complex::new(5.0, 0.0))
        );

        // x * 2 * 3 * 4 -> 24 * x
        let node = AstNode::Argument(0) * AstNode::Number(Complex::new(2.0, 0.0))
            * AstNode::Number(Complex::new(3.0, 0.0)) * AstNode::Number(Complex::new(4.0, 0.0));
        let simplified = node.simplify();
        assert_astnode_eq!(
            simplified,
            AstNode::Number(Complex::new(24.0, 0.0)) * AstNode::Argument(0)
        );

        // 2 * x + 3 -> not changed
        let node = AstNode::Number(Complex::new(2.0, 0.0)) * AstNode::Argument(0) + AstNode::Number(Complex::new(3.0, 0.0));
        let simplified = node.clone().simplify();
        assert_astnode_eq!(simplified, node)
    }

    #[test]
    fn test_simplify_function_call_full() {
        let node = AstNode::Number(Complex::new(2.0, 0.0)).pow(AstNode::Number(Complex::new(3.0, 0.0)));
        let simplified = node.simplify();
        assert_astnode_eq!(simplified, AstNode::Number(Complex::new(8.0, 0.0)));

        let node = AstNode::Number(Complex::new(2.0, 0.0)).exp();
        let simplified = node.clone().simplify();
        assert_astnode_eq!(simplified, AstNode::Number(Complex::from(2.0).exp()));
    }

    #[test]
    fn test_simplify_function_call_partial() {
        let node = AstNode::<f64>::Argument(0).pow(AstNode::Argument(1)).simplify();
        assert_astnode_eq!(
            node,
            AstNode::Argument(0).pow(AstNode::Argument(1))
        );
    }

    #[test]
    fn test_simplify_pow_to_powi() {
        let node = AstNode::Argument(0).pow(AstNode::Number(Complex::from(3.0))).simplify();
        assert_astnode_eq!(node, AstNode::Argument(0).powi(3));
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
                Rc::new(AstNode::Number(Complex::from(1.0))),
                Rc::new(AstNode::Number(Complex::from(2.0))),
            ],
        }.simplify();

        match node {
            AstNode::Number(val) => assert_abs_diff_eq!(val.re, 3.0, epsilon=1e-12),
            _ => panic!("Expected simplified to Number"),
        }
    }

    #[test]
    fn test_simplify_user_function_call_with_no_numbers() {
        let func = UserFn::new("sum", sum_func);

        let node = AstNode::UserFunctionCall {
            func,
            args: vec![
                Rc::new(AstNode::Number(Complex::ONE)),
                Rc::new(AstNode::Argument(0)),
            ],
        };

        let simplified = node.clone().simplify();

        assert_eq!(simplified, node);
    }

    #[test]
    fn test_compile_number() {
        let ast = AstNode::Number(Complex::new(1.0, 0.0));
        let tokens = ast.compile();
        assert_eq!(tokens, vec![Token::Number(Complex::new(1.0, 0.0))]);
    }

    #[test]
    fn test_compile_argument() {
        let ast = AstNode::<f64>::Argument(1);
        let tokens = ast.compile();
        assert_eq!(tokens, vec![Token::Argument(1)]);
    }

    #[test]
    fn test_compile_unary_operator() {
        let ast = -AstNode::Number(Complex::new(1.0, 0.0));
        let tokens = ast.compile();
        assert_eq!(
            tokens,
            vec![Token::Number(Complex::new(1.0, 0.0)), Token::UnaryOperator(UnaryOperatorKind::Negative)]
        );
    }

    #[test]
    fn test_compile_binary_operator() {
        let ast = AstNode::Number(Complex::new(1.0, 0.0)) + AstNode::Argument(1);
        let tokens = ast.compile();
        assert_eq!(
            tokens,
            vec![
                Token::Number(Complex::new(1.0, 0.0)),
                Token::Argument(1),
                Token::BinaryOperator(BinaryOperatorKind::Add),
            ]
        );
    }

    #[test]
    fn test_compile_function_single_argument() {
        let ast = AstNode::Number(Complex::new(1.0, 0.0)).sin();
        let tokens = ast.compile();
        assert_eq!(tokens, vec![Token::Number(Complex::new(1.0, 0.0)), Token::Function(FunctionKind::Sin)]);
    }

    #[test]
    fn test_compile_function_multi_arguments() {
        let ast = AstNode::Number(Complex::new(2.0, 0.0)).pow(AstNode::Argument(0));
        let tokens = ast.compile();
        assert_eq!(
            tokens,
            vec![
                Token::Number(Complex::new(2.0, 0.0)),
                Token::Argument(0),
                Token::Function(FunctionKind::Pow),
            ]
        );
    }

    #[test]
    fn test_compile_nested_expression() {
        // cos(1 + 2)
        let ast = (AstNode::Number(Complex::new(1.0, 0.0)) + AstNode::Number(Complex::new(2.0, 0.0))).cos();
        let tokens = ast.compile();
        assert_eq!(
            tokens,
            vec![
                Token::Number(Complex::new(1.0, 0.0)),
                Token::Number(Complex::new(2.0, 0.0)),
                Token::BinaryOperator(BinaryOperatorKind::Add),
                Token::Function(FunctionKind::Cos),
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
        let node = AstNode::Number(Complex::new(5.0, 0.0));
        let diff = node.differentiate(0).unwrap();
        assert_eq!(diff, AstNode::Number(Complex::ZERO));
    }

    #[test]
    fn test_differentiate_argument() {
        let node = AstNode::<f64>::Argument(1);
        let diff = node.clone().differentiate(1).unwrap();
        assert_eq!(diff, AstNode::Number(Complex::ONE));
        let diff_other = node.differentiate(0).unwrap();
        assert_eq!(diff_other, AstNode::Number(Complex::ZERO));
    }

    #[test]
    fn test_differentiate_unary_operator() {
        let node = -AstNode::<f64>::Argument(0);
        let diff = node.differentiate(0).unwrap();
        assert_eq!(diff, -AstNode::Number(Complex::ONE));
    }

    #[test]
    fn test_differentiate_binary_add() {
        let node = AstNode::Argument(0) + AstNode::Number(Complex::new(2.0, 0.0));
        let diff = node.differentiate(0).unwrap();
        // d/dx (x + 2) = 1 + 0
        assert_eq!(
            diff,
            AstNode::Number(Complex::ONE) + AstNode::Number(Complex::ZERO)
        );
    }

    #[test]
    fn test_differentiate_function_sin() {
        let node = AstNode::<f64>::Argument(0).sin();
        let diff = node.differentiate(0).unwrap();
        // d/dx sin(x) = cos(x) * 1
        assert_eq!(
            diff,
            AstNode::Argument(0).cos().mul(AstNode::Number(Complex::ONE))
        );
    }

    #[test]
    fn test_differentiate_derivative_order() {
        let node = AstNode::Derivative {
            expr: Rc::new(AstNode::<f64>::Argument(0)),
            var: 0,
            order: 1,
        };
        let diff = node.differentiate(0).unwrap();
        // d/dx (d/dx x) = d^2/dx^2 x
        assert_eq!(
            diff,
            AstNode::Derivative {
                expr: Rc::new(AstNode::Argument(0)),
                var: 0,
                order: 2,
            }
        );
    }

    #[test]
    fn test_differentiate_mul_x2() {
        // f(x) = x * x
        let node = AstNode::Argument(0).mul(AstNode::Argument(0)).differentiate(0)
            .unwrap().simplify();
        // d/dx (x * x) = 1 * x + x * 1 = 2x
        let expected = AstNode::Number(Complex::from(2.0)).mul(AstNode::Argument(0));
        assert_eq!(node, expected);
    }

    #[test]
    fn test_differentiate_powi_x3() {
        // f(x) = pow(x, 3)
        let node = AstNode::Argument(0).powi(3);
        let diff = node.differentiate(0).unwrap().simplify();
        // d/dx x^3 = 3 * x^(3-1) * 1 = 3 * x^2
        let expected = AstNode::Number(Complex::from(3.0))
            .mul(AstNode::Argument(0).powi(2));
        assert_eq!(diff, expected);
    }

    #[test]
    fn test_differentiate_div() {
        // f(x) = x / (x + 1)
        let node = AstNode::Argument(0).div(AstNode::Argument(0).add(AstNode::Number(Complex::from(1.0))));
        let diff = node.differentiate(0).unwrap().simplify();

        // d/dx [x / (x + 1)] = (1 * (x + 1) - x * 1) / (x + 1)^2 = (x + 1 - x) / (x + 1)^2 = (x + 1)^(-2)
        let expected = AstNode::Argument(0).add(AstNode::Number(Complex::ONE)).powi(-2);
        assert_eq!(diff, expected);
    }

    #[test]
    fn test_differentiate_chain_rule() {
        // f(x) = sin(x^2)
        let node = AstNode::Argument(0).powi(2).sin();
        let diff = node.differentiate(0).unwrap().simplify();

        // d/dx sin(x^2) = cos(x^2) * d/dx(x^2) = cos(x^2) * 2x
        let expected = AstNode::Number(Complex::from(2.0)).mul(AstNode::Argument(0).powi(2).cos()).mul(AstNode::Argument(0));
        assert_eq!(diff, expected);
    }
}
