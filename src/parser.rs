//! parser.rs
//!

use crate::lexer::Lexeme;
use crate::lexer::IMAGINARY_UNIT;
use crate::variable::Variables;
use num_complex::Complex;
use num_complex::ComplexFloat;
use phf::Map;
use phf_macros::phf_map;

macro_rules! lexeme_name_with_range {
    ($lexeme: expr) => {
        format!("{name} at {start}..{end}", name=$lexeme.text(), start=$lexeme.start(), end=$lexeme.end())
    };
}

/// Map of mathematical constants by their string representation.
static CONSTANTS: Map<&'static str, Complex<f64>> = phf_map! {
    "E" => Complex::new(std::f64::consts::E, 0.0),
    "FRAC_1_PI" => Complex::new(std::f64::consts::FRAC_1_PI, 0.0),
    "FRAC_1_SQRT_2" => Complex::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
    "FRAC_2_PI" => Complex::new(std::f64::consts::FRAC_2_PI, 0.0),
    "FRAC_2_SQRT_PI" => Complex::new(std::f64::consts::FRAC_2_SQRT_PI, 0.0),
    "FRAC_PI_2" => Complex::new(std::f64::consts::FRAC_PI_2, 0.0),
    "FRAC_PI_3" => Complex::new(std::f64::consts::FRAC_PI_3, 0.0),
    "FRAC_PI_4" => Complex::new(std::f64::consts::FRAC_PI_4, 0.0),
    "FRAC_PI_6" => Complex::new(std::f64::consts::FRAC_PI_6, 0.0),
    "FRAC_PI_8" => Complex::new(std::f64::consts::FRAC_PI_8, 0.0),
    "LN_2" => Complex::new(std::f64::consts::LN_2, 0.0),
    "LN_10" => Complex::new(std::f64::consts::LN_10, 0.0),
    "LOG2_10" => Complex::new(std::f64::consts::LOG2_10, 0.0),
    "LOG2_E" => Complex::new(std::f64::consts::LOG2_E, 0.0),
    "LOG10_2" => Complex::new(std::f64::consts::LOG10_2, 0.0),
    "LOG10_E" => Complex::new(std::f64::consts::LOG10_E, 0.0),
    "PI" => Complex::new(std::f64::consts::PI, 0.0),
    "SQRT_2" => Complex::new(std::f64::consts::SQRT_2, 0.0),
    "TAU" => Complex::new(std::f64::consts::TAU, 0.0),
};

#[derive(Debug, Clone, Copy, PartialEq)]
enum UnaryOperatorKind {
    Positive,   Negative,
}

impl UnaryOperatorKind {
    pub fn from(s: &str) -> Option<Self> {
        match s {
            "+" => Some(Self::Positive),
            "-" => Some(Self::Negative),
            _ => None,
        }
    }

    pub fn apply(&self, x: Complex<f64>) -> Complex<f64> {
        match self {
            Self::Positive => x,
            Self::Negative => -x,
        }
    }
}

impl std::fmt::Display for UnaryOperatorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Positive => "+",
            Self::Negative => "-",
        };
        write!(f, "{}", s)
    }
}

#[derive(Debug, Clone, PartialEq)]
struct BinaryOperatorInfo {
    /// Operator precedence (higher value means higer precedence).
    pub precedence: u8,

    /// Wheter the operator is left associative.
    pub is_left_assoc: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum BinaryOperatorKind {
    Add,    Sub,    Mul,    Div,
    Pow,
}

impl BinaryOperatorKind {
    pub fn info(&self) -> BinaryOperatorInfo {
        match self {
            Self::Add | Self::Sub
                => BinaryOperatorInfo { precedence: 0, is_left_assoc: true },
            Self::Mul | Self::Div
                => BinaryOperatorInfo { precedence: 1, is_left_assoc: true },
            Self::Pow
                => BinaryOperatorInfo { precedence: 2, is_left_assoc: false },
        }
    }

    pub fn from(s: &str) -> Option<Self> {
        match s {
            "+" => Some(Self::Add),
            "-" => Some(Self::Sub),
            "*" => Some(Self::Mul),
            "/" => Some(Self::Div),
            "^" => Some(Self::Pow),
            _ => None,
        }
    }

    pub fn apply(&self, l: Complex<f64>, r: Complex<f64>) -> Complex<f64> {
        match self {
            Self::Add => l + r,
            Self::Sub => l - r,
            Self::Mul => l * r,
            Self::Div => l / r,
            Self::Pow => l.powc(r),
        }
    }
}

impl std::fmt::Display for BinaryOperatorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Add => "+",
            Self::Sub => "-",
            Self::Mul => "*",
            Self::Div => "/",
            Self::Pow => "^",
        };
        write!(f, "{}", s)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum FunctionKind {
    Sin,    Cos,    Tan,
    Asin,   Acos,   Atan,
    Sinh,   Cosh,   Tanh,
    Asinh,  Acosh,  Atanh,
    Exp,    Ln,     Log10,
    Sqrt,   Abs,
    Conj,

    Pow,
}

impl FunctionKind {
    pub fn from(s: &str) -> Option<FunctionKind> {
        match s {
            "sin"       => Some(FunctionKind::Sin),
            "cos"       => Some(FunctionKind::Cos),
            "tan"       => Some(FunctionKind::Tan),
            "asin"      => Some(FunctionKind::Asin),
            "acos"      => Some(FunctionKind::Acos),
            "atan"      => Some(FunctionKind::Atan),
            "sinh"      => Some(FunctionKind::Sinh),
            "cosh"      => Some(FunctionKind::Cosh),
            "tanh"      => Some(FunctionKind::Tanh),
            "asinh"     => Some(FunctionKind::Asinh),
            "acosh"     => Some(FunctionKind::Acosh),
            "atanh"     => Some(FunctionKind::Atanh),
            "exp"       => Some(FunctionKind::Exp),
            "ln"        => Some(FunctionKind::Ln),
            "log10"     => Some(FunctionKind::Log10),
            "sqrt"      => Some(FunctionKind::Sqrt),
            "abs"       => Some(FunctionKind::Abs),
            "conj"      => Some(FunctionKind::Conj),

            "pow"       => Some(FunctionKind::Pow),

            _ => None,
        }
    }

    pub fn arg_num(&self) -> usize {
        match self {
            FunctionKind::Pow => 2,
            _ => 1,
        }
    }

    pub fn apply(&self, args: &[Complex<f64>]) -> Complex<f64> {
        match self {
            Self::Sin => args[0].sin(),
            Self::Cos => args[0].cos(),
            Self::Tan => args[0].tan(),
            Self::Asin => args[0].asin(),
            Self::Acos => args[0].acos(),
            Self::Atan => args[0].atan(),
            Self::Sinh => args[0].sinh(),
            Self::Cosh => args[0].cosh(),
            Self::Tanh => args[0].tanh(),
            Self::Asinh => args[0].asinh(),
            Self::Acosh => args[0].acosh(),
            Self::Atanh => args[0].atanh(),
            Self::Exp => args[0].exp(),
            Self::Ln => args[0].ln(),
            Self::Log10 => args[0].log10(),
            Self::Sqrt => args[0].sqrt(),
            Self::Abs => Complex::from(args[0].abs()),
            Self::Conj => args[0].conj(),

            Self::Pow => args[0].powc(args[1]),
        }
    }
}

impl std::fmt::Display for FunctionKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            FunctionKind::Sin   => "sin",
            FunctionKind::Cos   => "cos",
            FunctionKind::Tan   => "tan",
            FunctionKind::Asin  => "asin",
            FunctionKind::Acos  => "acos",
            FunctionKind::Atan  => "atan",
            FunctionKind::Sinh  => "sinh",
            FunctionKind::Cosh  => "cosh",
            FunctionKind::Tanh  => "tanh",
            FunctionKind::Asinh => "asinh",
            FunctionKind::Acosh => "acosh",
            FunctionKind::Atanh => "atanh",
            FunctionKind::Exp   => "exp",
            FunctionKind::Ln    => "ln",
            FunctionKind::Log10 => "log10",
            FunctionKind::Sqrt  => "sqrt",
            FunctionKind::Abs   => "abs",
            FunctionKind::Conj  => "conj",

            FunctionKind::Pow   => "pow",
        };
        write!(f, "{}", s)
    }
}

/// Token enum representing different types of tokens.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Token<'a> {
    /// Numerical value token holding the resolved value.
    ///
    /// - User-defined variable with external value resolved at parse time.
    /// - Constant token for predefined mathematical constants.
    /// - Real or Imaginary number token.
    Number(Complex<f64>),

    /// Function argument token by position index.
    ///
    /// Represents an argument index in the function's parameter list.
    Argument(usize),

    /// Operator token.
    Operator(Lexeme<'a>),

    /// Unary Operator token.
    UnaryOperator(UnaryOperatorKind),

    /// Binary Operator token.
    BinaryOperator(BinaryOperatorKind),

    /// Function token.
    Function(FunctionKind),

    /// Left parenthesis token '('.
    LParen(Lexeme<'a>),

    /// Right parenthesis token ')'.
    RParen(Lexeme<'a>),

    /// Comma token ',' used as argument separator.
    Comma(Lexeme<'a>),
}

impl<'a> Token<'a> {
    pub fn from(
        lexeme: &Lexeme<'a>,
        args: &[&str],
        vars: &Variables,
    ) -> Result<Self, String> {
        let text = lexeme.text();

        if let Some(val) = parse_real(text)
            .or_else(|| parse_imaginary(text))
            .or_else(|| CONSTANTS.get(text).cloned())
            .or_else(|| vars.get(text).copied())
        {
            return Ok(Token::Number(val));
        }

        if let Some(position) = args.iter().position(|&arg| arg == text) {
            return Ok(Token::Argument(position));
        }

        /* We can't know whether the text is unary operator or binary operator
         * because some operator's strings are the same.
         * So we register only its lexeme. */
        if let Some(_) = UnaryOperatorKind::from(text) {
            return Ok(Token::Operator(lexeme.clone()));
        }
        if let Some(_) = BinaryOperatorKind::from(text) {
            return Ok(Token::Operator(lexeme.clone()));
        }

        if let Some(func_kind) = FunctionKind::from(text) {
            return Ok(Token::Function(func_kind));
        }

        match text {
            "(" => Ok(Token::LParen(lexeme.clone())),
            ")" => Ok(Token::RParen(lexeme.clone())),
            "," => Ok(Token::Comma(lexeme.clone())),
            _ =>Err(format!("Unknown string {}", lexeme_name_with_range!(lexeme))),
        }
    }

}

pub fn make_function_list<'a>(tokens: Vec<Token<'a>>)
    -> Vec<Box<dyn Fn(
        &mut Vec<Complex<f64>>, // tempolary stack of value in token
        &[Complex<f64>] // arguments list
    )>>
{
    let mut func_list: Vec<Box<dyn Fn(&mut Vec<Complex<f64>>, &[Complex<f64>])>> = Vec::new();

    for token in tokens {
        match token {
            Token::Number(val) => {
                func_list.push(Box::new(
                    move |stack, _args| stack.push(val)
                ));
            },
            Token::Argument(idx) => {
                func_list.push(Box::new(
                    move |stack, args| stack.push(args[idx])
                ));
            },
            Token::UnaryOperator(oper) => {
                func_list.push(Box::new(
                    move |stack, _args| {
                        let expr = stack.pop().unwrap();
                        stack.push(oper.apply(expr));
                    }
                ));
            },
            Token::BinaryOperator(oper) => {
                func_list.push(Box::new(
                    move |stack, _args| {
                        let r = stack.pop().unwrap();
                        let l = stack.pop().unwrap();
                        stack.push(oper.apply(l, r));
                    }
                ))
            },
            Token::Function(func) => {
                func_list.push(Box::new(
                    move |stack, _args| {
                        let n = func.arg_num();
                        let mut args: Vec<Complex<f64>> = Vec::with_capacity(n);
                        args.resize(n, Complex::new(0.0, 0.0));

                        for i in (0..n).rev() {
                            args[i] = stack.pop().unwrap();
                        }
                        stack.push(func.apply(&args));
                    }
                ));
            },
            _ => unreachable!("Invalid tokens found: use compiled tokens"),
        }
    }

    func_list
}

#[derive(Debug, Clone, PartialEq)]
pub enum AstNode {
    Number(Complex<f64>),
    Argument(usize),
    UnaryOperator {
        kind: UnaryOperatorKind,
        expr: Box<AstNode>,
    },
    BinaryOperator {
        kind: BinaryOperatorKind,
        left: Box<AstNode>,
        right: Box<AstNode>,
    },
    FunctionCall {
        kind: FunctionKind,
        args: Vec<AstNode>,
    },
}

impl AstNode {
    pub fn from<'a>(
        lexemes: &[Lexeme<'a>],
        vars: &Variables,
        args: &[&str]
    ) -> Result<Self, String> {
        parse_to_ast(lexemes, vars, args)
    }

    pub fn simplify(self) -> Self {
        match self {
            Self::UnaryOperator { kind, expr } => {
                let expr = expr.simplify();
                if let AstNode::Number(val ) = expr {
                    AstNode::Number(kind.apply(val))
                } else {
                    AstNode::UnaryOperator { kind, expr: Box::new(expr) }
                }
            },
            Self::BinaryOperator { kind, left, right } => {
                let left = left.simplify();
                let right = right.simplify();
                fold_binary(kind, left, right)
            },
            Self::FunctionCall { kind, args } => {
                let args: Vec<_> = args.into_iter().map(|a| a.simplify()).collect();
                if args.iter().all(|a| matches!(a, AstNode::Number(_))) {
                    let nums: Vec<Complex<f64>> = args.iter().map(|a| {
                        if let AstNode::Number(v) = a { *v } else { unreachable!() }
                    }).collect();
                    AstNode::Number(kind.apply(&nums))
                } else {
                    AstNode::FunctionCall { kind, args }
                }
            },
            _ => self,
        }
    }

    fn from_unary<'a>(
        stack: &mut Vec<Self>,
        oper: UnaryOperatorKind,
    ) -> Result<(), String> {
        let expr = stack.pop()
            .ok_or(format!("Missing unary opeator {}", oper))?;
        stack.push(Self::UnaryOperator { kind: oper, expr: Box::new(expr) });
        Ok(())
    }

    fn from_binary<'a>(
        stack: &mut Vec<Self>,
        oper: BinaryOperatorKind,
    ) -> Result<(), String> {
        let right = stack.pop()
            .ok_or(format!("Missing right operand for {}", oper))?;
        let left = stack.pop()
            .ok_or(format!("missing left operand for {}", oper))?;
        stack.push(Self::BinaryOperator{
            kind: oper,
            left: Box::new(left),
            right: Box::new(right),
        });
        Ok(())
    }

    fn from_function<'a>(
        stack: &mut Vec<Self>,
        func: FunctionKind,
    ) -> Result<(), String> {
        let n = func.arg_num();
        let mut args = Vec::new();
        args.resize(n, Self::Argument(0)); // Dummy Self for initializing
        for i in (0..n).rev() { // this is expanded to `for (i=n-1; i >= 0; i--)` by LLVM
            let arg = stack.pop()
                .ok_or(format!("Missing function argument for {}", func))?;
            args[i] = arg;
        }
        stack.push(Self::FunctionCall { kind: func, args });
        Ok(())
    }

    fn execute<'a>(&self, tokens: &mut Vec<Token<'a>>) {
        match self {
            Self::Number(val) => tokens.push(Token::Number(*val)),
            Self::Argument(i) => tokens.push(Token::Argument(*i)),
            Self::UnaryOperator { kind, expr } => {
                Self::execute(expr, tokens);
                tokens.push(Token::UnaryOperator(*kind));
            },
            Self::BinaryOperator { kind, left, right } => {
                Self::execute(left, tokens);
                Self::execute(right, tokens);
                tokens.push(Token::BinaryOperator(*kind));
            }
            Self::FunctionCall { kind, args } => {
                for arg in args {
                    Self::execute(arg, tokens);
                }
                tokens.push(Token::Function(*kind));
            },
        }
    }

    pub fn compile<'a>(&self) -> Vec<Token<'a>> {
        let mut tokens: Vec<Token<'a>> = Vec::new();
        self.execute(&mut tokens);
        tokens
    }
}

fn parse_in_right_paren<'a>(
    ast_nodes: &mut Vec<AstNode>,
    token_stack: &mut Vec<Token<'a>>,
    lexeme: &Lexeme<'a>,
) -> Result<(), String> {
    while let Some(token) = token_stack.pop() {
        match token {
            Token::LParen(_) => break,
            Token::UnaryOperator(oper) => AstNode::from_unary(ast_nodes, oper)?,
            Token::BinaryOperator(oper) => AstNode::from_binary(ast_nodes, oper)?,
            Token::Function(func) => AstNode::from_function(ast_nodes, func)?,
            _ => {
                return Err(format!(
                    "Unexpected token in stack when parsing in RParen at {s}..{e}",
                    s=lexeme.start(), e=lexeme.end(),
                ))
            },
        }
    }
    Ok(())
}

fn parse_in_comma<'a>(
    ast_nodes: &mut Vec<AstNode>,
    token_stack: &mut Vec<Token<'a>>,
    lexeme: &Lexeme<'a>,
) -> Result<(), String> {
    // use Vec::last() to avoid removing Left Paren from the stack
    while let Some(token) = token_stack.last() {
        match token {
            Token::LParen(_) => break,
            Token::UnaryOperator(oper) => AstNode::from_unary(ast_nodes, oper.clone())?,
            Token::BinaryOperator(oper) => AstNode::from_binary(ast_nodes, oper.clone())?,
            _ => {
                return Err(format!(
                    "Unexpected token in stack when parsing in Comma at {s}..{e}",
                    s=lexeme.start(), e=lexeme.end(),
                ))
            },
        }
    }
    Ok(())
}

fn parse_in_unary_operator<'a>(
    token_stack: &mut Vec<Token<'a>>,
    lexeme: Lexeme<'a>,
) -> Result<(), String> {
    if let Some(oper_kind) = UnaryOperatorKind::from(lexeme.text()) {
        token_stack.push(Token::UnaryOperator(oper_kind));
        Ok(())
    } else {
        Err(format!("Unknown unary operator {}", lexeme_name_with_range!(lexeme)))
    }
}

fn parse_in_binary_operator<'a>(
    ast_nodes: &mut Vec<AstNode>,
    token_stack: &mut Vec<Token<'a>>,
    lexeme: Lexeme<'a>,
) -> Result<(), String> {
    if let Some(oper_kind) = BinaryOperatorKind::from(lexeme.text()) {
        let oper_info = oper_kind.info();
        while let Some(Token::BinaryOperator(top_oper)) = token_stack.last().cloned() {
            if (oper_info.is_left_assoc && (top_oper.info().precedence < oper_info.precedence))
                || (!oper_info.is_left_assoc && (top_oper.info().precedence <= oper_info.precedence))
            {
                break;
            }
            token_stack.pop();
            AstNode::from_binary(ast_nodes, top_oper)?;
        }
        token_stack.push(Token::BinaryOperator(oper_kind));
        Ok(())
    } else {
        Err(format!("Unknown binary operator {}", lexeme_name_with_range!(lexeme)))
    }
}

fn parse_to_ast<'a>(lexemes: &[Lexeme<'a>], vars: &Variables, args: &[&str]) -> Result<AstNode, String> {
    let mut ast_nodes: Vec<AstNode> = Vec::new();
    let mut token_stack: Vec<Token> = Vec::new();
    // record whether the previous token is finished by value or not to evaluate the token is unary operator or binary operator.
    let mut prev_is_value = false;
    let mut lexemes = lexemes.iter().peekable();

    while let Some(lexeme) = lexemes.next() {
        let token = Token::from(lexeme, args, vars)?;
        match token {
            Token::Number(val) => {
                ast_nodes.push(AstNode::Number(val));
                prev_is_value = true;
            },
            Token::Argument(pos) => {
                ast_nodes.push(AstNode::Argument(pos));
                prev_is_value = true;
            },
            Token::Operator(lexeme) => {
                match prev_is_value {
                    true => parse_in_binary_operator(&mut ast_nodes, &mut token_stack, lexeme)?,
                    false => parse_in_unary_operator(&mut token_stack, lexeme)?,
                };
                prev_is_value = false;
            },
            Token::Function(_) => {
                token_stack.push(token);
                prev_is_value = false;
            },
            Token::LParen(_) => {
                token_stack.push(token);
                prev_is_value = false; // The operator next to LParen is unary operator; ex) cos(-x), 3 * (-2)
            },
            Token::RParen(_) => {
                parse_in_right_paren(&mut ast_nodes, &mut token_stack, lexeme)?;
                prev_is_value = true; // The operator next to RParen is binary operator; ex) sin(x) + 2, (x+2)/(x-3)
            },
            Token::Comma(_) => {
                parse_in_comma(&mut ast_nodes, &mut token_stack, lexeme)?;
                prev_is_value = false; // The operator next to Comma is unary operator; ex) pow(x, -3)
            },
            _ => return Err(format!("Invalid token kind made from {}", lexeme_name_with_range!(lexeme))),
        }
    }

    while let Some(token) = token_stack.pop() {
        match token {
            Token::UnaryOperator(oper) => AstNode::from_unary(&mut ast_nodes, oper)?,
            Token::BinaryOperator(oper) => AstNode::from_binary(&mut ast_nodes, oper)?,
            Token::Function(func) => AstNode::from_function(&mut ast_nodes, func)?,
            _ => return Err("Unexpected token at the end".into()),
        }
    }

    let ret = ast_nodes.pop()
        .expect("Fail to parse to AST. There are NO AST node remaining.");

    if !ast_nodes.is_empty() {
        return Err("Fail to parse to AST. There are too AST node remaining.".into());
    }
    Ok(ret)
}

fn fold_binary(kind: BinaryOperatorKind, left: AstNode, right: AstNode) -> AstNode {
    match (left, right) {
        (AstNode::Number(l), AstNode::Number(r))
            => AstNode::Number(kind.apply(l, r)),
        (AstNode::BinaryOperator { kind: k1, left: l1, right: r1 }, AstNode::Number(r))
            if matches!(k1, BinaryOperatorKind::Add | BinaryOperatorKind::Mul) =>
        {
            if let AstNode::Number(r1_val) = *r1 {
                AstNode::BinaryOperator {
                    kind,
                    left: l1,
                    right: Box::new(AstNode::Number(k1.apply( r1_val, r))),
                }
            } else {
                AstNode::BinaryOperator {
                    kind,
                    left: Box::new(AstNode::BinaryOperator {
                        kind: k1,
                        left: l1,
                        right: r1
                    }),
                    right: Box::new(AstNode::Number(r)),
                }
            }
        }
        (l, r) => AstNode::BinaryOperator { kind, left: Box::new(l), right: Box::new(r) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;
    use crate::lexer;
    use approx::assert_abs_diff_eq;

    macro_rules! assert_astnode_eq {
        ($left:expr, $right:expr) => {{
            fn inner(left: &AstNode, right: &AstNode) {
                let epsilon = 1.0e-12;
                match (left, right) {
                    (AstNode::Number(l), AstNode::Number(r)) => {
                        assert_abs_diff_eq!(l.re(), r.re(), epsilon = epsilon);
                        assert_abs_diff_eq!(l.im(), r.im(), epsilon = epsilon);
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
    fn test_unary_operator_kind_from() {
        assert_eq!(UnaryOperatorKind::from("+"), Some(UnaryOperatorKind::Positive));
        assert_eq!(UnaryOperatorKind::from("-"), Some(UnaryOperatorKind::Negative));
        assert_eq!(UnaryOperatorKind::from("*"), None);
        assert_eq!(UnaryOperatorKind::from(""), None);
        assert_eq!(UnaryOperatorKind::from("x"), None);
    }

    #[test]
    fn test_binary_operator_kind_info() {
        let add_info = BinaryOperatorKind::Add.info();
        assert_eq!(add_info.precedence, 0);
        assert!(add_info.is_left_assoc);

        let sub_info = BinaryOperatorKind::Sub.info();
        assert_eq!(sub_info.precedence, 0);
        assert!(sub_info.is_left_assoc);

        let mul_info = BinaryOperatorKind::Mul.info();
        assert_eq!(mul_info.precedence, 1);
        assert!(mul_info.is_left_assoc);

        let div_info = BinaryOperatorKind::Div.info();
        assert_eq!(div_info.precedence, 1);
        assert!(div_info.is_left_assoc);

        let pow_info = BinaryOperatorKind::Pow.info();
        assert_eq!(pow_info.precedence, 2);
        assert!(!pow_info.is_left_assoc);
    }

    #[test]
    fn test_binary_operator_kind_from() {
        assert_eq!(BinaryOperatorKind::from("+"), Some(BinaryOperatorKind::Add));
        assert_eq!(BinaryOperatorKind::from("-"), Some(BinaryOperatorKind::Sub));
        assert_eq!(BinaryOperatorKind::from("*"), Some(BinaryOperatorKind::Mul));
        assert_eq!(BinaryOperatorKind::from("/"), Some(BinaryOperatorKind::Div));
        assert_eq!(BinaryOperatorKind::from("^"), Some(BinaryOperatorKind::Pow));

        assert_eq!(BinaryOperatorKind::from(""), None);
        assert_eq!(BinaryOperatorKind::from("x"), None);
        assert_eq!(BinaryOperatorKind::from("%"), None);
    }

    #[test]
    fn test_function_kind_from() {
        assert_eq!(FunctionKind::from("sin"), Some(FunctionKind::Sin));
        assert_eq!(FunctionKind::from("cos"), Some(FunctionKind::Cos));
        assert_eq!(FunctionKind::from("tan"), Some(FunctionKind::Tan));
        assert_eq!(FunctionKind::from("asin"), Some(FunctionKind::Asin));
        assert_eq!(FunctionKind::from("acos"), Some(FunctionKind::Acos));
        assert_eq!(FunctionKind::from("atan"), Some(FunctionKind::Atan));
        assert_eq!(FunctionKind::from("sinh"), Some(FunctionKind::Sinh));
        assert_eq!(FunctionKind::from("cosh"), Some(FunctionKind::Cosh));
        assert_eq!(FunctionKind::from("tanh"), Some(FunctionKind::Tanh));
        assert_eq!(FunctionKind::from("asinh"), Some(FunctionKind::Asinh));
        assert_eq!(FunctionKind::from("acosh"), Some(FunctionKind::Acosh));
        assert_eq!(FunctionKind::from("atanh"), Some(FunctionKind::Atanh));
        assert_eq!(FunctionKind::from("exp"), Some(FunctionKind::Exp));
        assert_eq!(FunctionKind::from("ln"), Some(FunctionKind::Ln));
        assert_eq!(FunctionKind::from("log10"), Some(FunctionKind::Log10));
        assert_eq!(FunctionKind::from("sqrt"), Some(FunctionKind::Sqrt));
        assert_eq!(FunctionKind::from("pow"), Some(FunctionKind::Pow));

        assert_eq!(FunctionKind::from(""), None);
        assert_eq!(FunctionKind::from("log"), None);
        assert_eq!(FunctionKind::from("abc"), None);
    }

    #[test]
    fn test_function_kind_arg_num() {
        let single_arg_functions = [
            FunctionKind::Sin,
            FunctionKind::Cos,
            FunctionKind::Tan,
            FunctionKind::Asin,
            FunctionKind::Acos,
            FunctionKind::Atan,
            FunctionKind::Sinh,
            FunctionKind::Cosh,
            FunctionKind::Tanh,
            FunctionKind::Asinh,
            FunctionKind::Acosh,
            FunctionKind::Atanh,
            FunctionKind::Exp,
            FunctionKind::Ln,
            FunctionKind::Log10,
            FunctionKind::Sqrt,
        ];

        for func in &single_arg_functions {
            assert_eq!(func.arg_num(), 1, "{:?} should have 1 argument", func);
        }

        let double_args_functions = [
            FunctionKind::Pow,
        ];

        for func in &double_args_functions {
            assert_eq!(func.arg_num(), 2, "{:?} should have 2 argument", func);
        }
    }

    #[test]
    fn test_number_token() {
        let lex = Lexeme::new("3.14", 0..4);
        let vars = Variables::new();
        let args: [&str; 0] = [];
        let token = Token::from(&lex, &args, &vars).unwrap();
        match token {
            Token::Number(val) => assert_eq!(val, Complex::new(3.14, 0.0)),
            _ => panic!("Expected Number token"),
        }
    }

    #[test]
    fn test_imaginary_number() {
        let lex = Lexeme::new("2i", 0..2);
        let vars = Variables::new();
        let args: [&str; 0] = [];
        let token = Token::from(&lex, &args, &vars).unwrap();
        match token {
            Token::Number(val) => assert_eq!(val, Complex::new(0.0, 2.0)),
            _ => panic!("Expected Number token"),
        }
    }

    #[test]
    fn test_constant_token() {
        let lex = Lexeme::new("PI", 0..2);
        let vars = Variables::new();
        let args: [&str; 0] = [];
        let token = Token::from(&lex, &args, &vars).unwrap();
        match token {
            Token::Number(val) => assert_eq!(val, Complex::new(std::f64::consts::PI, 0.0)),
            _ => panic!("Expected Number token"),
        }
    }

    #[test]
    fn test_argument_token() {
        let lex = Lexeme::new("arg0", 0..4);
        let vars = Variables::new();
        let args = ["arg0"];
        let token = Token::from(&lex, &args, &vars).unwrap();
        match token {
            Token::Argument(pos) => assert_eq!(pos, 0),
            _ => panic!("Expected Argument token"),
        }
    }

    #[test]
    fn test_operator_token() {
        let lex = Lexeme::new("+", 0..1);
        let vars = Variables::new();
        let args: [&str; 0] = [];
        let token = Token::from(&lex, &args, &vars).unwrap();
        match token {
            Token::Operator(_) => {}, // OK
            _ => panic!("Expected Operator token"),
        }
    }

    #[test]
    fn test_function_token() {
        let lex = Lexeme::new("sin", 0..3);
        let vars = Variables::new();
        let args: [&str; 0] = [];
        let token = Token::from(&lex, &args, &vars).unwrap();
        match token {
            Token::Function(f) => assert_eq!(f, FunctionKind::Sin),
            _ => panic!("Expected Function token"),
        }
    }

    #[test]
    fn test_parentheses_and_comma() {
        let lex_l = Lexeme::new("(", 0..1);
        let lex_r = Lexeme::new(")", 0..1);
        let lex_c = Lexeme::new(",", 0..1);
        let vars = Variables::new();
        let args: [&str; 0] = [];

        assert!(matches!(Token::from(&lex_l, &args, &vars).unwrap(), Token::LParen(_)));
        assert!(matches!(Token::from(&lex_r, &args, &vars).unwrap(), Token::RParen(_)));
        assert!(matches!(Token::from(&lex_c, &args, &vars).unwrap(), Token::Comma(_)));
    }

    #[test]
    fn test_unknown_string() {
        let lex = Lexeme::new("unknown", 0..7);
        let vars = Variables::new();
        let args: [&str; 0] = [];
        let res = Token::from(&lex, &args, &vars);
        assert!(res.is_err());
    }

    #[test]
    fn test_single_number_astnode() {
        let lexemes = lexer::from("42");
        let ast = parse_to_ast(&lexemes, &Variables::new(), &[]).unwrap();
        match ast {
            AstNode::Number(val) => assert_eq!(val, Complex::new(42.0, 0.0)),
            _ => panic!("Expected Number AST node"),
        }
    }

    #[test]
    fn test_unary_operator_negative_astnode() {
        let lexemes = lexer::from("- 3");
        let ast = parse_to_ast(&lexemes, &Variables::new(), &[]).unwrap();
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
        let ast = parse_to_ast(&lexemes, &Variables::new(), &[]).unwrap();
        // expected: (2 + (3 * 4))
        match ast {
            AstNode::BinaryOperator { kind, left, right } => {
                assert_eq!(kind, BinaryOperatorKind::Add);
                match *right {
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
        let ast = parse_to_ast(&lexemes, &Variables::new(), &[]).unwrap();
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
        let ast = parse_to_ast(&lexemes, &Variables::new(), &[]).unwrap();
        match ast {
            AstNode::FunctionCall { kind, args } => {
                assert_eq!(kind, FunctionKind::Sin);
                assert_eq!(args.len(), 1);
                assert_eq!(args[0], AstNode::Number(Complex::new(0.0, 0.0)));
            }
            _ => panic!("Expected Function node"),
        }
    }

    #[test]
    fn test_function_multiple_args_astnode() {
        let lexemes = lexer::from("pow ( 2 , 3 )");
        let ast = parse_to_ast(&lexemes, &Variables::new(), &[]).unwrap();
        match ast {
            AstNode::FunctionCall { kind, args } => {
                assert_eq!(kind, FunctionKind::Pow);
                assert_eq!(args.len(), 2);
                assert_eq!(args[0], AstNode::Number(Complex::new(2.0, 0.0)));
                assert_eq!(args[1], AstNode::Number(Complex::new(3.0, 0.0)));
            }
            _ => panic!("Expected Function node"),
        }
    }

    #[test]
    fn test_imaginary_number_astnode() {
        let lexemes = lexer::from("5i");
        let ast = parse_to_ast(&lexemes, &Variables::new(), &[]).unwrap();
        assert_eq!(ast, AstNode::Number(Complex::new(0.0, 5.0)));
    }

    #[test]
    fn test_unknown_token_astnode_error() {
        let lexemes = lexer::from("@");
        let res = parse_to_ast(&lexemes, &Variables::new(), &[]);
        assert!(res.is_err());
    }

    #[test]
    fn test_simplify_number() {
        let node = AstNode::Number(Complex::new(3.0, 0.0));
        assert_astnode_eq!(node.clone().simplify(), node);
    }

    #[test]
    fn test_simplify_unary_operator() {
        let node = AstNode::UnaryOperator {
            kind: UnaryOperatorKind::Negative,
            expr: Box::new(AstNode::Number(Complex::new(2.0, 0.0))),
        };
        let simplified = node.simplify();
        assert_astnode_eq!(simplified, AstNode::Number(Complex::new(-2.0, 0.0)));
    }

    #[test]
    fn test_simplify_binary_operator_full() {
        let node = AstNode::BinaryOperator {
            kind: BinaryOperatorKind::Add,
            left: Box::new(AstNode::Number(Complex::new(2.0, 0.0))),
            right: Box::new(AstNode::Number(Complex::new(3.0, 0.0))),
        };
        let simplified = node.simplify();
        assert_astnode_eq!(simplified, AstNode::Number(Complex::new(5.0, 0.0)));
    }

    #[test]
    fn test_simplify_binary_operator_partial() {
        let node = AstNode::BinaryOperator {
            kind: BinaryOperatorKind::Add,
            left: Box::new(AstNode::Argument(0)),
            right: Box::new(AstNode::Number(Complex::new(3.0, 0.0))),
        };
        let simplified = node.simplify();
        assert_astnode_eq!(
            simplified,
            AstNode::BinaryOperator {
                kind: BinaryOperatorKind::Add,
                left: Box::new(AstNode::Argument(0)),
                right: Box::new(AstNode::Number(Complex::new(3.0, 0.0))),
            }
        );
    }

    #[test]
    fn test_simplify_binary_operator_chain() {
        // x + 2 + 3 -> x + 5
        let node = AstNode::BinaryOperator {
            kind: BinaryOperatorKind::Add,
            left: Box::new(AstNode::BinaryOperator {
                kind: BinaryOperatorKind::Add,
                left: Box::new(AstNode::Argument(0)),
                right: Box::new(AstNode::Number(Complex::new(2.0, 0.0))),
            }),
            right: Box::new(AstNode::Number(Complex::new(3.0, 0.0))),
        };
        let simplified = node.simplify();
        assert_astnode_eq!(
            simplified,
            AstNode::BinaryOperator {
                kind: BinaryOperatorKind::Add,
                left: Box::new(AstNode::Argument(0)),
                right: Box::new(AstNode::Number(Complex::new(5.0, 0.0))),
            }
        );

        // x * 2 * 3 * 4 -> x * 24
        let node = AstNode::BinaryOperator {
            kind: BinaryOperatorKind::Mul,
            left: Box::new(AstNode::BinaryOperator {
                kind: BinaryOperatorKind::Mul,
                left: Box::new(AstNode::BinaryOperator {
                    kind: BinaryOperatorKind::Mul,
                    left: Box::new(AstNode::Argument(0)),
                    right: Box::new(AstNode::Number(Complex::new(2.0, 0.0))),
                }),
                right: Box::new(AstNode::Number(Complex::new(3.0, 0.0))),
            }),
            right: Box::new(AstNode::Number(Complex::new(4.0, 0.0))),
        };
        let simplified = node.simplify();
        assert_astnode_eq!(
            simplified,
            AstNode::BinaryOperator {
                kind: BinaryOperatorKind::Mul,
                left: Box::new(AstNode::Argument(0)),
                right: Box::new(AstNode::Number(Complex::new(24.0, 0.0))),
            }
        );

        // 2 * x + 3 -> not changed
        let node = AstNode::BinaryOperator {
            kind: BinaryOperatorKind::Add,
            left: Box::new(AstNode::BinaryOperator {
                kind: BinaryOperatorKind::Mul,
                left: Box::new(AstNode::Number(Complex::new(2.0, 0.0))),
                right: Box::new(AstNode::Argument(0))
            }),
            right: Box::new(AstNode::Number(Complex::new(3.0, 0.0))),
        };
        let simplified = node.clone().simplify();
        assert_astnode_eq!(simplified, node)
    }

    #[test]
    fn test_simplify_function_call_full() {
        let node = AstNode::FunctionCall {
            kind: FunctionKind::Pow,
            args: vec![
                AstNode::Number(Complex::new(2.0, 0.0)),
                AstNode::Number(Complex::new(3.0, 0.0)),
            ],
        };
        let simplified = node.simplify();
        assert_astnode_eq!(simplified, AstNode::Number(Complex::new(8.0, 0.0)));

        let node = AstNode::FunctionCall {
            kind: FunctionKind::Exp,
            args: vec![
                AstNode::Number(Complex::new(2.0, 0.0)),
            ],
        };
        let simplified = node.simplify();
        assert_astnode_eq!(simplified, AstNode::Number(Complex::new(2.0, 0.0).exp()));
    }

    #[test]
    fn test_simplify_function_call_partial() {
        let node = AstNode::FunctionCall {
            kind: FunctionKind::Pow,
            args: vec![
                AstNode::Argument(0),
                AstNode::Number(Complex::new(3.0, 0.0)),
            ],
        };
        let simplified = node.simplify();
        assert_astnode_eq!(
            simplified,
            AstNode::FunctionCall {
                kind: FunctionKind::Pow,
                args: vec![
                    AstNode::Argument(0),
                    AstNode::Number(Complex::new(3.0, 0.0)),
                ],
            }
        );
    }

    #[test]
    fn test_compile_number() {
        let ast = AstNode::Number(Complex::new(1.0, 0.0));
        let tokens = ast.compile();
        assert_eq!(tokens, vec![Token::Number(Complex::new(1.0, 0.0))]);
    }

    #[test]
    fn test_compile_argument() {
        let ast = AstNode::Argument(1);
        let tokens = ast.compile();
        assert_eq!(tokens, vec![Token::Argument(1)]);
    }

    #[test]
    fn test_compile_unary_operator() {
        let ast = AstNode::UnaryOperator {
            kind: UnaryOperatorKind::Negative,
            expr: Box::new(AstNode::Number(Complex::new(1.0, 0.0)))
        };
        let tokens = ast.compile();
        assert_eq!(
            tokens,
            vec![Token::Number(Complex::new(1.0, 0.0)), Token::UnaryOperator(UnaryOperatorKind::Negative)]
        );
    }

    #[test]
    fn test_compile_binary_operator() {
        let ast = AstNode::BinaryOperator {
            kind: BinaryOperatorKind::Add,
            left: Box::new(AstNode::Number(Complex::new(1.0, 0.0))),
            right: Box::new(AstNode::Argument(1))
        };
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
        let ast = AstNode::FunctionCall {
            kind: FunctionKind::Sin,
            args: vec![AstNode::Number(Complex::new(1.0, 0.0))]
        };
        let tokens = ast.compile();
        assert_eq!(tokens, vec![Token::Number(Complex::new(1.0, 0.0)), Token::Function(FunctionKind::Sin)]);
    }

    #[test]
    fn test_compile_function_multi_arguments() {
        let ast = AstNode::FunctionCall {
            kind: FunctionKind::Pow,
            args: vec![AstNode::Number(Complex::new(2.0, 0.0)), AstNode::Argument(0)]
        };
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
        let ast = AstNode::FunctionCall {
            kind: FunctionKind::Cos,
            args: vec![AstNode::BinaryOperator {
                kind: BinaryOperatorKind::Add,
                left: Box::new(AstNode::Number(Complex::new(1.0, 0.0))),
                right: Box::new(AstNode::Number(Complex::new(2.0, 0.0)))
            }],
        };
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
