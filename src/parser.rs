//! # parser.rs
//!
//! This module provides the functionality to parse mathematical expressions into an Abstract Syntax Tree (AST)
//! and then compile them into executable tokens.
//!
//! It supports:
//! - Real and complex numbers
//! - Variables (`Variables` from `variable.rs`)
//! - Constants (predefined math constants like PI, E, etc.)
//! - Unary and binary operators
//! - Built-in functions (sin, cos, ln, sqrt, etc.)
//! - User-defined functions (`UserDefinedFunction` and `UserDefinedTable`)
//!
//! The parsing process converts a sequence of lexemes (from the lexer) into an `AstNode` tree,
//! which can then be simplified or compiled into a sequence of executable tokens.
//!
//! # Notes
//! - Operator precedence and associativity are handled according to standard math rules.
//! - Unary and binary operators sharing symbols (like "-" for negation and subtraction) are disambiguated
//!   based on context.
//! - AST simplification will precompute nodes if all arguments are constant numbers.
//! - User-defined functions must be registered in `UserDefinedTable` before parsing expressions using them.

use crate::lexer::Lexeme;
use crate::lexer::IMAGINARY_UNIT;
use crate::{variable::FunctionCall, Variables, UserDefinedFunction, UserDefinedTable};
use num_complex::Complex;
use num_complex::ComplexFloat;
use phf::Map;
use phf_macros::phf_map;

macro_rules! lexeme_name_with_range {
    ($lexeme: expr) => {
        format!("{name} at {start}..{end}", name=$lexeme.text(), start=$lexeme.start(), end=$lexeme.end())
    };
}

const DIFFELENCIAL_OPERATOR_STR: &str = "diff";

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

pub mod constant {
    use crate::parser::CONSTANTS;

    /// Returns a list of supported mathematical constant names.
    pub fn names() -> Vec<&'static str> {
        CONSTANTS.keys().cloned().collect()
    }
}

#[doc(hidden)]
/// Internal macro to define all unary operators.
///
/// This macro is **not intended for public use**.
/// It centralizes the enum variants, string representation, and apply logic for unary operators.
macro_rules! unary_operator_kind {
    ($($name:ident => { symbol: $symbol:expr, apply: $apply:expr }),* $(,)?) => {
        /// Represents a unary operator in a mathematical expression.
        #[derive(Debug, Clone, Copy, PartialEq)]
        pub enum UnaryOperatorKind {
            $($name),*
        }

        impl UnaryOperatorKind {
            /// Converts a string representation to a `UnaryOperatorKind`.
            pub fn from(s: &str) -> Option<Self> {
                match s {
                    $( $symbol => Some(Self::$name), )*
                    _ => None,
                }
            }

            /// Applies the unary operator to a complex number.
            pub fn apply(&self, x: Complex<f64>) -> Complex<f64> {
                match self {
                    $( Self::$name => $apply(x), )*
                }
            }

            /// Returns a list of all supported unary operator symbols.
            pub fn names() -> Vec<&'static str> {
                vec![$($symbol),*]
            }
        }

        impl std::fmt::Display for UnaryOperatorKind {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                let s = match self {
                    $( Self::$name => $symbol, )*
                };
                write!(f, "{}", s)
            }
        }
    };
}

unary_operator_kind! {
    Positive => { symbol: "+", apply: |x: Complex<f64>| x },
    Negative => { symbol: "-", apply: |x: Complex<f64>| -x },
}


/// Information about a binary operator in a mathematical expression.
///
/// Contains the operator's precedence and associativity, which are used
/// when parsing expressions to determine the order of operations.
#[derive(Debug, Clone, PartialEq)]
pub struct BinaryOperatorInfo {
    /// Operator precedence (higher value means higher precedence).
    pub precedence: u8,

    /// Whether the operator is left-associative.
    pub is_left_assoc: bool,
}

#[doc(hidden)]
/// Internal macro to define all binary operators.
///
/// This macro is **not intended for public use**.
/// It centralizes the enum variants, string representation, precedence, associativity, and apply logic.
macro_rules! binary_operators {
    ($($name:ident => {
        symbol: $symbol:expr,
        precedence: $prec:expr,
        left_assoc: $assoc:expr,
        apply: $apply:expr
    }),* $(,)?) => {
        /// Represents a binary operator in a mathematical expression.
        #[derive(Debug, Clone, Copy, PartialEq)]
        pub enum BinaryOperatorKind {
            $($name),*
        }

        impl BinaryOperatorKind {
            /// Returns operator precedence and associativity.
            pub fn info(&self) -> BinaryOperatorInfo {
                match self {
                    $(Self::$name => BinaryOperatorInfo { precedence: $prec, is_left_assoc: $assoc },)*
                }
            }

            /// Converts a string to the corresponding operator.
            pub fn from(s: &str) -> Option<Self> {
                match s {
                    $($symbol => Some(Self::$name),)*
                    _ => None,
                }
            }

            /// Applies the operator to two complex numbers.
            pub fn apply(&self, l: Complex<f64>, r: Complex<f64>) -> Complex<f64> {
                match self {
                    $(Self::$name => $apply(l, r),)*
                }
            }

            /// Returns a list of all supported binary operator symbols.
            pub fn names() -> Vec<&'static str> {
                vec![$($symbol),*]
            }
        }

        impl std::fmt::Display for BinaryOperatorKind {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                let s = match self {
                    $(Self::$name => $symbol,)*
                };
                write!(f, "{}", s)
            }
        }
    };
}

binary_operators! {
    Add => { symbol: "+", precedence: 0, left_assoc: true,  apply: |l: Complex<f64>, r: Complex<f64>| l + r },
    Sub => { symbol: "-", precedence: 0, left_assoc: true,  apply: |l: Complex<f64>, r: Complex<f64>| l - r },
    Mul => { symbol: "*", precedence: 1, left_assoc: true,  apply: |l: Complex<f64>, r: Complex<f64>| l * r },
    Div => { symbol: "/", precedence: 1, left_assoc: true,  apply: |l: Complex<f64>, r: Complex<f64>| l / r },
    Pow => { symbol: "^", precedence: 2, left_assoc: false, apply: |l: Complex<f64>, r: Complex<f64>| l.powc(r) },
}


#[doc(hidden)]
/// Internal macro for defining built-in mathematical functions.
///
/// This macro is **not intended for public use**.
/// It centralizes the declaration of functions in one place and automatically
/// generates the [`FunctionKind`] enum and its implementations (`from`, `arity`,
/// `apply`, and `Display`).
///
/// # Developer Notes
/// - Each function must declare:
///   - The enum variant name
///   - Its canonical string name
///   - The number of arguments it takes
///   - How it is applied to its arguments
/// - To add or remove a built-in function, update the list inside this macro.
/// - User-defined functions should *not* be added here; use [`UserDefinedTable`] instead.
///
/// This macro ensures consistency across:
/// - Parsing (string → enum)
/// - Evaluation (apply)
/// - Formatting (Display)
macro_rules! functions {
    ($( $variant: ident => {
        name: $name:expr,
        arity: $arity:expr,
        apply: |$a:ident| $body:expr
    }, )*) => {
        /// Represents a mathematical function that can be applied to one or more complex numbers.
        ///
        /// Supports common trigonometric, hyperbolic, logarithmic, exponential, and arithmetic functions,
        /// as well as complex-specific operations like conjugation.
        #[derive(Debug, Clone, Copy, PartialEq)]
        pub enum FunctionKind {
            $( $variant, )*
        }

        impl FunctionKind {
            /// Converts a string representation of a function into a `FunctionKind`.
            ///
            /// Returns `None` if the string does not match any supported function.
            pub fn from(s: &str) -> Option<Self> {
                match s {
                    $( $name => Some(Self::$variant), )*
                    _ => None,
                }
            }

            /// Returns a list of all supported function names.
            pub fn names() -> Vec<&'static str> {
                vec![$($name),*]
            }
        }

        impl FunctionCall for FunctionKind {
            /// Returns arity, the number of arguments that the function takes.
            fn arity(&self) -> usize {
                match self {
                    $( Self::$variant => $arity, )*
                }
            }

            /// Applies the function to a slice of complex numbers and returns the result.
            ///
            /// The number of elements in `args` must match the value returned by `arity`.
            fn apply(&self, args: &[Complex<f64>]) -> Complex<f64> {
                match self {
                    $( Self::$variant => {
                        let $a = args;
                        $body
                    }, )*
                }
            }
        }

        impl std::fmt::Display for FunctionKind {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                let s = match self {
                    $( Self::$variant => $name, )*
                };
                write!(f, "{}", s)
            }
        }
    };
}

functions! {
    Sin     => { name: "sin",       arity: 1,   apply: |a| a[0].sin() },
    Cos     => { name: "cos",       arity: 1,   apply: |a| a[0].cos() },
    Tan     => { name: "tan",       arity: 1,   apply: |a| a[0].tan() },
    Asin    => { name: "asin",      arity: 1,   apply: |a| a[0].asin() },
    Acos    => { name: "acos",      arity: 1,   apply: |a| a[0].acos() },
    Atan    => { name: "atan",      arity: 1,   apply: |a| a[0].atan() },
    Sinh    => { name: "sinh",      arity: 1,   apply: |a| a[0].sinh() },
    Cosh    => { name: "cosh",      arity: 1,   apply: |a| a[0].cosh() },
    Tanh    => { name: "tanh",      arity: 1,   apply: |a| a[0].tanh() },
    Asinh   => { name: "asinh",     arity: 1,   apply: |a| a[0].asinh() },
    Acosh   => { name: "acosh",     arity: 1,   apply: |a| a[0].acosh() },
    Atanh   => { name: "atanh",     arity: 1,   apply: |a| a[0].atanh() },
    Exp     => { name: "exp",       arity: 1,   apply: |a| a[0].exp() },
    Ln      => { name: "ln",        arity: 1,   apply: |a| a[0].ln() },
    Log10   => { name: "log10",     arity: 1,   apply: |a| a[0].log10() },
    Sqrt    => { name: "sqrt",      arity: 1,   apply: |a| a[0].sqrt() },
    Abs     => { name: "abs",       arity: 1,   apply: |a| Complex::from(a[0].abs()) },
    Conj    => { name: "conj",      arity: 1,   apply: |a| a[0].conj() },
    Pow     => { name: "pow",       arity: 2,   apply: |a| a[0].powc(a[1]) },
    Powi    => { name: "powi",      arity: 2,   apply: |a| a[0].powi(a[1].re as i32) },
}

/// Represents a parsed token in a mathematical expression.
///
/// Tokens are produced by the lexer and consumed by the parser to build an AST.
/// This enum covers all possible token types, including numbers, operators,
/// functions, parentheses, commas, and user-defined functions.
#[derive(Debug, Clone, PartialEq)]
pub enum Token<'a> {
    /// Numerical value token holding a resolved complex number.
    ///
    /// This variant can represent:
    /// - User-defined variables with values resolved at parse time.
    /// - Predefined mathematical constants.
    /// - Literal real or imaginary numbers.
    Number(Complex<f64>),

    /// Function argument token by its position index in the argument list.
    Argument(usize),

    /// Generic operator token holding the original lexeme.
    Operator(Lexeme<'a>),

    /// Unary operator token (e.g., `+`, `-`).
    UnaryOperator(UnaryOperatorKind),

    /// Binary operator token (e.g., `+`, `-`, `*`, `/`, `^`).
    BinaryOperator(BinaryOperatorKind),

    /// Differential operator token.
    DiffOperator(Lexeme<'a>),

    /// Standard mathematical function token (e.g., `sin`, `cos`, `exp`).
    Function(FunctionKind),

    /// User-defined function token.
    UserFunction(UserDefinedFunction),

    /// Left parenthesis `'('`.
    LParen(Lexeme<'a>),

    /// Right parenthesis token `')'``.
    RParen(Lexeme<'a>),

    /// Comma `','`` used as argument separator.
    Comma(Lexeme<'a>),
}

impl<'a> Token<'a> {
    /// Attempts to parse a string as a real number.
    fn parse_real(s: &str) -> Option<Complex<f64>> {
        s.parse::<f64>().ok().map(Complex::from)
    }

    /// Attempts to parse a string as an imaginary number.
    fn parse_imaginary(s: &str) -> Option<Complex<f64>> {
        let num_part = s.strip_suffix(IMAGINARY_UNIT)?;
        if num_part.is_empty() {
            return Some(Complex::I);
        }
        match num_part.parse::<f64>() {
            Ok(val) => Some(Complex::new(0.0, val)),
            Err(_) => None,
        }
    }

    /// Converts a lexeme into a corresponding `Token`.
    ///
    /// Resolves numbers, variables, constants, operators, functions, and user-defined functions.
    /// Returns an error if the lexeme cannot be recognized.
    pub fn from(
        lexeme: &Lexeme<'a>,
        args: &[&str],
        vars: &'a Variables,
        users: &'a UserDefinedTable,
    ) -> Result<Self, String> {
        let text = lexeme.text();

        if let Some(val) = Self::parse_real(text)
            .or_else(|| Self::parse_imaginary(text))
            .or_else(|| CONSTANTS.get(text).cloned())
            .or_else(|| vars.get(text).copied())
        {
            return Ok(Token::Number(val));
        }

        if text == DIFFELENCIAL_OPERATOR_STR {
            return Ok(Token::DiffOperator(*lexeme));
        }

        if let Some(position) = args.iter().position(|&arg| arg == text) {
            return Ok(Token::Argument(position));
        }

        /* We can't know whether the text is unary operator or binary operator
         * because some operator's strings are the same.
         * So we register only its lexeme. */
        if UnaryOperatorKind::from(text).is_some() {
            return Ok(Token::Operator(*lexeme));
        }
        if BinaryOperatorKind::from(text).is_some() {
            return Ok(Token::Operator(*lexeme));
        }

        if let Some(func_kind) = FunctionKind::from(text) {
            return Ok(Token::Function(func_kind));
        }

        if let Some(user_func) = users.get(text) {
            return Ok(Token::UserFunction(user_func.clone()));
        }

        match text {
            "(" => Ok(Token::LParen(*lexeme)),
            ")" => Ok(Token::RParen(*lexeme)),
            "," => Ok(Token::Comma(*lexeme)),
            _ =>Err(format!("Unknown string {}", lexeme_name_with_range!(lexeme))),
        }
    }

}

/// Abstract Syntax Tree (AST) node representing a mathematical expression.
///
/// Each node corresponds to a part of an expression:
/// - numeric values,
/// - function arguments,
/// - unary/binary operators,
/// - function calls.
///
/// The AST allows for expression simplification and compilation into executable tokens.
#[derive(Debug, Clone, PartialEq)]
pub enum AstNode {
    /// Numeric literal.
    Number(Complex<f64>),

    /// Function argument by index.
    Argument(usize),

    /// Unary operator applied to an expression.
    UnaryOperator {
        kind: UnaryOperatorKind,
        expr: Box<AstNode>,
    },

    /// Binary operator applied to left and right expressions.
    BinaryOperator {
        kind: BinaryOperatorKind,
        left: Box<AstNode>,
        right: Box<AstNode>,
    },

    /// Differential operator
    Differentive {
        expr: Box<AstNode>,
        var: usize, // this is equal to the usize of AstNode::Argument
        order: usize,
    },

    /// Function call with evaluated argument expressions.
    FunctionCall {
        kind: FunctionKind,
        args: Vec<AstNode>,
    },

    /// User-defined Function call with evaluated argmuent expressions.
    UserFunctionCall {
        func: UserDefinedFunction,
        args: Vec<AstNode>,
    },
}

/// AstNode impl `from` and its helper impls
impl AstNode {
    /// Parses a slice of lexemes into an AST node.
    ///
    ///  Implements a full shunting-yard-like parser handling numbers, arguments, unary/binary operators,
    /// function calls, parentheses, and commas.
    ///
    /// # Parameters
    /// - `lexemes`: Slice of lexemes representing the expression.
    /// - `args`: List of argument names for functions.
    /// - `vars`: Table of variable values.
    /// - `users`: Table of user-defined functions.
    ///
    /// # Returns
    /// - `Ok(AstNode)` representing the root of the parsed AST.
    /// - `Err(String)` if parsing fails due to invalid syntax or unknown tokens.
    pub fn from<'a>(
        lexemes: &[Lexeme<'a>],
        args: &[&str],
        vars: &Variables,
        users: &UserDefinedTable,
    ) -> Result<Self, String> {
        let mut ast_nodes: Vec<Self> = Vec::new();
        let mut token_stack: Vec<Token> = Vec::new();
        // record whether the previous token is finished by value or not to evaluate the token is unary operator or binary operator.
        let mut prev_is_value = false;
        let lexemes = lexemes.iter().peekable();

        for lexeme in lexemes {
            let token = Token::from(lexeme, args, vars, users)?;
            match token {
                Token::Number(val) => {
                    ast_nodes.push(Self::Number(val));
                    prev_is_value = true;
                },
                Token::Argument(pos) => {
                    ast_nodes.push(Self::Argument(pos));
                    prev_is_value = true;
                },
                Token::Operator(lexeme) => {
                    match prev_is_value {
                        true => Self::parse_in_binary_operator(&mut ast_nodes, &mut token_stack, lexeme)?,
                        false => Self::parse_in_unary_operator(&mut token_stack, lexeme)?,
                    };
                    prev_is_value = false;
                },
                Token::DiffOperator(_) => {
                    token_stack.push(token);
                    prev_is_value = false;
                },
                Token::Function(_) | Token::UserFunction(_) => {
                    token_stack.push(token);
                    prev_is_value = false;
                },
                Token::LParen(_) => {
                    token_stack.push(token);
                    prev_is_value = false; // The operator next to LParen is unary operator; ex) cos(-x), 3 * (-2)
                },
                Token::RParen(_) => {
                    Self::parse_in_right_paren(&mut ast_nodes, &mut token_stack, lexeme)?;
                    prev_is_value = true; // The operator next to RParen is binary operator; ex) sin(x) + 2, (x+2)/(x-3)
                },
                Token::Comma(_) => {
                    Self::parse_in_comma(&mut ast_nodes, &mut token_stack, lexeme)?;
                    prev_is_value = false; // The operator next to Comma is unary operator; ex) pow(x, -3)
                },
                _ => return Err(format!("Invalid token kind made from {}", lexeme_name_with_range!(lexeme))),
            }
        }

        while let Some(token) = token_stack.pop() {
            match token {
                Token::UnaryOperator(oper) => Self::from_unary(&mut ast_nodes, oper)?,
                Token::BinaryOperator(oper) => Self::from_binary(&mut ast_nodes, oper)?,
                Token::DiffOperator(lexeme) => Self::from_diff(&mut ast_nodes, lexeme)?,
                Token::Function(func) => Self::from_function(&mut ast_nodes, func)?,
                Token::UserFunction(func) => Self::from_userfunction(&mut ast_nodes, func)?,
                _ => return Err("Unexpected token at the end".into()),
            }
        }

        let ret = ast_nodes.pop()
            .ok_or("Fail to parse to AST. There are NO AST node remaining.")?;

        if !ast_nodes.is_empty() {
            return Err("Fail to parse to AST. There are too AST node remaining.".into());
        }
        Ok(ret)
    }

    /// Parses tokens in a subexpression until a right parenthesis `)` is encountered.
    ///
    /// Pops tokens from `token_stack` and constructs AST nodes into `ast_nodes`.
    /// Handles unary operators, binary operators, and function calls.
    ///
    /// # Parameters
    /// - `ast_nodes`: Vector to accumulate AST nodes.
    /// - `token_stack`: Stack of tokens to process.
    /// - `lexeme`: Lexeme representing the right parenthesis.
    ///
    /// # Returns
    /// - `Ok(())` on success.
    /// - `Err(String)` if an unexpected token is found.
    fn parse_in_right_paren<'a>(
        ast_nodes: &mut Vec<Self>,
        token_stack: &mut Vec<Token<'a>>,
        lexeme: &Lexeme<'a>,
    ) -> Result<(), String> {
        while let Some(token) = token_stack.pop() {
            match token {
                Token::LParen(_) => break,
                Token::UnaryOperator(oper) => Self::from_unary(ast_nodes, oper)?,
                Token::BinaryOperator(oper) => Self::from_binary(ast_nodes, oper)?,
                Token::DiffOperator(lexeme) => Self::from_diff(ast_nodes, lexeme)?,
                Token::Function(func) => Self::from_function(ast_nodes, func)?,
                Token::UserFunction(func) => Self::from_userfunction(ast_nodes, func)?,
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

    /// Parses tokens in a subexpression until a comma `,` is encountered.
    ///
    /// Pops tokens from `token_stack` (without removing the left parenthesis) and constructs
    /// AST nodes into `ast_nodes`. Handles unary and binary operators.
    ///
    /// # Parameters
    /// - `ast_nodes`: Vector to accumulate AST nodes.
    /// - `token_stack`: Stack of tokens to process.
    /// - `lexeme`: Lexeme representing the comma.
    ///
    /// # Returns
    /// - `Ok(())` on success.
    /// - `Err(String)` if an unexpected token is found.
    fn parse_in_comma<'a>(
        ast_nodes: &mut Vec<Self>,
        token_stack: &mut Vec<Token<'a>>,
        lexeme: &Lexeme<'a>,
    ) -> Result<(), String> {
        // use Vec::last() to avoid removing Left Paren from the stack
        while let Some(token) = token_stack.pop() {
            match token {
                Token::LParen(_) => {
                    token_stack.push(token);
                    break;
                },
                Token::UnaryOperator(oper) => Self::from_unary(ast_nodes, oper)?,
                Token::BinaryOperator(oper) => Self::from_binary(ast_nodes, oper)?,
                Token::DiffOperator(lexeme) => Self::from_diff(ast_nodes, lexeme)?,
                Token::Function(func) => Self::from_function(ast_nodes, func)?,
                Token::UserFunction(func) => Self::from_userfunction(ast_nodes, func)?,
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

    /// Parses a unary operator token and pushes it onto the token stack.
    ///
    /// # Parameters
    /// - `token_stack`: Stack of tokens to process.
    /// - `lexeme`: Lexeme representing the unary operator.
    ///
    /// # Returns
    /// - `Ok(())` on success.
    /// - `Err(String)` if the lexeme does not represent a valid unary operator.
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

    /// Parses a binary operator token, resolves operator precedence, and pushes it onto the token stack.
    ///
    /// Implements the shunting-yard precedence rules for left- and right-associative operators.
    ///
    /// # Parameters
    /// - `ast_nodes`: Vector of AST nodes built so far.
    /// - `token_stack`: Stack of tokens to process.
    /// - `lexeme`: Lexeme representing the binary operator.
    ///
    /// # Returns
    /// - `Ok(())` on success.
    /// - `Err(String)` if the lexeme does not represent a valid binary operator.
    fn parse_in_binary_operator<'a>(
        ast_nodes: &mut Vec<Self>,
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
                Self::from_binary(ast_nodes, top_oper)?;
            }
            token_stack.push(Token::BinaryOperator(oper_kind));
            Ok(())
        } else {
            Err(format!("Unknown binary operator {}", lexeme_name_with_range!(lexeme)))
        }
    }

    /// Internal helper to create a unary operator AST node from a stack.
    ///
    /// This function pops the top operand from the stack and constructs a
    /// `UnaryOperator` AST node representing the specified unary operation.
    ///
    /// # Arguments
    ///
    /// * `stack` - A mutable reference to a stack of `AstNode`s. The operand for the unary
    ///   operator is expected at the top of the stack.
    /// * `oper` - The `UnaryOperatorKind` specifying which unary operation to create
    ///   (e.g., `Neg`, `Pos`, `Abs`).
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the `UnaryOperator` node was successfully created and pushed onto the stack.
    /// * `Err(String)` if the stack does not contain an operand for the unary operator.
    ///
    /// # Notes
    ///
    /// - This is an internal helper used during parsing to construct AST nodes for unary operations.
    fn from_unary(
        stack: &mut Vec<Self>,
        oper: UnaryOperatorKind,
    ) -> Result<(), String> {
        let expr = stack.pop()
            .ok_or(format!("Missing unary opeator {}", oper))?;
        stack.push(Self::UnaryOperator { kind: oper, expr: Box::new(expr) });
        Ok(())
    }

    /// Internal helper to create a binary operator AST node from a stack.
    ///
    /// This function pops the top two operands from the stack and constructs a
    /// `BinaryOperator` AST node representing the specified binary operation.
    ///
    /// # Arguments
    ///
    /// * `stack` - A mutable reference to a stack of `AstNode`s. The right-hand operand
    ///   is expected at the top of the stack, followed by the left-hand operand.
    /// * `oper` - The `BinaryOperatorKind` specifying which binary operation to create
    ///   (e.g., `Add`, `Sub`, `Mul`, `Div`, `Pow`).
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the `BinaryOperator` node was successfully created and pushed onto the stack.
    /// * `Err(String)` if the stack does not contain enough operands for the binary operator.
    ///
    /// # Notes
    ///
    /// - This is an internal helper used during parsing to construct AST nodes for binary operations.
    fn from_binary(
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

    /// Internal helper to create a function call AST node from a stack.
    ///
    /// This function pops the required number of arguments from the stack and constructs a
    /// `FunctionCall` AST node representing a call to the specified built-in function.
    ///
    /// # Arguments
    ///
    /// * `stack` - A mutable reference to a stack of `AstNode`s. The function arguments are
    ///   popped from this stack in reverse order (last argument first).
    /// * `func` - The `FunctionKind` representing the built-in function to call.
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the `FunctionCall` node was successfully created and pushed onto the stack.
    /// * `Err(String)` if the stack does not contain enough arguments for the function.
    ///
    /// # Notes
    ///
    /// - The function assumes that the arguments on the stack are correctly ordered according to
    ///   the parser's rules (last argument at the top of the stack).
    /// - This is an internal helper used during parsing to construct AST nodes for built-in
    ///   function calls.
    fn from_function(
        stack: &mut Vec<Self>,
        func: FunctionKind,
    ) -> Result<(), String> {
        let n = func.arity();
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

    /// Internal helper to create a user-defined function call AST node from a stack.
    ///
    /// This function pops the required number of arguments from the stack and constructs a
    /// `UserFunctionCall` AST node representing a call to the specified user-defined function.
    ///
    /// # Arguments
    ///
    /// * `stack` - A mutable reference to a stack of `AstNode`s. The function arguments are
    ///   popped from this stack in reverse order (last argument first).
    /// * `func` - The `UserDefinedFunction` to call, including its name and arity.
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the `UserFunctionCall` node was successfully created and pushed onto the stack.
    /// * `Err(String)` if the stack does not contain enough arguments for the function.
    ///
    /// # Notes
    ///
    /// - The function assumes that the arguments on the stack are correctly ordered according to
    ///   the parser's rules (last argument at the top of the stack).
    /// - This is an internal helper used during parsing to construct AST nodes for user-defined
    ///   function calls.
    fn from_userfunction(
        stack: &mut Vec<Self>,
        func: UserDefinedFunction,
    ) -> Result<(), String> {
        let n = func.arity();
        let mut args = Vec::new();
        args.resize(n, Self::Argument(0)); // Dummy Self for initializing
        for i in (0..n).rev() {
            let arg = stack.pop()
                .ok_or(format!("Missing function argument for {}", func.name()))?;
            args[i] = arg;
        }
        stack.push(Self::UserFunctionCall { func, args });
        Ok(())
    }

    /// Internal helper to create a differential operator AST node from a stack.
    ///
    /// This function pops the necessary operands from the stack to construct a `Differentive`
    /// AST node, representing the derivative of an expression. It supports both:
    /// - `diff(f(x), x)` for first-order derivatives, and
    /// - `diff(f(x), x, n)` for higher-order derivatives (integer `n` only).
    ///
    /// # Arguments
    ///
    /// * `stack` - A mutable reference to a stack of `AstNode`s. Operands are popped from this
    ///   stack to build the differential node.
    /// * `lexeme` - The lexeme corresponding to the differential operator, used for error messages.
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the differential node was successfully created and pushed onto the stack.
    /// * `Err(String)` if the stack does not contain sufficient operands, the differential
    ///   variable is invalid, or the order of differentiation is invalid (non-integer or negative).
    ///
    /// # Notes
    ///
    /// - Fractional calculus (non-integer order) is not supported.
    /// - The order of differentiation must be between 0 and `i8::MAX`.
    /// - This function assumes that the expression and variable nodes are correctly ordered
    ///   on the stack according to the parser's rules.
    fn from_diff(
        stack: &mut Vec<Self>,
        lexeme: Lexeme,
    ) -> Result<(), String> {
        let top = stack.pop()
            .ok_or(format!("Missing arguments for {}", lexeme_name_with_range!(lexeme)))?;

        if let Self::Number(z) = top {
            // the case of `diff(f(x), x, n)`
            if (z.im != 0.0) || (z.re.fract() != 0.0) {
                return Err(format!("Not supported fractional calculus, order {}, in {}", z, lexeme_name_with_range!(lexeme)));
            }
            let x = z.re();
            if (x < 0.0) || ((i8::MAX as f64) < x) {
                return Err(format!("Invalid differential order {} for {}", x, lexeme_name_with_range!(lexeme)));
            }
            let order = x as usize;
            let var = if let Self::Argument(idx) = stack.pop()
                .ok_or(format!("Missing differential variable for {}", lexeme_name_with_range!(lexeme)))?
            {
                idx
            } else {
                return Err(format!("Invalid differential variable for {}", lexeme_name_with_range!(lexeme)));
            };
            let expr = stack.pop()
                .ok_or(format!("Missing expr for differential operator {}", lexeme_name_with_range!(lexeme)))?;
            stack.push(Self::Differentive { expr: Box::new(expr), var, order });
        } else {
            // the case of `diff(f(x), x)`, which means `n = 1`
            let var = if let Self::Argument(idx) = stack.pop()
                .ok_or(format!("Missing differential variable for {}", lexeme_name_with_range!(lexeme)))?
            {
                idx
            } else {
                return Err(format!("Invalid differential variable for {}", lexeme_name_with_range!(lexeme)));
            };
            let expr = stack.pop()
                .ok_or(format!("Missing expr for differential operator {}", lexeme_name_with_range!(lexeme)))?;
            stack.push(Self::Differentive { expr: Box::new(expr), var, order: 1 });
        };
        Ok(())
    }
}

/// AstNode impl `simplify` and its helper impls
impl AstNode {
    /// Simplifies the AST by evaluating constant sub-expressions.
    ///
    /// Returns a new `AstNode` where all numeric computations that can be resolved
    /// at compile time are folded into `Number` nodes.
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
                Self::fold_binary(kind, left, right)
            },
            Self::FunctionCall { kind, args }
                => Self::fold_function(kind, args, |k, a| AstNode::FunctionCall { kind: k, args: a }),
            Self::UserFunctionCall { func, args }
                => Self::fold_function(func, args, |f, a| AstNode::UserFunctionCall { func: f, args: a }),
            _ => self,
        }
    }

    /// Internal helper to fold a function with arguments AST nodes if possible.
    fn fold_function<T, F>(func: T, args: Vec<Self>, make_node: F) -> Self
    where
        T: FunctionCall,
        F: Fn(T, Vec<Self>) -> Self,
    {
        let args: Vec<_> = args.into_iter().map(|a| a.simplify()).collect();
        if args.iter().all(|a| matches!(a, AstNode::Number(_))) {
            let nums: Vec<Complex<f64>> = args.iter().map(|a| {
                if let AstNode::Number(v) = a { *v } else { unreachable!() }
            }).collect();
            AstNode::Number(func.apply(&nums))
        } else {
            make_node(func, args)
        }
    }


    /// Internal helper to fold a binary operator with two AST nodes if possible.
    ///
    /// Performs constant folding when both operands are numbers. Also applies simple
    /// arithmetic optimizations for associative operators like `+` and `*`.
    ///
    /// # Parameters
    /// - `kind`: The binary operator kind.
    /// - `left`: Left-hand AST node.
    /// - `right`: Right-hand AST node.
    ///
    /// # Returns
    /// - `AstNode`: Simplified AST node after folding if applicable.
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
}

/// AstNode helper impl to create new AstNode
impl AstNode {
    /// Create a number 0.0 Ast node.
    fn zero() -> Self {
        AstNode::Number(Complex::ZERO)
    }

    /// Create a number 1.0 Ast node.
    fn one() -> Self {
        AstNode::Number(Complex::ONE)
    }

    /// Internal helper to create an additional operator AST node `self + other`.
    fn add(&self, other: &Self) -> Self {
        Self::BinaryOperator {
            kind: BinaryOperatorKind::Add,
            left: Box::new(self.clone()),
            right: Box::new(other.clone()),
        }
    }

    /// internal helper to create a subtracted operator AST node `self - other`.
    fn sub(&self, other: &Self) -> Self {
        Self::BinaryOperator {
            kind: BinaryOperatorKind::Sub,
            left: Box::new(self.clone()),
            right: Box::new(other.clone()),
        }
    }

    /// Internal helper to create a multiplied operator AST node `self * other`.
    fn mul(&self, other: &Self) -> Self {
        Self::BinaryOperator {
            kind: BinaryOperatorKind::Mul,
            left: Box::new(self.clone()),
            right: Box::new(other.clone()),
        }
    }

    /// Internal helper to create a divided operator AST node `self / right`.
    fn div(&self, right: &Self) -> Self {
        Self::BinaryOperator {
            kind: BinaryOperatorKind::Div,
            left: Box::new(self.clone()),
            right: Box::new(right.clone()),
        }
    }

    /// Internal helper to create a negative AST node `-self`.
    fn negative(&self) -> Self {
        Self::UnaryOperator {
            kind: UnaryOperatorKind::Negative,
            expr: Box::new(self.clone()),
        }
    }

    /// Internal helper to create a sin AST node `sin(x)`.
    fn sin(&self) -> Self {
        AstNode::FunctionCall { kind: FunctionKind::Sin, args: vec![self.clone()] }
    }

    /// Internal helper to create a cos AST node `cos(x)`.
    fn cos(&self) -> Self {
        AstNode::FunctionCall { kind: FunctionKind::Cos, args: vec![self.clone()] }
    }

    /// Internal helper to create a sinh AST node `sinh(x)`.
    fn sinh(&self) -> Self {
        AstNode::FunctionCall { kind: FunctionKind::Sinh, args: vec![self.clone()] }
    }

    /// Internal helper to create a cosh AST node `cosh(x)`.
    fn cosh(&self) -> Self {
        AstNode::FunctionCall { kind: FunctionKind::Cosh, args: vec![self.clone()] }
    }

    /// Internal helper to create an exp AST node `exp(x)`.
    fn exp(&self) -> Self {
        AstNode::FunctionCall { kind: FunctionKind::Exp, args: vec![self.clone()] }
    }

    /// Internal helper to create a sqrt AST node `sqrt(x)`.
    fn sqrt(&self) -> Self {
        AstNode::FunctionCall { kind: FunctionKind::Sqrt, args: vec![self.clone()] }
    }

    /// Internal helper to create an abs AST node `abs(x)`.
    fn abs(&self) -> Self {
        AstNode::FunctionCall { kind: FunctionKind::Abs, args: vec![self.clone()] }
    }

    /// Internal helper to create `pow(self, expr)` AST node.
    fn pow(&self, expr: &Self) -> Self {
        Self::FunctionCall {
            kind: FunctionKind::Pow,
            args: vec![self.clone(), expr.clone()],
        }
    }

    /// Internal helper to create `powi(self, expr)` AST node.
    fn powi(&self, expr: i32) -> Self {
        Self::FunctionCall {
            kind: FunctionKind::Powi,
            args: vec![self.clone(), AstNode::Number(Complex::from(expr as f64))],
        }
    }
}

/// AstNode impl `differentiate` and its helpers
impl AstNode {
    /// Compute the derivative of an AST node with respect to a given variable.
    ///
    /// This function recursively differentiates an `AstNode` representing a mathematical
    /// expression. It supports:
    /// - Numbers and arguments (variables),
    /// - Unary and binary operators,
    /// - Built-in functions,
    /// - User-defined functions,
    /// - Nested differential operators (`Differentive`) to handle higher-order derivatives.
    ///
    /// # Arguments
    ///
    /// * `var` - The index of the variable with respect to which the derivative is taken.
    ///
    /// # Returns
    ///
    /// A new `AstNode` representing the derivative of `self` with respect to the specified variable.
    ///
    /// # Panics
    ///
    /// - If a user-defined function does not have a derivative defined for the specified variable.
    ///
    /// # Notes
    ///
    /// - `BinaryOperator` nodes delegate to `diff_binary`.
    /// - `FunctionCall` nodes delegate to `diff_function`.
    /// - `UserFunctionCall` nodes require the user-defined function to provide a derivative.
    /// - Nested `Differentive` nodes increment the order if differentiating with respect to the same variable.
    pub fn differentiate(&self, var: usize) -> Result<Self, String> {
        match self {
            Self::Number(_) => Ok(Self::zero()),
            Self::Argument(idx)
                => if *idx == var {
                    Ok(Self::one())
                } else {
                    Ok(Self::zero())
                },
            Self::UnaryOperator { kind, expr }
                => Ok(Self::UnaryOperator { kind: *kind, expr: Box::new(expr.differentiate(var)?) }),
            Self::BinaryOperator { kind, left, right } => {
                Self::diff_binary(*kind, *left.clone(), *right.clone(), var)
            },
            Self::FunctionCall { kind, args } => {
                Self::diff_function(*kind, args, var)
            },
            Self::UserFunctionCall { func, args } => {
                match func.derivative(var) {
                    Some(deriv) => Ok(AstNode::UserFunctionCall { func: deriv, args: args.clone() }),
                    None => Err(format!("The deriv of {} for var[{}] is undefined", func.name(), var)),
                }
            },
            Self::Differentive { expr, var: inner_var, order } => {
                if *inner_var == var {
                    // d/dx (d/dx f(x)) = d^2/dx^2 f(x)
                    Ok(AstNode::Differentive { expr: expr.clone(), var, order: order + 1 })
                } else {
                    Ok(AstNode::Differentive {
                        expr: Box::new(expr.differentiate(var)?),
                        var: *inner_var,
                        order: *order,
                    })
                }
            }
        }
    }

    /// Differentiate a power expression with respect to a variable.
    ///
    /// Computes the derivative of the expression:
    ///
    /// ```text
    /// d/dx [ u(x) ^ v(x) ] = u(x) ^ v(x) * ( v'(x) * ln(u(x)) + v(x) * u'(x) / u(x) )
    /// ```
    ///
    /// This rule applies to the general case where both the base `u(x)` and the exponent
    /// `v(x)` are functions of `x`.
    ///
    /// # Arguments
    /// * `left` - The base expression `u(x)`.
    /// * `right` - The exponent expression `v(x)`.
    /// * `var` - The index of the variable with respect to which differentiation is performed.
    ///
    /// # Errors
    /// Returns an error if differentiation of the subexpressions fails.
    fn diff_pow(left: &Self, right: &Self, var: usize) -> Result<Self, String> {
        // d/dx [u(x) ^ v(x)] = u^v * (v' * ln(u) + v * u' / u)
        let u = left.clone();
        let v = right.clone();
        let du = u.differentiate(var)?;
        let dv = v.differentiate(var)?;
        let ln_u = Self::FunctionCall {
            kind: FunctionKind::Ln,
            args: vec![u.clone()],
        };
        Ok(u.pow(&v).mul(&(dv.mul(&ln_u)).add(&v.mul(&du).div(&u))))
    }

    /// Differentiate an integer power expression with respect to a variable.
    ///
    /// Computes the derivative of the expression:
    ///
    /// ```text
    /// d/dx [ u(x) ^ n ] = n * u(x) ^ (n - 1) * u'(x)
    /// ```
    ///
    /// where `n` is an integer constant (i.e., `powi` form).
    ///
    /// # Arguments
    /// * `left` - The base expression `u(x)`.
    /// * `right` - The integer exponent `n`.
    /// * `var` - The index of the variable with respect to which differentiation is performed.
    ///
    /// # Errors
    /// Returns an error if differentiation of the base fails.
    fn diff_powi(left: &Self, right: &Self, var: usize) -> Result<Self, String> {
        // d/dx [u(x) ^ n] = n * u(x) ^ (n-1) * u'(x)
        let u = left.clone();
        let n = right.clone();
        let du = u.differentiate(var)?;
        Ok(AstNode::FunctionCall {
            kind: FunctionKind::Powi,
            args: vec![u, n.sub(&Self::one())],
        }.mul(&n).mul(&du))
    }

    /// Differentiate a binary operator expression with respect to a variable.
    ///
    /// Supports the following binary operators:
    ///
    /// - **Addition** and **Subtraction**:
    ///   ```text
    ///   d/dx [ u ± v ] = u' ± v'
    ///   ```
    /// - **Multiplication**:
    ///   ```text
    ///   d/dx [ u * v ] = u' * v + u * v'
    ///   ```
    /// - **Division**:
    ///   ```text
    ///   d/dx [ u / v ] = (u' * v - u * v') / v^2
    ///   ```
    /// - **Power**:
    ///   Falls back to [`diff_pow`] for the general differentiation rule.
    ///
    /// # Arguments
    /// * `kind` - The binary operator kind.
    /// * `left` - The left-hand side expression `u(x)`.
    /// * `right` - The right-hand side expression `v(x)`.
    /// * `var` - The index of the variable with respect to which differentiation is performed.
    ///
    /// # Errors
    /// Returns an error if differentiation of the subexpressions fails.
    fn diff_binary(kind: BinaryOperatorKind, left: Self, right: Self, var:usize) -> Result<Self, String> {
        let dl = left.differentiate(var)?;
        let dr = right.differentiate(var)?;
        match kind {
            BinaryOperatorKind::Add | BinaryOperatorKind::Sub
                => Ok(Self::BinaryOperator { kind, left: Box::new(dl), right: Box::new(dr) }),
            BinaryOperatorKind::Mul
                => Ok(dl.mul(&right).add(&left.mul(&dr))),
            BinaryOperatorKind::Div
                => Ok(dl.mul(&right).sub(&left.mul(&dr)).div(&right.powi(2))),
            BinaryOperatorKind::Pow
                => Self::diff_pow(&left, &right, var),
        }
    }

    /// Differentiate a function call with respect to a variable.
    ///
    /// Implements the standard differentiation rules for elementary functions:
    ///
    /// - `sin(x)` → `cos(x) * x'`
    /// - `cos(x)` → `-sin(x) * x'`
    /// - `tan(x)` → `x' / cos(x)^2`
    /// - `asin(x)` → `x' / sqrt(1 - x^2)`
    /// - `acos(x)` → `-x' / sqrt(1 - x^2)`
    /// - `atan(x)` → `x' / (1 + x^2)`
    /// - `sinh(x)` → `cosh(x) * x'`
    /// - `cosh(x)` → `sinh(x) * x'`
    /// - `tanh(x)` → `x' / cosh(x)^2`
    /// - `asinh(x)` → `x' / sqrt(x^2 + 1)`
    /// - `acosh(x)` → `x' / sqrt(x^2 - 1)`
    /// - `atanh(x)` → `x' / (1 - x^2)`
    /// - `exp(x)` → `exp(x) * x'`
    /// - `ln(x)` → `x' / x`
    /// - `log10(x)` → `x' / (x * ln(10))`
    /// - `sqrt(x)` → `x' / (2 * sqrt(x))`
    /// - `abs(x)` → `(x / |x|) * x'` (undefined at `x = 0`)
    ///
    /// Special cases:
    /// - `conj(z)` → Not differentiable in the complex domain; returns an error.
    /// - `pow(u, v)` → Delegates to [`diff_pow`].
    /// - `powi(u, n)` → Delegates to [`diff_powi`].
    ///
    /// # Arguments
    /// * `kind` - The function kind.
    /// * `args` - The argument expressions.
    /// * `var` - The index of the variable with respect to which differentiation is performed.
    ///
    /// # Errors
    /// Returns an error if:
    /// - The function is not differentiable (`conj`).
    /// - Differentiation of the argument fails.
    fn diff_function(kind: FunctionKind, args: &[Self], var: usize) -> Result<Self, String> {
        let x = &args[0];
        let dx = args[0].differentiate(var)?;
        match kind {
            FunctionKind::Sin   => Ok(x.cos().mul(&dx)),
            FunctionKind::Cos   => Ok(x.sin().negative().mul(&dx)),
            FunctionKind::Tan   => Ok(dx.div(&x.cos().powi(2))),
            FunctionKind::Asin  => Ok(dx.div(&(Self::one().sub(&x.powi(2))))),
            FunctionKind::Acos  => Ok(dx.negative().div(&Self::one().sub(&x.powi(2)))),
            FunctionKind::Atan  => Ok(dx.div(&Self::one().add(&x.powi(2)))),
            FunctionKind::Sinh  => Ok(dx.mul(&x.cosh())),
            FunctionKind::Cosh  => Ok(dx.mul(&x.sinh())),
            FunctionKind::Tanh  => Ok(dx.div(&x.cosh().powi(2))),
            FunctionKind::Asinh => Ok(dx.div(&x.powi(2).add(&Self::one()).sqrt())),
            FunctionKind::Acosh => Ok(dx.div(&x.powi(2).sub(&Self::one()).sqrt())),
            FunctionKind::Atanh => Ok(dx.div(&Self::one().sub(&x.powi(2)))),
            FunctionKind::Exp   => Ok(dx.mul(&x.exp())),
            FunctionKind::Ln    => Ok(dx.div(x)),
            FunctionKind::Log10 => Ok(dx.mul(&AstNode::Number(Complex::from(std::f64::consts::LOG10_E))).div(x)),
            FunctionKind::Sqrt  => Ok(dx.mul(&AstNode::Number(Complex::from(0.5))).div(&x.sqrt())),
            // TODO: d/dx |f(x)| should return NaN if f(x) = 0, but we have NOT been able to express yet.
            FunctionKind::Abs   => Ok(x.div(&x.abs()).mul(&dx)),
            FunctionKind::Conj  => Err("The function `conj(z)` doesn't have any derivative functions at any coordinates.".into()),
            FunctionKind::Pow   => Self::diff_pow(&args[0], &args[1], var),
            FunctionKind::Powi  => Self::diff_powi(&args[0], &args[1], var),
        }
    }
}

/// AstNode impl `compile` and its helper impls
impl AstNode {
    /// Internal helper to compile the AST into a sequence of executable tokens.
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
            Self::UserFunctionCall { func, args } => {
                for arg in args {
                    Self::execute(arg, tokens);
                }
                tokens.push(Token::UserFunction(func.clone()));
            },
            Self::Differentive {..} => unreachable!("AstNode should not include Differentive"),
        }
    }

    /// Compiles the AST into a vector of `Token`s for evaluation.
    ///
    /// The resulting tokens can be used with a function list generated
    /// by `make_function_list` to evaluate the expression.
    pub fn compile<'a>(&self) -> Vec<Token<'a>> {
        let mut tokens: Vec<Token<'a>> = Vec::new();
        self.execute(&mut tokens);
        tokens
    }
}

#[cfg(test)]
mod unary_operator_kind_tests {
    use super::*;
    #[test]
    fn test_unary_operator_kind_from() {
        assert_eq!(UnaryOperatorKind::from("+"), Some(UnaryOperatorKind::Positive));
        assert_eq!(UnaryOperatorKind::from("-"), Some(UnaryOperatorKind::Negative));
        assert_eq!(UnaryOperatorKind::from("*"), None);
        assert_eq!(UnaryOperatorKind::from(""), None);
        assert_eq!(UnaryOperatorKind::from("x"), None);
    }
}

#[cfg(test)]
mod binary_operator_kind_tests {
    use super::*;

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
}

#[cfg(test)]
mod function_kind_tests {
    use super::*;

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
    fn test_function_kind_arity() {
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
            assert_eq!(func.arity(), 1, "{:?} should have 1 argument", func);
        }

        let double_args_functions = [
            FunctionKind::Pow,
        ];

        for func in &double_args_functions {
            assert_eq!(func.arity(), 2, "{:?} should have 2 argument", func);
        }
    }
}

#[cfg(test)]
mod token_tests {
    use super::*;

    #[test]
    fn test_number_token() {
        let lex = Lexeme::new("3.14", 0..4);
        let vars = Variables::new();
        let users = UserDefinedTable::new();
        let args: [&str; 0] = [];
        let token = Token::from(&lex, &args, &vars, &users).unwrap();
        match token {
            Token::Number(val) => assert_eq!(val, Complex::new(3.14, 0.0)),
            _ => panic!("Expected Number token"),
        }
    }

    #[test]
    fn test_imaginary_number() {
        let lex = Lexeme::new("2i", 0..2);
        let vars = Variables::new();
        let users = UserDefinedTable::new();
        let args: [&str; 0] = [];
        let token = Token::from(&lex, &args, &vars, &users).unwrap();
        match token {
            Token::Number(val) => assert_eq!(val, Complex::new(0.0, 2.0)),
            _ => panic!("Expected Number token"),
        }
    }

    #[test]
    fn test_constant_token() {
        let lex = Lexeme::new("PI", 0..2);
        let vars = Variables::new();
        let users = UserDefinedTable::new();
        let args: [&str; 0] = [];
        let token = Token::from(&lex, &args, &vars, &users).unwrap();
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
        let users = UserDefinedTable::new();
        let token = Token::from(&lex, &args, &vars, &users).unwrap();
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
        let users = UserDefinedTable::new();
        let token = Token::from(&lex, &args, &vars, &users).unwrap();
        match token {
            Token::Operator(_) => {}, // OK
            _ => panic!("Expected Operator token"),
        }
    }

    #[test]
    fn test_function_token() {
        let lex = Lexeme::new("sin", 0..3);
        let vars = Variables::new();
        let users = UserDefinedTable::new();
        let args: [&str; 0] = [];
        let token = Token::from(&lex, &args, &vars, &users).unwrap();
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
        let users = UserDefinedTable::new();
        let args: [&str; 0] = [];

        assert!(matches!(Token::from(&lex_l, &args, &vars, &users).unwrap(), Token::LParen(_)));
        assert!(matches!(Token::from(&lex_r, &args, &vars, &users).unwrap(), Token::RParen(_)));
        assert!(matches!(Token::from(&lex_c, &args, &vars, &users).unwrap(), Token::Comma(_)));
    }

    #[test]
    fn test_unknown_string() {
        let lex = Lexeme::new("unknown", 0..7);
        let vars = Variables::new();
        let users = UserDefinedTable::new();
        let args: [&str; 0] = [];
        let res = Token::from(&lex, &args, &vars, &users);
        assert!(res.is_err());
    }
}

#[cfg(test)]
mod astnode_tests {
    use super::*;
    use crate::{lexer, variable::UserDefinedFunction};
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
    fn test_single_number_astnode() {
        let lexemes = lexer::from("42");
        let ast = AstNode::from(&lexemes, &[], &Variables::new(), &UserDefinedTable::new()).unwrap();
        match ast {
            AstNode::Number(val) => assert_eq!(val, Complex::new(42.0, 0.0)),
            _ => panic!("Expected Number AST node"),
        }
    }

    #[test]
    fn test_unary_operator_negative_astnode() {
        let lexemes = lexer::from("- 3");
        let ast = AstNode::from(&lexemes, &[], &Variables::new(), &UserDefinedTable::new()).unwrap();
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
        let ast = AstNode::from(&lexemes, &[], &Variables::new(), &UserDefinedTable::new()).unwrap();
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
        let ast = AstNode::from(&lexemes, &[], &Variables::new(), &UserDefinedTable::new()).unwrap();
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
        let ast = AstNode::from(&lexemes, &[], &Variables::new(), &UserDefinedTable::new()).unwrap();
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
        let ast = AstNode::from(&lexemes, &[], &Variables::new(), &UserDefinedTable::new()).unwrap();
        match ast {
            AstNode::FunctionCall { kind, args } => {
                assert_eq!(kind, FunctionKind::Pow);
                assert_eq!(args.len(), 2);
                assert_eq!(args[0], AstNode::Number(Complex::new(2.0, 0.0)));
                assert_eq!(args[1], AstNode::Number(Complex::new(3.0, 0.0)));
            }
            _ => panic!("Expected Function node"),
        }

        let lexemes = lexer::from("pow ( sin(x) , 3 )");
        let ast = AstNode::from(&lexemes, &["x"], &Variables::new(), &UserDefinedTable::new()).unwrap();
        match ast {
            AstNode::FunctionCall { kind, args } => {
                assert_eq!(kind, FunctionKind::Pow);
                assert_eq!(args.len(), 2);
                assert_eq!(args[0], AstNode::FunctionCall {
                    kind: FunctionKind::Sin,
                    args: vec![AstNode::Argument(0)]
                });
                assert_eq!(args[1], AstNode::Number(Complex::new(3.0, 0.0)));
            }
            _ => panic!("Expected Function node"),
        }
    }

    #[test]
    fn test_imaginary_number_astnode() {
        let lexemes = lexer::from("5i");
        let ast = AstNode::from(&lexemes, &[], &Variables::new(), &UserDefinedTable::new()).unwrap();
        assert_eq!(ast, AstNode::Number(Complex::new(0.0, 5.0)));
    }

    #[test]
    fn test_unknown_token_astnode_error() {
        let lexemes = lexer::from("@");
        let res = AstNode::from(&lexemes, &[], &Variables::new(), &UserDefinedTable::new());
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


    // Dummy user-defined function
    fn sum_func(args: &[Complex<f64>]) -> Complex<f64> {
        args.iter().copied().sum()
    }

    #[test]
    fn test_simplify_user_function_call_with_numbers() {
        let func = UserDefinedFunction::new(
            "sum",
            sum_func,
            2,
        );

        let node = AstNode::UserFunctionCall {
            func,
            args: vec![
                AstNode::Number(Complex::from(1.0)),
                AstNode::Number(Complex::from(2.0)),
            ],
        }.simplify();

        match node {
            AstNode::Number(val) => assert_abs_diff_eq!(val.re, 3.0, epsilon=1e-12),
            _ => panic!("Expected simplified to Number"),
        }
    }

    #[test]
    fn test_simplify_user_function_call_with_no_numbers() {
        let func = UserDefinedFunction::new(
            "sum",
            sum_func,
            2,
        );

        let node = AstNode::UserFunctionCall {
            func,
            args: vec![
                AstNode::Number(Complex::ONE),
                AstNode::Argument(0),
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
        let node = AstNode::Argument(1);
        let diff = node.differentiate(1).unwrap();
        assert_eq!(diff, AstNode::Number(Complex::ONE));
        let diff_other = node.differentiate(0).unwrap();
        assert_eq!(diff_other, AstNode::Number(Complex::ZERO));
    }

    #[test]
    fn test_differentiate_unary_operator() {
        let node = AstNode::UnaryOperator {
            kind: UnaryOperatorKind::Negative,
            expr: Box::new(AstNode::Argument(0)),
        };
        let diff = node.differentiate(0).unwrap();
        assert_eq!(
            diff,
            AstNode::UnaryOperator {
                kind: UnaryOperatorKind::Negative,
                expr: Box::new(AstNode::Number(Complex::ONE)),
            }
        );
    }

    #[test]
    fn test_differentiate_binary_add() {
        let node = AstNode::BinaryOperator {
            kind: BinaryOperatorKind::Add,
            left: Box::new(AstNode::Argument(0)),
            right: Box::new(AstNode::Number(Complex::new(2.0, 0.0))),
        };
        let diff = node.differentiate(0).unwrap();
        // d/dx (x + 2) = 1 + 0
        assert_eq!(
            diff,
            AstNode::BinaryOperator {
                kind: BinaryOperatorKind::Add,
                left: Box::new(AstNode::Number(Complex::ONE)),
                right: Box::new(AstNode::Number(Complex::ZERO)),
            }
        );
    }

    #[test]
    fn test_differentiate_function_sin() {
        let node = AstNode::FunctionCall {
            kind: FunctionKind::Sin,
            args: vec![AstNode::Argument(0)],
        };
        let diff = node.differentiate(0).unwrap();
        // d/dx sin(x) = cos(x) * 1
        assert_eq!(
            diff,
            AstNode::BinaryOperator {
                kind: BinaryOperatorKind::Mul,
                left: Box::new(AstNode::FunctionCall {
                    kind: FunctionKind::Cos,
                    args: vec![AstNode::Argument(0)],
                }),
                right: Box::new(AstNode::Number(Complex::ONE)),
            }
        );
    }

    #[test]
    fn test_differentiate_differentive_order() {
        let node = AstNode::Differentive {
            expr: Box::new(AstNode::Argument(0)),
            var: 0,
            order: 1,
        };
        let diff = node.differentiate(0).unwrap();
        // d/dx (d/dx x) = d^2/dx^2 x
        assert_eq!(
            diff,
            AstNode::Differentive {
                expr: Box::new(AstNode::Argument(0)),
                var: 0,
                order: 2,
            }
        );
    }
}
