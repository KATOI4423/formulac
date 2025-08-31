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
use crate::variable::{Variables, UserDefinedFunction, UserDefinedTable};
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

            /// Returns arity, the number of arguments that the function takes.
            pub fn arity(&self) -> usize {
                match self {
                    $( Self::$variant => $arity, )*
                }
            }

            /// Applies the function to a slice of complex numbers and returns the result.
            ///
            /// The number of elements in `args` must match the value returned by `arity`.
            pub fn apply(&self, args: &[Complex<f64>]) -> Complex<f64> {
                match self {
                    $( Self::$variant => {
                        let $a = args;
                        $body
                    }, )*
                }
            }

            /// Returns a list of all supported function names.
            pub fn names() -> Vec<&'static str> {
                vec![$($name),*]
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
    Powi    => { name: "powi",      arity: 2,   apply: |a| powi(a[0], a[1]) },
}

fn powi(base: Complex<f64>, exp: Complex<f64>) -> Complex<f64> {
    let exp = exp.re();
    if exp.is_finite() && (i32::MIN as f64 <= exp) && (exp <= i32::MAX as f64) {
        base.powi(exp as i32)
    } else {
        Complex::new(f64::NAN, f64::NAN)
    }
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

    /// Standard mathematical function token (e.g., `sin`, `cos`, `exp`).
    Function(FunctionKind),

    /// User-defined function token.
    UserFunction(UserDefinedFunction<'a>),

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
        s.parse::<f64>().ok().map(|value| Complex::from(value))
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

        if let Some(user_func) = users.get(text) {
            return Ok(Token::UserFunction(user_func.clone()));
        }

        match text {
            "(" => Ok(Token::LParen(lexeme.clone())),
            ")" => Ok(Token::RParen(lexeme.clone())),
            "," => Ok(Token::Comma(lexeme.clone())),
            _ =>Err(format!("Unknown string {}", lexeme_name_with_range!(lexeme))),
        }
    }

}

/// Converts a list of parsed tokens into a list of executable functions.
///
/// This function transforms each `Token` into a boxed closure (`Fn`) that
/// operates on a temporary evaluation stack and an argument list. The resulting
/// closures implement the actual computation of numbers, operators, and functions
/// during expression evaluation.
///
/// # Parameters
/// - `tokens`: A vector of `Token`s produced by the parser, representing
///   a mathematical expression in a format suitable for evaluation.
///
/// # Returns
/// A vector of boxed closures (`Box<dyn Fn(...)>`) where each closure:
/// - Takes a mutable stack of `Complex<f64>` values representing intermediate results.
/// - Takes a slice of `Complex<f64>` representing function arguments.
/// - Performs the computation corresponding to the token and pushes the result
///   onto the stack.
///
/// # Behavior
/// - `Token::Number`: pushes its numeric value onto the stack.
/// - `Token::Argument`: pushes the corresponding argument from the provided list.
/// - `Token::UnaryOperator`: pops one value, applies the operator, and pushes the result.
/// - `Token::BinaryOperator`: pops two values, applies the operator, and pushes the result.
/// - `Token::Function` / `Token::UserFunction`: pops the required number of arguments,
///   applies the function, and pushes the result.
///
/// # Panics
/// - This function assumes that tokens are valid and compiled (i.e., no unprocessed
///   parentheses or commas). If invalid tokens are encountered, the closure will panic.
pub fn make_function_list<'a>(
    tokens: Vec<Token<'a>>,
) -> Vec<Box<dyn Fn(
    &mut Vec<Complex<f64>>, // tempolary stack of value in token
    &[Complex<f64>] // arguments list
) + 'a>>
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
                        let n = func.arity();
                        let mut args: Vec<Complex<f64>> = Vec::with_capacity(n);
                        args.resize(n, Complex::new(0.0, 0.0));

                        for i in (0..n).rev() {
                            args[i] = stack.pop().unwrap();
                        }
                        stack.push(func.apply(&args));
                    }
                ));
            },
            Token::UserFunction(func) => {
                func_list.push(Box::new(
                    move |stack, _args| {
                        let n = func.arity();
                        let mut args: Vec<Complex<f64>> = Vec::with_capacity(n);
                        args.resize(n, Complex::ZERO);

                        for i in (0..n).rev() {
                            args[i] = stack.pop().unwrap();
                        }
                        stack.push(func.apply(&args));
                    }
                ))
            }
            _ => unreachable!("Invalid tokens found: use compiled tokens"),
        }
    }

    func_list
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

    /// Function call with evaluated argument expressions.
    FunctionCall {
        kind: FunctionKind,
        args: Vec<AstNode>,
    },
}

impl AstNode {
    /// Parses a slice of lexemes into an AST node.
    ///
    /// # Parameters
    /// - `lexemes`: Slice of lexemes representing the expression.
    /// - `args`: List of argument names for functions.
    /// - `vars`: Table of variable values.
    /// - `users`: Table of user-defined functions.
    ///
    /// # Returns
    /// - `Ok(AstNode)` on successful parsing.
    /// - `Err(String)` if parsing fails.
    pub fn from<'a>(
        lexemes: &[Lexeme<'a>],
        args: &[&str],
        vars: &Variables,
        users: &UserDefinedTable,
    ) -> Result<Self, String> {
        parse_to_ast(lexemes, args, vars, users)
    }

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

    /// Internal helper to create a unary operator AST node from a stack.
    fn from_unary<'a>(
        stack: &mut Vec<Self>,
        oper: UnaryOperatorKind,
    ) -> Result<(), String> {
        let expr = stack.pop()
            .ok_or(format!("Missing unary opeator {}", oper))?;
        stack.push(Self::UnaryOperator { kind: oper, expr: Box::new(expr) });
        Ok(())
    }

    /// Internal helper to create a binary operator AST node from a stack.
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

    /// Internal helper to create a function call AST node from a stack.
    fn from_function<'a>(
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

/// Converts a slice of lexemes into an abstract syntax tree (AST).
///
/// Implements a full shunting-yard-like parser handling numbers, arguments, unary/binary operators,
/// function calls, parentheses, and commas.
///
/// # Parameters
/// - `lexemes`: Slice of lexemes representing the input expression.
/// - `args`: List of argument names for the expression.
/// - `vars`: Table of variable values.
/// - `users`: Table of user-defined functions.
///
/// # Returns
/// - `Ok(AstNode)` representing the root of the parsed AST.
/// - `Err(String)` if parsing fails due to invalid syntax or unknown tokens.
fn parse_to_ast<'a>(
    lexemes: &[Lexeme<'a>],
    args: &[&str],
    vars: &Variables,
    users: &UserDefinedTable,
) -> Result<AstNode, String>{
    let mut ast_nodes: Vec<AstNode> = Vec::new();
    let mut token_stack: Vec<Token> = Vec::new();
    // record whether the previous token is finished by value or not to evaluate the token is unary operator or binary operator.
    let mut prev_is_value = false;
    let mut lexemes = lexemes.iter().peekable();

    while let Some(lexeme) = lexemes.next() {
        let token = Token::from(lexeme, args, vars, users)?;
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
            Token::Function(_) | Token::UserFunction(_) => {
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

/// Folds a binary operator with two AST nodes if possible.
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
    fn test_single_number_astnode() {
        let lexemes = lexer::from("42");
        let ast = parse_to_ast(&lexemes, &[], &Variables::new(), &UserDefinedTable::new()).unwrap();
        match ast {
            AstNode::Number(val) => assert_eq!(val, Complex::new(42.0, 0.0)),
            _ => panic!("Expected Number AST node"),
        }
    }

    #[test]
    fn test_unary_operator_negative_astnode() {
        let lexemes = lexer::from("- 3");
        let ast = parse_to_ast(&lexemes, &[], &Variables::new(), &UserDefinedTable::new()).unwrap();
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
        let ast = parse_to_ast(&lexemes, &[], &Variables::new(), &UserDefinedTable::new()).unwrap();
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
        let ast = parse_to_ast(&lexemes, &[], &Variables::new(), &UserDefinedTable::new()).unwrap();
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
        let ast = parse_to_ast(&lexemes, &[], &Variables::new(), &UserDefinedTable::new()).unwrap();
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
        let ast = parse_to_ast(&lexemes, &[], &Variables::new(), &UserDefinedTable::new()).unwrap();
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
        let ast = parse_to_ast(&lexemes, &[], &Variables::new(), &UserDefinedTable::new()).unwrap();
        assert_eq!(ast, AstNode::Number(Complex::new(0.0, 5.0)));
    }

    #[test]
    fn test_unknown_token_astnode_error() {
        let lexemes = lexer::from("@");
        let res = parse_to_ast(&lexemes, &[], &Variables::new(), &UserDefinedTable::new());
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
