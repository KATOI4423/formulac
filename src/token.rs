//! # token.rs
//!
//! This source file is responsible for tokenizing formula strings.

use num_complex::Complex;
use phf::Map;
use phf_macros::phf_map;
use std::collections::VecDeque;

use crate::variable::Variables;

/// Function pointer type alias representing a mathematical function.
///
/// This type defines a function that takes a slice of complex numbers as input
/// arguments and returns a single complex number as the result.
///
/// This is used to represent operators and mathematical functions in the parser,
/// where the input slice length corresponds to the number of function arguments.
pub type Func = fn(&[Complex<f64>]) -> Complex<f64>;

/// Constant string representing an imaginary unit
const IMAGINARY_UNIT: &str = "i";

/// Represents a mathematical operator with its function, precedence, and associativity.
#[derive(Debug, Clone)]
pub struct Operator {
    /// The function implementing the operator's computation.
    function: Func,
    /// Operator precedence (higher value means higher precedence).
    precedence: u8,
    /// Whether the operator is left associative.
    is_left_assoc: bool,
}

impl Operator {
    /// Creates a new operator.
    ///
    /// # Arguments
    ///
    /// * `function` - Function pointer implementing the operator.
    /// * `precedence` - Operator precedence.
    /// * `is_left_assoc` - Left associativity flag.
    pub fn new(function: Func, precedence: u8, is_left_assoc: bool) -> Self {
        Self {
            function,
            precedence,
            is_left_assoc
        }
    }

    /// Executes the operator function with the given arguments.
    ///
    /// # Arguments
    ///
    /// * `args` - Slice of complex numbers as arguments.
    ///
    /// # Returns
    ///
    /// The computed complex number result.
    pub fn func(&self, args: &[Complex<f64>]) -> Complex<f64> {
        (self.function)(args)
    }

    /// Returns the operator precedence.
    pub fn precedence(&self) -> u8 {
        self.precedence
    }

    /// Returns whether the operator is left associative.
    pub fn is_left_assoc(&self) -> bool {
        self.is_left_assoc
    }
}

/// Represents a mathematical function with its implementation and expected argument count.
#[derive(Debug, Clone)]
pub struct Function {
    /// Function pointer implementing the mathematical function.
    function: Func,
    /// Number of arguments the function accepts.
    args_num: u8,
}

impl Function {
    /// Creates a new mathematical function.
    ///
    /// # Arguments
    ///
    /// * `function` - Function pointer implementing the function.
    /// * `args_num` - Number of expected arguments.
    pub fn new(function: Func, args_num: u8) -> Self {
        Self {
            function,
            args_num,
        }
    }

    /// Executes the function with the given arguments.
    ///
    /// # Arguments
    ///
    /// * `args` - Slice of complex numbers as arguments.
    ///
    /// # Returns
    ///
    /// The computed complex number result.
    pub fn func(&self, args: &[Complex<f64>]) -> Complex<f64> {
        (self.function)(args)
    }

    /// Returns the number of arguments this function expects.
    pub fn args_num(&self) -> u8 {
        self.args_num
    }
}

/// Token enum representing different types of tokens in the formula parser.
#[derive(Debug, Clone)]
pub enum Token
{
    /// Variable token holding the resolved value.
    ///
    /// User-defined variable with external value resolved at parse time.
    Variable(Complex<f64>),

    /// Function argument token by position index.
    ///
    /// Represents an argument index in the function's parameter list.
    Argument(usize),

    /// Constant token for predefined mathematical constants.
    Constant(f64),

    /// Real number token.
    Real(f64),

    /// Imaginary number token.
    Imaginary(f64),

    /// Operator token.
    Operator(Operator),

    /// Function token.
    Function(Function),

    /// Left parenthesis token '('.
    LParen,

    /// Right parenthesis token ')'.
    RParen,

    /// Comma token ',' used as argument separator.
    Comma,
}

/// A collection of tokens.
///
/// Uses a double-ended queue to allow efficient insertion/removal at both ends.
/// Typically used to store tokenized output before parsing.
pub type Tokens = VecDeque<Token>;

/// Adds two complex numbers.
fn add(args: &[Complex<f64>]) -> Complex<f64> {
    args[0] + args[1]
}
/// Subtracts the second complex number from the first.
fn sub(args: &[Complex<f64>]) -> Complex<f64> {
    args[0] - args[1]
}
/// Multiplies two complex numbers.
fn mul(args: &[Complex<f64>]) -> Complex<f64> {
    args[0] * args[1]
}
/// Divides the first complex number by the second.
fn div(args: &[Complex<f64>]) -> Complex<f64> {
    args[0] / args[1]
}

/// Macro to define unary functions easily from method names on Complex<f64>.
///
/// For example, define_unary_func!(sin) expands to
/// `fn sin(args: &[Complex<f64>]) -> Complex<f64> { args[0].sin() }`.
macro_rules! define_unary_func {
    ($name:ident) => {
        fn $name(args: &[Complex<f64>]) -> Complex<f64> {
            args[0].$name()
        }
    };
}

define_unary_func!(sin);
define_unary_func!(cos);
define_unary_func!(tan);
define_unary_func!(asin);
define_unary_func!(acos);
define_unary_func!(atan);
define_unary_func!(sinh);
define_unary_func!(cosh);
define_unary_func!(tanh);
define_unary_func!(asinh);
define_unary_func!(acosh);
define_unary_func!(atanh);
define_unary_func!(exp);
define_unary_func!(ln);
define_unary_func!(log10);
define_unary_func!(sqrt);

/// Raises the first argument to the power of the second.
fn pow(args: &[Complex<f64>]) -> Complex<f64> {
    args[0].powc(args[1])
}


/// Map of operators by their string representation.
static OPERATORS: Map<&'static str, Operator> = phf_map! {
    "+" => Operator{ function: add, precedence: 0, is_left_assoc: true },
    "-" => Operator{ function: sub, precedence: 0, is_left_assoc: true },
    "*" => Operator{ function: mul, precedence: 1, is_left_assoc: true },
    "/" => Operator{ function: div, precedence: 1, is_left_assoc: true },
    "^" => Operator{ function: pow, precedence: 2, is_left_assoc: false },
};

/// Map of mathematical constants by their string representation.
static CONSTANTS: Map<&'static str, f64> = phf_map! {
    "E" => std::f64::consts::E,
    "FRAC_1_PI" => std::f64::consts::FRAC_1_PI,
    "FRAC_1_SQRT_2" => std::f64::consts::FRAC_1_SQRT_2,
    "FRAC_2_PI" => std::f64::consts::FRAC_2_PI,
    "FRAC_2_SQRT_PI" => std::f64::consts::FRAC_2_SQRT_PI,
    "FRAC_PI_2" => std::f64::consts::FRAC_PI_2,
    "FRAC_PI_3" => std::f64::consts::FRAC_PI_3,
    "FRAC_PI_4" => std::f64::consts::FRAC_PI_4,
    "FRAC_PI_6" => std::f64::consts::FRAC_PI_6,
    "FRAC_PI_8" => std::f64::consts::FRAC_PI_8,
    "LN_10" => std::f64::consts::LN_10,
    "LN_2" => std::f64::consts::LN_2,
    "LOG10_E" => std::f64::consts::LOG10_E,
    "LOG2_E" => std::f64::consts::LOG2_E,
    "PI" => std::f64::consts::PI,
    "SQRT_2" => std::f64::consts::SQRT_2,
};

/// Map of functions by their string representation.
static FUNCTIONS: Map<&'static str, Function> = phf_map! {
    "sin"   => Function{ function: sin,     args_num: 1 },
    "cos"   => Function{ function: cos,     args_num: 1 },
    "tan"   => Function{ function: tan,     args_num: 1 },
    "asin"  => Function{ function: asin,    args_num: 1 },
    "acos"  => Function{ function: acos,    args_num: 1 },
    "atan"  => Function{ function: atan,    args_num: 1 },
    "sinh"  => Function{ function: sinh,    args_num: 1 },
    "cosh"  => Function{ function: cosh,    args_num: 1 },
    "tanh"  => Function{ function: tanh,    args_num: 1 },
    "asinh" => Function{ function: asinh,   args_num: 1 },
    "acosh" => Function{ function: acosh,   args_num: 1 },
    "atanh" => Function{ function: atanh,   args_num: 1 },
    "exp"   => Function{ function: exp,     args_num: 1 },
    "ln"    => Function{ function: ln,      args_num: 1 },
    "log10" => Function{ function: log10,   args_num: 1 },
    "sqrt"  => Function{ function: sqrt,    args_num: 1 },

    "pow"   => Function{ function: pow,     args_num: 2 },
};


/// Attempts to convert a string slice into a corresponding Token.
///
/// # Arguments
///
/// * `str` - The string slice to convert.
/// * `args` - List of argument variable names.
/// * `vars` - List of variables table.
///
/// # Returns
///
/// * `Ok(Token)` if the string corresponds to a known token (operator, function,
///   constant, argument, or punctuation).
/// * `Err(String)` if the string is unknown.
fn make_token(str: &str, args: &[&str], vars: &Variables) -> Result<Token, String> {
    if let Some(operator) = OPERATORS.get(str) {
        return Ok(Token::Operator(operator.clone()));
    }

    if let Some(function) = FUNCTIONS.get(str) {
        return Ok(Token::Function(function.clone()));
    }

    if let Some(constant) = CONSTANTS.get(str) {
        return Ok(Token::Constant(*constant));
    }

    if let Some(variable) = vars.get(str) {
        return Ok(Token::Variable(*variable))
    }

    if let Some(position) = args.iter().position(|&val| val == str) {
        return Ok(Token::Argument(position));
    }

    match str {
        "(" => return Ok(Token::LParen),
        ")" => return Ok(Token::RParen),
        "," => return Ok(Token::Comma),
        _ => (),
    }

    if str.ends_with(IMAGINARY_UNIT)
     && let Ok(val) = str[..(str.len() - IMAGINARY_UNIT.len())].parse::<f64>() {
        return Ok(Token::Imaginary(val));
    }

    if let Ok(val) = str.parse::<f64>() {
        return Ok(Token::Real(val));
    }

    return Err(format!("Unknown string {}", str));
}

/// Tokenizes a formula string into a sequence of Tokens.
///
/// Splits the formula into meaningful tokens (operators, functions, constants,
/// variables, parentheses, commas) according to known tokens and argument list.
///
/// # Arguments
///
/// * `formula` - The formula string to tokenize.
/// * `args` - The list of argument variable names.
/// * `vars` - The list of variables table.
///
/// # Returns
///
/// * `Ok(Tokens)` containing the parsed tokens if successful.
/// * `Err(String)` with an error message if tokenization fails.
pub fn divide_to_tokens(formula: &str, args: &[&str], vars: &Variables) -> Result<Tokens, String> {
    let mut tokens: Tokens = VecDeque::new();
    let mut current_start: usize = 0;
    let mut current_end: usize = 0;

    let is_token = |str: &str| -> bool {
        OPERATORS.contains_key(str) || FUNCTIONS.contains_key(str) || CONSTANTS.contains_key(str)
            || match str { "(" | ")" | "," => true, _ => false }
    };

    let is_number = |s: &str| -> bool {
        let target = if s.ends_with(IMAGINARY_UNIT) {
            &s[..(s.len()-IMAGINARY_UNIT.len())] // this range is always safe
        } else {
            s
        };
        target.parse::<f64>().is_ok()
    };

    let mut char_indices = formula.char_indices().peekable();

    while let Some((idx, ch)) = char_indices.next() {
        let next_end = idx + ch.len_utf8();

        if ch.is_whitespace() {
            if current_start != current_end {
                let token_str = &formula[current_start..current_end];
                tokens.push_back(make_token(token_str, args, vars)?);
            }
            current_start = next_end;
            current_end = current_start;
            continue;
        }

        let extended_str = &formula[current_start..next_end];
        if is_number(extended_str) || is_token(extended_str) {
            current_end = next_end;
            continue;
        }

        let current_str = &formula[current_start..current_end];
        let ch_str = &formula[idx..next_end];
        if is_number(current_str) || is_token(current_str) || is_token(ch_str) {
            if current_start != current_end {
                tokens.push_back(make_token(current_str, args, vars)?);
            }
            current_start = idx;
            current_end = next_end;
            continue;
        }

        current_end = next_end;
    }

    if current_start != current_end {
        let token_str = &formula[current_start..current_end];
        tokens.push_back(make_token(token_str, args, vars)?);
    }

    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens_to_debug_str(tokens: &Tokens) -> Vec<String> {
        tokens.iter().map(|t| format!("{:?}", t)).collect()
    }

    #[test]
    fn test_single_constant() {
        let tokens = divide_to_tokens("E", &[], &Variables::new()).unwrap();
        let expected = VecDeque::from([Token::Constant(std::f64::consts::E)]);
        assert_eq!(tokens_to_debug_str(&tokens), tokens_to_debug_str(&expected));
    }

    #[test]
    fn test_single_operator() {
        let tokens = divide_to_tokens("+", &[], &Variables::new()).unwrap();
        let expected = VecDeque::from([Token::Operator(OPERATORS.get("+").unwrap().clone())]);
        assert_eq!(tokens_to_debug_str(&tokens),tokens_to_debug_str(&expected));
    }

    #[test]
    fn test_argument_lockup() {
        let tokens = divide_to_tokens("y", &["y"], &Variables::new()).unwrap();
        let expected = VecDeque::from([Token::Argument(0)]);
        assert_eq!(tokens_to_debug_str(&tokens),tokens_to_debug_str(&expected));
    }

    #[test]
    fn test_function_call() {
        let tokens = divide_to_tokens("sin(PI)", &[], &Variables::new()).unwrap();
        let expected = VecDeque::from([
            Token::Function(FUNCTIONS.get("sin").unwrap().clone()),
            Token::LParen,
            Token::Constant(CONSTANTS.get("PI").unwrap().clone()),
            Token::RParen,
        ]);
        assert_eq!(tokens_to_debug_str(&tokens),tokens_to_debug_str(&expected));
    }

    #[test]
    fn test_multiple_argument() {
        let tokens = divide_to_tokens("x + y", &["x", "y"], &Variables::new()).unwrap();
        let expected = VecDeque::from([
            Token::Argument(0),
            Token::Operator(OPERATORS.get("+").unwrap().clone()),
            Token::Argument(1),
        ]);
        assert_eq!(tokens_to_debug_str(&tokens),tokens_to_debug_str(&expected));
    }

    #[test]
    fn test_unknown_token_error() {
        let token = "abc123";
        let err = divide_to_tokens(token, &[], &Variables::new()).unwrap_err();
        assert_eq!(err, format!("Unknown string {}", token));
    }

    #[test]
    fn test_parentheses_and_comma() {
        let tokens = divide_to_tokens("(,)", &[], &Variables::new()).unwrap();
        let expected = VecDeque::from([
            Token::LParen,
            Token::Comma,
            Token::RParen,
        ]);
        assert_eq!(tokens_to_debug_str(&tokens),tokens_to_debug_str(&expected));
    }

    #[test]
    fn test_whitespace() {
        let tokens = divide_to_tokens("ln (\tx\t)", &["x"], &Variables::new()).unwrap();
        let expected = VecDeque::from([
            Token::Function(FUNCTIONS.get("ln").unwrap().clone()),
            Token::LParen,
            Token::Argument(0),
            Token::RParen,
        ]);
        assert_eq!(tokens_to_debug_str(&tokens),tokens_to_debug_str(&expected));
    }

    #[test]
    fn test_real() {
        let tokens = divide_to_tokens("6.28", &[], &Variables::new()).unwrap();
        let expected = VecDeque::from([
            Token::Real(6.28)
        ]);
        assert_eq!(tokens_to_debug_str(&tokens), tokens_to_debug_str(&expected));
    }

    #[test]
    fn test_imaginary() {
        let tokens = divide_to_tokens("-1.5i", &[], &Variables::new()).unwrap();
        let expected = VecDeque::from([
            Token::Imaginary(-1.5)
        ]);
        assert_eq!(tokens_to_debug_str(&tokens), tokens_to_debug_str(&expected));
    }

    #[test]
    fn test_variable() {
        let tokens = divide_to_tokens("a", &[], &Variables::from(&[("a", 3.0)])).unwrap();
        let expected = VecDeque::from([
            Token::Variable(Complex::from(3.0))
        ]);
        assert_eq!(tokens_to_debug_str(&tokens), tokens_to_debug_str(&expected));
    }

    #[test]
    fn test_multibyte_token_boundary() {
        let token = "あ123";
        let err = divide_to_tokens(token, &[], &Variables::new()).unwrap_err();
        assert_eq!(err, format!("Unknown string {}", token));
    }
}
