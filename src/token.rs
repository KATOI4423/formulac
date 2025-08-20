//! # token.rs
//!
//! This source file is responsible for tokenizing formula strings.

use num_complex::Complex;
use phf::Map;
use phf_macros::phf_map;
use std::collections::VecDeque;

use crate::variable::{Variables, UserDefinedTable};

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
    /// Operator string for comparision
    str: &'static str,
}

impl Operator {
    /// Creates a new operator.
    ///
    /// # Arguments
    ///
    /// * `function` - Function pointer implementing the operator.
    /// * `precedence` - Operator precedence.
    /// * `is_left_assoc` - Left associativity flag.
    pub fn new(function: Func, precedence: u8, is_left_assoc: bool, str: &'static str) -> Self {
        Self {
            function,
            precedence,
            is_left_assoc,
            str,
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

    /// Returns the operator string
    pub fn str(&self) -> &str {
        self.str
    }
}

impl PartialEq for Operator {
    fn eq(&self, other: &Self) -> bool {
        // function pointer comparisons do not produce meaningful results since their addresses are not guaranteed to be unique,
        // so don't compare function pointers
        self.is_left_assoc == other.is_left_assoc
            && self.precedence == other.precedence
            && self.str == other.str
    }
}

/// Represents a mathematical function with its implementation and expected argument count.
#[derive(Debug, Clone)]
pub struct Function {
    /// Function pointer implementing the mathematical function.
    function: Func,
    /// Number of arguments the function accepts.
    args_num: u8,
    /// Function string for comparision
    str: &'static str,
}

impl Function {
    /// Creates a new mathematical function.
    ///
    /// # Arguments
    ///
    /// * `function` - Function pointer implementing the function.
    /// * `args_num` - Number of expected arguments.
    pub fn new(function: Func, args_num: u8, str: &'static str) -> Self {
        Self {
            function,
            args_num,
            str,
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

    /// Returns the function string
    pub fn str(&self) -> &str {
        self.str
    }
}

impl PartialEq for Function {
    fn eq(&self, other: &Self) -> bool {
        // function pointer comparisons do not produce meaningful results since their addresses are not guaranteed to be unique,
        // so don't compare function pointers
        (self.args_num == other.args_num) && (self.str == other.str)
    }
}

/// Token enum representing different types of tokens in the formula parser.
#[derive(Debug, Clone, PartialEq)]
pub enum Token
{
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
    "+" => Operator{ function: add, precedence: 0, is_left_assoc: true, str: "+" },
    "-" => Operator{ function: sub, precedence: 0, is_left_assoc: true, str: "-" },
    "*" => Operator{ function: mul, precedence: 1, is_left_assoc: true, str: "*" },
    "/" => Operator{ function: div, precedence: 1, is_left_assoc: true, str: "/" },
    "^" => Operator{ function: pow, precedence: 2, is_left_assoc: false, str: "^" },
};

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
    "LN_10" => Complex::new(std::f64::consts::LN_10, 0.0),
    "LN_2" => Complex::new(std::f64::consts::LN_2, 0.0),
    "LOG10_E" => Complex::new(std::f64::consts::LOG10_E, 0.0),
    "LOG2_E" => Complex::new(std::f64::consts::LOG2_E, 0.0),
    "PI" => Complex::new(std::f64::consts::PI, 0.0),
    "SQRT_2" => Complex::new(std::f64::consts::SQRT_2, 0.0),
};

/// Map of functions by their string representation.
static FUNCTIONS: Map<&'static str, Function> = phf_map! {
    "sin"   => Function{ function: sin,     args_num: 1,    str: "sin" },
    "cos"   => Function{ function: cos,     args_num: 1,    str: "cos" },
    "tan"   => Function{ function: tan,     args_num: 1,    str: "tan" },
    "asin"  => Function{ function: asin,    args_num: 1,    str: "asin" },
    "acos"  => Function{ function: acos,    args_num: 1,    str: "acos" },
    "atan"  => Function{ function: atan,    args_num: 1,    str: "atan" },
    "sinh"  => Function{ function: sinh,    args_num: 1,    str: "sinh" },
    "cosh"  => Function{ function: cosh,    args_num: 1,    str: "cosh" },
    "tanh"  => Function{ function: tanh,    args_num: 1,    str: "tanh" },
    "asinh" => Function{ function: asinh,   args_num: 1,    str: "asinh" },
    "acosh" => Function{ function: acosh,   args_num: 1,    str: "acosh" },
    "atanh" => Function{ function: atanh,   args_num: 1,    str: "atanh" },
    "exp"   => Function{ function: exp,     args_num: 1,    str: "exp" },
    "ln"    => Function{ function: ln,      args_num: 1,    str: "ln" },
    "log10" => Function{ function: log10,   args_num: 1,    str: "log10" },
    "sqrt"  => Function{ function: sqrt,    args_num: 1,    str: "sqrt" },

    "pow"   => Function{ function: pow,     args_num: 2,    str: "pow" },
};


/// Attempts to convert a string slice into a corresponding Token.
///
/// # Arguments
///
/// * `str` - The string slice to convert.
/// * `args` - List of argument variable names.
/// * `vars` - List of variables table.
/// * `users` - List of user defined tokens table.
///
/// # Returns
///
/// * `Ok(Token)` if the string corresponds to a known token (operator, function,
///   constant, argument, or punctuation).
/// * `Err(String)` if the string is unknown.
fn make_token(str: &str, args: &[&str], vars: &Variables, users: &UserDefinedTable) -> Result<Token, String> {
    if let Some(operator) = OPERATORS.get(str) {
        return Ok(Token::Operator(operator.clone()));
    }

    if let Some(function) = FUNCTIONS.get(str) {
        return Ok(Token::Function(function.clone()));
    }

    if let Some(constant) = CONSTANTS.get(str) {
        return Ok(Token::Number(*constant));
    }

    if let Some(variable) = vars.get(str) {
        return Ok(Token::Number(*variable))
    }

    if let Some(position) = args.iter().position(|&val| val == str) {
        return Ok(Token::Argument(position));
    }

    if let Some(token) = users.get(str) {
        match token {
            Token::Number(_) |
            Token::Operator(_) |
            Token::Function(_)
                => return Ok(token.clone()),
            Token::Argument(_) |
            Token::LParen | Token::RParen | Token::Comma
                => return Err(format!("Invaild user defined token: {:?}", token)),
        }
    }

    match str {
        "(" => return Ok(Token::LParen),
        ")" => return Ok(Token::RParen),
        "," => return Ok(Token::Comma),
        _ => (),
    }

    if str.ends_with(IMAGINARY_UNIT) {
        let num_part =  &str[..(str.len() - IMAGINARY_UNIT.len())];
        if let Ok(val) = num_part.parse::<f64>() {
            return Ok(Token::Number(Complex::new(0.0, val)));
        }
        if num_part.is_empty() {
            // Imaginary unit only
            return Ok(Token::Number(Complex::new(0.0, 1.0)));
        }
    }

    if let Ok(val) = str.parse::<f64>() {
        return Ok(Token::Number(Complex::new(val, 0.0)));
    }

    Err(format!("Unknown string \"{}\"", str))
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
/// * `users` - The list of user defined tokens table.
///
/// # Returns
///
/// * `Ok(Tokens)` containing the parsed tokens if successful.
/// * `Err(String)` with an error message if tokenization fails.
pub fn divide_to_tokens(formula: &str, args: &[&str], vars: &Variables, users: &UserDefinedTable) -> Result<Tokens, String> {
    let mut tokens: Tokens = VecDeque::new();
    let mut chars = formula.char_indices().peekable();
    let mut prev_is_value = false; // whether the previous token is finished by value or not

    while let Some((start_idx, ch)) = chars.next() {
        if ch.is_whitespace() {
            continue;
        }

        let end_idx = if ch.is_ascii_alphabetic() {
            // English letters block (for function, variable, constance name)
            let mut end = start_idx + ch.len_utf8();
            while let Some(&(_, next_ch)) = chars.peek() {
                if next_ch.is_ascii_alphabetic() || next_ch == '_' {
                    let (idx, ch) = chars.next().unwrap();
                    end = idx + ch.len_utf8();
                } else {
                    break;
                }
            }
            end
        } else if ch.is_ascii_digit()
            || ("+-".find(ch).is_some() && !prev_is_value)
        {
            // Numerical value (may have imaginary units at the end)
            let mut end = start_idx + ch.len_utf8();
            while let Some(&(_, next_ch)) = chars.peek() {
                if next_ch.is_ascii_digit() || ".eE+-".find(next_ch).is_some() {
                    let (idx, ch) = chars.next().unwrap();
                    end = idx + ch.len_utf8();
                } else if String::from(next_ch) == IMAGINARY_UNIT {
                    let (idx, ch) = chars.next().unwrap();
                    end = idx + ch.len_utf8();
                    break;
                } else {
                    break;
                }
            }
            end
        } else {
            // Symbol (operation, paren, comma, etc.)
            start_idx + ch.len_utf8()
        };

        let token_str = &formula[start_idx..end_idx];
        let token = make_token(token_str, args, vars, users)?;
        prev_is_value = matches!(token,
            Token::Number(_) | Token::Argument(_) | Token::RParen
        );
        tokens.push_back(token);
    }

    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_constant() {
        let tokens = divide_to_tokens("E", &[], &Variables::new(), &UserDefinedTable::new()).unwrap();
        let expected = VecDeque::from([Token::Number(std::f64::consts::E.into())]);
        assert_eq!(tokens, expected);
    }

    #[test]
    fn test_single_operator() {
        let tokens = divide_to_tokens("+", &[], &Variables::new(), &UserDefinedTable::new()).unwrap();
        let expected = VecDeque::from([Token::Operator(OPERATORS.get("+").unwrap().clone())]);
        assert_eq!(tokens, expected);
    }

    #[test]
    fn test_argument_lockup() {
        let tokens = divide_to_tokens("y", &["y"], &Variables::new(), &UserDefinedTable::new()).unwrap();
        let expected = VecDeque::from([Token::Argument(0)]);
        assert_eq!(tokens, expected);
    }

    #[test]
    fn test_function_call() {
        let tokens = divide_to_tokens("sin(PI)", &[], &Variables::new(), &UserDefinedTable::new()).unwrap();
        let expected = VecDeque::from([
            Token::Function(FUNCTIONS.get("sin").unwrap().clone()),
            Token::LParen,
            Token::Number(CONSTANTS.get("PI").unwrap().clone()),
            Token::RParen,
        ]);
        assert_eq!(tokens, expected);
    }

    #[test]
    fn test_function_no_argument() {
        let tokens = divide_to_tokens("sin()", &[], &Variables::new(), &UserDefinedTable::new()).unwrap();
        let expected = VecDeque::from([
            Token::Function(FUNCTIONS.get("sin").unwrap().clone()),
            Token::LParen,
            Token::RParen,
        ]);
        assert_eq!(tokens, expected);
    }

    #[test]
    fn test_multiple_argument() {
        let tokens = divide_to_tokens("x + y", &["x", "y"], &Variables::new(), &UserDefinedTable::new()).unwrap();
        let expected = VecDeque::from([
            Token::Argument(0),
            Token::Operator(OPERATORS.get("+").unwrap().clone()),
            Token::Argument(1),
        ]);
        assert_eq!(tokens, expected);
    }

    #[test]
    fn test_unknown_token_error() {
        let err = divide_to_tokens("abc123", &[], &Variables::new(), &UserDefinedTable::new()).unwrap_err();
        assert_eq!(err, "Unknown string \"abc\"");
    }

    #[test]
    fn test_parentheses_and_comma() {
        let tokens = divide_to_tokens("(,)", &[], &Variables::new(), &UserDefinedTable::new()).unwrap();
        let expected = VecDeque::from([
            Token::LParen,
            Token::Comma,
            Token::RParen,
        ]);
        assert_eq!(tokens, expected);
    }

    #[test]
    fn test_whitespace() {
        let tokens = divide_to_tokens("ln (\tx\t)", &["x"], &Variables::new(), &UserDefinedTable::new()).unwrap();
        let expected = VecDeque::from([
            Token::Function(FUNCTIONS.get("ln").unwrap().clone()),
            Token::LParen,
            Token::Argument(0),
            Token::RParen,
        ]);
        assert_eq!(tokens, expected);
    }

    #[test]
    fn test_real() {
        let tokens = divide_to_tokens("6.28", &[], &Variables::new(), &UserDefinedTable::new()).unwrap();
        let expected = VecDeque::from([
            Token::Number(Complex::new(6.28, 0.0))
        ]);
        assert_eq!(tokens, expected);
    }

    #[test]
    fn test_unary_operator() {
        let tokens = divide_to_tokens("-3.5", &[], &Variables::new(), &UserDefinedTable::new()).unwrap();
        let expected = VecDeque::from([
            Token::Number(Complex::new(-3.5, 0.0))
        ]);
        assert_eq!(tokens, expected);
    }

    #[test]
    fn test_scientific_notation() {
        let tokens = divide_to_tokens("1.0e+4", &[], &Variables::new(), &UserDefinedTable::new()).unwrap();
        let expected = VecDeque::from([
            Token::Number(Complex::new(1.0E+04, 0.0))
        ]);
        assert_eq!(tokens, expected);
    }

    #[test]
    fn test_start_with_period() {
        let err = divide_to_tokens(".5", &[], &Variables::new(), &UserDefinedTable::new()).unwrap_err();
        assert_eq!(err, "Unknown string \".\"");
    }

    #[test]
    fn test_imaginary() {
        let tokens = divide_to_tokens("1.5i", &[], &Variables::new(), &UserDefinedTable::new()).unwrap();
        let expected = VecDeque::from([
            Token::Number(Complex::new(0.0, 1.5))
        ]);
        assert_eq!(tokens, expected);
    }

    #[test]
    fn test_imaginary_unit() {
        let tokens = divide_to_tokens("i", &[], &Variables::new(), &UserDefinedTable::new()).unwrap();
        let expected = VecDeque::from([
            Token::Number(Complex::new(0.0, 1.0))
        ]);
        assert_eq!(tokens, expected);
    }

    #[test]
    fn test_variable() {
        let tokens = divide_to_tokens("a", &[], &Variables::from(&[("a", 3.0)]), &UserDefinedTable::new()).unwrap();
        let expected = VecDeque::from([
            Token::Number(Complex::from(3.0))
        ]);
        assert_eq!(tokens, expected);
    }

    #[test]
    fn test_e() {
        let tokens = divide_to_tokens("1e+5 + E", &[], &Variables::new(), &UserDefinedTable::new()).unwrap();
        let expected = VecDeque::from([
            Token::Number(Complex::new(1.0e5, 0.0)),
            Token::Operator(OPERATORS.get("+").unwrap().clone()),
            Token::Number(CONSTANTS.get("E").unwrap().clone()),
        ]);
        assert_eq!(tokens, expected);
    }

    #[test]
    fn test_multibyte_token_boundary() {
        let err = divide_to_tokens("あ123", &[], &Variables::new(), &UserDefinedTable::new()).unwrap_err();
        assert_eq!(err, "Unknown string \"あ\"");
    }

    #[test]
    fn test_user_defined_function_basic() {
        // Prepare user-defined function table
        let mut users = UserDefinedTable::new();

        // Define a simple 1-argument function: f(x) = x + 1
        let f_token = Token::Function(Function::new(
            |args| args[0] + Complex::new(1.0, 0.0),
            1,
            "f",
        ));

        users.register("f", f_token.clone());

        let vars = Variables::new();

        // Tokenize formula using user-defined function
        let tokens = divide_to_tokens("f(2)", &[], &vars, &users).unwrap();

        // Expect: [Token::Real(2.0), Token::Function(f)]
        assert_eq!(tokens.len(), 4); // "f", "(", "2", ")"
        assert_eq!(tokens, VecDeque::from([
            f_token, Token::LParen, Token::Number(Complex::from(2.0)), Token::RParen,
        ]));
    }
}
