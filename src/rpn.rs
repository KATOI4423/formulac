//! rpn.rs
//!
//! Converts an infix mathematical expression into Reverse Polish Notation (RPN)
//! using the Shunting-yard algorithm. The implementation handles parentheses,
//! commas (for function arguments), and operator precedence/associativity.

use crate::token::{divide_to_tokens, Operator, Token, Tokens};
use crate::variable::{Variables, UserDefinedTable};

/// Handles the case when a right parenthesis `)` is encountered.
///
/// Pops tokens from the stack and pushes them into the RPN output until a left
/// parenthesis `(` is found. If a function token is found directly before the
/// left parenthesis, it is also pushed to the output.
///
/// # Arguments
///
/// * `rpn` - The RPN output token list.
/// * `stack` - The operator/function/parenthesis stack.
///
/// # Errors
///
/// Returns an error if no matching left parenthesis is found.
///
/// # Algorithm
///
/// This follows the Shunting-yard algorithm's rule for handling a right parenthesis:
/// remove operators until `(` is encountered, then discard the `(`.
/// If there is a function before `(`, output it.
fn make_rpn_case_of_rparen(rpn: &mut Tokens, stack: &mut Tokens) -> Result<(), String> {
    loop {
        match stack.pop_back() {
            Some(Token::LParen) => {
                if let Some(Token::Function(_))= stack.back() {
                    rpn.push_back(stack.pop_back().unwrap());
                }
                return Ok(());
            },
            Some(token) => {
                rpn.push_back(token);
            },
            None => return Err("Invalid formula: Right Paren used, but Left Paren not found.".into()),
        }
    }
}

/// Handles the case when a comma `,` is encountered inside a function argument list.
///
/// Pops tokens from the stack into the RPN output until a left parenthesis `(`
/// is found.
///
/// # Arguments
///
/// * `rpn` - The RPN output token list.
/// * `stack` - The operator/function/parenthesis stack.
///
/// # Errors
///
/// Returns an error if no matching left parenthesis is found.
///
/// # Notes
///
/// In the Shunting-yard algorithm, commas separate function arguments.
fn make_rpn_case_of_comma(rpn: &mut Tokens, stack: &mut Tokens) -> Result<(), String> {
    loop {
        match stack.back() {
            Some(Token::LParen) => return Ok(()),
            Some(_) => rpn.push_back(stack.pop_back().unwrap()),
            None => return Err("Invalid formula: Comma used, but Left Paren not found.".into()),
        }
    }
}

/// Handles the case when an operator is encountered.
///
/// Compares the precedence and associativity of the current operator with the
/// operator on top of the stack. Pops operators from the stack into the RPN output
/// while the operator at the top of the stack has greater precedence (or equal
/// precedence and the current operator is left-associative).
///
/// # Arguments
///
/// * `oper` - The current operator being processed.
/// * `rpn` - The RPN output token list.
/// * `stack` - The operator/function/parenthesis stack.
///
/// # Notes
///
/// This implements the Shunting-yard algorithm's operator precedence handling.
fn make_rpn_case_of_operator(oper: Operator, rpn: &mut Tokens, stack: &mut Tokens) {
    while let Some(Token::Operator(poped)) = stack.back() {
        if (oper.is_left_assoc() && (oper.precedence() <= poped.precedence()))
            || (!oper.is_left_assoc() && (oper.precedence() < poped.precedence()))
        {
            rpn.push_back(stack.pop_back().unwrap());
        } else {
            break;
        }
    }
    stack.push_back(Token::Operator(oper));
}

/// Converts an infix mathematical expression into Reverse Polish Notation (RPN).
///
/// Implements the Shunting-yard algorithm, processing variables, constants,
/// operators, functions, parentheses, and commas.
///
/// # Arguments
///
/// * `formula` - The mathematical expression in infix notation.
/// * `args` - A list of argument names for functions.
/// * `vars` - The variable table containing predefined variables and values.
/// * `users` - The list of user defined tokens table.
///
/// # Returns
///
/// * `Ok(Tokens)` - The list of tokens in RPN order.
/// * `Err(String)` - If the formula contains syntax errors (e.g., mismatched parentheses).
///
/// # Example
pub fn make_rpn(formula: &str, args: &[&str], vars: &Variables, users: &UserDefinedTable) -> Result<Tokens, String> {
    let mut tokens = divide_to_tokens(formula, args, vars, users)?;
    let mut rpn = Tokens::new();
    let mut stack = Tokens::new();

    while let Some(token) = tokens.pop_front() {
        match token {
            Token::Number(_) |
            Token::Argument(_)
                => rpn.push_back(token),

            Token::Function(_) |
            Token::LParen
                => stack.push_back(token),

            Token::RParen
                => make_rpn_case_of_rparen(&mut rpn, &mut stack)?,

            Token::Comma
                => make_rpn_case_of_comma(&mut rpn, &mut stack)?,

            Token::Operator(oper)
                => make_rpn_case_of_operator(oper, &mut rpn, &mut stack),
        }
    }

    // Push any remaining stack contents to the RPN output
    while let Some(token) = stack.pop_back() {
        match token {
            Token::LParen | Token::RParen => return Err("Mismatched parentheses".into()),
            _ => rpn.push_back(token),
        }
    }
    Ok(rpn)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token::Function;
    use num_complex::Complex;
    use std::collections::VecDeque;

    #[test]
    fn test_make_rpn_basic() {
        let mut vars = Variables::new();
        vars.insert(&[("a", Complex::new(1.0, 0.0))]);

        // Test 1: simple arithmetic
        let rpn = make_rpn("3 + 4 * 2", &[], &vars, &UserDefinedTable::new()).unwrap();
        let expected = vec![
            Token::Number(Complex::new(3.0, 0.0)),
            Token::Number(Complex::new(4.0, 0.0)),
            Token::Number(Complex::new(2.0, 0.0)),
            Token::Operator(Operator::new(|args| args[0] * args[1], 1, true, "*")),
            Token::Operator(Operator::new(|args| args[0] + args[1], 0, true, "+")),
        ];
        assert_eq!(rpn.len(), expected.len());
        for (a, b) in rpn.iter().zip(expected.iter()) {
            assert_eq!(a, b);
        }

        // Test 2: function with argument
        let rpn = make_rpn("sin(a)", &["a"], &vars, &UserDefinedTable::new()).unwrap();
        assert_eq!(rpn.len(), 2);
        match &rpn[0] {
            Token::Number(_) => {},
            _ => panic!("Expected Variable token"),
        }
        match &rpn[1] {
            Token::Function(f) => assert_eq!(f.args_num(), 1),
            _ => panic!("Expected Function token"),
        }

        // Test 3: nested functions and parentheses
        let rpn = make_rpn("cos(1 + a)", &["a"], &vars, &UserDefinedTable::new()).unwrap();
        assert_eq!(rpn.len(), 4);
        // rpn[0]: Real(1.0), rpn[1]: Variable(a), rpn[2]: Operator(+), rpn[3]: Function(cos)
    }

    #[test]
    fn test_make_rpn_errors() {
        let vars = Variables::new();

        // unmatched right parenthesis
        let err = make_rpn("1 + )", &[], &vars, &UserDefinedTable::new()).unwrap_err();
        assert!(err.contains("Right Paren used, but Left Paren not found"));

        // unmatched left parenthesis
        let err = make_rpn("(1 + 2", &[], &vars, &UserDefinedTable::new()).unwrap_err();
        assert!(err.contains("wrong arguments for function") || err.contains("Mismatched parentheses"));
    }

    #[test]
    fn test_make_rpn_nested_functions() {
        // cos(sin(x)) -> RPN: x sin cos
        let rpn = make_rpn("cos(sin(x))", &["x"], &Variables::new(), &UserDefinedTable::new()).unwrap();

        // Expected RPN token sequence
        let expected = VecDeque::from([
            Token::Argument(0),
            Token::Function(Function::new(|args| args[0].sin(), 1, "sin")),
            Token::Function(Function::new(|args| args[0].cos(), 1, "cos")),
        ]);

        assert_eq!(rpn, expected, "RPN output does not match expected sequence");
    }

    #[test]
    fn test_user_defined_function_rpn() {
        let mut users = UserDefinedTable::new();
        let f_token = Token::Function(Function::new(
            |args| args[0] * Complex::new(2.0, 0.0),
            1,
            "double",
        ));
        users.register("double", f_token.clone());

        let vars = Variables::new();

        let rpn = make_rpn("double(3)", &[], &vars, &users).unwrap();
        let expected = VecDeque::from([
            Token::Number(Complex::from(3.0)),
            f_token,
        ]);

        assert_eq!(rpn, expected, "RPN output does not match expected sequence");
    }
}
