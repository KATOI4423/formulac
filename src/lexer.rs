//! lexer.rs
//!
//! This module provides a simple lexical analyzer (lexer) for mathematical expressions.
//! It splits an input string into a sequence of `Lexeme`s, each representing a
//! continuous piece of text with its corresponding position in the original string.
//!
//! The lexer handles identifiers, numeric literals (including decimal, scientific
//! notation, and imaginary unit), and single-character operators or punctuation.

use std::ops::Range;

/// Constant char representing an imaginary unit
pub const IMAGINARY_UNIT: char = 'i';

/// Represents a single lexeme etracted from the input string.
///
/// A `Lexeme` stores a text slice and its span (start..end indices) within
/// the original input string.
#[derive(Debug, Clone, PartialEq)]
pub struct Lexeme {
    text: String,
    span: Range<usize>,
}

impl Lexeme {
    /// Create a new `Lexeme`.
    ///
    /// # Argument
    ///
    /// * `text` - The slice of text corresponding to the lexeme.
    /// * `span` - The range of the lexeme in the original input string.
    pub fn new(text: &str, span: Range<usize>) -> Self {
        Self {
            text: text.to_string(),
            span
        }
    }

    /// Returns the text slice of the lexeme.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Returns the start index of the lexeme in the original input string.
    pub fn start(&self) -> &usize {
        &self.span.start
    }

    /// Returns the end index of the lexeme in the original input string.
    pub fn end(&self) -> &usize {
        &self.span.end
    }

    /// Returns the span index (start, end) of the lexeme in the original input string.
    pub fn span(&self) -> &Range<usize> {
        &self.span
    }
}

impl std::fmt::Display for Lexeme
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{name} [{start}, {end})", name=self.text, start=self.span.start, end=self.span.end)
    }
}

/// Type alias for a collection of lexemes.
pub type Lexemes = Vec<Lexeme>;

type CharIter<'a> = std::iter::Peekable<std::str::CharIndices<'a>>;

/// Parses an identifier starting at `start_idx`.
///
/// An identifier is a sequence of alphanumeric characters or underscores.
///
/// # Arguments
///
/// * `start_idx` - The starting index of the identifier.
/// * `chars` - The iterator over character indices, peekable.
///
/// # Returns
///
/// The ending index of the identifier.
fn parse_ident(start_idx: usize, chars: &mut CharIter) -> usize
{
    let mut end = start_idx + 1;
    while let Some(&(_, ch)) = chars.peek() {
        if ch.is_alphanumeric() || ch == '_' {
            let (idx, ch) = chars.next().unwrap();
            end = idx + ch.len_utf8();
        } else {
            break;
        }
    }

    end
}

/// Parses a numeric literal starting at `start_idx`.
///
/// Supports integers, decimals, scientific notation, and the imaginary unit.
///
/// # Arguments
///
/// * `start_idx` - The starting index of the identifier.
/// * `chars` - The iterator over character indices, peekable.
///
/// # Returns
///
/// The ending index of the numeric literal.
fn parse_number(start_idx: usize, chars: &mut CharIter) -> usize
{
    let mut end = start_idx + 1;
    let mut seen_e = false;

    while let Some(&(_, ch)) = chars.peek() {
        let accept = match ch {
            d if d.is_ascii_digit() || d == '.' => true,
            'e' | 'E' if !seen_e => { seen_e = true; true },
            '+' | '-' if seen_e => true,
            IMAGINARY_UNIT => {
                // imaginary unit meens the end of number token
                let (idx, ch) = chars.next().unwrap();
                end = idx + ch.len_utf8();
                break;
            },
            _ => false,
        };

        if accept {
            let (idx, ch) = chars.next().unwrap();
            end = idx + ch.len_utf8();
        } else {
            break;
        }
    }

    end
}

/// Splits the input string into a sequence of `Lexeme`s.
///
/// # Arguments
///
/// * `input` - THe input string to lex.
///
/// # Returns
///
/// A `VecDeque` of lexemes representing identifiers, numbers, and single-character tokens..
pub fn from(input: &str) -> Lexemes {
    let mut lexemes = Lexemes::default();
    let mut chars = input.char_indices().peekable();

    while let Some((start_idx, ch)) = chars.next() {
        if ch.is_whitespace() {
            continue;
        }

        let end_idx = match ch {
            '0'..='9' | '.' => parse_number(start_idx, &mut chars),
            'a'..='z' | 'A'..='Z' | '_' => parse_ident(start_idx, &mut chars),
            _ => start_idx + ch.len_utf8(),
        };

        if start_idx < end_idx {
            lexemes.push(Lexeme::new(&input[start_idx..end_idx], start_idx..end_idx));
        }
    }

    lexemes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lex_texts(input: &str) -> Vec<String> {
        from(input).iter().map(|l| l.text().to_string()).collect()
    }

    #[test]
    fn test_empty_and_whitespace() {
        let lexemes = from("");
        assert!(lexemes.is_empty());

        let lexemes = from("   \t\n  ");
        assert!(lexemes.is_empty());
    }

    #[test]
    fn test_identifiers() {
        // Simple identifier
        assert_eq!(lex_texts("x"), vec!["x"]);
        // Identifier with underscore and digits
        assert_eq!(lex_texts("var_1"), vec!["var_1"]);
        // Consecutive identifiers separated by space
        assert_eq!(lex_texts("a b_c D1"), vec!["a", "b_c", "D1"]);
    }

    #[test]
    fn test_numbers() {
        // Integer
        assert_eq!(lex_texts("123"), vec!["123"]);
        // Decimal
        assert_eq!(lex_texts("3.14"), vec!["3.14"]);
        // Scientific notation
        assert_eq!(lex_texts("1e10"), vec!["1e10"]);
        assert_eq!(lex_texts("2E-3"), vec!["2E-3"]);
        assert_eq!(lex_texts("5.0e+2"), vec!["5.0e+2"]);
        // Mixed with identifiers
        assert_eq!(lex_texts("x1 2.0"), vec!["x1", "2.0"]);
    }

    #[test]
    fn test_imaginary_numbers() {
        // Imaginary unit only
        assert_eq!(lex_texts("i"), vec!["i"]);
        // Imaginary with numeric prefix
        assert_eq!(lex_texts("3i"), vec!["3i"]);
        assert_eq!(lex_texts("3.14i"), vec!["3.14i"]);
        assert_eq!(lex_texts("2e10i"), vec!["2e10i"]);
    }

    #[test]
    fn test_single_char_tokens() {
        assert_eq!(lex_texts("()+-*/^,"), vec!["(", ")", "+", "-", "*", "/", "^", ","]);
    }

    #[test]
    fn test_mixed_expression() {
        let expr = "sin(x) + 3.0i - var_1 / 2e-3";
        let expected = vec!["sin", "(", "x", ")", "+", "3.0i", "-", "var_1", "/", "2e-3"];
        assert_eq!(lex_texts(expr), expected);
    }

    #[test]
    fn test_boundary_values() {
        // Leading zeros
        assert_eq!(lex_texts("0001 0.0"), vec!["0001", "0.0"]);
        // Extremely small/large exponents
        assert_eq!(lex_texts("1e-100 1e+100"), vec!["1e-100", "1e+100"]);
        // Decimal without leading digit
        assert_eq!(lex_texts(".5 0.5"), vec![".5", "0.5"]);
    }

    #[test]
    fn test_invalid_cases() {
        // Unexpected characters are treated as single-char tokens
        assert_eq!(lex_texts("@#%"), vec!["@", "#", "%"]);
        // Mixed invalid + valid
        assert_eq!(lex_texts("x$3i"), vec!["x", "$", "3i"]);
        // Multiple consecutive symbols
        assert_eq!(lex_texts("++--**//"), vec!["+", "+", "-", "-", "*", "*", "/", "/"]);
    }

    #[test]
    fn test_whitespace_sensitivity() {
        assert_eq!(lex_texts("  x   +  3 "), vec!["x", "+", "3"]);
        assert_eq!(lex_texts("\t\na\tb\n"), vec!["a", "b"]);
    }

    #[test]
    fn test_identifier_and_number_boundary() {
        // Identifier immediately followed by number
        assert_eq!(lex_texts("var123"), vec!["var123"]);
        // Number immediately followed by identifier (should split)
        assert_eq!(lex_texts("123abc"), vec!["123", "abc"]);
    }

    #[test]
    fn test_signed_numbers_and_operators() {
        // Single-character operators
        assert_eq!(lex_texts("+ -"), vec!["+", "-"]);
        // Signed numbers
        assert_eq!(lex_texts("+3 -4.5  -2e10 +0.1"), vec!["+", "3", "-", "4.5", "-", "2e10", "+", "0.1"]);
        // Mixed with identifiers
        assert_eq!(lex_texts("x+3 y-2"), vec!["x", "+", "3", "y", "-", "2"]);
        // Edge case: consecutive signs
        assert_eq!(lex_texts("x+-y"), vec!["x", "+", "-", "y"]);
    }
}
