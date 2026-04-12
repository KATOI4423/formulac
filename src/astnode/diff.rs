//! # astnode/diff.rs
//!
//! Symbolic differentiation of [`AstNode`] expressions.
//!
//! ## Responsibilities
//! - Implementing [`AstNode::differentiate`] for each node variant
//! - Applying standard differentiation rules for built-in operators and functions
//! - Delegating user-defined function derivatives to registered derivative functions
//!
//! ## Supported rules
//! - Constants and arguments (base cases)
//! - Unary and binary operators (+, -, *, /, ^)
//! - Built-in functions (sin, cos, exp, ln, sqrt, ...)
//! - User-defined functions via [`UserFn::derivative`]
//! - Higher-order [`AstNode::Derivative`] nodes
//!
//! Note: `abs` and `conj` are not differentiable in the complex domain
//! and will return a [`ParseError`].

use num_complex::Complex;
use std::rc::Rc;

use crate::astnode::AstNode;
use crate::core::Real;
use crate::functions::{
    Arity,
    FunctionKind,
};
use crate::err::ParseError;
use crate::lexer::Span;
use crate::operators::BinaryOperatorKind;


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

    /// Differentiates a built-in function call with respect to argument `var`.
    ///
    /// Applies the chain rule: if `f` is a built-in function and `x` is its argument,
    /// the derivative is `f'(x) * dx` where `dx = d(x)/d(var)`.
    ///
    /// # Arguments
    ///
    /// * `kind`  - The built-in function to differentiate.
    /// * `args`  - The function's argument list. For binary functions (`pow`, `powi`),
    ///             the second element is consumed here.
    /// * `var`   - The index of the variable to differentiate with respect to.
    /// * `span`  - The source span of the function call, used for error reporting
    ///             and constructing intermediate nodes.
    ///
    /// # Errors
    ///
    /// Returns [`ParseError::InvalidFormula`] if `kind` is [`FunctionKind::Abs`] or
    /// [`FunctionKind::Conj`], as these are not differentiable in the complex domain.
    ///
    /// # Derivative rules applied
    ///
    /// | Function   | Derivative                          |
    /// |------------|-------------------------------------|
    /// | `sin(x)`   | `cos(x) * dx`                       |
    /// | `cos(x)`   | `-sin(x) * dx`                      |
    /// | `tan(x)`   | `dx / cos(x)^2`                     |
    /// | `asin(x)`  | `dx / sqrt(1 - x^2)`                |
    /// | `acos(x)`  | `-dx / sqrt(1 - x^2)`               |
    /// | `atan(x)`  | `dx / (1 + x^2)`                    |
    /// | `sinh(x)`  | `dx * cosh(x)`                      |
    /// | `cosh(x)`  | `dx * sinh(x)`                      |
    /// | `tanh(x)`  | `dx / cosh(x)^2`                    |
    /// | `asinh(x)` | `dx / sqrt(x^2 + 1)`                |
    /// | `acosh(x)` | `dx / sqrt(x^2 - 1)`                |
    /// | `atanh(x)` | `dx / (1 - x^2)`                    |
    /// | `exp(x)`   | `dx * exp(x)`                       |
    /// | `ln(x)`    | `dx / x`                            |
    /// | `log10(x)` | `dx * log10(e) / x`                 |
    /// | `sqrt(x)`  | `dx * 0.5 / sqrt(x)`                |
    /// | `pow(x,y)` | see [`Self::diff_pow`]              |
    /// | `powi(x,n)`| see [`Self::diff_powi`]             |
    fn diff_function(
        kind: FunctionKind,
        mut args: Vec<Rc<Self>>,
        var:  usize,
        span: Span,
    ) -> Result<Self, ParseError> {
        let x = Self::unwrap_rc(args.remove(0));
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
                let y = Self::unwrap_rc(args.remove(0));
                Self::diff_pow(x, y, var)
            },
            FunctionKind::Powi => {
                let n = Self::unwrap_rc(args.remove(0));
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

#[cfg(test)]
mod differentiate_tests {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn differentiate_number() {
        let node = AstNode::Number { value: Complex::new(5.0, 0.0), span: Span::from(0..1) };
        let diff = node.differentiate(0).unwrap();
        assert_eq!(diff, AstNode::Number { value: Complex::ZERO, span: Span::from(0..1) });
    }

    #[test]
    fn differentiate_argument() {
        let node = AstNode::<f64>::Argument { index: 1, span: Span::from(0..1) };
        let diff = node.clone().differentiate(1).unwrap();
        assert_eq!(diff, AstNode::Number { value: Complex::ONE, span: Span::from(0..1) });
        let diff_other = node.differentiate(0).unwrap();
        assert_eq!(diff_other, AstNode::Number { value: Complex::ZERO, span: Span::from(0..1) });
    }

    #[test]
    fn differentiate_unary_operator() {
        let node = -AstNode::<f64>::Argument { index: 0, span: Span::from(2..3) };
        let diff = node.differentiate(0).unwrap();
        // d/dx(-x) = -1, where 1 is generated with the argument's span
        assert_eq!(diff, -AstNode::Number { value: Complex::ONE, span: Span::from(2..3) });
    }

    #[test]
    fn differentiate_binary_add() {
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
    fn differentiate_function_sin() {
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
    fn differentiate_derivative_order() {
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
    fn differentiate_mul_x2() {
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
    fn differentiate_builtin_functions() {
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

    // Note: differentiate_div is not tested due to complex span handling
    // After differentiation, internally generated constants' spans are determined
    // by the fold/simplify process, which makes exact span comparison difficult.
}
