"""
SymPy Symbolic Verification Engine (Unit 2).
Performs algebraic canonical reduction, polynomial ideal membership via Gröbner Bases,
clearing denominators/radicals, and dual residual computation for transitive ECF.
"""

import math
import re
from typing import Dict, List, Optional, Tuple, Any, Union
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor
)

from .owl_grounding import OWLGrounder
from ..models.step_frame import StepFrame, RubricStep


class SymPyVerifier:
    """
    Symbolic Verifier implementing:
    1. Parsing plain-text & LaTeX equations into SymPy expressions.
    2. Canonical algebraic equivalence via simplification.
    3. Gröbner Basis polynomial ideal entailment for skipped intermediate steps.
    4. Denominator & radical clearing safeguard for polynomial rings.
    5. Dual residual computation (RubricResidual vs. LocalPropagationResidual).
    """

    TRANSFORMATIONS = standard_transformations + (
        implicit_multiplication_application,
        convert_xor
    )

    def __init__(self, grounder: Optional[OWLGrounder] = None):
        self.grounder = grounder or OWLGrounder()
        # Protects multi-letter canonical symbols (e.g. "KE", "PE") from being
        # split by implicit multiplication (e.g. "KE" -> K*E) during parsing.
        canonical_targets = set(self.grounder.canonical_map.values())
        self._local_dict = {sym: sp.Symbol(sym) for sym in canonical_targets if sym.isidentifier()}

    def clean_math_string(self, raw_str: str) -> str:
        """Sanitizes text/LaTeX equation strings into standard SymPy syntax."""
        s = raw_str.strip()
        # Remove markdown wrapping $...$ or $$...$$
        s = re.sub(r'^\$+|\$+$', '', s).strip()
        # Replace LaTeX fractions \frac{a}{b} -> ((a)/(b))
        while "\\frac" in s:
            s = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'((\1)/(\2))', s)
        # Replace LaTeX sqrt \sqrt{a} -> sqrt(a)
        s = re.sub(r'\\sqrt\{([^{}]+)\}', r'sqrt(\1)', s)
        # Replace LaTeX cdot / times
        s = s.replace('\\cdot', '*').replace('\\times', '*').replace('×', '*')
        # Replace caret with ** for power
        s = s.replace('^', '**')
        # Canonicalize variable symbols using OWL grounder
        s = self.grounder.canonicalize_equation_string(s)
        return s

    def parse_equation(self, eq_str: str) -> Tuple[Optional[sp.Expr], Optional[sp.Expr]]:
        """Parses an equation 'LHS = RHS' into (LHS, RHS) SymPy expressions."""
        cleaned = self.clean_math_string(eq_str)
        if "=" in cleaned:
            parts = cleaned.split("=", 1)
            lhs_str, rhs_str = parts[0].strip(), parts[1].strip()
        else:
            # Single expression assumed to equal 0
            lhs_str, rhs_str = cleaned, "0"

        try:
            lhs = parse_expr(lhs_str, transformations=self.TRANSFORMATIONS, local_dict=self._local_dict)
            rhs = parse_expr(rhs_str, transformations=self.TRANSFORMATIONS, local_dict=self._local_dict)
            return lhs, rhs
        except Exception:
            return None, None

    def to_polynomial_numerator(self, expr: sp.Expr) -> sp.Poly:
        """
        Safeguard 1: Clears denominators and radical exponents to convert rational/radical
        expressions into purely polynomial representations over an integer/rational ring.
        """
        # Step 1: Simplify expression
        sim = sp.simplify(expr)
        
        # Step 2: Clear fractions / extract numerator
        num, den = sp.fraction(sim)
        
        # Step 3: Handle square roots if any remain in numerator
        # If expression is of form A - sqrt(B) = 0 => A^2 - B = 0
        sqrt_terms = [t for t in num.atoms(sp.Pow) if t.exp == sp.Rational(1, 2)]
        if sqrt_terms:
            # Square-elimination heuristic for single radical
            radical = sqrt_terms[0]
            rest = sp.collect(num, radical)
            # Replace radical^2 with its base
            base = radical.base
            poly_candidate = sp.simplify(num.subs(radical, sp.Symbol('_RAD_')))
            # If radical can be isolated: rest_without_rad = radical => (rest)^2 - base = 0
            rad_coeff = poly_candidate.coeff(sp.Symbol('_RAD_'))
            const_term = poly_candidate.subs(sp.Symbol('_RAD_'), 0)
            if rad_coeff != 0:
                cleared = sp.simplify(const_term**2 - (rad_coeff**2) * base)
                num = sp.fraction(cleared)[0]

        return sp.Poly(num, *num.free_symbols)

    def is_equivalent(self, lhs1: sp.Expr, rhs1: sp.Expr, lhs2: sp.Expr, rhs2: sp.Expr) -> bool:
        """
        Tests whether Eq1: lhs1 = rhs1 is algebraically equivalent to Eq2: lhs2 = rhs2.
        Checks:
        1. Direct difference: simplify((lhs1 - rhs1) - (lhs2 - rhs2)) == 0
        2. Proportional difference: simplify((lhs1 - rhs1) / (lhs2 - rhs2)) is a non-zero scalar
        3. Target variable solver check (e.g. v = sqrt(...) vs v^2 = ...)
        """
        diff1 = sp.simplify(lhs1 - rhs1)
        diff2 = sp.simplify(lhs2 - rhs2)

        # Direct cancellation
        if sp.simplify(diff1 - diff2) == 0 or sp.simplify(diff1 + diff2) == 0:
            return True

        # Ratio test (proportional constant multiple)
        try:
            ratio = sp.simplify(diff1 / diff2)
            if ratio.is_number and ratio != 0:
                return True
        except Exception:
            pass

        # Solver equivalence test on shared free symbols
        common_symbols = list(diff1.free_symbols.intersection(diff2.free_symbols))
        for sym in common_symbols:
            try:
                sols1 = sp.solve(diff1, sym)
                sols2 = sp.solve(diff2, sym)
                if sols1 and sols2:
                    # Check if sets of non-negative / real solutions match
                    matches = 0
                    for s1 in sols1:
                        for s2 in sols2:
                            if sp.simplify(s1 - s2) == 0 or sp.simplify(s1**2 - s2**2) == 0:
                                matches += 1
                    if matches > 0:
                        return True
            except Exception:
                continue

        return False

    def check_groebner_ideal_membership(
        self,
        student_eq: Tuple[sp.Expr, sp.Expr],
        intermediate_rubric_eqs: List[Tuple[sp.Expr, sp.Expr]]
    ) -> bool:
        """
        Uses Gröbner Bases (sympy.groebner) to test whether a combined algebraic leap
        belongs to the polynomial ideal generated by skipped rubric intermediate steps.
        
        Example:
        Rubric Step 1: F = m*a
        Rubric Step 2: v^2 - u^2 = 2*a*s
        Student Leap: F = m*(v^2 - u^2)/(2*s)
        Ideal: < F - m*a, v^2 - u^2 - 2*a*s >
        Student Poly: F*2*s - m*(v^2 - u^2)
        """
        try:
            # Collect all free symbols across all equations
            all_symbols: Set[sp.Symbol] = set()
            ideal_polys = []

            for lhs, rhs in intermediate_rubric_eqs:
                diff = sp.simplify(lhs - rhs)
                poly = self.to_polynomial_numerator(diff)
                ideal_polys.append(poly.as_expr())
                all_symbols.update(poly.free_symbols)

            # Process student equation
            s_diff = sp.simplify(student_eq[0] - student_eq[1])
            s_poly = self.to_polynomial_numerator(s_diff)
            all_symbols.update(s_poly.free_symbols)

            if not ideal_polys:
                return False

            gens = list(all_symbols)
            # Compute Gröbner basis with lexicographic / grevlex ordering
            gb = sp.groebner(ideal_polys, *gens, order='lex')
            
            # Reduce student polynomial with respect to Gröbner basis
            _, rem = sp.reduced(s_poly.as_expr(), gb, *gens, order='lex')
            return sp.simplify(rem) == 0
        except Exception:
            return False

    def evaluate_expression_numerically(
        self,
        expr_str_or_tuple: Union[str, Tuple[sp.Expr, sp.Expr]],
        target_variable: str,
        bindings: Dict[str, float]
    ) -> Optional[float]:
        """
        Substitutes given numerical bindings into an equation and solves for target_variable.
        Used to compute local step evaluations.
        """
        try:
            if isinstance(expr_str_or_tuple, str):
                lhs, rhs = self.parse_equation(expr_str_or_tuple)
            else:
                lhs, rhs = expr_str_or_tuple

            if lhs is None or rhs is None:
                return None

            diff = sp.simplify(lhs - rhs)
            canonical_bindings = self.grounder.canonicalize_bindings(bindings)
            canonical_target = self.grounder.canonicalize_symbol(target_variable)
            target_sym = sp.Symbol(canonical_target)

            # Substitute bindings (excluding target symbol)
            sub_dict = {
                sp.Symbol(k): v for k, v in canonical_bindings.items()
                if k != canonical_target
            }
            substituted = diff.subs(sub_dict)

            # Solve for target
            solutions = sp.solve(substituted, target_sym)
            if not solutions:
                # If target is already on one side and expr on the other
                val = float(substituted.evalf())
                return val

            # Filter for plausible physical solution (prefer non-negative / real)
            real_sols = [float(s.evalf()) for s in solutions if s.is_real]
            if real_sols:
                # If multiple, prefer positive
                pos_sols = [s for s in real_sols if s >= 0]
                return pos_sols[0] if pos_sols else real_sols[0]

            return float(solutions[0].evalf())
        except Exception:
            return None

    def compute_dual_residuals(
        self,
        student_value: Optional[float],
        master_expected_value: float,
        student_equation: Tuple[Optional[sp.Expr], Optional[sp.Expr]],
        target_var: str,
        student_prior_bindings: Dict[str, float]
    ) -> Tuple[float, float]:
        """
        Computes the two critical residuals for Transitive ECF:
        1. RubricResidual: |V_student - V_master_rubric|
        2. LocalPropagationResidual: |V_student - Eval(Formula_student, student_prior_bindings)|
        """
        if student_value is None:
            return 999.0, 999.0

        rubric_residual = abs(student_value - master_expected_value)

        # Compute what student's own formula yields when using student's prior bindings
        expected_from_student_bindings = None
        if student_equation[0] is not None and student_equation[1] is not None:
            expected_from_student_bindings = self.evaluate_expression_numerically(
                student_equation, target_var, student_prior_bindings
            )

        if expected_from_student_bindings is not None:
            local_residual = abs(student_value - expected_from_student_bindings)
        else:
            # If evaluation failed or formula invalid, local residual equals rubric residual
            local_residual = rubric_residual

        return round(rubric_residual, 4), round(local_residual, 4)

    def verify_step(
        self,
        step_frame: StepFrame,
        rubric_step: RubricStep,
        skipped_intermediate_rubric_steps: Optional[List[RubricStep]] = None,
        accumulated_student_bindings: Optional[Dict[str, float]] = None
    ) -> StepFrame:
        """
        End-to-end verification of a single derivation step against its corresponding rubric milestone:
        - Symbol canonicalization
        - Direct algebraic equivalence OR Gröbner basis ideal membership
        - Dual residual computation
        """
        if accumulated_student_bindings is None:
            accumulated_student_bindings = {}

        # 1. Parse student equation if present
        if step_frame.raw_equation:
            lhs, rhs = self.parse_equation(step_frame.raw_equation)
            step_frame.canonical_lhs = lhs
            step_frame.canonical_rhs = rhs
        
        # 2. Parse rubric equation
        r_lhs, r_rhs = self.parse_equation(rubric_step.canonical_equation)

        # 3. Test Symbolic Equivalence
        is_valid = False
        is_leap = False

        if step_frame.canonical_lhs is not None and step_frame.canonical_rhs is not None and r_lhs is not None and r_rhs is not None:
            # Test direct equivalence
            if self.is_equivalent(step_frame.canonical_lhs, step_frame.canonical_rhs, r_lhs, r_rhs):
                is_valid = True
            # Test Gröbner Basis ideal membership for combined skipped steps
            elif skipped_intermediate_rubric_steps:
                intermediate_pairs = []
                for s in skipped_intermediate_rubric_steps:
                    p_lhs, p_rhs = self.parse_equation(s.canonical_equation)
                    if p_lhs is not None and p_rhs is not None:
                        intermediate_pairs.append((p_lhs, p_rhs))
                intermediate_pairs.append((r_lhs, r_rhs))

                if self.check_groebner_ideal_membership(
                    (step_frame.canonical_lhs, step_frame.canonical_rhs),
                    intermediate_pairs
                ):
                    is_valid = True
                    is_leap = True
                    step_frame.skipped_intermediate_steps = [s.step_id for s in skipped_intermediate_rubric_steps]

        step_frame.is_symbolically_valid = is_valid
        step_frame.is_algebraic_leap = is_leap

        # 4. Target variable extraction (infer from LHS of equation)
        target_var = "v"
        if r_lhs is not None and len(r_lhs.free_symbols) >= 1:
            target_var = str(list(r_lhs.free_symbols)[0])
        elif step_frame.canonical_lhs is not None and len(step_frame.canonical_lhs.free_symbols) >= 1:
            target_var = str(list(step_frame.canonical_lhs.free_symbols)[0])

        # 5. Compute Dual Residuals
        rub_res, loc_res = self.compute_dual_residuals(
            student_value=step_frame.evaluated_value,
            master_expected_value=rubric_step.expected_output_value,
            student_equation=(step_frame.canonical_lhs, step_frame.canonical_rhs),
            target_var=target_var,
            student_prior_bindings=accumulated_student_bindings
        )
        step_frame.rubric_residual = rub_res
        step_frame.local_residual = loc_res

        return step_frame
