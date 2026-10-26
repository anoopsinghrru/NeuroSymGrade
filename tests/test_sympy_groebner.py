"""
Unit tests for SymPy Verifier, Gröbner Basis ideal membership, and Dual Residuals.
"""

import pytest
import sympy as sp
from src.symbolic_math.sympy_verifier import SymPyVerifier
from src.symbolic_math.owl_grounding import OWLGrounder
from src.models.step_frame import StepFrame, RubricStep


def test_owl_grounding_aliases():
    grounder = OWLGrounder()
    # Test displacement aliases
    assert grounder.canonicalize_symbol("delta_y") == "s"
    assert grounder.canonicalize_symbol("h") == "s"
    assert grounder.canonicalize_symbol("d") == "s"
    # Test initial velocity aliases
    assert grounder.canonicalize_symbol("v_0") == "u"
    assert grounder.canonicalize_symbol("v0") == "u"
    # Test equation rewriting
    eq = "\\Delta y = v_0 * t + 0.5 * g * t^2"
    canon = grounder.canonicalize_equation_string(eq)
    assert "s" in canon
    assert "u" in canon
    assert "a" in canon


def test_algebraic_equivalence():
    verifier = SymPyVerifier()
    # v^2 = u^2 + 2as vs v^2 - u^2 = 2as
    lhs1, rhs1 = verifier.parse_equation("v^2 = u^2 + 2*a*s")
    lhs2, rhs2 = verifier.parse_equation("v^2 - u^2 = 2*a*s")
    assert verifier.is_equivalent(lhs1, rhs1, lhs2, rhs2)


def test_groebner_basis_algebraic_leap():
    """
    Scenario 3:
    Rubric intermediate step 1: F = m*a
    Rubric intermediate step 2: v^2 - u^2 = 2*a*s
    Student combined leap: F = m*(v^2 - u^2)/(2*s)
    """
    verifier = SymPyVerifier()
    
    r1_lhs, r1_rhs = verifier.parse_equation("F = m*a")
    r2_lhs, r2_rhs = verifier.parse_equation("v^2 - u^2 = 2*a*s")
    
    s_lhs, s_rhs = verifier.parse_equation("F = (m*(v^2 - u^2))/(2*s)")
    
    is_entailed = verifier.check_groebner_ideal_membership(
        student_eq=(s_lhs, s_rhs),
        intermediate_rubric_eqs=[(r1_lhs, r1_rhs), (r2_lhs, r2_rhs)]
    )
    assert is_entailed is True


def test_dual_residuals_calculation():
    """
    Scenario 2:
    Step 1 has arithmetic slip: student calculated 25.0 instead of rubric 20.0.
    Formula: s = v * t
    """
    verifier = SymPyVerifier()
    
    # Prior bindings from student: v = 12.5, t = 2.0 (student calculated 25.0, rubric expects v = 10.0 => s = 20.0)
    student_prior_bindings = {"v": 12.5, "t": 2.0}
    eq_lhs, eq_rhs = verifier.parse_equation("s = v * t")
    
    rubric_res, local_res = verifier.compute_dual_residuals(
        student_value=25.0,
        master_expected_value=20.0,
        student_equation=(eq_lhs, eq_rhs),
        target_var="s",
        student_prior_bindings=student_prior_bindings
    )
    
    # Rubric residual: |25.0 - 20.0| = 5.0
    assert rubric_res == 5.0
    # Local residual: |25.0 - (12.5 * 2.0)| = 0.0 -> Perfect local propagation!
    assert local_res == 0.0
