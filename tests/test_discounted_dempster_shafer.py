"""
Unit tests for Discounted Dempster-Shafer Evidence Fusion and Yager's Combination Rule.
"""

import pytest
from src.uncertainty_fuzzy.dempster_shafer import DempsterShaferFusion, EvidenceMass


def test_dictator_paradox_eliminated():
    """
    Test 1:
    Even if Symbolic Verifier proves equation valid, Shafer discounting (alpha_2 = 0.90)
    ensures belief Bel({C}) does NOT collapse into 1.0 when neural confidence is low.
    """
    ds = DempsterShaferFusion(neural_discount=0.85, symbolic_discount=0.90)
    
    # Low neural confidence (0.10) with valid symbolic proof
    bel_c, k, pl_c = ds.evaluate_step_belief(neural_confidence=0.10, is_symbolically_valid=True)
    
    # Bel({C}) must be strictly less than 1.0 and reflective of uncertainty!
    assert bel_c < 0.90
    assert bel_c > 0.0
    # Plausibility must be higher than belief
    assert pl_c >= bel_c


def test_yager_rule_on_persuasive_hallucination():
    """
    Test 2 (Scenario 5 - Persuasive Math Hallucination):
    Student writes fluent, persuasive text (high neural confidence = 0.90),
    but the formula is mathematically invalid (symbolic is_valid = False).
    
    Under Yager's rule:
    - Conflict K must be high (> 0.50).
    - Conflict K is absorbed into uncertainty m(Theta), preventing unearned credit!
    """
    ds = DempsterShaferFusion(neural_discount=0.85, symbolic_discount=0.90)
    
    bel_c, k, pl_c = ds.evaluate_step_belief(neural_confidence=0.90, is_symbolically_valid=False)
    
    # Severe conflict K between neural claiming True and symbolic proving False
    assert k > 0.50
    # Bel({C}) must be near zero
    assert bel_c < 0.15
