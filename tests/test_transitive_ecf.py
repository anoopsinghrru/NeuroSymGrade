"""
Unit tests for Transitive ECF, Defeasible Non-Monotonic Logic, and Safeguards 2 & 3.
"""

import pytest
from src.models.step_frame import StepFrame, RubricStep
from src.reasoning_engine.prolog_bridge import DefeasiblePrologEngine
from src.reasoning_engine.explanation_facility import ExplanationFacility


def test_three_hop_transitive_ecf():
    """
    Scenario 2 (Multi-Hop Transitive ECF):
    Step 1: Arithmetic slip (LocalRes = 5.0, RubricRes = 5.0) -> root_arithmetic_slip!
    Step 2: Correctly uses Step 1's value in valid formula (LocalRes = 0.0, RubricRes = 10.0) -> ECF!
    Step 3: Correctly uses Step 2's value in valid formula (LocalRes = 0.0, RubricRes = 15.0) -> Transitive ECF!
    """
    engine = DefeasiblePrologEngine()
    
    # Assert Rubric
    engine.assert_rubric_step(RubricStep(1, "Kinematics", "v = u + a*t", {}, 20.0, 5.0))
    engine.assert_rubric_step(RubricStep(2, "Kinematics", "s = v * t", {}, 40.0, 5.0, prerequisites=[1]))
    engine.assert_rubric_step(RubricStep(3, "Work", "W = F * s", {}, 400.0, 5.0, prerequisites=[2]))

    # Step 1: Student made an arithmetic slip (calculated 25 instead of 20)
    s1 = StepFrame(
        step_id=1,
        raw_text="v = u + at = 0 + 10*2 = 25",
        invoked_concept="Kinematics",
        is_symbolically_valid=True,
        evaluated_value=25.0,
        rubric_residual=5.0,
        local_residual=5.0,  # Fails local calculation!
        depends_on=[]
    )
    engine.assert_student_step(s1)

    status1, mult1, exp1, tree1 = engine.evaluate_step(1)
    assert status1 == "failed_value"
    assert mult1 == 0.25

    # Step 2: Student computes s = v * t using their v=25 => s = 50 (rubric expects 40)
    s2 = StepFrame(
        step_id=2,
        raw_text="s = v * t = 25 * 2 = 50",
        invoked_concept="Kinematics",
        is_symbolically_valid=True,
        evaluated_value=50.0,
        rubric_residual=10.0,
        local_residual=0.0,  # Perfect local propagation!
        depends_on=[1]
    )
    engine.assert_student_step(s2)

    status2, mult2, exp2, tree2 = engine.evaluate_step(2)
    assert status2 == "ecf_override"
    assert mult2 == 0.85

    # Step 3: Student computes W = F * s using their s=50 => W = 500 (rubric expects 400)
    # Direct parent is Step 2. Transitive root is Step 1.
    s3 = StepFrame(
        step_id=3,
        raw_text="W = F * s = 10 * 50 = 500",
        invoked_concept="Work",
        is_symbolically_valid=True,
        evaluated_value=500.0,
        rubric_residual=100.0,
        local_residual=0.0,  # Perfect local propagation!
        depends_on=[2]
    )
    engine.assert_student_step(s3)

    status3, mult3, exp3, tree3 = engine.evaluate_step(3)
    # Transitive ECF must succeed for Step 3!
    assert status3 == "ecf_override"
    assert mult3 == 0.85

    # Test Explanation Facility
    ascii_tree = ExplanationFacility.render_proof_tree_ascii(tree3)
    assert "exception_ecf(Step_3)" in ascii_tree
    assert "Step_1" in ascii_tree


def test_safeguard_2_symbolic_veto():
    """
    Safeguard 2: Catch-all clause for invalid formulas (Scenario 5 - Persuasive Math Hallucination).
    """
    engine = DefeasiblePrologEngine()
    engine.assert_rubric_step(RubricStep(1, "Kinematics", "v = u + a*t", {}, 20.0, 5.0))

    s_bogus = StepFrame(
        step_id=1,
        raw_text="Clearly by momentum, v = u / (a * t) = 20",
        invoked_concept="Kinematics",
        is_symbolically_valid=False,  # Formula invalid!
        evaluated_value=20.0,
        rubric_residual=0.0,
        local_residual=0.0
    )
    engine.assert_student_step(s_bogus)

    status, mult, exp, tree = engine.evaluate_step(1)
    assert status == "symbolic_veto"
    assert mult == 0.0
    assert "Symbolic equation is algebraically invalid" in exp


def test_safeguard_3_loop_free_dag():
    """
    Safeguard 3: Cycles in student step dependencies are prevented from causing infinite recursion.
    """
    engine = DefeasiblePrologEngine()
    
    # Assert a step attempting an illegal backwards dependency
    s1 = StepFrame(step_id=1, raw_text="Step 1", is_symbolically_valid=True, depends_on=[2]) # 2 > 1 (illegal)
    s2 = StepFrame(step_id=2, raw_text="Step 2", is_symbolically_valid=True, depends_on=[1])
    
    engine.assert_student_step(s1)
    engine.assert_student_step(s2)
    
    # Check that Step 1 rejected dependency on 2 because 2 >= 1
    assert 2 not in engine.dependencies[1]
    # Check transitive dependency safely returns False without recursion crash
    assert not engine.depends_transitively(1, 2)
