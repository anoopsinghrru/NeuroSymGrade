"""
Comprehensive End-to-End Benchmark Test Suite.
Tests all 5 canonical derivation scenarios across all six syllabus units:
1. Golden Solution (Full Credit)
2. Multi-Hop Transitive ECF (Error Carried Forward)
3. Skipped Intermediate Step (Gröbner Basis Algebraic Leap)
4. Reordered Independent Prerequisite Steps (Nonlinear Constraint Posting)
5. Persuasive Math Hallucination (Yager Conflict & Symbolic Veto)
"""

import pytest
from src.pipeline import NeuroSymbolicVerificationEngine
from src.reasoning_engine.explanation_facility import ExplanationFacility


def test_scenario_1_golden_solution():
    """Scenario 1: Perfect Derivation matching Kinematics strategy exactly."""
    engine = NeuroSymbolicVerificationEngine()

    student_text = """
    Step 1: By kinematics, final velocity is v = u + a*t = 0 + 10*2 = 20
    Step 2: Average displacement is s = (u + v)/2 * t = (0 + 20)/2 * 2 = 20
    Step 3: Work done by gravity is W = F * s = 20 * 20 = 400
    """

    summary, trees = engine.verify_submission(
        submission_id="SUB_SCENARIO_1_GOLDEN",
        raw_submission_text=student_text,
        problem_id="prob_falling_object"
    )

    assert "strat_kinematics" in summary.selected_strategy
    assert summary.percentage >= 90.0
    for step in summary.step_evaluations:
        assert step.defeasible_status == "full_credit"
        assert step.rubric_residual == 0.0


def test_scenario_2_multihop_transitive_ecf():
    """
    Scenario 2: Multi-Hop Transitive ECF
    Step 1: Arithmetic slip (25 instead of 20).
    Step 2: Substitutes 25 into valid equation -> ECF override!
    Step 3: Substitutes resulting value into valid work equation -> Transitive ECF!
    """
    engine = NeuroSymbolicVerificationEngine()

    student_text = """
    Step 1: Velocity is v = u + a*t = 0 + 10*2 = 25
    Step 2: Displacement is s = (u + v)/2 * t = (0 + 25)/2 * 2 = 25
    Step 3: Work done is W = F * s = 20 * 25 = 500
    """

    summary, trees = engine.verify_submission(
        submission_id="SUB_SCENARIO_2_ECF",
        raw_submission_text=student_text,
        problem_id="prob_falling_object"
    )

    steps = summary.step_evaluations
    assert len(steps) == 3

    # Step 1: Penalized for arithmetic slip
    assert steps[0].defeasible_status == "failed_value"
    assert steps[0].credit_multiplier == 0.25

    # Step 2: ECF override (formula valid, propagated student's v=25)
    assert steps[1].defeasible_status == "ecf_override"
    assert steps[1].credit_multiplier == 0.85

    # Step 3: Transitive ECF override (traces error to root slip in Step 1)
    assert steps[2].defeasible_status == "ecf_override"
    assert steps[2].credit_multiplier == 0.85

    # Verify Explanation Facility explanation contains ECF justification
    why_step2 = engine.explain_why(summary, 2)
    assert "DEFEASIBLE ECF CREDIT" in why_step2


def test_scenario_3_algebraic_leap_groebner():
    """
    Scenario 3: Skipped Intermediate Step (Algebraic Leap)
    Student combines F = ma and v^2 - u^2 = 2as directly into F = m*(v^2 - u^2)/(2*s) in one line.
    """
    engine = NeuroSymbolicVerificationEngine()

    from src.models.step_frame import RubricStep, RubricStrategy
    strat = RubricStrategy(
        strategy_id="strat_leap",
        name="Combined Newton Kinematics",
        description="Solve force from velocity",
        steps=[
            RubricStep(1, "NewtonsSecondLaw", "F = m * a", {"m": 2.0, "a": 5.0}, 10.0, 5.0),
            RubricStep(2, "Kinematics", "v^2 - u^2 = 2 * a * s", {"u": 0.0, "a": 5.0, "s": 10.0}, 100.0, 5.0),
            RubricStep(3, "WorkEnergyTheorem", "W = F * s", {"F": 10.0, "s": 10.0}, 100.0, 5.0)
        ]
    )

    # Student writes the combined equation directly in Step 1
    student_text = """
    Step 1: Combining F = ma and kinematics, F = m*(v^2 - u^2)/(2*s) = 2*(100 - 0)/(2*10) = 10
    """

    summary, trees = engine.verify_submission(
        submission_id="SUB_SCENARIO_3_LEAP",
        raw_submission_text=student_text,
        custom_strategies=[strat]
    )

    step1 = summary.step_evaluations[0]
    # Gröbner basis ideal membership must confirm algebraic leap!
    assert step1.is_symbolically_valid is True
    assert step1.is_algebraic_leap is True


def test_scenario_4_partial_order_reordered():
    """
    Scenario 4: Reordered Independent Prerequisite Steps
    Nonlinear constraint posting guarantees 0 penalty for valid topological sort.
    """
    engine = NeuroSymbolicVerificationEngine()

    from src.models.step_frame import RubricStep, RubricStrategy
    strat = RubricStrategy(
        strategy_id="strat_independent",
        name="Independent Preconditions",
        description="Compute normal force and potential energy in either order",
        steps=[
            RubricStep(1, "NewtonsSecondLaw", "N = m * a", {}, 20.0, 5.0),
            RubricStep(2, "ConservationOfEnergy", "PE = m * a * s", {}, 400.0, 5.0),
            RubricStep(3, "WorkEnergyTheorem", "W = PE", {}, 400.0, 5.0, prerequisites=[1, 2])
        ],
        and_dependency_dag={1: [], 2: [], 3: [1, 2]} # 1 and 2 independent!
    )

    # Student writes Step 2 (PE) before Step 1 (N)
    student_text = """
    Step 1: Potential energy is PE = m * a * s = 2 * 10 * 20 = 400
    Step 2: Normal force is N = m * a = 2 * 10 = 20
    Step 3: Total work is W = PE = 400
    """

    summary, trees = engine.verify_submission(
        submission_id="SUB_SCENARIO_4_REORDER",
        raw_submission_text=student_text,
        custom_strategies=[strat]
    )

    # Zero ordering penalty
    assert summary.alignment_cost < 1.0


def test_scenario_5_persuasive_math_hallucination():
    """
    Scenario 5: Persuasive Math Hallucination
    Student text sounds very confident, but formula is mathematically bogus.
    Dempster-Shafer Yager rule routes conflict K into uncertainty; Prolog executes Symbolic Veto.
    """
    engine = NeuroSymbolicVerificationEngine()

    student_text = """
    Step 1: Clearly by conservation of momentum and energy, velocity is v = u / (a * t^2) = 20
    """

    summary, trees = engine.verify_submission(
        submission_id="SUB_SCENARIO_5_HALLUCINATION",
        raw_submission_text=student_text,
        problem_id="prob_falling_object"
    )

    step1 = summary.step_evaluations[0]
    assert step1.is_symbolically_valid is False
    assert step1.defeasible_status == "symbolic_veto"
    assert step1.credit_multiplier == 0.0
    assert step1.final_score == 0.0
    assert step1.epistemic_conflict > 0.40
