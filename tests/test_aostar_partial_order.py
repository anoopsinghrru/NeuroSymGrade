"""
Unit tests for AO* AND-OR Graph search and Nonlinear Partial-Order Constraint Posting.
"""

import pytest
from src.models.step_frame import StepFrame, RubricStep, RubricStrategy
from src.search_alignment.aostar_rubric_matcher import AOStarRubricMatcher, PartialOrderConstraintPoster


def test_partial_order_constraint_posting_zero_penalty():
    """
    Unit 4 Nonlinear Planning:
    Steps 1A (Normal Force) and 1B (Work-Energy Setup) are independent.
    Step 2 (Solve Velocity) requires BOTH 1A and 1B.
    A student writing [1B, 1A, 2] is a valid topological sort and must receive 0 penalty!
    """
    causal_constraints = {
        1: [],      # Step 1A: Independent
        2: [],      # Step 1B: Independent
        3: [1, 2]   # Step 2: Depends on 1A and 1B
    }

    # Student writes 2 (1B) then 1 (1A) then 3
    student_order = [2, 1, 3]
    penalty = PartialOrderConstraintPoster.compute_ordering_penalty(
        student_step_order=student_order,
        rubric_step_ids=[1, 2, 3],
        causal_constraints=causal_constraints
    )
    assert penalty == 0.0

    # Student writes 3 (Step 2) BEFORE Step 1 (Illegal violation!)
    illegal_order = [3, 1, 2]
    penalty_illegal = PartialOrderConstraintPoster.compute_ordering_penalty(
        student_step_order=illegal_order,
        rubric_step_ids=[1, 2, 3],
        causal_constraints=causal_constraints
    )
    assert penalty_illegal > 0.0


def test_aostar_or_strategy_selection():
    """
    AO* chooses between Strategy 1 (Kinematics) and Strategy 2 (Work-Energy).
    Student solved using Work-Energy. AO* must automatically select Strategy 2 with lowest cost.
    """
    # Strategy 1: Kinematics
    s1 = RubricStrategy(
        strategy_id="strat_kinematics",
        name="Kinematics Method",
        description="Solve using v^2 - u^2 = 2as",
        steps=[
            RubricStep(1, "Kinematics", "v^2 - u^2 = 2*a*s", {}, 20.0, 10.0)
        ]
    )

    # Strategy 2: Work-Energy
    s2 = RubricStrategy(
        strategy_id="strat_energy",
        name="Work Energy Method",
        description="Solve using W = delta KE",
        steps=[
            RubricStep(1, "WorkEnergyTheorem", "W = 0.5*m*v^2 - 0.5*m*u^2", {}, 20.0, 10.0)
        ]
    )

    matcher = AOStarRubricMatcher([s1, s2])

    # Student submission uses Work-Energy
    student_steps = [
        StepFrame(
            step_id=1,
            raw_text="By Work-Energy theorem: W = 0.5*m*v^2 - 0.5*m*u^2",
            invoked_concept="WorkEnergyTheorem",
            raw_equation="W = 0.5*m*v^2 - 0.5*m*u^2",
            is_symbolically_valid=True,
            rubric_residual=0.0
        )
    ]

    best_strat, cost, aligned_map, missing = matcher.align_best_strategy(student_steps)
    assert best_strat.strategy_id == "strat_energy"
    assert cost <= 0.5
    assert len(missing) == 0
