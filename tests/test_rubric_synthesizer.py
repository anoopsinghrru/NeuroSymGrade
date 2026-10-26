"""
Tests the RubricSynthesizer (Units 1, 2, 4, 5): compiling a master/reference
derivation into a gradeable RubricStrategy, with no LLM involved.
"""

from src.search_alignment.rubric_synthesizer import RubricSynthesizer
from src.pipeline import NeuroSymbolicVerificationEngine

GOLDEN_MASTER_SOLUTION = """
Step 1: Given u=0, a=10, t=2, velocity is v = u + a*t = 0 + 10*2 = 20
Step 2: Displacement is s = (u + v)/2 * t = (0 + 20)/2 * 2 = 20
Step 3: Since F=20, work done by gravity is W = F * s = 20 * 20 = 400
"""

BRAKING_MASTER_SOLUTION = """
Step 1: Given u=20, a=-4, time to stop is t = (0 - u)/a = (0 - 20)/-4 = 5
Step 2: Stopping distance is s = u*t + 0.5*a*t^2 = 20*5 + 0.5*-4*25 = 50
"""


def test_synthesizer_reproduces_known_rubric_structure():
    synth = RubricSynthesizer()
    strategy = synth.synthesize_strategy(
        GOLDEN_MASTER_SOLUTION, "strat_auto", "Auto-Synthesized Kinematics", total_marks=15.0
    )

    assert len(strategy.steps) == 3
    assert strategy.steps[0].canonical_equation == "v = u + a*t"
    assert strategy.steps[0].expected_output_value == 20.0
    assert strategy.steps[1].canonical_equation == "s = (u + v)/2 * t"
    assert strategy.steps[1].expected_output_value == 20.0
    assert strategy.steps[1].expected_bindings.get("v") == 20.0  # carried forward from step 1
    assert strategy.steps[2].canonical_equation == "W = F * s"
    assert strategy.steps[2].expected_output_value == 400.0

    # Marks are complexity-weighted but must sum exactly to the requested total.
    assert sum(s.max_marks for s in strategy.steps) == 15.0

    # Dependency DAG mirrors the auto-detected causal chain.
    assert strategy.and_dependency_dag == {1: [], 2: [1], 3: [2]}


def test_synthesized_rubric_grades_a_correct_submission_highly():
    synth = RubricSynthesizer()
    strategy = synth.synthesize_strategy(
        GOLDEN_MASTER_SOLUTION, "strat_auto", "Auto-Synthesized Kinematics", total_marks=15.0
    )

    engine = NeuroSymbolicVerificationEngine()
    summary, _ = engine.verify_submission(
        submission_id="SYNTH_GOLDEN",
        raw_submission_text=GOLDEN_MASTER_SOLUTION,
        problem_id="unused",
        custom_strategies=[strategy]
    )

    assert summary.percentage > 85.0
    assert all(s.is_symbolically_valid for s in summary.step_evaluations)


def test_synthesizer_handles_a_brand_new_topic_not_in_domain_rules():
    """Braking distance has no entry anywhere in config/domain_rules.json."""
    synth = RubricSynthesizer()
    strategy = synth.synthesize_strategy(
        BRAKING_MASTER_SOLUTION, "strat_braking", "Braking Distance Approach", total_marks=10.0
    )

    engine = NeuroSymbolicVerificationEngine()
    summary, _ = engine.verify_submission(
        submission_id="SYNTH_BRAKING",
        raw_submission_text=BRAKING_MASTER_SOLUTION,
        problem_id="unused",
        custom_strategies=[strategy]
    )

    assert summary.percentage > 80.0
    assert all(s.is_symbolically_valid for s in summary.step_evaluations)


def test_save_problem_to_config_round_trips(tmp_path):
    synth = RubricSynthesizer()
    problem_entry = synth.synthesize_problem(
        problem_id="prob_test_braking",
        title="Braking Distance",
        description="Test-only synthesized problem",
        strategy_specs=[{
            "strategy_id": "strat_braking",
            "name": "Braking Distance Approach",
            "description": "",
            "master_solution": BRAKING_MASTER_SOLUTION
        }],
        total_marks=10.0
    )

    config_path = tmp_path / "domain_rules_test.json"
    synth.save_problem_to_config(problem_entry, config_path=str(config_path))

    import json
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["problems"][0]["problem_id"] == "prob_test_braking"
    assert data["problems"][0]["strategies"][0]["steps"][0]["canonical_equation"] == "t = (0 - u)/a"
