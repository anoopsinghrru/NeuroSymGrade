"""
Tests the opt-in dimensional-analysis / unit-checking feature (Unit 4 ontology):
- OWLGrounder.extract_trailing_unit / check_unit correctness.
- RubricSynthesizer auto-opting a step into unit checking when the master
  solution itself states a unit.
- Pipeline penalty application (missing vs wrong units) and that it is a
  strict no-op for rubric steps that don't opt in (require_units=False,
  the default -- every pre-existing problem).
"""

from src.symbolic_math.owl_grounding import OWLGrounder
from src.search_alignment.rubric_synthesizer import RubricSynthesizer
from src.pipeline import NeuroSymbolicVerificationEngine

BRAKING_WITH_UNITS = (
    "Step 1: Given u=20, a=-4, time to stop is t = (0 - u)/a = (0 - 20)/-4 = 5 s\n"
    "Step 2: Stopping distance is s = u*t + 0.5*a*t^2 = 20*5 + 0.5*-4*25 = 50 m"
)


def test_extract_trailing_unit():
    g = OWLGrounder()
    assert g.extract_trailing_unit("v = u + a*t = 0 + 10*2 = 20") is None
    assert g.extract_trailing_unit("v = u + a*t = 0 + 10*2 = 20 m/s") == "m/s"
    assert g.extract_trailing_unit("W = F * s = 20 * 20 = 400 J") == "J"
    assert g.extract_trailing_unit("t = (0 - 20)/-4 = 5 s") == "s"


def test_check_unit_matches_aliases_and_rejects_wrong_units():
    g = OWLGrounder()
    assert g.check_unit("v", "v = ... = 20 m/s") == (True, "m/s")
    assert g.check_unit("v", "v = ... = 20 ms^-1") == (True, "ms^-1")
    assert g.check_unit("v", "v = ... = 20 cm/s") == (False, "cm/s")
    assert g.check_unit("v", "v = ... = 20") == (False, None)


def test_check_unit_returns_none_for_unregistered_symbol():
    g = OWLGrounder()
    matched, extracted = g.check_unit("not_a_real_symbol", "x = 5")
    assert matched is None and extracted is None


def test_synthesizer_auto_opts_in_when_master_solution_has_units():
    synth = RubricSynthesizer()
    strategy = synth.synthesize_strategy(
        BRAKING_WITH_UNITS, "strat_braking_units", "Braking With Units", total_marks=10.0
    )
    assert all(s.require_units for s in strategy.steps)


def test_synthesizer_does_not_opt_in_without_units_in_master_solution():
    no_units = (
        "Step 1: Given u=20, a=-4, time to stop is t = (0 - u)/a = (0 - 20)/-4 = 5\n"
        "Step 2: Stopping distance is s = u*t + 0.5*a*t^2 = 20*5 + 0.5*-4*25 = 50"
    )
    synth = RubricSynthesizer()
    strategy = synth.synthesize_strategy(no_units, "strat_braking_nounits", "No Units", total_marks=10.0)
    assert all(not s.require_units for s in strategy.steps)


def test_pipeline_full_credit_when_units_correct():
    synth = RubricSynthesizer()
    strategy = synth.synthesize_strategy(
        BRAKING_WITH_UNITS, "strat_braking_units", "Braking With Units", total_marks=10.0
    )
    engine = NeuroSymbolicVerificationEngine()
    summary, _ = engine.verify_submission(
        "UNIT_TEST_CORRECT", BRAKING_WITH_UNITS, problem_id="unused", custom_strategies=[strategy]
    )
    assert all(s.unit_valid is True for s in summary.step_evaluations)
    assert summary.percentage > 85.0


def test_pipeline_penalizes_missing_units():
    missing_units = (
        "Step 1: Time to stop is t = (0 - u)/a = (0 - 20)/-4 = 5\n"
        "Step 2: Stopping distance is s = u*t + 0.5*a*t^2 = 20*5 + 0.5*-4*25 = 50"
    )
    synth = RubricSynthesizer()
    strategy = synth.synthesize_strategy(
        BRAKING_WITH_UNITS, "strat_braking_units", "Braking With Units", total_marks=10.0
    )
    engine = NeuroSymbolicVerificationEngine()

    full_summary, _ = engine.verify_submission(
        "UNIT_TEST_FULL", BRAKING_WITH_UNITS, problem_id="unused", custom_strategies=[strategy]
    )
    missing_summary, _ = engine.verify_submission(
        "UNIT_TEST_MISSING", missing_units, problem_id="unused", custom_strategies=[strategy]
    )

    assert all(s.unit_valid is False and s.unit_provided is None for s in missing_summary.step_evaluations)
    # Missing-unit penalty (10%) must be strictly between "no penalty" and "wrong-unit penalty".
    assert missing_summary.total_marks_awarded < full_summary.total_marks_awarded
    ratio = missing_summary.total_marks_awarded / full_summary.total_marks_awarded
    assert 0.85 < ratio < 0.95


def test_pipeline_penalizes_wrong_units_more_than_missing_units():
    wrong_units = (
        "Step 1: Time to stop is t = (0 - u)/a = (0 - 20)/-4 = 5 minutes\n"
        "Step 2: Stopping distance is s = u*t + 0.5*a*t^2 = 20*5 + 0.5*-4*25 = 50 cm"
    )
    missing_units = (
        "Step 1: Time to stop is t = (0 - u)/a = (0 - 20)/-4 = 5\n"
        "Step 2: Stopping distance is s = u*t + 0.5*a*t^2 = 20*5 + 0.5*-4*25 = 50"
    )
    synth = RubricSynthesizer()
    strategy = synth.synthesize_strategy(
        BRAKING_WITH_UNITS, "strat_braking_units", "Braking With Units", total_marks=10.0
    )
    engine = NeuroSymbolicVerificationEngine()

    wrong_summary, _ = engine.verify_submission(
        "UNIT_TEST_WRONG", wrong_units, problem_id="unused", custom_strategies=[strategy]
    )
    missing_summary, _ = engine.verify_submission(
        "UNIT_TEST_MISSING2", missing_units, problem_id="unused", custom_strategies=[strategy]
    )

    assert all(s.unit_provided is not None and s.unit_valid is False for s in wrong_summary.step_evaluations)
    assert wrong_summary.total_marks_awarded < missing_summary.total_marks_awarded


def test_pipeline_is_a_no_op_for_rubrics_that_never_opt_in():
    """Every pre-existing (non-synthesized) problem has require_units unset (False)."""
    engine = NeuroSymbolicVerificationEngine()
    text = """
    Step 1: By kinematics, final velocity is v = u + a*t = 0 + 10*2 = 20
    Step 2: Average displacement is s = (u + v)/2 * t = (0 + 20)/2 * 2 = 20
    Step 3: Work done by gravity is W = F * s = 20 * 20 = 400
    """
    summary, _ = engine.verify_submission("UNIT_TEST_NOOP", text, problem_id="prob_falling_object")
    for s in summary.step_evaluations:
        assert s.unit_required is False
        assert s.unit_valid is None
    assert summary.total_marks_awarded == 13.74  # unchanged from before this feature existed
