"""
Tests the sentence-embedding fallback classifier (Unit 5 neural upgrade) that
replaces the old flat "ScientificLaw"/0.50 default whenever a step's wording
matches no keyword in CONCEPT_SIGNATURES, and proves the fallback is a strict
no-op for any text that already has a keyword match (zero regression risk).
"""

from src.perception_nlp.neural_slot_extractor import NeuralSlotExtractor

NO_KEYWORD_MATCH_TEXT = "Stopping distance is s = u*t + 0.5*a*t^2 = 20*5 + 0.5*-4*25 = 50"
ENERGY_TEXT_NO_KEYWORDS = "Kinetic energy is KE = PE = 100"
CLEAR_KEYWORD_TEXT = "By kinematics, final velocity is v = u + a*t = 0 + 10*2 = 20"


def test_fallback_classifies_previously_unclassifiable_text():
    ex = NeuralSlotExtractor()
    concept, confidence = ex.predict_concept_with_confidence(NO_KEYWORD_MATCH_TEXT)
    assert concept == "Kinematics"
    assert 0.5 < confidence <= 0.90


def test_fallback_classifies_energy_text_correctly():
    ex = NeuralSlotExtractor()
    concept, confidence = ex.predict_concept_with_confidence(ENERGY_TEXT_NO_KEYWORDS)
    assert concept == "ConservationOfEnergy"


def test_fallback_disabled_reverts_to_scientific_law_default():
    ex = NeuralSlotExtractor(use_semantic_fallback=False)
    concept, confidence = ex.predict_concept_with_confidence(NO_KEYWORD_MATCH_TEXT)
    assert concept == "ScientificLaw"
    assert confidence == 0.50


def test_keyword_path_is_unaffected_by_fallback_toggle():
    """Any text with an existing keyword match must classify identically
    whether or not the semantic fallback is enabled -- proves zero behavior
    change for every pre-existing benchmark scenario."""
    with_fallback = NeuralSlotExtractor(use_semantic_fallback=True)
    without_fallback = NeuralSlotExtractor(use_semantic_fallback=False)

    result_with = with_fallback.predict_concept_with_confidence(CLEAR_KEYWORD_TEXT)
    result_without = without_fallback.predict_concept_with_confidence(CLEAR_KEYWORD_TEXT)

    assert result_with == result_without
    assert result_with[0] == "Kinematics"


def test_all_five_benchmark_scenarios_identical_with_fallback_toggled():
    """End-to-end proof: disabling the fallback (old behavior) produces the
    exact same grading result as leaving it enabled (new behavior), for
    every documented CLI benchmark scenario."""
    import main as m
    from src.pipeline import NeuroSymbolicVerificationEngine

    for key in ["1", "2", "3", "4", "5"]:
        _, text = m.BENCHMARK_DEMOS[key]
        problem_id = m.SCENARIO_PROBLEMS[key]

        engine_with = NeuroSymbolicVerificationEngine()
        engine_without = NeuroSymbolicVerificationEngine()
        engine_without.extractor.use_semantic_fallback = False

        summary_with, _ = engine_with.verify_submission(f"FB_{key}", text, problem_id=problem_id)
        summary_without, _ = engine_without.verify_submission(f"NOFB_{key}", text, problem_id=problem_id)

        assert summary_with.total_marks_awarded == summary_without.total_marks_awarded
        assert summary_with.percentage == summary_without.percentage
