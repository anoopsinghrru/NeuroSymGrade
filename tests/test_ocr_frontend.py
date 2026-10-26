"""
Tests the OCR perception front-end (Unit 5) against a synthetically rendered
handwriting-style submission image, and confirms the extracted text is
consumable end-to-end by the neuro-symbolic pipeline.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.generate_sample_submission_image import generate, SAMPLE_TEXT
from src.perception_nlp.ocr_frontend import OCRSubmissionReader
from src.pipeline import NeuroSymbolicVerificationEngine


@pytest.fixture(scope="module")
def sample_image(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("ocr_fixture")
    return generate(str(out_dir / "sample_submission.png"))


@pytest.fixture(scope="module")
def ocr_reader():
    try:
        reader = OCRSubmissionReader()
        reader._get_reader()  # Forces model load/download once; skip the module if unavailable.
        return reader
    except Exception as exc:
        pytest.skip(f"EasyOCR model unavailable in this environment: {exc}")


def test_ocr_extracts_three_step_lines(sample_image, ocr_reader):
    text, confidence = ocr_reader.extract_text_with_confidence(sample_image)
    lines = [l for l in text.splitlines() if l.strip()]

    assert len(lines) == len(SAMPLE_TEXT)
    assert confidence > 0.3
    # Core numeric content and step structure must survive OCR + normalization.
    assert "Step 1" in lines[0]
    assert "25" in lines[0]
    assert "Step 2" in lines[1]
    assert "Step 3" in lines[2]
    assert "500" in lines[2]
    # No dollar-sign misreads should remain after normalization.
    assert "$" not in text


def test_ocr_output_is_consumable_by_pipeline(sample_image, ocr_reader):
    text, _ = ocr_reader.extract_text_with_confidence(sample_image)
    engine = NeuroSymbolicVerificationEngine()
    summary, trees = engine.verify_submission(
        submission_id="OCR_TEST",
        raw_submission_text=text,
        problem_id="prob_falling_object"
    )
    assert len(summary.step_evaluations) == 3
    assert summary.max_possible_marks > 0
