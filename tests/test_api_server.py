"""
Tests the FastAPI service layer (api/server.py) via FastAPI's TestClient --
no running server needed. Confirms the API is a thin, correct wrapper over
the same engine/synthesizer/OCR/report components already covered by the
other test modules (no grading logic is duplicated here, just the HTTP seam).
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.server import app
from tests.generate_sample_submission_image import generate

client = TestClient(app)

GOLDEN_TEXT = """
Step 1: By kinematics, final velocity is v = u + a*t = 0 + 10*2 = 20
Step 2: Average displacement is s = (u + v)/2 * t = (0 + 20)/2 * 2 = 20
Step 3: Work done by gravity is W = F * s = 20 * 20 = 400
"""


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "online"
    assert "prob_falling_object" in data["problems_loaded"]


def test_list_problems():
    resp = client.get("/problems")
    assert resp.status_code == 200
    assert "prob_falling_object" in resp.json()["problem_ids"]


def test_grade_text_golden_scenario():
    resp = client.post("/grade/text", json={
        "submission_id": "API_PYTEST_GOLDEN",
        "raw_submission_text": GOLDEN_TEXT,
        "problem_id": "prob_falling_object"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["metadata"]["submission_id"] == "API_PYTEST_GOLDEN"
    assert data["metadata"]["percentage"] > 85.0
    assert len(data["detailed_audit"]) == 3
    assert os.path.exists(data["_persisted_path"])


def test_grade_text_hallucination_flags_review():
    resp = client.post("/grade/text", json={
        "submission_id": "API_PYTEST_HALLUC",
        "raw_submission_text": "Step 1: Clearly by conservation of momentum and energy, "
                                "velocity is v = u / (a * t^2) = 20",
        "problem_id": "prob_falling_object"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["metadata"]["human_review_required_count"] >= 1
    assert data["detailed_audit"][0]["defeasible_status"] == "symbolic_veto"


def test_grade_text_malformed_problem_id_does_not_500():
    resp = client.post("/grade/text", json={
        "submission_id": "API_PYTEST_BADPROBLEM",
        "raw_submission_text": GOLDEN_TEXT,
        "problem_id": "this_problem_does_not_exist"
    })
    # Unknown problem_id falls back to the engine's default rubric rather than erroring.
    assert resp.status_code == 200


def test_synthesize_rubric_without_saving():
    resp = client.post("/rubric/synthesize", json={
        "problem_id": "prob_api_pytest_unsaved",
        "title": "API Test Problem",
        "total_marks": 10.0,
        "save": False,
        "strategies": [{
            "strategy_id": "strat_test",
            "name": "Test Strategy",
            "master_solution": "Step 1: Given u=20, a=-4, time to stop is "
                                "t = (0 - u)/a = (0 - 20)/-4 = 5"
        }]
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["problem_id"] == "prob_api_pytest_unsaved"
    assert data["strategies"][0]["steps"][0]["expected_output_value"] == 5.0
    assert "_saved_to" not in data

    # Since save=False, it must NOT appear in /problems.
    problems = client.get("/problems").json()["problem_ids"]
    assert "prob_api_pytest_unsaved" not in problems


def test_synthesize_rubric_with_saving_is_immediately_gradeable():
    resp = client.post("/rubric/synthesize", json={
        "problem_id": "prob_api_pytest_saved",
        "title": "API Test Problem (Saved)",
        "total_marks": 10.0,
        "save": True,
        "strategies": [{
            "strategy_id": "strat_test",
            "name": "Test Strategy",
            "master_solution": "Step 1: Given u=20, a=-4, time to stop is "
                                "t = (0 - u)/a = (0 - 20)/-4 = 5"
        }]
    })
    assert resp.status_code == 200
    assert "_saved_to" in resp.json()

    problems = client.get("/problems").json()["problem_ids"]
    assert "prob_api_pytest_saved" in problems

    grade_resp = client.post("/grade/text", json={
        "submission_id": "API_PYTEST_SAVED_GRADE",
        "raw_submission_text": "Step 1: Time to stop is t = (0 - u)/a = (0 - 20)/-4 = 5",
        "problem_id": "prob_api_pytest_saved"
    })
    assert grade_resp.status_code == 200
    assert grade_resp.json()["metadata"]["percentage"] > 80.0

    # Clean up: remove the test problem from the real config file.
    import json
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "domain_rules.json")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["problems"] = [p for p in cfg["problems"] if p["problem_id"] != "prob_api_pytest_saved"]
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def test_grade_image_ocr_roundtrip(tmp_path):
    img_path = generate(str(tmp_path / "api_ocr_sample.png"))
    with open(img_path, "rb") as f:
        resp = client.post(
            "/grade/image",
            params={"submission_id": "API_PYTEST_OCR", "problem_id": "prob_falling_object"},
            files={"file": ("sample.png", f, "image/png")}
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["metadata"]["submission_id"] == "API_PYTEST_OCR"
    assert data["metadata"]["ocr_confidence"] is not None
    assert "ocr_extracted_text" in data
    assert len(data["detailed_audit"]) == 3


def test_reports_listing_and_fetch():
    # Grade something first so there's definitely at least one report.
    client.post("/grade/text", json={
        "submission_id": "API_PYTEST_FOR_HISTORY",
        "raw_submission_text": GOLDEN_TEXT,
        "problem_id": "prob_falling_object"
    })

    listing = client.get("/reports")
    assert listing.status_code == 200
    reports = listing.json()["reports"]
    assert len(reports) > 0

    match = next(r for r in reports if r["submission_id"] == "API_PYTEST_FOR_HISTORY")
    detail = client.get(f"/reports/{match['report_id']}")
    assert detail.status_code == 200
    assert detail.json()["metadata"]["submission_id"] == "API_PYTEST_FOR_HISTORY"


def test_report_not_found_returns_404():
    resp = client.get("/reports/this_report_id_does_not_exist")
    assert resp.status_code == 404
