"""
Tests audit-report persistence and human-review flagging (Units 3 & 6).
"""

import json
import os

from src.pipeline import NeuroSymbolicVerificationEngine
from src.reasoning_engine.audit_report import (
    build_audit_report, persist_audit_report, list_audit_reports, load_audit_report
)


def test_build_audit_report_flags_high_conflict_steps():
    engine = NeuroSymbolicVerificationEngine()
    # Scenario 5-style hallucination: high DS conflict -> symbolic veto.
    text = "Step 1: Clearly by conservation of momentum and energy, velocity is v = u / (a * t^2) = 20"
    summary, trees = engine.verify_submission("AUDIT_TEST_HALLUC", text, problem_id="prob_falling_object")

    report = build_audit_report(summary, trees, problem_id="prob_falling_object")
    assert report["metadata"]["submission_id"] == "AUDIT_TEST_HALLUC"
    assert report["metadata"]["human_review_required_count"] >= 1
    assert report["detailed_audit"][0]["human_review_flag"] is True
    assert report["detailed_audit"][0]["review_reason"] is not None


def test_persist_and_list_and_load_round_trip(tmp_path):
    engine = NeuroSymbolicVerificationEngine()
    text = """
    Step 1: By kinematics, final velocity is v = u + a*t = 0 + 10*2 = 20
    Step 2: Average displacement is s = (u + v)/2 * t = (0 + 20)/2 * 2 = 20
    Step 3: Work done by gravity is W = F * s = 20 * 20 = 400
    """
    summary, trees = engine.verify_submission("AUDIT_TEST_GOLDEN", text, problem_id="prob_falling_object")
    report = build_audit_report(summary, trees, problem_id="prob_falling_object", ocr_confidence=0.95)

    out_dir = str(tmp_path / "reports")
    path = persist_audit_report(report, output_dir=out_dir)
    assert os.path.exists(path)

    listed = list_audit_reports(out_dir)
    assert len(listed) == 1
    assert listed[0]["submission_id"] == "AUDIT_TEST_GOLDEN"

    loaded = load_audit_report(report["metadata"]["report_id"], output_dir=out_dir)
    assert loaded is not None
    assert loaded["metadata"]["submission_id"] == "AUDIT_TEST_GOLDEN"
    assert len(loaded["detailed_audit"]) == 3


def test_low_ocr_confidence_flags_every_step(tmp_path):
    engine = NeuroSymbolicVerificationEngine()
    text = "Step 1: By kinematics, final velocity is v = u + a*t = 0 + 10*2 = 20"
    summary, trees = engine.verify_submission("AUDIT_TEST_LOWOCR", text, problem_id="prob_falling_object")

    report = build_audit_report(summary, trees, problem_id="prob_falling_object", ocr_confidence=0.4)
    assert report["metadata"]["human_review_required_count"] == len(summary.step_evaluations)
