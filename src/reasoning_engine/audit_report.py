"""
Audit Report Persistence (Units 3 & 6).
Serializes a DerivationEvaluationSummary + its backward-chaining proof trees
into a durable, timestamped JSON artifact -- the engine's own evidence trail,
instead of only ephemeral stdout printed by a CLI run.

The "human review required" flag is not a separate model call (unlike an
LLM-based grader's self-reported confidence): it is derived directly from
signals the engine already computes -- Dempster-Shafer epistemic conflict K
(Unit 3) and, when perception confidence is high but credit awarded is zero,
a likely perception/parsing mismatch worth a second look.
"""

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..models.step_frame import DerivationEvaluationSummary, StepFrame
from ..reasoning_engine.prolog_bridge import ProofNode

CONFLICT_REVIEW_THRESHOLD = 0.5
CONFIDENT_BUT_ZERO_CREDIT_THRESHOLD = 0.75


def _step_review_reason(step: StepFrame) -> Optional[str]:
    if step.epistemic_conflict > CONFLICT_REVIEW_THRESHOLD:
        return f"High Dempster-Shafer epistemic conflict K={step.epistemic_conflict} between perception and symbolic evidence."
    if step.credit_multiplier == 0.0 and step.neural_confidence >= CONFIDENT_BUT_ZERO_CREDIT_THRESHOLD:
        return (
            f"Perception confidence was high ({step.neural_confidence}) but the step received zero credit "
            f"({step.defeasible_status}) -- possibly a perception/OCR misread rather than a genuine error."
        )
    return None


def build_audit_report(
    summary: DerivationEvaluationSummary,
    proof_trees: Dict[int, ProofNode],
    problem_id: Optional[str] = None,
    ocr_confidence: Optional[float] = None
) -> Dict[str, Any]:
    """Builds the full audit report dict (metadata + per-step detailed audit)."""
    detailed_audit: List[Dict[str, Any]] = []
    flagged_steps: List[int] = []

    for step in summary.step_evaluations:
        reason = _step_review_reason(step)
        if reason:
            flagged_steps.append(step.step_id)

        detailed_audit.append({
            "step_id": step.step_id,
            "raw_text": step.raw_text,
            "concept": step.invoked_concept,
            "equation": step.raw_equation,
            "symbolic_valid": step.is_symbolically_valid,
            "algebraic_leap": step.is_algebraic_leap,
            "evaluated_value": step.evaluated_value,
            "rubric_residual": step.rubric_residual,
            "local_residual": step.local_residual,
            "neural_confidence": step.neural_confidence,
            "belief_support": step.belief_support,
            "epistemic_conflict": step.epistemic_conflict,
            "defeasible_status": step.defeasible_status,
            "credit_multiplier": step.credit_multiplier,
            "marks_awarded": step.final_score,
            "explanation": step.explanation,
            "human_review_flag": reason is not None,
            "review_reason": reason,
            "proof_tree": proof_trees[step.step_id].to_dict() if step.step_id in proof_trees else None
        })

    ocr_flagged = ocr_confidence is not None and ocr_confidence < CONFIDENT_BUT_ZERO_CREDIT_THRESHOLD
    if ocr_flagged:
        flagged_steps = sorted(set(flagged_steps) | {s.step_id for s in summary.step_evaluations})

    return {
        "metadata": {
            "report_id": f"{summary.submission_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "submission_id": summary.submission_id,
            "problem_id": problem_id,
            "selected_strategy": summary.selected_strategy,
            "alignment_cost": summary.alignment_cost,
            "score_final": f"{summary.total_marks_awarded}/{summary.max_possible_marks}",
            "total_marks_awarded": summary.total_marks_awarded,
            "max_possible_marks": summary.max_possible_marks,
            "percentage": summary.percentage,
            "ocr_confidence": ocr_confidence,
            "human_review_required_count": len(flagged_steps),
            "flagged_steps": flagged_steps
        },
        "detailed_audit": detailed_audit
    }


def persist_audit_report(report: Dict[str, Any], output_dir: str = "reports") -> str:
    """Writes the report to <output_dir>/AUDIT_REPORT_<submission_id>_<timestamp>.json."""
    os.makedirs(output_dir, exist_ok=True)
    safe_id = re.sub(r'[^A-Za-z0-9_\-]', '_', report["metadata"]["report_id"])
    path = os.path.join(output_dir, f"AUDIT_REPORT_{safe_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return path


def list_audit_reports(output_dir: str = "reports") -> List[Dict[str, Any]]:
    """Returns lightweight metadata for every persisted report (for a /history view)."""
    if not os.path.isdir(output_dir):
        return []
    entries = []
    for fname in sorted(os.listdir(output_dir), reverse=True):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(output_dir, fname), "r", encoding="utf-8") as f:
                data = json.load(f)
            entries.append({"file": fname, **data.get("metadata", {})})
        except Exception:
            continue
    return entries


def load_audit_report(report_id_or_filename: str, output_dir: str = "reports") -> Optional[Dict[str, Any]]:
    """Loads one full persisted report by its report_id or exact filename."""
    if not os.path.isdir(output_dir):
        return None
    for fname in os.listdir(output_dir):
        if not fname.endswith(".json"):
            continue
        if fname == report_id_or_filename or report_id_or_filename in fname:
            with open(os.path.join(output_dir, fname), "r", encoding="utf-8") as f:
                return json.load(f)
    return None
