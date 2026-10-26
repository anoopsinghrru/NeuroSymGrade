"""
Neuro-Symbolic Verification Engine -- FastAPI Service Layer.

Wraps the existing engine (src/pipeline.py), OCR front-end, rubric synthesizer,
and audit-report store behind a REST API, so the symbolic/defeasible/fuzzy
grading core can be driven by something other than the CLI (a frontend, a
script, a notebook) without duplicating any grading logic.

No LLM is involved anywhere in this service -- every endpoint is a thin
wrapper over the Units 1-6 algorithmic pipeline already in src/.

Run with: uvicorn api.server:app --reload --port 8000
Docs at:  http://127.0.0.1:8000/docs
"""

import os
import sys
import tempfile
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import NeuroSymbolicVerificationEngine
from src.reasoning_engine.audit_report import (
    build_audit_report, persist_audit_report, list_audit_reports, load_audit_report
)
from src.search_alignment.rubric_synthesizer import RubricSynthesizer
from src.perception_nlp.ocr_frontend import OCRSubmissionReader

app = FastAPI(
    title="NeuroSymGrade API",
    description="Neuro-Symbolic Verification & Grading Engine. REST API over the "
                "AO*/SymPy/Prolog/Dempster-Shafer/Fuzzy/GA grading engine. "
                "No LLM -- every verdict is produced by the algorithmic pipeline in src/.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_engine = NeuroSymbolicVerificationEngine()
_synthesizer = RubricSynthesizer(_engine.grounder)
_ocr_reader: Optional[OCRSubmissionReader] = None


def _get_ocr_reader() -> OCRSubmissionReader:
    global _ocr_reader
    if _ocr_reader is None:
        _ocr_reader = OCRSubmissionReader()
    return _ocr_reader


# ── Request/Response Schemas ────────────────────────────────────────────────

class GradeTextRequest(BaseModel):
    submission_id: str
    raw_submission_text: str
    problem_id: str = "prob_falling_object"


class StrategySpec(BaseModel):
    strategy_id: str
    name: str
    description: str = ""
    master_solution: str


class SynthesizeRequest(BaseModel):
    problem_id: str
    title: str
    description: str = ""
    total_marks: float = 15.0
    strategies: List[StrategySpec]
    save: bool = True


# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health() -> Dict[str, Any]:
    return {
        "status": "online",
        "version": "1.0.0",
        "problems_loaded": list(_engine.problems.keys())
    }


# ── Grading ──────────────────────────────────────────────────────────────────

@app.post("/grade/text", tags=["Grading"])
def grade_text(req: GradeTextRequest) -> Dict[str, Any]:
    """Grades a typed/plain-text submission against an existing rubric problem."""
    try:
        summary, trees = _engine.verify_submission(
            submission_id=req.submission_id,
            raw_submission_text=req.raw_submission_text,
            problem_id=req.problem_id
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    report = build_audit_report(summary, trees, problem_id=req.problem_id)
    path = persist_audit_report(report)
    report["_persisted_path"] = path
    return report


@app.post("/grade/image", tags=["Grading"])
async def grade_image(
    file: UploadFile = File(...),
    submission_id: str = "OCR_SUBMISSION",
    problem_id: str = "prob_falling_object"
) -> Dict[str, Any]:
    """Grades a scanned/handwritten submission image via OCR (EasyOCR) then the engine."""
    suffix = os.path.splitext(file.filename or "submission.png")[1] or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        reader = _get_ocr_reader()
        text, ocr_confidence = reader.extract_text_with_confidence(tmp_path)
        if not text.strip():
            raise HTTPException(status_code=422, detail="OCR extracted no text from the image.")

        summary, trees = _engine.verify_submission(
            submission_id=submission_id,
            raw_submission_text=text,
            problem_id=problem_id
        )
        report = build_audit_report(summary, trees, problem_id=problem_id, ocr_confidence=ocr_confidence)
        report["ocr_extracted_text"] = text
        path = persist_audit_report(report)
        report["_persisted_path"] = path
        return report
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# ── Rubric Synthesis ─────────────────────────────────────────────────────────

@app.post("/rubric/synthesize", tags=["Rubric Synthesis"])
def synthesize_rubric(req: SynthesizeRequest) -> Dict[str, Any]:
    """
    Compiles one or more master/reference derivations into a gradeable rubric
    (AO*/AND-OR strategy graph, canonical equations, complexity-weighted marks)
    -- no LLM, purely the project's own NLP/OWL/SymPy stack.
    """
    try:
        problem_entry = _synthesizer.synthesize_problem(
            problem_id=req.problem_id,
            title=req.title,
            description=req.description,
            strategy_specs=[s.model_dump() for s in req.strategies],
            total_marks=req.total_marks
        )
        if req.save:
            path = _synthesizer.save_problem_to_config(problem_entry)
            _engine._load_config()  # reload so the new problem is immediately gradeable
            problem_entry["_saved_to"] = path
        return problem_entry
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/problems", tags=["Rubric Synthesis"])
def list_problems() -> Dict[str, Any]:
    return {"problem_ids": list(_engine.problems.keys())}


# ── Audit Report History ────────────────────────────────────────────────────

@app.get("/reports", tags=["Reports"])
def get_reports() -> Dict[str, Any]:
    return {"reports": list_audit_reports()}


@app.get("/reports/{report_id}", tags=["Reports"])
def get_report(report_id: str) -> Dict[str, Any]:
    report = load_audit_report(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"Report '{report_id}' not found.")
    return report
