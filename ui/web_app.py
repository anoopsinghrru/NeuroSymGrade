"""
Streamlit Frontend for the Neuro-Symbolic Verification & Grading Engine.

A thin UI over the same engine the CLI (main.py) and REST API (api/server.py)
use -- no grading logic lives here, only presentation. Three tabs:
  1. Grade a Submission -- typed text or a scanned/handwritten image (OCR).
  2. Synthesize a Rubric -- compile a new gradeable problem from a reference
     derivation (Units 1/2/4/5 symbolic pipeline, no LLM).
  3. Report History -- browse persisted, timestamped audit reports and see
     which steps were flagged for human review and why.

Run with: streamlit run ui/web_app.py
"""

import os
import sys
import tempfile

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import NeuroSymbolicVerificationEngine
from src.reasoning_engine.explanation_facility import ExplanationFacility
from src.reasoning_engine.audit_report import (
    build_audit_report, persist_audit_report, list_audit_reports, load_audit_report
)
from src.search_alignment.rubric_synthesizer import RubricSynthesizer
from src.perception_nlp.ocr_frontend import OCRSubmissionReader
from ui.dashboard import EvaluationVisualizer

st.set_page_config(page_title="NeuroSymGrade", layout="wide")


@st.cache_resource
def get_engine() -> NeuroSymbolicVerificationEngine:
    return NeuroSymbolicVerificationEngine()


@st.cache_resource
def get_ocr_reader() -> OCRSubmissionReader:
    return OCRSubmissionReader()


engine = get_engine()
synthesizer = RubricSynthesizer(engine.grounder)
visualizer = EvaluationVisualizer()

st.title("NeuroSymGrade")
st.subheader("Neuro-Symbolic Verification & Grading Engine")
st.caption(
    "AO* Search · SymPy Gröbner Verification · Prolog Defeasible Logic · "
    "Discounted Dempster-Shafer · Fuzzy Scoring · Numba GA Calibration — no LLM in the grading loop."
)

tab_grade, tab_synth, tab_history = st.tabs(["Grade a Submission", "Synthesize a Rubric", "Report History"])

# ── Tab 1: Grade a Submission ───────────────────────────────────────────────
with tab_grade:
    st.subheader("Grade a derivation")
    problem_ids = list(engine.problems.keys())
    problem_id = st.selectbox("Rubric problem", problem_ids, index=0 if problem_ids else None)
    input_mode = st.radio("Input method", ["Typed / pasted text", "Scanned or handwritten image (OCR)"], horizontal=True)

    raw_text = None
    ocr_confidence = None

    if input_mode == "Typed / pasted text":
        raw_text = st.text_area(
            "Submission text",
            height=160,
            value="Step 1: By kinematics, final velocity is v = u + a*t = 0 + 10*2 = 20\n"
                  "Step 2: Average displacement is s = (u + v)/2 * t = (0 + 20)/2 * 2 = 20\n"
                  "Step 3: Work done by gravity is W = F * s = 20 * 20 = 400"
        )
    else:
        uploaded = st.file_uploader("Upload a submission image", type=["png", "jpg", "jpeg", "bmp"])
        if uploaded is not None:
            st.image(uploaded, caption="Uploaded submission", width=500)
            if st.button("Run OCR"):
                with st.spinner("Running EasyOCR (first run downloads ~65MB of weights)..."):
                    suffix = os.path.splitext(uploaded.name)[1] or ".png"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded.getvalue())
                        tmp_path = tmp.name
                    reader = get_ocr_reader()
                    raw_text, ocr_confidence = reader.extract_text_with_confidence(tmp_path)
                    os.remove(tmp_path)
                st.session_state["ocr_text"] = raw_text
                st.session_state["ocr_confidence"] = ocr_confidence

        raw_text = st.session_state.get("ocr_text")
        ocr_confidence = st.session_state.get("ocr_confidence")
        if raw_text:
            st.text_area("OCR-extracted text (editable before grading)", value=raw_text, height=140, key="ocr_text_edit")
            raw_text = st.session_state.get("ocr_text_edit", raw_text)
            st.caption(f"Mean OCR confidence: {ocr_confidence}")

    submission_id = st.text_input("Submission ID", value="STREAMLIT_SUBMISSION")

    if st.button("Grade Submission", type="primary", disabled=not raw_text):
        with st.spinner("Running neuro-symbolic pipeline..."):
            summary, trees = engine.verify_submission(
                submission_id=submission_id,
                raw_submission_text=raw_text,
                problem_id=problem_id
            )
            report = build_audit_report(summary, trees, problem_id=problem_id, ocr_confidence=ocr_confidence)
            report_path = persist_audit_report(report)

        col1, col2, col3 = st.columns(3)
        col1.metric("Score", report["metadata"]["score_final"])
        col2.metric("Percentage", f"{report['metadata']['percentage']}%")
        col3.metric("Flagged for Review", report["metadata"]["human_review_required_count"])
        st.caption(f"Strategy selected by AO* search: {summary.selected_strategy} · saved to `{report_path}`")

        for step in report["detailed_audit"]:
            icon = "⚠️" if step["human_review_flag"] else "✅" if step["marks_awarded"] > 0 else "❌"
            with st.expander(f"{icon} Step {step['step_id']} — {step['defeasible_status']} "
                              f"({step['marks_awarded']} marks)"):
                st.write(f"**Raw text:** {step['raw_text']}")
                st.write(f"**Equation:** `{step['equation']}`  |  **Symbolically valid:** {step['symbolic_valid']}")
                st.write(f"**Rubric residual:** {step['rubric_residual']}  |  **Local residual:** {step['local_residual']}")
                st.write(f"**Belief:** {step['belief_support']}  |  **Epistemic conflict K:** {step['epistemic_conflict']}")
                st.write(f"**Explanation:** {step['explanation']}")
                if step["human_review_flag"]:
                    st.warning(step["review_reason"])
                st.json(step["proof_tree"], expanded=False)

        ds_path = visualizer.plot_dempster_shafer_intervals(summary.step_evaluations)
        st.image(ds_path, caption="Dempster-Shafer Belief/Plausibility Intervals")

# ── Tab 2: Synthesize a Rubric ──────────────────────────────────────────────
with tab_synth:
    st.subheader("Compile a rubric from a reference derivation")
    st.caption(
        "No hand-written rubric JSON: write the correct derivation the same way a student would, "
        "and the existing NLP/OWL/SymPy stack compiles canonical equations, a causal prerequisite DAG, "
        "and complexity-weighted marks per step."
    )

    new_problem_id = st.text_input("New problem ID", value="prob_new_topic")
    new_title = st.text_input("Title", value="")
    new_description = st.text_area("Description", value="", height=80)
    total_marks = st.number_input("Total marks", min_value=1.0, value=15.0, step=1.0)
    strategy_name = st.text_input("Strategy name", value="Primary Approach")
    master_solution = st.text_area(
        "Master / reference solution",
        height=160,
        placeholder="Step 1: Given u=20, a=-4, time to stop is t = (0 - u)/a = (0 - 20)/-4 = 5\n"
                    "Step 2: Stopping distance is s = u*t + 0.5*a*t^2 = 20*5 + 0.5*-4*25 = 50"
    )

    if st.button("Compile Rubric", type="primary", disabled=not (new_problem_id and master_solution)):
        try:
            strategy = synthesizer.synthesize_strategy(
                master_solution_text=master_solution,
                strategy_id="strat_" + "".join(c if c.isalnum() else "_" for c in strategy_name.lower()),
                name=strategy_name,
                total_marks=total_marks
            )
            st.success(f"Compiled {len(strategy.steps)} step(s).")
            st.table([
                {
                    "Step": s.step_id, "Concept": s.concept, "Equation": s.canonical_equation,
                    "Expected Value": s.expected_output_value, "Max Marks": s.max_marks,
                    "Prerequisites": s.prerequisites
                }
                for s in strategy.steps
            ])

            problem_entry = synthesizer.synthesize_problem(
                problem_id=new_problem_id,
                title=new_title or new_problem_id,
                description=new_description,
                strategy_specs=[{
                    "strategy_id": strategy.strategy_id,
                    "name": strategy_name,
                    "description": "",
                    "master_solution": master_solution
                }],
                total_marks=total_marks
            )
            path = synthesizer.save_problem_to_config(problem_entry)
            engine._load_config()  # reload so it is immediately gradeable in Tab 1
            st.info(f"Saved to `{path}` — select '{new_problem_id}' in the Grade tab to use it immediately.")
        except Exception as exc:
            st.error(f"Synthesis failed: {exc}")

# ── Tab 3: Report History ───────────────────────────────────────────────────
with tab_history:
    st.subheader("Persisted audit reports")
    reports = list_audit_reports()
    if not reports:
        st.info("No reports persisted yet — grade a submission in the first tab.")
    else:
        options = {f"{r['submission_id']} — {r.get('score_final', '?')} ({r.get('timestamp', '')})": r["report_id"]
                   for r in reports}
        choice = st.selectbox("Select a report", list(options.keys()))
        if choice:
            full = load_audit_report(options[choice])
            st.json(full["metadata"], expanded=True)
            for step in full["detailed_audit"]:
                flag = " ⚠️ FLAGGED" if step["human_review_flag"] else ""
                st.write(f"**Step {step['step_id']}** [{step['defeasible_status']}] "
                         f"{step['marks_awarded']} marks{flag}")
