"""
Neuro-Symbolic Verification & Grading Engine (CLI Entrypoint).
Demonstrates end-to-end evaluation of technical derivations with defeasible reasoning.
"""

import sys
import os
import argparse

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.pipeline import NeuroSymbolicVerificationEngine
from src.reasoning_engine.explanation_facility import ExplanationFacility
from src.reasoning_engine.audit_report import build_audit_report, persist_audit_report
from src.uncertainty_fuzzy.ga_calibrator import GeneticAlgorithmCalibrator
from src.perception_nlp.ocr_frontend import OCRSubmissionReader
from src.search_alignment.rubric_synthesizer import RubricSynthesizer
from ui.dashboard import EvaluationVisualizer


BENCHMARK_DEMOS = {
    "1": (
        "Golden Solution (Exact Derivation)",
        """
        Step 1: By kinematics, final velocity is v = u + a*t = 0 + 10*2 = 20
        Step 2: Average displacement is s = (u + v)/2 * t = (0 + 20)/2 * 2 = 20
        Step 3: Work done by gravity is W = F * s = 20 * 20 = 400
        """
    ),
    "2": (
        "Multi-Hop Transitive ECF (Early Slip Propagated Correctly)",
        """
        Step 1: Velocity is v = u + a*t = 0 + 10*2 = 25
        Step 2: Displacement is s = (u + v)/2 * t = (0 + 25)/2 * 2 = 25
        Step 3: Work done is W = F * s = 20 * 25 = 500
        """
    ),
    "3": (
        "Algebraic Leap via Gröbner Basis (Combined Equation)",
        """
        Step 1: Combining F = ma and kinematics, F = m*(v^2 - u^2)/(2*s) = 2*(100 - 0)/(2*10) = 10
        """
    ),
    "4": (
        "Reordered Independent Steps (Partial-Order Planning)",
        """
        Step 1: Potential energy is PE = m * a * s = 2 * 10 * 20 = 400
        Step 2: Normal force is N = m * a = 2 * 10 = 20
        Step 3: Total work is W = PE = 400
        """
    ),
    "5": (
        "Persuasive Math Hallucination (Conflict K & Symbolic Veto)",
        """
        Step 1: Clearly by conservation of momentum and energy, velocity is v = u / (a * t^2) = 20
        """
    )
}

SCENARIO_PROBLEMS = {
    "1": "prob_falling_object",
    "2": "prob_falling_object",
    "3": "prob_combined_leap",
    "4": "prob_independent_dag",
    "5": "prob_falling_object",
}


def run_demo(scenario_key: str = "2", calibrate: bool = False, image_path: str = None, problem_id: str = None):
    print("=" * 80)
    print("NeuroSymGrade — Neuro-Symbolic Verification & Grading Engine with Non-Monotonic Logic")
    print("Units 1–6: AO* Search, SymPy Gröbner, Prolog Defeasible Logic, Discounted DS & Numba GA")
    print("=" * 80)

    ocr_confidence = None
    if image_path:
        print(f"\n[OCR] Reading scanned/handwritten submission from: {image_path}")
        print("[OCR] Loading EasyOCR model (first run downloads ~65MB of weights)...")
        reader = OCRSubmissionReader()
        text, ocr_confidence = reader.extract_text_with_confidence(image_path)
        title = f"OCR Submission ({image_path})"
        print(f"[OCR] Extracted text (mean confidence {ocr_confidence}):")
        print("-" * 50)
        print(text.strip())
        print("-" * 50)
    else:
        title, text = BENCHMARK_DEMOS.get(scenario_key, BENCHMARK_DEMOS["2"])
        print(f"\n[DEMO SELECTION] Scenario {scenario_key}: {title}\n")
        print("STUDENT SUBMISSION TEXT:")
        print("-" * 50)
        print(text.strip())
        print("-" * 50)

    engine = NeuroSymbolicVerificationEngine()

    if calibrate:
        print("\n[GA CALIBRATION] Running Numba-Accelerated Genetic Algorithm to tune fuzzy sets...")
        ga = GeneticAlgorithmCalibrator(pop_size=30, generations=20)
        params, history = ga.run_calibration()
        engine.fuzzy_scorer.params = params
        visualizer = EvaluationVisualizer()
        conv_path = visualizer.plot_ga_convergence(history)
        mf_path = visualizer.plot_fuzzy_memberships(params)
        print(f"-> GA Convergence plot saved to: {conv_path}")
        print(f"-> Calibrated Fuzzy Memberships saved to: {mf_path}")

    print("\n[VERIFICATION] Executing Neuro-Symbolic Pipeline...")
    resolved_problem_id = problem_id or SCENARIO_PROBLEMS.get(scenario_key, "prob_falling_object")
    submission_id = "OCR_SUBMISSION" if image_path else f"DEMO_SCENARIO_{scenario_key}"
    summary, trees = engine.verify_submission(
        submission_id=submission_id,
        raw_submission_text=text,
        problem_id=resolved_problem_id
    )

    # Print Full Audit Report
    report = ExplanationFacility.generate_full_audit_report(summary, trees)
    print("\n" + report)

    # Persist a durable, timestamped JSON audit report (not just stdout)
    json_report = build_audit_report(summary, trees, problem_id=resolved_problem_id, ocr_confidence=ocr_confidence)
    report_path = persist_audit_report(json_report)
    review_count = json_report["metadata"]["human_review_required_count"]
    print(f"[AUDIT] Persisted report to: {report_path}")
    if review_count > 0:
        print(f"[AUDIT] {review_count} step(s) flagged for human review: {json_report['metadata']['flagged_steps']}")

    # Plot Uncertainty Intervals
    visualizer = EvaluationVisualizer()
    ds_path = visualizer.plot_dempster_shafer_intervals(summary.step_evaluations)
    print(f"[PLOTS] Dempster-Shafer Uncertainty chart generated at: {ds_path}\n")

    print("=" * 80)
    print(f"FINAL RESULT: {summary.total_marks_awarded} / {summary.max_possible_marks} ({summary.percentage}%)")
    print(f"Strategy Selected by AO* Search: {summary.selected_strategy}")
    print("=" * 80)


def run_synthesize(spec_path: str):
    """
    Compiles a rubric from a JSON spec of master/reference derivations
    (see config/example_synth_spec.json) into config/domain_rules.json,
    using RubricSynthesizer -- no LLM, purely the project's own
    NLP/OWL/SymPy stack.
    """
    import json
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    synth = RubricSynthesizer()
    problem_entry = synth.synthesize_problem(
        problem_id=spec["problem_id"],
        title=spec["title"],
        description=spec.get("description", ""),
        strategy_specs=spec["strategies"],
        total_marks=float(spec.get("total_marks", 15.0))
    )
    path = synth.save_problem_to_config(problem_entry)

    print(f"[SYNTHESIZE] Compiled {len(problem_entry['strategies'])} strategy(ies) for "
          f"'{problem_entry['problem_id']}' and saved to: {path}\n")
    for strat in problem_entry["strategies"]:
        print(f"  Strategy: {strat['name']} ({strat['strategy_id']})")
        for step in strat["steps"]:
            print(f"    Step {step['step_id']} [{step['concept']}]: {step['canonical_equation']} "
                  f"= {step['expected_output_value']} ({step['max_marks']} marks)")
    print(f"\nGrade against it with: python main.py --problem {problem_entry['problem_id']} "
          f"--image <scan.jpg>  (or feed text directly via the API/Streamlit UI)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neuro-Symbolic Verification Engine CLI")
    parser.add_argument("--scenario", type=str, default="2", choices=["1", "2", "3", "4", "5"],
                        help="Benchmark scenario number (1 to 5)")
    parser.add_argument("--calibrate", action="store_true", help="Run GA calibrator before evaluation")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a scanned/handwritten submission image. "
                             "Runs OCR (EasyOCR) to extract text instead of using a benchmark scenario.")
    parser.add_argument("--problem", type=str, default=None,
                        help="Rubric problem_id to grade against when using --image "
                             "(default: prob_falling_object)")
    parser.add_argument("--synthesize", type=str, default=None,
                        help="Path to a rubric-synthesis spec JSON. Compiles a new rubric from "
                             "master/reference derivations into config/domain_rules.json and exits.")
    args = parser.parse_args()

    if args.synthesize:
        run_synthesize(args.synthesize)
    else:
        run_demo(scenario_key=args.scenario, calibrate=args.calibrate,
                  image_path=args.image, problem_id=args.problem)
