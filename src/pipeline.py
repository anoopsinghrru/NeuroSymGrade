"""
Unified Neuro-Symbolic Verification & Grading Pipeline Orchestrator.
Integrates Units 1–6 into an end-to-end evaluation engine:
NLP Perception -> StepFrames -> AO* Search -> SymPy Gröbner Verifier ->
Dempster-Shafer Fusion -> Prolog Defeasible Logic -> Fuzzy Centroid Scoring -> Explanation Trees.
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Any

from .models.step_frame import (
    StepFrame,
    RubricStep,
    RubricStrategy,
    DerivationEvaluationSummary
)
from .symbolic_math.owl_grounding import OWLGrounder
from .symbolic_math.sympy_verifier import SymPyVerifier
from .perception_nlp.neural_slot_extractor import NeuralSlotExtractor
from .search_alignment.aostar_rubric_matcher import AOStarRubricMatcher
from .uncertainty_fuzzy.dempster_shafer import DempsterShaferFusion
from .uncertainty_fuzzy.fuzzy_scorer import FuzzyScorer
from .reasoning_engine.prolog_bridge import DefeasiblePrologEngine, ProofNode
from .reasoning_engine.explanation_facility import ExplanationFacility


class NeuroSymbolicVerificationEngine:
    """
    Master Neuro-Symbolic Grading Engine.
    Combines neural perception, symbolic mathematics, AO* search alignment,
    non-monotonic defeasible reasoning, and fuzzy partial-credit scoring.
    """

    UNIT_MISSING_PENALTY = 0.90  # 10% deduction: correct value, unit omitted
    UNIT_WRONG_PENALTY = 0.75    # 25% deduction: correct value, wrong/mismatched unit

    def __init__(self, config_path: Optional[str] = None):
        self.grounder = OWLGrounder()
        self.verifier = SymPyVerifier(self.grounder)
        self.extractor = NeuralSlotExtractor(self.grounder)
        self.ds_fusion = DempsterShaferFusion(neural_discount=0.85, symbolic_discount=0.90)
        self.prolog_engine = DefeasiblePrologEngine()
        self.fuzzy_scorer = FuzzyScorer()

        self.problems: Dict[str, Dict[str, Any]] = {}
        self._load_config(config_path)

    def _load_config(self, config_path: Optional[str] = None):
        """Loads domain rules and rubrics from JSON."""
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "config", "domain_rules.json"
            )

        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for prob in data.get("problems", []):
                        self.problems[prob["problem_id"]] = prob
            except Exception:
                pass

    def get_rubric_strategies_for_problem(self, problem_id: str) -> List[RubricStrategy]:
        """Constructs RubricStrategy objects from loaded problem configuration."""
        prob = self.problems.get(problem_id)
        if not prob:
            # Default fallback strategy if problem ID not found
            s1 = RubricStrategy(
                strategy_id="strat_default",
                name="Standard Derivation",
                description="Default kinematic derivation",
                steps=[
                    RubricStep(1, "Kinematics", "v = u + a*t", {"u": 0.0, "a": 10.0, "t": 2.0}, 20.0, 5.0),
                    RubricStep(2, "Kinematics", "s = v * t", {"v": 20.0, "t": 2.0}, 40.0, 5.0, prerequisites=[1]),
                    RubricStep(3, "WorkEnergyTheorem", "W = F * s", {"F": 20.0, "s": 40.0}, 800.0, 5.0, prerequisites=[2])
                ],
                and_dependency_dag={1: [], 2: [1], 3: [2]}
            )
            return [s1]

        strategies = []
        for s_data in prob.get("strategies", []):
            steps = []
            for step_data in s_data.get("steps", []):
                r_step = RubricStep(
                    step_id=step_data["step_id"],
                    concept=step_data["concept"],
                    canonical_equation=step_data["canonical_equation"],
                    expected_bindings=step_data.get("expected_bindings", {}),
                    expected_output_value=float(step_data["expected_output_value"]),
                    max_marks=float(step_data["max_marks"]),
                    prerequisites=step_data.get("prerequisites", []),
                    require_units=step_data.get("require_units", False)
                )
                steps.append(r_step)

            # Convert string keys in dag to integers
            dag = {}
            for k, v in s_data.get("and_dependency_dag", {}).items():
                dag[int(k)] = [int(p) for p in v]

            strat = RubricStrategy(
                strategy_id=s_data["strategy_id"],
                name=s_data["name"],
                description=s_data["description"],
                steps=steps,
                and_dependency_dag=dag
            )
            strategies.append(strat)

        return strategies

    def verify_submission(
        self,
        submission_id: str,
        raw_submission_text: str,
        problem_id: str = "prob_falling_object",
        custom_strategies: Optional[List[RubricStrategy]] = None
    ) -> Tuple[DerivationEvaluationSummary, Dict[int, ProofNode]]:
        """
        Executes end-to-end verification and grading for an unstructured student submission.
        """
        # Step 1: Perception & Slot Extraction (Units 4 & 5)
        student_steps = self.extractor.parse_submission(raw_submission_text)

        # Step 2: AO* Derivation Graph Search Alignment (Unit 1 & 4)
        strategies = custom_strategies or self.get_rubric_strategies_for_problem(problem_id)
        matcher = AOStarRubricMatcher(strategies)
        best_strategy, alignment_cost, aligned_map, missing_rubric_steps = matcher.align_best_strategy(student_steps)

        # Reset Prolog Engine for fresh proof session
        self.prolog_engine.reset()
        for r_step in best_strategy.steps:
            self.prolog_engine.assert_rubric_step(r_step)

        # Step 3 & 4: Symbolic Math, Evidence Fusion, and Defeasible Reasoning
        # Seed initial bindings from problem constants / rubric expected_bindings
        accumulated_student_bindings: Dict[str, float] = {}
        for r_step in best_strategy.steps:
            for k, v in r_step.expected_bindings.items():
                if k not in accumulated_student_bindings:
                    accumulated_student_bindings[k] = float(v)

        proof_trees: Dict[int, ProofNode] = {}
        evaluated_steps: List[StepFrame] = []

        # Track rubric steps that were skipped via algebraic leap
        skipped_step_pool = list(missing_rubric_steps)

        for s_step in student_steps:
            # Find corresponding matched rubric step if any
            matched_r_step = None
            for r_id, aligned_s in aligned_map.items():
                if aligned_s is not None and aligned_s.step_id == s_step.step_id:
                    matched_r_step = next((r for r in best_strategy.steps if r.step_id == r_id), None)
                    break

            if matched_r_step is None:
                # Student provided a step outside the strategy; treat against nearest concept milestone
                matched_r_step = best_strategy.steps[min(s_step.step_id - 1, len(best_strategy.steps) - 1)]

            # 3.1. SymPy Canonical Verifier & Dual Residuals (Unit 2 & 4)
            self.verifier.verify_step(
                step_frame=s_step,
                rubric_step=matched_r_step,
                skipped_intermediate_rubric_steps=skipped_step_pool,
                accumulated_student_bindings=accumulated_student_bindings
            )

            # Update accumulated student bindings
            for k, v in s_step.variable_bindings.items():
                accumulated_student_bindings[k] = v
            if s_step.evaluated_value is not None:
                lhs_sym = None
                if s_step.canonical_lhs is not None and s_step.canonical_lhs.free_symbols:
                    lhs_sym = str(list(s_step.canonical_lhs.free_symbols)[0])
                elif matched_r_step and matched_r_step.canonical_equation:
                    r_lhs, _ = self.verifier.parse_equation(matched_r_step.canonical_equation)
                    if r_lhs is not None and r_lhs.free_symbols:
                        lhs_sym = str(list(r_lhs.free_symbols)[0])
                if lhs_sym:
                    accumulated_student_bindings[lhs_sym] = s_step.evaluated_value

            # 3.2. Discounted Dempster-Shafer Fusion (Unit 3)
            bel_c, conflict_k, pl_c = self.ds_fusion.evaluate_step_belief(
                neural_confidence=s_step.neural_confidence,
                is_symbolically_valid=s_step.is_symbolically_valid
            )
            s_step.belief_support = bel_c
            s_step.epistemic_conflict = conflict_k

            # 3.3. Defeasible Non-Monotonic Logic & Meta-Interpreter (Units 2, 3 & 6)
            self.prolog_engine.assert_student_step(s_step)
            status, mult, explanation, tree = self.prolog_engine.evaluate_step(s_step.step_id)
            s_step.defeasible_status = status
            s_step.credit_multiplier = mult
            s_step.explanation = explanation
            proof_trees[s_step.step_id] = tree

            # 3.4. Fuzzy Logic Controller & Centroid Defuzzification (Units 2 & 6)
            awarded_marks = self.fuzzy_scorer.evaluate_fuzzy_score(
                defeasible_multiplier=s_step.credit_multiplier,
                belief_support=s_step.belief_support,
                local_residual=s_step.local_residual,
                max_marks=matched_r_step.max_marks
            )

            # 3.5. Dimensional Analysis (Unit 4 ontology) -- opt-in per rubric step.
            # Backward-compatible: a rubric step with require_units unset (default
            # False) is never touched, so every pre-existing problem and documented
            # benchmark score is unaffected.
            if matched_r_step.require_units:
                r_lhs, _ = self.verifier.parse_equation(matched_r_step.canonical_equation)
                target_symbol = str(list(r_lhs.free_symbols)[0]) if r_lhs is not None and r_lhs.free_symbols else None
                if target_symbol:
                    unit_valid, unit_extracted = self.grounder.check_unit(target_symbol, s_step.raw_text)
                    s_step.unit_required = True
                    s_step.unit_provided = unit_extracted
                    s_step.unit_valid = unit_valid
                    if unit_valid is False:
                        penalty = self.UNIT_WRONG_PENALTY if unit_extracted is not None else self.UNIT_MISSING_PENALTY
                        awarded_marks = round(awarded_marks * penalty, 4)
                        s_step.explanation += (
                            f" [Unit check: expected a unit on {target_symbol}; "
                            f"{'found mismatched unit ' + repr(unit_extracted) if unit_extracted else 'no unit provided'} "
                            f"-> {int((1 - penalty) * 100)}% deduction applied.]"
                        )

            s_step.final_score = awarded_marks
            evaluated_steps.append(s_step)

        # Calculate totals
        max_marks = sum(r.max_marks for r in best_strategy.steps)
        total_awarded = sum(s.final_score for s in evaluated_steps)
        # Cap awarded marks at max_marks
        total_awarded = min(total_awarded, max_marks)
        pct = (total_awarded / max_marks * 100.0) if max_marks > 0 else 0.0

        summary = DerivationEvaluationSummary(
            submission_id=submission_id,
            selected_strategy=f"{best_strategy.name} ({best_strategy.strategy_id})",
            alignment_cost=alignment_cost,
            total_marks_awarded=round(total_awarded, 2),
            max_possible_marks=round(max_marks, 2),
            percentage=round(pct, 1),
            step_evaluations=evaluated_steps,
            global_explanation_tree={sid: tree.to_dict() for sid, tree in proof_trees.items()}
        )

        return summary, proof_trees

    def explain_why(self, summary: DerivationEvaluationSummary, step_id: int) -> str:
        """Interactive query: WHY did step K receive its grade?"""
        step = next((s for s in summary.step_evaluations if s.step_id == step_id), None)
        if not step:
            return f"Step {step_id} not found in submission."
        return ExplanationFacility.answer_why(step, None)

    def explain_how(self, proof_trees: Dict[int, ProofNode], step_id: int) -> str:
        """Interactive query: HOW was step K evaluated (unfolds deduction tree)?"""
        tree = proof_trees.get(step_id)
        if not tree:
            return f"Proof tree for step {step_id} not found."
        return ExplanationFacility.answer_how(tree)
