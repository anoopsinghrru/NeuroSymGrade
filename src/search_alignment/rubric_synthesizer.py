"""
Rubric Synthesizer (Units 1, 2, 4, 5).
Compiles a human-authored *master/reference derivation* (plain text, written the
same way a student would write one) into a machine rubric: canonical equations,
expected variable bindings, expected output values, a causal prerequisite DAG,
and complexity-weighted max-marks per step -- forming a RubricStrategy consumed
directly by AOStarRubricMatcher / NeuroSymbolicVerificationEngine.

This closes the gap between "the rubric is a hardcoded answer key" and "the
engine can compile a rubric from a worked solution" by reusing exactly the
same perception/symbolic stack already used to grade students:
- TextPreprocessor (Unit 5) splits steps and extracts equations/bindings.
- NeuralSlotExtractor (Unit 5) classifies each step's invoked concept.
- OWLGrounder (Unit 4) canonicalizes variable synonyms.
- SymPyVerifier (Unit 2) parses equations and measures derivation complexity
  (sp.count_ops) to proportionally weight each step's marks -- a step that
  requires more algebraic work is worth more, rather than an even split.
- The prerequisite DAG (Unit 1 AND-OR graph / Unit 4 partial-order planning)
  is inferred from the same causal-connective detector used to interpret
  student submissions, not hand-drawn.

No LLM is involved anywhere in this module.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import sympy as sp

from ..models.step_frame import RubricStep, RubricStrategy
from ..perception_nlp.text_preprocessor import TextPreprocessor
from ..perception_nlp.neural_slot_extractor import NeuralSlotExtractor
from ..symbolic_math.owl_grounding import OWLGrounder
from ..symbolic_math.sympy_verifier import SymPyVerifier


class RubricSynthesizer:
    """Compiles master/reference derivations into RubricStrategy objects."""

    def __init__(self, grounder: Optional[OWLGrounder] = None):
        self.grounder = grounder or OWLGrounder()
        self.preprocessor = TextPreprocessor()
        self.extractor = NeuralSlotExtractor(self.grounder)
        self.verifier = SymPyVerifier(self.grounder)

    def _step_complexity(self, canonical_equation: Optional[str]) -> int:
        """Symbolic operation count of a step's equation; used to weight marks."""
        if not canonical_equation:
            return 1
        lhs, rhs = self.verifier.parse_equation(canonical_equation)
        if lhs is None or rhs is None:
            return 1
        try:
            return max(1, sp.count_ops(lhs) + sp.count_ops(rhs))
        except Exception:
            return 1

    def synthesize_strategy(
        self,
        master_solution_text: str,
        strategy_id: str,
        name: str,
        description: str = "",
        total_marks: float = 15.0
    ) -> RubricStrategy:
        """
        Parses one reference derivation into a RubricStrategy (one OR-branch
        of the AND-OR rubric graph). The text is written exactly like a
        (correct) student submission -- "Step 1: ... = ...", etc.
        """
        raw_steps = self.preprocessor.split_into_steps(master_solution_text)

        accumulated_bindings: Dict[str, float] = {}
        prior_ids: List[int] = []
        steps: List[RubricStep] = []
        complexities: List[int] = []

        for step_id, raw_text in raw_steps:
            concept, _ = self.extractor.predict_concept_with_confidence(raw_text)
            eq_str, eval_value, literal_bindings = self.preprocessor.extract_equations_and_values(raw_text)

            canonical_bindings = self.grounder.canonicalize_bindings(literal_bindings)
            accumulated_bindings.update(canonical_bindings)

            lhs, rhs = (None, None)
            canonical_eq = eq_str
            if eq_str:
                lhs, rhs = self.verifier.parse_equation(eq_str)

            # Snapshot only the bindings this step's own equation actually needs,
            # carried forward from prior steps' outputs or this step's literal assignments.
            rhs_symbols = {str(s) for s in rhs.free_symbols} if rhs is not None else set()
            lhs_symbols = {str(s) for s in lhs.free_symbols} if lhs is not None else set()
            needed_symbols = (rhs_symbols | lhs_symbols) - (
                {str(list(lhs.free_symbols)[0])} if lhs is not None and lhs.free_symbols else set()
            )
            expected_bindings = {
                sym: accumulated_bindings[sym] for sym in needed_symbols if sym in accumulated_bindings
            }

            deps = self.preprocessor.extract_causal_dependencies(raw_text, step_id, prior_ids)
            prior_ids.append(step_id)

            complexity = self._step_complexity(canonical_eq)
            complexities.append(complexity)

            # Auto-opt-in to dimensional-analysis grading (Unit 4 ontology): if the
            # reference/master solution itself states a unit for this step's output,
            # the synthesized rubric requires students to state it too.
            lhs_symbol = str(list(lhs.free_symbols)[0]) if lhs is not None and lhs.free_symbols else None
            require_units = bool(
                lhs_symbol
                and lhs_symbol in self.grounder.CANONICAL_UNIT_MAP
                and self.grounder.extract_trailing_unit(raw_text) is not None
            )

            steps.append(RubricStep(
                step_id=step_id,
                concept=concept if concept != "ScientificLaw" else "General",
                canonical_equation=canonical_eq or "",
                expected_bindings=expected_bindings,
                expected_output_value=float(eval_value) if eval_value is not None else 0.0,
                max_marks=0.0,  # filled in below, proportional to complexity
                prerequisites=deps,
                require_units=require_units
            ))

            # Carry this step's own output forward for downstream steps.
            if lhs is not None and lhs.free_symbols and eval_value is not None:
                lhs_sym = str(list(lhs.free_symbols)[0])
                accumulated_bindings[lhs_sym] = float(eval_value)

        total_complexity = sum(complexities) or max(1, len(steps))
        for step, c in zip(steps, complexities):
            step.max_marks = round(total_marks * (c / total_complexity), 2)
        if steps:
            drift = round(total_marks - sum(s.max_marks for s in steps), 2)
            steps[-1].max_marks = round(steps[-1].max_marks + drift, 2)

        dag = {s.step_id: s.prerequisites for s in steps}

        return RubricStrategy(
            strategy_id=strategy_id,
            name=name,
            description=description,
            steps=steps,
            and_dependency_dag=dag
        )

    def synthesize_problem(
        self,
        problem_id: str,
        title: str,
        description: str,
        strategy_specs: List[Dict[str, str]],
        total_marks: float = 15.0
    ) -> Dict[str, Any]:
        """
        Builds a full problem entry (one or more OR-branch strategies, i.e. the
        AND-OR rubric graph) in the same schema `config/domain_rules.json` uses,
        from `strategy_specs`: a list of
        {"strategy_id", "name", "description", "master_solution"} dicts.
        """
        strategies_json = []
        for spec in strategy_specs:
            strategy = self.synthesize_strategy(
                master_solution_text=spec["master_solution"],
                strategy_id=spec["strategy_id"],
                name=spec["name"],
                description=spec.get("description", ""),
                total_marks=total_marks
            )
            strategies_json.append({
                "strategy_id": strategy.strategy_id,
                "name": strategy.name,
                "description": strategy.description,
                "and_dependency_dag": {str(k): v for k, v in strategy.and_dependency_dag.items()},
                "steps": [
                    {
                        "step_id": s.step_id,
                        "concept": s.concept,
                        "canonical_equation": s.canonical_equation,
                        "expected_bindings": s.expected_bindings,
                        "expected_output_value": s.expected_output_value,
                        "max_marks": s.max_marks,
                        "prerequisites": s.prerequisites,
                        "require_units": s.require_units
                    }
                    for s in strategy.steps
                ]
            })

        return {
            "problem_id": problem_id,
            "title": title,
            "description": description,
            "strategies": strategies_json
        }

    def save_problem_to_config(
        self,
        problem_entry: Dict[str, Any],
        config_path: Optional[str] = None
    ) -> str:
        """Appends or replaces a problem entry in config/domain_rules.json."""
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "config", "domain_rules.json"
            )

        data = {"problems": []}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)

        problems = data.setdefault("problems", [])
        existing_idx = next(
            (i for i, p in enumerate(problems) if p["problem_id"] == problem_entry["problem_id"]), None
        )
        if existing_idx is not None:
            problems[existing_idx] = problem_entry
        else:
            problems.append(problem_entry)

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return config_path
