"""
AO* Search on AND-OR Derivation Solution Graphs (Unit 1)
with Nonlinear Partial-Order Constraint Posting for AND-branches (Unit 4).
"""

from typing import Dict, List, Optional, Tuple, Set
import copy
from ..models.step_frame import StepFrame, RubricStep, RubricStrategy


class PartialOrderConstraintPoster:
    """
    Unit 4 Nonlinear Planning component using Constraint Posting.
    Maintains causal order constraints (S_i < S_j) and validates whether an observed
    permutation of student steps is a valid topological ordering of independent sub-goals.
    """

    @staticmethod
    def is_valid_topological_order(
        observed_step_ids: List[int],
        causal_constraints: Dict[int, List[int]]
    ) -> bool:
        """
        causal_constraints[j] = [i1, i2] means step j requires outputs from i1 and i2 (i.e. i1 < j and i2 < j).
        Validates if observed_step_ids respects this partial order.
        """
        seen: Set[int] = set()
        for sid in observed_step_ids:
            prereqs = causal_constraints.get(sid, [])
            for p in prereqs:
                if p in observed_step_ids and p not in seen:
                    # Prerequisite was written AFTER current step in an illegal order!
                    return False
            seen.add(sid)
        return True

    @staticmethod
    def compute_ordering_penalty(
        student_step_order: List[int],
        rubric_step_ids: List[int],
        causal_constraints: Dict[int, List[int]]
    ) -> float:
        """
        Returns 0.0 if the observed order is a valid topological sort under constraint posting.
        Returns a positive penalty proportional to violated causal constraints.
        """
        # Filter student steps to those that correspond to this rubric branch
        relevant_steps = [s for s in student_step_order if s in rubric_step_ids]
        if not relevant_steps:
            return 0.0

        if PartialOrderConstraintPoster.is_valid_topological_order(relevant_steps, causal_constraints):
            return 0.0

        # Count number of inverted causal pairs
        violations = 0
        seen = set()
        for s in relevant_steps:
            for p in causal_constraints.get(s, []):
                if p in relevant_steps and p not in seen:
                    violations += 1
            seen.add(s)

        return violations * 0.5


class AOStarRubricMatcher:
    """
    AO* Search implementation over an AND-OR Solution Graph.
    - OR Nodes: Competing solution methods (e.g., Kinematics vs. Energy Conservation).
    - AND Nodes: Conjunctive milestone requirements of a specific method.
    """

    def __init__(self, strategies: List[RubricStrategy]):
        self.strategies = {s.strategy_id: s for s in strategies}

    def compute_heuristic_step_distance(
        self,
        student_step: StepFrame,
        rubric_step: RubricStep
    ) -> float:
        """
        Heuristic cost h(n) measuring semantic and equation symbol divergence:
        h = w_concept * (1 - concept_match) + w_sym * (1 - symbol_overlap)
        """
        # Concept match
        s_concept = (student_step.invoked_concept or "").lower()
        r_concept = rubric_step.concept.lower()
        c_match = 1.0 if (s_concept and (s_concept in r_concept or r_concept in s_concept)) else 0.2

        # Variable symbol overlap
        import re
        s_vars = set(re.findall(r'[a-zA-Z_]+', student_step.raw_equation or ""))
        r_vars = set(re.findall(r'[a-zA-Z_]+', rubric_step.canonical_equation))
        math_funcs = {'sqrt', 'frac', 'cdot', 'sin', 'cos', 'tan', 'is'}
        s_vars -= math_funcs
        r_vars -= math_funcs

        overlap = len(s_vars.intersection(r_vars)) / max(len(r_vars), 1)
        sym_match = min(1.0, overlap)

        cost = (0.5 * (1.0 - c_match)) + (0.5 * (1.0 - sym_match))
        return round(cost, 4)

    def evaluate_strategy_alignment(
        self,
        student_steps: List[StepFrame],
        strategy: RubricStrategy
    ) -> Tuple[float, Dict[int, Optional[StepFrame]], List[RubricStep]]:
        """
        Evaluates the alignment cost of a specific strategy (OR-branch):
        1. Aligns each rubric milestone with best matching student StepFrame.
        2. Applies partial-order constraint posting penalty if independent steps violate causal DAG.
        3. Penalizes completely missing milestone steps.
        4. Penalizes unused/unmatched student steps to prevent single-step shortcuts from falsely winning.
        """
        aligned_map: Dict[int, Optional[StepFrame]] = {}
        missing_steps: List[RubricStep] = []
        cumulative_cost = 0.0

        used_student_step_ids = set()

        # Step 1: Greedy minimal-cost alignment for each rubric milestone
        for r_step in strategy.steps:
            best_match: Optional[StepFrame] = None
            best_cost = float('inf')

            for s_step in student_steps:
                if s_step.step_id in used_student_step_ids:
                    continue
                # Also consider steps that covered this via Gröbner algebraic leap
                if r_step.step_id in s_step.skipped_intermediate_steps:
                    best_match = s_step
                    best_cost = 0.0
                    break

                h = self.compute_heuristic_step_distance(s_step, r_step)
                if h < best_cost:
                    best_cost = h
                    best_match = s_step

            # If matched with reasonable cost (< 1.5), accept match
            if best_match is not None and best_cost < 1.5:
                aligned_map[r_step.step_id] = best_match
                used_student_step_ids.add(best_match.step_id)
                cumulative_cost += best_cost
            else:
                aligned_map[r_step.step_id] = None
                missing_steps.append(r_step)
                cumulative_cost += 2.0 # Penalty for completely missing required milestone

        # Step 2: Apply Nonlinear Partial-Order Constraint Posting
        ordered_matched_rubric_ids = []
        for s in student_steps:
            for r_id, matched_s in aligned_map.items():
                if matched_s is not None and matched_s.step_id == s.step_id:
                    ordered_matched_rubric_ids.append(r_id)
                    break

        rubric_all_ids = [r.step_id for r in strategy.steps]
        ordering_penalty = PartialOrderConstraintPoster.compute_ordering_penalty(
            student_step_order=ordered_matched_rubric_ids,
            rubric_step_ids=rubric_all_ids,
            causal_constraints=strategy.and_dependency_dag
        )
        cumulative_cost += ordering_penalty

        # Step 3: Unmatched student steps penalty (prevents small strategies from unfairly dominating)
        unmatched_count = len(student_steps) - len(used_student_step_ids)
        cumulative_cost += unmatched_count * 1.5

        return round(cumulative_cost, 4), aligned_map, missing_steps

    def align_best_strategy(
        self,
        student_steps: List[StepFrame]
    ) -> Tuple[RubricStrategy, float, Dict[int, Optional[StepFrame]], List[RubricStep]]:
        """
        AO* OR-node selection:
        Evaluates all alternative valid strategies and chooses the one with minimum alignment cost.
        """
        best_strategy = None
        min_cost = float('inf')
        best_alignment = {}
        best_missing = []

        for strat_id, strategy in self.strategies.items():
            cost, aligned_map, missing = self.evaluate_strategy_alignment(student_steps, strategy)
            if cost < min_cost:
                min_cost = cost
                best_strategy = strategy
                best_alignment = aligned_map
                best_missing = missing

        if best_strategy is None and self.strategies:
            first_strat = next(iter(self.strategies.values()))
            return first_strat, 999.0, {}, first_strat.steps

        return best_strategy, min_cost, best_alignment, best_missing
