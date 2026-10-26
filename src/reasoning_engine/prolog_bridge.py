"""
Prolog Reasoning Bridge & Pure-Python SLD Resolution Meta-Interpreter (Units 2, 3, 6).
Provides defeasible non-monotonic reasoning, transitive ECF overrides,
loop-free dependency validation, and structured backward-chaining proof trees.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from ..models.step_frame import StepFrame, RubricStep


@dataclass
class ProofNode:
    """Represents a node in the backward-chaining deduction proof tree (Unit 6)."""
    goal: str
    status: str                                    # 'PROVED', 'NEGATION_VERIFIED', 'FAILED'
    sub_traces: List['ProofNode'] = field(default_factory=list)
    details: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "status": self.status,
            "details": self.details,
            "sub_traces": [t.to_dict() for t in self.sub_traces]
        }


class DefeasiblePrologEngine:
    """
    Defeasible Non-Monotonic Engine implementing Reiter's Default Logic
    with Negation-as-Failure and Transitive Multi-Hop ECF.
    
    Includes Safeguard 2 (Symbolic Veto catch-all) and Safeguard 3 (Loop-free DAG).
    """

    def __init__(self):
        self.rubric_steps: Dict[int, RubricStep] = {}
        self.student_steps: Dict[int, StepFrame] = {}
        self.dependencies: Dict[int, List[int]] = {}

    def reset(self):
        self.rubric_steps.clear()
        self.student_steps.clear()
        self.dependencies.clear()

    def assert_rubric_step(self, step: RubricStep):
        self.rubric_steps[step.step_id] = step

    def assert_student_step(self, step: StepFrame):
        self.student_steps[step.step_id] = step
        # Safeguard 3: Filter dependencies to enforce DAG acyclicity (Parent < Child)
        valid_parents = [p for p in step.depends_on if p < step.step_id]
        self.dependencies[step.step_id] = valid_parents

    def depends_transitively(self, child_id: int, ancestor_id: int, visited: Optional[set] = None) -> bool:
        """
        Transitive dependency closure with cycle prevention (Safeguard 3).
        """
        if visited is None:
            visited = set()
        if child_id in visited:
            return False
        visited.add(child_id)

        immediate_parents = self.dependencies.get(child_id, [])
        if ancestor_id in immediate_parents:
            return True

        for parent in immediate_parents:
            if parent < child_id: # Enforce topological direction
                if self.depends_transitively(parent, ancestor_id, visited):
                    return True
        return False

    def is_root_arithmetic_slip(self, step_id: int) -> Tuple[bool, Optional[ProofNode]]:
        """
        A step is a root arithmetic slip if its formula is valid,
        it deviated from the master rubric (RubricRes > 0.05),
        AND its local arithmetic evaluation failed (LocalRes > 0.05).
        """
        step = self.student_steps.get(step_id)
        if not step:
            return False, None

        if step.is_symbolically_valid and step.rubric_residual > 0.05 and step.local_residual > 0.05:
            node = ProofNode(
                goal=f"root_arithmetic_slip(Step_{step_id})",
                status="PROVED",
                details=f"Valid formula but arithmetic calculation slipped: RubricRes={step.rubric_residual}, LocalRes={step.local_residual}"
            )
            return True, node
        return False, None

    def check_exception_ecf(self, step_id: int) -> Tuple[bool, Optional[ProofNode]]:
        """
        Defeasible Exception 1: Multi-Hop Error Carried Forward.
        Formula is valid, student evaluated their formula correctly given upstream values (LocalRes <= 0.05),
        and the discrepancy traces back to an ancestor slip.
        """
        step = self.student_steps.get(step_id)
        if not step:
            return False, None

        if not step.is_symbolically_valid:
            return False, None

        if step.rubric_residual > 0.05 and step.local_residual <= 0.05:
            # Check all ancestors
            for ancestor_id in self.student_steps.keys():
                if ancestor_id < step_id and self.depends_transitively(step_id, ancestor_id):
                    is_slip, slip_node = self.is_root_arithmetic_slip(ancestor_id)
                    if is_slip:
                        proof = ProofNode(
                            goal=f"exception_ecf(Step_{step_id})",
                            status="PROVED",
                            details=f"Formula is algebraically valid and accurately propagated from ancestor Step {ancestor_id}.",
                            sub_traces=[
                                ProofNode(
                                    goal=f"depends_transitively(Step_{step_id}, Step_{ancestor_id})",
                                    status="PROVED"
                                ),
                                slip_node
                            ]
                        )
                        return True, proof
        return False, None

    def check_exception_rounding(self, step_id: int) -> Tuple[bool, Optional[ProofNode]]:
        """
        Defeasible Exception 2: Minor Rounding / Truncation Slip (0.0 < RubricRes <= 0.05).
        """
        step = self.student_steps.get(step_id)
        if not step:
            return False, None

        if step.is_symbolically_valid and 0.0 < step.rubric_residual <= 0.05:
            proof = ProofNode(
                goal=f"exception_rounding(Step_{step_id})",
                status="PROVED",
                details=f"Minor scalar rounding error within threshold: RubricRes={step.rubric_residual} <= 0.05"
            )
            return True, proof
        return False, None

    def evaluate_step(self, step_id: int) -> Tuple[str, float, str, ProofNode]:
        """
        Evaluates a step using Non-Monotonic Defeasible Logic (Reiter's Default Logic + NAF).
        Returns: (status_tag, credit_multiplier, natural_language_explanation, proof_tree).
        """
        step = self.student_steps.get(step_id)
        rubric = self.rubric_steps.get(step_id)

        if not step:
            return "missing_step", 0.0, "Step was omitted in student submission.", ProofNode(f"evaluate_step({step_id})", "FAILED")

        # Case 5 (Safeguard 2): Symbolic Veto Catch-All for Invalid / Contradictory Formula
        if not step.is_symbolically_valid:
            proof = ProofNode(
                goal=f"evaluate_step(Step_{step_id}, symbolic_veto, 0.0)",
                status="PROVED",
                details="Symbolic equation is algebraically invalid, dimensionally inconsistent, or contradictory.",
                sub_traces=[
                    ProofNode(goal=f"student_step(Step_{step_id}, _, false, _, _)", status="PROVED")
                ]
            )
            return "symbolic_veto", 0.0, proof.details, proof

        # Case 1: Full Credit (Exact Symbolic & Numerical Match)
        if step.rubric_residual == 0.0:
            proof = ProofNode(
                goal=f"evaluate_step(Step_{step_id}, full_credit, 1.0)",
                status="PROVED",
                details="Verified: Both symbolic derivation and scalar output match the master rubric.",
                sub_traces=[
                    ProofNode(goal=f"student_step(Step_{step_id}, Concept, true, RubricRes==0.0)", status="PROVED")
                ]
            )
            return "full_credit", 1.0, proof.details, proof

        # Case 2: Multi-Hop ECF Override
        is_ecf, ecf_node = self.check_exception_ecf(step_id)
        if is_ecf and ecf_node:
            proof = ProofNode(
                goal=f"evaluate_step(Step_{step_id}, ecf_override, 0.85)",
                status="PROVED",
                details="Defeasible Override (ECF): Numerical discrepancy is justified by correctly propagated upstream arithmetic slip.",
                sub_traces=[ecf_node]
            )
            return "ecf_override", 0.85, proof.details, proof

        # Case 3: Rounding Slip Override
        is_rounding, rounding_node = self.check_exception_rounding(step_id)
        if is_rounding and rounding_node:
            proof = ProofNode(
                goal=f"evaluate_step(Step_{step_id}, rounding_override, 0.95)",
                status="PROVED",
                details="Defeasible Override (Rounding): Valid symbolic law with acceptable scalar truncation.",
                sub_traces=[
                    rounding_node,
                    ProofNode(goal=f"\\+ exception_ecf(Step_{step_id})", status="NEGATION_VERIFIED")
                ]
            )
            return "rounding_override", 0.95, proof.details, proof

        # Case 4: Default Penalty (Failed calculation with no valid ECF excuse)
        proof = ProofNode(
            goal=f"evaluate_step(Step_{step_id}, failed_value, 0.25)",
            status="PROVED",
            details="Penalty Applied: Incorrect scalar calculation with no valid upstream ECF justification.",
            sub_traces=[
                ProofNode(goal=f"student_step(Step_{step_id}, Concept, true, RubricRes={step.rubric_residual})", status="PROVED"),
                ProofNode(goal=f"\\+ exception_ecf(Step_{step_id})", status="NEGATION_VERIFIED"),
                ProofNode(goal=f"\\+ exception_rounding(Step_{step_id})", status="NEGATION_VERIFIED")
            ]
        )
        return "failed_value", 0.25, proof.details, proof
