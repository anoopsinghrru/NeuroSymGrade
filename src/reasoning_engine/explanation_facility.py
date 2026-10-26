"""
Explanation Facility for the Verification & Grading Engine (Unit 6).
Generates human-readable audit trails, renders ASCII/Markdown deduction trees,
and supports interactive WHY and HOW queries on proof trees.
"""

from typing import Dict, List, Any, Optional
from ..models.step_frame import StepFrame, DerivationEvaluationSummary
from .prolog_bridge import ProofNode


class ExplanationFacility:
    """
    Expert System Explanation Facility providing:
    - WHY did step K receive its grade?
    - HOW was the logical conclusion or defeasible exception proved?
    - Formatted ASCII/Markdown proof trees for student feedback and examiner viva.
    """

    @staticmethod
    def render_proof_tree_ascii(node: ProofNode, prefix: str = "", is_last: bool = True) -> str:
        """Renders a ProofNode hierarchy as a visual ASCII deduction tree."""
        connector = "+-- " if is_last else "|-- "
        lines = []
        
        status_symbol = {
            "PROVED": "[PROVED]",
            "NEGATION_VERIFIED": "[NOT-PROVED]",
            "FAILED": "[FAILED]"
        }.get(node.status, f"[{node.status}]")

        detail_text = f" -> {node.details}" if node.details else ""
        lines.append(f"{prefix}{connector}{status_symbol} {node.goal}{detail_text}")

        new_prefix = prefix + ("    " if is_last else "|   ")
        count = len(node.sub_traces)
        for i, child in enumerate(node.sub_traces):
            lines.append(ExplanationFacility.render_proof_tree_ascii(child, new_prefix, i == count - 1))

        return "\n".join(lines)

    @staticmethod
    def answer_why(step_frame: StepFrame, proof_tree: Optional[ProofNode]) -> str:
        """
        Answers the question: WHY did step K receive this specific score/credit multiplier?
        """
        sid = step_frame.step_id
        status = step_frame.defeasible_status
        multiplier = step_frame.credit_multiplier

        if status == "full_credit":
            return (
                f"Step {sid} received FULL CREDIT (100%) because its algebraic formula "
                f"matches the canonical law and its calculated value matched the master rubric exactly (residual = 0.0)."
            )
        elif status == "ecf_override":
            return (
                f"Step {sid} received DEFEASIBLE ECF CREDIT ({int(multiplier*100)}%) despite a rubric discrepancy. "
                f"Defeasible Exception 1 fired because the student's formula was algebraically sound and correctly "
                f"propagated a prior arithmetic slip from an upstream ancestor (LocalResidual = {step_frame.local_residual} <= 0.05)."
            )
        elif status == "rounding_override":
            return (
                f"Step {sid} received ROUNDING CREDIT ({int(multiplier*100)}%) because the algebraic formula was valid "
                f"and the numerical deviation ({step_frame.rubric_residual}) is within acceptable floating-point tolerance (<= 0.05)."
            )
        elif status == "symbolic_veto":
            return (
                f"Step {sid} received ZERO CREDIT (0%) via SYMBOLIC VETO. The equation entered is algebraically invalid, "
                f"contradictory, or dimensionally inconsistent, overriding any surface textual fluency."
            )
        elif status == "failed_value":
            return (
                f"Step {sid} received a DEFAULT PENALTY ({int(multiplier*100)}%). Although the formula was recognized, "
                f"the scalar calculation was incorrect (residual = {step_frame.rubric_residual}) and could not be justified "
                f"by any prior upstream error."
            )
        return f"Step {sid} status: {status}."

    @staticmethod
    def answer_how(proof_tree: ProofNode) -> str:
        """
        Answers the question: HOW was the conclusion reached? (Unfolds the backward-chaining proof trace).
        """
        return ExplanationFacility.render_proof_tree_ascii(proof_tree)

    @staticmethod
    def generate_full_audit_report(summary: DerivationEvaluationSummary, proof_trees: Dict[int, ProofNode]) -> str:
        """
        Generates a comprehensive Markdown audit report suitable for examiners and students.
        """
        lines = [
            f"# Verification & Grading Audit Report: {summary.submission_id}",
            f"**Selected Strategy:** {summary.selected_strategy}",
            f"**AO* Alignment Cost:** {summary.alignment_cost:.3f}",
            f"**Total Awarded Marks:** {summary.total_marks_awarded:.2f} / {summary.max_possible_marks:.2f} ({summary.percentage:.1f}%)",
            "",
            "## Step-by-Step Causal Deduction Trace",
            ""
        ]

        for step in summary.step_evaluations:
            tree = proof_trees.get(step.step_id)
            lines.append(f"### Step {step.step_id}: {step.invoked_concept or 'General Claim'}")
            lines.append(f"- **Raw Text:** *\"{step.raw_text}\"*")
            lines.append(f"- **Equation:** `{step.raw_equation or 'N/A'}`")
            lines.append(f"- **Symbolic Valid:** `{'True' if step.is_symbolically_valid else 'False'}`" + (" (Algebraic Leap via Gröbner Ideal)" if step.is_algebraic_leap else ""))
            lines.append(f"- **Evaluated Value:** `{step.evaluated_value}` | **Rubric Res:** `{step.rubric_residual}` | **Local Res:** `{step.local_residual}`")
            lines.append(f"- **Dempster-Shafer Conflict K:** `{step.epistemic_conflict:.3f}` | **Belief:** `{step.belief_support:.3f}`")
            lines.append(f"- **Defeasible Status:** `{step.defeasible_status}` (Multiplier: {step.credit_multiplier})")
            lines.append(f"- **Step Marks:** **{step.final_score:.2f}**")
            lines.append(f"- **Explanation:** {ExplanationFacility.answer_why(step, tree)}")
            if tree:
                lines.append("```text")
                lines.append(ExplanationFacility.render_proof_tree_ascii(tree))
                lines.append("```")
            lines.append("")

        return "\n".join(lines)
