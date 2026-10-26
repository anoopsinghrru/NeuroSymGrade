"""
Core data models for Neuro-Symbolic Verification Engine.
Defines StepFrames (Unit 4 Slot-and-Filler Frames), Rubric Step specifications,
and Verification Results.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import sympy as sp


@dataclass
class StepFrame:
    """
    Slot-and-Filler Frame representing a single student derivation step (Unit 4).
    Stores parsed linguistic, symbolic, and quantitative evidence.
    """
    step_id: int
    raw_text: str
    invoked_concept: Optional[str] = None          # Grounded OWL ontology class (e.g. 'Kinematics')
    raw_equation: Optional[str] = None             # Raw equation string extracted from text
    canonical_lhs: Optional[sp.Expr] = None        # Canonicalized SymPy Left-Hand Side
    canonical_rhs: Optional[sp.Expr] = None        # Canonicalized SymPy Right-Hand Side
    variable_bindings: Dict[str, float] = field(default_factory=dict) # Extracted scalar assignments
    evaluated_value: Optional[float] = None        # Student's calculated numerical result
    
    # Dual Residuals (Mathematical Upgrade for Transitive ECF)
    rubric_residual: float = 0.0                   # |V_student - V_master_rubric|
    local_residual: float = 0.0                    # |V_student - Eval(Formula, upstream_bindings)|
    
    # Verification Flags
    is_symbolically_valid: bool = False
    is_algebraic_leap: bool = False                # True if validated via Gröbner basis ideal membership
    skipped_intermediate_steps: List[int] = field(default_factory=list)
    
    # Dependency Tracking (Unit 4 Partial-Order Planning & Unit 2 Prolog)
    depends_on: List[int] = field(default_factory=list) # Immediate parent step IDs
    
    # Uncertainty & Scoring (Unit 3 & 5)
    neural_confidence: float = 0.0                 # Probability from neural slot extractor [0, 1]
    belief_support: float = 0.0                    # Dempster-Shafer Bel({C})
    epistemic_conflict: float = 0.0                # Dempster-Shafer K (or Yager uncertainty)
    defeasible_status: str = "unverified"          # 'full_credit', 'ecf_override', 'rounding_override', 'failed_value', 'symbolic_veto'
    credit_multiplier: float = 0.0                 # Output multiplier from Prolog [0.0, 1.0]
    final_score: float = 0.0                       # Defuzzified mark awarded for this step
    explanation: str = ""                          # Natural language audit trace from Explanation Facility

    # Dimensional Analysis (Unit 4 ontology; opt-in per rubric step via require_units)
    unit_required: bool = False
    unit_provided: Optional[str] = None            # Raw unit token extracted from student text, if any
    unit_valid: Optional[bool] = None               # None = not checked; True/False = checked result


@dataclass
class RubricStep:
    """Specification of an expected derivation milestone in the rubric."""
    step_id: int
    concept: str                                   # Expected OWL ontology class
    canonical_equation: str                        # Canonical equation string (e.g. 'v^2 - u^2 = 2*a*s')
    expected_bindings: Dict[str, float]            # Gold-standard variable values
    expected_output_value: float                   # Expected numerical scalar output
    max_marks: float                               # Maximum marks available for this step
    prerequisites: List[int] = field(default_factory=list) # Steps that must causally precede this step
    is_leaf: bool = False
    require_units: bool = False                     # Opt-in dimensional-analysis check (Unit 4 ontology)


@dataclass
class RubricStrategy:
    """Represents one valid solution branch (OR node) in an AND-OR Rubric Graph."""
    strategy_id: str
    name: str                                      # e.g., 'Kinematics_Method' or 'Work_Energy_Method'
    description: str
    steps: List[RubricStep]
    and_dependency_dag: Dict[int, List[int]] = field(default_factory=dict) # Partial-order causal constraints


@dataclass
class DerivationEvaluationSummary:
    """End-to-end evaluation report for an entire student derivation submission."""
    submission_id: str
    selected_strategy: str
    alignment_cost: float                          # AO* search optimal cost
    total_marks_awarded: float
    max_possible_marks: float
    percentage: float
    step_evaluations: List[StepFrame]
    global_explanation_tree: Dict[str, Any]
