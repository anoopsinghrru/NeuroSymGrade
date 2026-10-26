% ==============================================================================
% Defeasible Non-Monotonic Knowledge Base for Grading (Units 2 & 3)
% Implements Reiter's Default Logic and Negation-as-Failure (\+)
% with Transitive Multi-Hop Error Carried Forward (ECF) and Dual Residuals.
% ==============================================================================

:- dynamic rubric_step/4.
% rubric_step(StepID, ExpectedConcept, ExpectedValue, MaxMarks).

:- dynamic student_step/5.
% student_step(StepID, ExtractedConcept, SymbolicValid, RubricResidual, LocalResidual).

:- dynamic depends_on/2.
% depends_on(ChildStepID, ParentStepID).

% ------------------------------------------------------------------------------
% Safeguard 3: Loop-Free Transitive Dependency Closure (Acyclic DAG)
% ------------------------------------------------------------------------------
depends_transitively(Child, Ancestor) :-
    depends_on(Child, Ancestor).

depends_transitively(Child, Ancestor) :-
    depends_on(Child, Intermediate),
    Intermediate < Child, % Strict topological ordering safeguard prevents cycles
    depends_transitively(Intermediate, Ancestor).

% ------------------------------------------------------------------------------
% Root Calculation Slip Detection
% A step is a root arithmetic slip if its symbolic equation is valid,
% it deviated from the rubric (RubricRes > 0.05), AND its local evaluation failed (LocalRes > 0.05).
% ------------------------------------------------------------------------------
root_arithmetic_slip(StepID) :-
    student_step(StepID, _, true, RubricRes, LocalRes),
    RubricRes > 0.05,
    LocalRes > 0.05.

% ------------------------------------------------------------------------------
% Defeasible Exception 1: Multi-Hop Error Carried Forward (ECF)
% The formula is algebraically sound (true), student substituted their own upstream
% numbers accurately (LocalRes =< 0.05), and discrepancy traces back to an ancestor slip!
% ------------------------------------------------------------------------------
exception_ecf(StepID) :-
    student_step(StepID, _, true, RubricRes, LocalRes),
    RubricRes > 0.05,
    LocalRes =< 0.05,
    depends_transitively(StepID, AncestorStep),
    root_arithmetic_slip(AncestorStep).

% ------------------------------------------------------------------------------
% Defeasible Exception 2: Minor Rounding / Truncation Slip
% ------------------------------------------------------------------------------
exception_rounding(StepID) :-
    student_step(StepID, _, true, RubricRes, _),
    RubricRes > 0.0,
    RubricRes =< 0.05.

% ------------------------------------------------------------------------------
% Non-Monotonic Evaluation Rules (evaluate_step/4)
% Format: evaluate_step(StepID, StatusTag, CreditMultiplier, NaturalLanguageExplanation).
% ------------------------------------------------------------------------------

% Case 1: Exact Symbolic and Numerical Match
evaluate_step(StepID, full_credit, 1.0, 'Verified: Both symbolic derivation and scalar output match master rubric.') :-
    rubric_step(StepID, Concept, _, _),
    student_step(StepID, Concept, true, RubricRes, _),
    RubricRes == 0.0.

% Case 2: Non-Monotonic Override — Multi-Hop Error Carried Forward (ECF)
evaluate_step(StepID, ecf_override, 0.85, 'Defeasible Override (ECF): Numerical discrepancy is justified by correctly propagated upstream arithmetic slip.') :-
    rubric_step(StepID, Concept, _, _),
    student_step(StepID, Concept, true, RubricRes, _),
    RubricRes > 0.05,
    exception_ecf(StepID), !.

% Case 3: Non-Monotonic Override — Rounding Slip
evaluate_step(StepID, rounding_override, 0.95, 'Defeasible Override (Rounding): Valid symbolic law with acceptable scalar truncation.') :-
    rubric_step(StepID, Concept, _, _),
    student_step(StepID, Concept, true, _, _),
    exception_rounding(StepID),
    \+ exception_ecf(StepID), !.

% Case 4: Default Penalty (Failed numerical calculation with no valid ECF excuse)
evaluate_step(StepID, failed_value, 0.25, 'Penalty Applied: Incorrect scalar evaluation with no valid upstream ECF justification.') :-
    rubric_step(StepID, Concept, _, _),
    student_step(StepID, Concept, true, RubricRes, _),
    RubricRes > 0.05,
    \+ exception_ecf(StepID),
    \+ exception_rounding(StepID).

% Case 5 (Safeguard 2): Symbolic Veto Catch-All for Invalid or Contradictory Formulas
evaluate_step(StepID, symbolic_veto, 0.0, 'Symbolic Veto: Algebraic equation is invalid, dimensionally inconsistent, or contradictory.') :-
    student_step(StepID, _, false, _, _).
