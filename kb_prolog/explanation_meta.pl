% ==============================================================================
% Backward-Chaining Meta-Interpreter with Proof Trace Facility (Unit 6)
% Captures resolution proof trees to explain WHY marks were awarded or deducted.
% ==============================================================================

% Base Cases
prove_with_trace(true, [fact(true)]) :- !.

% Conjunction: Prove A, then prove B, and concatenate their proof traces
prove_with_trace((A, B), [conjunction(TraceA, TraceB)]) :- !,
    prove_with_trace(A, TraceA),
    prove_with_trace(B, TraceB).

% Negation as Failure (\+ Goal)
prove_with_trace(\+ Goal, [negation_verified(Goal)]) :-
    \+ Goal, !.

% Arithmetic comparisons and built-in evaluators
prove_with_trace(A > B, [verified(A > B)]) :-
    A > B, !.

prove_with_trace(A =< B, [verified(A =< B)]) :-
    A =< B, !.

prove_with_trace(A == B, [verified(A == B)]) :-
    A == B, !.

prove_with_trace(A \== B, [verified(A \== B)]) :-
    A \== B, !.

% Cut (!) handling
prove_with_trace(!, [cut_passed]) :- !.

% General Rule Unification: Clause resolution
prove_with_trace(Goal, [proved(Goal, SubTrace)]) :-
    clause(Goal, Body),
    prove_with_trace(Body, SubTrace).
