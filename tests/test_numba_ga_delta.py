"""
Unit tests for Delta-Encoded Numba-Accelerated Genetic Algorithm and Fuzzy Scorer.
"""

import pytest
import numpy as np
from src.uncertainty_fuzzy.ga_calibrator import (
    GeneticAlgorithmCalibrator,
    decode_delta_chromosome
)
from src.uncertainty_fuzzy.fuzzy_scorer import FuzzyScorer


def test_delta_chromosome_guarantees_monotonic_triangles():
    """
    Guarantees a < b < c for 100% of decoded triangles, even with extreme noisy genes!
    """
    raw_noisy_chromosome = np.array([
        0.0, 0.20, 0.30,   # Triangle 1
        0.2, 0.01, 0.05,   # Triangle 2 (tiny deltas)
        0.5, 0.15, 0.25,   # Triangle 3
        0.0, 0.10, 0.20,   # Triangle 4
        0.3, 0.30, 0.20,   # Triangle 5
        0.7, 0.10, 0.15    # Triangle 6
    ], dtype=np.float64)

    decoded = decode_delta_chromosome(raw_noisy_chromosome)

    for k in range(6):
        a = decoded[k * 3]
        b = decoded[k * 3 + 1]
        c = decoded[k * 3 + 2]
        # Strictly monotonic inequality must hold!
        assert a < b, f"Triangle {k} violated a < b: a={a}, b={b}"
        assert b < c, f"Triangle {k} violated b < c: b={b}, c={c}"


def test_ga_calibration_and_fuzzy_scoring():
    """
    Runs quick GA calibration and verifies fuzzy scorer output on all defeasible statuses.
    """
    calibrator = GeneticAlgorithmCalibrator(pop_size=20, generations=10)
    optimal_params, history = calibrator.run_calibration()

    assert len(optimal_params) == 18
    assert len(history) == 10
    # Convergence: final fitness should be greater than or equal to initial fitness
    assert history[-1] >= history[0]

    scorer = FuzzyScorer(membership_params=optimal_params)

    # Full credit test (10 max marks)
    full_score = scorer.evaluate_fuzzy_score(
        defeasible_multiplier=1.0,
        belief_support=0.90,
        local_residual=0.0,
        max_marks=10.0
    )
    assert full_score >= 8.5

    # Symbolic veto test (must be 0.0)
    veto_score = scorer.evaluate_fuzzy_score(
        defeasible_multiplier=0.0,
        belief_support=0.20,
        local_residual=10.0,
        max_marks=10.0
    )
    assert veto_score == 0.0

    # ECF partial credit test
    ecf_score = scorer.evaluate_fuzzy_score(
        defeasible_multiplier=0.85,
        belief_support=0.80,
        local_residual=0.0,
        max_marks=10.0
    )
    assert 7.0 <= ecf_score <= 9.0
