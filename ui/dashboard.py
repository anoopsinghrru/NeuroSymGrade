"""
Interactive Visualizer & Dashboard Generator for Viva Evaluations (Unit 6).
Generates ASCII deduction trees and Matplotlib diagnostic plots:
- Genetic Algorithm Convergence Curve
- Fuzzy Membership Function Curves
- Dempster-Shafer Uncertainty Intervals (Belief vs. Plausibility)
"""

import os
from typing import Dict, List, Optional
import numpy as np
import matplotlib
matplotlib.use('Agg') # Headless backend
import matplotlib.pyplot as plt

from src.models.step_frame import DerivationEvaluationSummary, StepFrame
from src.reasoning_engine.explanation_facility import ExplanationFacility
from src.reasoning_engine.prolog_bridge import ProofNode


class EvaluationVisualizer:
    """
    Renders diagnostic charts and interactive reports for viva evaluation.
    """

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(__file__), "static"
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_ga_convergence(self, history: List[float], filename: str = "ga_convergence.png") -> str:
        """Plots GA fitness improvement over generations."""
        plt.figure(figsize=(8, 4.5), dpi=150)
        generations = range(1, len(history) + 1)
        plt.plot(generations, history, marker='o', color='#2563eb', linewidth=2, markersize=4, label='Best Population Fitness')
        plt.title("Numba-Accelerated Genetic Algorithm: Fitness Convergence (Unit 6)", fontsize=12, fontweight='bold')
        plt.xlabel("Generation", fontsize=10)
        plt.ylabel("Fitness = 1 / (1 + MSE + L2_reg)", fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(frameon=True)
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, filename)
        plt.savefig(out_path)
        plt.close()
        return out_path

    def plot_fuzzy_memberships(self, params: np.ndarray, filename: str = "fuzzy_memberships.png") -> str:
        """Plots the calibrated triangular and shoulder fuzzy sets."""
        z = np.linspace(0.0, 1.0, 200)
        plt.figure(figsize=(9, 5), dpi=150)

        # Output Fuzzy Sets
        low_a, low_b, low_c = params[9], params[10], params[11]
        med_a, med_b, med_c = params[12], params[13], params[14]
        high_a, high_b, high_c = params[15], params[16], params[17]

        def trimf(x, a, b, c):
            return np.maximum(0.0, np.minimum((x - a) / max(b - a, 1e-5), (c - x) / max(c - b, 1e-5)))

        def right_shoulder(x, a, b):
            return np.where(x <= a, 0.0, np.where(x < b, (x - a) / max(b - a, 1e-5), 1.0))

        mu_low = trimf(z, low_a, low_b, low_c)
        mu_med = trimf(z, med_a, med_b, med_c)
        mu_high = right_shoulder(z, high_a, high_b)

        plt.plot(z, mu_low, label=f'Low (Poor) [{low_a:.2f}, {low_b:.2f}, {low_c:.2f}]', color='#dc2626', linewidth=2)
        plt.plot(z, mu_med, label=f'Med (Partial) [{med_a:.2f}, {med_b:.2f}, {med_c:.2f}]', color='#d97706', linewidth=2)
        plt.plot(z, mu_high, label=f'High (Exact) [{high_a:.2f}, {high_b:.2f}, 1.00]', color='#16a34a', linewidth=2)

        plt.title("Calibrated Fuzzy Membership Functions with Centroid Support (Units 2 & 6)", fontsize=12, fontweight='bold')
        plt.xlabel("Universe of Discourse: Awarded Fraction (z)", fontsize=10)
        plt.ylabel("Degree of Membership (μ)", fontsize=10)
        plt.ylim(-0.05, 1.05)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(frameon=True)
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, filename)
        plt.savefig(out_path)
        plt.close()
        return out_path

    def plot_dempster_shafer_intervals(
        self,
        step_evaluations: List[StepFrame],
        filename: str = "dempster_shafer_intervals.png"
    ) -> str:
        """Plots Belief vs. Plausibility intervals and Epistemic Conflict K for each step."""
        plt.figure(figsize=(8, 4.5), dpi=150)
        sids = [f"Step {s.step_id}" for s in step_evaluations]
        beliefs = [s.belief_support for s in step_evaluations]
        conflicts = [s.epistemic_conflict for s in step_evaluations]

        x = np.arange(len(sids))
        width = 0.35

        plt.bar(x - width/2, beliefs, width, label='Belief Bel({C})', color='#2563eb')
        plt.bar(x + width/2, conflicts, width, label='Conflict K (Yager)', color='#ef4444')

        plt.title("Dempster-Shafer Evidence Fusion: Belief Support vs. Epistemic Conflict (Unit 3)", fontsize=12, fontweight='bold')
        plt.xlabel("Derivation Steps", fontsize=10)
        plt.ylabel("Metric [0.0 - 1.0]", fontsize=10)
        plt.xticks(x, sids)
        plt.ylim(0.0, 1.05)
        plt.grid(True, linestyle='--', alpha=0.6, axis='y')
        plt.legend(frameon=True)
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, filename)
        plt.savefig(out_path)
        plt.close()
        return out_path
