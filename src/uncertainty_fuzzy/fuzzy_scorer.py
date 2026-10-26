"""
Fuzzy Logic Controller & Centroid Defuzzification Engine (Unit 2).
Translates defeasible multipliers, Dempster-Shafer belief support, and algebraic residuals
into continuous, calibrated partial-credit scores.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np


class FuzzyScorer:
    """
    Mamdani-style Fuzzy Inference Controller with Centroid Defuzzification.
    """

    def __init__(self, membership_params: Optional[np.ndarray] = None):
        """
        membership_params: Array of 18 parameters defining fuzzy set boundaries.
        If None, uses expert prior defaults.
        """
        # Default calibration parameters for triangular membership functions [a, b, c]
        if membership_params is not None and len(membership_params) == 18:
            self.params = membership_params
        else:
            # 6 fuzzy triangles: (Low, Med, High for Input), (Poor, Partial, Good for Output)
            self.params = np.array([
                # Defeasible Status [a, b, c]
                0.0, 0.25, 0.50,    # Low
                0.30, 0.60, 0.90,   # Med
                0.70, 0.90, 1.00,   # High
                # Output Score Multiplier [a, b, c]
                0.0, 0.15, 0.40,    # Low Score
                0.30, 0.60, 0.85,   # Medium Score
                0.75, 0.95, 1.00    # High Score
            ], dtype=np.float64)

    @staticmethod
    def trimf(x: float, a: float, b: float, c: float) -> float:
        """Triangular membership function evaluation."""
        if x <= a or x >= c:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a) if b > a else 1.0
        else:
            return (c - x) / (c - b) if c > b else 1.0

    @staticmethod
    def shoulder_high(z: float, a: float, b: float, c: float) -> float:
        """Right-shoulder membership for High output set."""
        if z <= a:
            return 0.0
        elif a < z < b:
            return (z - a) / (b - a) if b > a else 1.0
        else:
            return 1.0

    def evaluate_fuzzy_score(
        self,
        defeasible_multiplier: float,
        belief_support: float,
        local_residual: float,
        max_marks: float
    ) -> float:
        """
        Fuzzy inference combining Prolog defeasible status and Dempster-Shafer belief.
        Defuzzifies output via Centroid of Area (COA).
        """
        # If Prolog executed a hard symbolic veto (e.g. invalid equation), award 0.0 directly
        if defeasible_multiplier == 0.0:
            return 0.0

        # Calculate residual penalty factor
        res_factor = max(0.0, min(1.0, 1.0 - (local_residual / 5.0)))
        
        # Combined crisp input index
        x_in = (0.60 * defeasible_multiplier) + (0.30 * belief_support) + (0.10 * res_factor)
        x_in = max(0.0, min(1.0, x_in))

        # Decode output membership triangles from self.params
        out_low_a, out_low_b, out_low_c = self.params[9], self.params[10], self.params[11]
        out_med_a, out_med_b, out_med_c = self.params[12], self.params[13], self.params[14]
        out_high_a, out_high_b, out_high_c = self.params[15], self.params[16], self.params[17]

        # Input membership activations
        mu_low = self.trimf(x_in, self.params[0], self.params[1], self.params[2])
        mu_med = self.trimf(x_in, self.params[3], self.params[4], self.params[5])
        mu_high = self.shoulder_high(x_in, self.params[6], self.params[7], self.params[8])

        # Centroid Defuzzification across discrete universe z in [0.0, 1.0]
        z_vals = np.linspace(0.0, 1.0, 101)
        numerator = 0.0
        denominator = 0.0

        for z in z_vals:
            # Rule Aggregation (Mamdani Min-Max)
            # Rule 1: IF Low THEN Low
            # Rule 2: IF Med THEN Med
            # Rule 3: IF High THEN High
            w1 = min(mu_low, self.trimf(z, out_low_a, out_low_b, out_low_c))
            w2 = min(mu_med, self.trimf(z, out_med_a, out_med_b, out_med_c))
            w3 = min(mu_high, self.shoulder_high(z, out_high_a, out_high_b, out_high_c))
            mu_agg = max(w1, w2, w3)

            numerator += z * mu_agg
            denominator += mu_agg

        if denominator > 1e-6:
            z_star = numerator / denominator
        else:
            # Fallback to direct input index
            z_star = x_in

        # Bound and scale by max marks
        z_star = max(0.0, min(1.0, z_star))
        final_marks = round(z_star * max_marks, 2)
        return final_marks
