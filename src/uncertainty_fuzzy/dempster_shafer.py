"""
Dempster-Shafer Evidence Fusion Engine (Unit 3).
Implements Shafer's Sensor Discounting and Yager's Modified Combination Rule
to resolve the Dictator Paradox and quantify epistemic conflict K.
"""

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class EvidenceMass:
    """Belief masses assigned to subsets of Frame of Discernment Theta = {C, I}."""
    m_C: float       # Mass assigned to {Correct}
    m_I: float       # Mass assigned to {Incorrect}
    m_Theta: float   # Mass assigned to {Correct, Incorrect} (Epistemic Ignorance / Uncertainty)

    def normalize(self):
        total = self.m_C + self.m_I + self.m_Theta
        if total > 0:
            self.m_C /= total
            self.m_I /= total
            self.m_Theta /= total


class DempsterShaferFusion:
    """
    Fuses heterogeneous neural perception and symbolic mathematical proof evidence.
    Applies Shafer's sensor discounting factor and Yager's rule of combination.
    """

    def __init__(self, neural_discount: float = 0.85, symbolic_discount: float = 0.90):
        # alpha_1 for neural extractor, alpha_2 for symbolic verifier
        self.alpha_1 = neural_discount
        self.alpha_2 = symbolic_discount

    def get_neural_mass(self, confidence: float) -> EvidenceMass:
        """
        Constructs discounted mass function m1 from neural slot extractor confidence P(C).
        m1({C}) = alpha_1 * P, m1({I}) = alpha_1 * (1 - P), m1(Theta) = 1 - alpha_1
        """
        p = max(0.0, min(1.0, confidence))
        m_c = self.alpha_1 * p
        m_i = self.alpha_1 * (1.0 - p)
        m_theta = 1.0 - self.alpha_1
        return EvidenceMass(m_c, m_i, m_theta)

    def get_symbolic_mass(self, is_valid: bool) -> EvidenceMass:
        """
        Constructs discounted mass function m2 from SymPy algebraic verifier.
        Eliminates the Dictator Paradox by assigning m2(Theta) = 1 - alpha_2 = 0.10.
        """
        if is_valid:
            m_c = self.alpha_2
            m_i = 0.0
        else:
            m_c = 0.0
            m_i = self.alpha_2

        m_theta = 1.0 - self.alpha_2
        return EvidenceMass(m_c, m_i, m_theta)

    def combine_evidence_yager(
        self,
        m1: EvidenceMass,
        m2: EvidenceMass
    ) -> Tuple[EvidenceMass, float, float, float]:
        """
        Combines m1 and m2 using Yager's Modified Combination Rule.
        Returns: (combined_mass, conflict_K, belief_C, plausibility_C).
        
        Under Yager's rule, conflicting mass K is explicitly routed to m(Theta),
        preventing artificial inflation of belief under severe contradiction.
        """
        # Epistemic Conflict K
        k = (m1.m_C * m2.m_I) + (m1.m_I * m2.m_C)

        # Combined Mass on {C}
        m_c = (m1.m_C * m2.m_C) + (m1.m_C * m2.m_Theta) + (m1.m_Theta * m2.m_C)

        # Combined Mass on {I}
        m_i = (m1.m_I * m2.m_I) + (m1.m_I * m2.m_Theta) + (m1.m_Theta * m2.m_I)

        # Combined Mass on Theta (includes direct intersection + full conflict K)
        m_theta = (m1.m_Theta * m2.m_Theta) + k

        combined = EvidenceMass(round(m_c, 4), round(m_i, 4), round(m_theta, 4))

        # Belief: Bel({C}) = m({C})
        bel_c = round(m_c, 4)

        # Plausibility: Pl({C}) = m({C}) + m(Theta)
        pl_c = round(m_c + m_theta, 4)

        return combined, round(k, 4), bel_c, pl_c

    def evaluate_step_belief(
        self,
        neural_confidence: float,
        is_symbolically_valid: bool
    ) -> Tuple[float, float, float]:
        """
        End-to-end evidence fusion for a step.
        Returns: (belief_support, epistemic_conflict, plausibility).
        """
        m1 = self.get_neural_mass(neural_confidence)
        m2 = self.get_symbolic_mass(is_symbolically_valid)
        combined, k, bel_c, pl_c = self.combine_evidence_yager(m1, m2)
        return bel_c, k, pl_c
