"""
Neural Slot Extractor & Concept Classifier (Unit 5).
Maps unstructured derivation statements to domain concepts and populates
structured StepFrame objects (Unit 4).
"""

import math
import re
from typing import List, Dict, Tuple, Optional
from ..models.step_frame import StepFrame
from .text_preprocessor import TextPreprocessor
from ..symbolic_math.owl_grounding import OWLGrounder


class NeuralSlotExtractor:
    """
    Perception model converting unstructured student sentences into StepFrames.
    Estimates concept classification probabilities and extracts equation slots.
    """

    CONCEPT_SIGNATURES = {
        "Kinematics": [
            "velocity", "acceleration", "displacement", "speed", "initial velocity",
            "kinematic", "motion", "gravity", "v^2", "u^2", "at", "ut", "time"
        ],
        "NewtonsSecondLaw": [
            "newton", "second law", "force", "mass", "f = ma", "f=ma", "net force",
            "sigma f", "tension", "normal force", "friction"
        ],
        "WorkEnergyTheorem": [
            "work", "work-energy", "work energy", "w = delta ke", "w = f*d", "w=f*s",
            "kinetic energy change", "force times distance"
        ],
        "ConservationOfEnergy": [
            "conservation of energy", "total energy", "ke + pe", "potential energy",
            "mechanical energy", "initial energy", "final energy", "mgh", "0.5*m*v^2"
        ]
    }

    def __init__(self, grounder: Optional[OWLGrounder] = None):
        self.preprocessor = TextPreprocessor()
        self.grounder = grounder or OWLGrounder()

    def predict_concept_with_confidence(self, text: str) -> Tuple[str, float]:
        """
        Classifies the invoked concept and estimates semantic confidence P(Concept | Text).
        Uses normalized token frequency and keyword density.
        """
        lower_text = text.lower()
        scores = {}

        for concept, keywords in self.CONCEPT_SIGNATURES.items():
            matches = 0
            for kw in keywords:
                if kw in lower_text:
                    matches += 1
            scores[concept] = matches

        total_matches = sum(scores.values())
        if total_matches == 0:
            # Fallback to general physical law with baseline prior
            return "ScientificLaw", 0.50

        best_concept = max(scores, key=scores.get)
        best_count = scores[best_concept]

        # Softmax / Sigmoidal confidence scaling
        confidence = min(0.60 + (best_count * 0.12), 0.95)
        return best_concept, round(confidence, 3)

    def extract_step_frame(
        self,
        step_id: int,
        raw_text: str,
        prior_step_ids: List[int]
    ) -> StepFrame:
        """
        Converts a single student statement into a structured StepFrame.
        """
        concept, confidence = self.predict_concept_with_confidence(raw_text)
        eq_str, eval_val, bindings = self.preprocessor.extract_equations_and_values(raw_text)
        deps = self.preprocessor.extract_causal_dependencies(raw_text, step_id, prior_step_ids)

        # Ground bindings
        canonical_bindings = self.grounder.canonicalize_bindings(bindings)

        frame = StepFrame(
            step_id=step_id,
            raw_text=raw_text,
            invoked_concept=concept,
            raw_equation=eq_str,
            variable_bindings=canonical_bindings,
            evaluated_value=eval_val,
            depends_on=deps,
            neural_confidence=confidence
        )
        return frame

    def parse_submission(self, raw_submission: str) -> List[StepFrame]:
        """
        Parses a full multi-step submission into an ordered list of StepFrames.
        """
        raw_steps = self.preprocessor.split_into_steps(raw_submission)
        step_frames = []
        prior_ids = []

        for sid, text in raw_steps:
            frame = self.extract_step_frame(sid, text, prior_ids)
            step_frames.append(frame)
            prior_ids.append(sid)

        return step_frames
