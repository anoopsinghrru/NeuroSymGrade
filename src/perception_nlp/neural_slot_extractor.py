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

    Primary classification is keyword-density scoring (fast, deterministic, and
    exactly reproduces every documented benchmark score). When a step's wording
    matches none of the hand-curated keyword lists -- the known failure mode for
    topics outside kinematics phrasing -- it falls back to a genuine neural
    signal: sentence-embedding similarity (sentence-transformers) against a
    small set of concept exemplar sentences, rather than defaulting to a flat
    "ScientificLaw" / 0.50 guess. This only changes behavior for text that was
    previously unclassifiable; every case with an existing keyword match is
    untouched, so prior benchmark scores stay bit-for-bit identical.
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

    # Exemplar sentences for the semantic fallback -- deliberately phrased
    # differently from CONCEPT_SIGNATURES keywords so this is a genuinely
    # independent signal, not keyword matching in disguise.
    CONCEPT_EXEMPLARS = {
        "Kinematics": [
            "calculating stopping distance using initial speed and deceleration",
            "finding how far an object travels before coming to rest",
            "relating position, rate of change, and elapsed duration",
        ],
        "NewtonsSecondLaw": [
            "determining the net push or pull causing an object's motion to change",
            "relating an object's mass and its rate of speeding up",
        ],
        "WorkEnergyTheorem": [
            "energy transferred by a push acting over a distance",
            "change in an object's motion energy caused by a applied force",
        ],
        "ConservationOfEnergy": [
            "stored energy due to height converting into motion as something falls",
            "total energy of a system staying constant as it changes form",
        ],
    }
    SEMANTIC_FALLBACK_MIN_SIMILARITY = 0.25

    _embedding_model = None  # Lazy singleton shared across all instances/processes.

    def __init__(self, grounder: Optional[OWLGrounder] = None, use_semantic_fallback: bool = True):
        self.preprocessor = TextPreprocessor()
        self.grounder = grounder or OWLGrounder()
        self.use_semantic_fallback = use_semantic_fallback
        self._exemplar_embeddings = None

    def _get_embedding_model(self):
        if NeuralSlotExtractor._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            NeuralSlotExtractor._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        return NeuralSlotExtractor._embedding_model

    def _get_exemplar_embeddings(self):
        if self._exemplar_embeddings is None:
            model = self._get_embedding_model()
            self._exemplar_embeddings = {
                concept: model.encode(sentences, convert_to_numpy=True)
                for concept, sentences in self.CONCEPT_EXEMPLARS.items()
            }
        return self._exemplar_embeddings

    def _predict_via_embeddings(self, text: str) -> Tuple[Optional[str], Optional[float]]:
        """Semantic-similarity fallback classifier (Unit 5 neural upgrade)."""
        try:
            import numpy as np
            model = self._get_embedding_model()
            exemplar_embeddings = self._get_exemplar_embeddings()
            text_embedding = model.encode([text], convert_to_numpy=True)[0]
            text_norm = np.linalg.norm(text_embedding) + 1e-9

            best_concept, best_similarity = None, -1.0
            for concept, embeddings in exemplar_embeddings.items():
                norms = np.linalg.norm(embeddings, axis=1) + 1e-9
                sims = (embeddings @ text_embedding) / (norms * text_norm)
                top = float(np.max(sims))
                if top > best_similarity:
                    best_similarity, best_concept = top, concept

            if best_concept is None or best_similarity < self.SEMANTIC_FALLBACK_MIN_SIMILARITY:
                return None, None

            confidence = round(min(0.55 + best_similarity * 0.35, 0.90), 3)
            return best_concept, confidence
        except Exception:
            # Offline / model unavailable -- fail safe to the caller's default.
            return None, None

    def predict_concept_with_confidence(self, text: str) -> Tuple[str, float]:
        """
        Classifies the invoked concept and estimates semantic confidence P(Concept | Text).
        Uses normalized token frequency and keyword density, with a sentence-embedding
        fallback when no keyword in any concept's signature list is present.
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
            if self.use_semantic_fallback:
                concept, confidence = self._predict_via_embeddings(text)
                if concept is not None:
                    return concept, confidence
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
