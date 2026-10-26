r"""
OWL Grounding & Symbol Canonicalization Engine (Unit 4).
Maps varied student surface symbols (e.g. \Delta y, h, d, v_0, u) into canonical
SymPy symbols based on physical quantity taxonomies defined in config/rubric_ontology.owl.
"""

import os
import re
from typing import Dict, Optional, Tuple, Set
import xml.etree.ElementTree as ET


class OWLGrounder:
    """
    Grounds surface tokens and mathematical symbols to canonical domain variables
    using OWL ontology classes and alias mappings.
    """
    
    DEFAULT_CANONICAL_MAP = {
        # Displacement / Distance -> s
        "s": "s", "d": "s", "x": "s", "y": "s", "h": "s", 
        "delta_x": "s", "delta_y": "s", "\\Delta x": "s", "\\Delta y": "s",
        "deltax": "s", "deltay": "s", "Delta_y": "s", "Delta_x": "s",
        # Initial Velocity -> u
        "u": "u", "v0": "u", "v_0": "u", "v_i": "u", "vi": "u", "u_x": "u", "u_y": "u",
        # Final Velocity -> v (capital V kept distinct from U, which denotes PotentialEnergy)
        "v": "v", "V": "v", "vf": "v", "v_f": "v", "v_x": "v", "v_y": "v",
        # Acceleration / Gravity -> a
        "a": "a", "g": "a", "a_x": "a", "a_y": "a",
        # Time -> t
        "t": "t", "delta_t": "t", "\\Delta t": "t", "T": "t",
        # Force -> F
        "F": "F", "f": "F", "F_net": "F", "Fnet": "F", "F_ext": "F", "T_tension": "F",
        # Mass -> m
        "m": "m", "M": "m", "m1": "m", "m2": "m",
        # Work -> W
        "W": "W", "w": "W",
        # Kinetic Energy -> KE
        "KE": "KE", "K": "KE", "E_k": "KE", "Ek": "KE",
        # Potential Energy -> PE
        "PE": "PE", "U": "PE", "E_p": "PE", "Ep": "PE",
    }

    # Expected SI unit per canonical symbol (Unit 4 ontology: physical quantity -> unit).
    # Used by check_unit() for optional, backward-compatible dimensional-analysis grading.
    CANONICAL_UNIT_MAP = {
        "v": "m/s", "u": "m/s", "a": "m/s^2", "t": "s", "s": "m",
        "F": "N", "m": "kg", "W": "J", "KE": "J", "PE": "J",
    }

    # Accepted spellings for each canonical SI unit. Matching is case-insensitive
    # and whitespace-insensitive (see check_unit).
    UNIT_ALIASES = {
        "m/s": ["m/s", "ms^-1", "ms-1", "m s^-1", "meters/second", "meters per second", "m per s"],
        "m/s^2": ["m/s^2", "m/s2", "ms^-2", "m s^-2", "meters/second^2", "meters per second squared"],
        "s": ["s", "sec", "secs", "second", "seconds"],
        "m": ["m", "meter", "meters", "metre", "metres"],
        "N": ["n", "newton", "newtons"],
        "kg": ["kg", "kilogram", "kilograms"],
        "J": ["j", "joule", "joules"],
    }

    # Trailing unit token after the final numeric value in a derivation statement,
    # e.g. "... = 20 m/s" -> "m/s", "... = 400 J" -> "J".
    TRAILING_UNIT_REGEX = re.compile(
        r'[+-]?\d+\.?\d*(?:[eE][+-]?\d+)?\s*([a-zA-Z](?:[a-zA-Z0-9\^/\-\s]{0,12}[a-zA-Z0-9\^])?)\s*$'
    )

    def extract_trailing_unit(self, text: str) -> Optional[str]:
        """Extracts the unit token (if any) immediately following the final number in text."""
        stripped = text.strip().rstrip(".")
        match = self.TRAILING_UNIT_REGEX.search(stripped)
        if not match:
            return None
        token = match.group(1).strip()
        return token or None

    def check_unit(self, canonical_symbol: str, raw_text: str) -> Tuple[Optional[bool], Optional[str]]:
        """
        Checks whether raw_text declares the expected SI unit for canonical_symbol.
        Returns (matched, extracted_unit):
        - (None, None)   -- this symbol has no expected unit registered (nothing to check).
        - (False, None)  -- a unit was expected but none was found in the text.
        - (True/False, extracted_unit) -- a unit token was found; whether it matches.
        """
        expected = self.CANONICAL_UNIT_MAP.get(canonical_symbol)
        if expected is None:
            return None, None

        extracted = self.extract_trailing_unit(raw_text)
        if extracted is None:
            return False, None

        accepted = {alias.lower().replace(" ", "") for alias in self.UNIT_ALIASES.get(expected, [expected])}
        normalized = extracted.lower().replace(" ", "")
        return normalized in accepted, extracted

    CONCEPT_TO_LAW = {
        "kinematics": "Kinematics",
        "motion": "Kinematics",
        "newton": "NewtonsSecondLaw",
        "second_law": "NewtonsSecondLaw",
        "force": "NewtonsSecondLaw",
        "energy": "ConservationOfEnergy",
        "conservation_of_energy": "ConservationOfEnergy",
        "work_energy": "WorkEnergyTheorem",
        "work": "WorkEnergyTheorem"
    }

    def __init__(self, ontology_path: Optional[str] = None):
        self.canonical_map: Dict[str, str] = dict(self.DEFAULT_CANONICAL_MAP)
        self.ontology_path = ontology_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "config", "rubric_ontology.owl"
        )
        self._load_ontology()

    def _load_ontology(self):
        """Parse OWL file to augment symbol aliases and hierarchy."""
        if not os.path.exists(self.ontology_path):
            return
        try:
            tree = ET.parse(self.ontology_path)
            root = tree.getroot()
            namespaces = {
                'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
                'owl': 'http://www.w3.org/2002/07/owl#',
                'rdfs': 'http://www.w3.org/2000/01/rdf-schema#'
            }
            for cls in root.findall('owl:Class', namespaces):
                comment = cls.find('rdfs:comment', namespaces)
                label = cls.find('rdfs:label', namespaces)
                if comment is not None and comment.text and "aliases:" in comment.text:
                    aliases_str = comment.text.replace("aliases:", "").strip()
                    aliases = [a.strip() for a in aliases_str.split(",") if a.strip()]
                    cls_name = label.text if label is not None else ""
                    # Pick canonical representative
                    target = self._class_to_canonical_symbol(cls_name)
                    if target:
                        for alias in aliases:
                            self.canonical_map[alias] = target
        except Exception:
            # Fallback gracefully to default map
            pass

    def _class_to_canonical_symbol(self, class_name: str) -> Optional[str]:
        mapping = {
            "Displacement": "s",
            "InitialVelocity": "u",
            "FinalVelocity": "v",
            "Acceleration": "a",
            "Time": "t",
            "Force": "F",
            "Mass": "m",
            "Work": "W",
            "KineticEnergy": "KE",
            "PotentialEnergy": "PE",
        }
        return mapping.get(class_name)

    def canonicalize_symbol(self, symbol_str: str) -> str:
        """Map a single surface symbol string to its canonical equivalent."""
        cleaned = symbol_str.strip().replace(" ", "")
        return self.canonical_map.get(cleaned, cleaned)

    def canonicalize_equation_string(self, equation_str: str) -> str:
        """
        Rewrites a student equation string replacing all known aliases with canonical symbols.
        Respects word boundaries and LaTeX tokens.
        """
        eq = equation_str.strip()
        # Sort keys by length descending to avoid partial prefix collisions (e.g. delta_x before x)
        sorted_keys = sorted(self.canonical_map.keys(), key=lambda k: len(k), reverse=True)
        for alias in sorted_keys:
            canonical = self.canonical_map[alias]
            if alias == canonical:
                continue
            # Regex pattern with word boundary or non-alphanumeric boundary
            escaped_alias = re.escape(alias)
            pattern = rf'(?<![a-zA-Z0-9_]){escaped_alias}(?![a-zA-Z0-9_])'
            eq = re.sub(pattern, canonical, eq)
        return eq

    def canonicalize_bindings(self, bindings: Dict[str, float]) -> Dict[str, float]:
        """Convert a dictionary of student variable bindings into canonical keys."""
        canonical_b = {}
        for k, v in bindings.items():
            can_k = self.canonicalize_symbol(k)
            canonical_b[can_k] = float(v)
        return canonical_b

    def ground_concept(self, text_or_concept: str) -> str:
        """Map an extracted concept phrase to a formal OWL class."""
        normalized = text_or_concept.lower().strip().replace(" ", "_").replace("'", "")
        for key, law in self.CONCEPT_TO_LAW.items():
            if key in normalized:
                return law
        return "ScientificLaw"
