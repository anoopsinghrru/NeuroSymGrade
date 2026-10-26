"""
NLP Text Preprocessing & Causal Dependency Parser (Unit 5).
Splits multi-line student submissions into derivation steps, extracts mathematical
equations/LaTeX blocks, scalar variable assignments, and causal connectives.
"""

import re
from typing import List, Dict, Tuple, Optional, Any


class TextPreprocessor:
    """
    Parses unstructured text submissions into candidate derivation sentences,
    isolating mathematical claims, causal connectives, and variable assignments.
    """

    CAUSAL_CONNECTIVES = [
        "therefore", "since", "because", "substituting", "substitute", "plugging in",
        "hence", "thus", "yields", "gives", "by conservation of", "according to",
        "from newtons", "from step"
    ]

    EQUATION_REGEX = re.compile(
        r'(?:(?:\$+[^\$]+\$+)|(?:[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*[^,\n;]+))'
    )

    SCALAR_ASSIGNMENT_REGEX = re.compile(
        r'(?<![a-zA-Z0-9_\*\+\-\/\^\(\)])\b([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\b'
    )

    def split_into_steps(self, raw_submission: str) -> List[Tuple[int, str]]:
        """
        Splits a submission into sequential derivation statements.
        Handles:
        - "Step 1: ...", "Step 2: ..."
        - "1. ...", "2. ..."
        - Newlines / semicolons with content
        """
        lines = [line.strip() for line in raw_submission.strip().splitlines() if line.strip()]
        steps = []
        current_step_id = 1

        for line in lines:
            # Check for explicit numbering
            match_step = re.match(r'^(?:step\s*)?(\d+)[\.\:\-\)]\s*(.*)$', line, re.IGNORECASE)
            if match_step:
                sid = int(match_step.group(1))
                content = match_step.group(2).strip()
                steps.append((sid, content if content else line))
                current_step_id = sid + 1
            else:
                steps.append((current_step_id, line))
                current_step_id += 1

        return steps

    def extract_clean_lhs(self, lhs_raw: str) -> str:
        """Isolates the mathematical variable or formula from leading English text."""
        s = lhs_raw.strip()
        # Split by punctuation like comma, colon, semicolon
        parts = re.split(r'[,:;]', s)
        candidate = parts[-1].strip()

        # Strip common leading verbal phrases
        candidate = re.sub(
            r'^(?:by|from|using|we have|therefore|since|because|clearly|the|total|net|final|initial|average|calculated|velocity|displacement|force|energy|work|potential energy|kinetic energy|normal force)\s+(?:is|gives|yields|=)?\s*',
            '', candidate, flags=re.IGNORECASE
        ).strip()

        # If candidate has "is <var>", take what follows "is"
        candidate = re.sub(r'^.*?\bis\s+', '', candidate, flags=re.IGNORECASE).strip()
        candidate = re.sub(r'^.*?\byields\s+', '', candidate, flags=re.IGNORECASE).strip()
        candidate = re.sub(r'^.*?\bgives\s+', '', candidate, flags=re.IGNORECASE).strip()

        # If multiple tokens remain, pick the last math token if it resembles a variable/formula
        tokens = candidate.split()
        if len(tokens) > 1:
            last = tokens[-1]
            if re.match(r'^[a-zA-Z_][a-zA-Z0-9_\^\*\+\-/\\]*$', last):
                return last

        return candidate if candidate else s

    def extract_equations_and_values(self, text: str) -> Tuple[Optional[str], Optional[float], Dict[str, float]]:
        """
        Extracts:
        - The primary equation string.
        - The student's calculated numerical result (the right-most scalar evaluated value).
        - Extracted scalar variable bindings (e.g. u=0, a=9.8).
        """
        bindings: Dict[str, float] = {}
        for m in self.SCALAR_ASSIGNMENT_REGEX.finditer(text):
            var_name, val_str = m.group(1), m.group(2)
            try:
                bindings[var_name] = float(val_str)
            except ValueError:
                pass

        # If text has multiple clauses separated by comma/semicolon and multiple contain '=':
        # e.g., "Combining F = ma and kinematics, F = m*(v^2 - u^2)/(2*s) = 10"
        # The primary derivation equation is the clause with the longest equation/evaluation chain.
        clauses = [c.strip() for c in re.split(r'[,;]', text) if '=' in c]
        target_clause = text
        if clauses:
            # Pick clause with most '=' signs (the derivation chain)
            target_clause = max(clauses, key=lambda c: c.count('='))

        eq_str = None
        eval_value = None

        # Look for equations containing '=' in target_clause
        parts = target_clause.split("=")
        if len(parts) >= 2:
            first_lhs = parts[0].strip()
            lhs_clean = self.extract_clean_lhs(first_lhs)
            first_rhs = parts[1].strip()
            eq_str = f"{lhs_clean} = {first_rhs}"

            # The final segment often contains the evaluated numerical answer
            last_part = parts[-1].strip()
            num_match = re.search(r'([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)', last_part)
            if num_match:
                try:
                    eval_value = float(num_match.group(1))
                except ValueError:
                    pass

        # If eval_value wasn't found from chain, check if any binding matches a target
        if eval_value is None and bindings:
            eval_value = list(bindings.values())[-1]

        return eq_str, eval_value, bindings

    def extract_causal_dependencies(self, text: str, step_id: int, prior_step_ids: List[int]) -> List[int]:
        """
        Detects if text mentions earlier steps explicitly (e.g. "from step 1", "using step 1")
        or defaults to depending on immediate predecessor (step_id - 1).
        """
        deps = []
        # Explicit reference: "step 1", "from (1)"
        for pid in prior_step_ids:
            if re.search(rf'\b(?:step|\#|\()?\s*{pid}\b', text, re.IGNORECASE):
                deps.append(pid)

        # If no explicit reference and prior step exists, assume causal continuation
        if not deps and prior_step_ids:
            deps.append(prior_step_ids[-1])

        # Ensure dependencies strictly obey topological order (pid < step_id)
        return [p for p in deps if p < step_id]
