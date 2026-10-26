# NeuroSymGrade

### Neuro-Symbolic Verification & Grading Engine with Non-Monotonic Logic

A neuro-symbolic grading and causal verification engine designed to overcome the critical failure modes of standard LLM evaluators and rigid script graders in rigorous STEM derivations.

The engine translates unstructured student text, LaTeX, or code derivations into formal **Slot-and-Filler StepFrames**, aligns multi-step strategies over an **AND-OR Rubric Graph**, verifies algebraic equivalence and polynomial ideal membership using **SymPy Gröbner Bases**, fuses uncertain perception with symbolic proof via **Discounted Dempster-Shafer (Yager's Rule)**, evaluates **Transitive Multi-Hop Error Carried Forward (ECF)** with **Prolog Defeasible Logic**, and calibrates fuzzy partial-credit boundaries using a **Numba-Accelerated Genetic Algorithm**.

---

## 1. Syllabus Mapping (Units 1–6)

| Syllabus Unit | AI Methodology Implemented | Project Module | Functional Role |
| :--- | :--- | :--- | :--- |
| **Unit 1: Search** | **AO\* Search**, AND-OR Solution Graph, Heuristic Cost Functions | `src/search_alignment/aostar_rubric_matcher.py` | Automatically discovers which valid solution strategy the student attempted (OR branch) and evaluates sequential prerequisites (AND branch). |
| **Unit 2: Logic & Languages** | **Predicate Logic**, **Non-Monotonic Logic**, **Fuzzy Logic**, SymPy, Prolog | `src/symbolic_math/sympy_verifier.py`, `kb_prolog/defeasible_rules.pl` | Canonical symbolic equivalence, Gröbner polynomial ideal membership, default rules, and continuous partial-credit scoring. |
| **Unit 3: Knowledge & Uncertainty** | **Dempster-Shafer Theory**, Epistemic Conflict $K$, Yager's Combination Rule | `src/uncertainty_fuzzy/dempster_shafer.py` | Fuses noisy neural confidence ($m_1$) with discounted symbolic verification ($m_2$). High conflict $K$ triggers symbolic veto to defeat LLM hallucination. |
| **Unit 4: Slot-and-Filler & Planning** | **Frames**, **OWL Ontology Grounding**, **Nonlinear Planning with Constraint Posting** | `src/models/step_frame.py`, `config/rubric_ontology.owl`, `src/symbolic_math/owl_grounding.py` | Extracts structured StepFrames; canonicalizes variable synonyms ($\Delta y, h \rightarrow s$); validates independent prerequisites in any topological order. |
| **Unit 5: NLP & Neural Networks** | **Syntactic & Dependency Parsing**, Causal Connective Extraction, Concept Classification, **OCR (CRAFT detection + CRNN recognition)** | `src/perception_nlp/text_preprocessor.py`, `src/perception_nlp/neural_slot_extractor.py`, `src/perception_nlp/ocr_frontend.py` | Converts free-text derivations into claims, equations, and variable bindings; extracts causal markers (`"since"`, `"therefore"`); reads scanned/handwritten submissions via EasyOCR instead of requiring typed text. |
| **Unit 6: Expert Systems & Optimization** | **Backward-Chaining Explanation Facility**, **Numba-Accelerated Genetic Algorithm** | `src/reasoning_engine/explanation_facility.py`, `src/uncertainty_fuzzy/ga_calibrator.py` | Produces an auditable deduction proof tree (`WHY`/`HOW` queries); evolves fuzzy membership thresholds using delta-encoded chromosomes. |

---

## 2. Five Critical Mathematical Safeguards

1. **Elimination of the Dempster-Shafer Dictator Paradox (Unit 3):**
   - Applies Shafer's sensor discounting ($\alpha_2 = 0.90, m_2(\Theta) = 0.10$).
   - Uses Yager's Modified Combination Rule routing conflict $K$ into universal uncertainty $\Theta$, preventing unearned credit while avoiding degenerate certainty collapse.
2. **Multi-Hop Transitive ECF with Dual Residuals (Units 2 & 3):**
   - Separates $\mathcal{R}_{\text{rubric}} = |V_{\text{student}} - V_{\text{master}}|$ from $\mathcal{R}_{\text{local}} = |V_{\text{student}} - \text{Eval}(\text{Formula}_{\text{student}}, \text{Bindings}_{\text{student}})|$.
   - Evaluates multi-hop transitive chains via `depends_transitively/2` so Step 3 and beyond receive ECF credit if they trace back to an ancestor slip.
3. **Combined Algebraic Leaps via Gröbner Bases (Units 2 & 4):**
   - When a student combines multiple derivation steps (e.g. $F = ma$ and $v^2 - u^2 = 2as \implies F = \frac{m(v^2 - u^2)}{2s}$), the engine computes the Gröbner basis of the intermediate rubric ideal and tests polynomial ideal membership.
4. **Nonlinear Planning via Partial-Order Constraint Posting (Units 1 & 4):**
   - AND-branch sub-goals are modeled as a Partial-Order DAG ($\mathcal{P} = \langle \mathcal{S}, \mathcal{O} \rangle$). Valid topological sorts incur zero reordering penalty.
5. **Delta-Encoded Monotonic Chromosomes in Numba GA (Unit 6):**
   - Encodes fuzzy triangles using non-negative delta-offsets $[a, \Delta_1, \Delta_2]$, mathematically guaranteeing $a < b < c$ on 100% of crossover and mutation iterations without repair loops.

---

## 3. Directory Layout

```
├── config/
│   ├── rubric_ontology.owl               # OWL Knowledge Graph for domain laws & symbol synonyms (Unit 4)
│   └── domain_rules.json                 # JSON schema of derivation problems, rubrics & DAGs
├── kb_prolog/
│   ├── defeasible_rules.pl               # Non-monotonic logic, transitive ECF & dual residuals (Units 2, 3)
│   └── explanation_meta.pl               # Backward-chaining meta-interpreter for proof trees (Unit 6)
├── src/
│   ├── models/step_frame.py              # StepFrame, RubricStep, RubricStrategy, Summary (Unit 4)
│   ├── perception_nlp/
│   │   ├── text_preprocessor.py          # Sentence splitting & equation/clause isolation (Unit 5)
│   │   ├── neural_slot_extractor.py      # Concept classifier & StepFrame slot extractor (Unit 5)
│   │   └── ocr_frontend.py               # EasyOCR handwriting/scan reader -> raw submission text (Unit 5)
│   ├── symbolic_math/
│   │   ├── owl_grounding.py              # OWL Symbol canonicalization & alias resolver (Unit 4)
│   │   └── sympy_verifier.py             # Gröbner bases, canonical reduction & dual residuals (Unit 2)
│   ├── search_alignment/
│   │   ├── aostar_rubric_matcher.py      # AO* search on AND-OR trees + Partial-Order Planning (Unit 1, 4)
│   │   └── rubric_synthesizer.py         # Compiles a rubric from a reference derivation (Units 1,2,4,5; no LLM)
│   ├── uncertainty_fuzzy/
│   │   ├── dempster_shafer.py            # Discounted DS & Yager's conflict rule (Unit 3)
│   │   ├── fuzzy_scorer.py               # Fuzzy membership & centroid defuzzification (Unit 2)
│   │   └── ga_calibrator.py              # Numba-accelerated GA with delta-encoded chromosomes (Unit 6)
│   ├── reasoning_engine/
│   │   ├── prolog_bridge.py              # Pure-Python SLD resolution engine & PySwip wrapper
│   │   ├── explanation_facility.py       # Formats audit deduction trees & WHY/HOW queries (Unit 6)
│   │   └── audit_report.py               # Persists timestamped JSON audit reports + human-review flagging
│   └── pipeline.py                       # Unified Neuro-Symbolic Verification Orchestrator
├── api/
│   └── server.py                         # FastAPI service layer (grading, synthesis, report history)
├── tests/
│   ├── test_sympy_groebner.py
│   ├── test_aostar_partial_order.py
│   ├── test_transitive_ecf.py
│   ├── test_discounted_dempster_shafer.py
│   ├── test_numba_ga_delta.py
│   ├── test_ocr_frontend.py              # OCR extraction + end-to-end pipeline consumption
│   ├── test_rubric_synthesizer.py        # Rubric synthesis correctness + a brand-new topic
│   ├── test_audit_report.py              # Report persistence + human-review flagging
│   ├── generate_sample_submission_image.py  # Renders a synthetic handwriting-style demo image
│   └── test_end_to_end_scenarios.py      # All 5 canonical benchmark scenarios
├── ui/
│   ├── dashboard.py                      # Matplotlib diagnostic plot generator
│   ├── web_app.py                        # Streamlit frontend (grade / synthesize / report history)
│   └── static/                           # Output directory for generated figures
├── reports/                               # Persisted timestamped JSON audit reports (created at runtime)
└── main.py                               # CLI entrypoint for evaluating answers & viva demos
```

---

## 4. Running the Project

### Execute All Unit & Benchmark Tests (51/51 Passing)
```bash
python -m pytest tests/ -v
```

### Run Benchmark Demos via CLI
```bash
# Scenario 1: Golden Solution (Perfect Derivation, 100% Credit)
python main.py --scenario 1

# Scenario 2: Multi-Hop Transitive ECF (Early Slip Propagated Correctly)
python main.py --scenario 2 --calibrate

# Scenario 3: Skipped Intermediate Step (Gröbner Basis Algebraic Leap)
python main.py --scenario 3

# Scenario 4: Reordered Independent Prerequisite Steps (Partial-Order Planning)
python main.py --scenario 4

# Scenario 5: Persuasive Math Hallucination (Conflict K & Symbolic Veto)
python main.py --scenario 5
```

### Grade a Scanned / Handwritten Submission via OCR
Instead of a hardcoded benchmark string, the engine can read a photo or scan of
a student's handwritten derivation directly, using **EasyOCR** (a lightweight
CRAFT-detection + CRNN-recognition pipeline — no GPU required) as the
perception front-end:

```bash
python main.py --image path/to/submission.jpg --problem prob_falling_object
```

A synthetic demo image is generated for you:

```bash
python tests/generate_sample_submission_image.py
python main.py --image ui/static/sample_handwritten_submission.png
```

The OCR stage (`src/perception_nlp/ocr_frontend.py`) reconstructs reading-ordered
lines from EasyOCR's word/phrase-level detections, applies light domain-aware
normalization for recurring handwriting-OCR confusions (e.g. a cursive "s"
misread as "$"), and reports a mean recognition confidence — a genuine
perception-uncertainty signal, as opposed to the keyword-based heuristic
confidence used by `neural_slot_extractor.py` for typed input. The first run
downloads ~65MB of model weights. Realistic OCR noise (e.g. case-sensitive
confusions in cursive handwriting) that the symbolic layer correctly rejects
via **Safeguard 2 (Symbolic Veto)** is expected and demonstrates the
engine's robustness to imperfect perception rather than a bug.

---

## 6. Beyond the CLI: Persistence, Rubric Synthesis, API & Web UI

The engine is no longer only a scripted CLI demo against three hardcoded
problems. Four additions make it a real, extensible system — **without
introducing an LLM anywhere in the grading loop**; every verdict is still
produced by the Units 1–6 symbolic/defeasible/fuzzy pipeline.

### 6.1 Persisted, timestamped audit reports with human-review flagging
Every grading run (`main.py`, the API, or the Streamlit UI) now persists a
full JSON audit trail to `reports/AUDIT_REPORT_<submission_id>_<timestamp>.json`
via `src/reasoning_engine/audit_report.py`, instead of only printing to
stdout. Each step is automatically flagged for human review when the
engine's own uncertainty signals say so — **no extra model call**, just the
Dempster-Shafer conflict `K` (Unit 3) it already computes:
- `epistemic_conflict K > 0.5` — perception and symbolic evidence genuinely disagree.
- `credit_multiplier == 0.0` while `neural_confidence >= 0.75` — the step
  *sounded* right but scored zero, often a perception/OCR misread worth a
  second look rather than a genuine conceptual error.
- Low OCR confidence on the whole submission.

### 6.2 Rubric Synthesizer — compiling rubrics instead of hand-writing them
`src/search_alignment/rubric_synthesizer.py` compiles a plain-text
*reference/master derivation* (written exactly like a correct student
submission) into a full `RubricStrategy`: canonical equations, expected
bindings carried forward step-to-step, a causal prerequisite DAG (reusing
the same causal-connective detector used to read student submissions), and
**complexity-weighted marks** (`sympy.count_ops` per step, not an even
split). This directly answers "is the rubric just a hardcoded answer key?"
— it isn't; `config/domain_rules.json` is now a *compiled artifact*, not a
hand-authored one.

```bash
# Compile a brand-new problem (e.g. braking distance -- not in domain_rules.json)
# from a spec of master solutions:
python main.py --synthesize config/example_synth_spec.json

# It's immediately gradeable:
python main.py --problem prob_braking_distance --image <scan.jpg>
```

### 6.3 FastAPI service layer
`api/server.py` wraps the engine, OCR, synthesizer, and report store behind
a REST API (`/grade/text`, `/grade/image`, `/rubric/synthesize`, `/reports`),
so the grading core can be driven by something other than the CLI:

```bash
uvicorn api.server:app --reload --port 8000
# Interactive docs at http://127.0.0.1:8000/docs
```

### 6.4 Streamlit web UI
`ui/web_app.py` is a three-tab frontend (Grade a Submission / Synthesize a
Rubric / Report History) over the same engine — type or upload a scan,
see the per-step audit trail with review flags and the Dempster-Shafer
chart inline, or compile and immediately grade against a new topic:

```bash
streamlit run ui/web_app.py
```

---

## 5. Sample Audit Deduction Tree Output

```text
# Verification & Grading Audit Report: DEMO_SCENARIO_2
Selected Strategy: Kinematics & Newton Approach (strat_kinematics)
Total Awarded Marks: 9.38 / 15.00 (62.5%)

### Step 1: Kinematics
- Raw Text: "Velocity is v = u + a*t = 0 + 10*2 = 25"
- Evaluated Value: 25.0 | Rubric Res: 5.0 | Local Res: 5.0
- Defeasible Status: failed_value (Multiplier: 0.25)
+-- [PROVED] evaluate_step(Step_1, failed_value, 0.25)
    |-- [PROVED] student_step(Step_1, Concept, true, RubricRes=5.0)
    |-- [NOT-PROVED] \+ exception_ecf(Step_1)
    +-- [NOT-PROVED] \+ exception_rounding(Step_1)

### Step 2: Kinematics
- Raw Text: "Displacement is s = (u + v)/2 * t = (0 + 25)/2 * 2 = 25"
- Evaluated Value: 25.0 | Rubric Res: 5.0 | Local Res: 0.0
- Defeasible Status: ecf_override (Multiplier: 0.85)
+-- [PROVED] evaluate_step(Step_2, ecf_override, 0.85)
    +-- [PROVED] exception_ecf(Step_2)
        |-- [PROVED] depends_transitively(Step_2, Step_1)
        +-- [PROVED] root_arithmetic_slip(Step_1)

### Step 3: WorkEnergyTheorem
- Raw Text: "Work done is W = F * s = 20 * 25 = 500"
- Evaluated Value: 500.0 | Rubric Res: 100.0 | Local Res: 0.0
- Defeasible Status: ecf_override (Multiplier: 0.85)
+-- [PROVED] evaluate_step(Step_3, ecf_override, 0.85)
    +-- [PROVED] exception_ecf(Step_3)
        |-- [PROVED] depends_transitively(Step_3, Step_1)
        +-- [PROVED] root_arithmetic_slip(Step_1)
```

---

## 7. Grading-Rigor Fixes: Honest Concept Classification & Dimensional Analysis

Two targeted upgrades close real correctness gaps in the grading logic itself
(not infrastructure), both proven to be **zero-regression** for every
pre-existing problem and documented benchmark score.

### 7.1 Semantic fallback for concept classification (Unit 5)
`NeuralSlotExtractor` was pure keyword-density counting -- accurate for
kinematics phrasing, but it silently defaulted to a flat `"ScientificLaw"`
label (0.50 confidence) for anything outside that vocabulary, which could
cause the AO* matcher to misalign steps (confirmed: an energy-conservation
submission had its Step 2 and Step 3 swapped against the rubric). It now
falls back to **sentence-embedding similarity** (`sentence-transformers`,
`all-MiniLM-L6-v2`) against concept exemplar sentences *only* when no
keyword matches at all -- a genuine independent neural signal, not keyword
matching in disguise. Every case with an existing keyword match is
untouched: `tests/test_semantic_fallback_classifier.py` proves all 5 CLI
benchmark scenarios produce bit-for-bit identical scores whether the
fallback is enabled or disabled.

### 7.2 Opt-in dimensional analysis / unit checking (Unit 4 ontology)
The engine previously compared only numeric residuals -- a student writing
`20` instead of `20 m/s` got full credit. `OWLGrounder` now maps each
canonical symbol to its expected SI unit and extracts/validates a trailing
unit token from the student's text (`check_unit`). This is **opt-in per
rubric step** (`require_units`, default `False`), so no pre-existing
problem's score changes. `RubricSynthesizer` auto-opts a step in when the
*master/reference solution itself* states a unit for that step -- author a
solution with units, and grading units becomes part of the rubric for free.
Missing a required unit costs 10% of that step's marks; providing the
wrong unit (e.g. `cm/s` instead of `m/s`) costs 25%:

```bash
# prob_braking_distance was re-synthesized with units in its master solution
# (config/example_synth_spec.json), so it now grades units automatically.
python main.py --problem prob_braking_distance --image <scan_with_units.jpg>
```
