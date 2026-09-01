# Activity Diagram Multi-Agent Research Framework

A research-oriented Python prototype for generating UML Activity Diagrams from natural-language requirements using an OpenAI-backed multi-agent pipeline.

## Research pipeline

Requirement Agent
-> Planning Agent
-> Generator Agent
-> Deterministic Validator
-> Semantic Reviewer
-> Defect Classifier
-> Typed Repair Router
-> Regression Validation
-> final PlantUML + SVG/PNG (when PlantUML is available)

The canonical artifact is a structured `ActivityDiagram` intermediate representation (IR). PlantUML is only a rendering/export format.

## Repair taxonomy

R1 Syntax
R2 Structural / UML
R3 Requirement Coverage
R4 Semantic Alignment
R5 Control Flow
R6 Decision / Guard
R7 Concurrency / Synchronization
R8 Data / Object Flow
R9 Exception / Event Flow
R10 Termination / Liveness
R11 Hallucination / Extraneous Behavior
R12 Granularity / Abstraction
R13 Layout / Visualization

## Requirements

- Python 3.11+
- OpenAI API key
- Optional: Java + PlantUML or Docker if you want rendered images

## Quick start

```bash
python -m venv .venv
# Windows:
# .venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# Put your OPENAI_API_KEY in .env

python -m scripts.run_pipeline --input data/examples/login_requirement.txt
```

Output is written to `outputs/`.

## Model

The default model is `gpt-5.6-luna`, selected for cost-sensitive/high-volume use in this prototype. Change `OPENAI_MODEL` in `.env` as needed.

The OpenAI integration uses the Responses API and JSON Schema Structured Outputs so agent boundaries stay machine-readable.

## Example

```bash
python -m scripts.run_pipeline \
  --input data/examples/payment_requirement.txt \
  --max-iterations 3
```

## Experiments

```bash
python -m scripts.run_experiment --dataset data/examples/sample_dataset.jsonl --experiment direct_llm
python -m scripts.run_experiment --dataset data/examples/sample_dataset.jsonl --experiment cot
python -m scripts.run_experiment --dataset data/examples/sample_dataset.jsonl --experiment self_refine
python -m scripts.run_experiment --dataset data/examples/sample_dataset.jsonl --experiment ladex
python -m scripts.run_experiment --dataset data/examples/sample_dataset.jsonl --experiment multi_agent
```

Then:

```bash
python -m scripts.calculate_metrics --results results/raw
```

## Data layout

- `data/raw/` keeps original datasets untouched.
- `data/processed/` stores extracted requirements/plans/graphs.
- `data/gold/` stores expert gold diagrams, mappings, traces and annotations.
- `results/` stores experiment outputs and metrics.

## Important research note

The included LADEX baseline is an *experimental reproduction scaffold*, not the authors' official implementation. For a paper, implement the exact published protocol from the LADEX paper and cite it directly.

Likewise, PURE/PAGED source files are not bundled. Put licensed/allowed dataset files under `data/raw/`.

## Suggested next research work

1. Build the human gold Activity Diagram + Requirement Trace Matrix set from the selected PURE documents.
2. Lock the evaluation protocol before comparing systems.
3. Run direct LLM, planning-only, self-refine/LADEX-style, and full multi-agent baselines on the exact same inputs.
4. Add ablations for planning, deterministic validation, semantic review, typed repair, and iterative repair.
5. Report structural validity, requirement coverage, unsupported behavior, graph similarity, behavioral similarity, complexity, latency/cost, repair success, and expert ratings.
