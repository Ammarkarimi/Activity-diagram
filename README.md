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

## Agents

| Agent | Role |
|---|---|
| Orchestrator | Owns pipeline state, routes every agent output, decides iterate / accept / rollback / stop, keeps the best candidate |
| Requirement Agent | Splits the specification into atomic requirements R1..Rn (per chunk for long documents) |
| Decomposition Agent | Long documents only: partitions requirements into modules (sub-activities) and the flow between them |
| Planning Agent | Builds the ActivityPlan (nodes, edges, decisions, loops, forks, exceptions) |
| Generator Agent | Turns the plan into the ActivityDiagram IR |
| Validator (deterministic) | Structural, coverage, hallucination and PlantUML syntax checks |
| Semantic Reviewer | LLM review of the diagram against the requirements, given the structural results |
| Feedback Agent | Prioritizes all defects; can only reorder existing ones |
| Repair Agent / Router | Repairs the selected defects (deterministic repairs first, LLM otherwise); the orchestrator re-validates and accepts only non-regressing repairs |

## Long documents (20+ pages)

Sending a whole 20-page SRS to every agent overflows context windows and
degrades quality, so long inputs go through a hierarchical pipeline
(`src/pipeline/hierarchical.py`):

1. **Chunking** (`src/document/chunker.py`): section-aware split on headings
   (Markdown, numbered `3.2.1`, ALL CAPS) and paragraphs, with a token budget
   per chunk. Every chunk records its heading path and carries the tail of the
   previous chunk as read-only context.
2. **Chunked extraction**: the Requirement Agent runs per chunk, in parallel.
   Results are consolidated into one global R1..Rn list: dependencies are
   remapped and overlap duplicates removed. Section provenance is kept on
   every requirement. A failing chunk is reported without aborting the run.
3. **Decomposition**: the Decomposition Agent groups requirements into
   cohesive modules (at most `--max-requirements-per-module` each) and
   proposes module-to-module flow (sequence, conditional, or independent use
   cases). The plan is then normalized deterministically: every requirement
   lands in exactly one module, oversized modules are split, and dangling or
   unreachable transitions are fixed. If the agent fails, or the requirement
   list is too large for one prompt, requirements are grouped by section.
4. **Per-module agent loop**: each module runs through the full
   Planning -> Generator -> Validator -> Reviewer -> Feedback -> Repair loop,
   in parallel. Each module sees only its own requirements and its entry/exit
   context, so prompt size does not grow with document length.
5. **Composition**: an **overview diagram** (one call-behaviour action per
   module) and a **full diagram** with every module inlined. Normal module
   ends continue to the next module. Final nodes labelled as failures
   (rejected, cancelled, error, ...) remain terminal. Both diagrams are
   validated and compiled to PlantUML.
6. **Report**: global requirement coverage, per-module defects and metrics
   in `report.md`.

```bash
# auto picks hierarchical mode when the input exceeds ~12000 tokens (about 20 pages)
python -m scripts.run_pipeline --input data/pure/peering.txt
python -m scripts.run_pipeline --input specs/srs.pdf --mode hierarchical \
    --chunk-tokens 2500 --max-requirements-per-module 25 --max-workers 4
```

Output folder `outputs/<id>_<timestamp>/`:

- `00_chunks.json`, `01_requirements.json`, `02_decomposition.json`
- `modules/M*_*/`: every agent's intermediate output per module
- `modules/M*.puml`: one diagram per module
- `viewer.html`: open in a browser to browse every diagram with scrolling, zoom and pan
- `overview.puml`, `full.puml` (+ SVG/PNG when PlantUML is available)
- `parts/`: the full diagram cut into one diagram per module, each with its swimlanes and
  "From Part N" / "Continue in Part N" links to the others
- `*.invalid.puml`: a diagram that failed the PlantUML syntax check (the error is in `report.md`)
- `03_final_state.json`, `report.md`

Inputs may be `.txt`, `.md`, `.pdf` (`pypdf`) or `.docx` (`python-docx`).

## Before a diagram is final

A diagram is only finalised after these checks:

- **Outline checks**: a stop inside a fork, two outcomes of a decision with
  the same guard, and a loop or fork the outline's own analysis lists but
  does not draw are blocking; a step drawn twice is reported.
- **Stable comparison**: an unchanged candidate is not reviewed twice, repairs
  stop when only LOW findings remain, and candidates are ranked by blocking,
  then serious defects (reviewer scores only break ties).

- **Blocking defects** (unreachable nodes, broken flow or termination, an action
  with several exits, PlantUML errors, requirements not shown) get up to 3 extra
  repair rounds after `--max-iterations`, and a candidate without them always
  beats one with them. Structural defects no rule can fix go to the LLM repair
  agent instead of stopping the loop.
- **Coverage** only counts requirements traced to reachable nodes: an
  unreachable node is never drawn.
- **Decisions hidden in actions** (an action with several guarded exits) are
  split into an action plus a decision automatically.
- **PlantUML** must actually draw the diagram (a layout crash counts as an
  error); if it cannot, the loops are redrawn as plain branches and checked
  again.
- **Swimlanes**: every branch starts in its decision's lane, multi-way
  decisions with lanes are nested `if`/`else` (PlantUML drops arrows of
  cross-lane `elseif` chains), and long lane titles wrap. Lane names are
  matched to actors using the specification's own abbreviations
  ("Peering Agent (PA)").

`tools/plantuml.jar` is not in git: run `python -m scripts.install_plantuml`
on every new checkout, otherwise diagrams cannot be checked (the CLI and the
UI warn about this).

## Planners

`--planner outline` (default) plans the diagram as a block-structured outline
(steps, if/else, while loops and forks, each step in its actor's lane), the
way an expert writes an activity diagram. The outline is converted to the IR
for validation and review, repaired by editing the outline, and drawn
directly as PlantUML blocks, so the diagram never needs jumps between distant
parts of the flow. `--planner graph` keeps the earlier
Planning -> Generator -> Refinement agents.

Specifications above ~2000 tokens are extracted chunk by chunk and merged by
the Consolidation Agent (restatements and explanations are kept for
traceability but not drawn) before one diagram is planned.

## Output files (single mode)

`outputs/<sample>.puml/.svg/.png` is the final diagram and
`outputs/<sample>_before_repair.puml/.svg/.png` the diagram as first drawn,
before validation and repair. `outputs/<sample>_state.json` holds the final
state, including `metrics.before_repair` / `metrics.after_repair` (defects,
blocking defects, coverage, quality). Every stage is kept in
`outputs/<sample>_<timestamp>/`:

| File | Content |
|---|---|
| `01_requirement_agent.json` | extracted requirements |
| `01a_consolidation_agent.json` | restatements / details set aside (long specs) |
| `01b_actor_agent.json` | swimlanes (chosen and proposed) |
| `02_outline_agent.json` | the outline (outline planner) |
| `02_planning_agent.json`, `03_generator_agent.json`, `03b_refinement_agent.json` | graph planner stages |
| `initial.puml` | diagram before validation and repair |
| `04_iteration_NN_validation.json`, `04_iteration_NN.puml` | validation, review, defects and PlantUML per iteration |
| `05_iteration_NN_feedback_agent.json` | defect prioritisation |
| `06_iteration_NN_repair_agent.json` | repair and its changes |
| `07_iteration_NN_repaired_diagram.json`, `07_iteration_NN_repaired_outline.json` | repaired candidate |
| `08_final_state.json`, `final.puml` | final state and diagram |

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
- Recommended: Java + PlantUML, used to check every diagram's syntax and size before it is
  written and to render SVG/PNG. Run `python -m scripts.install_plantuml` to download
  `plantuml.jar` into `tools/` (or set `PLANTUML_JAR`, or put `plantuml` on PATH). Without it
  only built-in syntax checks run.

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

## Evaluating against the gold standards

Run the pipeline several times per sample (results vary from run to run),
then score the outputs two ways:

```bash
# 1. Similarity to the expert's diagram (the gold scores 1.0 by definition).
#    Lexical label matching, plus --semantic: steps paired by meaning by an
#    LLM judge (majority of 5 judgements). Use a stronger judge than the
#    pipeline's model.
python -m scripts.compare_with_gold --generated results/raw/<run> --semantic --judge-model gpt-5.5

# 2. Fidelity to the SPECIFICATION, scoring gold and generated diagrams the
#    same way: coverage of the behaviours the text states, share of steps
#    the text does not support, and their harmonic mean.
python -m scripts.evaluate_fidelity --generated results/raw/<run> --model gpt-5.5
```

Both write Markdown tables with mean ± std per sample. The behaviour list
used by `evaluate_fidelity` is built once per specification (cached in
`results/reference_behaviours/`) with an evaluation-only prompt, so every
diagram is judged against the same list. Gold diagrams contain an expert's
interpretation (steps from outside the given text, deliberate abstraction),
so report both measures.

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
