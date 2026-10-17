# Generate–Verify–Repair: Multi-Agent UML Activity Diagram Generation

This tool generates UML activity diagrams from natural-language requirements.
Separate agents extract the requirements, plan the flow and build the diagram.
The diagram is then checked by deterministic rules and by an LLM reviewer, and
the defects found are repaired in a loop. A repair is kept only if
re-verification shows the diagram did not get worse.

The repository contains the tool (command line and web UI), the six PURE
specifications used in our evaluation, the reference (gold standard) diagrams
for them, and the outputs of the runs reported in the paper.

## Repository layout

```
app.py                         Streamlit web interface
src/                           the pipeline: agents, validation, repair, PlantUML generation
scripts/                       command-line entry points (run, score, install PlantUML)
tests/                         pytest test suite
configs/                       validation rules, model list and token prices
data/pure/                     input specifications (functional requirements from PURE)
data/pure_gold_standard/       reference activity diagrams (PlantUML .txt + rendered .png)
outputs/Agentic AI Results/    outputs of the evaluation runs reported in the paper
```

### Input specifications (`data/pure/`)

| File | PURE document |
|---|---|
| `agentmom.txt` | AgentMom (agent message communication) |
| `central_trading.txt` | Central Trading System |
| `inventory.txt` | Inventory (university asset management) |
| `libra.txt` | Libra (cluster job scheduler) |
| `peering.txt` | CDN Peering |
| `video_search.txt` | Video Search |

Each file holds the functional requirements part of the PURE document as
plain text. The PURE dataset is described in A. Ferrari, G. O. Spagnolo and
S. Gnesi, "PURE: A Dataset of Public Requirements Documents", RE 2017.

### Reference diagrams (`data/pure_gold_standard/`)

One activity diagram per specification, written in PlantUML (`.txt`) with a
rendered image (`.png`). The number in the file name is the PURE document
year. The diagrams were drawn by the authors and reviewed by a software
engineering expert.

## Requirements

- Python 3.11 or newer
- An OpenAI API key
- Java 8 or newer (for PlantUML). Every diagram is syntax-checked and rendered
  with PlantUML. Without Java only the built-in checks run and no images are
  drawn.

## Setup

```bash
git clone https://github.com/Ammarkarimi/Activity-diagram.git
cd Activity-diagram

python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# Windows (cmd):         .venv\Scripts\activate.bat
# Linux/macOS:           source .venv/bin/activate

pip install -r requirements.txt

# downloads tools/plantuml.jar (not kept in git)
python -m scripts.install_plantuml

cp .env.example .env        # Windows: copy .env.example .env
```

Open `.env` and set your key:

| Variable | Meaning | Value used in the paper |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API key (required) | |
| `OPENAI_MODEL` | model used by every agent | `gpt-5.4-mini` |
| `OPENAI_TIMEOUT` | seconds allowed per LLM call | `300` |
| `LOG_LEVEL` | `INFO` or `DEBUG` | `INFO` |

Check the installation (no API key needed):

```bash
python -m pytest
```

## Running the web interface

```bash
streamlit run app.py
```

The app opens at http://localhost:8501. Add `--server.address 127.0.0.1` to
keep it reachable only from your own machine.

1. In the sidebar, choose the model, the pipeline mode (*Auto* is fine), the
   planner (*Outline*, the default, was used in the paper) and the number of
   repair iterations (default 3).
2. Under *Requirements source*, pick one of the specifications in
   `data/pure`, upload a `.txt`, `.md`, `.pdf` or `.docx` file, or paste text.
3. Click **Generate activity diagram**. A short specification takes one to
   three minutes.

The results page has these tabs:

- **Diagram**: the final diagram and, if the loop changed it, the diagram
  before repair. Both can be downloaded as SVG, PNG or PlantUML.
- **Iterations**: the repair loop, one iteration at a time: the candidate
  diagram (red: steps named by a defect sent to repair; green: steps changed by
  the previous repair), the defects found, the repair and whether it was
  accepted or rolled back, and the defects resolved since the previous
  iteration.
- **Requirements**: the extracted requirements with type and actors.
- **PlantUML**: the diagram source.
- **Usage**: tokens and cost per iteration and per agent.

A full run can also be downloaded as a `.zip`. Each UI run is stored in
`outputs/ui/<name>_<timestamp>/`.

**Viewing the published results without an API key.** Choose *Saved run* as
the requirements source, select a `*_state.json` file under
`outputs/Agentic AI Results/` and click **Open run**. The run opens with all
its iterations and no LLM call is made.

## Reproducing the evaluation

### 1. Generate the diagrams

Run the pipeline once per specification, with the settings used in the paper
(outline planner, 3 repair iterations, `gpt-5.4-mini` in `.env`):

```bash
for spec in agentmom central_trading inventory libra peering video_search; do
  python -m scripts.run_pipeline --input data/pure/$spec.txt --output-dir outputs/my_run
done
```

On Windows PowerShell:

```powershell
foreach ($spec in "agentmom","central_trading","inventory","libra","peering","video_search") {
  python -m scripts.run_pipeline --input "data/pure/$spec.txt" --output-dir outputs/my_run
}
```

Each run costs between 0.05 and 0.26 USD and takes one to four minutes. LLM
output varies between runs, so your diagrams will not be identical to ours.

`run_pipeline` options: `--model`, `--max-iterations` (default 3),
`--planner {outline,graph}` (default `outline`),
`--mode {auto,single,hierarchical}` (default `auto`), `--output-dir`
(default `outputs`), `--sample-id`.

### 2. What a run writes

For a specification `<spec>`, the output folder contains:

| File | Content |
|---|---|
| `<spec>.puml/.svg/.png` | final diagram (after the repair loop) |
| `<spec>_before_repair.puml/.svg/.png` | diagram produced by generation alone |
| `<spec>_state.json` | final state: requirements, all candidates, defects, repair history, metrics, tokens and cost |
| `<spec>_<timestamp>/` | the output of every agent, listed below |

| File in `<spec>_<timestamp>/` | Content |
|---|---|
| `01_requirement_agent.json` | extracted requirements |
| `01a_consolidation_agent.json` | requirements set aside as restatements or context (long inputs) |
| `01b_actor_agent.json` | swimlanes |
| `02_outline_agent.json` | the activity plan |
| `initial.puml` | diagram before verification |
| `04_iteration_NN_validation.json`, `04_iteration_NN.puml` | rule and reviewer results, defects and diagram per iteration |
| `05_iteration_NN_feedback_agent.json` | prioritised defects |
| `06_iteration_NN_repair_agent.json` | repair and its change list |
| `07_iteration_NN_repaired_outline.json` | repaired candidate |
| `08_final_state.json`, `final.puml` | final state and diagram |

The runs reported in the paper are in `outputs/Agentic AI Results/` in
exactly this format.

### 3. Compare with the reference diagrams

The numbers in the paper come from a manual, node-by-node comparison of
`<spec>_before_repair.puml` (generation only) and `<spec>.puml` (after repair)
with the reference diagram in `data/pure_gold_standard/`, using four measures:

| Measure | Definition |
|---|---|
| Structural correctness | well-formed elements / all elements (PlantUML delimiters, start/end, action and decision nodes) |
| Semantic correctness | generated nodes matching a reference node / generated nodes |
| Semantic completeness | reference nodes found in the generated diagram / reference nodes |
| Swimlane identification | reference swimlanes found / reference swimlanes |

For a quick automatic check, `compare_with_gold` matches the action labels of
each `<spec>.puml` against the reference and reports recall, precision, F1
and ordering agreement:

```bash
python -m scripts.compare_with_gold --generated outputs/my_run
# optional: pair steps by meaning with an LLM judge
python -m scripts.compare_with_gold --generated outputs/my_run --semantic --judge-model gpt-5.4
```

The tables are written to `results/gold_comparison/` (not tracked in git). Automatic matching is
stricter than a human reader, so its scores are lower than the manual ones.

## How the tool works

| Agent | Kind | Input → output |
|---|---|---|
| Orchestrator | rules | pipeline state → next step; accept, roll back or stop |
| Requirement Agent | LLM | specification → typed requirements R1…Rn |
| Planning Agent | LLM | requirements, swimlanes → block-structured plan |
| Generator | rules | plan → graph (intermediate representation) and PlantUML |
| Validator | rules | graph, requirements → structural and coverage defects |
| Semantic Reviewer | LLM | specification, graph, rule results → semantic defects |
| Feedback Agent | LLM | all defects → prioritised defects |
| Repair Agent | LLM | selected defects, plan → repaired plan, change list |

- Every diagram is held as a typed graph (the intermediate representation).
  All checks run on this graph; PlantUML is generated from it for rendering
  and a final syntax check.
- Specifications longer than about 2,000 tokens are split at section
  boundaries and extracted chunk by chunk in parallel.
- Defects that make a diagram wrong (broken flow, invalid decisions or forks,
  PlantUML errors, missing requirements) are *blocking*. The loop runs three
  iterations by default plus up to three more while blocking defects remain,
  and the best candidate becomes the final diagram.
- Documents above about 12,000 tokens (roughly 20 pages) can run in
  hierarchical mode (`--mode hierarchical`): requirements are grouped into
  modules, each module runs the loop, and the module diagrams are composed
  into an overview and a full diagram.
- `--planner graph` uses an alternative graph planner with an LLM generator.

## Citation

If you use this tool or the reference diagrams, please cite our paper:

A. Limkar, M. A. Karimi and S. Tiwari, "Generate–Verify–Repair: A
Multi-Agent Framework for UML Activity Diagram Generation from Natural
Language Requirements", Software Engineering Research Lab (SERL), DAU
Gandhinagar.
