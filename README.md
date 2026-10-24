# Generate–Verify–Repair: Multi-Agent UML Activity Diagram Generation

A multi-agent tool that generates UML activity diagrams from natural-language
requirements. It drafts a diagram, checks it, and repairs the defects it finds
in a loop, keeping a repair only when the diagram does not get worse.

- Tool: https://activity-diagram.streamlit.app/
- Demonstration video: https://youtu.be/An76JRiyT6o

## Overview

1. **Generation.** The Requirement Agent splits the specification into typed
   requirements, the Actor Agent finds the swimlanes, the Planning Agent
   writes the activity plan, the Generator Agent turns it into a diagram, and
   the Refinement Agent removes duplicated steps.
2. **Verification.** Deterministic rules check the structure, requirement
   coverage and PlantUML syntax; an LLM reviewer compares the diagram with the
   specification.
3. **Repair.** The Feedback Agent ranks the defects and the Repair Agent fixes
   the most important ones. The repaired diagram is verified again and is
   rolled back if it got worse.

The web interface shows the final diagram, every iteration of the loop
(defects found, repair applied, accepted or rolled back), the extracted
requirements, the PlantUML source, and token usage and cost.

The repository also contains the six PURE specifications used in our
evaluation (`data/pure/`), their reference diagrams
(`data/pure_gold_standard/`), and the runs reported in the paper
(`outputs/Agentic AI Results/`).

## Requirements

- Python 3.11 or newer
- An OpenAI API key
- Java 8 or newer, for PlantUML (rendering and syntax checks)

## Setup

```bash
git clone https://github.com/Ammarkarimi/Activity-diagram.git
cd Activity-diagram

python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# Linux/macOS:           source .venv/bin/activate

pip install -r requirements.txt
python -m scripts.install_plantuml   # downloads tools/plantuml.jar

cp .env.example .env                 # Windows: copy .env.example .env
```

Open `.env` and set `OPENAI_API_KEY`. The paper used `OPENAI_MODEL=gpt-5.4-mini`.

## Run

```bash
streamlit run app.py
```

Open http://localhost:8501, choose the model and the number of repair
iterations in the sidebar, upload or paste a specification (or pick an
example), and click **Generate activity diagram**.

To view the published runs without an API key, choose **Saved run** as the
requirements source and open any run under `outputs/Agentic AI Results/`.

### Docker

```bash
docker build -t activity-diagram .
docker run -p 7860:7860 activity-diagram
```

Then open http://localhost:7860.

## Citation

M. A. Karimi, A. Limkar and S. Tiwari, "Generate–Verify–Repair: A Multi-Agent
Framework for UML Activity Diagram Generation from Natural Language
Requirements", Software Engineering Research Lab (SERL), DAU Gandhinagar.
