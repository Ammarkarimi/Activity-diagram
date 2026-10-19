"""Web UI for the activity-diagram pipeline.

Run with:  streamlit run app.py

The UI only calls the existing pipelines (MultiAgentPipeline and
HierarchicalPipeline); it does not change their behaviour.
"""
from __future__ import annotations

import base64
import html
import io
import json
import logging
import os
import re
import tempfile
import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.document.chunker import estimate_tokens
from src.document.loader import load_document
from src.generation.plantuml_tool import ensure_plantuml, plantuml_command, run_plantuml
from src.llm.openai_client import OpenAIClient
from src.models.domain import PipelineState
from src.reporting.iterations import ADDED, FLAGGED, iteration_views

load_dotenv()

ROOT = Path(__file__).resolve().parent
UI_OUTPUT_DIR = ROOT / "outputs" / "ui"
EXAMPLE_DIRS = [ROOT / "data" / "pure"]
# Folder searched for saved runs. A public deployment sets it to the published
# results so that visitors never see each other's runs.
SAVED_RUNS_DIR = ROOT / os.getenv("SAVED_RUNS_DIR", "outputs")

# Models offered in the selector. The backend uses the OpenAI Responses API,
# so any OpenAI model with structured-output support works ("Custom" below).
MODELS = ["gpt-5.4-mini", "gpt-5.4", "gpt-5.5", "gpt-4.1", "gpt-4o"]
CUSTOM_MODEL = "Custom…"

MODE_LABELS = {
    "auto": "Auto (by document length)",
    "single": "Single diagram",
    "hierarchical": "Hierarchical (long documents)",
}


# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Activity Diagram Generator",
    page_icon="◆",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1400px;}
      h1 {font-weight: 600; letter-spacing: -0.01em; font-size: 2.1rem !important;}
      .app-subtitle {color: #5f6b7a; margin-top: -0.6rem; margin-bottom: 1.4rem;}
      div[data-testid="stMetricValue"] {font-size: 1.6rem;}
      div[data-testid="stMetric"] {
        background: #f7f9fc; border: 1px solid #e3e8ef; border-radius: 8px;
        padding: 0.7rem 0.9rem;
      }
      div[data-testid="stMetricLabel"] p {color: #5f6b7a; font-size: 0.82rem;}
      .pipeline {color: #5f6b7a; font-size: 0.82rem; line-height: 1.6;}
      #MainMenu, footer, [data-testid="stAppDeployButton"] {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------
# Pipeline execution (background thread + live log)
# ------------------------------------------------------------------
class _ListHandler(logging.Handler):
    def __init__(self, sink: list[str]) -> None:
        super().__init__(level=logging.INFO)
        self.sink = sink
        self.setFormatter(logging.Formatter("%(asctime)s  %(name)s  %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.sink.append(self.format(record))
        except Exception:
            pass


def _run_pipeline(job: dict) -> None:
    """Executed in a worker thread; writes results into ``job``."""
    try:
        text, mode, out = job["text"], job["mode"], job["out"]
        llm = OpenAIClient(model=job["model"], api_key=job["api_key"])
        if mode == "hierarchical":
            from src.pipeline.hierarchical import HierarchicalPipeline

            state = HierarchicalPipeline(
                model=job["model"],
                llm=llm,
                chunk_tokens=job["chunk_tokens"],
                max_requirements_per_module=job["max_reqs"],
                max_workers=job["max_workers"],
                planner=job["planner"],
            ).run(
                sample_id=job["sample_id"],
                requirement_text=text,
                max_iterations=job["max_iterations"],
                output_dir=out,
                document_title=job["sample_id"],
            )
        else:
            from src.pipeline.orchestrator import MultiAgentPipeline

            state = MultiAgentPipeline(model=job["model"], planner=job["planner"], llm=llm).run(
                sample_id=job["sample_id"],
                requirement_text=text,
                max_iterations=job["max_iterations"],
                output_dir=out,
            )
            # Same files the CLI (scripts/run_pipeline.py) writes.
            sid = job["sample_id"]
            (out / f"{sid}_state.json").write_text(
                json.dumps(state.model_dump(mode="json"), indent=2), encoding="utf-8"
            )
            (out / f"{sid}.puml").write_text(state.final_plantuml, encoding="utf-8")
        job["state"] = state
    except Exception as exc:  # surfaced in the UI
        job["error"] = f"{type(exc).__name__}: {exc}"


def run_with_progress(job: dict) -> None:
    logs: list[str] = []
    handler = _ListHandler(logs)
    root = logging.getLogger()
    previous_level = root.level
    root.setLevel(logging.INFO)
    root.addHandler(handler)

    worker = threading.Thread(target=_run_pipeline, args=(job,), daemon=True)
    started = time.time()
    worker.start()
    try:
        with st.status("Running multi-agent pipeline…", expanded=True) as status:
            info = st.empty()
            log_box = st.empty()
            while worker.is_alive():
                elapsed = int(time.time() - started)
                info.markdown(
                    f"**Model:** `{job['model']}` &nbsp;·&nbsp; **Mode:** {job['mode']} "
                    f"&nbsp;·&nbsp; **Elapsed:** {elapsed // 60:02d}:{elapsed % 60:02d}"
                )
                log_box.code("\n".join(logs[-12:]) or "Starting…", language=None)
                time.sleep(1.0)
            worker.join()
            log_box.code("\n".join(logs[-12:]), language=None)
            if "error" in job:
                status.update(label="Pipeline failed", state="error", expanded=True)
            else:
                status.update(
                    label=f"Completed in {time.time() - started:.0f} s",
                    state="complete",
                    expanded=False,
                )
    finally:
        root.removeHandler(handler)
        root.setLevel(previous_level)
    job["logs"] = logs
    job["elapsed"] = time.time() - started


# ------------------------------------------------------------------
# Result helpers
# ------------------------------------------------------------------
def zip_folder(folder: Path) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                zf.write(path, Path(folder.name) / path.relative_to(folder))
    return buffer.getvalue()


def collect_diagrams(job: dict) -> dict[str, dict]:
    """Label -> {svg, png, puml} for every rendered diagram of the run."""
    state = job["state"]
    renders = state.metrics.get("render", {})
    diagrams: dict[str, dict] = {}

    if job["mode"] != "hierarchical":
        diagrams["Activity diagram"] = {**renders, "puml_text": state.final_plantuml}
        # The diagram as first drawn, before validation and repair.
        sid = job["sample_id"]
        before = {ext: job["out"] / f"{sid}_before_repair.{ext}" for ext in ("svg", "png", "puml")}
        before_text = before["puml"].read_text(encoding="utf-8") if before["puml"].exists() else ""
        job["before_same"] = bool(before_text) and before_text.strip() == (state.final_plantuml or "").strip()
        if before_text and not job["before_same"]:
            diagrams["Before repair"] = {
                "svg": str(before["svg"]) if before["svg"].exists() else None,
                "png": str(before["png"]) if before["png"].exists() else None,
                "puml_text": before_text,
            }
        return diagrams

    modules = {m.module.id: m for m in state.modules}
    parts = {f"part_{p['number']:02d}": p for p in state.metrics.get("parts", [])}

    def add(label: str, key: str, puml_text: str) -> None:
        if key in renders:
            diagrams[label] = {**renders[key], "puml_text": puml_text}

    add("Full diagram", "full", state.full_plantuml)
    add("Overview (modules)", "overview", state.overview_plantuml)
    for key, part in parts.items():
        source = renders.get(key, {}).get("source")
        text = Path(source).read_text(encoding="utf-8") if source and Path(source).exists() else ""
        add(f"Part {part['number']}: {part['name']}", key, text)
    for mid, module in modules.items():
        add(f"Module {mid}: {module.module.name}", mid, module.plantuml)
    return diagrams


def show_svg(svg_path: str | None, fit: bool, height: int = 700, svg_bytes: bytes | None = None) -> None:
    # Shown as an <img> so the SVG (whose labels come from LLM output) is never
    # executed as markup.
    data = base64.b64encode(svg_bytes if svg_bytes is not None else Path(svg_path).read_bytes()).decode("ascii")
    size = "max-width:100%;height:auto;" if fit else "max-width:none;"
    st.html(
        f'''<div style="height:{height}px;overflow:auto;background:#fff;border:1px solid #e3e8ef;
                    border-radius:8px;padding:16px;box-sizing:border-box;text-align:center;">
             <img src="data:image/svg+xml;base64,{data}" style="{size}" alt="Activity diagram">
           </div>'''
    )


@st.cache_data(show_spinner=False, max_entries=64)
def plantuml_svg(text: str) -> bytes | None:
    proc = run_plantuml(["-tsvg", "-pipe"], input_text=text)
    if proc is None or proc.returncode != 0 or "<svg" not in proc.stdout:
        return None
    return proc.stdout.encode("utf-8")


def saved_runs() -> dict[str, Path]:
    """Label -> state file of every finished run under SAVED_RUNS_DIR, newest first."""
    root = SAVED_RUNS_DIR
    files = [p for p in root.rglob("*_state.json") if not p.name[:1].isdigit()] if root.exists() else []
    return {
        p.relative_to(ROOT).as_posix(): p
        for p in sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
    }


def open_saved_run(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "candidates" not in data:
        raise ValueError("only single-diagram runs can be opened here")
    state = PipelineState.model_validate(data)
    config = path.parent / "ui_run_config.json"
    model = json.loads(config.read_text(encoding="utf-8")).get("model") if config.exists() else None
    run_dir = Path(state.metrics.get("run_output_dir") or path.parent)
    run_dir = run_dir if run_dir.is_absolute() else ROOT / run_dir
    return {
        "state": state,
        "sample_id": path.name[: -len("_state.json")],
        "mode": "single",
        "planner": state.metrics.get("planner", "outline"),
        "model": model or os.getenv("OPENAI_MODEL", "n/a"),
        "out": path.parent,
        "zip_dir": run_dir if run_dir.exists() else path.parent,
    }


_SEVERITY_COLOURS = {"CRITICAL": "#b71c1c", "HIGH": "#d84315", "MEDIUM": "#f9a825", "LOW": "#78909c"}


def _tag(text: str, colour: str) -> str:
    return (f'<span style="background:{colour};color:#fff;border-radius:4px;padding:1px 6px;'
            f'font-size:0.72rem;font-weight:600;margin-right:4px;">{html.escape(text)}</span>')


def _defect_card(defect, *, blocking: bool, selected: bool, requirements: dict[str, str],
                 resolved: bool = False) -> str:
    severity = getattr(defect.severity, "value", str(defect.severity))
    tags = _tag(severity, _SEVERITY_COLOURS.get(severity, "#78909c"))
    if blocking:
        tags += _tag("BLOCKING", "#6a1b9a")
    if selected:
        tags += _tag("SENT TO REPAIR", "#1565c0")
    if resolved:
        tags = _tag("RESOLVED", "#2e7d32") + tags
    traced = "".join(
        f'<div style="margin-top:3px;color:#37474f;"><b>{html.escape(r)}</b>: {html.escape(requirements[r])}</div>'
        for r in defect.requirement_ids if r in requirements
    )
    border = "#2e7d32" if resolved else ("#1565c0" if selected else "#e3e8ef")
    return (f'<div style="border:1px solid {border};border-left:4px solid {border};border-radius:6px;'
            f'padding:6px 9px;margin-bottom:6px;font-size:0.85rem;background:#fff;">'
            f'{tags}<span style="color:#5f6b7a;font-size:0.75rem;">{html.escape(defect.category)}</span>'
            f'<div style="margin-top:4px;">{html.escape(defect.description)}</div>{traced}</div>')


def show_iterations(job: dict) -> None:
    """Every candidate the loop evaluated: issues found, the repair, and what the next one fixed."""
    from src.pipeline.orchestrator import BLOCKING_DEFECT_PREFIXES

    state = job["state"]
    views = iteration_views(state, BLOCKING_DEFECT_PREFIXES)
    if not views:
        st.info("This run has no recorded iterations.")
        return
    best = state.best_candidate_iteration
    names = {
        v.iteration: f"Iteration {v.iteration + 1}"
        + (" · initial" if v.iteration == 0 else "")
        + (" · final" if v.iteration == best else "")
        for v in views
    }
    chosen = st.radio("Iteration", list(names), format_func=names.get, horizontal=True, key="iteration_pick")
    index = next(i for i, v in enumerate(views) if v.iteration == chosen)
    view = views[index]
    previous = views[index - 1] if index else None
    requirements = {r.id: r.text for r in state.requirements}

    def change(now, before, fmt="{:+d}"):
        return fmt.format(now - before) if previous is not None and now != before else None

    cols = st.columns(4)
    cols[0].metric("Defects", len(view.defects),
                   change(len(view.defects), len(previous.defects) if previous else 0), delta_color="inverse")
    cols[1].metric("Blocking defects", len(view.blocking_ids),
                   change(len(view.blocking_ids), len(previous.blocking_ids) if previous else 0), delta_color="inverse")
    coverage = round(100 * view.candidate.metrics.requirement_coverage)
    before = round(100 * previous.candidate.metrics.requirement_coverage) if previous else coverage
    cols[2].metric("Requirement coverage", f"{coverage}%", change(coverage, before, "{:+d} pts"))
    cols[3].metric("Steps drawn", sum(1 for n in view.candidate.diagram.nodes if n.type.value == "action"))

    svg = plantuml_svg(view.plantuml)
    if svg:
        show_svg(None, True, height=560, svg_bytes=svg)
    else:
        st.code(view.plantuml, language="text")
    st.html(
        '<div style="font-size:0.8rem;color:#5f6b7a;">'
        f'<span style="background:{FLAGGED};border:1px solid #c62828;padding:0 10px;margin-right:4px;"></span>'
        'step named by a defect sent to repair &nbsp;&nbsp;'
        f'<span style="background:{ADDED};border:1px solid #2e7d32;padding:0 10px;margin-right:4px;"></span>'
        'step added by the previous repair</div>'
    )

    issues, outcome = st.columns(2)
    with outcome:
        if view.resolved:
            st.markdown(f"**Resolved since Iteration {view.iteration}**")
            st.html("".join(_defect_card(d, blocking=False, selected=False, requirements=requirements, resolved=True)
                            for d in view.resolved))
        if view.added_steps:
            st.caption("Added: " + ", ".join(view.added_steps))
        if view.removed_steps:
            st.caption("Removed: " + ", ".join(view.removed_steps))

        st.markdown("**Repair**")
        if not view.repairs:
            last = view.iteration == views[-1].iteration
            st.info("No repair in this iteration: the loop stopped here." if last else "No repair in this iteration.")
        for repair in view.repairs:
            reason = str(repair.get("reason", "")).replace("_", " ").lower()
            if repair.get("accepted"):
                st.success(f"Accepted ({reason}). The repaired diagram is Iteration {view.iteration + 2}.")
            else:
                st.error(f"Rejected ({reason}). Rolled back to this diagram.")
            for item in repair.get("changes") or []:
                st.markdown(f"- {item}")

    with issues:
        st.markdown("**Issues found by verification**")
        order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        ranked = sorted(view.defects, key=lambda d: (d.id not in view.selected_ids, d.id not in view.blocking_ids,
                                                     order.get(getattr(d.severity, "value", d.severity), 4)))
        if ranked:
            st.html("".join(_defect_card(d, blocking=d.id in view.blocking_ids, selected=d.id in view.selected_ids,
                                         requirements=requirements) for d in ranked))
        else:
            st.success("No defects: the diagram passed every check.")


def summary_metrics(job: dict) -> list[tuple[str, str]]:
    state, m = job["state"], job["state"].metrics
    elapsed = m.get("total_execution_time_seconds", job.get("elapsed", 0))
    if job["mode"] == "hierarchical":
        return [
            ("Requirements", str(m.get("requirement_count", len(state.requirements)))),
            ("Modules", str(m.get("module_count", len(state.modules)))),
            ("Coverage", f"{100 * m.get('requirement_coverage', 0):.1f}%"),
            ("Nodes / edges", f"{m.get('full_diagram_nodes', 0)} / {m.get('full_diagram_edges', 0)}"),
            ("Open defects", str(m.get("remaining_module_defects", 0))),
            ("Runtime", f"{elapsed:.0f} s"),
        ]
    summary = m.get("research_summary", {})
    diagram = state.diagram
    return [
        ("Requirements", str(len(state.requirements))),
        ("Coverage", f"{100 * summary.get('best_requirement_coverage', 0):.1f}%"),
        ("Quality score", f"{m.get('best_score', 0):.1f}"),
        ("Nodes / edges", f"{len(diagram.nodes)} / {len(diagram.edges)}" if diagram else "–"),
        ("Open defects", str(len(state.defects))),
        ("Runtime", f"{elapsed:.0f} s"),
    ]


def _usage_row(name: str, u: dict | None) -> dict:
    u = u or {}
    cost = u.get("cost_usd")
    return {
        "Stage": name,
        "Calls": u.get("calls", 0),
        "Input tokens": u.get("input_tokens", 0),
        "Cached input": u.get("cached_input_tokens", 0),
        "Output tokens": u.get("output_tokens", 0),
        "Reasoning": u.get("reasoning_tokens", 0),
        "Cost (USD)": f"{cost:.4f}" if cost is not None else "n/a",
    }


def usage_tables(job: dict) -> tuple[dict, list[dict], list[dict]]:
    """(total, rows per iteration or module, rows per agent)."""
    usage = job["state"].metrics.get("token_usage") or {}
    total = usage.get("total") or {}
    if job["mode"] == "hierarchical":
        stages = [_usage_row(f"Document: {agent}", u) for agent, u in usage.get("document_level", {}).items()]
        stages += [_usage_row(f"Module {mid}", u) for mid, u in usage.get("by_module", {}).items()]
    else:
        stages = [_usage_row(phase.replace("_", " ").capitalize(), u) for phase, u in usage.get("by_phase", {}).items()]
    agents = [
        {"Agent": agent, **{k: v for k, v in _usage_row(agent, u).items() if k != "Stage"}}
        for agent, u in usage.get("by_agent", {}).items()
    ]
    return total, stages, agents


def report_markdown(job: dict) -> str | None:
    if job["mode"] == "hierarchical":
        report = Path(job["state"].metrics["run_output_dir"]) / "report.md"
        return report.read_text(encoding="utf-8") if report.exists() else None
    return None


@st.cache_resource(show_spinner="Setting up PlantUML…")
def _plantuml_ready() -> bool:
    """Fetch plantuml.jar once per server when Java is there but the jar is not."""
    try:
        return ensure_plantuml()
    except Exception as exc:  # no network: the app still runs, without images
        logging.getLogger(__name__).warning("PlantUML download failed: %s", exc)
        return False


_plantuml_ready()


# ------------------------------------------------------------------
# Sidebar: configuration
# ------------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")

    api_key = st.text_input(
        "OpenAI API key",
        type="password",
        help="Used only for your own runs in this browser session; it is not stored.",
    ).strip() or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        st.info("Enter an OpenAI API key to generate diagrams. Saved runs open without a key.")

    env_model = os.getenv("OPENAI_MODEL", MODELS[0])
    options = MODELS if env_model in MODELS else [env_model, *MODELS]
    choice = st.selectbox("LLM model", options + [CUSTOM_MODEL], index=options.index(env_model))
    model = st.text_input("Model name", placeholder="e.g. gpt-5.4-nano") if choice == CUSTOM_MODEL else choice

    mode = st.selectbox(
        "Pipeline mode",
        list(MODE_LABELS),
        format_func=MODE_LABELS.get,
        help="Hierarchical mode chunks long documents and generates one diagram per module "
        "before composing the full diagram.",
    )
    planner = st.selectbox(
        "Planner",
        ["outline", "graph"],
        format_func={"outline": "Outline (nested blocks)", "graph": "Graph (plan + generator)"}.get,
        help="Outline plans, repairs and draws the diagram as nested if/while/fork blocks, "
        "like an expert's diagram.",
    )
    max_iterations = st.slider("Max repair iterations", 0, 6, 3)

    with st.expander("Advanced (hierarchical)"):
        long_doc_tokens = st.number_input("Auto threshold (tokens)", 500, 50000, 12000, step=500)
        chunk_tokens = st.number_input("Chunk size (tokens)", 500, 20000, 2500, step=250)
        max_reqs = st.number_input("Max requirements per module", 5, 100, 25)
        max_workers = st.number_input("Parallel workers", 1, 16, 4)

    st.divider()
    st.markdown(
        '<div class="pipeline"><b>Agent pipeline</b><br>'
        "Requirement → Outline planner (or Planning → Generator) → Validator → "
        "Semantic Reviewer → Feedback → Repair → Regression check</div>",
        unsafe_allow_html=True,
    )
    if plantuml_command() is None:
        st.warning(
            "PlantUML (Java + `tools/plantuml.jar`) not found: diagrams can't be checked or rendered. "
            "Run `python -m scripts.install_plantuml`."
        )


# ------------------------------------------------------------------
# Main: input
# ------------------------------------------------------------------
st.title("UML Activity Diagram Generator")
st.markdown(
    '<p class="app-subtitle">Multi-agent generation of UML activity diagrams '
    "from natural-language requirements</p>",
    unsafe_allow_html=True,
)

examples = {p.name: p for d in EXAMPLE_DIRS if d.exists() for p in sorted(d.glob("*.txt"))}
source = st.radio("Requirements source", ["Upload document", "Paste text", "Example", "Saved run"], horizontal=True)

text, sample_id = "", "requirements"
if source == "Upload document":
    uploaded = st.file_uploader("Specification (.txt, .md, .pdf, .docx)", type=["txt", "md", "pdf", "docx"])
    if uploaded is not None:
        sample_id = Path(uploaded.name).stem
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / uploaded.name
            path.write_bytes(uploaded.getvalue())
            try:
                text = load_document(path)
            except Exception as exc:
                st.error(str(exc))
elif source == "Paste text":
    text = st.text_area("Requirements", height=220, placeholder="The customer enters their card details…")
    sample_id = "pasted_requirements"
elif source == "Saved run":
    runs = saved_runs()
    if runs:
        picked = st.selectbox("Finished run (opens its results without calling the LLM)", list(runs))
        if st.button("Open run", type="primary"):
            try:
                st.session_state["job"] = open_saved_run(runs[picked])
                st.session_state.pop("zip", None)
            except Exception as exc:
                st.error(f"Could not open the run: {exc}")
    else:
        st.info(f"No finished runs found under {SAVED_RUNS_DIR.relative_to(ROOT).as_posix()}/.")
else:
    if examples:
        name = st.selectbox("Example specification", list(examples))
        sample_id = Path(name).stem
        text = examples[name].read_text(encoding="utf-8", errors="replace")
    else:
        st.info("No examples found under data/pure.")

if text.strip():
    tokens = estimate_tokens(text)
    resolved = mode if mode != "auto" else ("hierarchical" if tokens > long_doc_tokens else "single")
    with st.expander(f"Preview · {len(text.split()):,} words · ~{tokens:,} tokens · runs in {resolved} mode"):
        st.text(text[:6000] + ("\n…" if len(text) > 6000 else ""))

sample_id = re.sub(r"[^A-Za-z0-9_-]+", "_", sample_id).strip("_") or "requirements"
can_run = bool(text.strip()) and bool(model) and bool(api_key)

if source != "Saved run" and st.button("Generate activity diagram", type="primary", disabled=not can_run):
    tokens = estimate_tokens(text)
    resolved = mode if mode != "auto" else ("hierarchical" if tokens > long_doc_tokens else "single")
    out = UI_OUTPUT_DIR / f"{sample_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{sample_id}_input.txt").write_text(text, encoding="utf-8")
    job = {
        "text": text,
        "sample_id": sample_id,
        "model": model,
        "mode": resolved,
        "max_iterations": int(max_iterations),
        "planner": planner,
        "chunk_tokens": int(chunk_tokens),
        "max_reqs": int(max_reqs),
        "max_workers": int(max_workers),
        "out": out,
    }
    (out / "ui_run_config.json").write_text(
        json.dumps({k: (str(v) if isinstance(v, Path) else v) for k, v in job.items() if k != "text"}, indent=2),
        encoding="utf-8",
    )
    job["api_key"] = api_key  # added after the config is written, so it never reaches disk
    run_with_progress(job)
    job.pop("api_key", None)
    (out / "pipeline.log").write_text("\n".join(job.get("logs", [])), encoding="utf-8")
    st.session_state["job"] = job
    st.session_state.pop("zip", None)


# ------------------------------------------------------------------
# Main: results
# ------------------------------------------------------------------
job = st.session_state.get("job")
if job:
    if "error" in job:
        st.error(job["error"])
        st.stop()

    st.divider()
    head_left, head_right = st.columns([3, 1], vertical_alignment="bottom")
    head_left.subheader("Results")
    head_left.caption(f"Model: {job['model']}  ·  Mode: {job['mode']}  ·  Output: {job['out'].relative_to(ROOT).as_posix()}")
    usage_total, usage_stages, usage_agents = usage_tables(job)
    if usage_total:
        cost = usage_total.get("cost_usd")
        head_left.caption(
            f"LLM usage: {usage_total.get('total_tokens', 0):,} tokens in {usage_total.get('calls', 0)} calls  ·  "
            f"Cost: {'$%.4f' % cost if cost is not None else 'n/a (model not in configs/pricing.json)'}"
        )
    if "zip" not in st.session_state:
        st.session_state["zip"] = zip_folder(job.get("zip_dir") or job["out"])
    head_right.download_button(
        "Download full output (.zip)",
        data=st.session_state["zip"],
        file_name=f"{(job.get('zip_dir') or job['out']).name}.zip",
        mime="application/zip",
        type="primary",
        width="stretch",
    )

    state = job["state"]
    diagrams = collect_diagrams(job)
    report = report_markdown(job)
    has_iterations = job["mode"] != "hierarchical" and bool(getattr(state, "candidates", None))
    tab_names = ["Diagram"] + (["Iterations"] if has_iterations else []) + ["Requirements", "PlantUML", "Usage"]
    tab_names += ["Report"] if report else []
    tabs = dict(zip(tab_names, st.tabs(tab_names)))

    with tabs["Diagram"]:
        if job.get("before_same"):
            st.caption("No repair changed the diagram, so the diagram before repair is also the final one.")
        if not diagrams:
            st.warning("No diagram was rendered.")
        else:
            c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
            selected = c1.selectbox("Diagram", list(diagrams), label_visibility="collapsed")
            fit = c2.toggle("Fit to width", value=True)
            entry = diagrams[selected]
            if entry.get("svg") and Path(entry["svg"]).exists():
                show_svg(entry["svg"], fit)
            elif entry.get("png") and Path(entry["png"]).exists():
                st.image(entry["png"])
            else:
                st.info("PlantUML/Java not available, so no image was rendered. See the PlantUML tab.")
            d1, d2, d3, _ = st.columns([1, 1, 1, 3])
            for col, kind, mime in ((d1, "svg", "image/svg+xml"), (d2, "png", "image/png")):
                path = entry.get(kind)
                if path and Path(path).exists():
                    col.download_button(f"Download {kind.upper()}", Path(path).read_bytes(),
                                        file_name=Path(path).name, mime=mime, width="stretch")
            if entry.get("puml_text"):
                d3.download_button("Download .puml", entry["puml_text"],
                                   file_name=f"{job['sample_id']}.puml", width="stretch")

    if has_iterations:
        with tabs["Iterations"]:
            show_iterations(job)

    with tabs["Requirements"]:
        rows = [
            {
                "ID": r.id,
                "Type": getattr(r.type, "value", r.type),
                "Requirement": r.text,
                "Actors": ", ".join(r.actors),
                "Section": r.section,
            }
            for r in state.requirements
        ]
        if not any(row["Section"] for row in rows):
            for row in rows:
                row.pop("Section")
        st.dataframe(rows, width="stretch", hide_index=True)

    with tabs["PlantUML"]:
        if diagrams:
            name = st.selectbox("Source", list(diagrams), key="puml_source")
            st.code(diagrams[name].get("puml_text") or "", language="text")

    with tabs["Usage"]:
        if not usage_total:
            st.info("No token usage was recorded for this run.")
        else:
            label = "module" if job["mode"] == "hierarchical" else "iteration"
            st.markdown(f"**Per {label}**")
            st.dataframe(usage_stages + [_usage_row("Total", usage_total)], width="stretch", hide_index=True)
            st.markdown("**Per agent**")
            st.dataframe(usage_agents, width="stretch", hide_index=True)
            st.caption("Prices: configs/pricing.json (USD per 1M tokens). Reasoning tokens are included in output tokens.")

    if report:
        with tabs["Report"]:
            st.markdown(report)
