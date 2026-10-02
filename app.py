"""Web UI for the activity-diagram pipeline.

Run with:  streamlit run app.py

The UI only calls the existing pipelines (MultiAgentPipeline and
HierarchicalPipeline); it does not change their behaviour.
"""
from __future__ import annotations

import base64
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

load_dotenv()

ROOT = Path(__file__).resolve().parent
UI_OUTPUT_DIR = ROOT / "outputs" / "ui"
EXAMPLE_DIRS = [ROOT / "data" / "examples", ROOT / "data" / "pure"]

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
        if mode == "hierarchical":
            from src.pipeline.hierarchical import HierarchicalPipeline

            state = HierarchicalPipeline(
                model=job["model"],
                chunk_tokens=job["chunk_tokens"],
                max_requirements_per_module=job["max_reqs"],
                max_workers=job["max_workers"],
            ).run(
                sample_id=job["sample_id"],
                requirement_text=text,
                max_iterations=job["max_iterations"],
                output_dir=out,
                document_title=job["sample_id"],
            )
        else:
            from src.pipeline.orchestrator import MultiAgentPipeline

            state = MultiAgentPipeline(model=job["model"]).run(
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


def show_svg(svg_path: str, fit: bool, height: int = 700) -> None:
    # Shown as an <img> so the SVG (whose labels come from LLM output) is never
    # executed as markup.
    data = base64.b64encode(Path(svg_path).read_bytes()).decode("ascii")
    size = "max-width:100%;height:auto;" if fit else "max-width:none;"
    st.html(
        f'''<div style="height:{height}px;overflow:auto;background:#fff;border:1px solid #e3e8ef;
                    border-radius:8px;padding:16px;box-sizing:border-box;text-align:center;">
             <img src="data:image/svg+xml;base64,{data}" style="{size}" alt="Activity diagram">
           </div>'''
    )


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


# ------------------------------------------------------------------
# Sidebar: configuration
# ------------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")

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
    max_iterations = st.slider("Max repair iterations", 0, 6, 3)

    with st.expander("Advanced (hierarchical)"):
        long_doc_tokens = st.number_input("Auto threshold (tokens)", 500, 50000, 3000, step=500)
        chunk_tokens = st.number_input("Chunk size (tokens)", 500, 20000, 2500, step=250)
        max_reqs = st.number_input("Max requirements per module", 5, 100, 25)
        max_workers = st.number_input("Parallel workers", 1, 16, 4)

    st.divider()
    st.markdown(
        '<div class="pipeline"><b>Agent pipeline</b><br>'
        "Requirement → Planning → Generator → Validator → Semantic Reviewer → "
        "Feedback → Typed Repair → Regression check</div>",
        unsafe_allow_html=True,
    )
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is not set (add it to `.env`).")


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
source = st.radio("Requirements source", ["Upload document", "Paste text", "Example"], horizontal=True)

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
else:
    if examples:
        name = st.selectbox("Example specification", list(examples))
        sample_id = Path(name).stem
        text = examples[name].read_text(encoding="utf-8", errors="replace")
    else:
        st.info("No examples found under data/.")

if text.strip():
    tokens = estimate_tokens(text)
    resolved = mode if mode != "auto" else ("hierarchical" if tokens > long_doc_tokens else "single")
    with st.expander(f"Preview · {len(text.split()):,} words · ~{tokens:,} tokens · runs in {resolved} mode"):
        st.text(text[:6000] + ("\n…" if len(text) > 6000 else ""))

sample_id = re.sub(r"[^A-Za-z0-9_-]+", "_", sample_id).strip("_") or "requirements"
can_run = bool(text.strip()) and bool(model) and bool(os.getenv("OPENAI_API_KEY"))

if st.button("Generate activity diagram", type="primary", disabled=not can_run):
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
        "chunk_tokens": int(chunk_tokens),
        "max_reqs": int(max_reqs),
        "max_workers": int(max_workers),
        "out": out,
    }
    (out / "ui_run_config.json").write_text(
        json.dumps({k: (str(v) if isinstance(v, Path) else v) for k, v in job.items() if k != "text"}, indent=2),
        encoding="utf-8",
    )
    run_with_progress(job)
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
        st.session_state["zip"] = zip_folder(job["out"])
    head_right.download_button(
        "Download full output (.zip)",
        data=st.session_state["zip"],
        file_name=f"{job['out'].name}.zip",
        mime="application/zip",
        type="primary",
        width="stretch",
    )

    for col, (label, value) in zip(st.columns(6), summary_metrics(job)):
        col.metric(label, value)

    state = job["state"]
    diagrams = collect_diagrams(job)
    report = report_markdown(job)
    tab_names = ["Diagram", "Requirements", "PlantUML", "Usage"] + (["Report"] if report else [])
    tabs = st.tabs(tab_names)

    with tabs[0]:
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

    with tabs[1]:
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

    with tabs[2]:
        if diagrams:
            name = st.selectbox("Source", list(diagrams), key="puml_source")
            st.code(diagrams[name].get("puml_text") or "", language="text")

    with tabs[3]:
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
        with tabs[4]:
            st.markdown(report)
