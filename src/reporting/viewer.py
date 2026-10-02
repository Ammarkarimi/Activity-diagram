"""A self-contained HTML viewer for the diagrams of a hierarchical run.

Opens from disk (no server, no internet): the overview, the full diagram and
every linked part, with scrolling, zoom and drag-to-pan, so diagrams larger
than an editor preview can show are still readable.
"""
from __future__ import annotations

import html
import json

from src.models.domain import HierarchicalState


def build_viewer(state: HierarchicalState, sample_id: str, part_sources: dict[int, str] | None = None) -> str:
    """``part_sources`` maps part numbers to their PlantUML text."""
    m = state.metrics
    part_sources = part_sources or {}
    parts = m.get("parts") or []
    views = [
        {
            "id": "overview",
            "title": "Overview",
            "subtitle": "One step per part",
            "svg": f"{sample_id}_overview.svg",
            "source": state.overview_plantuml,
            "from": [],
            "to": [],
        },
        {
            "id": "full",
            "title": "Full diagram",
            "subtitle": "All parts in one diagram",
            "svg": f"{sample_id}_full.svg",
            "source": state.full_plantuml,
            "from": [],
            "to": [],
        },
    ]
    for part in parts:
        if not part.get("stem"):
            continue
        views.append(
            {
                "id": f"part-{part['number']}",
                "title": f"Part {part['number']}: {part['name']}",
                "subtitle": part["module"],
                "svg": f"parts/{part['stem']}.svg",
                "source": part_sources.get(part["number"], ""),
                "from": [f"part-{n}" for n in part.get("comes_from", [])],
                "to": [f"part-{n}" for n in part.get("continues_to", [])],
            }
        )
    data = json.dumps(views, ensure_ascii=False).replace("</", "<\\/")
    title = html.escape(f"{state.decomposition.system_name if state.decomposition else sample_id} diagrams")
    return _TEMPLATE.replace("__TITLE__", title).replace("__DATA__", data)


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { --bg:#f6f7f9; --panel:#fff; --line:#d9dde3; --text:#1f2328; --muted:#636c76; --accent:#0b62d6; }
  * { box-sizing: border-box; }
  body { margin:0; font:14px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; color:var(--text); background:var(--bg);
         display:grid; grid-template-columns: 280px 1fr; height:100vh; }
  nav { border-right:1px solid var(--line); background:var(--panel); overflow:auto; padding:12px 0; }
  nav h1 { font-size:15px; margin:0 16px 12px; }
  nav h2 { font-size:11px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); margin:16px 16px 4px; }
  nav button { display:block; width:100%; text-align:left; border:0; background:none; padding:7px 16px; font:inherit; color:inherit; cursor:pointer; }
  nav button small { display:block; color:var(--muted); font-size:12px; }
  nav button:hover { background:#eef2f7; }
  nav button[aria-current="true"] { background:#e3edfc; box-shadow: inset 3px 0 var(--accent); }
  main { display:flex; flex-direction:column; min-width:0; min-height:0; }
  header { display:flex; flex-wrap:wrap; gap:8px; align-items:center; padding:8px 12px; border-bottom:1px solid var(--line); background:var(--panel); }
  header strong { margin-right:auto; font-size:15px; }
  header button, header a { border:1px solid var(--line); background:var(--panel); border-radius:6px; padding:4px 10px; font:inherit; color:inherit; cursor:pointer; text-decoration:none; }
  header button:hover, header a:hover { border-color:var(--accent); }
  #zoom { min-width:52px; text-align:center; color:var(--muted); }
  #links { display:flex; flex-wrap:wrap; gap:6px; padding:6px 12px; border-bottom:1px solid var(--line); background:var(--panel); }
  #links:empty { display:none; }
  #links button { border:1px solid var(--line); background:#f3f6fa; border-radius:999px; padding:2px 10px; font:inherit; font-size:13px; cursor:pointer; }
  #stage { flex:1; overflow:auto; cursor:grab; background:
           linear-gradient(90deg, #eef0f3 1px, transparent 1px) 0 0/24px 24px,
           linear-gradient(#eef0f3 1px, transparent 1px) 0 0/24px 24px, #fff; }
  #stage.dragging { cursor:grabbing; }
  #stage img { display:block; margin:16px; max-width:none; user-select:none; -webkit-user-drag:none; }
  #missing, #source { margin:24px; }
  #source { white-space:pre; font:12px/1.5 ui-monospace, Consolas, monospace; background:#fff; border:1px solid var(--line); padding:12px; overflow:auto; }
  [hidden] { display:none !important; }
  @media (max-width: 760px) { body { grid-template-columns: 1fr; grid-template-rows: auto 1fr; } nav { max-height:35vh; } }
</style>
</head>
<body>
<nav>
  <h1>__TITLE__</h1>
  <div id="menu"></div>
</nav>
<main>
  <header>
    <strong id="title"></strong>
    <button id="prev" title="Previous part">&larr;</button>
    <button id="next" title="Next part">&rarr;</button>
    <button id="fitw" title="Fit width (W)">Fit width</button>
    <button id="fitp" title="Fit page (F)">Fit page</button>
    <button id="out" title="Zoom out (-)">&minus;</button>
    <span id="zoom"></span>
    <button id="in" title="Zoom in (+)">+</button>
    <button id="one" title="Actual size (0)">100%</button>
    <button id="src" title="Show the PlantUML source">Source</button>
    <a id="open" target="_blank" rel="noopener">Open SVG</a>
  </header>
  <div id="links"></div>
  <div id="stage"><img id="img" alt=""></div>
  <p id="missing" hidden>No rendered image for this diagram. Install PlantUML
    (<code>python -m scripts.install_plantuml</code>) and run the pipeline again, or open the source.</p>
  <pre id="source" hidden></pre>
</main>
<script>
const views = __DATA__;
const byId = Object.fromEntries(views.map(v => [v.id, v]));
const $ = id => document.getElementById(id);
const stage = $("stage"), img = $("img");
let current = null, zoom = 1;

function menu() {
  const parts = views.filter(v => v.id.startsWith("part-"));
  const groups = [["Diagrams", views.filter(v => !v.id.startsWith("part-"))], ["Parts", parts]];
  for (const [name, items] of groups) {
    if (!items.length) continue;
    const h = document.createElement("h2"); h.textContent = name; $("menu").append(h);
    for (const v of items) {
      const b = document.createElement("button");
      b.dataset.id = v.id;
      b.innerHTML = "<span></span><small></small>";
      b.firstChild.textContent = v.title;
      const links = [];
      if (v.from.length) links.push("from " + v.from.map(id => id.slice(5)).join(", "));
      if (v.to.length) links.push("continues in " + v.to.map(id => id.slice(5)).join(", "));
      b.lastChild.textContent = links.join(" · ") || v.subtitle;
      b.onclick = () => show(v.id);
      $("menu").append(b);
    }
  }
}

function setZoom(z) {
  zoom = Math.min(4, Math.max(0.05, z));
  if (img.naturalWidth) img.style.width = Math.round(img.naturalWidth * zoom) + "px";
  $("zoom").textContent = Math.round(zoom * 100) + "%";
}
function fit(mode) {
  if (!img.naturalWidth) return;
  const w = (stage.clientWidth - 40) / img.naturalWidth, h = (stage.clientHeight - 40) / img.naturalHeight;
  setZoom(mode === "page" ? Math.min(w, h, 1) : Math.min(w, 1));
}
function zoomAt(factor, x, y) {
  const before = zoom, sx = stage.scrollLeft + x, sy = stage.scrollTop + y;
  setZoom(zoom * factor);
  const k = zoom / before;
  stage.scrollLeft = sx * k - x; stage.scrollTop = sy * k - y;
}

function show(id) {
  const v = byId[id]; if (!v) return;
  current = v;
  for (const b of document.querySelectorAll("nav button")) b.setAttribute("aria-current", b.dataset.id === id);
  $("title").textContent = v.title;
  $("open").href = v.svg;
  $("links").innerHTML = "";
  for (const [label, ids] of [["From", v.from], ["Continues in", v.to]]) {
    for (const target of ids) {
      if (!byId[target]) continue;
      const b = document.createElement("button");
      b.textContent = label + ": " + byId[target].title;
      b.onclick = () => show(target);
      $("links").append(b);
    }
  }
  $("source").hidden = true; stage.hidden = false; $("missing").hidden = true;
  $("source").textContent = v.source || "(no PlantUML source)";
  img.onload = () => fit("width");
  img.onerror = () => { stage.hidden = true; $("missing").hidden = false; };
  img.src = v.svg;
  stage.scrollTo(0, 0);
  const parts = views.filter(x => x.id.startsWith("part-"));
  const i = parts.indexOf(v);
  $("prev").disabled = i <= 0; $("next").disabled = i < 0 || i >= parts.length - 1;
  $("prev").onclick = () => i > 0 && show(parts[i - 1].id);
  $("next").onclick = () => i >= 0 && i < parts.length - 1 && show(parts[i + 1].id);
  try { history.replaceState(null, "", "#" + id); } catch (e) {}
}

$("fitw").onclick = () => fit("width");
$("fitp").onclick = () => fit("page");
$("in").onclick = () => zoomAt(1.25, stage.clientWidth / 2, stage.clientHeight / 2);
$("out").onclick = () => zoomAt(0.8, stage.clientWidth / 2, stage.clientHeight / 2);
$("one").onclick = () => setZoom(1);
$("src").onclick = () => {
  const showSource = $("source").hidden;
  $("source").hidden = !showSource;
  stage.hidden = showSource || !img.naturalWidth;
  $("missing").hidden = showSource || !!img.naturalWidth;
};
stage.addEventListener("wheel", e => {
  if (!e.ctrlKey && !e.metaKey) return;
  e.preventDefault();
  const r = stage.getBoundingClientRect();
  zoomAt(e.deltaY < 0 ? 1.15 : 1 / 1.15, e.clientX - r.left, e.clientY - r.top);
}, { passive: false });
let drag = null;
stage.addEventListener("mousedown", e => { drag = { x: e.clientX, y: e.clientY, l: stage.scrollLeft, t: stage.scrollTop }; stage.classList.add("dragging"); e.preventDefault(); });
window.addEventListener("mousemove", e => { if (drag) { stage.scrollLeft = drag.l - (e.clientX - drag.x); stage.scrollTop = drag.t - (e.clientY - drag.y); } });
window.addEventListener("mouseup", () => { drag = null; stage.classList.remove("dragging"); });
window.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT") return;
  if (e.key === "+" || e.key === "=") $("in").click();
  else if (e.key === "-") $("out").click();
  else if (e.key === "0") $("one").click();
  else if (e.key === "w") fit("width");
  else if (e.key === "f") fit("page");
  else if (e.key === "ArrowRight" && !$("next").disabled) $("next").click();
  else if (e.key === "ArrowLeft" && !$("prev").disabled) $("prev").click();
});
menu();
show(byId[location.hash.slice(1)] ? location.hash.slice(1) : "full");
</script>
</body>
</html>
"""
