from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.models.domain import ActivityDiagram, ActivityEdge, ActivityNode, NodeType

Tail = tuple[str, "str | None"]


@dataclass
class _Frame:
    kind: str
    node: str
    collected: list[Tail] = field(default_factory=list)
    has_else: bool = False


_IF = re.compile(r"^if\s*\((.*)\)\s*(?:then\s*(?:\((.*)\))?)?\s*$", re.IGNORECASE)
_ELSEIF = re.compile(r"^(?:elseif|else\s+if)\s*\((.*)\)\s*(?:then\s*(?:\((.*)\))?)?\s*$", re.IGNORECASE)
_ELSE = re.compile(r"^else\s*(?:\((.*)\))?\s*$", re.IGNORECASE)
_WHILE = re.compile(r"^while\s*\((.*?)\)\s*(?:is\s*\((.*)\))?\s*$", re.IGNORECASE)
_ENDWHILE = re.compile(r"^end\s*while\s*(?:\((.*)\))?\s*$", re.IGNORECASE)
_REPEAT_WHILE = re.compile(
    r"^repeat\s+while\s*\((.*?)\)\s*(?:is\s*\((.*?)\))?\s*(?:not\s*\((.*?)\))?\s*$", re.IGNORECASE
)
_LANE = re.compile(r"^\|(?:[^|]*\|)?([^|]+)\|$")
_ARROW_LABEL = re.compile(r"^-+>\s*(.*?);?\s*$")


class ActivityPlantUMLParser:
    """Parses PlantUML *new-syntax* activity diagrams into the ActivityDiagram IR.

    Supports start/stop/end/kill/detach, actions (single and multi-line),
    if/elseif/else/endif, while/endwhile, repeat/repeat while,
    fork/split (again/end), swimlanes, arrow labels, and skips notes,
    partitions, titles and skinparams. Gold-standard and generated diagrams
    go through the same parser so they can be compared on equal terms.
    """

    def parse(self, text: str) -> ActivityDiagram:
        self.nodes: list[ActivityNode] = []
        self.edges: list[ActivityEdge] = []
        self.lane: str | None = None
        tails: list[Tail] = []
        stack: list[_Frame] = []
        title = "Activity Diagram"

        lines = self._logical_lines(text)
        index = 0
        while index < len(lines):
            line = lines[index]
            index += 1
            low = line.lower()

            if low.startswith("note") and ":" not in line and not low.startswith("note as"):
                # multi-line note block
                while index < len(lines) and not lines[index].lower().startswith("end note"):
                    index += 1
                index += 1
                continue
            if low.startswith(("note", "skinparam", "@startuml", "@enduml", "'", "hide", "legend")):
                continue
            if low.startswith("title"):
                title = line[5:].strip() or title
                continue
            if low.startswith("partition") or line in {"{", "}"} or low.startswith("group") or low == "end group":
                continue

            lane = _LANE.match(line)
            if lane:
                self.lane = lane.group(1).strip()
                continue

            if low == "start":
                node = self._node(NodeType.INITIAL, "Start")
                self._connect(tails, node)
                tails = [(node, None)]
                continue
            if low in {"stop", "end", "kill"}:
                node = self._node(NodeType.FINAL, "End")
                self._connect(tails, node)
                tails = []
                continue
            if low == "detach":
                tails = []
                continue

            arrow = _ARROW_LABEL.match(line)
            if arrow:
                label = arrow.group(1).strip() or None
                tails = [(src, guard or label) for src, guard in tails]
                continue

            if line.startswith(":"):
                label = line[1:]
                label = re.sub(r"[;|<>\]}/]\s*$", "", label).strip()
                node = self._node(NodeType.ACTION, self._clean(label))
                self._connect(tails, node)
                tails = [(node, None)]
                continue

            m = _ELSEIF.match(line)
            if m and stack and stack[-1].kind == "if":
                frame = stack[-1]
                frame.collected += tails
                # elseif = nested decision on the "else" path of the previous one
                node = self._node(NodeType.DECISION, self._clean(m.group(1)))
                self._connect([(frame.node, "else")], node)
                frame.node = node
                tails = [(node, self._clean(m.group(2) or "yes"))]
                continue

            m = _IF.match(line)
            if m:
                node = self._node(NodeType.DECISION, self._clean(m.group(1)))
                self._connect(tails, node)
                stack.append(_Frame("if", node))
                tails = [(node, self._clean(m.group(2) or "yes"))]
                continue

            m = _ELSE.match(line)
            if m and stack and stack[-1].kind == "if":
                frame = stack[-1]
                frame.collected += tails
                frame.has_else = True
                tails = [(frame.node, self._clean(m.group(1) or "else"))]
                continue

            if low in {"endif", "end if"} and stack and stack[-1].kind == "if":
                frame = stack.pop()
                collected = frame.collected + tails
                if not frame.has_else:
                    collected.append((frame.node, "else"))
                tails = self._merge(collected)
                continue

            m = _WHILE.match(line)
            if m:
                node = self._node(NodeType.DECISION, self._clean(m.group(1)))
                self._connect(tails, node)
                stack.append(_Frame("while", node))
                tails = [(node, self._clean(m.group(2) or "yes"))]
                continue

            m = _ENDWHILE.match(line)
            if m and stack and stack[-1].kind == "while":
                frame = stack.pop()
                self._connect(tails, frame.node)
                tails = [(frame.node, self._clean(m.group(1) or "exit"))]
                continue

            if low == "repeat":
                node = self._node(NodeType.MERGE, "repeat")
                self._connect(tails, node)
                stack.append(_Frame("repeat", node))
                tails = [(node, None)]
                continue

            m = _REPEAT_WHILE.match(line)
            if m and stack and stack[-1].kind == "repeat":
                frame = stack.pop()
                node = self._node(NodeType.DECISION, self._clean(m.group(1)))
                self._connect(tails, node)
                self._connect([(node, self._clean(m.group(2) or "yes"))], frame.node)
                tails = [(node, self._clean(m.group(3) or "no"))]
                continue
            if low.startswith("backward"):
                continue

            if low in {"fork", "split"}:
                node = self._node(NodeType.FORK, "fork")
                self._connect(tails, node)
                stack.append(_Frame("fork", node))
                tails = [(node, None)]
                continue
            if low in {"fork again", "split again"} and stack and stack[-1].kind == "fork":
                stack[-1].collected += tails
                tails = [(stack[-1].node, None)]
                continue
            if (low.startswith("end fork") or low.startswith("end split") or low.startswith("end merge")) and stack and stack[-1].kind == "fork":
                frame = stack.pop()
                collected = frame.collected + tails
                node = self._node(NodeType.JOIN, "join")
                self._connect(collected, node)
                tails = [(node, None)]
                continue
            # Unknown statement: ignored.

        return ActivityDiagram(title=title, nodes=self.nodes, edges=self.edges)

    # ------------------------------------------------------------

    def _merge(self, collected: list[Tail]) -> list[Tail]:
        if len(collected) <= 1:
            return collected
        node = self._node(NodeType.MERGE, "merge")
        self._connect(collected, node)
        return [(node, None)]

    def _node(self, node_type: NodeType, label: str) -> str:
        node_id = f"N{len(self.nodes) + 1}"
        self.nodes.append(ActivityNode(id=node_id, type=node_type, label=label, lane=self.lane))
        return node_id

    def _connect(self, tails: list[Tail], target: str) -> None:
        for source, guard in tails:
            self.edges.append(
                ActivityEdge(id=f"E{len(self.edges) + 1}", source=source, target=target, guard=guard)
            )

    @staticmethod
    def _clean(text: str | None) -> str:
        return " ".join((text or "").replace("\\n", " ").split())

    @staticmethod
    def _logical_lines(text: str) -> list[str]:
        """Strip comments and join multi-line ``:action`` statements."""
        result: list[str] = []
        buffer: str | None = None
        for raw in text.splitlines():
            line = raw.strip()
            if buffer is not None:
                buffer += " " + line
                if re.search(r"[;|<>\]}/]\s*$", line):
                    result.append(buffer)
                    buffer = None
                continue
            if not line or line.startswith("'"):
                continue
            if line.startswith(":") and not re.search(r"[;|<>\]}/]\s*$", line):
                buffer = line
                continue
            result.append(line)
        if buffer is not None:
            result.append(buffer)
        return result
