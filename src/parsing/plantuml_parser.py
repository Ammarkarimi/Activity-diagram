from __future__ import annotations

import re

from src.models.domain import ActivityDiagram, ActivityEdge, ActivityNode, NodeType


class MinimalPlantUMLParser:
    """Parses the subset emitted by PlantUMLGenerator.

    The canonical source of truth remains the ActivityDiagram IR.
    """

    def parse(self, text: str) -> ActivityDiagram:
        nodes = []
        edges = []

        for line in text.splitlines():
            line = line.strip()
            if line.startswith('"') and ' as ' in line and '-->' not in line:
                m = re.match(r'"(.*?)"\s+as\s+(\w+)', line)
                if m:
                    nodes.append(ActivityNode(id=m.group(2), type=NodeType.ACTION, label=m.group(1)))
            elif line.startswith("(*) -->"):
                m = re.search(r' as (\w+)$', line)
                if m:
                    nodes.append(ActivityNode(id=m.group(1), type=NodeType.INITIAL, label="Start"))
            elif '--> (*)' in line:
                m = re.match(r'"(.*?)"\s+as\s+(\w+)\s+-->\s+\(\*\)', line)
                if m:
                    nodes.append(ActivityNode(id=m.group(2), type=NodeType.FINAL, label=m.group(1)))
            elif '-->' in line:
                m = re.match(r'(\w+)\s+-->\s+(\w+)(?:\s+:\s+(.*))?', line)
                if m:
                    edges.append(
                        ActivityEdge(
                            id=f"E{len(edges)+1}",
                            source=m.group(1),
                            target=m.group(2),
                            guard=m.group(3),
                        )
                    )

        return ActivityDiagram(nodes=nodes, edges=edges)
