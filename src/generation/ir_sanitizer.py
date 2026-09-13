from __future__ import annotations

import logging
import re
from typing import Dict, List, Set

from src.models.domain import ActivityDiagram, ActivityEdge, ActivityNode, EdgeType

logger = logging.getLogger(__name__)


class IRSanitizer:
    """
    Sanitizes ActivityDiagram IR objects produced by the LLM before they reach
    the PlantUML compiler or validators.
    """

    @staticmethod
    def sanitize(diagram: ActivityDiagram) -> ActivityDiagram:
        """
        Sanitize an ActivityDiagram instance according to predefined rules.
        Returns a new ActivityDiagram instance.
        """
        if not diagram:
            return diagram

        def normalize_id(id_str: str) -> str:
            if not id_str:
                return id_str
            id_str = id_str.strip().replace(" ", "_")
            id_str = re.sub(r"[^a-zA-Z0-9_\-]", "_", id_str)
            return id_str

        old_to_new_node_ids: Dict[str, str] = {}
        seen_node_ids: Set[str] = set()
        new_nodes: List[ActivityNode] = []

        # 1, 4, 9: Normalize node IDs, remove duplicates, fix requirement_ids
        for node in diagram.nodes:
            old_id = node.id
            new_id = normalize_id(old_id)
            
            old_to_new_node_ids[old_id] = new_id

            if new_id not in seen_node_ids:
                seen_node_ids.add(new_id)
                new_req_ids = node.requirement_ids if node.requirement_ids is not None else []
                
                new_nodes.append(
                    ActivityNode(
                        id=new_id,
                        type=node.type,
                        label=node.label,
                        requirement_ids=new_req_ids,
                    )
                )

        if len(new_nodes) != len(diagram.nodes):
            logger.warning(
                f"Removed {len(diagram.nodes) - len(new_nodes)} duplicate nodes."
            )

        if any(old_id != new_id for old_id, new_id in old_to_new_node_ids.items()):
            changed = sum(1 for old_id, new_id in old_to_new_node_ids.items() if old_id != new_id)
            logger.warning(f"Normalized {changed} node IDs.")

        seen_edge_ids: Set[str] = set()
        new_edges: List[ActivityEdge] = []
        dangling_edges_removed = 0

        # 2, 5, 6, 7, 8, 9, 11: Edge sanitization
        for edge in diagram.edges:
            old_id = edge.id
            new_id = normalize_id(old_id)

            # 3, 11: Update source and target references
            new_source = old_to_new_node_ids.get(edge.source, normalize_id(edge.source))
            new_target = old_to_new_node_ids.get(edge.target, normalize_id(edge.target))

            # 6: Remove dangling edges
            if new_source not in seen_node_ids or new_target not in seen_node_ids:
                dangling_edges_removed += 1
                continue

            # 5: Remove duplicate edges
            if new_id in seen_edge_ids:
                continue

            seen_edge_ids.add(new_id)

            # 7: Fix edge type
            new_type = edge.type
            if new_type is None or not isinstance(new_type, EdgeType) and new_type not in [e.value for e in EdgeType]:
                new_type = EdgeType.CONTROL
            elif isinstance(new_type, str) and new_type in [e.value for e in EdgeType]:
                new_type = EdgeType(new_type)

            # 8: Fix edge guard
            new_guard = None if edge.guard == "" else edge.guard

            # 9: Fix requirement_ids
            new_req_ids = edge.requirement_ids if edge.requirement_ids is not None else []

            new_edges.append(
                ActivityEdge(
                    id=new_id,
                    source=new_source,
                    target=new_target,
                    type=new_type,
                    guard=new_guard,
                    requirement_ids=new_req_ids,
                )
            )

        if dangling_edges_removed > 0:
            logger.warning(f"Removed {dangling_edges_removed} dangling edges.")

        total_edges_ignored = len(diagram.edges) - len(new_edges) - dangling_edges_removed
        if total_edges_ignored > 0:
            logger.warning(f"Removed {total_edges_ignored} duplicate edges.")

        # 10: Fix title
        new_title = diagram.title if diagram.title else "Activity Diagram"

        return ActivityDiagram(nodes=new_nodes, edges=new_edges, title=new_title)
