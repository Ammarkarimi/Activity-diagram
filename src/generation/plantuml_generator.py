from __future__ import annotations

from collections import defaultdict

import networkx as nx

from src.models.domain import (
    ActivityDiagram,
    ActivityEdge,
    ActivityNode,
    NodeType,
)


class PlantUMLGenerator:
    """
    Deterministic ActivityDiagram IR -> PlantUML compiler.

    The IR is the source of truth.

    This compiler recognizes:

    - sequence
    - decisions
    - loops
    - forks
    - joins
    - merges
    - terminal nodes
    """

    def render(
        self,
        diagram: ActivityDiagram,
    ) -> str:

        self._validate(diagram)

        graph = self._build_graph(
            diagram
        )

        nodes = diagram.node_map()

        initial_nodes = [
            node
            for node in diagram.nodes
            if node.type == NodeType.INITIAL
        ]

        if len(initial_nodes) != 1:
            raise ValueError(
                "PlantUML compiler requires "
                "exactly one initial node."
            )

        initial = initial_nodes[0]

        lines: list[str] = []

        lines.append("@startuml")
        lines.append(
            f"title {self._escape(diagram.title)}"
        )
        lines.append("")
        lines.append("start")
        lines.append("")

        visited: set[str] = set()
        self._final_reached = False

        self._compile_node(
            graph=graph,
            nodes=nodes,
            diagram=diagram,
            node_id=initial.id,
            visited=visited,
            lines=lines,
        )

        if not self._final_reached:
            lines.append("")
            lines.append("stop")

        lines.append("")
        lines.append("@enduml")

        return "\n".join(lines)

    # ============================================================
    # GRAPH
    # ============================================================

    def _build_graph(
        self,
        diagram: ActivityDiagram,
    ) -> nx.DiGraph:

        graph = nx.DiGraph()

        for node in diagram.nodes:

            graph.add_node(
                node.id,
                type=node.type.value,
                label=node.label,
            )

        for edge in diagram.edges:

            graph.add_edge(
                edge.source,
                edge.target,
                edge_id=edge.id,
                guard=edge.guard,
            )

        return graph

    # ============================================================
    # NODE COMPILER
    # ============================================================

    def _compile_node(
        self,
        *,
        graph: nx.DiGraph,
        nodes: dict[str, ActivityNode],
        diagram: ActivityDiagram,
        node_id: str,
        visited: set[str],
        lines: list[str],
    ) -> None:

        if node_id in visited:
            return

        visited.add(node_id)

        node = nodes[node_id]

        # --------------------------------------------------------
        # INITIAL
        # --------------------------------------------------------

        if node.type == NodeType.INITIAL:

            self._compile_successors(
                graph=graph,
                nodes=nodes,
                diagram=diagram,
                node_id=node_id,
                visited=visited,
                lines=lines,
            )

            return

        # --------------------------------------------------------
        # FINAL
        # --------------------------------------------------------

        if node.type == NodeType.FINAL:
            lines.append("stop")
            self._final_reached = True
            return

        # --------------------------------------------------------
        # ACTION
        # --------------------------------------------------------

        if node.type == NodeType.ACTION:

            lines.append(
                f":{self._escape(node.label)};"
            )

            self._compile_successors(
                graph=graph,
                nodes=nodes,
                diagram=diagram,
                node_id=node_id,
                visited=visited,
                lines=lines,
            )

            return

        # --------------------------------------------------------
        # DECISION
        # --------------------------------------------------------

        if node.type == NodeType.DECISION:

            self._compile_decision(
                graph=graph,
                nodes=nodes,
                diagram=diagram,
                node=node,
                visited=visited,
                lines=lines,
            )

            return

        # --------------------------------------------------------
        # MERGE
        # --------------------------------------------------------

        if node.type == NodeType.MERGE:

            self._compile_successors(
                graph=graph,
                nodes=nodes,
                diagram=diagram,
                node_id=node_id,
                visited=visited,
                lines=lines,
            )

            return

        # --------------------------------------------------------
        # FORK
        # --------------------------------------------------------

        if node.type == NodeType.FORK:

            self._compile_fork(
                graph=graph,
                nodes=nodes,
                diagram=diagram,
                node=node,
                visited=visited,
                lines=lines,
            )

            return

        # --------------------------------------------------------
        # JOIN
        # --------------------------------------------------------

        if node.type == NodeType.JOIN:

            self._compile_successors(
                graph=graph,
                nodes=nodes,
                diagram=diagram,
                node_id=node_id,
                visited=visited,
                lines=lines,
            )

            return

        # --------------------------------------------------------
        # OBJECT
        # --------------------------------------------------------

        if node.type == NodeType.OBJECT:

            lines.append(
                f"floating note right: "
                f"{self._escape(node.label)}"
            )

            self._compile_successors(
                graph=graph,
                nodes=nodes,
                diagram=diagram,
                node_id=node_id,
                visited=visited,
                lines=lines,
            )

            return

        # --------------------------------------------------------
        # NOTE
        # --------------------------------------------------------

        if node.type == NodeType.NOTE:

            lines.append(
                f"note right: "
                f"{self._escape(node.label)}"
            )

            self._compile_successors(
                graph=graph,
                nodes=nodes,
                diagram=diagram,
                node_id=node_id,
                visited=visited,
                lines=lines,
            )

    # ============================================================
    # SUCCESSORS
    # ============================================================

    def _compile_successors(
        self,
        *,
        graph: nx.DiGraph,
        nodes: dict[str, ActivityNode],
        diagram: ActivityDiagram,
        node_id: str,
        visited: set[str],
        lines: list[str],
    ) -> None:

        successors = list(
            graph.successors(node_id)
        )

        if not successors:
            return

        # --------------------------------------------------------
        # One successor
        # --------------------------------------------------------

        if len(successors) == 1:

            target = successors[0]

            # Don't recursively traverse cycles.
            if target in visited:
                return

            self._compile_node(
                graph=graph,
                nodes=nodes,
                diagram=diagram,
                node_id=target,
                visited=visited,
                lines=lines,
            )

            return

        # --------------------------------------------------------
        # More than one successor from a normal node
        #
        # This should normally be a decision/fork.
        # We do not invent semantics here.
        # --------------------------------------------------------

        raise ValueError(
            f"Node {node_id} has multiple "
            "outgoing edges but is not a "
            "decision or fork."
        )

    # ============================================================
    # DECISION COMPILER
    # ============================================================

    def _compile_decision(
        self,
        *,
        graph: nx.DiGraph,
        nodes: dict[str, ActivityNode],
        diagram: ActivityDiagram,
        node: ActivityNode,
        visited: set[str],
        lines: list[str],
    ) -> None:

        outgoing = [
            edge
            for edge in diagram.edges
            if edge.source == node.id
        ]

        if len(outgoing) < 2:

            raise ValueError(
                f"Decision {node.id} needs "
                "at least two outgoing branches."
            )

        # Detect a simple loop:
        #
        # Decision
        #   |
        #   +--> body --> Decision
        #
        # This is a post-condition loop shape.

        loop_edge = None
        normal_edges = []

        for edge in outgoing:

            if self._reaches(
                graph,
                edge.target,
                node.id,
            ):
                loop_edge = edge
            else:
                normal_edges.append(edge)

        # --------------------------------------------------------
        # Post-condition loop
        # --------------------------------------------------------

        if loop_edge and normal_edges:

            exit_edge = normal_edges[0]

            lines.append(
                "repeat"
            )

            # Compile one path leading into the loop.
            self._compile_loop_body(
                graph=graph,
                nodes=nodes,
                diagram=diagram,
                start=loop_edge.target,
                loop_target=node.id,
                visited=visited,
                lines=lines,
                excluded_targets={
                    exit_edge.target,
                },
            )

            continue_guard = (
                self._guard(
                    loop_edge.guard,
                    "continue",
                )
            )

            lines.append(
                f"repeat while "
                f"({self._escape_guard(continue_guard)}) "
                f"is (true)"
            )

            exit_target = exit_edge.target

            if exit_target not in visited:

                self._compile_node(
                    graph=graph,
                    nodes=nodes,
                    diagram=diagram,
                    node_id=exit_target,
                    visited=visited,
                    lines=lines,
                )

            return

        # --------------------------------------------------------
        # Normal IF / ELSEIF / ELSE
        # --------------------------------------------------------

        merge_point = self._find_merge_point(graph, node.id, outgoing)
        branch_visited_sets = []

        for i, edge in enumerate(outgoing):

            is_first = (i == 0)
            is_last = (i == len(outgoing) - 1)

            guard = self._guard(
                edge.guard,
                "yes" if is_first else "otherwise" if is_last else "no"
            )

            if is_first:
                lines.append(
                    f"if ({self._escape(node.label)}) "
                    f"then ({self._escape_guard(guard)})"
                )
            elif is_last:
                lines.append(
                    f"else ({self._escape_guard(guard)})"
                )
            else:
                lines.append(
                    f"elseif ({self._escape(node.label)}) "
                    f"then ({self._escape_guard(guard)})"
                )

            branch_visited = visited.copy()
            if merge_point:
                branch_visited.add(merge_point)

            self._compile_branch(
                graph=graph,
                nodes=nodes,
                diagram=diagram,
                target=edge.target,
                visited=branch_visited,
                lines=lines,
            )
            branch_visited_sets.append(branch_visited)

        lines.append("endif")

        for bv in branch_visited_sets:
            visited.update(bv)

        if merge_point:
            visited.discard(merge_point)
            self._compile_node(
                graph=graph,
                nodes=nodes,
                diagram=diagram,
                node_id=merge_point,
                visited=visited,
                lines=lines,
            )

    def _find_merge_point(
        self,
        graph: nx.DiGraph,
        decision_id: str,
        outgoing_edges: list[ActivityEdge],
    ) -> str | None:
        
        reachable_sets = []
        for edge in outgoing_edges:
            target = edge.target
            reachable = set(nx.descendants(graph, target))
            reachable.add(target)
            reachable_sets.append(reachable)
            
        intersection = set()
        for i in range(len(reachable_sets)):
            for j in range(i + 1, len(reachable_sets)):
                intersection.update(reachable_sets[i].intersection(reachable_sets[j]))
                
        if not intersection:
            return None
            
        closest = None
        min_dist = float('inf')
        for node in intersection:
            try:
                dist = nx.shortest_path_length(graph, decision_id, node)
                if dist < min_dist:
                    min_dist = dist
                    closest = node
            except nx.NetworkXNoPath:
                pass
                
        return closest

    # ============================================================
    # LOOP BODY
    # ============================================================

    def _compile_loop_body(
        self,
        *,
        graph: nx.DiGraph,
        nodes: dict[str, ActivityNode],
        diagram: ActivityDiagram,
        start: str,
        loop_target: str,
        visited: set[str],
        lines: list[str],
        excluded_targets: set[str],
    ) -> None:

        original_visited = visited.copy()
        visited.add(loop_target)
        visited.update(excluded_targets)

        self._compile_node(
            graph=graph,
            nodes=nodes,
            diagram=diagram,
            node_id=start,
            visited=visited,
            lines=lines,
        )

        if loop_target not in original_visited:
            visited.discard(loop_target)
        for ex in excluded_targets:
            if ex not in original_visited:
                visited.discard(ex)

    # ============================================================
    # BRANCH
    # ============================================================

    def _compile_branch(
        self,
        *,
        graph: nx.DiGraph,
        nodes: dict[str, ActivityNode],
        diagram: ActivityDiagram,
        target: str,
        visited: set[str],
        lines: list[str],
    ) -> None:

        if target not in nodes:
            raise ValueError(
                f"Unknown branch target {target}"
            )

        if nodes[target].type in {
            NodeType.FINAL,
        }:
            # Compile it to get the stop emitted if needed
            self._compile_node(
                graph=graph,
                nodes=nodes,
                diagram=diagram,
                node_id=target,
                visited=visited,
                lines=lines,
            )
            return

        self._compile_node(
            graph=graph,
            nodes=nodes,
            diagram=diagram,
            node_id=target,
            visited=visited,
            lines=lines,
        )

    # ============================================================
    # FORK
    # ============================================================

    def _compile_fork(
        self,
        *,
        graph: nx.DiGraph,
        nodes: dict[str, ActivityNode],
        diagram: ActivityDiagram,
        node: ActivityNode,
        visited: set[str],
        lines: list[str],
    ) -> None:

        branches = list(
            graph.successors(node.id)
        )

        if len(branches) < 2:
            raise ValueError(
                f"Fork {node.id} needs at least "
                "two branches."
            )

        lines.append("fork")

        for index, target in enumerate(branches):

            if index > 0:
                lines.append(
                    "fork again"
                )

            self._compile_node(
                graph=graph,
                nodes=nodes,
                diagram=diagram,
                node_id=target,
                visited=visited.copy(),
                lines=lines,
            )

        lines.append("end fork")

    # ============================================================
    # GRAPH UTILITIES
    # ============================================================

    @staticmethod
    def _reaches(
        graph: nx.DiGraph,
        source: str,
        target: str,
    ) -> bool:

        if source == target:
            return True

        try:
            return nx.has_path(
                graph,
                source,
                target,
            )
        except nx.NetworkXError:
            return False

    # ============================================================
    # VALIDATION
    # ============================================================

    @staticmethod
    def _validate(
        diagram: ActivityDiagram,
    ) -> None:

        node_ids = {
            node.id
            for node in diagram.nodes
        }

        if not node_ids:
            raise ValueError(
                "Activity diagram contains no nodes."
            )

        if len(node_ids) != len(
            diagram.nodes
        ):
            raise ValueError(
                "Duplicate node IDs."
            )

        edge_ids = {
            edge.id
            for edge in diagram.edges
        }

        if len(edge_ids) != len(
            diagram.edges
        ):
            raise ValueError(
                "Duplicate edge IDs."
            )

        for edge in diagram.edges:

            if edge.source not in node_ids:
                raise ValueError(
                    f"Edge {edge.id} has "
                    f"missing source {edge.source}."
                )

            if edge.target not in node_ids:
                raise ValueError(
                    f"Edge {edge.id} has "
                    f"missing target {edge.target}."
                )

    # ============================================================
    # FORMATTERS
    # ============================================================

    @staticmethod
    def _escape(
        text: str,
    ) -> str:

        return (
            (text or "")
            .replace('"', "'")
            .replace("\n", " ")
            .replace(";", ",")
            .replace("|", " ")
            .replace("{", " ")
            .replace("}", " ")
            .replace("<", " ")
            .replace(">", " ")
            .strip()
        )

    @staticmethod
    def _escape_guard(
        text: str,
    ) -> str:

        return (
            (text or "")
            .replace("(", "")
            .replace(")", "")
            .replace('"', "'")
            .strip()
        )

    @staticmethod
    def _guard(
        guard: str | None,
        default: str,
    ) -> str:

        if not guard:
            return default

        value = guard.strip()

        if value.startswith("["):
            value = value[1:]

        if value.endswith("]"):
            value = value[:-1]

        return value.strip() or default