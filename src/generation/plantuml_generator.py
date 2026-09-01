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

        self._compile_node(
            graph=graph,
            nodes=nodes,
            diagram=diagram,
            node_id=initial.id,
            visited=visited,
            lines=lines,
        )

        # Only add stop if final exists.
        if any(
            n.type == NodeType.FINAL
            for n in diagram.nodes
        ):
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
                start=node.id,
                loop_target=node.id,
                visited=visited,
                lines=lines,
                excluded_targets={
                    loop_edge.target,
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
        # Normal IF / ELSE
        # --------------------------------------------------------

        first = outgoing[0]
        second = outgoing[1]

        first_guard = self._guard(
            first.guard,
            "yes",
        )

        second_guard = self._guard(
            second.guard,
            "no",
        )

        lines.append(
            f"if ({self._escape(node.label)}) "
            f"then ({self._escape_guard(first_guard)})"
        )

        self._compile_branch(
            graph=graph,
            nodes=nodes,
            diagram=diagram,
            target=first.target,
            visited=visited,
            lines=lines,
        )

        lines.append(
            f"else ({self._escape_guard(second_guard)})"
        )

        self._compile_branch(
            graph=graph,
            nodes=nodes,
            diagram=diagram,
            target=second.target,
            visited=visited,
            lines=lines,
        )

        lines.append("endif")

        # Additional branches.
        for extra in outgoing[2:]:

            guard = self._guard(
                extra.guard,
                "otherwise",
            )

            lines.append(
                f"else "
                f"({self._escape_guard(guard)})"
            )

            self._compile_branch(
                graph=graph,
                nodes=nodes,
                diagram=diagram,
                target=extra.target,
                visited=visited,
                lines=lines,
            )

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

        successors = list(
            graph.successors(start)
        )

        for target in successors:

            if target in excluded_targets:
                continue

            if target == loop_target:
                continue

            if target in visited:
                continue

            self._compile_node(
                graph=graph,
                nodes=nodes,
                diagram=diagram,
                node_id=target,
                visited=visited,
                lines=lines,
            )

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

        # Don't globally suppress a branch just because the merge
        # point was visited by the other branch. The compiler
        # terminates branch traversal at merge/final/revisited nodes.
        if target not in nodes:
            raise ValueError(
                f"Unknown branch target {target}"
            )

        if nodes[target].type in {
            NodeType.FINAL,
        }:
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