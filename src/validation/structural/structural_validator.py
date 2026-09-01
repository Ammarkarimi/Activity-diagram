from __future__ import annotations

import networkx as nx

from src.models.domain import (
    ActivityDiagram,
    Defect,
    NodeType,
    Severity,
    ValidationResult,
)
from src.validation.syntax.syntax_validator import (
    SyntaxValidator,
)


class StructuralValidator:

    def __init__(self) -> None:

        self.syntax = (
            SyntaxValidator()
        )

    def validate(
        self,
        diagram: ActivityDiagram,
    ) -> ValidationResult:

        defects: list[Defect] = []

        rule_counts = {
            "STRUCT-001": 0,
            "STRUCT-002": 0,
            "STRUCT-003": 0,
            "STRUCT-004": 0,
            "DECISION-001": 0,
            "DECISION-002": 0,
            "FLOW-001": 0,
            "CONCURRENCY-001": 0,
            "TERM-001": 0,
        }

        syntax_result = (
            self.syntax.validate(
                diagram
            )
        )

        defects.extend(
            syntax_result.defects
        )

        # ============================================================
        # NODE IDS
        # ============================================================

        node_ids = {
            node.id
            for node in diagram.nodes
        }

        # ============================================================
        # INITIAL NODE
        # ============================================================

        initial_nodes = [
            node
            for node in diagram.nodes
            if node.type
            == NodeType.INITIAL
        ]

        if len(initial_nodes) != 1:

            rule_counts[
                "STRUCT-001"
            ] += 1

            defects.append(
                Defect(
                    id="D-STRUCT-001",
                    category="STRUCTURAL",
                    severity=Severity.HIGH,
                    description=(
                        "Expected exactly "
                        "one initial node."
                    ),
                    node_ids=[
                        node.id
                        for node
                        in initial_nodes
                    ],
                    edge_ids=[],
                    requirement_ids=[],
                    evidence=(
                        f"Found "
                        f"{len(initial_nodes)} "
                        "initial nodes."
                    ),
                    suggested_action=(
                        "Add or remove initial "
                        "nodes so the diagram "
                        "has exactly one."
                    ),
                )
            )

        # ============================================================
        # FINAL NODE
        # ============================================================

        final_nodes = [
            node
            for node in diagram.nodes
            if node.type
            == NodeType.FINAL
        ]

        if not final_nodes:

            rule_counts[
                "STRUCT-002"
            ] += 1

            defects.append(
                Defect(
                    id="D-STRUCT-002",
                    category="STRUCTURAL",
                    severity=Severity.HIGH,
                    description=(
                        "No final node found."
                    ),
                    node_ids=[],
                    edge_ids=[],
                    requirement_ids=[],
                    evidence="No final node exists.",
                    suggested_action=(
                        "Add a final node."
                    ),
                )
            )

        # ============================================================
        # GRAPH
        # ============================================================

        graph = nx.DiGraph()

        graph.add_nodes_from(
            node_ids
        )

        # ============================================================
        # EDGE REFERENCES
        # ============================================================

        for edge in diagram.edges:

            if (
                edge.source
                not in node_ids
                or edge.target
                not in node_ids
            ):

                rule_counts[
                    "STRUCT-004"
                ] += 1

                defects.append(
                    Defect(
                        id=(
                            f"D-STRUCT-004-"
                            f"{edge.id}"
                        ),
                        category="STRUCTURAL",
                        severity=Severity.CRITICAL,
                        description=(
                            f"Edge {edge.id} "
                            "references a "
                            "missing node."
                        ),
                        node_ids=[],
                        edge_ids=[
                            edge.id
                        ],
                        requirement_ids=[],
                        evidence=(
                            f"{edge.source} "
                            f"-> {edge.target}"
                        ),
                        suggested_action=(
                            "Repair the edge "
                            "source/target."
                        ),
                    )
                )

                continue

            graph.add_edge(
                edge.source,
                edge.target,
                edge_id=edge.id,
                guard=edge.guard,
            )

        # ============================================================
        # REACHABILITY
        # ============================================================

        if initial_nodes:

            reachable: set[str] = set()

            for initial in initial_nodes:

                reachable.update(
                    nx.descendants(
                        graph,
                        initial.id,
                    )
                )

                reachable.add(
                    initial.id
                )

            unreachable = (
                node_ids
                - reachable
            )

            if unreachable:

                rule_counts[
                    "STRUCT-003"
                ] += len(unreachable)

                defects.append(
                    Defect(
                        id="D-STRUCT-003",
                        category="STRUCTURAL",
                        severity=Severity.HIGH,
                        description=(
                            "Unreachable "
                            "diagram nodes."
                        ),
                        node_ids=sorted(
                            unreachable
                        ),
                        edge_ids=[],
                        requirement_ids=[],
                        evidence=(
                            str(
                                sorted(
                                    unreachable
                                )
                            )
                        ),
                        suggested_action=(
                            "Connect all "
                            "diagram nodes "
                            "to the main "
                            "workflow."
                        ),
                    )
                )

            # --------------------------------------------------------
            # Initial -> Final
            # --------------------------------------------------------

            reachable_final = False

            for initial in initial_nodes:

                for final in final_nodes:

                    try:

                        if nx.has_path(
                            graph,
                            initial.id,
                            final.id,
                        ):
                            reachable_final = True
                            break

                    except nx.NetworkXError:
                        pass

                if reachable_final:
                    break

            if not reachable_final:

                rule_counts[
                    "TERM-001"
                ] += 1

                defects.append(
                    Defect(
                        id="D-TERM-001",
                        category="TERMINATION",
                        severity=Severity.HIGH,
                        description=(
                            "No path exists "
                            "from the initial "
                            "node to a final "
                            "node."
                        ),
                        node_ids=[],
                        edge_ids=[],
                        requirement_ids=[],
                        evidence="Graph reachability analysis.",
                        suggested_action=(
                            "Repair the control "
                            "flow so at least "
                            "one valid path "
                            "terminates."
                        ),
                    )
                )

        # ============================================================
        # DECISIONS
        # ============================================================

        for node in diagram.nodes:

            if node.type != NodeType.DECISION:
                continue

            outgoing = [
                edge
                for edge in diagram.edges
                if edge.source == node.id
            ]

            if len(outgoing) < 2:

                rule_counts[
                    "DECISION-001"
                ] += 1

                defects.append(
                    Defect(
                        id=(
                            f"D-DECISION-001-"
                            f"{node.id}"
                        ),
                        category="DECISION",
                        severity=Severity.HIGH,
                        description=(
                            f"Decision "
                            f"{node.id} has "
                            "fewer than "
                            "two outgoing "
                            "branches."
                        ),
                        node_ids=[
                            node.id
                        ],
                        edge_ids=[
                            edge.id
                            for edge
                            in outgoing
                        ],
                        requirement_ids=(
                            node.requirement_ids
                        ),
                        evidence=(
                            f"{len(outgoing)} "
                            "outgoing edges."
                        ),
                        suggested_action=(
                            "Add the required "
                            "decision branches."
                        ),
                    )
                )

            for edge in outgoing:

                if not edge.guard:

                    rule_counts[
                        "DECISION-002"
                    ] += 1

                    defects.append(
                        Defect(
                            id=(
                                f"D-DECISION-002-"
                                f"{edge.id}"
                            ),
                            category="DECISION",
                            severity=Severity.MEDIUM,
                            description=(
                                f"Decision edge "
                                f"{edge.id} has "
                                "no guard."
                            ),
                            node_ids=[
                                node.id
                            ],
                            edge_ids=[
                                edge.id
                            ],
                            requirement_ids=(
                                edge.requirement_ids
                            ),
                            evidence=(
                                "Missing guard."
                            ),
                            suggested_action=(
                                "Add a meaningful "
                                "guard condition."
                            ),
                        )
                    )

        # ============================================================
        # DEAD END ACTIONS
        # ============================================================

        for node in diagram.nodes:

            if node.type != NodeType.ACTION:
                continue

            if graph.out_degree(
                node.id
            ) == 0:

                rule_counts[
                    "FLOW-001"
                ] += 1

                defects.append(
                    Defect(
                        id=(
                            f"D-FLOW-001-"
                            f"{node.id}"
                        ),
                        category="CONTROL_FLOW",
                        severity=Severity.MEDIUM,
                        description=(
                            f"Action "
                            f"{node.id} is "
                            "a dead end."
                        ),
                        node_ids=[
                            node.id
                        ],
                        edge_ids=[],
                        requirement_ids=(
                            node.requirement_ids
                        ),
                        evidence=node.label,
                        suggested_action=(
                            "Connect the action "
                            "to its next "
                            "workflow step or "
                            "final node."
                        ),
                    )
                )

        # ============================================================
        # CONCURRENCY
        # ============================================================

        for node in diagram.nodes:

            if node.type != NodeType.FORK:
                continue

            branches = list(
                graph.successors(
                    node.id
                )
            )

            if len(branches) < 2:

                rule_counts[
                    "CONCURRENCY-001"
                ] += 1

                defects.append(
                    Defect(
                        id=(
                            f"D-CON-001-"
                            f"{node.id}"
                        ),
                        category="CONCURRENCY",
                        severity=Severity.HIGH,
                        description=(
                            f"Fork "
                            f"{node.id} has "
                            "fewer than "
                            "two branches."
                        ),
                        node_ids=[
                            node.id
                        ],
                        edge_ids=[],
                        requirement_ids=(
                            node.requirement_ids
                        ),
                        evidence=(
                            f"{len(branches)} "
                            "branches."
                        ),
                        suggested_action=(
                            "Add at least "
                            "two parallel "
                            "branches."
                        ),
                    )
                )

        # ============================================================
        # SCORE
        # ============================================================

        denominator = max(
            1,
            len(diagram.nodes)
            + len(diagram.edges),
        )

        score = max(
            0.0,
            1.0
            - (
                len(defects)
                / denominator
            ),
        )

        return ValidationResult(
            passed=not defects,
            score=score,
            defects=defects,
            rule_counts=rule_counts,
            summary=(
                "Structural validation "
                f"found {len(defects)} "
                "defect(s)."
            ),
        )