from __future__ import annotations

import sys
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

    ``links`` marks nodes that connect a part of a split diagram to another
    part: "in" actions are drawn as an incoming signal ("From Part 2"),
    "out" final nodes as an outgoing signal ("Continue in Part 4") where the
    flow leaves this part.
    """

    def __init__(self, links: dict[str, str] | None = None) -> None:
        self.links = links or {}

    def render(
        self,
        diagram: ActivityDiagram,
        structured_loops: bool = True,
    ) -> str:
        """Compile the IR to PlantUML.

        ``structured_loops=False`` draws every loop as plain branching with
        connector jumps back to its head instead of repeat/while blocks; used
        when PlantUML cannot lay out the structured version.
        """

        self._validate(diagram)

        # The compiler walks the graph recursively; diagrams composed from a
        # long specification can have hundreds of nodes.
        sys.setrecursionlimit(max(sys.getrecursionlimit(), 10_000))

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

        self._analyze(graph, nodes, initial.id)

        if structured_loops:
            text = self._emit(graph, nodes, diagram, initial)
            if self._blocks_balanced(text):
                return text

        # Loop back edges that leave from inside a nested branch cannot be
        # expressed with repeat/while blocks (and some structured loops crash
        # PlantUML's layout, see structured_loops). Fall back to plain branching
        # (back edges become connector jumps) rather than emitting invalid
        # PlantUML.
        self._repeat_headers = set()
        self._while_headers = set()
        return self._emit(graph, nodes, diagram, initial)

    def _emit(
        self,
        graph: nx.DiGraph,
        nodes: dict[str, ActivityNode],
        diagram: ActivityDiagram,
        initial: ActivityNode,
    ) -> str:

        lines: list[str] = []

        lines.append("@startuml")
        lines.append(
            f"title {self._escape(diagram.title)}"
        )
        lines.append("")

        visited: set[str] = set()
        self._final_reached = False
        self._current_lane: str | None = None
        self._lane_names: dict[str, str] = {}
        # Edges to a node drawn elsewhere are drawn as connector jumps:
        # "(A)" + "detach" at the source and "(A)" just before the target.
        self._connectors: dict[str, str] = {}
        # Nodes control reaches by falling out of the enclosing blocks (merge
        # points of open IF/fork blocks, head of the current while loop).
        self._fallthrough: list[str] = []

        # PlantUML requires the first swimlane to be declared before "start".
        first_lane = self._first_lane(graph, nodes, initial.id)
        if first_lane:
            self._emit_lane(first_lane, lines)

        lines.append("start")
        lines.append("")

        self._compile_node(
            graph=graph,
            nodes=nodes,
            diagram=diagram,
            node_id=initial.id,
            visited=visited,
            path=set(),
            lines=lines,
        )

        if not self._final_reached:
            lines.append("")
            lines.append("stop")

        lines.append("")
        lines.append("@enduml")

        # Node markers become the target side of connector jumps.
        resolved = []
        for line in lines:
            if line.startswith(self._NODE_MARK):
                target = line[len(self._NODE_MARK):]
                if target in self._connectors:
                    resolved.append(f"({self._connectors[target]})")
                continue
            resolved.append(line)

        return "\n".join(resolved)

    _NODE_MARK = "\x00node:"

    def _emit_final(self, node: ActivityNode, lines: list[str]) -> None:
        self._final_reached = True
        if self.links.get(node.id) == "out":
            self._emit_lane(node.lane, lines)
            lines.append(f":{self._escape(node.label)}>")
            lines.append("detach")
        else:
            lines.append("stop")

    def _jump(
        self,
        target: str,
        nodes: dict[str, ActivityNode],
        lines: list[str],
    ) -> None:
        """Continue at ``target``, which is drawn somewhere else."""
        # Leaving the innermost open block reaches its merge point anyway.
        if self._fallthrough and target == self._fallthrough[-1]:
            return
        if nodes[target].type == NodeType.FINAL:
            self._emit_final(nodes[target], lines)
            return
        if target not in self._connectors:
            index = len(self._connectors)
            self._connectors[target] = chr(ord("A") + index % 26) + (str(index // 26) if index >= 26 else "")
        lines.append(f"({self._connectors[target]})")
        lines.append("detach")

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

    def _analyze(
        self,
        graph: nx.DiGraph,
        nodes: dict[str, ActivityNode],
        initial_id: str,
    ) -> None:
        """Find loops and merge points.

        A back edge u -> v is one whose target dominates its source; v is
        then a loop header. A decision header becomes a ``while`` loop, any
        other header a ``repeat`` loop closed by its back edges. Merge points
        are computed on the forward graph (no back edges), see _merge_for.
        """
        self._nodes = nodes
        reachable = nx.descendants(graph, initial_id) | {initial_id}
        idom = nx.immediate_dominators(graph.subgraph(reachable), initial_id)

        def dominates(a: str, b: str) -> bool:
            while True:
                if a == b:
                    return True
                parent = idom.get(b)
                if parent is None or parent == b:
                    return False
                b = parent

        self._back_edges = {
            (u, v)
            for u, v in graph.edges
            if u in reachable and dominates(v, u)
        }

        self._loop_nodes: dict[str, set[str]] = defaultdict(set)
        for source, header in self._back_edges:
            body = self._loop_nodes[header]
            body.add(header)
            stack = [source]
            while stack:
                current = stack.pop()
                if current in body:
                    continue
                body.add(current)
                stack.extend(graph.predecessors(current))

        self._while_headers = {
            header
            for header in self._loop_nodes
            if nodes[header].type == NodeType.DECISION
        }
        self._repeat_headers = set(self._loop_nodes) - self._while_headers

        # Forward graph: no back edges, and no cycles left over from
        # irreducible flow (cycles entered at more than one node).
        forward = nx.DiGraph()
        forward.add_nodes_from(graph.nodes)
        forward.add_edges_from(
            edge for edge in graph.edges if edge not in self._back_edges
        )
        while True:
            try:
                cycle = nx.find_cycle(forward)
            except nx.NetworkXNoCycle:
                break
            forward.remove_edge(*cycle[-1][:2])

        self._forward = forward
        self._order = {
            node_id: index
            for index, node_id in enumerate(nx.topological_sort(forward))
        }
        self._reach_cache: dict[str, set[str]] = {}

    def _reach(self, node_id: str) -> set[str]:
        if node_id not in self._reach_cache:
            self._reach_cache[node_id] = nx.descendants(self._forward, node_id) | {node_id}
        return self._reach_cache[node_id]

    def _terminates(self, node_id: str) -> bool:
        """True when every forward path from the node ends in a final node."""
        return all(
            self._nodes[n].type == NodeType.FINAL
            for n in self._reach(node_id)
            if self._forward.out_degree(n) == 0
        )

    def _merge_for(self, targets: list[str]) -> str | None:
        """Node where the branches towards ``targets`` rejoin.

        Branches that end in a final node of their own (``stop``) are
        ignored, so an early exit does not push the merge point to the end of
        the diagram. When the branches never rejoin and all but the longest
        one stop, the longest one continues after the IF block (same
        behaviour, less nesting).
        """
        reach = [self._reach(target) for target in targets]
        shared = [
            i
            for i in range(len(reach))
            if any(reach[i] & reach[j] for j in range(len(reach)) if j != i)
        ]
        if shared:
            common = set.intersection(*(reach[i] for i in shared))
            if not common:
                common = {
                    n
                    for i in shared
                    for j in shared
                    if i < j
                    for n in reach[i] & reach[j]
                }
            return min(common, key=self._order.__getitem__)

        if len(targets) < 2:
            return None
        sizes = sorted(((len(r), t) for r, t in zip(reach, targets)), reverse=True)
        if sizes[0][0] == sizes[1][0]:
            return None
        longest = sizes[0][1]
        if all(self._terminates(t) for t in targets if t != longest):
            return longest
        return None

    def _repeat_close(
        self,
        graph: nx.DiGraph,
        source: str,
        target: str,
    ) -> list[str] | None:
        """Actions on a branch that leads straight back to a repeat header.

        Returns the (possibly empty) chain of single-step actions between
        the decision and the back edge, or None if the branch is no such
        loop-back.
        """
        chain: list[str] = []
        previous, current = source, target
        while True:
            if current in self._repeat_headers and (previous, current) in self._back_edges:
                return chain
            if (
                self._nodes[current].type != NodeType.ACTION
                or graph.out_degree(current) != 1
                or graph.in_degree(current) != 1
                or len(chain) == 3
            ):
                return None
            chain.append(current)
            previous, current = current, next(iter(graph.successors(current)))

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
        path: set[str],
        lines: list[str],
    ) -> None:

        if node_id in path:
            return
        if node_id in visited:
            return

        next_path = set(path)
        next_path.add(node_id)
        visited.add(node_id)

        node = nodes[node_id]
        lines.append(self._NODE_MARK + node_id)
        if node_id in self._repeat_headers:
            lines.append("repeat")
        self._emit_lane(node.lane, lines)

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
                path=next_path,
                lines=lines,
            )

            return

        # --------------------------------------------------------
        # FINAL
        # --------------------------------------------------------

        if node.type == NodeType.FINAL:
            self._emit_final(node, lines)
            return

        # --------------------------------------------------------
        # ACTION
        # --------------------------------------------------------

        if node.type == NodeType.ACTION:

            end = "<" if self.links.get(node_id) == "in" else ";"
            lines.append(f":{self._escape(node.label)}{end}")

            self._compile_successors(
                graph=graph,
                nodes=nodes,
                diagram=diagram,
                node_id=node_id,
                visited=visited,
                path=next_path,
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
                path=next_path,
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
                path=next_path,
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
                path=next_path,
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
                path=next_path,
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
                path=next_path,
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
                path=next_path,
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
        path: set[str],
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

            # Unconditional jump back to the start of a repeat loop.
            if target in self._repeat_headers and (node_id, target) in self._back_edges:
                lines.append("repeat while (continue)")
                return

            # Don't recursively traverse cycles.
            if target in visited or target in path:
                self._jump(target, nodes, lines)
                return

            self._compile_node(
                graph=graph,
                nodes=nodes,
                diagram=diagram,
                node_id=target,
                visited=visited,
                path=path,
                lines=lines,
            )

            return

        # --------------------------------------------------------
        # More than one successor from a normal node.
        #
        # This situation can happen when a generator emits a branch without a
        # dedicated decision node. Preserve the graph by compiling it as a
        # synthetic decision instead of crashing and producing an empty output.
        # --------------------------------------------------------

        synthetic_node = nodes[node_id].model_copy(deep=True)
        synthetic_node.type = NodeType.DECISION
        if not synthetic_node.label:
            synthetic_node.label = "Decision"

        self._compile_decision(
            graph=graph,
            nodes=nodes,
            diagram=diagram,
            node=synthetic_node,
            visited=visited,
            path=path,
            lines=lines,
        )

        return

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
        path: set[str],
        lines: list[str],
        outgoing: list[ActivityEdge] | None = None,
    ) -> None:

        # ``outgoing`` is given when compiling a subset of the branches as a
        # plain IF block (e.g. several exits of a loop).
        handle_loops = outgoing is None
        if outgoing is None:
            outgoing = [
                edge
                for edge in diagram.edges
                if edge.source == node.id
            ]

        label = self._escape(node.label)
        common = dict(
            graph=graph,
            nodes=nodes,
            diagram=diagram,
            visited=visited,
            path=path,
            lines=lines,
        )

        # --------------------------------------------------------
        # End of a repeat loop: a branch jumps back to the header.
        # --------------------------------------------------------

        # Actions between the decision and the back edge are drawn on the
        # backward path of the loop.
        backward: list[str] = []
        repeat_edges = []
        for edge in outgoing:
            chain = self._repeat_close(graph, node.id, edge.target)
            if chain is not None and not any(n in visited for n in chain):
                repeat_edges.append(edge)
                if not backward:
                    backward = chain

        if handle_loops and repeat_edges:

            exits = [edge for edge in outgoing if edge not in repeat_edges]
            if backward:
                visited.update(backward)
                text = " / ".join(self._escape(self._nodes[n].label) for n in backward)
                lines.append(f"backward:{text};")
            condition = (
                f"repeat while ({label}) "
                f"is ({self._escape_guard(self._guard(repeat_edges[0].guard, 'yes'))})"
            )

            if len(exits) == 1:
                lines.append(
                    f"{condition} "
                    f"not ({self._escape_guard(self._guard(exits[0].guard, 'no'))})"
                )
                self._compile_branch(target=exits[0].target, **common)
            else:
                lines.append(condition)
                if exits:
                    self._compile_decision(node=node, outgoing=exits, **common)

            return

        # --------------------------------------------------------
        # Head of a while loop.
        # --------------------------------------------------------

        if handle_loops and node.id in self._while_headers:

            loop = self._loop_nodes[node.id]
            body = [edge for edge in outgoing if edge.target in loop]
            exits = [edge for edge in outgoing if edge.target not in loop]

            lines.append(
                f"while ({label}) "
                f"is ({self._escape_guard(self._guard(body[0].guard, 'yes'))})"
            )

            # The loop body must not swallow the nodes after the loop.
            reserved = {edge.target for edge in exits} - visited
            visited.update(reserved)

            body = [edge for edge in body if edge.target != node.id]
            self._fallthrough.append(node.id)
            if len(body) == 1:
                self._compile_branch(target=body[0].target, **common)
            elif body:
                self._compile_decision(node=node, outgoing=body, **common)
            self._fallthrough.pop()

            visited.difference_update(reserved)

            if len(exits) == 1:
                lines.append(
                    f"endwhile ({self._escape_guard(self._guard(exits[0].guard, 'no'))})"
                )
                self._compile_branch(target=exits[0].target, **common)
            else:
                lines.append("endwhile")
                if exits:
                    self._compile_decision(node=node, outgoing=exits, **common)

            return

        # --------------------------------------------------------
        # Normal IF / ELSEIF / ELSE
        # --------------------------------------------------------

        if not outgoing:
            # Dead-end decision (flagged by the structural validator).
            lines.append(f":{label};")
            return

        # Branches stop at the merge point; it is emitted once after endif.
        # A merge point that is already visited belongs to an enclosing
        # block (control falls through to it) or was drawn elsewhere.
        merge_point = self._merge_for([edge.target for edge in outgoing])
        # A shared final node is not a merge point: each branch stops on its
        # own, so no "stop" follows the block.
        if merge_point in path or (
            merge_point is not None and nodes[merge_point].type == NodeType.FINAL
        ):
            merge_point = None
        owns_merge = merge_point is not None and merge_point not in visited
        if owns_merge:
            self._fallthrough.append(merge_point)
            visited.add(merge_point)

        # Branches share ``visited``: a node already drawn by an earlier
        # branch is reached by a connector jump instead of being drawn twice.
        for i, edge in enumerate(outgoing):

            is_first = (i == 0)
            is_last = (i == len(outgoing) - 1)

            guard = self._guard(
                edge.guard,
                "yes" if is_first else "otherwise" if is_last else "no"
            )

            if is_first:
                lines.append(
                    f"if ({label}) "
                    f"then ({self._escape_guard(guard)})"
                )
            elif is_last:
                lines.append(
                    f"else ({self._escape_guard(guard)})"
                )
            else:
                lines.append(
                    f"elseif ({label}) "
                    f"then ({self._escape_guard(guard)})"
                )

            self._compile_branch(
                graph=graph,
                nodes=nodes,
                diagram=diagram,
                target=edge.target,
                visited=visited,
                path=path.copy(),
                lines=lines,
            )

        lines.append("endif")

        if merge_point is not None and not owns_merge:
            self._jump(merge_point, nodes, lines)

        if owns_merge:
            self._fallthrough.pop()
            visited.discard(merge_point)
            self._compile_node(
                graph=graph,
                nodes=nodes,
                diagram=diagram,
                node_id=merge_point,
                visited=visited,
                path=path,
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
        path: set[str],
        lines: list[str],
    ) -> None:

        if target not in nodes:
            raise ValueError(
                f"Unknown branch target {target}"
            )

        if target in path or target in visited:
            self._jump(target, nodes, lines)
            return

        self._compile_node(
            graph=graph,
            nodes=nodes,
            diagram=diagram,
            node_id=target,
            visited=visited,
            path=path,
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
        path: set[str],
        lines: list[str],
    ) -> None:

        branches = list(
            graph.successors(node.id)
        )

        if len(branches) < 2:
            # Degenerate fork (flagged by the structural validator).
            self._compile_successors(
                graph=graph,
                nodes=nodes,
                diagram=diagram,
                node_id=node.id,
                visited=visited,
                path=path,
                lines=lines,
            )
            return

        # Branches stop at the join; it is emitted once after "end fork".
        merge_point = self._merge_for(branches)
        # A shared final node is not a merge point: each branch stops on its
        # own, so no "stop" follows the block.
        if merge_point in path or (
            merge_point is not None and nodes[merge_point].type == NodeType.FINAL
        ):
            merge_point = None
        owns_merge = merge_point is not None and merge_point not in visited
        if owns_merge:
            self._fallthrough.append(merge_point)
            visited.add(merge_point)

        lines.append("fork")

        for index, target in enumerate(branches):

            if index > 0:
                lines.append(
                    "fork again"
                )

            self._compile_branch(
                graph=graph,
                nodes=nodes,
                diagram=diagram,
                target=target,
                visited=visited,
                path=path.copy(),
                lines=lines,
            )

        lines.append("end fork")

        if merge_point is not None and not owns_merge:
            self._jump(merge_point, nodes, lines)

        if owns_merge:
            self._fallthrough.pop()
            visited.discard(merge_point)
            self._compile_node(
                graph=graph,
                nodes=nodes,
                diagram=diagram,
                node_id=merge_point,
                visited=visited,
                path=path,
                lines=lines,
            )

    # ============================================================
    # GRAPH UTILITIES
    # ============================================================

    @staticmethod
    def _blocks_balanced(text: str) -> bool:
        """Check that if/repeat/while/fork blocks open and close in order."""
        stack: list[str] = []
        for raw in text.splitlines():
            line = raw.strip()
            if line.startswith(("elseif (", "else (")) or line == "else":
                if not stack or stack[-1] != "if":
                    return False
            elif line.startswith("if ("):
                stack.append("if")
            elif line == "endif":
                if not stack or stack.pop() != "if":
                    return False
            elif line.startswith("repeat while"):
                if not stack or stack.pop() != "repeat":
                    return False
            elif line == "repeat":
                stack.append("repeat")
            elif line.startswith("while ("):
                stack.append("while")
            elif line.startswith("endwhile"):
                if not stack or stack.pop() != "while":
                    return False
            elif line == "fork":
                stack.append("fork")
            elif line == "fork again":
                if not stack or stack[-1] != "fork":
                    return False
            elif line == "end fork":
                if not stack or stack.pop() != "fork":
                    return False
        return not stack

    def _emit_lane(
        self,
        lane: str | None,
        lines: list[str],
    ) -> None:
        lane = self._lane_name(lane)
        if not lane or lane == self._current_lane:
            return
        lines.append(f"|{self._escape_lane(lane)}|")
        self._current_lane = lane

    def _lane_name(self, lane: str | None) -> str:
        """Spellings that differ only in case or spacing share one lane."""
        lane = " ".join((lane or "").split())
        if not lane:
            return ""
        return self._lane_names.setdefault(lane.casefold(), lane)

    @staticmethod
    def _first_lane(
        graph: nx.DiGraph,
        nodes: dict[str, ActivityNode],
        initial_id: str,
    ) -> str | None:
        for node_id in nx.bfs_tree(graph, initial_id):
            lane = (nodes[node_id].lane or "").strip()
            if lane:
                return lane
        return None

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
    def _escape_lane(text: str) -> str:
        return (
            text.replace("|", " ")
            .replace("\n", " ")
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