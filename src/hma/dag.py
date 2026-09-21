"""Subtask DAG with a deterministic scheduler.

The planner emits a DAG of sub-questions; the orchestrator executes it in
topological order, wave by wave. Cycles are rejected at parse time — a
hallucinating planner that emits A→B→A must fail loudly, not hang.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Node:
    id: str
    prompt: str
    deps: list[str] = field(default_factory=list)


class CycleError(ValueError):
    pass


class UnknownDepError(ValueError):
    pass


class DAG:
    def __init__(self, nodes: list[Node]):
        self.nodes = {n.id: n for n in nodes}
        if len(self.nodes) != len(nodes):
            raise ValueError("duplicate node ids")
        for n in nodes:
            for d in n.deps:
                if d not in self.nodes:
                    raise UnknownDepError(f"{n.id} depends on unknown node {d!r}")
        self._assert_acyclic()

    def _assert_acyclic(self) -> None:
        state: dict[str, int] = {}  # 0=visiting 1=done

        def visit(nid: str) -> None:
            if state.get(nid) == 0:
                raise CycleError(f"cycle through {nid}")
            if state.get(nid) == 1:
                return
            state[nid] = 0
            for d in self.nodes[nid].deps:
                visit(d)
            state[nid] = 1

        for nid in sorted(self.nodes):
            visit(nid)

    def waves(self) -> list[list[str]]:
        """Topological layers: each wave's nodes depend only on earlier waves.

        Independent subtasks in the same wave can fan out to a thread pool;
        ordering inside every list is lexicographic for reproducibility.
        """
        done: set[str] = set()
        out: list[list[str]] = []
        remaining = set(self.nodes)
        while remaining:
            wave = sorted(n for n in remaining if set(self.nodes[n].deps) <= done)
            if not wave:  # unreachable given _assert_acyclic, kept as a guard
                raise CycleError("unschedulable frontier")
            out.append(wave)
            done |= set(wave)
            remaining -= set(wave)
        return out

    def execution_order(self) -> list[str]:
        return [n for wave in self.waves() for n in wave]
