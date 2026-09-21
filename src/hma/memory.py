"""Shared blackboard memory.

Agents never talk to each other directly; they publish findings to the
blackboard and retrieve from it. That decoupling is what lets executors
skip redundant research: if a peer already published evidence covering a
sub-question, the orchestrator reuses it and logs a memory hit.

Retrieval is a dependency-free TF-overlap scorer — at blackboard scale
(hundreds of entries) it beats nothing on quality and beats every vector
index on simplicity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


@dataclass(frozen=True)
class Finding:
    key: str                 # subtask id that produced it
    answer: str              # short natural-language claim
    evidence_ids: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    prompt: str = ""         # originating subtask prompt (for reuse matching)
    seq: int = 0             # write order, for conflict tie-breaks

    @property
    def tokens(self) -> set[str]:
        return _tokens(self.key + " " + self.answer + " " + self.prompt
                       + " " + " ".join(self.tags))


@dataclass
class Conflict:
    topic: str
    a: Finding
    b: Finding


class Blackboard:
    def __init__(self, dedup_jaccard: float = 0.85, conflict_jaccard: float = 0.60):
        self._entries: dict[str, Finding] = {}
        self._order: list[str] = []
        self._n = 0
        self.dedup_jaccard = dedup_jaccard
        self.conflict_jaccard = conflict_jaccard
        self.hits = 0
        self.misses = 0
        self.conflicts: list[Conflict] = []

    def __len__(self) -> int:
        return len(self._entries)

    def publish(self, f: Finding) -> str:
        """Write-through dedup: return the id of the stored (or pre-existing) finding."""
        for fid, old in self._entries.items():
            if self._similar(f.tokens, old.tokens) >= self.dedup_jaccard and \
               f.evidence_ids and f.evidence_ids == old.evidence_ids:
                return fid  # same claim from the same evidence: drop duplicate
        self._n += 1
        stored = Finding(**{**f.__dict__, "seq": self._n})
        self._entries[stored.key] = stored
        self._order.append(stored.key)
        self._detect_conflicts(stored)
        return stored.key

    def _detect_conflicts(self, new: Finding) -> None:
        topic = new.tags[0] if new.tags else new.key
        for other in self._entries.values():
            if other.key == new.key:
                continue
            other_topic = other.tags[0] if other.tags else other.key
            if other_topic == topic and other.answer.strip().lower() != new.answer.strip().lower():
                self.conflicts.append(Conflict(topic=topic, a=other, b=new))

    def lookup(self, key: str) -> Finding | None:
        return self._entries.get(key)

    def retrieve(self, query: str, top_k: int = 3) -> list[Finding]:
        """Token-overlap retrieval; counts a hit when something passes threshold."""
        qt = _tokens(query)
        scored = [(self._similar(qt, f.tokens), k, f) for k, f in self._entries.items()]
        scored.sort(key=lambda x: (-x[0], x[1]))
        best = [(s, f) for s, _, f in scored if s > 0.15][:top_k]
        if best:
            self.hits += 1
        else:
            self.misses += 1
        return [f for _, f in best]

    @staticmethod
    def _similar(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def all_findings(self) -> list[Finding]:
        return [self._entries[k] for k in self._order]

    def stats(self) -> dict[str, float | int]:
        total = self.hits + self.misses
        return {
            "entries": len(self),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
            "conflicts": len(self.conflicts),
        }
