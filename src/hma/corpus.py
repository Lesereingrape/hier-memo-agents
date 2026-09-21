"""Tiny built-in corpus with lexical retrieval.

Eight short documents on agent systems; `search` is transparent bag-of-words
scoring so demo runs are deterministic and auditable — you can always see
exactly why a passage was retrieved, which is the point of a research system
built to be verified.
"""

from __future__ import annotations

import re

from .evidence import Document

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


DEMO_DOCS: list[Document] = [
    Document("d1", "harness-overview",
             "An agent harness wraps a model with tools, sandboxes and retries. "
             "The harness, not the model alone, determines whether long tasks finish. "
             "Harness changes move success rates by double digits on the same weights."),
    Document("d2", "memory-in-agents",
             "Shared memory lets parallel agents avoid repeating each other's work. "
             "Episodic blackboards record findings with provenance so results can be reused. "
             "Without deduplication, multi-agent teams multiply their token bill."),
    Document("d3", "planner-executor",
             "Hierarchical agents split work into a planner and several executors. "
             "The planner decomposes a question into a dependency graph of subtasks. "
             "Executors run leaf tasks and publish finished evidence upward."),
    Document("d4", "citation-grounding",
             "Verifiable citations require each claim to link to a verbatim source span. "
             "Post-hoc verification catches fabricated quotes before publication. "
             "Orphan claims with no resolvable evidence should fail validation loudly."),
    Document("d5", "multi-agent-teams",
             "Multi-agent frameworks coordinate roles such as researcher and reviewer. "
             "Communication overhead can erase the gains of specialization. "
             "Blackboard architectures scale better than pairwise agent chat."),
    Document("d6", "evaluation-statistics",
             "Agent benchmark scores need confidence intervals to be meaningful. "
             "Paired comparisons on identical task sets remove task-difficulty noise. "
             "Single-run rankings mostly measure luck, not capability."),
    Document("d7", "tool-use-errors",
             "Tool call failures are the dominant error mode in production agents. "
             "Good harnesses measure recovery rate, not just final accuracy. "
             "Retrying an identical failing call is the most common wasteful loop."),
    Document("d8", "distillation-cost",
             "Distillation transfers behavior from a large model to a compact one. "
             "Dark knowledge in soft labels can beat hard labels at equal size. "
             "Compact distilled models make agent swarms cheap enough to run wide."),
]


class Corpus:
    def __init__(self, docs: list[Document] | None = None):
        self.docs = docs if docs is not None else DEMO_DOCS
        self._tok = {d.doc_id: _tokens(d.title + " " + d.text) for d in self.docs}

    def doc(self, doc_id: str) -> Document:
        return next(d for d in self.docs if d.doc_id == doc_id)

    def search(self, query: str, top_k: int = 3) -> list[tuple[float, Document, str]]:
        """Return (score, doc, best_sentence) for the top_k docs by token overlap."""
        qt = _tokens(query)
        out = []
        for d in self.docs:
            inter = len(qt & self._tok[d.doc_id])
            score = inter / len(qt | self._tok[d.doc_id]) if qt else 0.0
            if score > 0:
                sent = max(_sentences(d.text), key=lambda s: len(qt & _tokens(s)))
                out.append((score, d, sent))
        out.sort(key=lambda x: (-x[0], x[1].doc_id))
        return out[:top_k]


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=\.)\s+", text) if s.strip()]
