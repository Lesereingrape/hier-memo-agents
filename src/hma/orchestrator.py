"""The orchestrator: plan → execute DAG waves → synthesize → verify.

Coordination rules that make this more than a prompt chain:
- executors consult the blackboard before researching; a strong memory hit
  skips the model call entirely (and is counted, so you can measure savings)
- every quote the model returns is checked verbatim against the corpus
  before it becomes citable evidence — the model cannot invent sources
- the finished report is re-verified from scratch by `evidence.verify`
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .corpus import Corpus
from .dag import DAG, Node
from .evidence import Claim, ProvenanceStore, Report, Verification, verify
from .llm import LLMClient
from .memory import Blackboard, Finding, _tokens

REUSE_THRESHOLD = 0.45


@dataclass
class RunResult:
    report: Report
    verification: Verification
    memory_stats: dict
    llm_calls: int
    skipped_by_memory: int = 0
    rejected_quotes: list[str] = field(default_factory=list)


class Orchestrator:
    def __init__(self, llm: LLMClient, corpus: Corpus, board: Blackboard | None = None):
        self.llm = llm
        self.corpus = corpus
        self.store = ProvenanceStore(corpus.docs)
        self.board = board or Blackboard()

    def run(self, question: str, top_k: int = 2) -> RunResult:
        plan = self.llm.complete_json({"op": "plan", "question": question})
        dag = DAG([Node(id=s["id"], prompt=s["prompt"], deps=list(s.get("deps", [])))
                   for s in plan["subtasks"]])

        skipped = 0
        rejected: list[str] = []
        for wave in dag.waves():
            for nid in wave:
                if nid == "s0":  # synthesis node runs after the loop, on all findings
                    continue
                node = dag.nodes[nid]
                if self._recall(node.prompt) is not None:
                    skipped += 1
                    continue
                rejected += self._research(node, top_k)

        claims_json = self.llm.complete_json({
            "op": "synthesize", "question": question,
            "findings": [{"key": f.key, "answer": f.answer,
                          "evidence_ids": list(f.evidence_ids)}
                         for f in self.board.all_findings()],
        })
        report = Report(question=question,
                        claims=[Claim(text=c["text"], evidence_ids=list(c["evidence_ids"]))
                                for c in claims_json["claims"]])
        return RunResult(report=report, verification=verify(report, self.store),
                         memory_stats=self.board.stats(), llm_calls=self.llm.n_calls,
                         skipped_by_memory=skipped, rejected_quotes=rejected)

    def _recall(self, prompt: str) -> Finding | None:
        """Memory first: topic-level reuse, so parallel subtasks of one question
        stay independent while a later question skips topics already covered."""
        focus = _tokens(prompt.rpartition("[")[2].rstrip("] "))
        pt = _tokens(prompt)
        for f in self.board.all_findings():
            if not f.evidence_ids:
                continue
            if focus:
                if _jaccard(focus, _tokens(" ".join(f.tags))) >= 0.9:
                    self.board.hits += 1
                    return f
            elif _jaccard(pt, f.tokens) >= REUSE_THRESHOLD:
                self.board.hits += 1
                return f
        return None

    def _research(self, node: Node, top_k: int) -> list[str]:
        """One model call; returns descriptions of quotes that failed verbatim checks."""
        candidates = [{"doc_id": d.doc_id, "quote": s}
                      for _, d, s in self.corpus.search(node.prompt, top_k=top_k)]
        resp = self.llm.complete_json({
            "op": "research", "subtask": {"id": node.id, "prompt": node.prompt},
            "candidates": candidates,
            "deps": [self.board.lookup(d) for d in node.deps],
        })
        evidence_ids: list[str] = []
        rejected: list[str] = []
        for ev in resp.get("evidence", []):
            quote = str(ev.get("quote", ""))
            try:
                evidence_ids.append(self.store.cite(f"{node.id}:{ev['doc_id']}",
                                                    str(ev["doc_id"]), quote).ev_id)
            except (KeyError, ValueError):
                rejected.append(f"{node.id}: non-verbatim quote {quote[:60]!r}")
        self.board.publish(Finding(key=node.id, answer=str(resp.get("answer", "")),
                                   evidence_ids=tuple(evidence_ids),
                                   tags=tuple(resp.get("tags", [])),
                                   prompt=node.prompt))
        return rejected


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0
