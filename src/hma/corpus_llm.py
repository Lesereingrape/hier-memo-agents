"""Deterministic LLM backend for demos and tests.

`CorpusLLM` answers the same JSON contracts as a hosted model, but its
"reasoning" is transparent retrieval: plan fans out over the best-matching
documents, research quotes the best-matching sentence, synthesize wires
claims to the evidence ids they came from. This keeps the whole pipeline —
including provenance verification — exercisable in CI with zero network.
"""

from __future__ import annotations

from .corpus import Corpus


class CorpusLLM:
    def __init__(self, corpus: Corpus, max_subtasks: int = 4):
        self.corpus = corpus
        self.max_subtasks = max_subtasks
        self.n_calls = 0

    def complete_json(self, request: dict) -> dict:
        self.n_calls += 1
        op = request.get("op")
        if op == "plan":
            return self._plan(request["question"])
        if op == "research":
            return self._research(request)
        if op == "synthesize":
            return self._synthesize(request)
        raise ValueError(f"CorpusLLM got unsupported op {op!r}")

    def _plan(self, question: str) -> dict:
        hits = self.corpus.search(question, top_k=self.max_subtasks)
        subtasks = [{"id": f"r{i + 1}", "prompt": f"{question} [{d.title}]",
                     "deps": []} for i, (_, d, _) in enumerate(hits)]
        if not subtasks:  # nothing matched lexically: one broad subtask
            subtasks = [{"id": "r1", "prompt": question, "deps": []}]
        subtasks.append({"id": "s0", "prompt": f"synthesize: {question}",
                         "deps": [s["id"] for s in subtasks]})
        return {"subtasks": subtasks}

    def _research(self, request: dict) -> dict:
        candidates = request.get("candidates", [])
        if not candidates:
            return {"answer": "no evidence found", "evidence": [], "tags": []}
        best = candidates[0]
        return {"answer": best["quote"],
                "evidence": [{"doc_id": best["doc_id"], "quote": best["quote"]}],
                "tags": [self.corpus.doc(best["doc_id"]).title]}

    def _synthesize(self, request: dict) -> dict:
        findings = request.get("findings", [])
        claims: list[dict] = []
        seen: set[str] = set()
        for f in findings:
            if not f.get("evidence_ids") or f["answer"] in seen:
                continue
            seen.add(f["answer"])
            claims.append({"text": f["answer"], "evidence_ids": list(f["evidence_ids"])})
        all_ev = sorted({e for c in claims for e in c["evidence_ids"]})
        if claims:
            claims.append({"text": f"Synthesis of {len(claims)} findings over {len(all_ev)} sources.",
                           "evidence_ids": all_ev})
        return {"claims": claims}


__all__ = ["CorpusLLM"]
