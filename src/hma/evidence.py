"""Evidence objects and citation provenance verification.

The synthesizer may only emit claims wired to evidence ids; every evidence
id must resolve to a **verbatim span** in the corpus. `verify` re-checks a
finished report from scratch — if a downstream edit breaks the chain, or a
model hallucinated a quote, the report fails validation. No unverifiable
claims by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class Evidence:
    ev_id: str
    doc_id: str
    quote: str  # must be a verbatim substring of the document text


class ProvenanceStore:
    def __init__(self, docs: list[Document]):
        self.docs = {d.doc_id: d for d in docs}
        self.evidence: dict[str, Evidence] = {}
        for d in self.docs.values():
            self.evidence[d.doc_id] = Evidence(d.doc_id, d.doc_id, d.text)

    def cite(self, ev_id: str, doc_id: str, quote: str) -> Evidence:
        doc = self.docs.get(doc_id)
        if doc is None:
            raise KeyError(f"unknown document {doc_id!r}")
        if quote not in doc.text:
            raise ValueError(f"quote is not a verbatim span of {doc_id}: {quote[:60]!r}")
        ev = Evidence(ev_id, doc_id, quote)
        self.evidence[ev_id] = ev
        return ev

    def get(self, ev_id: str) -> Evidence | None:
        return self.evidence.get(ev_id)


@dataclass
class Claim:
    text: str
    evidence_ids: list[str] = field(default_factory=list)


@dataclass
class Report:
    question: str
    claims: list[Claim]
    used_subtask_ids: list[str] = field(default_factory=list)


@dataclass
class Verification:
    ok: bool
    unknown_evidence: list[str]
    broken_spans: list[str]   # evidence id whose quote no longer verbatim
    orphan_claims: list[str]  # claim text with no resolvable evidence

    def summary(self) -> str:
        if self.ok:
            return "VERIFIED: every claim resolves to a verbatim corpus span"
        parts = []
        if self.unknown_evidence:
            parts.append(f"unknown evidence ids: {self.unknown_evidence}")
        if self.broken_spans:
            parts.append(f"broken spans: {self.broken_spans}")
        if self.orphan_claims:
            parts.append(f"unsupported claims: {self.orphan_claims}")
        return "FAILED: " + "; ".join(parts)


def verify(report: Report, store: ProvenanceStore) -> Verification:
    unknown: list[str] = []
    broken: list[str] = []
    orphans: list[str] = []
    for claim in report.claims:
        resolved = 0
        for ev_id in claim.evidence_ids:
            ev = store.get(ev_id)
            if ev is None:
                if ev_id not in unknown:
                    unknown.append(ev_id)
                continue
            doc = store.docs.get(ev.doc_id)
            if doc is None or ev.quote not in doc.text:
                broken.append(ev_id)
            else:
                resolved += 1
        if resolved == 0:
            orphans.append(claim.text)
    return Verification(ok=not (unknown or broken or orphans),
                        unknown_evidence=unknown, broken_spans=broken,
                        orphan_claims=orphans)
