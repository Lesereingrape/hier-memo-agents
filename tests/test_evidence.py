import pytest

from hma.evidence import Claim, Document, ProvenanceStore, Report, verify

DOC = Document("d1", "t", "Alpha beta gamma. Delta epsilon zeta.")


def test_cite_requires_verbatim_span():
    store = ProvenanceStore([DOC])
    ev = store.cite("e1", "d1", "Alpha beta gamma.")
    assert ev.doc_id == "d1"
    with pytest.raises(ValueError):
        store.cite("e2", "d1", "Alpha delta gamma.")  # stitched quote
    with pytest.raises(KeyError):
        store.cite("e3", "nope", "Alpha beta gamma.")


def test_verify_passes_for_grounded_report():
    store = ProvenanceStore([DOC])
    store.cite("e1", "d1", "Alpha beta gamma.")
    report = Report("q", [Claim("first", ["e1"]), Claim("second", ["d1"])])
    v = verify(report, store)
    assert v.ok, v.summary()


def test_verify_flags_orphan_and_unknown_evidence():
    store = ProvenanceStore([DOC])
    store.cite("e1", "d1", "Alpha beta gamma.")
    report = Report("q", [Claim("real", ["e1"]), Claim("ghost", ["nope"])])
    v = verify(report, store)
    assert not v.ok
    assert v.orphan_claims == ["ghost"]
    assert v.unknown_evidence == ["nope"]


def test_verify_flags_broken_span_after_corpus_edit():
    """Simulates evidence drifting from source: quote must stay verbatim."""
    store = ProvenanceStore([DOC])
    store.cite("e1", "d1", "Alpha beta gamma.")
    report = Report("q", [Claim("x", ["e1"])])
    assert verify(report, store).ok
    store.docs["d1"] = Document("d1", "t", "Totally different text now.")
    v = verify(report, store)
    assert not v.ok
    assert v.broken_spans == ["e1"]
