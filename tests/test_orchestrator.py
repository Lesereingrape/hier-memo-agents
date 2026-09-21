from hma.corpus import Corpus
from hma.corpus_llm import CorpusLLM
from hma.llm import ScriptedLLM
from hma.memory import Blackboard
from hma.orchestrator import REUSE_THRESHOLD, Orchestrator

Q = "how does shared memory help multi agent teams and what do citations require"


def _plan(n_subtasks=3):
    return {"subtasks": [
        {"id": f"r{i}", "prompt": f"shared memory for multi agent teams topic{i}", "deps": []}
        for i in range(n_subtasks)
    ] + [{"id": "s0", "prompt": "synthesize",
          "deps": [f"r{i}" for i in range(n_subtasks)]}]}


def test_end_to_end_produces_verified_cited_report():
    orch = Orchestrator(CorpusLLM(Corpus()), Corpus())
    res = orch.run(Q)
    assert res.verification.ok, res.verification.summary()
    assert res.report.claims
    assert all(c.evidence_ids for c in res.report.claims)
    assert not res.rejected_quotes


def test_second_question_reuses_blackboard():
    corpus = Corpus()
    board = Blackboard()
    orch = Orchestrator(CorpusLLM(corpus), corpus, board=board)
    first = orch.run(Q)
    calls_after_first = first.llm_calls
    second = orch.run(Q)  # same question again: research should be skipped
    assert second.skipped_by_memory > 0
    research_calls = second.llm_calls - calls_after_first - 2  # plan + synthesize
    assert research_calls < first.llm_calls  # strictly cheaper second run


def test_hallucinated_quote_is_rejected_and_report_still_verified():
    scripted = ScriptedLLM([
        _plan(1),
        {"answer": "fabricated fact",
         "evidence": [{"doc_id": "d1", "quote": "this quote exists nowhere in the corpus"}],
         "tags": ["harness-overview"]},
        {"claims": []},
    ])
    orch = Orchestrator(scripted, Corpus())
    res = orch.run(Q)
    assert res.rejected_quotes, "fake quote must be flagged"
    assert "d1" not in [e for f in orch.board.all_findings() for e in f.evidence_ids]


def test_reuse_threshold_boundary_behavior():
    corpus = Corpus()
    orch = Orchestrator(CorpusLLM(corpus), corpus)
    assert 0.0 < REUSE_THRESHOLD < 1.0
    res = orch.run("tool call failures recovery rate agents")
    assert res.verification.ok
