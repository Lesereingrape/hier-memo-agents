from hma.memory import Blackboard, Finding


def _f(key, answer, ev=(), tags=()):
    return Finding(key=key, answer=answer, evidence_ids=tuple(ev), tags=tuple(tags))


def test_publish_and_lookup():
    bb = Blackboard()
    bb.publish(_f("r1", "shared memory avoids duplicate work", ["e1"], ["memory"]))
    assert bb.lookup("r1") is not None
    assert len(bb) == 1


def test_identical_claim_from_same_evidence_is_deduplicated():
    bb = Blackboard()
    id1 = bb.publish(_f("r1", "harnesses decide task success", ["e1"], ["harness"]))
    id2 = bb.publish(_f("r1", "harnesses decide task success", ["e1"], ["harness"]))
    assert id1 == id2 == "r1"
    assert len(bb) == 1


def test_retrieve_counts_hits_and_misses():
    bb = Blackboard()
    bb.publish(_f("r1", "planners decompose questions into subtask graphs",
                  ["e1"], ["plan"]))
    got = bb.retrieve("how do planners decompose subtask graphs")
    assert got and got[0].key == "r1"
    assert bb.retrieve("completely unrelated quantum pastry topic") == []
    s = bb.stats()
    assert s["hits"] == 1 and s["misses"] == 1


def test_conflicting_findings_on_same_topic_are_recorded():
    bb = Blackboard()
    bb.publish(_f("r1", "distillation beats retrieval", ["e1"], ["method"]))
    bb.publish(_f("r2", "retrieval beats distillation", ["e2"], ["method"]))
    assert len(bb.conflicts) == 1
    c = bb.conflicts[0]
    assert c.topic == "method" and {c.a.key, c.b.key} == {"r1", "r2"}


def test_no_conflict_for_identical_answers():
    bb = Blackboard()
    bb.publish(_f("r1", "same claim text", ["e1"], ["t"]))
    bb.publish(_f("r2", "same claim text", ["e2"], ["t"]))
    assert bb.conflicts == []
