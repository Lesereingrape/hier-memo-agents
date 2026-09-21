import pytest

from hma.dag import DAG, CycleError, Node, UnknownDepError


def _dag(*nodes: Node) -> DAG:
    return DAG(list(nodes))


def test_waves_respect_dependencies():
    dag = _dag(
        Node("a", "pa"),
        Node("b", "pb"),
        Node("c", "pc", deps=["a", "b"]),
    )
    assert dag.waves() == [["a", "b"], ["c"]]
    assert dag.execution_order() == ["a", "b", "c"]


def test_cycle_detected():
    with pytest.raises(CycleError):
        _dag(Node("a", "x", deps=["b"]), Node("b", "y", deps=["a"]))


def test_self_cycle_detected():
    with pytest.raises(CycleError):
        _dag(Node("a", "x", deps=["a"]))


def test_unknown_dep_rejected():
    with pytest.raises(UnknownDepError):
        _dag(Node("a", "x", deps=["ghost"]))


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError):
        _dag(Node("a", "x"), Node("a", "y"))


def test_independent_nodes_sorted_lexicographically():
    dag = _dag(Node("z", ""), Node("a", ""), Node("m", ""))
    assert dag.execution_order() == ["a", "m", "z"]
