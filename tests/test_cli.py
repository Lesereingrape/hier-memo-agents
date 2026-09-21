from hma.cli import main


def test_cli_run(capsys):
    assert main(["run", "what do citations require and how does shared memory help"]) == 0
    out = capsys.readouterr()
    assert "VERIFIED" in out.out
    assert "llm_calls" in out.err


def test_cli_session(capsys):
    assert main(["session", "--n", "2"]) == 0
    out = capsys.readouterr().out
    assert out.count("===") == 2
    assert "blackboard after session" in out
