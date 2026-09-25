"""Guard: the README's sample outputs must be what the CLI prints, byte for byte.

`hma run` and `hma session` are deterministic offline commands, so there is no excuse for a
transcript in the docs that the code cannot reproduce - and there was one: the previous
README quoted a claim citing `[r2:d4]` that no run of this repository produces. The blocks
between the RUN / SESSION markers are now spliced from live stdout by
`experiments/make_readme.py`, and this test re-runs both commands in-process and compares.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import make_readme  # noqa: E402

MARKER = "<!-- {m}:START -->\n```text\n(.*?)\n```\n<!-- {m}:END -->"


def _block(marker: str) -> str:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    match = re.search(MARKER.format(m=marker), text, re.DOTALL)
    assert match, f"README is missing a fenced {marker} block"
    return match.group(1)


def test_run_block_is_the_cli_output():
    assert _block("RUN") == make_readme.run_block(), (
        "README run block is stale; run `python experiments/make_readme.py --write`")


def test_session_block_is_the_cli_accounting():
    assert _block("SESSION") == make_readme.session_block(), (
        "README session block is stale; run `python experiments/make_readme.py --write`")


def _stats(block: str) -> list[dict[str, str]]:
    """Parse `llm_calls=17 skipped_by_memory=1 memory={...}` into its counter fields."""
    rows = []
    for line in block.splitlines():
        if not line.startswith("llm_calls="):
            continue
        counters, board = line.split(" memory=", 1)
        rows.append({**dict(p.split("=", 1) for p in counters.split()),
                     "memory": ast.literal_eval(board)})
    return rows


def test_the_quoted_reuse_holds_in_a_live_run():
    """The prose reads the third session line as 'one topic came back from the blackboard'."""
    rows = _stats(_block("SESSION"))
    assert len(rows) == make_readme.SESSION_N, "the block quotes one line per question"
    assert all(int(r["llm_calls"]) > 0 for r in rows)
    assert int(rows[-1]["skipped_by_memory"]) > int(rows[0]["skipped_by_memory"]), (
        "README sells the session as showing memory reuse; the call accounting says it does not")
    assert rows[-1]["memory"]["hits"] > rows[0]["memory"]["hits"], (
        "the skip is quoted as a blackboard hit; the telemetry disagrees")


def test_the_memory_stats_the_block_prints_are_the_boards_own():
    """The trailing dict on each stats line is real telemetry, not decoration."""
    for row in _stats(_block("SESSION")):
        stats = row["memory"]
        assert set(stats) >= {"entries", "hits", "misses", "hit_rate", "conflicts"}
        assert 0.0 <= stats["hit_rate"] <= 1.0
        assert stats["hits"] + stats["misses"] <= stats["entries"]


def test_the_advertised_test_count_is_the_suite_that_exists():
    problems = make_readme.check_readme(ROOT / "README.md")
    assert not problems, "; ".join(problems)
