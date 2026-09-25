"""Splice the README's two sample-output blocks from what the shipped CLI actually prints.

`hier-memo-agents` advertises a deterministic offline backend, so the sample output in its
README should be a property of the code rather than a transcript of one afternoon. This
script runs `hma run` and `hma session` in-process, captures their stdout, and writes the
result between the RUN / SESSION markers; `tests/test_readme_output_is_real.py` asserts the
README already equals it. Nothing here paraphrases the CLI - the only edit applied to the
captured text is which lines the README quotes.

    python experiments/make_readme.py            # print what the README must contain
    python experiments/make_readme.py --write    # splice it in
"""

from __future__ import annotations

import argparse
import io
import re
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hma.cli import main

QUESTION = "how does shared memory help multi-agent teams and what do citations need?"
SESSION_N = 3


def _capture(argv: list[str]) -> list[str]:
    """Run the CLI in-process and return everything it writes, stdout and stderr together.

    Both streams go into one buffer so the lines come back in the order the CLI printed
    them, which is what a reader sees in a terminal - `hma` deliberately reports its call
    accounting on stderr and its report on stdout.
    """
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        code = main(argv)
    assert code == 0, f"hma {' '.join(argv)} exited {code}"
    text = buf.getvalue().replace("\r\n", "\n").replace("\r", "\n")
    return [ln.rstrip() for ln in text.split("\n") if ln.strip()]


def run_block() -> str:
    """Everything `hma run <question>` prints, which is the whole point of the command."""
    return "\n".join(_capture(["run", QUESTION]))


def session_block() -> str:
    """The call-accounting lines of `hma session --n 3`, in order, plus the board summary.

    The per-question reports are omitted because the block above already shows one; the
    lines kept here are the ones that make the memory-economy claim checkable, printed as
    the CLI prints them - leading indentation included.
    """
    lines = _capture(["session", "--n", str(SESSION_N)])
    stats = [ln for ln in lines if ln.lstrip().startswith("llm_calls=")]
    tail = [ln for ln in lines if ln.lstrip().startswith("blackboard after session")]
    assert len(stats) == SESSION_N, f"session printed {len(stats)} call lines, not {SESSION_N}"
    assert len(tail) == 1, "the session summary line disappeared"
    return "\n".join([ln.lstrip() for ln in stats] + tail)


def blocks() -> dict[str, str]:
    return {"RUN": run_block(), "SESSION": session_block()}


def _write(path: Path, marker: str, block: str) -> None:
    text = path.read_text(encoding="utf-8")
    start, end = f"<!-- {marker}:START -->", f"<!-- {marker}:END -->"
    assert start in text and end in text, f"{path} is missing the {marker} markers"
    head, _, rest = text.partition(start)
    _, _, tail = rest.partition(end)
    assert "```" not in block, "this block contains a code fence; the splice would break"
    nl = "\n"
    path.write_text(f"{head}{start}{nl}```text{nl}{block}{nl}```{nl}{end}{tail}",
                    encoding="utf-8")


def check_readme(path: Path) -> list[str]:
    """Problems with the README's hand-written claims that live outside the blocks."""
    text = path.read_text(encoding="utf-8")
    n_tests = test_count(path.parent / "tests")
    claims = {int(m) for m in re.findall(r"(\d+) tests", text)}
    if not claims:
        return ["the README no longer states how many tests the suite has"]
    if claims != {n_tests}:
        return [f"README says {sorted(claims)} tests, tests/ defines {n_tests}"]
    return []


def test_count(tests_dir: Path) -> int:
    return sum(len(re.findall(r"^def test_", p.read_text(encoding="utf-8"), re.MULTILINE))
               for p in sorted(tests_dir.glob("test_*.py")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="make_readme")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = Path(args.root)
    readme = root / "README.md"
    rendered = blocks()
    if args.write:
        for marker, block in rendered.items():
            _write(readme, marker, block)
        print(f"README {' and '.join(m.lower() for m in rendered)} blocks rewritten")
    else:
        for marker, block in rendered.items():
            print(f"<!-- {marker} -->\n{block}\n")
    for problem in check_readme(readme):
        print(f"warning: {problem}", file=sys.stderr)
