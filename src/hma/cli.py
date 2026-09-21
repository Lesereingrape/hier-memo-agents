"""`hma` command line: one-shot research run and multi-question session."""

from __future__ import annotations

import argparse
import sys

DEMO_QUESTIONS = [
    "how does shared memory help multi-agent teams and what do citations need?",
    "what makes an agent harness succeed on long tasks and how do tool errors behave?",
    "can distillation and shared memory make agent teams cheaper?",
]


def _build(model: str | None):
    from .corpus import Corpus
    from .corpus_llm import CorpusLLM
    if model and model != "corpus":
        from .llm import OpenAICompatLLM
        return OpenAICompatLLM(model=model), Corpus()
    return CorpusLLM(Corpus()), Corpus()


def cmd_run(args: argparse.Namespace) -> int:
    from .orchestrator import Orchestrator

    llm, corpus = _build(args.model)
    orch = Orchestrator(llm, corpus)
    res = orch.run(args.question)
    _print_result(args.question, res)
    return 0 if res.verification.ok else 1


def cmd_session(args: argparse.Namespace) -> int:
    """Related questions on one persistent board: measures cross-task memory reuse."""
    from .orchestrator import Orchestrator

    llm, corpus = _build(args.model)
    orch = Orchestrator(llm, corpus)
    for q in DEMO_QUESTIONS[: args.n]:
        res = orch.run(q)
        _print_result(q, res)
    s = orch.board.stats()
    print(f"\nblackboard after session: {s['entries']} findings, "
          f"hit_rate={s['hit_rate']:.2f}, conflicts={s['conflicts']}")
    return 0


def _print_result(question: str, res) -> None:
    print(f"\n=== {question}")
    for c in res.report.claims:
        print(f"  - {c.text}  [{', '.join(c.evidence_ids)}]")
    print(f"  verification: {res.verification.summary()}")
    print(f"  llm_calls={res.llm_calls} skipped_by_memory={res.skipped_by_memory} "
          f"memory={res.memory_stats}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hma", description="hierarchical memory agents")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="answer one question with the full pipeline")
    r.add_argument("question")
    r.add_argument("--model", default=None,
                   help="backend: 'corpus' (default, offline) or a model name "
                        "for an OpenAI-compatible endpoint via $OPENAI_API_KEY")
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("session", help="run builtin related questions on one board")
    s.add_argument("--n", type=int, default=3)
    s.add_argument("--model", default=None)
    s.set_defaults(fn=cmd_session)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
