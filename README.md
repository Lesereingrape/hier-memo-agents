# hier-memo-agents (`hma`)

> A hierarchical planner–executor research system with a **shared blackboard memory** and **citation provenance that fails validation when broken**.

`hma` decomposes a question into a subtask DAG, fans executors over a corpus, publishes findings to a shared memory that deduplicates and detects conflicts, and finally emits a report where **every claim must resolve to a verbatim source span** — or the run exits nonzero. Zero dependencies, deterministic, a full test suite that runs in well under a second, no API keys needed to try the whole pipeline.

```
question ──► Planner ──► DAG waves ──► Executors ──► Synthesizer ──► verify()
                            │              │                            │
                       cycle-safe     Blackboard                  evidence chain:
                       topological    dedup · conflicts           claim → quote → doc
                                      topic reuse (skip calls)    any edit breaks it
```

![ci](https://github.com/Lesereingrape/hier-memo-agents/actions/workflows/ci.yml/badge.svg)

## What this is not

Not another "agents chatting with each other" demo. The three hard problems in multi-agent systems are coordinated, measured, and tested here:

1. **Scheduling** — the planner's subtasks form a DAG; cycles and unknown deps (classic hallucinating-planner failure) are rejected before execution, and independent nodes are scheduled into reproducible waves.
2. **Redundant work** — executors ask the blackboard *before* spending a model call. Topic-level reuse means a second question covering an already-researched topic skips the research call entirely, and the skip is counted (`skipped_by_memory`).
3. **Trustworthy output** — quotes returned by the model are checked verbatim against the corpus *before* they become citable evidence; fabricated quotes are rejected and reported. The finished report is re-verified from scratch: unknown evidence ids, broken spans, or orphan claims all fail the run.

## Quickstart

```bash
pip install -e .
hma run "how does shared memory help multi-agent teams and what do citations need?"
hma session --n 3     # related questions on one persistent board; watch reuse kick in
```

Verified output (offline `CorpusLLM` backend):

<!-- RUN:START -->
```text
=== how does shared memory help multi-agent teams and what do citations need?
  - Shared memory lets parallel agents avoid repeating each other's work.  [r1:d2]
  - Without deduplication, multi-agent teams multiply their token bill.  [r2:d2]
  - Synthesis of 2 findings over 2 sources.  [r1:d2, r2:d2]
  verification: VERIFIED: every claim resolves to a verbatim corpus span
  llm_calls=6 skipped_by_memory=0 memory={'entries': 4, 'hits': 0, 'misses': 0, 'hit_rate': 0.0, 'conflicts': 3}
```
<!-- RUN:END -->

Session stats show the memory economy: questions 1–2 spend 6 model calls each; the third question, which revisits an already-covered topic, needs 5:

<!-- SESSION:START -->
```text
llm_calls=6 skipped_by_memory=0 memory={'entries': 4, 'hits': 0, 'misses': 0, 'hit_rate': 0.0, 'conflicts': 3}
llm_calls=12 skipped_by_memory=0 memory={'entries': 4, 'hits': 0, 'misses': 0, 'hit_rate': 0.0, 'conflicts': 3}
llm_calls=17 skipped_by_memory=1 memory={'entries': 4, 'hits': 1, 'misses': 0, 'hit_rate': 1.0, 'conflicts': 4}
blackboard after session: 4 findings, hit_rate=1.00, conflicts=4
```
<!-- SESSION:END -->

Both blocks are the stdout of those two commands, spliced in by
`python experiments/make_readme.py --write` and pinned by
`tests/test_readme_output_is_real.py` - a sample output no code can print is worse than
no sample output at all.

## Bring your own model

The agent↔model seam is one method: `complete_json(request) -> dict` with three contracts (`plan`, `research`, `synthesize`).

```python
from hma import Corpus, Orchestrator
from hma.llm import OpenAICompatLLM

llm = OpenAICompatLLM(model="gpt-4o-mini")   # any OpenAI-compatible endpoint
orch = Orchestrator(llm, Corpus())
res = orch.run("what decides long-task success in agent systems?")
assert res.verification.ok
```

`CorpusLLM` is a deterministic retrieval-driven stand-in implementing the same contracts — which is what makes the provenance guarantees testable in CI. Swapping in a hosted model changes nothing upstream or downstream: the verbatim-quote check and `verify()` guardrails are enforced by the framework, not the model.

## Design notes

- **Blackboard over pairwise chat.** Agents communicate only through findings (key, answer, evidence ids, topic tags) — no N² message threads. Jaccard dedup drops identical claims re-derived from the same evidence; same-topic-different-answer findings are logged as `Conflict`s for a reviewer instead of silently overwriting.
- **Prompt-cost accounting.** `skipped_by_memory` is a first-class result field: if your "multi-agent" system never measures call savings, it's probably paying for them.
- **Provenance as data, not prose.** Evidence ids (`r1:d2`) are deterministic, so a report diff shows exactly which subtask→document link changed.

## Tests prove the guarantees, not the happy path

26 tests, ~0.2 s: DAG cycle/dep/duplicate rejection · memory dedup + conflict detection · `cite()` refusing stitched quotes · `verify()` catching orphan claims and spans that drift after corpus edits · the orchestrator rejecting a scripted hallucinated quote · cross-question memory reuse · the two sample-output blocks above re-generated from the CLI and compared byte for byte.

Regenerate the blocks after changing the pipeline, so the docs cannot keep a transcript the
code no longer produces:

```bash
python experiments/make_readme.py --write
```

## Roadmap

- [ ] Retraction semantics: findings invalidated when upstream evidence breaks
- [ ] Budget-aware scheduling (stop the DAG when the token bill is spent)
- [ ] Evaluation adapter against [agent-harness-eval](https://github.com/Lesereingrape/agent-harness-eval) trace format
- [ ] Real-corpus mode: ingests your documents/URLs with the same verbatim rules

## License

Apache-2.0
