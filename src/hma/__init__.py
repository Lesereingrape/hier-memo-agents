"""hma — hierarchical memory agents.

A planner decomposes research questions into a subtask DAG, executors
research against a corpus while sharing a deduplicating blackboard memory,
and the synthesizer may only emit claims that verify back to verbatim
source spans.
"""

from .corpus import DEMO_DOCS, Corpus
from .corpus_llm import CorpusLLM
from .dag import DAG, CycleError, Node, UnknownDepError
from .evidence import Claim, ProvenanceStore, Report, verify
from .llm import OpenAICompatLLM, ScriptedLLM
from .memory import Blackboard, Finding
from .orchestrator import Orchestrator, RunResult

__version__ = "0.1.0"

__all__ = [
           "DAG",
           "DEMO_DOCS",
           "Blackboard",
           "Claim",
           "Corpus",
           "CorpusLLM",
           "CycleError",
           "Finding",
           "Node",
           "OpenAICompatLLM",
           "Orchestrator",
           "ProvenanceStore",
           "Report",
           "RunResult",
           "ScriptedLLM",
           "UnknownDepError",
           "verify",
]
