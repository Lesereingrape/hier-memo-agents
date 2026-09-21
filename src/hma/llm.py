"""LLM client seam: a JSON-in / JSON-out protocol with swappable backends.

Every agent decision is one structured request (`plan`, `research`,
`synthesize`). `CorpusLLM` is a deterministic, retrieval-driven stand-in so
the whole system runs and is testable without any API key; `OpenAICompatLLM`
plugs the identical contracts into any OpenAI-compatible endpoint.
"""

from __future__ import annotations

import json
import os
from typing import Protocol


class LLMClient(Protocol):
    def complete_json(self, request: dict) -> dict: ...
    @property
    def n_calls(self) -> int: ...


class ScriptedLLM:
    """Returns canned responses in order — the workhorse for unit tests."""

    def __init__(self, responses: list[dict]):
        self._responses = list(responses)
        self.calls: list[dict] = []
        self.n_calls = 0

    def complete_json(self, request: dict) -> dict:
        self.calls.append(request)
        self.n_calls += 1
        if not self._responses:
            raise RuntimeError("ScriptedLLM exhausted")
        return self._responses.pop(0)


class OpenAICompatLLM:
    """Chat-completions backend for any OpenAI-compatible endpoint.

    Optional dependency: requires `pip install hma[http]` (httpx only).
    """

    def __init__(self, model: str = "gpt-4o-mini", api_base: str = "https://api.openai.com/v1",
                 api_key_env: str = "OPENAI_API_KEY", temperature: float = 0.0):
        try:
            import httpx
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("OpenAICompatLLM needs `pip install hma[http]`") from e
        key = os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(f"missing API key in ${api_key_env}")
        self._http = httpx.Client(base_url=api_base, timeout=120,
                                  headers={"Authorization": f"Bearer {key}"})
        self.model = model
        self.temperature = temperature
        self.n_calls = 0

    def complete_json(self, request: dict) -> dict:
        self.n_calls += 1
        resp = self._http.post(
            "/chat/completions",
            json={
                "model": self.model,
                "temperature": self.temperature,
                "response_format": {"type": "json_object"},
                "messages": [{
                    "role": "system",
                    "content": "You power a multi-agent research system. Reply with "
                               "ONLY a JSON object matching the requested schema.",
                }, {"role": "user", "content": json.dumps(request)}],
            },
        )
        resp.raise_for_status()
        return json.loads(resp.json()["choices"][0]["message"]["content"])
