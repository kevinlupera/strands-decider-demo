"""Synchronous client for a locally served Strands Decider 2B model.

Strands Decider is a *decision model* (a "system one" model): it does not generate
text. It reads a `state` plus one or more typed questions and returns a calibrated
probability over the options of each question, in a single forward pass. See
https://strandsagents.com/blog/introducing-strands-decider/ and
https://github.com/strands-labs/strands-decider

This client talks to the server you start with:

    strands-decider serve StrandsAgents/strands-decider-2B-hobson-v19 --port 8099

The transport is deliberately `urllib` (no extra dependency) and synchronous, which
mirrors the official example client and keeps requests to the model strictly
sequential -- the way the server is best driven.

OFFLINE FALLBACK
----------------
If no server is reachable, `Decider(allow_fallback=True)` answers with a transparent,
deterministic keyword heuristic and marks every answer `"fallback": True`. This keeps
the demos runnable without the ~2B model downloaded, and lets you show the *pattern*
and the cost/routing math on any machine. A fallback answer is NOT a real decision-model
classification and must never be presented as one -- the demos label it explicitly.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any

NAME = "strands-decider"
ENV_URL = "STRANDS_DECIDER_URL"
SERVE_HINT = (
    "strands-decider serve StrandsAgents/strands-decider-2B-hobson-v19 --port 8099"
)
MODEL_ID = "hobson-latest"  # sent as `model`; the server answers as whatever it loaded
DEFAULT_PORT = 8099
DEFAULT_URL = os.environ.get(ENV_URL, f"http://127.0.0.1:{DEFAULT_PORT}")


class ServerUnavailable(RuntimeError):
    """The decider server is not answering. Carries the command that starts one."""


class Decider:
    """One state, many typed questions, one round trip.

    Questions in a single `ask` cannot see each other, so their answers stay
    independent -- but the state is encoded once and shared, which is why asking
    several questions costs far less than several requests. Ask everything at once.
    """

    def __init__(
        self,
        url: str = DEFAULT_URL,
        timeout: float = 120.0,
        allow_fallback: bool = False,
    ) -> None:
        self.url = url.rstrip("/")
        # Generous: on Apple silicon the first request at a given input length pays a
        # one-off MPS/MLX shape compile that can be seconds on a long state.
        self.timeout = timeout
        self.allow_fallback = allow_fallback
        self.calls = 0
        self.input_tokens = 0
        self.last_latency_ms: float | None = None
        self.using_fallback = False

    # ---- question builders ------------------------------------------------

    @staticmethod
    def noul(instructions: str, criteria: Mapping[str, str] | None = None) -> dict[str, Any]:
        """Yes/no. The answer is P(true); with two outcomes that *is* the confidence.

        `criteria` describes what `"true"`/`"false"` mean and sharpens the boundary
        considerably -- worth writing as carefully as the question itself.
        """
        q: dict[str, Any] = {"type": "noul", "instructions": instructions}
        if criteria:
            q["criteria"] = dict(criteria)
        return q

    @staticmethod
    def choice(instructions: str, criteria: Mapping[str, str]) -> dict[str, Any]:
        """Pick one of N. `criteria` maps option name -> what that option means."""
        return {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}

    @staticmethod
    def score(instructions: str, criteria: list[str]) -> dict[str, Any]:
        """Rate against an ordered rubric, lowest level first."""
        return {"type": "score", "instructions": instructions, "criteria": list(criteria)}

    # ---- asking -----------------------------------------------------------

    def _unavailable(self, exc: Exception) -> ServerUnavailable:
        return ServerUnavailable(
            f"no {NAME} server at {self.url} ({exc}). Start one with:\n    {SERVE_HINT}"
        )

    def ask(self, state: Any, questions: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
        """Evaluate `state` against `questions`; answers return under the same keys.

        Falls back to the local heuristic only when `allow_fallback=True` and the
        server is unreachable.
        """
        body = json.dumps(
            {"state": state, "model": MODEL_ID, "questions": dict(questions)}
        ).encode()
        request = urllib.request.Request(
            f"{self.url}/v1/systemone",
            data=body,
            headers={"content-type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read())
        except urllib.error.URLError as exc:
            if self.allow_fallback:
                return self._fallback(state, questions)
            raise self._unavailable(exc) from exc
        self.calls += 1
        self.input_tokens += payload.get("usage", {}).get("input_tokens", 0)
        self.last_latency_ms = payload.get("latency_ms")
        return payload["answers"]

    def health(self) -> dict[str, Any]:
        """The loaded checkpoint, window and device. Raises if nothing is serving."""
        try:
            with urllib.request.urlopen(f"{self.url}/health", timeout=10) as response:
                return json.loads(response.read())
        except urllib.error.URLError as exc:
            raise self._unavailable(exc) from exc

    def banner(self) -> str:
        """A one-line description of the serving model, or the fallback notice."""
        try:
            h = self.health()
        except ServerUnavailable:
            if self.allow_fallback:
                self.using_fallback = True
                return (
                    f"{NAME}: NO SERVER -> using OFFLINE HEURISTIC FALLBACK "
                    f"(not a real decision-model classification). Start the real "
                    f"model with:\n    {SERVE_HINT}"
                )
            raise
        return (
            f"{NAME}: {h.get('checkpoint', '?')} on {h.get('base_model', '?')} "
            f"[{h.get('device', '?')}, window {h.get('max_length', '?')}] at {self.url}"
        )

    def __str__(self) -> str:
        mode = " (fallback)" if self.using_fallback else ""
        return f"{self.calls} decisions, {self.input_tokens} input tokens{mode}"

    # ---- offline fallback -------------------------------------------------

    def _fallback(
        self, state: Any, questions: Mapping[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """Deterministic keyword heuristic. Transparent stand-in, never a real decision.

        Every answer carries `"fallback": True` so callers can label it. The numbers
        are plausible but hand-rolled; they exist to keep the demo runnable, not to
        stand in for the model's calibrated output.
        """
        self.using_fallback = True
        self.calls += 1
        text = str(state).lower()
        start = time.perf_counter()
        answers: dict[str, Any] = {}
        for key, q in questions.items():
            qtype = q["type"]
            instr = q.get("instructions", "").lower()
            if qtype == "noul":
                answers[key] = {"type": "noul", "noul": self._heur_noul(key, instr, text),
                                "fallback": True}
            elif qtype == "choice":
                opts = list(q.get("criteria", {}).keys())
                answers[key] = {"type": "choice", "choice": opts[0] if opts else "",
                                "confidence": 0.5, "fallback": True}
            else:  # score
                levels = q.get("criteria", [])
                answers[key] = {"type": "score", "score": float(len(levels) - 1) / 2 if levels else 0.0,
                                "confidence": 0.5, "fallback": True}
        self.last_latency_ms = (time.perf_counter() - start) * 1000.0
        return answers

    @staticmethod
    def _heur_noul(key: str, instr: str, text: str) -> float:
        """Crude but deterministic yes/no probabilities for the demo questions."""
        if "grounded" in key or "grounded" in instr:
            # Treat a state that mentions a concrete place/value as 'grounded'.
            grounded_markers = any(w in text for w in (
                "city:", "location:", "in paris", "in seattle", "in tokyo", "in madrid",
            ))
            return 0.82 if grounded_markers else 0.18
        if "premature" in key or "premature" in instr or "clarify" in instr:
            missing = any(w in text for w in ("none", "null", "unknown", "guessed", "assume"))
            return 0.80 if missing else 0.20
        if "complex" in key or "complex" in instr:
            # Signals of a hard request: length, or words that imply multi-step work.
            hard_words = (
                "design", "prove", "proof", "refactor", "diagnose", "architecture",
                "enumerate", "compare", "justify", "trade-off", "tradeoff", "rigorous",
                "fault-tolerant", "distributed", "consensus", "isolation",
            )
            score = 0.0
            if len(text) > 120:
                score += 0.4
            if any(w in text for w in hard_words):
                score += 0.45
            return min(0.9, max(0.1, score)) if score else 0.2
        return 0.5
