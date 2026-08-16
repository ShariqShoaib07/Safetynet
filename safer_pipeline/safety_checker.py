"""
safety_checker.py

The Safety Planning LLM call site. Unlike the task planner, this does NOT
run every tick -- it's event-driven, fired only when world_diff.yaml has
new content (a zone-crossing occurred). Its last verdict is held and
reused by the arbitrator on every tick until the next diff event replaces
it (see main.py's polling loop).

Also does the second job discussed with the supervisor: if Safety-Weighted
ToT is enabled, this same LLM scores the Task Planner's candidate actions
so the framework can pick the highest-safety branch (kept to one layer,
not recursive).
"""

import os
import re

from llm_client import LLMClient, load_prompt
from observation import build_safety_observation
from dialogue_history import DialogueHistory

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "safety_prompt.txt")

_VERDICT_RE = re.compile(r"VERDICT:\s*(SAFE|CAUTION|STOP)\s*(?::\s*(.+))?", re.IGNORECASE)
_SCORE_RE = re.compile(r"SCORE:\s*(.+?)\s*\|\s*(\d+)\s*\|\s*REASON:\s*(.+)")


class SafetyChecker:
    def __init__(self, model: str = "gpt-4o-mini", api_key_env: str = "SAFETY_LLM_API_KEY"):
        self.client = LLMClient(model=model, api_key_env=api_key_env, role="safety_checker")

    def check_event(self, diff_state: dict, history: DialogueHistory) -> dict:
        """Called only when world_diff.yaml changes. Returns
        {"verdict": "SAFE"|"CAUTION"|"STOP", "reason": str}."""
        event_text = build_safety_observation(diff_state)
        prompt = load_prompt(
            PROMPT_PATH,
            safety_event=event_text,
            candidates="(none -- this is an event-triggered call)",
            dialogue_history=history.as_text(),
        )
        raw = self.client.generate([{"role": "user", "content": prompt}])
        history.add("safety_checker", raw)

        m = _VERDICT_RE.search(raw)
        if not m:
            # Fail safe: if we can't parse a verdict, treat it as STOP
            # rather than silently defaulting to SAFE.
            return {"verdict": "STOP", "reason": "unparseable safety response -- failing safe"}
        return {"verdict": m.group(1).upper(), "reason": (m.group(2) or "").strip()}

    def score_candidates(self, candidates: list[dict], history: DialogueHistory) -> list[dict]:
        """Called from the arbitrator when Safety-Weighted ToT is active.
        Returns candidates annotated with a "safety_score" (0-10),
        highest first."""
        if not candidates:
            return []

        candidate_text = "\n".join(
            f"- {c['action']} (reason: {c['reason']})" for c in candidates
        )
        prompt = load_prompt(
            PROMPT_PATH,
            safety_event="(none -- this is a candidate-scoring call)",
            candidates=candidate_text,
            dialogue_history=history.as_text(),
        )
        raw = self.client.generate([{"role": "user", "content": prompt}])
        history.add("safety_checker", raw)

        scores = {}
        for line in raw.splitlines():
            m = _SCORE_RE.search(line)
            if m:
                scores[m.group(1).strip()] = int(m.group(2))

        scored = []
        for c in candidates:
            c = dict(c)
            c["safety_score"] = scores.get(c["action"], 0)  # unscored = treat as unsafe
            scored.append(c)
        return sorted(scored, key=lambda c: -c["safety_score"])
