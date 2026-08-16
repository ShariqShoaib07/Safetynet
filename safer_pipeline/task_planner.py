"""
task_planner.py

The Task Planning LLM call site. Runs every tick, reads the current
world state, and proposes up to 3 candidate actions (Safety-Weighted
Tree-of-Thought, kept to a single branching layer per supervisor
feedback -- see meeting notes, "keep everything in one layer rather
than many layers, don't make it too complex").

Candidate parsing is intentionally forgiving (regex-lite line matching),
mirroring COHERENT's own parse_answer() approach of matching against a
known action vocabulary rather than requiring strict JSON from the LLM --
strict-format compliance from chat models is unreliable enough that
COHERENT builds a second reformatting call around it (see LLM_oracle.py).
We keep that lesson but simplify: one parse pass, fall back to "no
candidates" rather than a second LLM call, since v1's goal is a working
pipeline, not squeezing out format-compliance edge cases yet.
"""

import os
import re

from llm_client import LLMClient, load_prompt
from observation import build_task_observation
from actions import available_actions_text, action_names
from dialogue_history import DialogueHistory

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "task_prompt.txt")

_CANDIDATE_RE = re.compile(r"CANDIDATE:\s*(.+?)\s*\|\s*REASON:\s*(.+)")


class TaskPlanner:
    def __init__(self, model: str = "gpt-4o-mini", api_key_env: str = "TASK_LLM_API_KEY"):
        self.client = LLMClient(model=model, api_key_env=api_key_env, role="task_planner")

    def propose(self, world_state: dict, task_goal: str,
                candidate_object_ids: list[str],
                history: DialogueHistory) -> list[dict]:
        """Returns a list of {"action": str, "reason": str} candidates,
        up to 3, in the order the LLM proposed them (not yet safety-scored
        -- that happens in safety_checker.py)."""
        observation = build_task_observation(world_state)
        action_list = available_actions_text(candidate_object_ids)

        prompt = load_prompt(
            PROMPT_PATH,
            task_goal=task_goal,
            observation=observation,
            dialogue_history=history.as_text(),
            actionlist=action_list,
        )

        raw = self.client.generate([{"role": "user", "content": prompt}])
        candidates = self._parse(raw)
        history.add("task_planner", raw if candidates else f"(no progress) {raw}")
        return candidates

    def _parse(self, raw: str) -> list[dict]:
        if raw.strip().upper().startswith("NO PROGRESS POSSIBLE"):
            return []

        candidates = []
        known_actions = action_names()
        for line in raw.splitlines():
            m = _CANDIDATE_RE.search(line)
            if not m:
                continue
            action_str, reason = m.group(1).strip(), m.group(2).strip()
            # sanity check: the action name (ignoring target) must be one we know
            base_action = action_str.strip("[]").split("]")[0].strip("[")
            if any(name in action_str for name in known_actions):
                candidates.append({"action": action_str, "reason": reason})
        return candidates[:3]
