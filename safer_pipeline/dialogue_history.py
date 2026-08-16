"""
dialogue_history.py

Rolling log both the Task Planning LLM and Safety Planning LLM read from
and write to, so each call has recent context instead of reasoning from
a blank slate every tick. Adapted from COHERENT's dialogue-history
mechanism in PEFA/LLM_oracle.py (self.dialogue_history, last N turns
included in the oracle prompt each call).
"""

from collections import deque
from dataclasses import dataclass, field
import time


@dataclass
class Turn:
    timestamp: float
    source: str        # "task_planner" | "safety_checker" | "cbf"
    content: str


class DialogueHistory:
    def __init__(self, max_turns: int = 10):
        self._turns = deque(maxlen=max_turns)

    def add(self, source: str, content: str):
        self._turns.append(Turn(time.time(), source, content))

    def as_text(self) -> str:
        if not self._turns:
            return "(no prior history yet)"
        lines = []
        for t in self._turns:
            lines.append(f"[{t.source}] {t.content}")
        return "\n".join(lines)

    def __len__(self):
        return len(self._turns)
