"""
main.py

The orchestration loop. Reads world.yaml every tick for the Task
Planning LLM, watches world_diff.yaml's mtime for Safety Planning LLM
triggers, arbitrates between them, gates the result through the CBF
layer, and logs everything to the shared dialogue history.

This is deliberately a polling loop, not a file-watcher (inotify/etc.) --
matches how world_model_v2.py itself already works (fixed UPDATE_INTERVAL
writes), so the two halves of the system move in lockstep without adding
an extra dependency. Swap in a real file-watcher later if polling latency
ever actually matters.

PLACEHOLDER STATE: task_goal, candidate_object_ids, and the "execute
final action" step at the bottom are all stand-ins -- there's no real
goal definition or robot execution API wired up yet since those are
still open questions (robot arm action space, what "execute" even means
for the physical/sim setup). Everything up to that point (observation
building, both LLM calls, arbitration, CBF gating, logging) is real and
runnable end-to-end with placeholder LLM keys.
"""

import os
import time

import yaml

from observation import load_world
from task_planner import TaskPlanner
from safety_checker import SafetyChecker
from arbitration import arbitrate
from cbf import ControlBarrierFunction
from dialogue_history import DialogueHistory

WORLD_YAML = os.path.join(os.path.dirname(__file__), "..", "safetynet", "world.yaml")
WORLD_DIFF_YAML = os.path.join(os.path.dirname(__file__), "..", "safetynet", "world_diff.yaml")

TICK_SECONDS = 1.0          # how often the task planner re-evaluates
TASK_GOAL = "PLACEHOLDER -- define the actual task goal here"
USE_TOT_SCORING = True      # Safety-Weighted Tree-of-Thought, single layer


def _candidate_object_ids(world_state: dict, limit: int = 10) -> list[str]:
    """Placeholder target-selection: just the nearest few non-static
    objects. Replace with real task-relevant object selection once the
    task goal format is defined."""
    objects = [o for o in (world_state.get("objects") or []) if not o.get("static")]
    objects.sort(key=lambda o: o.get("risk_score", 0.0))
    return [o["id"] for o in objects[:limit]]


def run():
    history = DialogueHistory(max_turns=10)
    task_planner = TaskPlanner()
    safety_checker = SafetyChecker()
    cbf = ControlBarrierFunction()

    last_diff_mtime = None
    last_safety_verdict = {"verdict": "SAFE", "reason": "no events yet"}

    print("[main] SAFER pipeline starting. Using placeholder LLM keys until "
          "TASK_LLM_API_KEY / SAFETY_LLM_API_KEY are set.")

    while True:
        if not os.path.exists(WORLD_YAML):
            print(f"[main] waiting for {WORLD_YAML} to exist "
                  f"(is world_model_v2.py running?)")
            time.sleep(TICK_SECONDS)
            continue

        world_state = load_world(WORLD_YAML)

        # --- Safety Planning LLM: only fires on a new world_diff.yaml ---
        if os.path.exists(WORLD_DIFF_YAML):
            mtime = os.path.getmtime(WORLD_DIFF_YAML)
            if mtime != last_diff_mtime:
                last_diff_mtime = mtime
                diff_state = load_world(WORLD_DIFF_YAML)
                last_safety_verdict = safety_checker.check_event(diff_state, history)
                print(f"[main] new zone-crossing event -> verdict: "
                      f"{last_safety_verdict['verdict']} "
                      f"({last_safety_verdict['reason']})")

        # --- Task Planning LLM: runs every tick ---
        candidate_ids = _candidate_object_ids(world_state)
        candidates = task_planner.propose(world_state, TASK_GOAL, candidate_ids, history)

        scored = None
        if USE_TOT_SCORING and candidates:
            scored = safety_checker.score_candidates(candidates, history)

        # --- Arbitration (code, not an LLM) ---
        proposed = arbitrate(candidates, last_safety_verdict, scored_candidates=scored)

        # --- CBF gate: final, non-LLM authority ---
        final_action = cbf.gate(proposed, world_state.get("objects") or [])

        history.add("cbf", f"final_action={final_action['action']} "
                            f"(gated={final_action['cbf']['gated']})")

        print(f"[main] tick -> final_action: {final_action}")

        # PLACEHOLDER: this is where the action would actually be sent to
        # the robot arm (physical driver or sim). Not wired up yet.
        # execute(final_action)

        time.sleep(TICK_SECONDS)


if __name__ == "__main__":
    run()
