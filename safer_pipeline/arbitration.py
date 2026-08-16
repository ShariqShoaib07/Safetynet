"""
arbitration.py

Combines the Task Planner's candidate(s) with the Safety Checker's latest
verdict/scores into one proposed action for the CBF layer to gate.

This is plain code, NOT a third LLM -- see the architecture discussion:
SAFER's real structure is Task LLM + Safety LLM + a deterministic CBF
layer, not three LLMs. This module is the small piece of glue logic that
sits between the two LLM outputs and the CBF, so it needed to live
somewhere, but it's rules, not a model call.

Two situations this handles:
  1. No Safety-Weighted ToT scoring available yet (or it's off) -- just
     apply the latest event-driven verdict as an override on the task
     planner's top candidate.
  2. Safety-Weighted ToT scoring is available -- pick the
     highest-safety-scored candidate, but a STOP verdict from the latest
     zone-crossing event still overrides everything, since an active
     STOP means something is in the danger zone RIGHT NOW, which
     out-ranks a branch-scoring judgment from a few ticks ago.
"""


def arbitrate(task_candidates: list[dict], safety_verdict: dict,
              scored_candidates: list[dict] | None = None) -> dict:
    """Returns the final proposed action dict, before CBF gating:
    {"action": str, "reason": str, "source": str}
    """
    # A live STOP verdict always wins, no matter what the task planner wants.
    if safety_verdict.get("verdict") == "STOP":
        return {
            "action": "stop",
            "reason": f"Safety verdict STOP: {safety_verdict.get('reason', 'unspecified')}",
            "source": "safety_override",
        }

    if not task_candidates:
        return {
            "action": "hold",
            "reason": "Task planner found no progress possible",
            "source": "task_planner",
        }

    if scored_candidates:
        best = scored_candidates[0]
        if best["safety_score"] < 4:
            # Even the "best" candidate scored poorly -- don't act on it.
            return {
                "action": "hold",
                "reason": f"Best candidate '{best['action']}' scored too low "
                          f"({best['safety_score']}/10) to execute",
                "source": "arbitration",
            }
        action_note = ""
        if safety_verdict.get("verdict") == "CAUTION":
            action_note = f" (CAUTION active: {safety_verdict.get('reason', '')})"
        return {
            "action": best["action"],
            "reason": f"{best['reason']}{action_note}",
            "source": "task_planner+safety_scoring",
        }

    # No ToT scoring available -- fall back to the task planner's top
    # candidate, annotated with whatever the latest verdict was.
    top = task_candidates[0]
    action_note = ""
    if safety_verdict.get("verdict") == "CAUTION":
        action_note = f" (CAUTION active: {safety_verdict.get('reason', '')})"
    return {
        "action": top["action"],
        "reason": f"{top['reason']}{action_note}",
        "source": "task_planner",
    }
