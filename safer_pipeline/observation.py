"""
observation.py

Turns safetynet's world.yaml / world_diff.yaml into plain text for the
LLM prompts. This replaces COHERENT's `agent_obs2text()` (which walked a
sim scene-graph of nodes/edges) -- our "graph" is just the flat objects
list world_model_v2.py already writes out, so this is simpler, not a
port of that function.

Two builders:
  - build_task_observation(world_state)  -> for the Task Planning LLM,
    reads the full world.yaml snapshot every tick.
  - build_safety_observation(diff_state) -> for the Safety Planning LLM,
    reads only the added/changed/removed since the last diff -- this is
    what actually triggers a safety-check call in main.py.
"""

import yaml


def load_world(path: str) -> dict:
    """Loads world.yaml/world_diff.yaml defensively -- world_model_v2.py
    writes these on its own schedule/error paths (e.g. an empty
    shutdown snapshot after a camera crash, or a non-atomic write caught
    mid-line by our poll), so a load failure or an unexpectedly empty/
    null file here should never crash the pipeline -- just skip this tick."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (yaml.YAMLError, OSError) as e:
        print(f"[observation] WARNING: failed to read {path} this tick ({e}) -- skipping")
        return {}
    return data or {}


def _describe_object(obj: dict) -> str:
    cls = obj.get("class", "object")
    oid = obj.get("id", "?")
    zone = obj.get("zone", "unknown")
    risk = obj.get("risk_score", 0.0)
    pos = obj.get("position", [0, 0, 0])
    parts = [f"{cls} ({oid}) at [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}], "
             f"zone={zone}, risk={risk:.3f}"]

    if obj.get("class") == "person":
        auth = obj.get("authorized", False)
        name = obj.get("name") or "unidentified"
        parts.append(f"authorized={auth} (name={name})")

    if obj.get("static"):
        parts.append("static")
    if not obj.get("visible", True):
        parts.append("not currently visible (last known position)")

    return ", ".join(parts)


def build_task_observation(world_state: dict, max_objects: int = 40) -> str:
    """Full-state description for the Task Planning LLM. Caps the object
    count (world.yaml can have hundreds of stale/low-confidence entries --
    see CONF_THRESHOLD in world_model_v2.py) so the prompt doesn't blow up;
    prioritizes non-static, higher-risk, currently-visible objects first,
    since those are what actually matter for planning the next action.
    """
    objects = world_state.get("objects") or []

    def sort_key(o):
        return (
            not o.get("visible", False),      # visible first
            o.get("static", False),           # dynamic before static
            -o.get("risk_score", 0.0),        # higher risk first
        )

    objects_sorted = sorted(objects, key=sort_key)[:max_objects]

    lines = [f"World snapshot at {world_state.get('timestamp', 'unknown time')}."]
    lines.append(f"{len(objects)} tracked objects total "
                 f"(showing the {len(objects_sorted)} most relevant):")
    for obj in objects_sorted:
        lines.append(f"  - {_describe_object(obj)}")
    return "\n".join(lines)


def build_safety_observation(diff_state: dict) -> str:
    """Event description for the Safety Planning LLM -- only the
    added/changed/removed entries since the last diff, matching
    world_diff.yaml's own structure (zone-crossing-triggered, see
    DIFFED_FIELDS / compute_zone in world_model_v2.py)."""
    lines = [f"Zone-crossing event at {diff_state.get('timestamp', 'unknown time')}."]

    added = diff_state.get("added", [])
    changed = diff_state.get("changed", [])
    removed = diff_state.get("removed", [])

    if added:
        lines.append(f"NEWLY DETECTED ({len(added)}):")
        for obj in added:
            lines.append(f"  - {_describe_object(obj)}")
    if changed:
        lines.append(f"CHANGED ({len(changed)}):")
        for obj in changed:
            lines.append(f"  - {_describe_object(obj)}")
    if removed:
        # removed entries are typically just IDs -- handle both shapes
        lines.append(f"REMOVED ({len(removed)}):")
        for obj in removed:
            oid = obj.get("id", obj) if isinstance(obj, dict) else obj
            lines.append(f"  - {oid}")

    if not (added or changed or removed):
        lines.append("(No changes -- this diff should not have triggered a call.)")

    return "\n".join(lines)
