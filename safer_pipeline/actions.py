"""
actions.py

Placeholder action space for v1 -- a single robot arm, tested physically
and in simulation. Treat everything here as a tunable placeholder, same
spirit as the constants in world_model_v2.py's config block: this is a
minimal set to get the pipeline running end-to-end, not the final word on
what the arm can do. Swap in the real arm API's actual action set once
that's pinned down.

Kept deliberately close to COHERENT's `[action] <target>` string format
(see get_available_plans() in coherent_upstream/.../PEFA/LLM.py) so the
Task Planning LLM prompt's output-parsing logic can reuse that same
pattern rather than inventing a new one.
"""

# Each entry: (action_name, takes_target: bool, description)
ROBOT_ARM_ACTIONS = [
    ("move_to", True, "Move the arm toward a named object/position"),
    ("grasp", True, "Close the gripper on a named object"),
    ("release", False, "Open the gripper, releasing whatever is held"),
    ("hold", False, "Freeze in place, do nothing further this tick"),
    ("retreat", False, "Move to a predefined safe/home position"),
    ("stop", False, "Immediate stop -- used by the safety layer/CBF, "
                     "not something the task planner should normally choose"),
]


def available_actions_text(candidate_object_ids: list[str]) -> str:
    """Builds the "A. [action] <target>" style list COHERENT's prompts
    expect (see PEFA/prompt/robot_arm_prompt.txt), so the Task Planning
    LLM prompt can just substitute this in as #ACTIONLIST#."""
    lines = []
    letter = ord("A")
    for name, takes_target, _ in ROBOT_ARM_ACTIONS:
        if takes_target:
            for obj_id in candidate_object_ids:
                lines.append(f"{chr(letter)}. [{name}] <{obj_id}>")
                letter += 1
        else:
            lines.append(f"{chr(letter)}. [{name}]")
            letter += 1
    return "\n".join(lines)


def action_names() -> list[str]:
    return [a[0] for a in ROBOT_ARM_ACTIONS]
