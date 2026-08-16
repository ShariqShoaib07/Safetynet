"""
cbf.py

The Control Barrier Function layer -- the final, non-LLM safety gate.

This is deliberately NOT a language model. Its entire job is to be the
one part of the pipeline that cannot be talked out of its constraints:
whatever the Task Planning LLM proposes and the Safety Planning LLM
approves still passes through here before it reaches the robot, and if
it violates a hard distance constraint, it gets blocked or clamped --
no exceptions, no reasoning, just math.

Design note: this formalizes logic that already exists informally in
safetynet/world_model_v2.py -- specifically `compute_zone()`, which uses
tighter DANGER_ZONE_AUTHORIZED_M/CAUTION_ZONE_AUTHORIZED_M radii for
authorized people and wider DANGER_ZONE_M/CAUTION_ZONE_M radii for
everyone/everything else. A CBF constraint is exactly this idea, made
mathematically explicit: h(x) = distance_to_object - safe_radius, and the
constraint is h(x) >= 0.

CURRENT LIMITATION (flag for later): the zone constants below are
duplicated from safetynet/world_model_v2.py's SAFETY ZONE / BOUNDARY
CONFIG block rather than imported from it. world_model_v2.py pulls in
torch/YOLO/RealSense/InsightFace at import time, which is far too heavy a
dependency for a pipeline module that should be able to run (and be
tested) without a camera attached. If the constants in world_model_v2.py
change, these must be updated to match by hand until both are refactored
to import from one shared config module -- worth doing before this goes
in the paper, so there's a single source of truth.
"""

from dataclasses import dataclass
import math

# --- Duplicated from safetynet/world_model_v2.py -- KEEP IN SYNC ---------
DANGER_ZONE_M = 0.5
CAUTION_ZONE_M = 1.5
DANGER_ZONE_AUTHORIZED_M = 0.15
CAUTION_ZONE_AUTHORIZED_M = 0.45
# ---------------------------------------------------------------------


@dataclass
class BarrierViolation:
    object_id: str
    object_class: str
    distance_m: float
    safe_radius_m: float
    margin_m: float  # negative = violated, how far into the danger zone


class ControlBarrierFunction:
    """Evaluates a proposed action against every object in the current
    world state and decides: allow, clamp (e.g. downgrade to STOP/HOLD),
    or block outright.

    v1 is intentionally simple -- it checks the robot's *current* position
    against every tracked object's danger radius (a static/instantaneous
    check), rather than forward-simulating the proposed action's full
    trajectory. That's a real limitation for a fast-moving arm, but it
    matches where the rest of the pipeline is right now (single robot,
    still defining the action space) and gives us a working, honest v1 to
    build the real trajectory-aware constraint on top of later.
    """

    def __init__(self, danger_zone_m=DANGER_ZONE_M, caution_zone_m=CAUTION_ZONE_M,
                 danger_zone_authorized_m=DANGER_ZONE_AUTHORIZED_M,
                 caution_zone_authorized_m=CAUTION_ZONE_AUTHORIZED_M):
        self.danger_zone_m = danger_zone_m
        self.caution_zone_m = caution_zone_m
        self.danger_zone_authorized_m = danger_zone_authorized_m
        self.caution_zone_authorized_m = caution_zone_authorized_m

    def _safe_radius(self, obj: dict) -> float:
        """Same authorization-aware radius logic as
        world_model_v2.compute_zone() -- authorized people get the tighter
        radius, everything/everyone else gets the wider one."""
        authorized = obj.get("authorized", False) and obj.get("class") == "person"
        return self.danger_zone_authorized_m if authorized else self.danger_zone_m

    def evaluate(self, world_objects: list[dict], robot_position=(0.0, 0.0, 0.0)) -> tuple[bool, list[BarrierViolation]]:
        """Returns (is_safe, violations).

        world_objects: the `objects` list straight out of world.yaml.
        robot_position: robot's current position in the same frame as the
        tracked objects. Defaults to the origin, matching how
        world_model_v2.py already treats distances as "distance from
        camera/robot" (see compute_zone using np.linalg.norm(position)
        directly, no separate robot-position subtraction).
        """
        violations = []
        for obj in world_objects:
            pos = obj.get("position")
            if pos is None:
                continue
            dist = math.dist(pos, robot_position)
            safe_radius = self._safe_radius(obj)
            margin = dist - safe_radius
            if margin < 0:
                violations.append(BarrierViolation(
                    object_id=obj.get("id", "?"),
                    object_class=obj.get("class", "?"),
                    distance_m=round(dist, 3),
                    safe_radius_m=safe_radius,
                    margin_m=round(margin, 3),
                ))
        return (len(violations) == 0), violations

    def gate(self, proposed_action: dict, world_objects: list[dict],
              robot_position=(0.0, 0.0, 0.0)) -> dict:
        """The actual gate called from main.py. Takes whatever the
        arbitrator settled on and returns the FINAL action that's allowed
        to reach the robot.

        proposed_action: dict, shape TBD alongside the arbitrator's output
        format -- expected to at minimum carry an "action" field.

        Returns a dict with the same shape as proposed_action, plus a
        "cbf" sub-dict recording what happened, so nothing is silently
        overridden without a trace in the logs / dialogue history.
        """
        is_safe, violations = self.evaluate(world_objects, robot_position)

        result = dict(proposed_action)
        if is_safe:
            result["cbf"] = {"gated": False, "violations": []}
            return result

        # Hard override -- CBF wins regardless of what the LLMs decided.
        result["action"] = "STOP"
        result["cbf"] = {
            "gated": True,
            "original_action": proposed_action.get("action"),
            "violations": [v.__dict__ for v in violations],
        }
        return result
