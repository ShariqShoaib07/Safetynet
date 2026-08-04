"""
World Model v5 -- Open-vocabulary detection + AUTHORIZATION-based person ID.

This is a different problem than generic re-id: the robot doesn't need
to know "is this the same person as 10 seconds ago" for its own sake --
it needs to know "is this person on my authorized list, so I can work
close to them, or not, so I should keep distance." That's access
control, not tracking, and it FAILS CLOSED: if we're not confident,
the person is treated as unauthorized. A missed authorization just
means the robot is cautious for a moment. A false authorization is a
safety miss.

Design vs v4:
  1. Enroll authorized lab workers once with enroll_worker.py --
     produces known_workers.yaml (name -> averaged face embedding).
  2. Per-frame: person tracked SPATIALLY (cheap, same as before) so a
     known person's box doesn't churn ids while just standing/walking.
  3. Periodically (every AUTH_CHECK_INTERVAL_SEC of wall-clock time, and
     sighting of a new track) run face detection + match against
     known_workers.yaml with a strict threshold.
       - Confident match  -> authorized: true, name: "..."
       - No match / no face visible / low confidence -> authorized: false
     Once a track is confirmed authorized, we keep believing it for a
     grace period even if a later frame briefly loses the face angle
     (walking, turning) -- but it re-verifies periodically rather than
     trusting the first read forever, and any REVERSAL (once-authorized
     face no longer matches on a strong read) immediately drops back to
     unauthorized rather than staying stuck on a stale "safe" label.
  4. world.yaml carries `authorized` and `name` per person so the SAFER
     planner can key proximity behavior off it directly.
  5. Non-person objects unchanged: spatial-only, class label, no
     persistent per-instance identity system.

Run:
  python enroll_worker.py "Dr. Ali Murtaza"      # once per known worker
  python world_model_v2.py
"""

import time
import numpy as np
import cv2
import yaml
import torch
import pyrealsense2 as rs
from ultralytics import YOLOWorld
from insightface.app import FaceAnalysis

# ==========================================================================
# ============ SAFETY ZONE / BOUNDARY CONFIG -- EDIT VALUES HERE =========
# ==========================================================================
# Every number that controls the proximity "boundary" -- how close is
# dangerous, how close is a warning, how the radar/rings are drawn -- is
# collected here in ONE place. Nothing else in the file needs to change
# to retune the boundary; every other usage just reads these constants.

DANGER_ZONE_M = 0.5     # meters -- UNAUTHORIZED person (or any non-person object) inside this
                          # radius of the robot = "danger"
CAUTION_ZONE_M = 1.5    # meters -- same, but "caution" (outside danger, inside this)
                          # beyond CAUTION_ZONE_M => "safe"
                          # THESE ARE PLACEHOLDER DEFAULTS -- tune them against your actual
                          # robot's real reach/workspace, not against these numbers.

DANGER_ZONE_AUTHORIZED_M = 0.15   # meters -- AUTHORIZED person only. Tighter than the
CAUTION_ZONE_AUTHORIZED_M = 0.45  # unauthorized radii above, NOT zero -- an authorized worker
                                     # is trusted to behave safely, but still needs SOME live
                                     # margin: even a known worker standing right on top of the
                                     # robot is a real safety situation, not a non-event. Same
                                     # ratio (caution = 3x danger) as the unauthorized pair above.
                                     # PLACEHOLDER DEFAULTS -- tune alongside DANGER_ZONE_M/
                                     # CAUTION_ZONE_M.

RISK_GLOW_SIGMA_M = CAUTION_ZONE_M / 2.15
RISK_GLOW_SIGMA_AUTHORIZED_M = CAUTION_ZONE_AUTHORIZED_M / 2.15
                          # Sigma for the CONTINUOUS risk value (compute_risk_value / risk_score
                          # in world.yaml / the heat map glow) -- picked so the value is ~1.0 at
                          # the robot, still clearly hot at the respective DANGER_ZONE, and fades
                          # to a dim ~0.1 near the respective CAUTION_ZONE. Same boundaries as
                          # above, just smooth instead of stepped -- retune together.

RADAR_RANGE_M = 3.0     # meters -- how far out the top-down radar window displays.
                          # People beyond this are still tracked/authorized normally,
                          # they just won't be plotted until they come within range.
RADAR_SIZE_PX = 520      # pixels -- radar window is a RADAR_SIZE_PX x RADAR_SIZE_PX square

GROUND_SQUASH_FACTOR = 0.55  # 0.0-1.0 -- how flat the on-camera-feed safety ellipse looks
                               # (see draw_safety_rings). This is a VISUAL APPROXIMATION on
                               # the main camera view only, not a true floor projection -- the
                               # radar window above is the geometrically honest boundary view.
                               # Tune by eye against your actual camera mount if you use it.

# ==========================================================================

# ----------------------------- CONFIG -----------------------------------

WIDTH, HEIGHT, FPS = 640, 480, 30
UPDATE_INTERVAL = 1.0
DRAW_GRACE_SEC = 1.0  # DISPLAY ONLY -- how long to keep drawing an object at its last known
                       # position after it stops being redetected each exact frame. This is
                       # purely cosmetic (smooths out the visible flicker in the preview window
                       # caused by momentary misses on cluttered/small objects); it does NOT
                       # change what's in world.yaml or world_diff.yaml -- the registry already
                       # correctly retains objects and marks them invisible on its own, this
                       # just makes the live window less distracting to watch.
CONF_THRESHOLD = 0.15  # lowered from 0.25 to catch small/cluttered desk & shelf items --
                         # tradeoff: more false positives are possible now. If you start
                         # seeing objects that clearly aren't there, raise this back up.

SPATIAL_MATCH_STATIC = 0.15
SPATIAL_MATCH_DYNAMIC = 0.6
PERSON_MATCH_RADIUS = 0.55      # loosened from 0.4 -- that was too tight for natural body
                                  # movement (leaning toward a laptop/desk can shift the
                                  # estimated 3D position more than 0.4m in one frame-to-frame
                                  # step, causing spurious track loss/recreation even for someone
                                  # who never actually left). Safe to loosen now: the close-
                                  # together-strangers risk this radius originally guarded against
                                  # is independently covered by the immediate-demote-on-confident-
                                  # mismatch logic (see auth.check() call site) -- even if a wrong
                                  # detection briefly claims a track, the next face check corrects
                                  # it within ~AUTH_CHECK_INTERVAL_SEC, not several seconds later.
STATIC_REACQUIRE_RADIUS = 0.35  # wider radius used ONLY when re-matching a static object that
                                  # was just occluded/lost (visible=False last frame). We already
                                  # know exactly where it is with high confidence, so a slightly
                                  # imprecise depth reading right as it reappears (partial
                                  # occlusion, motion blur from whoever walked past it) shouldn't
                                  # be enough to spawn a duplicate track. Continuously-visible
                                  # static objects still use the tight SPATIAL_MATCH_STATIC radius
                                  # so two distinct nearby objects of the same class don't merge.
STATIC_CONFIRM_TIME = 6.0   # raised from 3.0 -- was too eager to lock something as static right
                              # as the flat position-diff threshold below got retired in favor of
                              # zone-crossing; this alone is now the only thing standing between a
                              # brand-new detection and getting frozen, so it needs more margin.
STATIC_JITTER = 0.05
HYSTERESIS_FRAMES = 5

# Face auth thresholds -- FAIL CLOSED. Higher = stricter = fewer false
# authorizations, more "unknown, keep distance" false negatives. Tune
# this UP if a stranger ever gets matched to a known worker. Never tune
# it down without physically testing against the exact people involved.
AUTH_MATCH_THRESHOLD = 0.55        # cosine similarity needed to call it a match
AUTH_CHECK_INTERVAL_SEC = 1.0      # how often (wall-clock) to re-verify a tracked person's face

# Authorization no longer decays on a pure timer. A continuously-tracked
# person (same spatial track, never lost) keeps their authorization even
# if their face isn't re-confirmable for a while -- e.g. they turned
# sideways or looked down. Spatial track continuity IS the evidence that
# it's still the same physical person; re-confirming the face every
# single interval was never actually necessary for that case, and was
# the reason a side profile pose caused a false "unauthorized" demotion.
#
# Authorization is dropped in exactly two situations instead:
#   1. IMMEDIATELY, if a face check runs, a face IS found, and it
#      confidently does NOT match any known worker (handled where
#      auth.check() is called -- unrelated to this constant).
#   2. After the track has been continuously LOST (not just facing away
#      -- actually undetected, e.g. walked out of frame / fully
#      occluded) for longer than AUTH_LOST_GRACE_SEC below.
AUTH_LOST_GRACE_SEC = 6.0          # how long a track can be genuinely invisible (not just
                                     # turned away) before its authorization is dropped

# --- Clothing-signature reconnection bridge -----------------------------
# IMPORTANT SCOPE: clothing similarity NEVER grants authorization on its
# own. It only helps correctly ROUTE a reappearing person back to their
# own already-earned, still-valid authorization record -- the one that's
# already sitting there with authorized=True during the AUTH_LOST_GRACE_
# SEC window above. Without this, someone who reappears from a different
# spot/angle than where they were lost can fail the spatial match and
# get a brand-new (unauthorized) track created, even though their real
# record is still valid and just sitting unused. This closes that gap;
# it does not create a new way to become authorized without a face match
# ever having happened.
CLOTHING_MATCH_THRESHOLD = 0.65    # HSV histogram correlation (0-1) needed to treat a
                                     # reappearing person as the same one who just left
CLOTHING_HIST_EMA_ALPHA = 0.8      # how much weight the OLD clothing signature keeps each
                                     # time it's refreshed on a fresh face confirmation

KNOWN_WORKERS_YAML = "known_workers.yaml"
OUTPUT_YAML = "world.yaml"              # full state snapshot -- send this to the LLM on the FIRST call
OUTPUT_DIFF_YAML = "world_diff.yaml"    # delta since the last write -- send this on every call AFTER the first

# Fields diffed per person/object, and how much a numeric field has to
# change before it counts as a real change worth reporting. Velocity,
# confidence, and last_seen are intentionally excluded from the diff --
# they fluctuate constantly even when nothing meaningful happened, and
# including them would make the diff file basically as big as the full
# state every cycle, defeating the point.
# POSITION_DIFF_THRESHOLD is RETIRED -- every class now uses the same
# zone-crossing rule that used to be person-only (see compute_zone() and
# compute_and_write_diff()). A stationary box that gets kicked into
# caution/danger reports immediately, exactly like a person would; a box
# sitting untouched produces no diff at all, same as before. Kept here,
# commented, only as a reminder this concept existed if a supplementary
# raw-distance fallback is ever wanted again.
# POSITION_DIFF_THRESHOLD = 0.1  # meters
VELOCITY_DIFF_THRESHOLD = 0.4   # m/s -- now applies to EVERY class (previously person-only).
                                 # Raised from an earlier 0.15 -- that was too sensitive to
                                 # ordinary positional sensor jitter (a few cm of frame-to-frame
                                 # noise alone produces ~0.1-0.15 m/s of apparent velocity even
                                 # when something is standing/sitting still), which was firing
                                 # diffs on noise, not real movement. 0.4 m/s is comfortably above
                                 # jitter and still well below normal walking speed (~1+ m/s), so
                                 # a genuine approach -- or a shove -- still gets caught.
DIFFED_FIELDS = ["static", "visible", "authorized", "name"]

# Proximity zone constants (DANGER_ZONE_M, CAUTION_ZONE_M) now live in the
# SAFETY ZONE / BOUNDARY CONFIG block at the top of this file. Zone now
# applies to EVERY tracked class, not just people -- a chair or a box is
# just as capable of ending up in the danger zone (someone kicks it,
# knocks it, or the robot drives toward it) as a person is. An object's
# position is reported in the diff ONLY when it crosses from one zone
# into another -- not on every small move within the same zone. Zone is
# ALSO now what gates whether an object is even allowed to freeze as
# "static" -- see WorldObject.update() / _check_static() -- so something
# sitting in caution/danger is never allowed to go stale in world.yaml.


def compute_zone(position, authorized=False):
    """authorized=True (only ever meaningful for a person) uses the
    tighter DANGER_ZONE_AUTHORIZED_M/CAUTION_ZONE_AUTHORIZED_M radii
    instead of the default ones -- trusted, but never a zero-risk
    'safe no matter how close' pass. Objects and unauthorized people
    always use the default (larger) radii."""
    dist = float(np.linalg.norm(position))
    danger = DANGER_ZONE_AUTHORIZED_M if authorized else DANGER_ZONE_M
    caution = CAUTION_ZONE_AUTHORIZED_M if authorized else CAUTION_ZONE_M
    if dist < danger:
        return "danger"
    elif dist < caution:
        return "caution"
    return "safe"


def compute_risk_value(position, authorized=False):
    """Continuous 0-1 companion to compute_zone(), using the SAME
    authorization-aware sigma (RISK_GLOW_SIGMA_M or _AUTHORIZED_M) that
    drives the heat map glow -- so a person/object's numeric risk_score
    in world.yaml is always consistent with both its discrete 'zone' and
    what you'd see glowing on the heat map. ~1.0 right at the robot,
    still high at the respective DANGER_ZONE, fading to ~0.1 near the
    respective CAUTION_ZONE, ~0 well beyond it."""
    dist = float(np.linalg.norm(position))
    sigma = RISK_GLOW_SIGMA_AUTHORIZED_M if authorized else RISK_GLOW_SIGMA_M
    return float(np.exp(-(dist ** 2) / (2 * sigma ** 2)))

VOCAB = [
    "person", "robot arm", "mobile robot", "table", "chair", "box",
    "bottle", "cup", "spray can", "cylinder", "tool", "wire", "cable",
    "laptop", "monitor", "keyboard", "backpack", "book", "sensor",
    "drone", "camera", "door", "plant", "bag", "container", "wheel",
    "gripper", "screwdriver", "battery", "controller", "marker", "tape",
    "pipe", "valve", "switch", "light", "fan", "motor", "pump", "conveyor",
    "rack", "shelf", "panel", "frame", "hose", "tube", "rod", "bar",
    "plate", "sheet", "block", "brick", "stone", "rock", "cone", "pyramid",
    "sphere", "ring", "disk", "gear", "pulley", "spring", "coil", "magnet",
    "cable tie", "trolley", "cart", "bin", "bucket", "tray", "basket",
    "crate", "pallet", "barrel", "drum", "tank", "canister", "vessel",
    "flask", "beaker", "test tube", "pipette", "microscope", "telescope",
    "camera lens", "projector", "screen",
    # -- desk/office clutter (previously missing, caused low recall) --
    "computer mouse", "mouse pad", "tissue box", "first aid box",
    "stack of papers", "notebook", "notepad", "sticky notes", "pen",
    "pencil", "pen holder", "smartphone", "mobile phone", "charger",
    "phone charger", "power strip", "extension cord", "power adapter",
    "drawer", "monitor stand", "whiteboard", "clipboard", "sticker",
    "cardboard box", "cardboard", "styrofoam", "scissors", "cutter",
    "ruler", "id card", "lanyard", "wallet", "key", "keys", "wristwatch",
    "eyeglasses", "cap", "cloth", "fabric", "bag of cloth",
    # -- electronics/robotics workbench clutter --
    "circuit board", "pcb", "breadboard", "resistor", "capacitor",
    "wire spool", "soldering iron", "multimeter", "remote control",
    "small motor", "servo motor", "electronic module", "arduino board",
    "esp32 board", "usb cable", "usb drive", "sd card", "connector",
    "jumper wires", "toolbox", "tool kit", "calculator", "adapter plug",
    "power bank", "led strip", "circuit component"
]

# --------------------------------------------------------------------------


class WorkerAuth:
    """Loads the enrolled worker face embeddings and answers
    'is this face one of the authorized workers'. Read-only at runtime --
    enrollment happens separately via enroll_worker.py."""

    def __init__(self, registry_path=KNOWN_WORKERS_YAML, threshold=AUTH_MATCH_THRESHOLD):
        self.threshold = threshold
        try:
            with open(registry_path) as f:
                data = yaml.safe_load(f) or {}
            self.known = {k: np.array(v) for k, v in data.items()}
        except FileNotFoundError:
            self.known = {}
        if not self.known:
            print(f"WARNING: {registry_path} not found or empty. "
                  f"Every person will be treated as UNAUTHORIZED. "
                  f"Run enroll_worker.py first.")
        self.app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        # Smaller det_size than enrollment (320) -- this runs every
        # AUTH_CHECK_INTERVAL_SEC during live tracking, so it needs to be
        # fast. Enrollment only runs a handful of times, so it keeps the
        # higher-quality 320 for a better averaged embedding. The output
        # embedding is always 512-d regardless of det_size, so this
        # mismatch doesn't hurt match quality, just detection speed.
        self.app.prepare(ctx_id=0, det_size=(160, 160))

    def check(self, person_crop_bgr):
        """Returns (authorized: bool, name: str|None, face_found: bool).

        face_found matters: if a face WAS detected but confidently
        didn't match anyone known, that's a real signal ("this is
        definitely not an enrolled worker") and the caller should act
        on it immediately. If no face was found at all (bad angle,
        too far, motion blur), that's just a missed reading, not
        evidence of anything -- the caller should NOT treat it the
        same way.
        """
        if person_crop_bgr.size == 0 or not self.known:
            return False, None, False
        faces = self.app.get(person_crop_bgr)
        if not faces:
            return False, None, False
        emb = faces[0].normed_embedding

        best_name, best_score = None, -1.0
        for name, known_emb in self.known.items():
            score = float(np.dot(emb, known_emb))
            if score > best_score:
                best_score, best_name = score, name

        if best_score >= self.threshold:
            return True, best_name, True
        return False, None, True


class WorldObject:
    _counters = {}

    def __init__(self, cls_name, position, dims, confidence, fixed_id=None):
        if fixed_id is not None:
            self.id = fixed_id
        else:
            idx = WorldObject._counters.get(cls_name, 0) + 1
            WorldObject._counters[cls_name] = idx
            self.id = f"{cls_name.replace(' ', '_')}_{idx}"

        self.cls_name = cls_name
        self.position = np.array(position, dtype=float)
        self.dims = dims
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.confidence = confidence
        self.position_history = [(time.time(), self.position.copy())]
        self.static = False
        self.static_locked_position = None
        self.flip_counter = 0
        self.visible = True
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.last_update_time = time.time()

        # Authorization state -- FAILS CLOSED by default.
        self.authorized = False
        self.worker_name = None
        self.last_auth_check = 0.0
        self.invisible_since = None  # timestamp the track was LAST seen going
                                       # visible->invisible; None while visible
        self.zone = compute_zone(self.position, self.authorized)  # now tracked for every class, not just person
        self.risk_score = compute_risk_value(self.position, self.authorized)  # continuous companion to zone
        self.clothing_hist = None  # only ever set on a face-CONFIRMED authorization

    def predicted_position(self):
        if self.static and self.static_locked_position is not None:
            return self.static_locked_position
        dt = time.time() - self.last_update_time
        return self.position + self.velocity * dt

    def update(self, position, dims, confidence):
        now = time.time()
        dt = max(now - self.last_update_time, 1e-3)
        raw_position = np.array(position, dtype=float)

        # Zone is computed from the RAW live reading, every frame, for
        # EVERY class -- including something already static-locked. This
        # is what keeps a frozen object's zone accurate even if the
        # robot itself moves and changes the true distance, and it's
        # what lets the force-unlock below react the instant things get
        # risky, rather than only next time _check_static() happens to run.
        self.zone = compute_zone(raw_position, self.authorized)
        self.risk_score = compute_risk_value(raw_position, self.authorized)

        # Zone-gated freezing: nothing is allowed to STAY static-locked
        # once it's outside 'safe'. A person standing close, or a chair
        # someone might kick, could move at any instant -- freezing its
        # position would risk masking a real, sudden change from a
        # downstream safety consumer. Force-unlock immediately, no
        # hysteresis delay, the moment zone leaves 'safe'.
        if self.static and self.zone != "safe":
            self.static = False
            self.static_locked_position = None
            self.flip_counter = 0

        if self.static and self.static_locked_position is not None:
            displacement = np.linalg.norm(raw_position - self.static_locked_position)
            if displacement > STATIC_JITTER * 3:
                self.flip_counter += 1
                if self.flip_counter < HYSTERESIS_FRAMES:
                    final_position = self.static_locked_position  # ignore this reading, not confirmed movement yet
                else:
                    self.static = False
                    self.static_locked_position = None
                    self.flip_counter = 0
                    final_position = raw_position
            else:
                # Genuinely still static -- FREEZE the position exactly,
                # don't blend in noisy readings. This is what keeps a
                # static object's entry in world.yaml byte-identical
                # between writes until it actually moves, instead of
                # drifting by a millimeter every 0.5s from sensor noise.
                self.flip_counter = 0
                final_position = self.static_locked_position
        else:
            final_position = raw_position

        self.velocity = (final_position - self.position) / dt
        self.position = np.array(final_position, dtype=float)
        self.dims = dims
        self.confidence = confidence

        self.position_history.append((now, self.position.copy()))
        self.position_history = [(t, p) for t, p in self.position_history if now - t < STATIC_CONFIRM_TIME + 1]

        self._check_static(now)

        self.last_update_time = now
        self.last_seen = now
        self.visible = True
        self.invisible_since = None

    def apply_auth_result(self, authorized, name, now):
        """Caller should only invoke this with authorized=False when a
        face WAS confidently read and did NOT match any known worker --
        not simply because no face was visible this check. That
        distinction is what lets this demote immediately on a real
        mismatch (e.g. the wrong person's detection got assigned to
        this track), while a missed/bad-angle read on the correct
        person (still spatially tracked, just facing away) is left
        alone entirely -- see AUTH_LOST_GRACE_SEC / expire_stale_
        authorizations for the only other path that can demote."""
        self.authorized = authorized
        self.worker_name = name if authorized else None
        self.last_auth_check = now

    def mark_invisible(self):
        # Only stamp invisible_since on the FIRST frame a previously-
        # visible track goes missing -- this is what lets us measure how
        # long the track has been genuinely lost, not just how long ago
        # it was last reconfirmed.
        if self.visible and self.invisible_since is None:
            self.invisible_since = time.time()
        self.visible = False

    def _check_static(self, now):
        if self.static:
            return
        if self.zone != "safe":
            return  # never allowed to lock while in caution/danger -- see update()
        if now - self.first_seen < STATIC_CONFIRM_TIME:
            return
        if len(self.position_history) < 3:
            return
        positions = np.array([p for _, p in self.position_history])
        spread = np.max(np.linalg.norm(positions - positions.mean(axis=0), axis=1))
        if spread < STATIC_JITTER:
            self.static = True
            self.static_locked_position = positions.mean(axis=0)
            self.velocity = np.array([0.0, 0.0, 0.0])
            self.flip_counter = 0

    def to_dict(self):
        pos = self.static_locked_position if self.static_locked_position is not None else self.position
        d = {
            "id": self.id,
            "class": self.cls_name,
            "position": [round(float(v), 2) for v in pos],
            "dimensions": [round(float(v), 2) for v in self.dims],
            "velocity": [round(float(v), 2) for v in self.velocity],
            "static": bool(self.static),
            "visible": bool(self.visible),
            "confidence": round(float(self.confidence), 2),
            "last_seen": round(time.time() - self.last_seen, 1),
            "zone": self.zone,  # 'danger' / 'caution' / 'safe' -- now for every class
            "risk_score": round(float(self.risk_score), 3),  # continuous 0-1 companion to zone
        }
        if self.cls_name == "person":
            d["authorized"] = bool(self.authorized)
            d["name"] = self.worker_name  # None if unauthorized/unknown
        return d


class WorldRegistry:
    def __init__(self):
        self.objects = {}
        self._last_snapshot = {}  # obj_id -> to_dict() as of the last diff write

    def resolve_person_identity(self, obj, name):
        """Call this right after a face-authorization match succeeds.

        The numeric id (person_14, person_21, ...) is assigned purely
        from spatial tracking and has nothing to do with WHO the person
        is -- if a track is lost (occlusion, walking out of frame) and
        the person reappears, spatial matching alone creates a brand
        new id. This is what caused 'Shariq' to jump from person_14 to
        person_21.

        Fix: once we know a face belongs to 'name', check whether ANY
        other tracked person object (even currently invisible) already
        has that same worker_name. If so, that's the same real person's
        original identity -- merge this fresh track's telemetry into
        the ORIGINAL object and discard the duplicate, instead of
        letting a second id exist for the same person.
        """
        if name is None:
            return obj
        for other_id, other in list(self.objects.items()):
            if other is obj:
                continue
            if other.cls_name == "person" and other.worker_name == name:
                other.position = obj.position
                other.velocity = obj.velocity
                other.dims = obj.dims
                other.confidence = obj.confidence
                other.position_history = obj.position_history
                other.static = False
                other.static_locked_position = None
                other.flip_counter = 0
                other.visible = True
                other.last_seen = obj.last_seen
                other.last_update_time = obj.last_update_time
                other.authorized = True
                other.worker_name = name
                other.last_auth_check = obj.last_auth_check
                if obj.id in self.objects:
                    del self.objects[obj.id]
                return other
        return obj

    def try_clothing_reconnect(self, obj, histogram, now):
        """Call this for a BRAND NEW person track, before/alongside the
        face check. If this new detection's clothing closely matches
        someone who was AUTHORIZED and is currently within their
        AUTH_LOST_GRACE_SEC window (i.e. genuinely lost, but their
        authorization record hasn't expired yet), merge into that
        original record instead of leaving this as an orphan new
        unauthorized track.

        SCOPE, IMPORTANT: this can only ever reconnect to a record that
        ALREADY has authorized=True from a real face match. It never
        sets authorized=True on its own, never creates new trust, and
        never touches a record whose grace window has already expired
        (those are correctly gone). It only fixes mis-routing.
        """
        if histogram is None:
            return obj
        best_match, best_score = None, CLOTHING_MATCH_THRESHOLD
        for other_id, other in self.objects.items():
            if other is obj or other.cls_name != "person":
                continue
            if not other.authorized or other.clothing_hist is None:
                continue
            if other.invisible_since is None:
                continue  # still visible/tracked elsewhere -- not a reconnection case
            if (now - other.invisible_since) > AUTH_LOST_GRACE_SEC:
                continue  # outside the trust window -- about to (or already) expire, don't touch
            score = clothing_histogram_similarity(other.clothing_hist, histogram)
            if score > best_score:
                best_match, best_score = other, score

        if best_match is None:
            return obj

        best_match.position = obj.position
        best_match.velocity = obj.velocity
        best_match.dims = obj.dims
        best_match.confidence = obj.confidence
        best_match.position_history = obj.position_history
        best_match.static = False
        best_match.static_locked_position = None
        best_match.flip_counter = 0
        best_match.visible = True
        best_match.invisible_since = None
        best_match.last_seen = obj.last_seen
        best_match.last_update_time = obj.last_update_time
        best_match.zone = obj.zone
        # authorized / worker_name / clothing_hist / last_auth_check are
        # left as best_match's own values -- that trust was already
        # earned by an actual face match, we're only correcting routing.
        if obj.id in self.objects:
            del self.objects[obj.id]
        return best_match

    def process_frame_detections(self, person_detections, object_detections):
        """Match ALL of this frame's detections against tracks at once,
        using each track's position as it was BEFORE this frame's
        updates.

        Matching is done as a GLOBAL nearest-pair-first assignment, not
        detection-by-detection in whatever order YOLO happened to emit
        boxes. Previously, each detection independently grabbed the
        nearest still-available track; when two people stood close
        together, whichever detection was processed FIRST (arbitrary
        order) could claim a track that actually belonged to the OTHER
        person, if that other person's own detection hadn't been looked
        at yet. Sorting every valid (detection, track) pair by distance
        and assigning the closest pairs first removes that order
        dependency -- the overall best matches win regardless of which
        detection came first in the list.

        person_detections: list of (position, dims, confidence) tuples.
        object_detections: list of (cls_name, position, dims, confidence).

        Returns: list of (WorldObject, is_new_person_track) in the same
        order as person_detections, followed by the object matches in
        order as object_detections.
        """
        # Snapshot predicted positions BEFORE any mutation this frame.
        snapshot = {obj_id: obj.predicted_position() for obj_id, obj in self.objects.items()}
        claimed_tracks = set()

        # ---------------------------- people ----------------------------
        people_ids = [oid for oid, o in self.objects.items() if o.cls_name == "person"]
        pairs = []
        for det_idx, (position, dims, confidence) in enumerate(person_detections):
            for oid in people_ids:
                dist = np.linalg.norm(np.array(position) - snapshot[oid])
                if dist < PERSON_MATCH_RADIUS:
                    pairs.append((dist, det_idx, oid))
        pairs.sort(key=lambda p: p[0])

        det_to_track = {}
        for dist, det_idx, oid in pairs:
            if det_idx in det_to_track or oid in claimed_tracks:
                continue
            det_to_track[det_idx] = oid
            claimed_tracks.add(oid)

        person_results = []
        for det_idx, (position, dims, confidence) in enumerate(person_detections):
            oid = det_to_track.get(det_idx)
            if oid is not None:
                obj = self.objects[oid]
                obj.update(position, dims, confidence)
                person_results.append((obj, False))
            else:
                new_obj = WorldObject("person", position, dims, confidence)
                self.objects[new_obj.id] = new_obj
                claimed_tracks.add(new_obj.id)
                person_results.append((new_obj, True))

        # --------------------------- objects -----------------------------
        pairs = []
        for det_idx, (cls_name, position, dims, confidence) in enumerate(object_detections):
            for oid, obj in self.objects.items():
                if obj.cls_name != cls_name:
                    continue
                if obj.static:
                    radius = SPATIAL_MATCH_STATIC if obj.visible else STATIC_REACQUIRE_RADIUS
                else:
                    radius = SPATIAL_MATCH_DYNAMIC
                dist = np.linalg.norm(np.array(position) - snapshot[oid])
                if dist < radius:
                    pairs.append((dist, det_idx, oid))
        pairs.sort(key=lambda p: p[0])

        det_to_track = {}
        for dist, det_idx, oid in pairs:
            if det_idx in det_to_track or oid in claimed_tracks:
                continue
            det_to_track[det_idx] = oid
            claimed_tracks.add(oid)

        object_results = []
        for det_idx, (cls_name, position, dims, confidence) in enumerate(object_detections):
            oid = det_to_track.get(det_idx)
            if oid is not None:
                obj = self.objects[oid]
                obj.update(position, dims, confidence)
                object_results.append(obj)
            else:
                new_obj = WorldObject(cls_name, position, dims, confidence)
                self.objects[new_obj.id] = new_obj
                claimed_tracks.add(new_obj.id)
                object_results.append(new_obj)

        return person_results, object_results

    def end_frame(self, matched_ids_this_frame):
        for obj_id, obj in self.objects.items():
            if obj_id not in matched_ids_this_frame:
                obj.mark_invisible()

    def expire_stale_authorizations(self, now):
        """Call once per frame, for the whole registry. Demotes an
        authorized person ONLY if their track has been continuously,
        genuinely invisible (walked out of frame / fully occluded) for
        longer than AUTH_LOST_GRACE_SEC. A person who's still being
        spatially tracked every frame -- just facing away from the
        camera -- is NEVER touched by this, no matter how long it's
        been since their face was last reconfirmed."""
        for obj in self.objects.values():
            if obj.cls_name != "person" or not obj.authorized:
                continue
            if obj.invisible_since is not None and (now - obj.invisible_since) > AUTH_LOST_GRACE_SEC:
                obj.authorized = False
                obj.worker_name = None

    def write_yaml(self, path=OUTPUT_YAML):
        # Sorted by (class, id) so the file has a stable, predictable
        # order every write -- otherwise dict iteration order jumps
        # around as objects get created/updated in whatever order YOLO
        # happened to emit detections that frame.
        ordered = sorted(self.objects.values(), key=lambda o: (o.cls_name, o.id))
        world = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "objects": [o.to_dict() for o in ordered],
        }
        # default_flow_style=None -> PyYAML only uses compact "[a, b, c]"
        # flow style for collections that are PURE SCALARS (position,
        # dimensions, velocity), and block style (one field per line) for
        # each object's dict. Was default_flow_style=True (flow style
        # for EVERYTHING, including the object dicts) -- that saved a
        # few tokens but crammed the whole file onto one or two
        # unreadable mega-lines, making it hard to eyeball for mistakes.
        # This is the readable middle ground: still compact where it
        # doesn't matter (number arrays), one line per field where it
        # does (so you can actually scan/diff an object at a glance).
        with open(path, "w") as f:
            yaml.dump(world, f, sort_keys=False, default_flow_style=None, width=1000)

    def compute_and_write_diff(self, path=OUTPUT_DIFF_YAML):
        """Compares the current state to the state as of the LAST call
        to this method, writes only what changed to world_diff.yaml,
        and updates the stored snapshot for next time.

        EVERY class (person or object): a position is only reported when
        its PROXIMITY ZONE changes (danger/caution/safe -- see
        compute_zone()), not on every small move within the same zone.
        Velocity spikes and authorized/name/static/visible changes still
        trigger independently, since those matter regardless of zone. A
        stationary chair that gets kicked into the danger zone reports
        immediately, exactly like a person would; one sitting untouched
        in 'safe' produces nothing, cycle after cycle.

        Usage in your LLM pipeline:
          - Send world.yaml (full state) on the very first planning call.
          - Send world_diff.yaml on every call after that.
          - Call this method every write cycle regardless -- it always
            advances the snapshot, so diffs stay incremental.
        """
        ordered = sorted(self.objects.values(), key=lambda o: (o.cls_name, o.id))
        current = {o.id: o.to_dict() for o in ordered}

        added, changed, removed = [], [], []

        for obj_id, cur in current.items():
            prev = self._last_snapshot.get(obj_id)
            if prev is None:
                added.append(cur)
                continue

            field_diffs = {}

            zone_changed = prev.get("zone") != cur.get("zone")
            vel_delta = np.linalg.norm(np.array(cur["velocity"]) - np.array(prev["velocity"]))
            velocity_changed = vel_delta > VELOCITY_DIFF_THRESHOLD
            other_changed = any(prev.get(f) != cur.get(f) for f in DIFFED_FIELDS)

            if zone_changed or velocity_changed or other_changed:
                # Something significant fired -- include position AND
                # risk_score too so the LLM has full spatial/risk context,
                # not just the field that happened to trip the trigger.
                # risk_score is NOT a trigger on its own -- it drifts by
                # tiny amounts every frame from ordinary jitter even when
                # nothing meaningful happened, so using it to fire a diff
                # would defeat the whole point of diffing. It only ever
                # rides along with a real trigger (zone/velocity/other).
                field_diffs["position"] = cur["position"]
                field_diffs["risk_score"] = cur["risk_score"]
                if zone_changed:
                    field_diffs["zone"] = cur["zone"]
                if velocity_changed:
                    field_diffs["velocity"] = cur["velocity"]
                for field in DIFFED_FIELDS:
                    if prev.get(field) != cur.get(field):
                        field_diffs[field] = cur[field]

            if field_diffs:
                changed.append({"id": obj_id, "class": cur["class"], **field_diffs})

        for obj_id in sorted(self._last_snapshot):
            if obj_id not in current:
                removed.append(obj_id)

        diff = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "added": added,
            "changed": changed,
            "removed": removed,
        }
        with open(path, "w") as f:
            yaml.dump(diff, f, sort_keys=False, default_flow_style=None, width=1000)

        self._last_snapshot = current
        return diff


# --------------------------- Perception helpers ---------------------------

def deproject_pixel_to_point(depth_frame, intrinsics, x, y):
    depth = depth_frame.get_distance(int(x), int(y))
    if depth <= 0:
        return None
    return rs.rs2_deproject_pixel_to_point(intrinsics, [float(x), float(y)], depth)


def estimate_dimensions(intrinsics, x1, y1, x2, y2, center_depth):
    if center_depth <= 0:
        return [0.1, 0.1, 0.1]
    width_m = ((x2 - x1) / intrinsics.fx) * center_depth
    height_m = ((y2 - y1) / intrinsics.fy) * center_depth
    depth_m = max(width_m, height_m) * 0.5
    return [round(width_m, 3), round(height_m, 3), round(depth_m, 3)]


def crop_box(color_image, x1, y1, x2, y2):
    h, w = color_image.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=np.uint8)
    return color_image[y1:y2, x1:x2]


def compute_clothing_histogram(color_image, x1, y1, x2, y2):
    """HSV histogram of the person's torso region (middle 50% of the
    bbox height, full width) -- torso clothing is more stable across
    frames than the whole bbox, which drags in floor/background/hair.
    This is used ONLY for reconnecting a lost track's own already-
    earned authorization -- see the CLOTHING_MATCH_THRESHOLD comment."""
    h, w = color_image.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    box_h = y2 - y1
    ty1, ty2 = y1 + int(box_h * 0.25), y1 + int(box_h * 0.75)
    if ty2 <= ty1:
        ty1, ty2 = y1, y2
    torso = color_image[ty1:ty2, x1:x2]
    if torso.size == 0:
        return None
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def clothing_histogram_similarity(hist_a, hist_b):
    return float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))


def real_radius_to_pixels(intrinsics, real_radius_m, depth_m):
    """How many pixels wide is a real_radius_m circle, at this depth,
    for this camera's focal length? Pinhole projection: apparent size
    is inversely proportional to distance. This is what makes the ring
    correctly shrink as the person gets farther and grow as they get
    closer -- it's a real safety radius, not a fixed decorative circle."""
    if depth_m <= 0.05:
        depth_m = 0.05  # avoid a blown-up circle if depth reads near-zero
    return int((real_radius_m * intrinsics.fx) / depth_m)


# def draw_dashed_ellipse(img, center, axes, color, thickness=2, dash_deg=14, gap_deg=10):
#     """cv2 has no built-in dashed ellipse -- approximate one with short
#     arcs. Used for the 'warning' zone ring (danger zone ring is drawn
#     solid instead, to visually read as more urgent)."""
#     if axes[0] <= 0 or axes[1] <= 0:
#         return
#     angle = 0.0
#     while angle < 360.0:
#         end = min(angle + dash_deg, 360.0)
#         cv2.ellipse(img, center, axes, 0, angle, end, color, thickness)
#         angle += dash_deg + gap_deg


# GROUND_SQUASH_FACTOR now lives in the SAFETY ZONE / BOUNDARY CONFIG
# block at the top of this file (0.0 = perfect circle, 1.0 = flattened
# to a line). This is a visual approximation on the camera feed only --
# not a true floor homography. See that block for the full explanation.


# def draw_safety_rings(img, intrinsics, foot_x, foot_y, depth_m):
#     """Draws both proximity rings anchored at the person's FEET (not
#     their body centroid) and squashed into ellipses so they visually
#     read as hugging the ground around the person, rather than floating
#     at chest height facing the camera. Radii are still real-world-scaled
#     via depth (see real_radius_to_pixels) using the SAME DANGER_ZONE_M /
#     CAUTION_ZONE_M constants that drive world_diff.yaml's zone-crossing
#     logic -- the visual and the actual safety-relevant output can never
#     disagree with each other, only the on-screen SHAPE is an approximation.
#     Solid red = no-go/danger radius. Dashed orange = warning/caution radius."""
#     danger_px = real_radius_to_pixels(intrinsics, DANGER_ZONE_M, depth_m)
#     caution_px = real_radius_to_pixels(intrinsics, CAUTION_ZONE_M, depth_m)
#     center = (int(foot_x), int(foot_y))

#     danger_axes = (danger_px, max(int(danger_px * (1 - GROUND_SQUASH_FACTOR)), 2))
#     caution_axes = (caution_px, max(int(caution_px * (1 - GROUND_SQUASH_FACTOR)), 2))

#     draw_dashed_ellipse(img, center, caution_axes, (0, 165, 255), thickness=2)  # caution -- dashed orange
#     cv2.ellipse(img, center, danger_axes, 0, 0, 360, (0, 0, 255), 2)            # danger -- solid red


def render_radar_view(people, intrinsics=None):
    """Top-down (bird's-eye) view of the safety zones, using the SAME
    real-world (x, depth) coordinates already computed for every person
    -- no camera calibration, height, or tilt needed, unlike the on-
    camera-feed ellipses in draw_safety_rings(). This is the
    geometrically HONEST boundary view: the danger/caution circles here
    are true, undistorted circles at their real radii, and every
    person's dot is plotted at their true real-world position relative
    to the robot. Meant for a separate, dedicated presentation window.

    people: list of WorldObject (cls_name == 'person'), whatever's
    currently visible this frame.
    """
    img = np.full((RADAR_SIZE_PX, RADAR_SIZE_PX, 3), 25, dtype=np.uint8)
    center_x, center_y = RADAR_SIZE_PX // 2, RADAR_SIZE_PX - 40  # robot near the bottom, facing "up"
    scale = (RADAR_SIZE_PX / 2) / RADAR_RANGE_M  # pixels per meter

    # Range rings (light grey, every 1m) for scale reference
    for r_m in range(1, int(RADAR_RANGE_M) + 1):
        cv2.circle(img, (center_x, center_y), int(r_m * scale), (60, 60, 60), 1)
        cv2.putText(img, f"{r_m}m", (center_x + int(r_m * scale) + 4, center_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (110, 110, 110), 1)

    # Optional camera field-of-view cone, purely for audience context
    if intrinsics is not None and intrinsics.fx > 0:
        half_fov = np.arctan((WIDTH / 2) / intrinsics.fx)
        for sign in (-1, 1):
            end_x = int(center_x + sign * RADAR_RANGE_M * scale * np.sin(half_fov))
            end_y = int(center_y - RADAR_RANGE_M * scale * np.cos(half_fov))
            cv2.line(img, (center_x, center_y), (end_x, end_y), (70, 70, 70), 1)

    # Safety zones -- TRUE circles, no perspective distortion. Two pairs:
    # the larger/default pair (unauthorized people + all objects) in
    # red/orange, and the tighter pair that applies ONLY to an
    # AUTHORIZED person in blue/cyan -- drawn thinner so it visually
    # reads as "the smaller version", not a second/duplicate boundary.
    cv2.circle(img, (center_x, center_y), int(CAUTION_ZONE_M * scale), (0, 165, 255), 2)
    cv2.circle(img, (center_x, center_y), int(DANGER_ZONE_M * scale), (0, 0, 255), 2)
    cv2.circle(img, (center_x, center_y), int(CAUTION_ZONE_AUTHORIZED_M * scale), (255, 220, 0), 1)
    cv2.circle(img, (center_x, center_y), int(DANGER_ZONE_AUTHORIZED_M * scale), (255, 0, 0), 1)
    cv2.putText(img, "red/orange = unauthorized+objects, blue/cyan = authorized",
                (10, RADAR_SIZE_PX - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    # Robot icon (triangle, pointing "up" / forward)
    tri = np.array([[center_x, center_y - 14], [center_x - 10, center_y + 10],
                     [center_x + 10, center_y + 10]], dtype=np.int32)
    cv2.fillPoly(img, [tri], (200, 200, 200))
    cv2.putText(img, "ROBOT", (center_x - 28, center_y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # Plot every person
    for obj in people:
        x_m = float(obj.position[0])   # lateral (left/right)
        z_m = float(obj.position[2])   # depth (forward distance from robot)
        px = int(center_x + x_m * scale)
        py = int(center_y - z_m * scale)
        if 0 <= px < RADAR_SIZE_PX and 0 <= py < RADAR_SIZE_PX:
            color = (0, 255, 0) if obj.authorized else (0, 0, 255)
            cv2.circle(img, (px, py), 8, color, -1)
            label = obj.worker_name if obj.authorized else obj.id
            cv2.putText(img, label, (px + 10, py + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.putText(img, "Safety Zone Radar (top-down)", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
    return img


# RISK_GLOW_SIGMA_M now lives in the SAFETY ZONE / BOUNDARY CONFIG block
# at the top of this file, alongside DANGER_ZONE_M / CAUTION_ZONE_M -- it
# drives both this heat map's glow AND the risk_score field in world.yaml,
# so the two can never disagree with each other.


def _build_risk_field(registry, grid_n=130):
    """Shared Gaussian risk field, used by BOTH render_risk_heatmap and
    render_probability_map below -- so the glow you see and the percent
    numbers you see are guaranteed to be reading off the exact same
    values, never two independently-computed versions that could drift
    apart. Only currently-dynamic (obj.static is False), visible objects
    contribute -- see render_risk_heatmap's docstring for why. Combined
    with np.maximum across objects, not addition, for the same reason.

    Returns the (grid_n, grid_n) field array (0-1) plus the grid's
    meter-space coordinate arrays (x_m, z_m), same coordinate frame as
    render_radar_view (x lateral, z depth, robot at the origin)."""
    size = RADAR_SIZE_PX
    scale_g = (grid_n / 2) / RADAR_RANGE_M       # grid-cells per meter
    center_gx, center_gy = grid_n // 2, grid_n - int(40 * grid_n / size)
    gx_idx, gy_idx = np.meshgrid(np.arange(grid_n), np.arange(grid_n))
    x_m = (gx_idx - center_gx) / scale_g          # lateral (left/right)
    z_m = (center_gy - gy_idx) / scale_g          # depth (forward from robot)

    field = np.zeros((grid_n, grid_n), dtype=np.float32)
    for obj in registry.objects.values():
        if not obj.visible or obj.static:
            continue  # only currently-dynamic things contribute
        ox, oz = float(obj.position[0]), float(obj.position[2])
        sigma = RISK_GLOW_SIGMA_AUTHORIZED_M if obj.authorized else RISK_GLOW_SIGMA_M
        dist2 = (x_m - ox) ** 2 + (z_m - oz) ** 2
        contribution = np.exp(-dist2 / (2 * sigma ** 2)).astype(np.float32)
        field = np.maximum(field, contribution)
    return field, x_m, z_m


def render_risk_heatmap(registry, intrinsics=None):
    """A DIFFERENT window from render_radar_view above. That one draws
    exact zone rings + a dot per person -- clean, discrete, good for
    reading an exact distance. This one is a genuine risk HEAT MAP: a
    smooth glow, built ONLY from objects that are currently DYNAMIC
    (obj.static is False) -- anything static-locked is, by construction
    (see WorldObject._check_static()/update()), sitting safely in the
    'safe' zone and frozen, so it isn't a live risk and doesn't compete
    visually with what's actually capable of hurting someone right now.
    Includes every class equally (person, chair, box, ...) -- not just
    people -- since zone/risk now applies to all of them the same way.

    See render_probability_map() below for the same field, but with
    actual numeric percentages, a legend, and contour lines -- this one
    is deliberately just the raw glow for a fast, at-a-glance read.

    Same coordinate frame, RADAR_RANGE_M/RADAR_SIZE_PX scale, and robot
    placement as render_radar_view, so the two windows line up visually
    if you put them side by side.
    """
    size = RADAR_SIZE_PX
    center_x, center_y = size // 2, size - 40
    scale = (size / 2) / RADAR_RANGE_M  # pixels per meter

    field, _, _ = _build_risk_field(registry)

    norm = np.clip(field * 255, 0, 255).astype(np.uint8)
    norm = cv2.resize(norm, (size, size), interpolation=cv2.INTER_CUBIC)
    img = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

    # Reference rings + robot marker, drawn in white so they read clearly
    # against the JET colormap regardless of what's underneath them.
    for r_m in range(1, int(RADAR_RANGE_M) + 1):
        cv2.circle(img, (center_x, center_y), int(r_m * scale), (255, 255, 255), 1)
    cv2.circle(img, (center_x, center_y), int(CAUTION_ZONE_M * scale), (255, 255, 255), 1)
    cv2.circle(img, (center_x, center_y), int(DANGER_ZONE_M * scale), (255, 255, 255), 2)
    cv2.circle(img, (center_x, center_y), int(CAUTION_ZONE_AUTHORIZED_M * scale), (255, 255, 255), 1)
    cv2.circle(img, (center_x, center_y), int(DANGER_ZONE_AUTHORIZED_M * scale), (255, 255, 255), 1)
    tri = np.array([[center_x, center_y - 14], [center_x - 10, center_y + 10],
                     [center_x + 10, center_y + 10]], dtype=np.int32)
    cv2.fillPoly(img, [tri], (255, 255, 255))
    cv2.putText(img, "Risk Heat Map (dynamic only, top-down)", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


# Iso-probability contour lines drawn on the probability map, and their
# labels (as a percentage). PLACEHOLDER set -- add/remove levels freely.
PROBABILITY_CONTOUR_LEVELS = [0.10, 0.25, 0.50, 0.75, 0.90]


def render_probability_map(registry, intrinsics=None):
    """A THIRD top-down window, alongside render_radar_view (exact zone
    rings + a dot per person) and render_risk_heatmap (the raw glow).
    This one takes the EXACT SAME Gaussian field as the heat map (see
    _build_risk_field -- shared, not recomputed, so the two can never
    disagree) and makes the actual probability VALUES readable instead
    of just a color gradient:
      - a colorbar legend on the right mapping color -> percent
      - labeled iso-probability contour lines (PROBABILITY_CONTOUR_LEVELS)
      - a live risk-percentage readout next to every dynamic object
    """
    size = RADAR_SIZE_PX
    center_x, center_y = size // 2, size - 40
    scale = (size / 2) / RADAR_RANGE_M

    field, _, _ = _build_risk_field(registry)

    norm_small = np.clip(field * 255, 0, 255).astype(np.uint8)
    norm = cv2.resize(norm_small, (size, size), interpolation=cv2.INTER_CUBIC)
    img = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

    # Contours computed on the upscaled (full-size) field so the lines
    # come out smooth instead of blocky from the coarse evaluation grid.
    field_full = cv2.resize(field, (size, size), interpolation=cv2.INTER_CUBIC)
    for level in PROBABILITY_CONTOUR_LEVELS:
        mask = (field_full >= level).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cv2.drawContours(img, contours, -1, (255, 255, 255), 1)
        biggest = max(contours, key=cv2.contourArea)
        top_point = tuple(biggest[biggest[:, :, 1].argmin()][0])
        label_y = max(int(top_point[1]) - 6, 12)
        cv2.putText(img, f"{int(level * 100)}%", (int(top_point[0]) - 10, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    tri = np.array([[center_x, center_y - 14], [center_x - 10, center_y + 10],
                     [center_x + 10, center_y + 10]], dtype=np.int32)
    cv2.fillPoly(img, [tri], (255, 255, 255))

    # Live percentage label at every object contributing to the field --
    # reuses obj.risk_score directly (same formula, already
    # authorization-aware) rather than resampling the grid at a point.
    for obj in registry.objects.values():
        if not obj.visible or obj.static:
            continue
        px = int(center_x + float(obj.position[0]) * scale)
        py = int(center_y - float(obj.position[2]) * scale)
        if 0 <= px < size and 0 <= py < size:
            cv2.circle(img, (px, py), 4, (255, 255, 255), -1)
            cv2.putText(img, f"{obj.id} {obj.risk_score * 100:.0f}%", (px + 8, py + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Colorbar legend (color -> probability %) pasted onto a wider canvas.
    bar_w, margin = 40, 70
    canvas = np.full((size, size + bar_w + margin, 3), 25, dtype=np.uint8)
    canvas[:, :size] = img
    grad = np.linspace(255, 0, size, dtype=np.uint8).reshape(-1, 1)
    grad = np.repeat(grad, bar_w, axis=1)
    grad_colored = cv2.applyColorMap(grad, cv2.COLORMAP_JET)
    canvas[:, size + 10:size + 10 + bar_w] = grad_colored
    for pct in (0, 25, 50, 75, 100):
        y = int(size - (pct / 100) * (size - 1))
        cv2.putText(canvas, f"{pct}%", (size + 10 + bar_w + 4, min(max(y, 10), size - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1)

    cv2.putText(canvas, "Probability Map (risk %, dynamic only)", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return canvas


# --------------------------------- Main -----------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  WARNING: torch.cuda.is_available() is False -- torch is running "
              "CPU-only even if you have a GPU. Reinstall torch with CUDA support: "
              "pip install torch --index-url https://download.pytorch.org/whl/cu121")

    try:
        import onnxruntime as ort
        ort_providers = ort.get_available_providers()
        print(f"  onnxruntime providers: {ort_providers}")
        if "CUDAExecutionProvider" not in ort_providers:
            print("  WARNING: CUDAExecutionProvider not available -- face matching "
                  "(InsightFace) will run on CPU regardless of your GPU. Fix with: "
                  "pip uninstall onnxruntime && pip install onnxruntime-gpu")
    except ImportError:
        pass

    # YOLO-World-L is noticeably heavy for real-time on CPU. If there's
    # no GPU, fall back to the S weights (already in this folder) --
    # meaningfully faster, still open-vocabulary, small accuracy trade.
    # Set FORCE_MODEL below if you want to override this choice manually.
    FORCE_MODEL = None  # e.g. "yolov8l-worldv2.pt" or "yolov8s-worldv2.pt" to force a choice
    if FORCE_MODEL:
        model_path = FORCE_MODEL
    elif device == "cuda":
        model_path = "yolov8l-worldv2.pt"
    else:
        model_path = "yolov8s-worldv2.pt"

    print(f"Loading YOLO-World ({model_path})...")
    model = YOLOWorld(model_path)
    model.set_classes(VOCAB)

    print("Loading face authorization (InsightFace)...")
    auth = WorkerAuth()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    registry = WorldRegistry()
    last_write = 0.0
    frame_count = 0
    fps_window_start = time.time()
    fps_frame_count = 0

    print("Running. Press 'q' to quit.")
    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            results = model.predict(color_image, conf=CONF_THRESHOLD, verbose=False, device=device)[0]
            frame_count += 1
            fps_frame_count += 1
            if time.time() - fps_window_start > 2.0:
                fps = fps_frame_count / (time.time() - fps_window_start)
                print(f"FPS: {fps:.1f}")
                fps_frame_count = 0
                fps_window_start = time.time()
            now = time.time()

            matched_ids_this_frame = set()

            # --- Phase 1: gather every detection's 3D position first.
            # Matching happens in one batch AFTER this (process_frame_
            # detections), not per-detection as we go -- doing it per-
            # detection let a second person "steal" a first person's
            # already-claimed track when they're close together.
            person_boxes = []      # (x1,y1,x2,y2)
            person_dets = []       # (position, dims, confidence)
            object_boxes = []      # (x1,y1,x2,y2)
            object_dets = []       # (cls_name, position, dims, confidence)

            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                cls_name = results.names[cls_id]
                confidence = float(box.conf[0])

                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                point = deproject_pixel_to_point(depth_frame, intrinsics, cx, cy)
                if point is None:
                    continue

                center_depth = depth_frame.get_distance(cx, cy)
                dims = estimate_dimensions(intrinsics, x1, y1, x2, y2, center_depth)

                if cls_name == "person":
                    person_boxes.append((x1, y1, x2, y2))
                    person_dets.append((point, dims, confidence))
                else:
                    object_boxes.append((x1, y1, x2, y2))
                    object_dets.append((cls_name, point, dims, confidence))

            # --- Phase 2: exclusive batch matching for this whole frame ---
            person_results, object_results = registry.process_frame_detections(person_dets, object_dets)

            # --- Phase 3: auth-check + draw ---
            for (x1, y1, x2, y2), (position, dims, confidence), (obj, is_new_track) in \
                    zip(person_boxes, person_dets, person_results):

                if is_new_track:
                    # Before treating this as a brand-new person, see if
                    # their clothing matches someone who was AUTHORIZED
                    # and got lost very recently (still within their
                    # trust window) -- if so, this is very likely the
                    # same person reappearing from a different angle
                    # than where they were lost, and we should route
                    # back to their existing valid record instead of
                    # spawning an orphan unauthorized track.
                    clothing_hist = compute_clothing_histogram(color_image, x1, y1, x2, y2)
                    reconnected = registry.try_clothing_reconnect(obj, clothing_hist, now)
                    if reconnected is not obj:
                        obj = reconnected
                        is_new_track = False  # successfully routed back to an existing, valid record

                should_check = is_new_track or (now - obj.last_auth_check) > AUTH_CHECK_INTERVAL_SEC
                if should_check:
                    crop = crop_box(color_image, x1, y1, x2, y2)
                    authorized, name, face_found = auth.check(crop)
                    if authorized:
                        obj.apply_auth_result(True, name, now)
                        # Merge back into this worker's ORIGINAL track id
                        # if one already exists (e.g. this is a fresh
                        # track created after Shariq walked back into
                        # frame) -- keeps the id stable instead of
                        # incrementing forever across occlusions.
                        obj = registry.resolve_person_identity(obj, name)
                        # Capture/refresh the clothing signature ONLY on
                        # an actual face-confirmed match -- this is the
                        # sole place clothing_hist ever gets set, which
                        # is what keeps the reconnection bridge above
                        # tied to real, earned trust and nothing else.
                        fresh_hist = compute_clothing_histogram(color_image, x1, y1, x2, y2)
                        if fresh_hist is not None:
                            if obj.clothing_hist is None:
                                obj.clothing_hist = fresh_hist
                            else:
                                blended = CLOTHING_HIST_EMA_ALPHA * obj.clothing_hist + \
                                          (1 - CLOTHING_HIST_EMA_ALPHA) * fresh_hist
                                cv2.normalize(blended, blended, 0, 1, cv2.NORM_MINMAX)
                                obj.clothing_hist = blended
                    elif face_found:
                        # A face WAS visible and confidently did NOT match
                        # any known worker -- demote immediately. This is
                        # what catches a wrong-identity assignment (e.g.
                        # an unauthorized person's detection briefly got
                        # matched to an authorized person's track when
                        # they stood close together) instead of letting
                        # it silently coast on the grace period.
                        obj.apply_auth_result(False, None, now)
                    # else: no face detected at all this check (bad angle,
                    # too far, motion blur) -- not evidence of anything,
                    # leave current state alone and let the grace period
                    # below handle natural decay if checks keep missing.

                matched_ids_this_frame.add(obj.id)

                color = (0, 255, 0) if obj.authorized else (0, 0, 255)  # green=ok, red=unauthorized
                label = f"{obj.id} [{obj.worker_name or 'UNAUTHORIZED'}]"
                cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(color_image, label, (int(x1), int(y1) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Real-world-radius safety rings, using THIS FRAME's raw
                # detected depth (not obj.position, which could be a
                # frozen static-lock) so the ring tracks true live
                # distance. Anchored at the bbox's bottom-center (an
                # approximation of the person's feet/ground contact
                # point) rather than the box center, and squashed into
                # an ellipse -- see GROUND_SQUASH_FACTOR -- so it reads
                # as hugging the floor around them instead of floating
                # at chest height facing the camera.
                # foot_x = (x1 + x2) / 2
                # foot_y = y2
                # depth_m = float(position[2])
                # draw_safety_rings(color_image, intrinsics, foot_x, foot_y, depth_m)

            for (x1, y1, x2, y2), obj in zip(object_boxes, object_results):
                matched_ids_this_frame.add(obj.id)
                color = (0, 255, 0) if obj.static else (0, 165, 255)
                cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(color_image, obj.id, (int(x1), int(y1) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # --- Phase 4 (display-only): draw "last known position" ghost
            # boxes for anything not redetected THIS exact frame but seen
            # recently. Reprojects the object's stored 3D position back to
            # a 2D pixel using the camera intrinsics. Thinner line so it
            # reads as "recently seen" rather than a live detection. This
            # does not touch the registry or the yaml output at all.
            for obj in registry.objects.values():
                if obj.id in matched_ids_this_frame:
                    continue
                age = now - obj.last_seen
                if age > DRAW_GRACE_SEC:
                    continue
                depth = float(obj.position[2])
                if depth <= 0:
                    continue
                px, py = rs.rs2_project_point_to_pixel(intrinsics, obj.position.tolist())
                half_w = max(int((obj.dims[0] * intrinsics.fx) / (2 * depth)), 5)
                half_h = max(int((obj.dims[1] * intrinsics.fy) / (2 * depth)), 5)
                gx1, gy1 = int(px - half_w), int(py - half_h)
                gx2, gy2 = int(px + half_w), int(py + half_h)

                if obj.cls_name == "person":
                    gcolor = (0, 200, 0) if obj.authorized else (0, 0, 200)
                    glabel = f"{obj.id} [{obj.worker_name or 'UNAUTHORIZED'}]"
                else:
                    gcolor = (0, 180, 0) if obj.static else (0, 120, 180)
                    glabel = obj.id
                cv2.rectangle(color_image, (gx1, gy1), (gx2, gy2), gcolor, 1)
                cv2.putText(color_image, glabel, (gx1, gy1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, gcolor, 1)

            registry.end_frame(matched_ids_this_frame)
            registry.expire_stale_authorizations(now)

            if time.time() - last_write > UPDATE_INTERVAL:
                registry.write_yaml()             # full state -- feed this to the LLM on its FIRST call
                registry.compute_and_write_diff() # delta since last write -- feed this on every call after that
                last_write = time.time()

            visible_people = [o for o in registry.objects.values()
                               if o.cls_name == "person" and o.visible]
            radar_img = render_radar_view(visible_people, intrinsics)
            cv2.imshow("Safety Zone Radar (top-down)", radar_img)

            heatmap_img = render_risk_heatmap(registry, intrinsics)
            cv2.imshow("Risk Heat Map (dynamic only, top-down)", heatmap_img)

            prob_img = render_probability_map(registry, intrinsics)
            cv2.imshow("Probability Map (risk %, top-down)", prob_img)

            cv2.imshow("World Model - Authorization (press q to quit)", color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        registry.write_yaml()
        registry.compute_and_write_diff()
        print(f"Final world state saved to {OUTPUT_YAML}, diff saved to {OUTPUT_DIFF_YAML}")


if __name__ == "__main__":
    main()