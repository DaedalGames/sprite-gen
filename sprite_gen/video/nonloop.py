# SPDX-License-Identifier: Apache-2.0
"""Single-play cuts: terminal condition, endpoint sampling and playback metadata."""
from __future__ import annotations

import math
from typing import Any, Callable

from sprite_gen._deps import np

KINDS = ("once", "lifetime")
MOTION_TEXT = {'death': 'collapses directly onto the ground, settles into a completely prone position, and remains lying entirely motionless for the final 0.4 seconds without shifting, rising, or returning to the starting posture.', 'fall': 'drops briefly downward in place, impacts and settles onto the ground, and holds that grounded resting position completely motionless for the final 0.4 seconds without bouncing, recovering, or returning to the start.', 'land': 'descends vertically a short distance, absorbs ground impact by compressing into a low crouch, and holds that crouched posture completely motionless for the final 0.4 seconds without standing up or returning to the start.', 'cast': 'raises a casting implement or the forward body into an active casting pose and holds that gesture completely motionless for the final 0.4 seconds, displaying no glowing lights, bursts, or particle effects.', 'hurt_hold': 'recoils backward from an impact once and immediately holds the recoiled posture completely motionless for the final 0.4 seconds without relaxing, recovering, or returning to the start.', 'transform': 'changes into a final form through a single completed morph and holds that new form completely motionless for the final 0.4 seconds without shifting, reverting, or returning to the starting form.'}
STATES = frozenset(MOTION_TEXT)
COMMON_TEXT = 'Each action is executed exactly once without looping or returning to the initial state, holding the final pose completely motionless for the final 0.4 seconds, keeping the full body and any props entirely inside the frame under a fixed camera against a flat pure chroma background with no shadows, particles, visual effects, or smears, while preserving the original art style and permitting changes to character design and proportions solely for the transform action.'
ANATOMY_TEXT = {'quadruped': 'Preserve quadruped anatomical structure without requiring an upright or standing stance, allowing natural weight transfer, crouching, and full ground collapse during death or falls without requiring all four limbs to touch the ground on every frame.', 'legless': 'Preserve continuous limbless anatomy without forcing an upright orientation, allowing undulating, coiled, or fully flattened ground configurations during death or falls without introducing limbs.'}
CYCLIC_STATES = frozenset({"idle", "walk", "run", "talk", "blink", "breathe"})
STABLE_SECONDS = 0.2
STABLE_MAX = 0.025  # distance / visible subject mass, independent of canvas padding


def hold_ending(D: np.ndarray, masses: np.ndarray, fps: float) -> tuple[int, int, dict[str, Any]]:
    """Require a visible, stable final window, including slow drift against its endpoint."""
    n = len(D)
    count = max(4, math.ceil(fps * STABLE_SECONDS))
    if n <= count + 1 or float(masses.min()) <= 0:
        raise ValueError("once needs a visible subject and a stable final window")
    scale = float(np.median(masses[-count:]))
    tail = D[-count:, -count:] / scale
    diameter = float(tail.max())
    if diameter > STABLE_MAX:
        raise ValueError(f"once has no stable end pose: final {count} frames move {diameter:.4f} of the subject (limit {STABLE_MAX})")
    departure = D[0] / max(float(masses[0]), scale)
    active = np.flatnonzero(departure > STABLE_MAX * 2)
    if not len(active) or int(active[0]) >= n - count:
        raise ValueError("once has no performed action before its stable end pose")
    start = max(0, int(active[0]) - 1)
    return start, n - 1, {"kind": "hold", "frames": count, "start": n - count,
                         "diameter": diameter, "max": STABLE_MAX, "normalization": "subject-mass"}


Ending = Callable[[np.ndarray, np.ndarray, float], tuple[int, int, dict[str, Any]]]


def select_cut(D: np.ndarray, masses: np.ndarray, *, fps: float, kind: str = "once",
               ending: Ending = hold_ending) -> dict[str, Any]:
    """One cut body; an ending returns inclusive bounds and its measured evidence."""
    start, end, terminal = ending(D, masses, fps)
    if not 0 <= start < end < len(D):
        raise ValueError(f"{kind}: invalid terminal cut [{start}, {end}]")
    energy = np.diag(D, 1)[start:end]
    event = start + int(np.argmax(energy)) + 1
    return {"kind": kind, "start": start, "length": end - start + 1,
            "end_frame": end, "event_frame": event, "terminal": terminal,
            "period_global": None, "ratio": 0.0, "review_recommended": False}


def samples(length: int, count: int) -> list[int]:
    """Inclusive endpoint schedule for a single-play strip or preview."""
    if count < 2 or length < 2:
        raise ValueError("a single-play cut needs at least two output frames")
    return [round(k * (length - 1) / (count - 1)) for k in range(count)]


def metadata(cycle: dict[str, Any], indices: list[int]) -> dict[str, Any]:
    event = cycle["event_frame"] - cycle["start"]
    return {"loop": False, "hold_last": cycle["kind"] == "once", "end_frame": len(indices) - 1,
            "events": [{"frame": min(range(len(indices)), key=lambda k: abs(indices[k] - event)),
                        "name": "motion_peak"}]}


def validate_body(frames: list[Any]) -> None:
    """A terminal pose may change shape, but no keyed body may leave its source canvas."""
    for k, frame in enumerate(frames):
        alpha = frame.getchannel("A").point(lambda a: 255 if a >= 8 else 0)
        box = alpha.getbbox()
        if box is None:
            raise ValueError(f"once: subject is empty at frame {k}")
        if box[0] == 0 or box[1] == 0 or box[2] == frame.width or box[3] == frame.height:
            raise ValueError(f"once: subject touches canvas edge at frame {k}; enlarge the canvas and film again")
