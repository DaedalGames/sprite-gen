# SPDX-License-Identifier: Apache-2.0
"""What a walk's legs say, read off the solid pixels of each frame.

Two stages read them. `video-cycle-align` turns each loop to start as a heel lands
(`align.foot_strike`), and the two-cycle screen (sprite_gen/video/period.py) records how a gait's
steps fall in a shorter return it found — evidence for whoever looks at it, never what decides.
Both read the same per-frame signals (`strike_signals`) and choose between them by the same swings
(`legs_by`):

- `stride`: the width of the foot band (the lowest FOOT_BAND of the frame's own height) as a
  fraction of that height. Seen from the side or a diagonal the feet open and close once a step.
- `reach`: the lowest solid row. Seen from the front or back the foot nearer the viewer is drawn
  lower, once a step.
- `top`: the body's top line, for a body with no legs to read. It rises and falls once a step
  whichever leg leads, so the step count reads it as the witness that a crossing interval holds
  two steps (`local_cycle.bob_word`) — against the lowest row and the head's width, since the
  picture moving or zoomed with the body moves the line as well.
- `gap`: the widest empty run between the foot band's two ends, of the frame's height — two feet
  seen apart.

The step count's witness reads two more (`body_signals`), recorded nowhere else:

- `crown`: the head's width, the log of the mean width (in alpha) of the band CROWN of the clip's
  median body height below the first solid row. A picture zoomed in and out widens and narrows
  it as it lifts and lowers the top line; a body rising and falling on its legs leaves it.
- `edge`: 1 where the lowest solid row is the frame's last row — the feet may run off the
  frame, and the lowest row is the frame's, not theirs — else 0.

Which of the two feet is which is read on cues the view draws (`depth_cue`, `shade_cue`): from the front or back the
foot nearer the viewer is drawn lower, from the side or a diagonal the far leg is drawn in shade. `video-cycle-align`
names the foot that lands by them (`align.strike_foot`), and the step count asks whether the other leg leads one
crossing interval on (`local_cycle.read_cycles`): a cue can only refuse cycles there, never witness them, since a light
fixed on the screen turns it with how far apart the feet are drawn, whichever leg leads. Legs drawn alike to a cue
leave it still: its range over the clip under half the step it is drawn in (a TONE of luma, a pixel row of the body's
height) is the arithmetic's, not the feet's (`local_cycle.cue_word`).

The swing of the stride or the lowest row says there is something to read, not that it is legs:
a legless body that squashes as it bounces widens its foot band once a cycle, where a walker's
widens once a step. Counting steps therefore also asks for two feet seen apart (`feet_apart`).
"""

from __future__ import annotations

import math
from typing import Any

from PIL import Image

from sprite_gen._deps import np

FOOT_BAND = 0.08  # fraction of the frame's own height, measured up from its lowest opaque row
LOW_ALPHA = 128  # the legs are read off solid pixels only
BODY_RUN = 0.5  # the body's top line: the first row at least half as wide as the frame's widest
CROWN = (0.02, 0.10)  # the head's top band: from 2 % to 10 % of the body's median height below its first solid row
# A walk seen from the side or a diagonal opens its feet by a good part of its height each step;
# from the front or back its foot band hardly changes width, but the foot nearer the viewer is
# drawn lower by a few hundredths of it.
STRIDE_MIN = 0.15  # the foot band's swing (of the body's height) that says the feet open along the picture
REACH_MIN = 0.015  # the lowest row's swing (of the body's height) that says a foot steps toward the viewer
# Two feet seen apart: an empty run at least this wide (of the body's height) between the foot
# band's ends, in at least FEET_APART_SHARE of the frames. A walker's feet part for a good share of
# each step, from any view; a legless body's foot band is one run.
FEET_APART = 0.02
FEET_APART_SHARE = 0.2
SHADE_BAND = 0.15  # the shade cue reads the feet and lower legs: the lowest 15 % of the body
TONE = 1 / 255  # one 8-bit level of luma (0..1): the step the shade cue's tones are drawn in; the depth cue's is a pixel row


def _longest_runs(solid: np.ndarray) -> np.ndarray:
    """The longest horizontal run of solid pixels in each row."""
    edges = np.diff(np.pad(solid.astype(np.int8), ((0, 0), (1, 1))), axis=1)
    runs = np.zeros(solid.shape[0], dtype=int)
    for y in np.nonzero(solid.any(axis=1))[0]:
        starts, ends = np.nonzero(edges[y] == 1)[0], np.nonzero(edges[y] == -1)[0]
        runs[y] = int((ends - starts).max())
    return runs


def _widest_gap(cols: np.ndarray) -> int:
    """The longest run of False between the first and the last True of `cols`."""
    xs = np.nonzero(cols)[0]
    inner = ~cols[xs[0] : xs[-1] + 1]
    if not inner.any():
        return 0
    edges = np.diff(np.pad(inner.astype(np.int8), 1))
    return int((np.nonzero(edges == -1)[0] - np.nonzero(edges == 1)[0]).max())


def strike_signals(frames: list[Image.Image]) -> dict[str, list[float]]:
    """Per frame, from the solid pixels: `stride`, `reach`, `top`, `gap` (see the module) and
    `height`, the frame's own body height in pixels. `top` passes over an ear, a hat's point or an
    antenna: the first row whose longest solid run is at least BODY_RUN of the frame's widest."""
    out: dict[str, list[float]] = {"stride": [], "reach": [], "top": [], "gap": [], "height": []}
    for f in frames:
        solid = np.asarray(f.getchannel("A")) >= LOW_ALPHA
        rows = np.nonzero(solid.any(axis=1))[0]
        if rows.size == 0:
            raise ValueError("a loop frame has no solid body")
        y0, y1 = int(rows[0]), int(rows[-1])
        height = y1 - y0 + 1
        cols = solid[y1 + 1 - max(1, round(height * FOOT_BAND)) : y1 + 1].any(axis=0)
        xs = np.nonzero(cols)[0]
        runs = _longest_runs(solid)
        out["stride"].append((xs[-1] - xs[0] + 1) / height)
        out["reach"].append(float(y1))
        out["top"].append(float(np.nonzero(runs >= BODY_RUN * runs.max())[0][0]))
        out["gap"].append(_widest_gap(cols) / height)
        out["height"].append(float(height))
    return out


def body_signals(frames: list[Image.Image]) -> dict[str, list[float]]:
    """Per frame, from the solid pixels, what the step count's witness reads besides `strike_signals`:
    `crown` and `edge` (see the module). `crown` reads the same band of every frame, of the clip's
    median body height."""
    edge, firsts, heights, widths = [], [], [], []
    for f in frames:
        alpha = np.asarray(f.getchannel("A"))
        rows = np.nonzero((alpha >= LOW_ALPHA).any(axis=1))[0]
        if rows.size == 0:
            raise ValueError("a loop frame has no solid body")
        y0, y1 = int(rows[0]), int(rows[-1])
        edge.append(float(y1 == alpha.shape[0] - 1))
        firsts.append(y0)
        heights.append(y1 - y0 + 1)
        widths.append(alpha.sum(axis=1, dtype=np.float64) / 255.0)
    median = float(np.median(heights)) if frames else 0.0
    a0, a1 = max(1, math.ceil(CROWN[0] * median)), max(2, math.ceil(CROWN[1] * median))
    crown = [float(np.log(max(1e-6, w[y0 + a0 : y0 + a1].sum() / (a1 - a0)))) for y0, w in zip(firsts, widths)]
    return {"crown": crown, "edge": edge}


def swings(sig: dict[str, list[float]]) -> dict[str, float]:
    """How far the stride and the lowest row swing over the frames, of the median body height."""
    height = float(np.median(sig["height"]))
    return {k: float(max(sig[k]) - min(sig[k])) / (1.0 if k == "stride" else height) for k in ("stride", "reach")}


def legs_by(sig: dict[str, list[float]]) -> list[str]:
    """The signals that swing enough to read legs by, the stride first."""
    swing = swings(sig)
    return [k for k, least in (("stride", STRIDE_MIN), ("reach", REACH_MIN)) if swing[k] >= least]


def feet_apart(sig: dict[str, list[float]]) -> float:
    """The share of the frames that show two feet apart (a gap of FEET_APART or more)."""
    return float(np.mean([g >= FEET_APART for g in sig["gap"]])) if sig["gap"] else 0.0


def step_signal(sig: dict[str, list[float]]) -> dict[str, Any]:
    """The signal a gait's steps are counted on, or why there is none: the stride where it swings
    by STRIDE_MIN, else the lowest row where it swings by REACH_MIN — the turn's rule without a
    view — and only where two feet are seen apart."""
    by = legs_by(sig)
    apart = feet_apart(sig)
    record: dict[str, Any] = {"feet_apart": round(apart, 3), **{f"{k}_swing": round(v, 4) for k, v in swings(sig).items()}}
    if not by:
        return {**record, "by": None, "why": f"no legs to read: the foot band swings under {STRIDE_MIN} of the body's height "
                                              f"and the lowest row under {REACH_MIN}"}
    if apart < FEET_APART_SHARE:
        return {**record, "by": None, "why": f"no two feet seen apart ({apart:.0%} of the frames, legs from {FEET_APART_SHARE:.0%})"}
    return {**record, "by": by[0], "values": [float(v) for v in sig[by[0]]]}


def feet_halves(frame: Image.Image) -> tuple[np.ndarray, np.ndarray, int]:
    """(solid mask, the picture's right half of the feet as a column mask, body height): the halves
    split at the middle of the foot band (the lowest FOOT_BAND of the frame's own height)."""
    solid = np.asarray(frame.getchannel("A")) >= LOW_ALPHA
    rows = np.nonzero(solid.any(axis=1))[0]
    y1, height = int(rows[-1]), int(rows[-1] - rows[0] + 1)
    xs = np.nonzero(solid[y1 + 1 - max(1, round(height * FOOT_BAND)) : y1 + 1].any(axis=0))[0]
    return solid, np.arange(solid.shape[1]) >= (xs[0] + xs[-1]) / 2, height


def depth_cue(frame: Image.Image) -> float:
    """How much lower the picture's right foot is drawn than its left, as a fraction of the body's
    height (the lowest solid row of each half)."""
    solid, right, height = feet_halves(frame)
    low = [int(np.nonzero(solid[:, half].any(axis=1))[0].max()) if solid[:, half].any() else 0 for half in (~right, right)]
    return (low[1] - low[0]) / height


def shade_cue(frame: Image.Image, facing: str) -> float:
    """How much lighter the front foot is drawn than the back one (mean luma 0..1 of the lowest
    SHADE_BAND of the body, front half against back half; the front is the side faced)."""
    solid, right, height = feet_halves(frame)
    rows = np.nonzero(solid.any(axis=1))[0]
    band = np.zeros_like(solid)
    band[rows[-1] + 1 - max(1, round(height * SHADE_BAND)) : rows[-1] + 1] = True
    rgb = np.asarray(frame.convert("RGB"), dtype=np.float32) / 255.0
    luma = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    front = right if facing == "right" else ~right
    parts = [luma[solid & band & cols[None, :]] for cols in (front, ~front)]
    return float(parts[0].mean() - parts[1].mean()) if all(p.size for p in parts) else 0.0
