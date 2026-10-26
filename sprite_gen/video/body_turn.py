# SPDX-License-Identifier: Apache-2.0
"""A walk whose body turns inside its clip, read frame by frame (`video-loop --anchor motion-auto`, `--body-turn`;
docs/video-pipeline.md section 4, "The body turn").

A clip filmed from a still walks at the still's angle, or should: a back-diagonal walk that turns to a full back view
half way and back holds no two steps in one facing, and a cut across the turn plays the body swinging round. The
reading (`torso-side-v1`) puts each frame's torso between the clip's first frame — the still it was filmed from, a
standing pose, lead-in or not — and that frame mirrored about its head:
`side = (v.s - v.m) / (1 - s.m)`, 1 where the torso shows the side the still's does, 0 where it is as near the still as
its mirror (a body seen square, from behind or in front), under 0 past it. A pose the walk changes moves both
correlations alike; a turn moves them apart. Each clip is read against its own first frame, never against a kind of
motion or a direction's name. Not against the first frame after a lead-in (the video model reframing the still to the
walk's size, which the search leaves out): that frame is a stride's pose, whose mirror the walk draws half a cycle on, so
a walk that does not turn reads a turn every stride (a synthetic one, least -0.24); the still filmed small reads blurred,
but only nearer 0 (a synthetic walk filmed from 0.6 of its size for 8 frames: +0.41 where it does not turn, its turn
-0.14).

Three named bounds:

- SIDE_SHOWN_MIN: how far the still's head and shoulders (the top 45 % of its body) lie off their mirror (1 - s.m) for
  the clip to be read at all. Under it the still is seen square — a front or back walk, whose far step lays the torso
  nearer its mirror each step — and its facing is not read: a square still that turns to a diagonal is not seen.
- TURN_SIDE_MAX: a frame whose torso reads under it is turned.
- TURN_RUN_MIN: a turn is that many turned frames in a row or more, counted in frames — the time the clip plays them —
  not in drawings: a walk held on twos holds a turn as long as one drawn on ones does, in half the drawings. A shorter
  dip is the walk's own.

Each crop is centred on the head: the middle of the solid pixels of the top 15 % of the body (HEAD_SHARE), read again on
those within HEAD_REACH of the body's height of it until it settles. A part swung up beside the head (a tail on a beat of
its own, a held prop) would pull a plain mass centre aside every swing, and the torso would read as turning with it (a
synthetic side walk with a 60 px tail on a 45-frame beat read a turn every swing, least -0.21); where nothing lies that
far out, the middle is the mass centre itself. Where nothing lies within HEAD_REACH of the middle — the top's parts lie
apart either side of it: a prop held as high as the head and apart from it, or a body too small for a pixel to lie that
near — the top does not tell which part is the head, and the frame is not read (None in `side`, its why in `unread`); a
still not read so leaves the clip unread, cut as the search alone cuts it. The largest part is not taken for the head: a
prop can weigh what the head does in those rows (a synthetic staff's ornament beside the head weighs the same), and a
crop about the prop would read turns the body does not make, which the cut search would leave out, or
`--body-turn refuse` refuse.

A crop reads only what it holds. A still with no solid pixel across its torso's rows (a part apart above the body tops
its box, its narrowest row in the gap between), a crop of no width, or one of a single value throughout (an empty crop)
is not read — the still leaving the clip unread, a frame None — never read as a torso at 0, as near the still as its
mirror.

The torso (`torso_rows`) is the rows from the neck — the narrowest row of the first frame between NECK_RANGE of its
height — down half the body below it. A body with no neck narrower than its head (a hood, a robe) has its narrowest row
at the range's top, so its crop starts in the head and holds the head and the chest, which turn with it."""
from __future__ import annotations

from typing import Any

from PIL import Image

from sprite_gen._deps import np

READING = "torso-side-v1"
SIDE_SHOWN_MIN = 0.10
TURN_SIDE_MAX = 0.0
TURN_RUN_MIN = 4
MODES = ("warn", "refuse")  # --body-turn: a cut that holds a turn, where none without it holds a cycle, warned of or refused
SOLID = 128  # an alpha at least this is the body
HEAD_SHARE = 0.15  # the head's centre: the middle of the solid pixels of the top 15 % of the body's box
HEAD_REACH = 0.3  # ... of those within this share of the body's height of it, read again until it settles
UPPER = (0.0, 0.45)  # the still's head and shoulders, read for SIDE_SHOWN_MIN
NECK_RANGE = (0.10, 0.55)  # where the neck is looked for, in shares of the body's height from its top
ROWS = np.arange(0.01, 1.0, 0.02)  # the rows the profile is read at, each 2 % of the height tall
CROP = (48, 64)  # every crop is read at this size
LUMA = np.array([0.299, 0.587, 0.114])


def _body(frame: Image.Image) -> tuple[np.ndarray, int, int, float] | str:
    """The frame's solid pixels, its box's top row and height, and its head's centre column (HEAD_SHARE, HEAD_REACH);
    where they are not read, why, said of the frame: no solid pixel, or no head to centre on — the top's parts lie
    apart either side of their middle, none within HEAD_REACH of it."""
    solid = np.asarray(frame.getchannel("A")) >= SOLID
    rows = np.nonzero(solid.any(axis=1))[0]
    if not len(rows):
        return "has no solid body to read"
    top, height = int(rows[0]), int(rows[-1]) - int(rows[0]) + 1
    mass = solid[top:top + max(2, int(HEAD_SHARE * height))].sum(axis=0).astype(np.float64)
    cols = np.arange(len(mass))
    centre = float((mass * cols).sum() / mass.sum())
    for _ in range(20):
        near = np.abs(cols - centre) <= HEAD_REACH * height
        if not mass[~near].any():
            break
        if not mass[near].any():
            return (f"has no head to centre on: the top {HEAD_SHARE * 100:g} % of its body holds parts apart, none "
                    f"within {HEAD_REACH:g} of its height of their middle")
        moved = float((mass * near * cols).sum() / (mass * near).sum())
        settled = abs(moved - centre) < 0.25
        centre = moved
        if settled:
            break
    return solid, top, height, centre


def _unit(*planes: np.ndarray) -> np.ndarray:
    v = np.concatenate([p.ravel() for p in planes])
    v = v - v.mean()
    return v / (np.linalg.norm(v) or 1.0)


def _crop(frame: Image.Image, body: tuple[np.ndarray, int, int, float], rows: tuple[float, float],
          held: float | None = None) -> tuple[np.ndarray, np.ndarray, float] | str:
    """The crop of `rows` (shares of the body's height from its top) about the head's centre, `held` wide (a share of
    the height; where None, the frame's own extent across those rows, about the head), read at CROP: unit vectors of
    its luma times alpha and its alpha, of the same mirrored about the head's centre, and the width held. Where it
    holds nothing to read, why, said of the frame — no solid pixel across those rows (the width the frame's own), a
    crop of no width, or one of a single value throughout (an empty crop's 0): not read as a torso at 0."""
    solid, top, height, centre = body
    r0 = top + int(rows[0] * height)
    r1 = max(r0 + 2, top + int(rows[1] * height))
    span = f"rows {rows[0]:.2f}-{rows[1]:.2f} of its height"
    if held is None:
        cols = np.nonzero(solid[r0:r1].any(axis=0))[0]
        if not len(cols):
            return f"has no solid pixel in its {span}"
        held = 2 * max(centre - cols.min(), cols.max() - centre) / height
    half = held * height / 2
    box = (int(round(centre - half)), r0, int(round(centre + half)), r1)
    if box[2] <= box[0]:
        return f"has a crop of no width over its {span}"
    y = np.asarray(frame.crop(box).resize(CROP, Image.Resampling.LANCZOS), dtype=np.float64) / 255.0
    luma, alpha = (y[..., :3] @ LUMA) * y[..., 3], y[..., 3]
    v = _unit(luma, alpha)
    if not v.any():
        return f"has nothing to read in its crop of {span}"
    return v, _unit(luma[:, ::-1], alpha[:, ::-1]), held


def neck(body: tuple[np.ndarray, int, int, float]) -> float:
    """The neck: of the rows in NECK_RANGE, the narrowest (the span holding the middle 96 % of the row's solid pixels,
    over the height), the topmost on a tie — a share of the body's height."""
    solid, top, height, _ = body
    widths = {}
    for f in ROWS:
        if not NECK_RANGE[0] <= f <= NECK_RANGE[1]:
            continue
        lo = top + int((f - 0.01) * height)
        mass = solid[lo:max(lo + 1, top + int((f + 0.01) * height))].sum(axis=0)
        share = np.cumsum(mass) / max(1.0, float(mass.sum()))
        widths[float(f)] = float(np.searchsorted(share, 0.98) - np.searchsorted(share, 0.02)) / height
    return min(widths, key=lambda f: (widths[f], f))


def torso_rows(at: float) -> tuple[float, float]:
    """The torso's rows: from the neck `at` down half the body below it."""
    return at, at + (1 - at) / 2


def turns(side: list[float | None]) -> list[dict[str, Any]]:
    """The turns of a clip read `side`: each run of TURN_RUN_MIN frames or more in a row whose side is under
    TURN_SIDE_MAX (a frame not read, None, ends a run), its first and last frame and its least side and where."""
    out = []
    k = 0
    while k < len(side):
        end = k
        while end < len(side) and side[end] is not None and side[end] < TURN_SIDE_MAX:
            end += 1
        if end - k >= TURN_RUN_MIN:
            at = min(range(k, end), key=lambda j: side[j])
            out.append({"from": k, "to": end - 1, "least": side[at], "at": at})
        k = max(end, k + 1)
    return out


def read(frames: list[Image.Image]) -> dict[str, Any]:
    """The clip's facing against its first frame (`frames`, the clip as keyed, RGBA): the record the loop report keeps
    (`body_turn`) — the reading, its bounds, the frame read against (`reference`), the still's own side (`still_side`),
    whether it was read (`why` where not), the neck, each frame's `side` and the `turns`. A frame not read is None in
    `side`, and `unread` says why, each why with its frames (where any is)."""
    out: dict[str, Any] = {"reading": READING, "bounds": {"SIDE_SHOWN_MIN": SIDE_SHOWN_MIN, "TURN_SIDE_MAX": TURN_SIDE_MAX,
                                                          "TURN_RUN_MIN": TURN_RUN_MIN},
                           "reference": 0, "turns": []}
    if not frames:
        return {**out, "read": False, "still_side": None, "why": "the clip has no frames"}
    first = frames[0]
    still = _body(first)
    if isinstance(still, str):
        return {**out, "read": False, "still_side": None, "why": f"the clip's first frame {still}"}
    upper = _crop(first, still, UPPER)
    if isinstance(upper, str):
        return {**out, "read": False, "still_side": None, "why": f"the clip's first frame {upper}"}
    s, m, _ = upper
    out["still_side"] = round(1 - float(s @ m), 4)
    if out["still_side"] < SIDE_SHOWN_MIN:
        return {**out, "read": False,
                "why": f"the still is seen square: its head and shoulders lie {out['still_side']:.2f} off their mirror, under "
                       f"{SIDE_SHOWN_MIN:g} (a front or back view), and its facing is not read"}
    at = neck(still)
    out["neck"] = round(at, 2)
    rows = torso_rows(at)
    torso = _crop(first, still, rows)
    if isinstance(torso, str):
        return {**out, "read": False, "why": f"the clip's first frame {torso}, its torso's from the neck down"}
    s, m, held = torso
    shown = 1 - float(s @ m)
    out["torso_still_side"] = round(shown, 4)
    if shown <= 1e-6:
        return {**out, "read": False, "why": "the still's torso is its own mirror: no side to read it by"}
    side: list[float | None] = []
    unread: dict[str, list[int]] = {}
    for k, frame in enumerate(frames):
        body = _body(frame)
        crop = body if isinstance(body, str) else _crop(frame, body, rows, held)
        if isinstance(crop, str):
            side.append(None)
            unread.setdefault(crop, []).append(k)
            continue
        v, _, _ = crop
        side.append(round((float(v @ s) - float(v @ m)) / shown, 4))
    out = {**out, "read": True, "side": side, "turns": turns(side)}
    if unread:
        out["unread"] = [{"frames": ks, "why": why} for why, ks in unread.items()]
    return out


def spans(frames: list[int]) -> str:
    """Frames in runs, in words: `40-42`, `22-23 and 36-45`."""
    runs: list[list[int]] = []
    for k in sorted(frames):
        if runs and k == runs[-1][1] + 1:
            runs[-1][1] = k
        else:
            runs.append([k, k])
    return " and ".join(f"{a}-{b}" if b > a else f"{a}" for a, b in runs)


def turned(reading: dict[str, Any]) -> set[int]:
    """The clip frames inside the reading's turns."""
    return {k for turn in reading["turns"] for k in range(turn["from"], turn["to"] + 1)}


class Turn:
    """A cut's word on the turn, for the cut search (`local_cycle.detect`'s, `by_steps`' and `counted`'s `turn`): why
    the cut s/L (in the search's frames, the clip's after its first `skip`) is refused — it holds a frame of a turn —
    or None. Each refusal is counted (`refused`)."""

    def __init__(self, reading: dict[str, Any], skip: int = 0):
        self.frames = turned(reading)
        self.skip = skip
        self.refused = 0

    def held(self, start: int, length: int) -> list[int]:
        """The turned frames of the clip a cut s/L in the search's frames holds."""
        return sorted(self.frames & set(range(start + self.skip, start + self.skip + length)))

    def __call__(self, start: int, length: int) -> str | None:
        held = self.held(start, length)
        if not held:
            return None
        self.refused += 1
        return f"the body turns (clip frames {spans(held)})"


def _over(reading: dict[str, Any]) -> str:
    spans = [f"{t['from']}-{t['to']}" for t in reading["turns"]]
    least = min(reading["turns"], key=lambda t: t["least"])
    return (f"its torso faces nearer the still's mirror than the still over frames {' and '.join(spans)} (least "
            f"{least['least']:+.2f} at frame {least['at']}; a turn is a torso under {TURN_SIDE_MAX:g} for {TURN_RUN_MIN} "
            "frames or more)")


def refusal(reading: dict[str, Any], *, cut: tuple[int, int] | None = None, held: list[int] | None = None) -> str:
    """The refusal of `--body-turn refuse`, after `video-loop: `: the turn, and that no cut without it holds two steps
    that repeat — or, where a cut was found that holds it (`cut`, in the clip's frames, and its turned frames `held`),
    which — and the clip to be filmed again at the still's angle."""
    found = (f", and the cut {cut[0]}/{cut[1]} holds frames {spans(held)} of it" if cut is not None and held
             else ", and no cut that leaves those frames out holds two steps that repeat")
    return f"the body turns in the walk — {_over(reading)}{found}: film it again with the torso and hips kept at the still's angle"


def kept(reading: dict[str, Any], *, cut: tuple[int, int], held: list[int]) -> str:
    """The warning of `--body-turn warn` over a cut that holds the turn (in the clip's frames), where no cut without it
    holds two steps that repeat."""
    return (f"the body turns in the walk — {_over(reading)}, and the cut {cut[0]}/{cut[1]} holds frames {spans(held)} "
            "of it: no cut that leaves those frames out holds two steps that repeat — kept as cut; --body-turn refuse "
            "refuses such a clip, to be filmed again at the still's angle")


def unfound(reading: dict[str, Any]) -> str:
    """The warning of `--body-turn warn` where no cut holds two steps that repeat, with the turn or without it."""
    return (f"the body turns in the walk — {_over(reading)}: no cut holds two steps that repeat, with those frames or "
            "without them; --body-turn refuse names the turn as the reason, to be filmed again at the still's angle")
