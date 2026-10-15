# SPDX-License-Identifier: Apache-2.0
"""Frames off a walk loop's flow, found by direction and speed, taken out and made again by RIFE.

A video model redraws thin hair a little differently every frame, and now and then a ponytail
lands somewhere it was not one frame earlier, or a tail snaps back at a wrap that does not close:
a frame drawn where the flow does not go. The flow reading (`flow_scores`, `flow_targets`) finds
it by how far it lies off the line between its neighbours, in the usual steps of the region it is
read in; the frames off the flow, one or two drawings in a row, are taken out and made again by
RIFE between the frames either side, each at its own time, and every other frame stays the video's
own (docs/loop-repair.md section 2). Re-making every frame instead (an offset of half a frame) also
softens the frames that were fine, and was judged "not corrected" (2026-10-03). A frame the
ghost screen names a filmed ghost is replaced in the same stage, before the flow is read: by
RIFE's frame between its neighbours where both are clean and it has no fault, else by the clean
frame beside it (docs/loop-repair.md section 4).

Measured on the loop as it plays — cyclic, the last frame followed by the first — and inside
the union box of the body over the whole loop, the way the strip cells are cut, so a frame's
score does not depend on where the canvas put the body.

The size reading (`jump_scores`) counts how much a step changes, whole body and hair behind it.
It picked the frames until the flow reading did, and names a fast swing a jump; the jump guard of
`video-loop-compare` still reads it (docs/loop-comparison.md, "The jump guard").
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from PIL import Image

from sprite_gen._deps import np
from sprite_gen.video import held as held_mod
from sprite_gen.video.rife import Interpolate, crossfade, ghost, outline_shift, smear
from sprite_gen.video.interpolation_quality import clean_beside, faults, ghost_screen, looks

# Interpolation calls at most per loop, a drawing made one call, including rejected proposals. A loop that needs more is
# jolting everywhere, not jumping once, and a few made frames do not fix it (the jolt index, section 3, says so instead).
MAX_REPAIRS = 3
# The hair behind a right-facing body, below the head, as fractions of the union box: where a
# ponytail cut at the wrong moment jumps (the size reading, and the jolt index's hair). A left-facing
# body has it on the other side.
HAIR_BOX_RIGHT = (0.0, 0.30, 0.45, 0.80)
ALPHA_SOLID = 8  # alpha at or above this is body, as everywhere in video-loop
BOX_MARGIN = 8  # px around the union box, the strip cells' own side margin


def union_box(frames: list[Image.Image]) -> tuple[int, int, int, int]:
    """The body's box over the whole loop, widened by the strip cells' margin (the bottom stays on the feet)."""
    boxes = [f.getchannel("A").point(lambda v: 255 if v >= ALPHA_SOLID else 0).getbbox() for f in frames]
    boxes = [b for b in boxes if b]
    if not boxes:
        raise ValueError("every loop frame is fully transparent")
    return (min(b[0] for b in boxes) - BOX_MARGIN, max(0, min(b[1] for b in boxes) - BOX_MARGIN),
            max(b[2] for b in boxes) + BOX_MARGIN, max(b[3] for b in boxes))


def hair_box(facing: str) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = HAIR_BOX_RIGHT
    return (x0, y0, x1, y1) if facing != "left" else (1 - x1, y0, 1 - x0, y1)


def coverage(frames: list[Image.Image], box: tuple[int, int, int, int]) -> list[np.ndarray]:
    """Alpha 0..1 of every frame inside `box` (outside the frame counts as empty)."""
    return [np.asarray(f.crop(box), dtype=np.float32)[..., 3] / 255.0 for f in frames]


def steps(alphas: list[np.ndarray], frac: tuple[float, float, float, float] | None = None) -> np.ndarray:
    """Mean coverage change from frame k to k+1, the last to the first at the end; inside `frac`
    (fractions of the box) when given."""
    if frac is not None:
        h, w = alphas[0].shape
        alphas = [a[int(h * frac[1]):int(h * frac[3]), int(w * frac[0]):int(w * frac[2])] for a in alphas]
    n = len(alphas)
    return np.array([float(np.abs(alphas[(k + 1) % n] - alphas[k]).mean()) for k in range(n)])


def jump_scores(frames: list[Image.Image], *, facing: str = "right", box: tuple[int, int, int, int] | None = None) -> dict[str, Any]:
    """The size reading, per step: the whole body's change and the hair's, each over its own median; the score
    is the larger. The jump guard of `video-loop-compare` reads it (`restoration.jump`); the repair reads the flow."""
    al = coverage(frames, box or union_box(frames))
    whole, hair = steps(al), steps(al, hair_box(facing))
    mw, mh = float(np.median(whole)), float(np.median(hair))
    w = whole / mw if mw > 0 else np.zeros_like(whole)
    h = hair / mh if mh > 0 else np.zeros_like(hair)
    return {"whole": w, "hair": h, "score": np.maximum(w, h)}


# --- the flow reading (docs/loop-repair.md section 2, "The flow reading") ---------------------------

# A loop plays its frames as a path: each frame a point, the body's coverage and shading blurred to
# the size of a step. Where a part moves the way it is going — slow or fast, a leg swung through — a
# frame lies near the line between the frames either side of it. A frame drawn where the flow does
# not go (a tail flung against its sway for one frame, a tail snapping back at a wrap that does not
# close) lies far off that line, whatever its size. The size reading above counts how much a step
# changes and cannot tell the two apart: on an ordinary walk it names a fast swing a jump.
FLOW_BLUR = 0.03  # the blur's sigma, of the body height: a shape moved one ordinary step still overlaps where it was
FLOW_GRID = 4  # the union box in FLOW_GRID x FLOW_GRID cells, read in windows of 2 x 2 cells at a stride of one cell
FLOW_LIMIT = 3.0  # a frame this many of its window's usual steps off the line between its neighbours leaves the flow
FLOW_SPAN = 2  # frames in a row a group takes out at most, to be made again between the frames either side
# The share of a held loop's runs that show their drawing for the hold, a frame more or less (a slip): under it the
# runs are not the hold, and the loop is read a drawing a frame (`flow_drawings`).
HELD_RUNS = 0.75
_LUMA = np.array([0.299, 0.587, 0.114])


def _blur(planes: np.ndarray, sigma: float) -> np.ndarray:
    """A Gaussian blur of `sigma` px over axes 1 and 2 of (n, h, w, c), outside the grid empty."""
    r = max(1, int(np.ceil(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    for axis in (1, 2):
        pad = [(0, 0)] * planes.ndim
        pad[axis] = (r, r)
        padded = np.pad(planes, pad)
        size = planes.shape[axis]
        planes = sum(k[i] * np.take(padded, np.arange(i, i + size), axis=axis) for i in range(2 * r + 1))
    return planes


def flow_states(frames: list[Image.Image], box: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Each frame of the loop as a point of its path, (frames, h, w, 2): its coverage and its shading
    (luma times coverage) inside `box` (the union box), averaged down to a grid of a third of the blur
    and blurred by FLOW_BLUR of the body height. Raises ValueError for a loop with no body."""
    pixels = np.stack([np.asarray(f.crop(box or union_box(frames)), dtype=np.float64) / 255.0 for f in frames])
    alpha = pixels[..., 3]
    rows = np.where(alpha.max(axis=(0, 2)) > 0.1)[0]
    if rows.size == 0:
        raise ValueError("every loop frame is fully transparent")
    sigma = FLOW_BLUR * max(1, int(rows.max() - rows.min()))
    planes = np.stack([alpha, (pixels[..., :3] @ _LUMA) * alpha], axis=-1)
    n, height, width, c = planes.shape
    f = max(1, min(int(round(sigma / 1.5)), height, width))
    h, w = height // f, width // f
    small = planes[:, :h * f, :w * f].reshape(n, h, f, w, f, c).mean(axis=(2, 4))
    return _blur(small, sigma / f)


def flow_windows(h: int, w: int) -> list[tuple[int, int, int, int]]:
    """The windows of a (h, w) grid the flow is read in, (y0, y1, x0, x1): 2 x 2 of FLOW_GRID x FLOW_GRID
    cells at a stride of one cell, so a part between two cells is whole in one window."""
    ys = np.linspace(0, h, FLOW_GRID + 1).astype(int)
    xs = np.linspace(0, w, FLOW_GRID + 1).astype(int)
    return [(int(ys[i]), int(ys[i + 2]), int(xs[j]), int(xs[j + 2])) for i in range(FLOW_GRID - 1) for j in range(FLOW_GRID - 1)]


def _flow_read(states: np.ndarray, windows: list[tuple[int, int, int, int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per frame and window, cyclic: how far the frame lies off the line between its two neighbours (half the
    change from the step into it to the step out of it), the step out of it (frame k to k+1), and the cosine
    between the step into it and the step out of it. Each (frames, windows)."""
    n = len(states)
    off, step, turn = (np.zeros((n, len(windows))) for _ in range(3))
    for i, (y0, y1, x0, x1) in enumerate(windows):
        x = states[:, y0:y1, x0:x1].reshape(n, -1)
        out = np.roll(x, -1, axis=0) - x
        into = np.roll(out, 1, axis=0)
        step[:, i] = np.linalg.norm(out, axis=1)
        off[:, i] = np.linalg.norm(out - into, axis=1) / 2
        norms = np.linalg.norm(into, axis=1) * step[:, i]
        turn[:, i] = np.where(norms > 0, (into * out).sum(axis=1) / np.where(norms > 0, norms, 1.0), 1.0)
    return off, step, turn


def _flow_scale(step: np.ndarray) -> np.ndarray:
    """Per window, the usual step it is read against: its median step, and no less than the median of
    those over the windows, so a window where little moves is not read on its noise."""
    median = np.median(step, axis=0)
    return np.maximum(median, max(float(np.median(median)), 1e-12))


def flow_drawings(states: np.ndarray) -> tuple[list[list[int]], int]:
    """The drawings of a loop's path, as runs of the frames that show each (each run from the frame that
    begins it, in play order), and the hold. A loop filmed on twos or threes shows each drawing for two or
    three frames: the repeat is the same point of the path, and its step of nothing beside the drawing's
    step of two would read as a lurch at every drawing. The hold is `held.measure`'s, read on the steps of
    this path as a ring; a frame whose step from the frame before it is under its boundary repeats that
    frame's drawing. A loop drawn every frame (hold 1) is a drawing a frame.

    A loop whose only large steps are the steps into and out of a few jumps can read as held there: its
    ordinary steps the repeats, its jumps the drawings. On those drawings its jumps would read as ordinary
    steps, and a drawing made again would stand for every frame of a run of five. Its runs are not its
    hold: a hold is taken only where at least HELD_RUNS of the runs are the hold, a frame more or less,
    and the loop is otherwise read a drawing a frame."""
    n = len(states)
    steps = [float(np.linalg.norm(states[(k + 1) % n] - states[k])) for k in range(n)]
    # The hold is read on the rhythm of the steps; the rate only says how many drawings a second, unused here.
    rhythm = held_mod.measure(steps, fps=1.0, cyclic=True)
    hold = int(rhythm.get("hold") or 1)
    repeat = [hold > 1 and steps[(t - 1) % n] < rhythm["held_below"] for t in range(n)]
    if all(repeat) or not any(repeat):
        return [[t] for t in range(n)], 1
    begin = repeat.index(False)
    runs: list[list[int]] = []
    for i in range(n):
        t = (begin + i) % n
        if repeat[t]:
            runs[-1].append(t)
        else:
            runs.append([t])
    if sum(abs(len(run) - hold) <= 1 for run in runs) < HELD_RUNS * len(runs):
        return [[t] for t in range(n)], 1
    return runs, hold


def _flow(frames: list[Image.Image], box: tuple[int, int, int, int] | None) -> dict[str, Any]:
    """The loop's path read on its drawings: their states, the runs of frames that show them, the windows,
    and per drawing and window the distance off the path, the step out and the turn, and the windows' usual
    step."""
    states = flow_states(frames, box)
    runs, hold = flow_drawings(states)
    path = states[[run[0] for run in runs]]
    windows = flow_windows(states.shape[1], states.shape[2])
    off, step, turn = _flow_read(path, windows)
    h, w = states.shape[1], states.shape[2]
    return {"path": path, "runs": runs, "hold": hold, "windows": windows, "off": off, "step": step, "turn": turn,
            "scale": _flow_scale(step), "fractions": [(x0 / w, y0 / h, x1 / w, y1 / h) for y0, y1, x0, x1 in windows]}


def _drawing_scores(flow: dict[str, Any]) -> dict[str, Any]:
    """Per drawing: `off`, `window`, `turn`, `into` and `out` in its worst window (`flow_scores`)."""
    rel = flow["off"] / flow["scale"]
    worst = rel.argmax(axis=1)
    at = np.arange(len(rel))
    out = flow["step"] / flow["scale"]
    return {"off": rel[at, worst], "window": worst, "turn": flow["turn"][at, worst],
            "into": np.roll(out, 1, axis=0)[at, worst], "out": out[at, worst]}


def flow_scores(frames: list[Image.Image], *, box: tuple[int, int, int, int] | None = None) -> dict[str, Any]:
    """Per frame, how far its drawing lies off the loop's flow (`off`): the distance from the line between
    the drawings either side of it, over its window's usual step, in the window where that is largest
    (`window`, its index in `windows`, fractions of the box (x0, y0, x1, y1)); there the cosine between
    the step into the drawing and the step out of it (`turn`, below 0 against the way it came, 1 where
    either is still) and the two steps in the window's usual steps (`into`, `out`). Every frame that
    shows a drawing (`flow_drawings`, `hold`) reads the drawing's values."""
    flow = _flow(frames, box)
    per = _drawing_scores(flow)
    of = np.zeros(len(frames), dtype=int)
    for d, run in enumerate(flow["runs"]):
        of[run] = d
    return {**{k: v[of] for k, v in per.items()}, "windows": flow["fractions"], "hold": flow["hold"]}


def between_time(a: int, frame: int, b: int, count: int) -> float:
    """The time of `frame` on the way from frame `a` to frame `b` of a loop `count` frames long, 0..1, in play
    order (the wrap included): a frame made there shows the pose at its own time, so the loop keeps its length,
    every frame its time, and a drawing held on twos its two frames."""
    return ((frame - a) % count) / ((b - a) % count)


def flow_targets(frames: list[Image.Image], *, limit: float = FLOW_LIMIT, span: int = FLOW_SPAN,
                 box: tuple[int, int, int, int] | None = None, fixed: Iterable[int] = ()) -> dict[str, Any]:
    """The frames that leave the loop's flow, in groups to take out and make again between the frames either
    side (docs/loop-repair.md section 2, "The flow reading"). Read on the loop's drawings (`flow_drawings`):
    a frame on its own where every frame is a drawing, all the frames that show it on twos or threes.

    The worst drawing at or over `limit` first. A group is up to `span` drawings in a row that take that
    drawing or lie beside it; its in-between is the straight line of the path from the drawing before the
    group to the drawing after, each drawing at its own time (`between_time`). Of the groups whose in-between
    brings every drawing from the one before them to the one after under `limit`, the one taken is the one
    whose every drawing the in-between moves furthest (`move`, the least over its drawings of the move in its
    worst window, in the window's usual steps): the drawings off the flow, not the ones beside them that read
    off only because a neighbour is, and no drawing the in-between hardly moves. A pair flung together is taken
    out whole, where an in-between beside it would bring the reading under `limit` by smoothing its edges and
    leave it standing; a part that snaps back at the wrap is one frame on either side of it, where two would
    spread the snap over a frame the in-between hardly moves. Ties go to the smaller result, then the fewer
    drawings. The path then reads with that in-between, and the next worst drawing is read. A group never
    takes, or is made from, a drawing a group before it took, or a drawing that shows a frame in `fixed`
    (made, given way or refused already). A drawing no group brings under `limit` is `unfixed`.

    Each group: its `frames` in play order, the `drawings` it takes (the frames that show each), the
    `neighbours` it is made from (the first frame of the drawing before it and of the drawing after), each
    drawing's time `t` between them, `off` (the worst of its drawings before) with its `turn`, `into`, `out` and
    `window`, `move`, and `off_after`. The in-between is the straight line of the blurred path, a stand-in for
    the frames RIFE makes there: it says which frames to take out, not how the made frames will look."""
    flow = _flow(frames, box)
    first = _drawing_scores(flow)
    runs, windows, scale = flow["runs"], flow["windows"], flow["scale"]
    n, count = len(runs), len(frames)
    record: dict[str, Any] = {"limit": limit, "span": span, "blur": FLOW_BLUR, "grid": FLOW_GRID, "hold": flow["hold"],
                              "drawings": n, "off_max_before": round(float(first["off"].max()), 4), "groups": [], "unfixed": []}
    if n < 4:
        record.update(stopped="loop shorter than 4 drawings", off_max_after=record["off_max_before"])
        return record

    def reading(path: np.ndarray) -> np.ndarray:
        return (_flow_read(path, windows)[0] / scale).max(axis=1)

    def moved(made: np.ndarray, path: np.ndarray, group: list[int]) -> float:
        return min(max(float(np.linalg.norm(made[g, y0:y1, x0:x1] - path[g, y0:y1, x0:x1]) / scale[i])
                       for i, (y0, y1, x0, x1) in enumerate(windows)) for g in group)

    def said(d: int) -> dict[str, Any]:
        return {"off": round(float(first["off"][d]), 4), "turn": round(float(first["turn"][d]), 4),
                "into": round(float(first["into"][d]), 4), "out": round(float(first["out"][d]), 4),
                "window": [round(v, 4) for v in flow["fractions"][int(first["window"][d])]]}

    of = {t: d for d, run in enumerate(runs) for t in run}
    taken = {of[t] for t in fixed}
    path, unfixed = flow["path"].copy(), set()
    current = reading(path)
    while True:
        open_ = [d for d in np.argsort(-current, kind="stable") if current[d] >= limit and d not in taken and d not in unfixed]
        if not open_:
            break
        d = int(open_[0])
        tried: list[tuple[float, float, list[int], int, int, np.ndarray]] = []
        for size in range(1, min(span, n - 2) + 1):
            # the groups of `size` that take d or lie beside it: d is between the drawing before them and the one after
            for start in range(d - size, d + 2):
                group = [(start + i) % n for i in range(size)]
                a, b = (start - 1) % n, (start + size) % n
                if {a, b, *group} & taken:
                    continue
                made = path.copy()
                for g in group:
                    made[g] = path[a] + between_time(runs[a][0], runs[g][0], runs[b][0], count) * (path[b] - path[a])
                after = reading(made)
                tried.append((float(max(after[(a + i) % n] for i in range(size + 2))), moved(made, path, group), group, a, b, made))
        ok = [x for x in tried if x[0] < limit]
        if not ok:
            unfixed.add(d)
            record["unfixed"].append({"frames": runs[d], **said(d),
                                      "off_after": round(min((x[0] for x in tried), default=float(current[d])), 4)})
            continue
        result, move, group, a, b, made = max(ok, key=lambda x: (round(x[1], 9), -x[0], -len(x[2])))
        worst = max(group, key=lambda g: float(first["off"][g]))
        record["groups"].append({"frames": [t for g in group for t in runs[g]], "drawings": [runs[g] for g in group],
                                 "neighbours": [runs[a][0], runs[b][0]],
                                 "t": [round(between_time(runs[a][0], runs[g][0], runs[b][0], count), 4) for g in group],
                                 **said(worst), "move": round(move, 4), "off_after": round(result, 4)})
        taken |= set(group)
        path, current = made, reading(made)
    record["off_max_after"] = round(float(current.max()), 4)
    return record


def _give_way(frames: list[Image.Image], j: int, ghosts: dict[int, float], interpolate: Interpolate | None,
              out: list[Image.Image]) -> dict[str, Any]:
    """The round of a filmed ghost at `j` (`why` ghost), its cell replaced in `out` where it gives way.

    Where both frames beside it are clean, RIFE's frame half way between them is proposed and taken
    without a fault (`taken` rife); else, or where there is no interpolator, the clean frame beside it
    the cycle alignment would take (`interpolation_quality.clean_beside`, `taken` {source}); where
    both are filmed ghosts too it is kept (`taken` None)."""
    n = len(frames)
    a, b = (j - 1) % n, (j + 1) % n
    measure: dict[str, Any] | None = None
    wrong: list[str] = []
    no_proposal = ("a frame beside it is a filmed ghost too" if a in ghosts or b in ghosts
                   else "no interpolator" if interpolate is None else None)
    if no_proposal is None:
        assert interpolate is not None
        made = interpolate(frames[a], frames[b], 0.5)
        if made.mode != "RGBA" or made.size != frames[j].size:
            raise ValueError("repair interpolator must return an RGBA frame of the original size")
        measure = {**smear(made, frames[a], frames[b]), "outline_shift": outline_shift(made, frames[a], frames[b]),
                   "ghost": ghost(made), "crossfade": crossfade(made, frames[a], frames[b], 0.5)}
        wrong = faults(measure)
        if not made.getchannel("A").getbbox():
            wrong.append("empty")
    beside = clean_beside(j, ghosts, n)
    taken: Any = "rife" if measure is not None and not wrong else {"source": beside} if beside is not None else None
    if taken == "rife":
        out[j] = made
    elif taken is not None:
        out[j] = frames[beside]
    return {"why": "ghost", "target": j, "neighbours": [a, b],
            "original": {**smear(frames[j], frames[a], frames[b]), "ghost": ghosts[j]},
            "proposal": measure, **({"no_proposal": no_proposal} if no_proposal else {}), "faults": wrong,
            "look": looks(measure) if measure is not None else [], "taken": taken,
            "outcome": "accepted" if taken == "rife" else "given-way" if taken is not None else "kept",
            **({"replaced": j} if taken is not None else {})}


def repair_jumps(frames: list[Image.Image], interpolate: Interpolate | None, *, facing: str = "right",
                 limit: float = FLOW_LIMIT, span: int = FLOW_SPAN, max_frames: int = MAX_REPAIRS,
                 screen: dict[str, Any] | None = None) -> tuple[list[Image.Image], dict[str, Any]]:
    """Replace each filmed ghost the screen names, then take out each group of frames off the loop's flow and
    make it again with RIFE between the frames either side, worst first (docs/loop-repair.md section 2).

    `screen` is the ghost screen of `frames` as filmed (`interpolation_quality.ghost_screen`, read
    here when not given). Each ghost it names is a round of its own (`why` ghost, in frame order,
    outside the call budget): RIFE's frame between its two neighbours where both are clean and the
    frame has no fault, else the clean frame beside it (the frame after it, else the one before it),
    else, both being filmed ghosts too, the ghost kept (`_give_way`). The flow is then read on the
    loop as the ghosts left it (`flow_targets`, `why` flow): the worst group first, one call a drawing
    it takes, each drawing made at its own time between the group's neighbours (`between_time`) and
    shown on every frame the drawing was — the loop keeps its length, every frame its time, a drawing
    held on twos its two frames. At most `max_frames` calls; a group the calls left cannot pay for
    stops the stage (`left`). Never made from or beside a frame already made or given way: two made
    frames side by side are made from each other and melt the legs. A group whose made drawings carry
    a shared interpolation-quality fault — a ghost too, on its own coverage, where the screen reads
    the loop — keeps every frame it would take, as filmed, and nothing is made from or beside them
    (`refused`); a made drawing that cross-fades two drawings inside its silhouette is still taken,
    its round naming it (`look`). `interpolate` may be None: a ghost then gives way to the clean frame
    beside it, and the flow search stops at the first group it would make, naming it (`unmade`). The
    flow is read again after each group, on the union box of the original loop. `facing` reads nothing
    here: it is recorded for the jump guard of `video-loop-compare` (`restoration.jump`)."""
    out = list(frames)
    n = len(out)
    box = union_box(frames)
    screen = screen if screen is not None else ghost_screen(frames)
    reads = bool(screen["reads"])
    record: dict[str, Any] = {
        "reading": "flow", "limit": limit, "span": span, "max_frames": max_frames, "facing": facing,
        "off_max_before": round(float(flow_scores(frames, box=box)["off"].max()), 4), "replaced": [], "rounds": [],
        "refused": [], "attempts": 0,
    }
    if n < 4:
        record.update(stopped="loop shorter than 4 frames", off_max_after=record["off_max_before"])
        return out, record
    replaced: list[int] = []
    ghosts = {g["frame"]: g["ghost"] for g in screen["ghosts"]}
    for j in sorted(ghosts):
        round_ = _give_way(frames, j, ghosts, interpolate, out)
        record["rounds"].append(round_)
        if "replaced" in round_:
            replaced.append(j)
    refused: list[int] = []
    calls = 0
    while True:
        plan = flow_targets(out, limit=limit, span=span, box=box, fixed=[*replaced, *refused])
        record.setdefault("hold", plan["hold"])
        record.setdefault("drawings", plan["drawings"])
        if not plan["groups"]:
            stopped = ("no frame off the flow" if not plan["unfixed"] else
                       f"no group of up to {span} drawings brings the frames left off the flow under {limit:g} (`unfixed`)")
            break
        group = plan["groups"][0]
        need = len(group["drawings"])
        if calls + need > max_frames:
            stopped = (f"{max_frames} interpolation calls used" if calls >= max_frames else
                       f"the next group needs {need} interpolation calls, {max_frames - calls} left")
            record["left"] = [{"frames": g["frames"], "off": g["off"]} for g in plan["groups"]]
            break
        if interpolate is None:
            record["unmade"] = {k: group[k] for k in ("frames", "neighbours", "off")}
            stopped = "a frame to make off the flow and no interpolator"
            break
        ia, ib = group["neighbours"]
        a, b = out[ia], out[ib]
        made: list[tuple[list[int], Image.Image, dict[str, Any]]] = []
        for drawing in group["drawings"]:
            t = between_time(ia, drawing[0], ib, n)
            frame = interpolate(a, b, t)
            if frame.mode != "RGBA" or frame.size != out[drawing[0]].size:
                raise ValueError("repair interpolator must return an RGBA frame of the original size")
            measure = {**smear(frame, a, b), "outline_shift": outline_shift(frame, a, b),
                       "ghost": ghost(frame) if reads else None, "crossfade": crossfade(frame, a, b, t)}
            wrong = faults(measure) + ([] if frame.getchannel("A").getbbox() else ["empty"])
            made.append((drawing, frame, {
                "frames": drawing, "t": round(t, 4), "proposal": measure, "faults": wrong, "look": looks(measure),
                "original": {**smear(out[drawing[0]], a, b), "ghost": ghost(out[drawing[0]]) if reads else None}}))
        calls += need
        wrong = sorted({f for *_, m in made for f in m["faults"]})
        round_ = {"why": "flow", **group, "made": [m for *_, m in made], "faults": wrong,
                  "look": sorted({w for *_, m in made for w in m["look"]}), "outcome": "rejected" if wrong else "accepted"}
        if wrong:
            refused.extend(group["frames"])
        else:
            for drawing, frame, _ in made:
                for k in drawing:
                    out[k] = frame
            replaced.extend(group["frames"])
            round_["replaced"] = group["frames"]
        record["rounds"].append(round_)
    record.update(replaced=replaced, refused=refused, attempts=calls, stopped=stopped, unfixed=plan["unfixed"],
                  off_max_after=plan["off_max_before"])
    return out, record


# --- the jolt index (docs/loop-repair.md section 3) ------------------------------------------

# Reference bounds, at the edge of what was kept on the 2026-10-03 takes (section 3): nothing kept
# exceeds them, and the back-diagonal take refused because "it jumps" does. A loop beyond them is
# warned about; it fails only when a caller passes a bound (--jolt-max / --head-step-max), because
# eleven judged takes are too few to refilm on, and at these values 16 of 30 Lite takes would be.
JOLT_REFERENCE = 0.43  # alpha jolt index; the kept takes reached 0.414 on strip cells, 0.4225 on the cut frames video-loop reads
HEAD_STEP_REFERENCE = 0.75  # head's sideways move in one frame, % of body height; kept <= 0.68, refused 0.82
HEAD_BAND = 0.20  # the head is the top fifth of the body's height over the loop
HEAD_MEDIAN_FLOOR = 0.05  # % of body height: a median step below this is the tracker's rounding
# The top band's step into the loop's first frame, over the loop's median step, beyond which the
# wrap pops (docs/loop-repair.md section 3, "The seam pop"). It sits between walks holding a part
# that sways on its own beat and walks without one, on the cut as it plays and on the source frames
# the cut is chosen from. Above it the cut is chosen again (`--anchor motion-auto`), and a pop that
# stays is a warning line.
SEAM_POP_REFERENCE = 10.0
# The held side's seam (docs/loop-repair.md section 3, "The held side"): a held part that swings on
# its own beat misses below its top too — under the hand, where the top band does not reach. Read
# on the silhouette's outer edge on the side the top band sits on, row by row: how far the cut's
# wrap is from the clip's own way on there, in place and in speed, over that row's median step in
# the loop; the HELD_EDGE_QUANTILE-th percentile of the rows. Under HELD_EDGE_REFERENCE the held
# side closes. On a walk without a held part the edge is the hair or an arm, and the same reading
# runs as high as on a staff that does not close, so it is read only on a cut whose top band pops
# (`--anchor motion-auto` chooses again there), where the top band has found the held part and its
# side. The bound sits between cuts that close a held staff and cuts where it swings back at the wrap.
HELD_EDGE_REFERENCE = 1.2
HELD_EDGE_QUANTILE = 80
HELD_EDGE_FLOOR = 0.005  # of the body height, a pixel at least: a row's median step under it is the edge's rounding
HELD_EDGE_ALPHA = 128  # alpha at or above this is the edge (0..255)


def jolt_index(step_values: np.ndarray) -> float:
    """How far each step strays from the mean of its two neighbours, on average, over the median
    step. 0 for a loop whose steps change evenly; alternating big and small steps read high."""
    n = len(step_values)
    med = float(np.median(step_values))
    if n < 3 or med <= 0:
        return 0.0
    return float(np.mean([abs(step_values[k] - (step_values[k - 1] + step_values[(k + 1) % n]) / 2) for k in range(n)]) / med)


def _track_stats(values: np.ndarray) -> dict[str, float]:
    n = len(values)
    d = np.abs(np.array([values[(k + 1) % n] - values[k] for k in range(n)]))
    med = float(np.median(d))
    return {"step_max_pct": round(float(d.max()), 3), "step_median_pct": round(med, 3),
            "max_over_median": round(float(d.max()) / max(med, HEAD_MEDIAN_FLOOR), 3),
            "range_pct": round(float(np.ptp(values)), 3), "worst_into_frame": int((np.argmax(d) + 1) % n)}


def head_track(alphas: list[np.ndarray]) -> dict[str, Any]:
    """The head's place frame by frame, in % of the body's height over the loop: x is the coverage
    centroid of the top fifth (HEAD_BAND), y the body's top line. A take whose head pops sideways
    reads here while its coverage change stays ordinary (2026-10-03, Lite back diagonal)."""
    x, y, height = _head_series(alphas)
    return {"body_height_px": height, "x": _track_stats(x), "y": _track_stats(y)}


def _head_series(alphas: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, int]:
    """`head_track`'s two places per frame (% of the body height) and the body height in px."""
    rows = np.where(np.max(np.stack(alphas), axis=0).max(axis=1) > 0.1)[0]
    if rows.size == 0:
        raise ValueError("every loop frame is fully transparent")
    top, bottom = int(rows.min()), int(rows.max())
    height = max(1, bottom - top)
    band = slice(top, top + max(1, int(height * HEAD_BAND)))
    xs, ys = [], []
    for a in alphas:
        b = a[band]
        mass = float(b.sum())
        xs.append(float((b.sum(axis=0) * np.arange(b.shape[1])).sum() / mass) if mass > 0 else float("nan"))
        filled = np.where(a.sum(axis=1) > 0.5)[0]
        ys.append(float(filled.min()) if filled.size else float("nan"))
    x = np.array(xs) / height * 100
    y = np.array(ys) / height * 100
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("a loop frame has no head to track")
    return x, y, height


def seam_pop(alphas: list[np.ndarray]) -> dict[str, Any]:
    """How far the top of the silhouette jumps into the loop's first frame (docs/loop-repair.md
    section 3, "The seam pop").

    The top band `head_track` reads is the head, or the tip of whatever the body holds higher than its
    head — a staff, a flag, a raised spear. A thin part like that can sway on its own beat, slower than
    the steps: a cut one step long then ends with it somewhere else, and it jumps at the wrap while
    the body closes. The seam ratio is an area measure and hardly sees a thin part move: a tip that
    jumps many times its usual step changes few pixels.

    Per axis, the wrap's step (last frame -> first) over the loop's median step, a pixel at least: a
    step under a pixel is the tracker's rounding. It pops when the wrap is that axis' largest step
    and exceeds SEAM_POP_REFERENCE times the median; `pop` is the larger of the two, 0 when neither
    axis has its largest step at the wrap. A loop whose top jitters more inside than at the wrap
    (hair redrawn every frame) has its wrap ordinary for it."""
    x, y, height = _head_series(alphas)
    floor = max(HEAD_MEDIAN_FLOOR, 100.0 / height)
    record: dict[str, Any] = {"body_height_px": height, "reference": SEAM_POP_REFERENCE}
    pop = 0.0
    for axis, values in (("x", x), ("y", y)):
        n = len(values)
        d = np.abs(np.array([values[(k + 1) % n] - values[k] for k in range(n)]))
        over = float(d[-1]) / max(float(np.median(d)), floor)
        worst = bool(d[-1] >= d.max())
        record[axis] = {"wrap_pct": round(float(d[-1]), 3), "step_median_pct": round(float(np.median(d)), 3),
                        "wrap_over_median": round(over, 3), "wrap_is_largest": worst}
        if worst:
            pop = max(pop, over)
    record["pop"] = round(pop, 3)
    record["pops"] = pop > SEAM_POP_REFERENCE
    return record


def held_side(masks: list[np.ndarray]) -> str:
    """The side of the body the top band (HEAD_BAND) sits on, "left" or "right": where a part held
    higher than the head is held. `masks` are a cut's frames, True where the body is."""
    stack = np.stack(masks)
    rows = np.where(stack.any(axis=(0, 2)))[0]
    if rows.size == 0:
        raise ValueError("every loop frame is fully transparent")
    top, bottom = int(rows.min()), int(rows.max())
    band = stack[:, top:top + max(1, int(max(1, bottom - top) * HEAD_BAND))]
    cols = np.arange(stack.shape[2])
    body, tip = stack.sum(axis=(0, 1)), band.sum(axis=(0, 1))
    return "right" if float((tip * cols).sum()) / float(tip.sum()) >= float((body * cols).sum()) / float(body.sum()) else "left"


def side_edges(masks: list[np.ndarray], side: str) -> np.ndarray:
    """(frames, rows): each row's outermost body column on `side`, NaN where the row is empty."""
    stack = np.stack(masks)
    filled = stack.any(axis=2)
    if side == "right":
        edge = stack.shape[2] - 1 - stack[:, :, ::-1].argmax(axis=2)
    else:
        edge = stack.argmax(axis=2)
    out = edge.astype(np.float64)
    out[~filled] = np.nan
    return out


def held_edge(edges: np.ndarray, start: int, length: int) -> dict[str, Any]:
    """The held side's seam of a cut (start, length) of a clip whose side edges are `edges`
    (`side_edges`), section 3, "The held side".

    A loop closes where its last frame is the frame before its first and its first the frame after its
    last: per row, the cut's first frame against the clip's frame after its last, its last against the
    clip's frame before its first, and the step into and out of each against the clip's own (a part
    that comes back to its place moving the other way swings back at the wrap). Their mean over the
    row's median step in the loop, a floor of HELD_EDGE_FLOOR; the HELD_EDGE_QUANTILE-th percentile
    over the rows is `seam`, and the held side `closes` under HELD_EDGE_REFERENCE. Raises ValueError
    for a cut with fewer than two clip frames on either side, or no row to read."""
    n = len(edges)
    s, L = start, length
    if s < 2 or s + L + 1 >= n:
        raise ValueError("the clip has no two frames either side of the cut to read its wrap against")
    loop = edges[s:s + L]
    rows = np.where(np.isfinite(loop).any(axis=0))[0]
    if rows.size == 0:
        raise ValueError("every loop frame is fully transparent")
    steps = np.abs(np.diff(np.vstack([loop, loop[:1]]), axis=0))
    terms = np.stack([np.abs(edges[s] - edges[s + L]),
                      np.abs((edges[s + 1] - edges[s]) - (edges[s + L + 1] - edges[s + L])),
                      np.abs(edges[s + L - 1] - edges[s - 1]),
                      np.abs((edges[s + L - 1] - edges[s + L - 2]) - (edges[s - 1] - edges[s - 2]))])
    with np.errstate(invalid="ignore"):
        counted = np.isfinite(steps).sum(axis=0)
        median = np.where(counted > 0, np.nanmedian(np.where(counted > 0, steps, 0.0), axis=0), np.nan)
        read = np.isfinite(terms).sum(axis=0)
        mean = np.where(read > 0, np.nansum(terms, axis=0) / np.maximum(read, 1), np.nan)
    floor = max(1.0, HELD_EDGE_FLOOR * float(rows.max() - rows.min()))
    ratio = mean / np.maximum(median, floor)
    ratio = ratio[np.isfinite(ratio)]
    if ratio.size == 0:
        raise ValueError("no row has the held side's edge in the frames its wrap is read on")
    seam = round(float(np.percentile(ratio, HELD_EDGE_QUANTILE)), 3)
    return {"seam": seam, "closes": seam < HELD_EDGE_REFERENCE, "rows": int(ratio.size), "reference": HELD_EDGE_REFERENCE}


def held_edge_verdict(shape: dict[str, Any] | None) -> list[str]:
    """The held side of a cut chosen again on its top band (`cycle.seam_pop.held_edge`), in words, when
    the chosen cut does not close it or it could not be read; empty otherwise (and for a cut whose top
    band did not pop, where it is not read)."""
    held = (shape or {}).get("held_edge")
    if not held:
        return []
    if "skipped" in held:
        return [f"the top of the silhouette jumps at the first choice, and the side its held part is on could not be read "
                f"({held['skipped']}): whether the part closes below its top at this cut is not known"]
    chosen = held["chosen"]
    if "skipped" in chosen:
        return [f"the held part's side ({held['side']}) could not be read at this cut ({chosen['skipped']}): whether it "
                "closes below its top is not known"]
    if chosen["closes"]:
        return []
    line = (f"the held part does not close at the wrap on the {held['side']} side of the silhouette, below its top: "
            f"{chosen['seam']:.2f}x its rows' median step (over {held['reference']:g}); ")
    capped = held.get("two_step_capped")
    if capped:
        # An explicit --max-len is the caller's ceiling: windows two lengths long past it are not read.
        low, high = capped["lengths_over"]
        return [line + f"no window one length long closes it, and windows two lengths long ({low}-{high} frames) are over "
                f"--max-len {capped['max_len']}, so they were not read"]
    if not held.get("two_step", {}).get("read", True):
        # The steps were counted (video-loop --steps): the cut is the count's, nothing longer is read.
        return [line + "no window of the counted length closes it"]
    return [line + "no window one or two lengths long closes it — it swings on a beat of its own"]


def measure_jolt(frames: list[Image.Image], *, facing: str = "right") -> dict[str, Any]:
    """The jolt index of a loop as it plays, and the head's frame-to-frame moves."""
    al = coverage(frames, union_box(frames))
    whole, hair = steps(al), steps(al, hair_box(facing))
    med = float(np.median(whole))
    try:
        head: dict[str, Any] = head_track(al)
        pop: dict[str, Any] = seam_pop(al)
    except ValueError as exc:
        head = {"skipped": str(exc)}  # recorded, and the head gate says it could not read it
        pop = {"skipped": str(exc)}
    return {
        "index": round(jolt_index(whole), 4),
        "hair_index": round(jolt_index(hair), 4),
        "step_max_over_median": round(float(whole.max()) / med, 4) if med > 0 else None,
        "head": head,
        "seam_pop": pop,
    }


def seam_pop_verdict(measured: dict[str, Any], *, head_step_max: float | None = HEAD_STEP_REFERENCE) -> list[str]:
    """The top band's jump into the loop's first frame, in words, when it pops (section 3, "The seam
    pop"); empty otherwise. A sideways jump the head bound already names (`jolt_verdict`, into frame
    0) is not said twice."""
    pop = measured.get("seam_pop") or {}
    if "skipped" in pop or not pop.get("pops"):
        return []
    head = measured["head"].get("x", {})
    said = head_step_max is not None and head.get("step_max_pct", 0) > head_step_max and head.get("worst_into_frame") == 0
    axes = [(axis, pop[axis]) for axis in ("y", "x") if pop[axis]["wrap_is_largest"]
            and pop[axis]["wrap_over_median"] > pop["reference"] and not (axis == "x" and said)]
    if not axes:
        return []
    words = {"x": "sideways", "y": "up or down"}
    return ["the top of the silhouette jumps into the loop's first frame "
            + ", ".join(f"{words[a]} {v['wrap_pct']:.2f} % of the body height, {v['wrap_over_median']:.1f}x its median step" for a, v in axes)
            + f" (over {pop['reference']:g}x): a part held above the head swings on its own beat and does not close at this cut"]


def jolt_verdict(measured: dict[str, Any], *, jolt_max: float | None = JOLT_REFERENCE,
                 head_step_max: float | None = HEAD_STEP_REFERENCE) -> list[str]:
    """What exceeds the bounds, in words; empty when the loop stays inside. A bound of None is not checked."""
    over = []
    if jolt_max is not None and measured["index"] > jolt_max:
        over.append(f"jolt index {measured['index']:.3f} exceeds {jolt_max}")
    if head_step_max is None:
        return over
    if "skipped" in measured["head"]:
        over.append(f"the head could not be tracked ({measured['head']['skipped']})")
        return over
    head = measured["head"]["x"]
    if head["step_max_pct"] > head_step_max:
        over.append(f"the head moves sideways {head['step_max_pct']:.2f} % of the body height in one frame "
                    f"(into frame {head['worst_into_frame']}), over {head_step_max} %")
    return over
