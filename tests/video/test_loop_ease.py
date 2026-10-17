# SPDX-License-Identifier: Apache-2.0
"""The eased retime (docs/loop-repair.md section 2, "The eased retime"): where a part rushes — a run of one or
two steps far faster than the steps around it, going the way it was going — the frames either side are made
again at times the search chooses, so the part's speed rises and falls gradually; the body's feet, mass and
ground line move no further in a step than they did anywhere in that stretch, nor any window of the rest of the
body further than its own largest step there, read on the frames made; the loop keeps its length, and every
frame outside the stretch is the video's own. A frame flung against the way is the flow repair's, not a rush;
an ordinary walk has none.

The loops are the walker of tests/video/test_loop_flow.py, whose tail can run short of the cycle so that it
rushes on at the wrap (`lag`). The interpolator is a stand-in that draws the walker between two frames the way
an optical-flow in-between does: the body at the time between them, the tail moved on between its two places.
Drawn over 36 frames a cycle, the stand-in's frames made at the times the search chooses step further in the
legs, the arm and the head than the line between the frames as filmed the search reads them on, and every such
stretch is refused on the frames made. The walkers eased here are drawn over 14 to 48 frames, where the frames
made keep the rest of the body within its own steps."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from PIL import Image, ImageOps

from sprite_gen.video import loop as loop_mod
from sprite_gen.video import repair
from tests.video.test_loop_flow import N, _against, _frame, _loop
from tests.video.test_loop_repair import _keyed, _no_rife

LAG = 0.25  # the tail a quarter of a cycle short: the flow repair takes one frame at the wrap, and a rush is left
WRAP = 24  # frames a cycle of the walker eased after the flow repair's frame at the wrap
SHORT, SHORT_LAG = 16, 0.15  # a walker whose tail rushes at the wrap under the flow reading's limit


def _walk(n: int = N, *, lag: float = 0.0, hold: int = 1, flung: dict[int, float] | None = None) -> list[Image.Image]:
    """`_loop` of tests/video/test_loop_flow.py with every dark part drawn whole (`dark_whole`), so that a
    frame drawn between two phases has no more dark pixels than they have (the smear reading counts them)."""
    flung = flung or {}
    drawn = [t - t % hold for t in range(n)]
    return [_frame(d, n, tail_off=_against(d, n) * flung.get(d, 0.0), tail_lag=lag * d / n, dark_whole=True) for d in drawn]


class Lerp:
    """A stand-in RIFE for the walker: between filmed frames a and b at t, the body at the time t of the way from
    a to b and the tail moved on t of the way from its place in a to its place in b (forward, as it sways). Records
    each call as (a, b, t) by the filmed frames' indices, and refuses a frame it did not film."""

    def __init__(self, filmed: list[Image.Image], *, lag: float = 0.0, hold: int = 1, n: int = N):
        self.at = {id(f): k for k, f in enumerate(filmed)}
        self.n, self.lag, self.hold = n, lag, hold
        self.calls: list[tuple[int, int, float]] = []
        self.made: list[Image.Image] = []

    def _sway(self, k: int) -> float:
        d = k - k % self.hold
        return 2 * math.pi * d * (1 - self.lag) / self.n

    def __call__(self, a: Image.Image, b: Image.Image, t: float) -> Image.Image:
        ka, kb = self.at[id(a)], self.at[id(b)]  # KeyError: made from a frame that was not filmed
        self.calls.append((ka, kb, round(t, 4)))
        da, db = ka - ka % self.hold, kb - kb % self.hold
        body = da + t * ((db - da) % self.n)
        sa, sb = self._sway(ka), self._sway(kb)
        while sb < sa:
            sb += 2 * math.pi
        sway = sa + t * (sb - sa)
        self.made.append(_frame(body, self.n, tail_lag=(2 * math.pi * body / self.n - sway) / (2 * math.pi),
                                dark_whole=True))
        return self.made[-1]


def _drift(frames: list[Image.Image], segment: list[int]) -> np.ndarray:
    """The largest step over the segment's steps, in play order, of where the body stands: the foot line's centre
    and the body's mass (`--anchor feet`'s drift reference) and the lowest solid row."""
    def where(f: Image.Image) -> np.ndarray:
        box = f.getchannel("A").point(lambda v: 255 if v >= 8 else 0).getbbox()
        rows = np.nonzero((np.asarray(f.getchannel("A")) >= 128).any(axis=1))[0]
        return np.array([loop_mod.foot_centre(f, box), loop_mod.body_centre(f, box), float(rows.max())])

    points = [where(frames[k]) for k in segment]
    return np.max(np.abs(np.diff(points, axis=0)), axis=0)


def _ease(record: dict) -> dict:
    (round_,) = [r for r in record["rounds"] if r["why"] == "ease"]
    return round_


def _rest(frames: list[Image.Image], before: list[Image.Image], after: list[Image.Image], round_: dict
          ) -> tuple[np.ndarray, np.ndarray]:
    """Every window the rush is not read in, its largest step over the stretch on the loop `before` the ease and on the
    loop `after` it: the flow reading's own steps, on the union box of `frames` (the loop as filmed) and the drawings of
    `before`."""
    box = repair.union_box(frames)
    flow = repair._flow(before, box)
    then = repair._flow(after, box, (flow["runs"], flow["hold"]))
    first = [run[0] for run in flow["runs"]]
    stretch = [first.index(k) for k in (round_["anchors"][0], *(d[0] for d in round_["drawings"]), round_["anchors"][1])]
    part = [w for w, fraction in enumerate(flow["fractions"]) if [round(v, 4) for v in fraction] in round_["part"]["windows"]]
    rest = [w for w in range(len(flow["windows"])) if w not in part]
    return flow["step"][stretch[:-1]][:, rest].max(axis=0), then["step"][stretch[:-1]][:, rest].max(axis=0)


def _kept_rest(frames: list[Image.Image], flow_only: list[Image.Image], out: list[Image.Image], round_: dict) -> None:
    """The stretch taken kept every window the rush is not read in within the largest step it took over the stretch as
    it played, read again here on the loop as it now plays, and the round says so (`rest`)."""
    before, after = _rest(frames, flow_only, out, round_)
    assert np.all(after <= before + 1e-9)
    assert round_["rest"]["faster"] == [] and len(round_["rest"]["windows"]) == len(before)
    assert all(a <= b + 1e-4 for a, b in zip(round_["rest"]["step_max_after"], round_["rest"]["step_max_before"]))


def test_a_tail_that_rushes_on_at_the_wrap_is_eased_over_the_frames_either_side():
    n = WRAP
    frames = _walk(n, lag=LAG)
    flow_only, before = repair.repair_jumps(frames, Lerp(frames, lag=LAG, n=n), ease=False)
    assert [r["why"] for r in before["rounds"]] == ["flow"]  # the flow takes one frame at the wrap, and leaves a rush
    assert before["surge"] is not None and before["surge_after"] is not None
    lerp = Lerp(frames, lag=LAG, n=n)
    out, record = repair.repair_jumps(frames, lerp)
    round_ = _ease(record)
    assert round_["outcome"] == "accepted" and round_["width"] in repair.EASE_WIDTHS
    segment = [*[round_["anchors"][0]], *round_["frames"], round_["anchors"][1]]
    assert [(k - segment[0]) % n for k in segment] == list(range(len(segment)))  # one stretch, in play order
    assert {n - 1, 0} <= set(round_["frames"]) and len(round_["frames"]) == 2 * round_["width"] + round_["surge"]["steps"] - 1
    assert len(out) == n and all(out[k] is frames[k] for k in range(n) if k not in round_["frames"])
    # every frame is made from two frames as filmed, never from a frame made — the flow's frame at the wrap is made again
    flow_frames = set(before["rounds"][0]["frames"])
    assert flow_frames <= set(round_["frames"]) and round_["supersedes"] == sorted(flow_frames)
    assert all(not {a, b} & flow_frames for a, b, _ in lerp.calls)
    # the rush is spread: the part's largest step over the stretch falls by a quarter or more, its speed changes less
    # from step to step, and no rush is left
    assert round_["part"]["step_max_after"] <= 3 / 4 * round_["part"]["step_max_before"]
    assert round_["accel"]["chosen"] < round_["accel"]["before"] / 2 and record["surge_after"] is None
    # no drift: the feet, the body's mass and the ground line move no further in a step than they did there, to within
    # the pixel they are read on; and the rest of the body no faster
    assert np.all(_drift(out, segment) <= _drift(flow_only, segment) + repair.DRIFT_PIXEL)
    _kept_rest(frames, flow_only, out, round_)
    changed = {k for k in round_["frames"] if out[k] is not frames[k]}
    assert round_["replaced"] == sorted(changed, key=round_["frames"].index) and len(changed) >= len(round_["frames"]) - 2
    assert set(record["replaced"]) == flow_frames | changed


def test_a_rush_the_flow_reading_lets_pass_is_eased_too():
    # A tail 15 % of a cycle short over 16 frames reads under the flow's limit: no frame is taken out, and it still rushes.
    frames = _walk(SHORT, lag=SHORT_LAG)
    assert repair.flow_targets(frames)["groups"] == []
    flow_only, _ = repair.repair_jumps(frames, Lerp(frames, lag=SHORT_LAG, n=SHORT), ease=False)
    out, record = repair.repair_jumps(frames, Lerp(frames, lag=SHORT_LAG, n=SHORT))
    round_ = _ease(record)
    assert round_["outcome"] == "accepted" and round_["supersedes"] == []
    assert record["surge"]["ratio"] >= repair.SURGE_RATIO and record["surge_after"] is None
    assert len(out) == SHORT and all(out[k] is frames[k] for k in range(SHORT) if k not in round_["frames"])
    _kept_rest(frames, flow_only, out, round_)


def test_a_stretch_whose_frames_made_hurry_the_rest_of_the_body_is_refused():
    # Over 36 frames, easing a tail a tenth of a cycle short gives the rush more time and the steps around it less; the
    # search reads the rest of the body on the line between the frames as filmed and finds times that keep it, and the
    # frames made, which go the body's own way between them, step further there (a validator's counterexample on 9dd5050:
    # the arm's window 1.736 to 2.030, the stretch taken). Read on the frames made, the stretch is refused for that alone,
    # and the loop stays as the flow left it.
    frames = _walk(lag=0.1)
    flow_only, _ = repair.repair_jumps(frames, Lerp(frames, lag=0.1), ease=False)
    out, record = repair.repair_jumps(frames, Lerp(frames, lag=0.1))
    round_ = _ease(record)
    assert round_["outcome"] == "rejected" and round_["rejected_for"] == ["the rest faster"]
    faster = [i for i, w in enumerate(round_["rest"]["windows"]) if w in round_["rest"]["faster"]]
    assert faster and all(round_["rest"]["step_max_after"][i] > round_["rest"]["step_max_before"][i] for i in faster)
    assert [f.tobytes() for f in out] == [f.tobytes() for f in flow_only] and record["replaced"] == []
    assert record["surge_after"] is not None and "replaced" not in round_


def test_an_ordinary_walk_has_no_rush_and_nothing_is_made():
    for frames in (_loop(), _loop(hold=2), _walk(), _walk(hold=2)):
        assert repair.surges(frames) == []
        out, record = repair.repair_jumps(frames, lambda a, b, t: pytest.fail("nothing to make"))
        assert record["surge"] is None and all(o is f for o, f in zip(out, frames))
        assert not [r for r in record["rounds"] if r["why"] == "ease"]


def test_a_frame_flung_against_the_way_is_not_a_rush():
    # Its steps in and out are large, and the second goes back the way the first came: the flow repair's frame.
    assert repair.surges(_loop(flung={12: 12})) == [] and repair.surges(_walk(flung={12: 12})) == []


def test_on_twos_each_drawing_is_made_once_and_shown_for_both_its_frames():
    n = 48  # 24 drawings on twos
    frames = _walk(n, lag=LAG, hold=2)
    flow_only, _ = repair.repair_jumps(frames, Lerp(frames, lag=LAG, hold=2, n=n), ease=False)
    lerp = Lerp(frames, lag=LAG, hold=2, n=n)
    out, record = repair.repair_jumps(frames, lerp)
    round_ = _ease(record)
    assert record["hold"] == 2 and round_["outcome"] == "accepted"
    assert all(len(d) == 2 and out[d[0]] is out[d[1]] for d in round_["drawings"])
    assert len(lerp.calls) <= len(round_["drawings"]) + len(record["rounds"][0]["drawings"])
    assert len(out) == n and all(out[k] is frames[k] for k in range(n) if k not in round_["frames"])
    _kept_rest(frames, flow_only, out, round_)


def test_a_made_frame_that_moves_the_body_further_than_the_stretch_did_refuses_the_ease():
    n = WRAP
    frames = _walk(n, lag=LAG)
    flow_only, _ = repair.repair_jumps(frames, Lerp(frames, lag=LAG, n=n), ease=False)
    lerp = Lerp(frames, lag=LAG, n=n)

    def pushed(a: Image.Image, b: Image.Image, t: float) -> Image.Image:
        made = lerp(a, b, t)
        if len(lerp.calls) <= 1:  # the flow's frame at the wrap, as before
            return made
        moved = Image.new("RGBA", made.size, (0, 0, 0, 0))
        moved.paste(made, (12 if len(lerp.calls) % 2 else 0, 0))  # every other made frame pushed sideways
        return moved

    out, record = repair.repair_jumps(frames, pushed)
    round_ = _ease(record)
    assert round_["outcome"] == "rejected" and "drift" in round_["rejected_for"] and round_["drift"]["over"]
    assert [f.tobytes() for f in out] == [f.tobytes() for f in flow_only]  # the loop as the flow left it
    assert set(record["replaced"]) == set(record["rounds"][0]["frames"])


def test_a_made_frame_at_fault_refuses_the_ease_and_the_loop_stays_as_the_flow_left_it():
    n = WRAP
    frames = _walk(n, lag=LAG)
    flow_only, _ = repair.repair_jumps(frames, Lerp(frames, lag=LAG, n=n), ease=False)
    lerp = Lerp(frames, lag=LAG, n=n)

    def melted(a: Image.Image, b: Image.Image, t: float) -> Image.Image:
        made = lerp(a, b, t)
        if len(lerp.calls) != 3:
            return made
        pixels = np.asarray(made).copy()
        dark = (pixels[..., :3].max(axis=2) < 70) & (pixels[..., 3] > 0)
        pixels[dark, :3] = (200, 200, 220)  # the outlines melted into the fill
        return Image.fromarray(pixels, "RGBA")

    out, record = repair.repair_jumps(frames, melted)
    round_ = _ease(record)
    assert round_["outcome"] == "rejected" and "outline" in round_["faults"] and "fault" in round_["rejected_for"]
    assert [f.tobytes() for f in out] == [f.tobytes() for f in flow_only]


def test_without_an_interpolator_a_rush_is_named_and_nothing_is_made():
    frames = _walk(lag=0.1)
    out, record = repair.repair_jumps(frames, None)
    assert record["surge"] is not None and record["ease_skipped"] == "no interpolator"
    assert all(o is f for o, f in zip(out, frames)) and record["replaced"] == []


def test_it_eases_the_same_frames_whichever_way_the_body_faces():
    frames = _walk(SHORT, lag=SHORT_LAG)
    mirrored = [ImageOps.mirror(f) for f in frames]
    lerp = Lerp(frames, lag=SHORT_LAG, n=SHORT)
    index = {id(m): k for k, m in enumerate(mirrored)}

    def mirror_lerp(a: Image.Image, b: Image.Image, t: float) -> Image.Image:
        return ImageOps.mirror(lerp(frames[index[id(a)]], frames[index[id(b)]], t))

    _, record = repair.repair_jumps(frames, Lerp(frames, lag=SHORT_LAG, n=SHORT))
    _, flipped = repair.repair_jumps(mirrored, mirror_lerp)
    assert _ease(flipped)["frames"] == _ease(record)["frames"] and _ease(flipped)["outcome"] == "accepted"
    assert _ease(record)["outcome"] == "accepted"


# --- the search (repair.ease_times) -------------------------------------------------------------------------------


def test_the_search_keeps_every_frame_at_its_own_time_where_the_part_moves_evenly():
    times = np.arange(0, 8.0 + 1e-9, 1 / repair.EASE_GRID)
    progress = times[None, :] * 1.0  # one usual step a frame
    body = np.stack([times * 2.0, times * 0.5, np.zeros_like(times)])  # the body walks on evenly
    result = repair.ease_times(times, progress, body, np.array([2.0, 0.5, 1.0]), np.array([1.0]), np.array([1.0]),
                               own=list(range(1, 8)))
    assert result["accel"] == pytest.approx(0.0, abs=1e-9) and result["times"] == pytest.approx(list(range(1, 8)))


def test_the_search_spreads_a_rush_as_far_as_the_body_lets_it():
    # The part covers nine usual steps in the fourth of eight steps, one in each of the others. The body moves one
    # px a step and may move up to two: the times can bunch round the rush, but no step may cover more than two.
    times = np.arange(0, 8.0 + 1e-9, 1 / repair.EASE_GRID)
    per_step = np.array([1, 1, 1, 9, 1, 1, 1, 1], dtype=float)
    progress = np.interp(times, np.arange(9), np.concatenate([[0], np.cumsum(per_step)]))[None, :]
    body = np.stack([times, np.zeros_like(times), np.zeros_like(times)])
    result = repair.ease_times(times, progress, body, np.array([2.0, 1.0, 1.0]), np.array([1.0]), np.array([1.0]),
                               own=list(range(1, 8)))
    chosen = np.array([0.0, *result["times"], 8.0])
    assert np.all(np.diff(chosen) <= 2.0 + 1e-9) and np.all(np.diff(chosen) > 0)
    steps = np.diff(np.interp(chosen, times, progress[0]))
    assert steps.max() < 9 and result["accel"] < 8  # spread over more than one step
    assert np.all(np.abs(np.diff(np.concatenate([[1.0], steps, [1.0]]))) <= result["accel"] + 1e-9)


# --- video-loop --repair auto (sprite_gen/video/loop.py) ---------------------------------------------------------


class NearestLerp(Lerp):
    """`Lerp` for frames as video-loop hands them over (read back from disk and cleaned): each is the filmed frame
    nearest it."""

    def __init__(self, filmed: list[Image.Image], **kw):
        super().__init__(filmed, **kw)
        self.filmed = [np.asarray(f, dtype=np.int16) for f in filmed]

    def __call__(self, a: Image.Image, b: Image.Image, t: float) -> Image.Image:
        near = [int(np.argmin([np.abs(np.asarray(x, dtype=np.int16) - f).mean() for f in self.filmed])) for x in (a, b)]
        self.at = {id(a): near[0], id(b): near[1]}
        return super().__call__(a, b, t)


def _cut(tmp_path, frames: list[Image.Image], **kw):
    return loop_mod.run_loop(_keyed(tmp_path, frames), tmp_path / "out", fps=24.0, state="walk", min_len=None, max_len=None,
                             n_out=None, seam_max=1000.0, name="w", report_path=tmp_path / "w.json", cycle_mode="fixed",
                             start=0, length=len(frames), anchor="none", **kw)


def test_video_loop_eases_the_rush_and_plays_the_frames_made(tmp_path, capsys):
    frames = _walk(SHORT, lag=SHORT_LAG)
    lerp = NearestLerp(frames, lag=SHORT_LAG, n=SHORT)
    rep = _cut(tmp_path, frames, interpolate=lerp)
    jr = rep["jump_repair"]
    round_ = _ease(jr)
    assert round_["outcome"] == "accepted" and jr["applied"] is True and jr["surge_after"] is None
    assert set(round_["replaced"]) <= set(jr["replaced"]) and rep["seam_measurement"] == "rendered-cells"
    cells = sorted((tmp_path / "out" / "cycle").glob("frame-*.png"))
    assert len(cells) == SHORT  # the loop keeps its length
    made = {k: m for m in round_["made"] if m["proposal"] is not None for k in m["frames"]}
    assert made and len(made) == len(lerp.made)
    for k, image in zip(sorted(made, key=round_["frames"].index), lerp.made):
        assert np.array_equal(np.asarray(Image.open(cells[k])), np.asarray(image))
    assert json.loads((tmp_path / "w.json").read_text())["jump_repair"]["rounds"][-1]["why"] == "ease"


def test_a_rush_without_rife_is_cut_as_filmed_with_a_warning(tmp_path, monkeypatch, capsys):
    _no_rife(monkeypatch, tmp_path)
    frames = _walk(lag=0.1)
    rep = _cut(tmp_path, frames)
    jr = rep["jump_repair"]
    assert jr["applied"] is False and jr["replaced"] == [] and jr["ease_skipped"] == "no interpolator"
    assert jr["why"] == "RIFE not installed — the rush cut as filmed" and jr["install"] == "sprite-gen rife install"
    warning = [line for line in capsys.readouterr().err.splitlines() if "rushes" in line]
    assert len(warning) == 1 and "sprite-gen rife install" in warning[0] and "not eased" in warning[0]


def test_repair_on_without_rife_refuses_a_rush(tmp_path, monkeypatch):
    _no_rife(monkeypatch, tmp_path)
    with pytest.raises(SystemExit, match=r"for a part that rushes.*sprite-gen rife install"):
        _cut(tmp_path, _walk(lag=0.1), repair="on")


def test_a_stretch_never_leaves_more_than_half_the_loop_made(monkeypatch):
    # Fourteen frames: four drawings a side would remake eight, more than the seven video-loop-compare reads back as the
    # repair's; three a side remakes six.
    frames = _walk(14, lag=0.15)
    monkeypatch.setattr(repair, "MADE_SHARE", 1.0)
    _, unbound = repair.repair_jumps(frames, Lerp(frames, lag=0.15, n=14))
    assert _ease(unbound)["width"] == 4 and len(_ease(unbound)["frames"]) == 8
    monkeypatch.setattr(repair, "MADE_SHARE", 0.5)
    flow_only, _ = repair.repair_jumps(frames, Lerp(frames, lag=0.15, n=14), ease=False)
    out, record = repair.repair_jumps(frames, Lerp(frames, lag=0.15, n=14))
    round_ = _ease(record)
    assert round_["outcome"] == "accepted" and round_["width"] == 3 and len(record["replaced"]) <= 14 // 2
    _kept_rest(frames, flow_only, out, round_)
