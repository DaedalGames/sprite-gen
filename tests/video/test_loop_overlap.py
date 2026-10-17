# SPDX-License-Identifier: Apache-2.0
"""The overlapped wrap (docs/loop-repair.md section 2, "The overlapped wrap"): a walk cut two steps long closes for the
legs, and a tail on a beat of its own closes on no cut of two steps, so the seam gate refuses every one. The clip goes on
past the cut; the frames beside the wrap are made again between the clip's frame a cycle apart and the cut's own, and
the loop passes the gate, still two steps, its body standing where it stood.

The walker is seen from the front, 26 frames a cycle, its tail swinging round on a beat of 41 frames, in a 73-frame clip.
The interpolator is a stand-in that draws the walker between two frames the way an optical-flow in-between does: the body
at the time between them, the tail's tip on the line between its two places. Synthetic frames only."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from sprite_gen.video import loop, repair
from tests.video.test_loop_repair import _no_rife

FRAMES, CYCLE, TAIL = 73, 26, 41
REACH, LIFT, TUFT = 40, 14, 18  # the tail's tip swings round this far sideways and up and down, px; its tuft's radius


def _tip(k: float) -> tuple[float, float]:
    a = 2 * math.pi * k / TAIL
    return REACH * math.sin(a), LIFT * math.cos(a)


def _walker(body: float, tip: tuple[float, float]) -> Image.Image:
    """The walker at `body` frames into its walk, its tail's tip at `tip`: legs side by side, the nearer foot drawn lower
    each step, arms swinging opposite, and a tail from the hips ending in a tuft."""
    ph = 2 * math.pi * (body + 5) / CYCLE
    im = Image.new("RGBA", (300, 240))
    d = ImageDraw.Draw(im)
    cx, y = 150, 40 + round(2 * math.cos(2 * ph))
    drop = round(6 * math.sin(ph))
    tx, ty = tip
    d.line((cx, y + 60, cx + tx, y + 72 + ty), fill=(180, 120, 40, 255), width=18)
    d.ellipse((cx + tx - TUFT, y + 72 + ty - TUFT, cx + tx + TUFT, y + 72 + ty + TUFT), fill=(200, 140, 60, 255))
    d.rectangle((cx - 12, y + 80, cx - 1, y + 120 + drop), fill=(60, 60, 70, 255))
    d.rectangle((cx, y + 80, cx + 11, y + 120 - drop), fill=(60, 60, 70, 255))
    d.rectangle((cx - 16, y + 30, cx + 16, y + 84), fill=(200, 80, 80, 255))
    d.rectangle((cx - 12, y, cx + 12, y + 28), fill=(230, 200, 170, 255))
    sw = round(8 * math.sin(ph))
    d.rectangle((cx - 22, y + 34 + sw, cx - 16, y + 70 + sw), fill=(225, 225, 225, 255))
    d.rectangle((cx + 16, y + 34 - sw, cx + 22, y + 70 - sw), fill=(225, 225, 225, 255))
    d.point((cx - 6, y + 40 + round(body) % 11), fill=(250, 250, 250, 255))  # no two frames the same picture
    return im


def _filmed(k: int) -> Image.Image:
    return _walker(k, _tip(k))


def _key(frame: Image.Image) -> str:
    """A frame's picture wherever the canvas puts it: its pixels inside its own box."""
    return hashlib.sha256(frame.crop(frame.getbbox()).tobytes()).hexdigest()


class Between:
    """A stand-in RIFE for the walker: between two frames of the clip — wherever the loop placed them — at t, the body at
    the time t of the way from the first to the second (the shorter way round its cycle), the tail's tip t of the way along
    the line between its two places, placed where the second frame is. Records each call as (a, b, t) by clip frame, and
    refuses a frame the clip did not film."""

    def __init__(self, frames: int = FRAMES):
        self.of = {_key(_filmed(k)): k for k in range(frames)}
        self.calls: list[tuple[int, int, float]] = []

    def __call__(self, a: Image.Image, b: Image.Image, t: float) -> Image.Image:
        ka, kb = self.of[_key(a)], self.of[_key(b)]  # KeyError: made from a frame the clip did not film
        self.calls.append((ka, kb, round(t, 4)))
        apart = (ka - kb + CYCLE / 2) % CYCLE - CYCLE / 2
        (xa, ya), (xb, yb) = _tip(ka), _tip(kb)
        drawn = _walker(kb + (1 - t) * apart, (xa + t * (xb - xa), ya + t * (yb - ya)))
        # where the loop put b: its box against the box of the frame the clip filmed
        dx, dy = (p - q for p, q in zip(b.getbbox()[:2], _filmed(kb).getbbox()[:2]))
        out = Image.new("RGBA", b.size)
        out.paste(drawn, (dx, dy))
        return out


def _keyed(root: Path) -> Path:
    keyed = root / "keyed"
    keyed.mkdir(parents=True)
    for k in range(FRAMES):
        _filmed(k).save(keyed / f"{k:03}.png")
    return keyed


def _cut(root: Path, keyed: Path, *, repair_mode: str = "auto", interpolate=None) -> dict:
    """The walker cut as `video-loop --state walk --anchor motion-auto` cuts it; the report, whether it passed or not."""
    try:
        loop.run_loop(keyed, root / "out", fps=24.0, state="walk", min_len=None, max_len=None, n_out=None,
                      seam_max=loop.SEAM_RATIO_MAX, name="loop", report_path=root / "loop.json", anchor="motion-auto",
                      repair=repair_mode, interpolate=interpolate)
    except SystemExit:
        pass
    return json.loads((root / "loop.json").read_text())


def test_a_tail_on_its_own_beat_is_closed_at_the_wrap_with_the_clips_frames_a_cycle_apart(tmp_path):
    keyed = _keyed(tmp_path)
    between = Between()
    report = _cut(tmp_path, keyed, interpolate=between)
    c = report["cycle"]
    s, L = c["start"], c["length"]
    assert c["steps"] == {"count": 2, "by": "crossings"} and abs(L - CYCLE) <= 1
    overlap = report["wrap_overlap"]
    assert overlap["seam"] > loop.SEAM_RATIO_MAX  # the cut as filmed: the gate refuses it
    assert report["status"] == "passed" and report["resampled_seam_ratio"] <= loop.SEAM_RATIO_MAX
    taken = overlap["tries"][-1]
    assert taken["outcome"] == "accepted" and (overlap["width"], overlap["seam_after"]) == (taken["width"], taken["seam"])
    assert taken["seam"] <= loop.SEAM_RATIO_MAX and taken["steps"]["verdict"] == "two" and not taken["drift"]["over"]
    assert taken["rest"]["faster"] == [] and taken["faults"] == [] and taken["rushes"] == []
    assert taken["part"]["step_max_after"] < taken["part"]["step_max_before"]
    # each frame of a run shown between the clip's frame a cycle apart and the cut's own, at the weight the search chose,
    # the weights never going back toward the frame a cycle apart; RIFE asked only for a frame between the two
    assert overlap["calls"] == len(between.calls) == sum(0 < r["t"] < 1 for t in overlap["tries"] for r in t.get("made", []))
    calls = list(between.calls)
    for tried in overlap["tries"]:
        sign = 1 if tried["side"] == "after" else -1
        assert tried["frames"] == repair.overlap_run(tried["side"], tried["width"], L)
        weights = [r["t"] for r in tried.get("made", [])]
        assert weights == sorted(weights)
        for r in tried.get("made", []):
            q = r["frame"]
            late, early = (s + q + L, s + q) if sign > 0 else (s + q, s + q - L)
            if 0 < r["t"] < 1:
                ka, kb, t = calls.pop(0)
                assert (ka, kb) == (late, early) and abs(t - r["t"]) < 1e-3
    run = overlap["frames"]
    # every other frame is the clip's own, and nothing else was made
    assert report["jump_repair"]["overlapped"] == run and report["jump_repair"]["replaced"] == []
    cells = sorted((tmp_path / "out" / "cycle").glob("frame-*.png"))
    assert len(cells) == L
    filmed = {_key(_filmed(k)): k for k in range(FRAMES)}
    assert [filmed.get(_key(Image.open(p).convert("RGBA"))) for p in cells] == [None if q in run else s + q for q in range(L)]


def test_the_cut_as_filmed_is_refused_at_the_seam_with_the_repair_off(tmp_path):
    report = _cut(tmp_path, _keyed(tmp_path), repair_mode="off")
    assert report["status"] == "failed" and report["error"].startswith("video-loop: loop seam ratio")
    assert report["cycle"]["steps"] == {"count": 2, "by": "crossings"} and "wrap_overlap" not in report


def test_without_rife_the_wrap_is_cut_as_filmed_and_says_so(tmp_path, monkeypatch, capsys):
    _no_rife(monkeypatch, tmp_path)
    report = _cut(tmp_path, _keyed(tmp_path))
    assert report["status"] == "failed" and report["error"].startswith("video-loop: loop seam ratio")
    overlap = report["wrap_overlap"]
    assert overlap["frames"] == [] and overlap["install"] == "sprite-gen rife install" and overlap["calls"] == 0
    assert "the overlapped wrap was not made" in capsys.readouterr().err


def test_without_rife_and_the_repair_on_the_loop_fails_asking_for_it(tmp_path, monkeypatch):
    _no_rife(monkeypatch, tmp_path)
    report = _cut(tmp_path, _keyed(tmp_path), repair_mode="on")
    assert report["status"] == "failed" and "at a wrap a part on its own beat does not close" in report["error"]
    assert "sprite-gen rife install" in report["error"]


def test_a_made_frame_with_a_fault_refuses_the_run_and_the_wrap_is_cut_as_filmed(tmp_path, capsys):
    """A stand-in that returns nothing: every run is refused (`fault`), each width tried once, and the gate refuses the
    cut as filmed."""
    calls = []

    def empty(a: Image.Image, b: Image.Image, t: float) -> Image.Image:
        calls.append(t)
        return Image.new("RGBA", b.size)

    report = _cut(tmp_path, _keyed(tmp_path), interpolate=empty)
    assert report["status"] == "failed" and report["error"].startswith("video-loop: loop seam ratio")
    overlap = report["wrap_overlap"]
    assert overlap["frames"] == [] and [t["width"] for t in overlap["tries"]] == list(repair.OVERLAP_WIDTHS)
    assert all(t["outcome"] == "rejected" and "fault" in t["rejected_for"] for t in overlap["tries"])
    # the overlapped wrap's calls first; the jump repair then reads the cut as filmed, its wrap off the flow
    made = [r["t"] for t in overlap["tries"] for r in t["made"] if 0 < r["t"] < 1]
    assert overlap["calls"] == len(made) and calls[:len(made)] == pytest.approx(made, abs=1e-3)
    assert "the overlapped wrap was refused" in capsys.readouterr().err


def test_the_weights_ease_from_the_late_frame_to_the_early_off_the_ease_only_where_the_body_needs_it():
    """`overlap_weights` on lines drawn by hand: three frames between a late frame and an early one. Unbounded, they are
    the ease in and out (`overlap_weight`) on the grid; a window of the rest of the body that moves with the share taken
    in a step, bounded in the second step to a tenth, moves the weights off the ease there and no further; a body measure
    that may not move leaves no weighting."""
    P, G = 4, repair.OVERLAP_GRID + 1
    grid = np.linspace(0.0, 1.0, G)
    step = np.zeros((P, G, G, 2))
    step[..., 0] = 1.0 + 4.0 * np.abs(grid[None, :] - grid[:, None])  # the part, unbounded
    body = np.zeros((P, G, G, 3))
    eased = [round(repair.overlap_weight(p, P - 1) * (G - 1)) / (G - 1) for p in range(1, P)]
    free = repair.overlap_weights(step, body, [], np.zeros(0), np.ones(3))
    assert free is not None and free["weights"] == pytest.approx(eased) and free["off_ease"] < 0.5 / (G - 1) + 1e-9
    step[1, ..., 1] = np.abs(grid[None, :] - grid[:, None])  # the rest of the body, in the second step only
    held = repair.overlap_weights(step, body, [1], np.array([0.1]), np.ones(3))
    assert held is not None and held["weights"][1] - held["weights"][0] <= 0.1 + 1e-9
    assert held["weights"] == sorted(held["weights"]) and held["off_ease"] > free["off_ease"]
    body[...] = 1.0
    assert repair.overlap_weights(step, body, [], np.zeros(0), np.full(3, 0.5)) is None
