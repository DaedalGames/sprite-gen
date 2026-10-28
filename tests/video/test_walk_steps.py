# SPDX-License-Identifier: Apache-2.0
"""A walk is cut two steps long — the left foot forward once and the right once — wherever the clip shows its steps
(docs/loop-repair.md section 3, "The step count"; sprite_gen/video/crossings.py, local_cycle.by_steps).

Each walker is drawn so the pixel search alone makes one of the faults a walk loop is cut with:

- a diagonal walk with its legs drawn alike and a tail on its own beat, cut one step long (a half step loop:
  one foot goes forward);
- a walk slower than the search's window, cut one step long;
- a front walk with a big tail on its own beat, cut three steps long by the gait fallback (the wrap puts the
  nearer foot back: a stomp in place);
- a slow front walk in a short clip, refused outright (no periodic cycle found);
- a walk that stalls, its cut playing the stalled step.

Two whose feet meet on the screen once a cycle, read as cycles (`local_cycle.read_cycles`) and cut one cycle, their
body rising and falling twice an interval between crossings — the witness that reads no leg: a walk seen from behind
whose far foot passes hidden behind the near one, on ones, twos or threes, a cycle a whole number of frames or not,
its legs drawn apart or alike (the parent engine counted the cycle one step and cut two cycles, or kept the search's
cut unwritten); and a walk on a front diagonal whose near foot steps out toward the camera and passes the far one
apart, read so where its view is given. The guards: from the side, and where the view's foot cue says the other leg
leads an interval on, the count stands however alike a step on draws; and where nothing that reads no leg witnesses
two steps an interval — legs drawn alike from any view, alike and wide as a hidden pass is, alike and wide under a
light fixed on the screen, the swing foot lifted under such a light, a far leg a few tones off with no view, and a
walk whose feet do meet once a cycle but whose body does not bob — the crossings stay on steps, named in a warning.
And one the count alone gets wrong, where the search's cut stands: a walk on threes whose tail closes on no cut of
two steps.

A walk the search already cuts two steps long, joined at the walk's own point, is cut exactly as before; a body
whose feet cannot be read is cut as before. Synthetic frames only."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from conftest import feet_unseen
from sprite_gen.video import crossings, legs, local_cycle, loop

FRAMES = 73


def _side(k: float, *, cycle: float, phase: float = 0, tail_cycle: float = 1, tail: int = 0, arms: bool = True) -> Image.Image:
    """A side or diagonal walker: legs drawn alike, half a cycle apart and seen apart for most of a step; arms opposite,
    the far one darker (`arms`); a tail swinging up and down behind on its own beat (`tail_cycle`, `tail` px long)."""
    ph = 2 * math.pi * (k + phase) / cycle
    im = Image.new("RGBA", (300, 220))
    d = ImageDraw.Draw(im)
    x, y = 90, 40 + round(3 * math.cos(2 * ph))
    sw = 16 * math.sin(ph)
    if arms:
        d.rectangle((x + 14 - sw - 4, y + 34, x + 14 - sw + 4, y + 72), fill=(150, 150, 150, 255))
    for sign in (1, -1):
        lx = x + 14 + 14 * sign * math.sin(ph)
        d.rectangle((lx - 5, y + 80, lx + 5, y + 126), fill=(60, 60, 70, 255))
    if tail:
        a = 0.8 * math.sin(2 * math.pi * k / tail_cycle)
        d.line((x, y + 50, x - tail * math.cos(a), y + 50 - tail * math.sin(a)), fill=(180, 120, 40, 255), width=12)
    d.rectangle((x, y + 30, x + 28, y + 84), fill=(200, 80, 80, 255))
    d.rectangle((x + 2, y, x + 26, y + 28), fill=(230, 200, 170, 255))
    if arms:
        d.rectangle((x + 14 + sw - 4, y + 34, x + 14 + sw + 4, y + 72), fill=(225, 225, 225, 255))
    d.point((x + 6, y + 40 + int(k) % 11), fill=(250, 250, 250, 255))  # no two frames the same picture
    return im


def _front(k: float, *, cycle: float, phase: float = 0, tail_cycle: float = 1, tail: int = 0) -> Image.Image:
    """A walker seen from the front: legs side by side and touching (never seen apart), the nearer foot drawn lower
    each step; arms swinging opposite; a tail swinging sideways at the hips on its own beat."""
    ph = 2 * math.pi * (k + phase) / cycle
    im = Image.new("RGBA", (260, 220))
    d = ImageDraw.Draw(im)
    cx, y = 130, 40 + round(2 * math.cos(2 * ph))
    drop = round(6 * math.sin(ph))
    if tail:
        a = math.sin(2 * math.pi * k / tail_cycle)
        d.line((cx, y + 72, cx + tail * a, y + 90), fill=(180, 120, 40, 255), width=14)
    d.rectangle((cx - 12, y + 80, cx - 1, y + 120 + drop), fill=(60, 60, 70, 255))
    d.rectangle((cx, y + 80, cx + 11, y + 120 - drop), fill=(60, 60, 70, 255))
    d.rectangle((cx - 16, y + 30, cx + 16, y + 84), fill=(200, 80, 80, 255))
    d.rectangle((cx - 12, y, cx + 12, y + 28), fill=(230, 200, 170, 255))
    sw = round(8 * math.sin(ph))
    d.rectangle((cx - 22, y + 34 + sw, cx - 16, y + 70 + sw), fill=(225, 225, 225, 255))
    d.rectangle((cx + 16, y + 34 - sw, cx + 22, y + 70 - sw), fill=(225, 225, 225, 255))
    d.point((cx - 6, y + 40 + int(k) % 11), fill=(250, 250, 250, 255))
    return im


def _behind(k: int, *, cycle: float, hold: int, alike: bool = False, bob: int = 3) -> Image.Image:
    """A walker seen from behind on a diagonal, drawn on twos or threes (`hold`): the near leg swings ahead and back,
    and the far one is seen beside it only while it is behind — ahead, it stands hidden behind the near leg. The feet
    are seen apart once a cycle and together through the other step, its pass hidden among them: one crossing shows a
    cycle. The body bobs `bob` px once a step; the far leg darker (`alike`: drawn as the near one); arms opposite, the
    far one darker; each repeat of a drawing redrawn a little (a pixel aside, a few tones)."""
    redraw = k % hold
    ph = 2 * math.pi * (k - redraw) / cycle
    im = Image.new("RGBA", (300, 220))
    d = ImageDraw.Draw(im)
    x, y = 90 + redraw, 40 + round(bob * math.cos(2 * ph))
    sw = 16 * math.sin(ph)
    near = x + 14 + sw
    far = x + 14 - sw if sw > 0 else near
    d.rectangle((x + 14 - sw - 4, y + 34, x + 14 - sw + 4, y + 72), fill=(150, 150, 150, 255))
    d.rectangle((far - 5, y + 80, far + 5, y + 126), fill=(70, 70, 80, 255) if alike else (50, 50, 60, 255))
    d.rectangle((near - 5, y + 80, near + 5, y + 126), fill=(70, 70, 80, 255))
    d.rectangle((x, y + 30, x + 28, y + 84), fill=(200 + 12 * redraw, 80, 80, 255))
    d.rectangle((x + 2, y, x + 26, y + 28), fill=(230, 200, 170, 255))
    d.rectangle((x + 14 + sw - 4, y + 34, x + 14 + sw + 4, y + 72), fill=(225, 225, 225, 255))
    return im


def _outward(k: float, *, cycle: float, bob: int = 3) -> Image.Image:
    """A walker on a front diagonal whose near foot steps out toward the camera: it swings wide ahead and back, and the
    far foot hardly moves, so the feet meet on the screen once a cycle, where the near foot passes the far one on its
    way back; its pass on the way out is made apart. The body bobs `bob` px once a step; arms opposite, the far one
    darker; the far leg a tone off the near."""
    ph = 2 * math.pi * k / cycle
    im = Image.new("RGBA", (300, 220))
    d = ImageDraw.Draw(im)
    x, y = 90, 40 + round(bob * math.cos(2 * ph))
    sw = 16 * math.sin(ph)
    near, far = x + 14 + sw, x + 34 - 4 * math.sin(ph)
    d.rectangle((x + 14 - sw - 4, y + 34, x + 14 - sw + 4, y + 72), fill=(150, 150, 150, 255))
    d.rectangle((far - 5, y + 80, far + 5, y + 126), fill=(61, 61, 71, 255))
    d.rectangle((near - 5, y + 80, near + 5, y + 126), fill=(60, 60, 70, 255))
    d.rectangle((x, y + 30, x + 28, y + 84), fill=(200, 80, 80, 255))
    d.rectangle((x + 2, y, x + 26, y + 28), fill=(230, 200, 170, 255))
    d.rectangle((x + 14 + sw - 4, y + 34, x + 14 + sw + 4, y + 72), fill=(225, 225, 225, 255))
    return im


def _alike(k: float, *, cycle: float, phase: float = 0, tone: int = 1, lean: float = 0, leg_tone: int | None = None,
           leg_half: int = 5, lift: int = 0, light: tuple[str, int] | None = None, bob: int = 3) -> Image.Image:
    """A side walker whose two steps are the same silhouette: the far leg and arm `tone` tones lighter than the near
    ones (the far leg `leg_tone`, where given: 0 draws the legs alike), the arms swinging `lean` px further back than
    forward — nothing else tells a step from the next. Each leg is drawn `leg_half` px either side of its line: at 10
    the feet are seen together over half the interval at a pass, as where the far foot passes hidden. The leg swinging
    forward is lifted up to `lift` px, most at its pass; `light` lays a light fixed on the screen over the legs
    (`_lit`). The body bobs `bob` px once a step."""
    ph = 2 * math.pi * (k + phase) / cycle
    im = Image.new("RGBA", (300, 220))
    d = ImageDraw.Draw(im)
    drawn = Image.new("L", im.size)
    x, y = 90, 40 + round(bob * math.cos(2 * ph))
    for sign, t in ((1, 0), (-1, tone)):
        sw = sign * (16 * math.sin(ph) + lean)
        d.rectangle((x + 14 - sw - 4, y + 34, x + 14 - sw + 4, y + 72), fill=(190 + t, 190 + t, 190 + t, 255))
    for sign, t in ((1, 0), (-1, tone if leg_tone is None else leg_tone)):
        lx = x + 14 + 14 * sign * math.sin(ph)
        up = round(lift * max(0.0, sign * math.cos(ph)))
        box = (lx - leg_half, y + 80 - up, lx + leg_half, y + 126 - up)
        d.rectangle(box, fill=(60 + t, 60 + t, 70 + t, 255))
        ImageDraw.Draw(drawn).rectangle(box, fill=255)
    d.rectangle((x, y + 30, x + 28, y + 84), fill=(200, 80, 80, 255))
    ImageDraw.Draw(drawn).rectangle((x, y + 30, x + 28, y + 84), fill=0)
    d.rectangle((x + 2, y, x + 26, y + 28), fill=(230, 200, 170, 255))
    return _lit(im, drawn, light)


def _lit(im: Image.Image, legs_drawn: Image.Image, light: tuple[str, int] | None) -> Image.Image:
    """`im` under a light fixed on the screen, laid over the pixels drawn as legs (`legs_drawn`), both legs alike:
    `("x", n)` up to n tones lighter to the right across the feet, `("y", n)` up to n tones lighter lower down the legs.
    It lights a foot by where it is drawn on the screen, not by which leg it is."""
    if light is None:
        return im
    axis, tones = light
    a = np.array(im)
    h, w = a.shape[:2]
    field = (np.clip((np.arange(w) - 80) * tones // 48, 0, tones)[None, :].repeat(h, 0) if axis == "x" else
             np.clip((np.arange(h) - 120) * tones // 46, 0, tones)[:, None].repeat(w, 1))
    on = np.asarray(legs_drawn) > 0
    a[..., :3][on] = np.clip(a[..., :3][on].astype(int) + field[on][:, None], 0, 255).astype(np.uint8)
    return Image.fromarray(a)


def _carried(draw, motion: str, room: int = 60):
    """`draw`'s walker carried up and down the screen with the whole picture (a motion of the picture, not the walk's)
    on a canvas `room` px taller either side: `"arc"` along a smooth arc, 24 px down at either end of the 73 frames
    and level in the middle; `"cubic"` from 48 px up at the start to 48 down at the end, still in the middle;
    `"sway"` 12 px down, level, 12 up and back every 92 frames; `"ramp"` 24 px down over the clip, a row every three
    frames."""
    move = {"arc": lambda k: round(24 * ((k - 36) / 36) ** 2), "cubic": lambda k: round(48 * ((k - 36) / 36) ** 3),
            "sway": lambda k: round(12 * math.sin(2 * math.pi * k / 92)), "ramp": lambda k: round(24 * k / 72)}[motion]

    def frame(k: int) -> Image.Image:
        im = draw(k)
        out = Image.new("RGBA", (im.width, im.height + 2 * room))
        out.paste(im, (0, room + move(k)))
        return out
    return frame


def _tailed(k: int) -> Image.Image:
    """A side walker on threes, 30 frames a cycle, its legs hurrying through each pass and slowing apart, with a big
    tail swinging on a beat of its own (34 frames); each repeat of a drawing redrawn a few tones."""
    redraw = k % 3
    t = k - redraw
    u = 2 * math.pi * t / 30
    ph = u - 0.25 * math.sin(u)
    im = Image.new("RGBA", (300, 220))
    d = ImageDraw.Draw(im)
    x, y = 90, 40 + round(3 * math.cos(2 * ph))
    sw = 16 * math.sin(ph)
    d.rectangle((x + 14 - sw - 4, y + 34, x + 14 - sw + 4, y + 72), fill=(150, 150, 150, 255))
    for sign in (1, -1):
        lx = x + 14 + 14 * sign * math.sin(ph)
        d.rectangle((lx - 5, y + 80, lx + 5, y + 126), fill=(60, 60, 70, 255))
    a = 0.8 * math.sin(2 * math.pi * t / 34)
    d.line((x, y + 50, x - 70 * math.cos(a), y + 50 - 70 * math.sin(a)), fill=(180, 120, 40, 255), width=12)
    d.rectangle((x, y + 30, x + 28, y + 84), fill=(200 + 3 * redraw, 80, 80, 255))
    d.rectangle((x + 2, y, x + 26, y + 28), fill=(230, 200, 170, 255))
    d.rectangle((x + 14 + sw - 4, y + 34, x + 14 + sw + 4, y + 72), fill=(225, 225, 225, 255))
    return im


def _blob(k: int) -> Image.Image:
    """A legless body bouncing every 16 frames: one foot band, nothing to step."""
    im = Image.new("RGBA", (200, 200))
    h = 60 + round(8 * math.cos(2 * math.pi * k / 16))
    ImageDraw.Draw(im).ellipse((70, 170 - h, 130, 170), fill=(90, 160, 220, 255))
    ImageDraw.Draw(im).point((90, 150 - k % 9), fill=(250, 250, 250, 255))
    return im


def _keyed(root: Path, draw, frames: int = FRAMES, hold: tuple[int, int] | None = None) -> Path:
    """The walker's frames; `hold` (from, frames) holds the walk still that long at `from`."""
    keyed = root / "keyed"
    keyed.mkdir(parents=True)
    for k in range(frames):
        t = k if hold is None or k < hold[0] else hold[0] if k < hold[0] + hold[1] else k - hold[1]
        draw(t).save(keyed / f"{k:03}.png")
    return keyed


def _cut(out: Path, keyed: Path, *extra: str) -> dict:
    assert loop.main(["--frames-dir", str(keyed), "--out-dir", str(out), "--state", "walk",
                      "--anchor", "motion-auto", "--repair", "off", *extra]) == 0
    return json.loads((out / "loop.loop.report.json").read_text())


def _two_steps(report: dict, cycle: float) -> dict:
    """The cut covers two steps by the clip's crossings — one cycle of the walker, `cycle` frames — and says so."""
    c = report["cycle"]
    counted = c["step_count"]
    assert counted["readable"] and counted["chosen_by"] == "two steps" and c["method"] == "observed-two-steps-v1"
    assert counted["cut"]["verdict"] == "two" and counted["cut"]["k"] == [2] and counted["cut"]["ring"] == 2
    assert abs(counted["cut"]["phase"]) <= crossings.PHASE_TOLERANCE
    assert abs(c["length"] - cycle) <= 1.5
    assert c["steps"] == {"count": 2, "by": "crossings"}
    assert report["status"] == "passed"
    return counted


@pytest.mark.slow
def test_a_half_step_diagonal_walk_is_cut_two_steps(tmp_path):
    """Legs drawn alike, 14.5 frames a step, a tail on a 45-frame beat: one step on, the legs are swapped and the
    tail is nearer where it was than a cycle on, so the search takes 14 frames — one foot goes forward. The
    crossings count it one step; the cut is two."""
    report = _cut(tmp_path / "out", _keyed(tmp_path, lambda k: _side(k, cycle=29, tail_cycle=45, tail=60, arms=False)))
    counted = _two_steps(report, 29)
    assert counted["mode"] == "gap" and counted["step_range"][1] == 14.5
    assert counted["search"]["verdict"] == "refuted" and counted["search"]["k"] == [1] and counted["search"]["cut"][1] in (14, 15)
    # Its feet pass briefly and no view is given: whether its crossings mark cycles is not asked.
    assert "cycles" not in counted
    strip = json.loads((tmp_path / "out" / "loop.strip.json").read_text())
    assert strip["steps"] == 2 and strip["cycle_frames"] == report["cycle"]["length"]
    # The step screen's pixel suspicion of a cycle twice as long is settled by the count.
    screen = report["cycle"]["step_screen"]
    assert screen["suspect"] is False and (screen.get("settled_by") == "crossings" or "pixel_suspect" not in screen)


@pytest.mark.slow
def test_a_walk_slower_than_the_window_is_cut_two_steps(tmp_path):
    """A 46-frame cycle (1.92 s) in a 73-frame clip: past the window's 36, so the search cut a 23-frame step. Two
    steps are 46 frames, and the clip holds 0.59 of a cycle past them."""
    report = _cut(tmp_path / "out", _keyed(tmp_path, lambda k: _side(k, cycle=46, phase=13)))
    counted = _two_steps(report, 46)
    assert report["window"][1] < report["cycle"]["length"]
    assert counted["search"]["k"] == [1]
    assert report["cycle"]["coverage"] == round((FRAMES - report["cycle"]["length"]) / report["cycle"]["length"], 4) > loop.COVERAGE_MIN


@pytest.mark.slow
def test_a_tail_on_its_own_beat_no_longer_cuts_three_steps(tmp_path):
    """A front walk, 13.5 frames a step, its big tail on a 41-frame beat: the search refuses the walk's 27 frames
    (repeat error over two steps), and the gait fallback takes the tail's 41 — three steps, the wrap putting the
    nearer foot back: a stomp in place. The crossings count three and cut two."""
    report = _cut(tmp_path / "out", _keyed(tmp_path, lambda k: _front(k, cycle=27, phase=5, tail_cycle=41, tail=70)))
    counted = _two_steps(report, 27)
    assert counted["mode"] == "sign" and counted["alternation"] <= crossings.ALTERNATION_MAX
    assert "gait_fallback" in report and counted["search"]["k"] == [3] and counted["search"]["cut"][1] in (40, 41, 42)


@pytest.mark.slow
def test_a_slow_walk_in_a_short_clip_the_search_refuses_is_cut_two_steps(tmp_path):
    """A 34-frame front walk with a tail on a 52-frame beat: no window repeats within two steps of its own, so the
    search and the gait fallback find no cycle. The crossings show two."""
    report = _cut(tmp_path / "out", _keyed(tmp_path, lambda k: _front(k, cycle=34, phase=5, tail_cycle=52, tail=70)))
    counted = _two_steps(report, 34)
    assert counted["search"]["refused"].startswith("no periodic cycle found")


@pytest.mark.slow
def test_a_walk_that_stalls_is_not_cut_across_the_stalled_step(tmp_path):
    """A side walk held still 20 frames: the step the hold is in is far longer than the walk's own, and no frame of
    it is played (the search's cut played its last frames, and its seam did not close)."""
    report = _cut(tmp_path / "out", _keyed(tmp_path, lambda k: _side(k, cycle=29, phase=3), frames=96, hold=(30, 20)))
    counted = _two_steps(report, 29)
    c = [x if isinstance(x, (int, float)) else x[1] for x in counted["crossings"]]
    hi = counted["step_range"][2]
    stalled = [(a, b) for a, b in zip(c, c[1:]) if b - a > hi]
    assert stalled == [(26.0, 60.5)]
    s, L = report["cycle"]["start"], report["cycle"]["length"]
    assert s + L - 1 < 26 or s > 60.5
    assert counted["search"]["code"] == "step-long"


def _one_cycle(out: Path, cycle: float) -> dict:
    """The cut in `out` is one cycle of the walker, `cycle` frames, two steps by its crossings read as cycles, and says
    so — in its strip too."""
    report = json.loads((out / "loop.loop.report.json").read_text())
    c = report["cycle"]
    counted = c["step_count"]
    assert abs(c["length"] - cycle) <= 1
    assert c["steps"] == {"count": 2, "by": "crossings"}
    assert counted["mode"] == "gap" and counted["cycles"]["cycles"] is True
    assert abs(counted["step_range"][1] - cycle / 2) <= 1
    assert counted["cut"]["verdict"] == "two" and counted["cut"]["k"] == [2] and counted["cut"]["ring"] == 2
    assert abs(counted["cut"]["phase"]) <= crossings.PHASE_TOLERANCE
    assert json.loads((out / "loop.strip.json").read_text())["steps"] == 2
    assert report["status"] == "passed"
    return counted


@pytest.mark.slow
@pytest.mark.parametrize("cycle, hold", [(20, 2), (24, 3), (17.5, 1), (18.5, 1)])
def test_a_walk_whose_far_foot_hides_is_cut_one_cycle(tmp_path, cycle, hold):
    """Seen from behind on a diagonal, the far foot passes hidden behind the near one: the feet are seen together over
    more than half the interval between two crossings, one crossing a cycle. The clip one interval on is the same
    picture, a cycle on, and the body rises and falls twice an interval — a step each rise: the count reads the
    crossings as cycles, two steps an interval. The cut is one cycle, `steps: 2` (the parent engine counted the cycle
    one step: on twos and threes it kept the search's cut unwritten, and on ones, a cycle a whole number of frames or
    not, cut two cycles, 35 and 37 frames). With no view, no foot cue is read."""
    report = _cut(tmp_path / "out", _keyed(tmp_path, lambda k: _behind(k, cycle=cycle, hold=hold)))
    counted = _one_cycle(tmp_path / "out", cycle)
    said = counted["cycles"]
    assert report["drawings"]["hold"] == hold
    assert said["asked"].startswith("the feet together") and said["overlap"] > crossings.HIDE_SHARE
    assert said["one"] < said["half"] and "cue" not in said and "unwitnessed" not in said
    assert said["bob"]["says"] == "cycles" and said["bob"]["half"] >= local_cycle.BOB_TWICE and said["bob"]["swing"] >= local_cycle.BOB_FLOOR
    assert "rises and falls twice an interval" in said["why"]


@pytest.mark.slow
@pytest.mark.parametrize("view", [None, "back_diagonal"])
@pytest.mark.parametrize("cycle, hold", [(17.5, 1), (20, 2)])
def test_a_walk_whose_far_foot_hides_is_cut_one_cycle_however_alike_its_legs(tmp_path, view, cycle, hold):
    """The same walker with its far leg drawn as the near one: the view's shade of the feet does not move (`alike`) and
    says nothing, and the body's bob, which reads no leg, still witnesses two steps an interval: one cycle (the parent
    engine left legs drawn alike to the cue on steps and cut two cycles, with a warning)."""
    _cut(tmp_path / "out", _keyed(tmp_path, lambda k: _behind(k, cycle=cycle, hold=hold, alike=True)), *(("--view", view) if view else ()))
    said = _one_cycle(tmp_path / "out", cycle)["cycles"]
    assert said["bob"]["says"] == "cycles" and (view is None) == ("cue" not in said)
    assert view is None or said["cue"]["says"] == "alike"


def _unwitnessed(report: dict, interval: float) -> dict:
    """The cut is two crossing intervals long, `interval` frames each, two steps by its crossings left on steps because
    nothing that reads no leg witnesses two steps an interval (`cycles.unwitnessed`), and a warning names it."""
    c = report["cycle"]
    counted = c["step_count"]
    said = counted["cycles"]
    assert abs(c["length"] - 2 * interval) <= 1.5 and c["steps"] == {"count": 2, "by": "crossings"}
    assert counted["cut"]["verdict"] == "two" and report["status"] == "passed"
    assert said["one"] < said["half"] and said["one"] <= said["two"] + local_cycle.CYCLE_SLACK
    assert said["cycles"] is False and said["unwitnessed"] is True and said["bob"]["says"] is None
    assert "nothing that reads no leg witnesses two steps an interval" in said["why"]
    assert any(local_cycle.UNWITNESSED in w and f"the cut {c['start']}/{c['length']}" in w for w in report["jolt"]["warnings"])
    return said


@pytest.mark.slow
@pytest.mark.parametrize("walker, view, interval", [("behind", None, 17.5), ("behind", "back_diagonal", 17.5),
                                                    ("outward", "front_diagonal", 16)])
def test_a_walk_whose_feet_meet_once_a_cycle_but_whose_body_does_not_bob_is_cut_two_intervals_with_a_warning(tmp_path, walker, view,
                                                                                                         interval):
    """The two walkers whose feet meet once a cycle, their body held still: its top line does not rise and fall, and
    nothing else that reads no leg can tell a step on from a cycle on. The crossings stay on steps: two intervals, two
    cycles, named in a warning — never one interval read as two steps on a word that cannot tell (the parent engine cut
    one cycle on the picture where a pass hid, and on the foot cue's turn half an interval on from the view)."""
    draw = (lambda k: _behind(k, cycle=interval, hold=1, bob=0)) if walker == "behind" else (lambda k: _outward(k, cycle=interval, bob=0))
    report = _cut(tmp_path / "out", _keyed(tmp_path, draw), *(("--view", view) if view else ()))
    said = _unwitnessed(report, interval)
    assert said["bob"]["swing"] < local_cycle.BOB_FLOOR and "does not rise and fall" in said["bob"]["why"]


@pytest.mark.slow
@pytest.mark.parametrize("cycle", [16, 18])
def test_a_diagonal_walk_whose_near_foot_steps_out_is_cut_one_cycle_from_its_view(tmp_path, cycle):
    """On a front diagonal the near foot steps out toward the camera and passes the far one apart: the feet meet on the
    screen once a cycle, and briefly — no pass hides. Given the view, the count asks the picture: the clip one interval
    on is the same picture, the legs as they were, the view's foot cue does not turn over there, and the body rises and
    falls twice an interval. One cycle, `steps: 2` (the parent engine counted each meeting a step and cut two cycles,
    32 and 36 frames). Without the view it is not asked, and cuts as before."""
    keyed = _keyed(tmp_path, lambda k: _outward(k, cycle=cycle))
    _cut(tmp_path / "view", keyed, "--view", "front_diagonal")
    counted = _one_cycle(tmp_path / "view", cycle)
    assert counted["cycles"]["overlap"] <= crossings.HIDE_SHARE and counted["cycles"]["asked"] == "the front_diagonal view, the legs side by side"
    # The far leg a tone off the near one moves the cue a tone where the feet part: over the floor, and it does not turn
    # over one interval on — it does not refuse. Its turn half an interval on is no witness (a light fixed on the screen
    # turns a cue so too); the body's bob is.
    cue = counted["cycles"]["cue"]
    assert cue["says"] is None and cue["range"] >= cue["floor"] == pytest.approx(local_cycle.CUE_FLOOR * legs.TONE, rel=0.01)
    assert cue["one"] > local_cycle.CUE_SWAPPED and "half" not in cue
    assert counted["cycles"]["bob"]["says"] == "cycles"
    plain = _cut(tmp_path / "plain", keyed)
    assert "cycles" not in plain["cycle"]["step_count"] and abs(plain["cycle"]["length"] - 2 * cycle) <= 1


@pytest.mark.slow
@pytest.mark.parametrize("tone, lean", [(1, 0), (4, 1)])
def test_a_side_walk_whose_two_steps_look_alike_is_cut_two_steps_however_near_a_step_on(tmp_path, tone, lean):
    """The case the count was made for, guarded: from the side the two steps are one silhouette, the near leg and arm
    a tone or a few apart from the far ones. The walk is slower than the window, so the search cuts one step, and at
    one tone the clip a step on draws nearer than the next frame (0.89 playback steps). Its feet pass briefly — no pass
    can hide — and from the side the picture is not asked: the count stands, two steps, a cycle long."""
    keyed = _keyed(tmp_path, lambda k: _alike(k, cycle=46, phase=13, tone=tone, lean=lean))
    for out, extra in (("plain", ()), ("side", ("--view", "side"))):
        counted = _two_steps(_cut(tmp_path / out, keyed, *extra), 46)
        assert counted["search"]["verdict"] == "refuted" and counted["search"]["k"] == [1]
        assert "cycles" not in counted


@pytest.mark.slow
def test_a_diagonal_walk_whose_foot_cue_turns_over_a_step_on_is_cut_two_steps(tmp_path):
    """The same walker filmed as a front diagonal: its legs a step on are one silhouette, so the clip one interval on is
    as near as a cycle on, but its far leg is drawn a few tones off the near one, and the view's foot cue turns over one
    interval on — the other leg leads. The count stands: two steps, a cycle long."""
    report = _cut(tmp_path / "out", _keyed(tmp_path, lambda k: _alike(k, cycle=46, phase=13, tone=4, lean=1)),
                  "--view", "front_diagonal")
    counted = _two_steps(report, 46)
    said = counted["cycles"]
    assert said["cycles"] is False and said["one"] <= said["two"] + local_cycle.CYCLE_SLACK
    assert said["cue"]["one"] <= local_cycle.CUE_SWAPPED and said["cue"]["says"] == "steps" and "the other leg leads" in said["why"]


@pytest.mark.slow
@pytest.mark.parametrize("view", ["front", "front_diagonal", "back", "back_diagonal"])
def test_a_walk_whose_legs_are_drawn_alike_is_cut_two_steps_from_any_view(tmp_path, view):
    """The same walker with its legs drawn alike, its arms still four tones apart, filmed from the front, the back or a
    diagonal: one interval on (a step on) is the same picture, as near as two on, and the view's foot cue does not move
    — the shade of legs drawn alike, feet drawn on one row — its range far under half the step it is drawn in. The body
    rises and falls once an interval, a step: nothing witnesses two steps an interval, and the crossings stay on steps,
    as from the side: two steps, 46 frames, and a warning naming why (the parent engine read the still cue's float
    roundings, or no correlation, as the same leg leading and cut 23 frames, one step, counted two)."""
    report = _cut(tmp_path / "out", _keyed(tmp_path, lambda k: _alike(k, cycle=46, phase=13, tone=4, lean=1, leg_tone=0)),
                  "--view", view)
    _two_steps(report, 46)
    said = _unwitnessed(report, 23)
    assert said["asked"] == f"the {view} view, the legs side by side" and said["overlap"] <= crossings.HIDE_SHARE
    assert said["cue"]["says"] == "alike" and said["cue"]["range"] < said["cue"]["floor"] and "one" not in said["cue"]
    assert said["bob"]["half"] < -local_cycle.BOB_TWICE


@pytest.mark.slow
@pytest.mark.parametrize("view", [None, "side", "front_diagonal", "back"])
def test_a_walk_whose_legs_are_drawn_alike_and_wide_is_cut_two_steps_where_a_pass_could_hide(tmp_path, view):
    """The same walker with its legs drawn alike and twice as wide: the feet are seen together over half the interval at
    each pass, as where the far foot passes hidden, so the count asks the picture with or without a view — and one
    interval on (a step on) is the same picture, as near as two on. The view's foot cue does not move (with no view
    none is read), and the body rises and falls once an interval: nothing witnesses two steps an interval, and the
    crossings stay on steps: two steps, 46 frames, and a warning naming why (the parent engine let the picture mark
    cycles and cut 23 frames, one step, counted two)."""
    keyed = _keyed(tmp_path, lambda k: _alike(k, cycle=46, phase=13, tone=4, lean=1, leg_tone=0, leg_half=10))
    report = _cut(tmp_path / "out", keyed, *(("--view", view) if view else ()))
    _two_steps(report, 46)
    said = _unwitnessed(report, 23)
    assert said["asked"].startswith("the feet together") and said["overlap"] > crossings.HIDE_SHARE
    assert (said["cue"]["says"] == "alike" and said["cue"]["range"] < said["cue"]["floor"]) if view else "cue" not in said


@pytest.mark.slow
@pytest.mark.parametrize("view", [None, "side", "front_diagonal", "back_diagonal"])
def test_a_walk_whose_legs_are_drawn_alike_under_a_light_fixed_on_the_screen_is_cut_two_steps(tmp_path, view):
    """The wide walker with its legs drawn alike, lit from the right by a light fixed on the screen (up to 12 tones
    across the feet): the shade of the feet moves with how far apart they are drawn, whichever leg leads, and so turns
    over half an interval on. The body rises and falls once an interval: nothing that reads no leg witnesses two steps
    an interval, and the crossings stay on steps — 46 frames, a warning (the parent engine let the picture mark cycles
    where a pass could hide, and cut 23 frames, one step, counted two, from every one of these views)."""
    report = _cut(tmp_path / "out", _keyed(tmp_path, lambda k: _alike(k, cycle=46, phase=13, tone=4, lean=1, leg_tone=0, leg_half=10,
                                                                       light=("x", 12))), *(("--view", view) if view else ()))
    _two_steps(report, 46)
    said = _unwitnessed(report, 23)
    assert view is None or (said["cue"]["says"] is None and said["cue"]["range"] >= said["cue"]["floor"])


@pytest.mark.slow
@pytest.mark.parametrize("view", ["front", "front_diagonal", "back", "back_diagonal"])
def test_a_walk_whose_swing_foot_lifts_under_a_light_fixed_on_the_screen_is_cut_two_steps(tmp_path, view):
    """The walker with its legs drawn alike, the leg swinging forward lifted 6 px at its pass, lit lower down by a light
    fixed on the screen: the depth and the shade of the feet move with the swing, whichever leg swings, and turn over
    half an interval on. Nothing that reads no leg witnesses two steps an interval: 46 frames, a warning (the parent
    engine read the cue's turn half an interval on as the other leg leading a step on, and cut 23 frames)."""
    report = _cut(tmp_path / "out", _keyed(tmp_path, lambda k: _alike(k, cycle=46, phase=13, tone=4, lean=1, leg_tone=0, lift=6,
                                                                       light=("y", 12))), "--view", view)
    _two_steps(report, 46)
    assert _unwitnessed(report, 23)["cue"]["says"] in (None, "alike")


@pytest.mark.slow
@pytest.mark.parametrize("view", [None, "side", "front_diagonal"])
def test_a_wide_walk_whose_far_leg_is_drawn_a_few_tones_off_is_cut_two_steps(tmp_path, view):
    """The wide walker with its far leg four tones off the near one: a pass can hide. From the side or a diagonal the
    view's shade turns over one interval on — the other leg leads — and refuses cycles; with no view no cue is read,
    and the body rising and falling once an interval witnesses nothing: two steps, 46 frames either way, a warning
    only where nothing refused (the parent engine let the picture mark cycles with no view and cut 23 frames)."""
    report = _cut(tmp_path / "out", _keyed(tmp_path, lambda k: _alike(k, cycle=46, phase=13, tone=4, lean=1, leg_half=10)),
                  *(("--view", view) if view else ()))
    counted = _two_steps(report, 46)
    if view is None:
        _unwitnessed(report, 23)
    else:
        said = counted["cycles"]
        assert said["cycles"] is False and said["cue"]["says"] == "steps" and "unwitnessed" not in said
        assert not any(local_cycle.UNWITNESSED in w for w in report["jolt"]["warnings"])


@pytest.mark.slow
@pytest.mark.parametrize("motion, bob, room", [("arc", 3, 0), ("arc", 0, 60), ("sway", 0, 60), ("ramp", 0, 60)])
def test_a_walk_whose_crossings_are_its_steps_carried_up_and_down_is_cut_two_steps(tmp_path, motion, bob, room):
    """The wide walker with its legs drawn alike, carried up and down the screen with the whole picture (`_carried`;
    its legs, their steps and their crossings every 23 frames as they were): bobbing, along a smooth arc 24 px deep;
    its body held still, along the same arc, a sway 12 px either way four intervals long, and a ramp 24 px down a row
    every three frames. None of it is the body rising and falling twice an interval: less its flow the bobbing
    walker's top line rises and falls once an interval, turned over half an interval on, and the still one's moves
    under BOB_ROWS (its rows rounding up and down as the picture drifts). Nothing witnesses two steps an interval: two
    steps, 46 frames, a warning (the parent engine read the flow's own likeness half an interval on as the body rising
    and falling twice an interval and cut each one step, counted two, and no warning said so)."""
    walker = lambda k: _alike(k, cycle=46, phase=13, tone=4, lean=1, leg_tone=0, leg_half=10, bob=bob)  # noqa: E731
    report = _cut(tmp_path / "out", _keyed(tmp_path, _carried(walker, motion, room)))
    _two_steps(report, 46)
    said = _unwitnessed(report, 23)
    assert said["asked"].startswith("the feet together")
    if bob:
        assert said["bob"]["half"] <= -local_cycle.BOB_TWICE
    else:
        assert said["bob"]["rows"] < local_cycle.BOB_ROWS and "does not rise and fall" in said["bob"]["why"]


@pytest.mark.slow
@pytest.mark.parametrize("motion, room", [("arc", 0), ("cubic", 60)])
def test_a_walk_whose_feet_meet_once_a_cycle_carried_up_and_down_is_cut_one_cycle(tmp_path, motion, room):
    """The walker whose far foot passes hidden, its legs drawn alike, carried along the arc or the cubic with the
    whole picture: less its flow its top line rises and falls twice an interval as it did with the picture still,
    one cycle."""
    walker = lambda k: _behind(k, cycle=17.5, hold=1, alike=True)  # noqa: E731
    _cut(tmp_path / "out", _keyed(tmp_path, _carried(walker, motion, room)))
    said = _one_cycle(tmp_path / "out", 17.5)["cycles"]
    assert said["bob"]["says"] == "cycles" and said["bob"]["quarter"] <= -local_cycle.BOB_TWICE


@pytest.mark.slow
def test_a_cut_of_two_steps_that_does_not_close_leaves_the_searchs_cut(tmp_path):
    """On threes, a big tail on its own beat: the search's cut is one the count does not refute (it plays feet held
    together past a pass: uncertain), and no cut of two steps joins the walk at its own point. The seam gate reads the
    first tried between its drawings and passes it (read frame by frame, every one tried was over the gate), but out
    of phase it is no two-step cut that closes. The search's cut stands and the loop passes; the record and a warning
    line name the two-step cut declined (the parent engine took it, and the seam gate failed the loop)."""
    report = _cut(tmp_path / "out", _keyed(tmp_path, _tailed))
    c = report["cycle"]
    counted = c["step_count"]
    assert report["status"] == "passed" and report["drawings"]["hold"] == 3
    assert counted["chosen_by"] == "search" and counted["search"]["verdict"] == "uncertain"
    assert [c["start"], c["length"]] == counted["search"]["cut"] and "steps" not in c
    declined = counted["declined"]
    assert declined["gated"] is False and abs(declined["phase"]) > crossings.PHASE_TOLERANCE
    assert [t["seam"] for t in counted["seam_gate"]["tried"]] == [declined["seam"]] and declined["seam"] <= report["seam_max"]
    assert any("the search's cut" in w and "stands" in w for w in report["jolt"]["warnings"])


def _same_cut(a: Path, b: Path) -> None:
    for name in ("loop.strip.png", "loop.gif", "loop.webp"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    frames = sorted(p.name for p in (a / "cycle").glob("*.png"))
    assert frames and frames == sorted(p.name for p in (b / "cycle").glob("*.png"))
    assert all((a / "cycle" / f).read_bytes() == (b / "cycle" / f).read_bytes() for f in frames)


@pytest.mark.slow
def test_a_walk_the_search_cuts_two_steps_in_phase_is_cut_as_before(tmp_path):
    """The same front walk with a small tail: the search cuts its 27 frames, two steps joined at the walk's own point,
    and the cut is kept — every image as the search alone cut it; the strip says `steps: 2`."""
    keyed = _keyed(tmp_path, lambda k: _front(k, cycle=27, phase=5, tail_cycle=41, tail=40))
    report = _cut(tmp_path / "read", keyed)
    counted = report["cycle"]["step_count"]
    assert counted["chosen_by"] == "search" and counted["cut"]["verdict"] == "two"
    assert report["cycle"]["method"] == "local-repeat-drift-v1" and report["cycle"]["steps"] == {"count": 2, "by": "crossings"}
    with feet_unseen():
        plain = _cut(tmp_path / "unread", keyed)
    assert (plain["cycle"]["start"], plain["cycle"]["length"]) == (report["cycle"]["start"], report["cycle"]["length"])
    _same_cut(tmp_path / "read", tmp_path / "unread")
    strips = [json.loads((tmp_path / d / "loop.strip.json").read_text()) for d in ("read", "unread")]
    assert strips[0].pop("steps") == 2 and "steps" not in strips[1] and strips[0] == strips[1]


@pytest.mark.slow
def test_a_body_with_no_feet_to_read_is_cut_as_the_search_cuts_it(tmp_path):
    keyed = _keyed(tmp_path, _blob, frames=64)
    report = _cut(tmp_path / "out", keyed)
    counted = report["cycle"]["step_count"]
    assert counted["readable"] is False and counted["chosen_by"] == "search" and counted["why"]
    assert counted["search"]["cut"] == [report["cycle"]["start"], report["cycle"]["length"]]
    assert "steps" not in report["cycle"] and "steps" not in json.loads((tmp_path / "out" / "loop.strip.json").read_text())


@pytest.mark.slow
def test_a_count_given_is_the_word_on_the_searchs_cut(tmp_path):
    """--steps counts the cut the search takes: the crossings are not asked."""
    report = _cut(tmp_path / "out", _keyed(tmp_path, lambda k: _side(k, cycle=46, phase=13)), "--steps", "2")
    assert "step_count" not in report["cycle"] and report["cycle"]["steps"]["by"] == "given"
    assert report["cycle"]["length"] in (22, 23, 24)  # the search's one step, kept on the count


def _distances(period: int = 13, frames: int = 120) -> np.ndarray:
    t = np.arange(frames)
    angle = t * 2 * np.pi / period
    points = np.column_stack([np.cos(angle), np.sin(angle)])
    return np.abs(points[:, None] - points[None, :]).mean(axis=2).astype(np.float32)


@pytest.mark.parametrize("uncertain_from", [None, 5])
def test_the_search_refuses_what_the_crossings_refute_and_names_their_count(uncertain_from):
    """`detect` with `judge`: a window the crossings count other than two is refused, and a search that finds nothing
    lists each refused window's count, those the crossings refute after the rest."""
    def judge(s, L):
        if uncertain_from is not None and s >= uncertain_from:
            return {"verdict": "uncertain", "code": "partial", "k": [], "why": "plays frames no crossing confirms"}
        return {"verdict": "refuted", "code": "refuted", "k": [1], "why": "one step, each seen"}

    D = _distances()
    if uncertain_from is not None:
        found = local_cycle.detect(D, np.zeros(len(D)), min_len=12, max_len=14, gait_floor=8, periodicity_min=loop.PERIODICITY_MIN,
                                   double_tolerance=loop.GAIT_DOUBLE_TOL, double_search=loop.GAIT_DOUBLE_SEARCH, judge=judge)
        assert found["start"] >= uncertain_from
        return
    with pytest.raises(ValueError, match="no periodic cycle found") as refused:
        local_cycle.detect(D, np.zeros(len(D)), min_len=12, max_len=14, gait_floor=8, periodicity_min=loop.PERIODICITY_MIN,
                           double_tolerance=loop.GAIT_DOUBLE_TOL, double_search=loop.GAIT_DOUBLE_SEARCH, judge=judge)
    shown = refused.value.diagnostics["candidates"]
    assert shown and all(row["step_count"]["verdict"] == "refuted" for row in shown)
    assert any("one step, each seen in the clip's crossings" == row["refused"] for row in shown)
    assert "refute after the rest" in refused.value.diagnostics["order"]


def _reading(run: int) -> crossings.Reading:
    """A clip whose feet meet every 15 frames (crossings at 8, 23 … 68), seen together `run` frames at each meeting
    (9: over half the interval, a pass can hide there; 3: brief)."""
    middles = (8, 23, 38, 53, 68)
    return crossings.Reading(FRAMES, "gap", [(float(c), float(c)) for c in middles],
                             runs=[(c - run // 2, c + run // 2) for c in middles])


def _bob(beat: float | None) -> dict:
    """Leg signals whose body's top line rises and falls 3 rows either way every `beat` frames (None: held still), the
    body 100 rows tall: every 7.5 frames it rises and falls twice an interval of `_reading`, every 15 once."""
    t = np.arange(FRAMES)
    top = 40 + (np.zeros(FRAMES) if beat is None else 3 * np.cos(2 * np.pi * (t - 8) / beat))
    return {"top": list(top), "height": [100.0] * FRAMES}


def _choose(found: dict, *, period: int, seam: float, run: int = 9, search_seam: float = 1.0, view: str | None = None,
            cue: list[float] | None = None, cue_step: float | None = None, bob: float | None = None) -> tuple[dict, dict]:
    """`by_steps` on a clip whose picture goes round every `period` frames and whose feet meet every 15 (`_reading`),
    its crossings read as `read_cycles` reads them from `view`, its foot `cue` (drawn in steps of `cue_step`, where
    given) and its body's bob (`_bob`, every `bob` frames), with the search's cut `found`: every two-step cut's seam
    reads `seam` at the gate, its top calm and its flow off 1.0 either side, and the search's own cut `search_seam`."""
    D = _distances(period=period, frames=FRAMES)
    reading = _reading(run)
    local_cycle.read_cycles(reading, D, view=view, cue=cue, signals=_bob(bob), **({} if cue_step is None else {"cue_step": cue_step}))
    return local_cycle.by_steps(
        D, np.zeros(FRAMES), reading, found=dict(found), refused=None, min_len=12, max_len=FRAMES,
        seam_flow=lambda s, L: {"off": [1.0, 1.0], "turn": [0.0, 0.0], "window": [[0, 0, 1, 1], [0, 0, 1, 1]]},
        wrap_pop=lambda s, L: {"pops": False, "pop": 0.1},
        seam_gate=lambda s, L: search_seam if (s, L) == (found["start"], found["length"]) else seam, flow_limit=3.0, seam_max=2.0,
        gait_floor=12, periodicity_min=loop.PERIODICITY_MIN, signals=None)


@pytest.mark.parametrize("seam, search_seam, kept", [(3.0, 1.0, True), (1.0, 1.0, False), (3.0, 3.0, False)])
def test_a_cut_the_count_does_not_refute_gives_way_only_to_a_two_step_cut_that_closes(seam, search_seam, kept):
    """The search's cut 10/33 covers two steps a fifth of a step off (out of phase, not refuted), and the seam gate
    passes it. A two-step cut in phase replaces it only where the seam gate passes that too; where none passes, the
    search's cut stands and the record names the two-step cut it declined, with a warning line (the parent engine took
    the two-step cut either way). A search's cut the gate refuses as well gives way: the two-step cut's wrap may yet be
    closed by the overlapped wrap."""
    picked, record = _choose({"start": 10, "length": 33}, period=30, seam=seam, search_seam=search_seam)
    assert record["search"]["verdict"] == "two" and abs(record["search"]["phase"]) > crossings.PHASE_TOLERANCE
    if kept:
        assert (picked["start"], picked["length"]) == (10, 33) and record["chosen_by"] == "search"
        assert record["declined"]["seam"] == seam and record["declined"]["gated"] is True
        assert picked["steps"] == {"count": 2, "by": "crossings"}
        warned = local_cycle.step_verdict({**picked, "step_count": record})
        assert len(warned) == 1 and "the search's cut 10/33 stands" in warned[0]
    else:
        assert record["chosen_by"] == "two steps" and "declined" not in record and picked["length"] != 33


def test_a_cut_the_count_refutes_gives_way_to_two_steps_closing_or_not():
    """A search's cut the count refutes, one step seen, is a fault of its own: the two-step cut is taken though its seam
    fails the gate, and the gate says whether the loop passes. The picture goes round every 30 frames: one interval on
    it is half a cycle off, and the crossings mark steps."""
    picked, record = _choose({"start": 10, "length": 15}, period=30, seam=3.0)
    assert record["cycles"]["cycles"] is False and record["cycles"]["one"] > record["cycles"]["half"]
    assert record["search"]["verdict"] == "refuted" and record["search"]["k"] == [1]
    assert record["chosen_by"] == "two steps" and picked["steps"] == {"count": 2, "by": "crossings"} and abs(picked["length"] - 30) <= 1


@pytest.mark.parametrize("bob", [7.5, 15, None])
def test_where_a_pass_can_hide_and_the_picture_repeats_each_interval_the_body_says_whether_the_crossings_mark_cycles(bob):
    """The same meetings on a clip whose picture goes round every 15 frames, the feet together 9 frames at each: a pass
    can hide there, and one interval on the clip is the same picture as two on. Where the body rises and falls twice an
    interval, the crossings mark cycles: the step is half the interval, and the search's cut 10/15 is two steps joined at
    the walk's own point — it stands, `steps` written. Where it rises and falls once an interval, or is held still, the
    picture alone cannot tell a step on from a cycle on (legs drawn alike are the same picture a step on): the crossings
    stay on steps, two steps of 15, `unwitnessed`, a warning naming it (the parent engine let the picture mark cycles)."""
    picked, record = _choose({"start": 10, "length": 15}, period=15, seam=1.0, bob=bob)
    said = record["cycles"]
    assert said["asked"].startswith("the feet together") and said["one"] <= said["two"] + local_cycle.CYCLE_SLACK and "cue" not in said
    if bob == 7.5:
        assert said["cycles"] is True and said["bob"]["says"] == "cycles" and said["bob"]["half"] >= local_cycle.BOB_TWICE
        assert record["step_range"][1] == 7.5 and "unwitnessed" not in said
        assert (picked["start"], picked["length"]) == (10, 15) and picked["steps"] == {"count": 2, "by": "crossings"}
        assert record["chosen_by"] == "search" and record["cut"]["verdict"] == "two"
        assert local_cycle.step_verdict({**picked, "step_count": record}) == []
    else:
        assert said["cycles"] is False and said["unwitnessed"] is True and said["bob"]["says"] is None
        assert ("does not rise and fall" if bob is None else "does not rise and fall twice an interval") in said["bob"]["why"]
        assert ("half" not in said["bob"]) if bob is None else said["bob"]["half"] < -local_cycle.BOB_TWICE
        assert record["chosen_by"] == "two steps" and abs(picked["length"] - 30) <= 1
        warned = [w for w in local_cycle.step_verdict({**picked, "step_count": record}) if local_cycle.UNWITNESSED in w]
        assert len(warned) == 1 and said["bob"]["why"] in warned[0]


@pytest.mark.parametrize("view", [None, "side"])
def test_where_the_feet_meet_briefly_the_count_stands_from_the_side_or_no_view(view):
    """The same clip, its feet seen together only at the moment they meet: no pass can hide, and from the side (or with
    no view given) a step on drawing as near as a cycle on is what legs drawn alike do. Not asked: two steps."""
    picked, record = _choose({"start": 10, "length": 15}, period=15, seam=1.0, run=3, view=view, bob=7.5)
    assert "cycles" not in record
    assert record["search"]["verdict"] == "refuted" and record["chosen_by"] == "two steps" and abs(picked["length"] - 30) <= 1


@pytest.mark.parametrize("cue, bob, cycles", [("one", 7.5, False), ("half", 15, False), ("half", 7.5, True)])
def test_from_a_diagonal_the_foot_cue_only_refuses_cycles(cue, bob, cycles):
    """The same brief meetings from a front diagonal: asked, and one interval on is the same picture. Where the view's
    foot cue turns over one interval on (the near foot's shade leading one interval, the far one's the next) the other
    leg leads there: the count stands, two steps of 15, whatever the body says. A cue that turns over half an interval
    on and holds one on is no witness — a light fixed on the screen turns the shade of feet drawn apart so, whichever
    leg leads: with the body rising and falling once an interval, the crossings stay on steps with a warning (the
    parent engine read the turn as the other leg leading a step on and kept 15 frames as two steps); rising and falling
    twice, they mark cycles and the search's 15 frames are the cycle."""
    t = np.arange(FRAMES)
    values = list(np.sin(2 * np.pi * (t - 8) / (30 if cue == "one" else 15)))
    picked, record = _choose({"start": 10, "length": 15}, period=15, seam=1.0, run=3, view="front_diagonal", cue=values,
                             cue_step=legs.TONE, bob=bob)
    said = record["cycles"]
    assert said["asked"] == "the front_diagonal view, the legs side by side" and said["cycles"] is cycles
    assert said["cue"]["says"] == ("steps" if cue == "one" else None) and "half" not in said["cue"]
    assert (said["cue"]["one"] <= local_cycle.CUE_SWAPPED) is (cue == "one")
    assert said.get("unwitnessed", False) is (cue == "half" and not cycles)
    assert abs(picked["length"] - (15 if cycles else 30)) <= (0 if cycles else 1)
    warned = local_cycle.step_verdict({**picked, "step_count": record})
    assert any(local_cycle.UNWITNESSED in w for w in warned) is (cue == "half" and not cycles)


@pytest.mark.parametrize("view, run", [("front_diagonal", 3), ("back_diagonal", 9), (None, 9)])
@pytest.mark.parametrize("cue", ["still", "own beat"])
@pytest.mark.parametrize("bob", [7.5, None])
def test_a_cue_that_does_not_tell_the_legs_apart_leaves_the_word_to_the_body(view, run, cue, bob):
    """The same meetings, the clip one interval on the same picture, from a diagonal (passes brief or long) or with no
    view where a pass can hide, and a foot cue that does not tell the legs apart: legs drawn alike leave it still — a
    few float roundings, its range far under half a tone (`legs.TONE`), not read, however well the roundings, which
    repeat with the legs, correlate — or it flickers on a beat of its own (every 4 frames) and does not turn over one
    interval on. It refuses nothing; with no view it is not read at all. The body says: rising and falling twice an
    interval, the crossings mark cycles, one cycle of 15 (the parent engine left a still cue on steps, two cycles);
    held still, they stay on steps, two steps of 15, with a warning."""
    t = np.arange(FRAMES)
    values = 1e-8 * np.sin(2 * np.pi * (t - 8) / 15) if cue == "still" else 0.05 * np.cos(2 * np.pi * t / 4)
    picked, record = _choose({"start": 10, "length": 15}, period=15, seam=1.0, run=run, view=view, cue=list(values),
                             bob=bob, **({"cue_step": legs.TONE} if cue == "still" else {}))
    said = record["cycles"]
    assert said["one"] < said["half"] and said["one"] <= said["two"] + local_cycle.CYCLE_SLACK
    if view is None:
        assert "cue" not in said
    else:
        assert said["cue"]["says"] == ("alike" if cue == "still" else None)
        assert ("one" not in said["cue"]) if cue == "still" else said["cue"]["one"] > local_cycle.CUE_SWAPPED
    assert said["cycles"] is (bob is not None) and said.get("unwitnessed", False) is (bob is None)
    assert abs(picked["length"] - (15 if bob else 30)) <= (0 if bob else 1)
    assert any(local_cycle.UNWITNESSED in w for w in local_cycle.step_verdict({**picked, "step_count": record})) is (bob is None)


@pytest.mark.parametrize("swing, beat, says", [(3, 7.5, "cycles"), (3, 15, None), (0.2, 7.5, None), (3, 30, None)])
def test_the_body_witnesses_two_steps_an_interval_where_its_top_line_rises_and_falls_twice(swing, beat, says):
    """`bob_word` on an interval of 15 frames, the body 100 rows tall: a top line rising and falling 3 rows either way
    every 7.5 frames (twice an interval: as it was half an interval on, turned over a quarter on) witnesses two steps
    an interval; every 15 (once: a step an interval, turned over half an interval on) or every 30 (once a cycle of two
    intervals: a quarter turn half an interval on, and much of it taken out as flow) does not; 0.2 rows either way, its
    range under BOB_FLOOR of the body's height and under BOB_ROWS rows, does not rise and fall, however it correlates."""
    t = np.arange(FRAMES)
    line = list(40 + swing * np.cos(2 * np.pi * t / beat))
    word = local_cycle.bob_word(line, 15, height=100)
    assert word["says"] == says and word["floor"] == local_cycle.BOB_FLOOR and word["twice"] == local_cycle.BOB_TWICE
    left = local_cycle.off_flow(line, 15)
    assert word["swing"] == pytest.approx(np.ptp(left) / 100, abs=1e-4)
    assert word["rows"] == pytest.approx(np.ptp(left), abs=0.01)
    if swing < 1:
        assert "half" not in word and "does not rise and fall" in word["why"]
    elif beat < 30:
        twice = word["half"] >= local_cycle.BOB_TWICE and word["quarter"] <= -local_cycle.BOB_TWICE
        assert twice is (says == "cycles")


@pytest.mark.parametrize("flow", ["still", "arc", "cubic", "sway", "ramp"])
@pytest.mark.parametrize("beat", [7.5, 15, None])
def test_the_body_is_read_less_the_flow_of_the_picture(beat, flow):
    """`bob_word` on an interval of 15 frames, the body 100 rows tall, its top line read in whole rows: rising and
    falling 3 rows either way twice an interval (every 7.5 frames), once (every 15) or not at all, on a picture held
    still or carried up and down with the line: along an arc 24 rows deep, a cubic 48 rows either way, a sway 12 rows
    either way four intervals long, a ramp of 6 rows over the clip. Less its flow the line is the body's own: twice an
    interval witnesses two steps an interval wherever the picture goes, and once or not at all never does; a line that
    does not rise and fall is left under BOB_ROWS by the picture drifting in whole rows."""
    t = np.arange(FRAMES)
    flows = {"still": 0 * t, "arc": 24 * ((t - 36) / 36) ** 2, "cubic": 48 * ((t - 36) / 36) ** 3,
             "sway": 12 * np.sin(2 * np.pi * t / 60), "ramp": 6 * t / 72}
    top = np.round(40 + (0 if beat is None else 3 * np.cos(2 * np.pi * t / beat)) + flows[flow])
    word = local_cycle.bob_word(list(top), 15, height=100)
    assert (word["says"] == "cycles") is (beat == 7.5)
    if beat is None and flow != "sway":
        assert word["rows"] < local_cycle.BOB_ROWS and "does not rise and fall" in word["why"]


def test_the_flow_taken_out_leaves_a_rise_and_fall_once_or_twice_an_interval_whole():
    """`off_flow` over an interval of 15 frames (and of 17.5, its end frames a quarter weighted): a cubic flow is taken
    out wholly, and a rise and fall once or twice an interval laid on it is left as it was, over the frames the
    interval fits about (7 either end; 9 at 17.5), less its straight line."""
    t = np.arange(FRAMES)
    flow = 48 * ((t - 36) / 36) ** 3 - 5 * t / 72 + 40
    for interval, reach in ((15, 7), (17.5, 9)):
        for beat in (interval, interval / 2):
            rise = 3 * np.cos(2 * np.pi * t / beat)
            left = local_cycle.off_flow(list(flow + rise), interval)
            assert left.size == FRAMES - 2 * reach
            inner, u = rise[reach:FRAMES - reach], np.arange(FRAMES - 2 * reach)
            whole = inner - np.polyval(np.polyfit(u, inner, 1), u)
            assert np.max(np.abs(left - whole)) < (1e-9 if interval == 15 else 0.05)
        assert np.max(np.abs(local_cycle.off_flow(list(flow), interval))) < 1e-9


@pytest.mark.parametrize("verdict, named", [("uncertain", True), ("two", False)])
def test_a_walk_that_starts_and_stops_in_the_take_is_named_to_film_again(verdict, named):
    """Feet together 15 frames at the head and 14 at the tail, longer than any pass the clip shows whole (8, 3): the
    walk stands still there. Where the cut kept is not two steps, no cut of two closes between the stands and the loop
    names the take to film again (`start-stop`), its stands, crossings and cut; a cut counted two steps is a walk
    looped, and nothing is named. Feet that only pass name nothing."""
    reading = crossings.Reading(FRAMES, "gap", [(10.0, 10.0), (20.0, 20.0), (40.5, 40.5)],
                                runs=[(3, 17), (19, 21), (37, 44), (59, 72)], unconfirmed=[(59, 72)])
    assert reading.stands == [(3, 17), (59, 72)] and reading.run_limit == pytest.approx(8 * math.sqrt(2))
    record = reading.record()
    assert record["stands"] == [[3, 17], [59, 72]] and record["run_limit"] == round(8 * math.sqrt(2), 2)
    cut = {"verdict": verdict, "code": "tail-stall" if verdict == "uncertain" else "two", "why": "plays the clip's tail"}
    retake = loop.stand_retake({"start": 37, "length": 19, "step_count": {**record, "cut": cut}})
    if named:
        assert retake["reason"] == "start-stop" and retake["reasons"] == ["start-stop"] and retake["remedy"] == "retake"
        assert retake["start_stop"]["stands"] == [[3, 17], [59, 72]] and retake["start_stop"]["cut"] == [37, 19]
        assert retake["start_stop"]["code"] == "tail-stall"
        assert loop.retakes(None, retake) == retake
        assert loop.retakes({"reason": "part-pop", "reasons": ["part-pop"], "part_pop": {}, "remedy": "retake"}, retake)["reasons"] == \
            ["part-pop", "start-stop"]
    else:
        assert retake is None
    passing = _reading(3)
    assert passing.stands == [] and "stands" not in passing.record()
    assert loop.stand_retake({"start": 10, "length": 15, "step_count": {**passing.record(), "cut": {"verdict": "uncertain"}}}) is None
