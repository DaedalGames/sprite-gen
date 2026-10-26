# SPDX-License-Identifier: Apache-2.0
"""A walk whose body turns inside its clip (docs/video-pipeline.md section 4, "The body turn";
sprite_gen/video/body_turn.py): each frame's torso is read between the clip's first frame and that frame mirrored
(`torso-side-v1`), a turn is the torso under TURN_SIDE_MAX for TURN_RUN_MIN frames or more, and `video-loop --anchor
motion-auto` takes no cut that holds a frame of one while another holds a cycle. Where none does, `--body-turn warn` (the
default) takes the cut it took before and says so; `--body-turn refuse` refuses the clip in a line of its own.

The walker is a back diagonal seen from behind: legs side by side swinging half a cycle apart, arms either side swinging
opposite, the arm on the facing side bright and the other dim, a light panel on the torso and the cheek on the head
toward the facing. Over a turn the facing goes over to the still's mirror and back (`facing` 1 the still's three-quarter
view, 0 square from behind, under 0 past it), the arms' tones crossing. A walker drawn square from behind (`square`) is
the back walk whose still shows no side. A frame the reading cannot read — no head to centre on, no solid pixel across
the still's torso rows, a crop of no width or with nothing in it, no frames — is not read, never read as 0 or as a
middle no pixel holds. Synthetic frames only."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from conftest import feet_unseen
from sprite_gen.video import local_cycle, loop

FRAMES = 73
CYCLE = 24
W, H = 300, 240
TURN = (30, 40)  # the frames the turning walker turns over, the still's mirror reached at their middle
LOW = -0.6  # the facing the turn reaches
APART = 240  # how much lower the walker is drawn, on a canvas as much taller, where a part is apart above it


def _facing(turns, t: float, low: float = LOW, edge: int = 3) -> float:
    """1 outside the turns; inside each (a, b), down to `low` over `edge` frames either end."""
    f = 1.0
    for a, b in turns:
        if a <= t <= b:
            f = min(f, 1.0 - (1.0 - low) * min(1.0, (min(t - a, b - t) + 1) / (edge + 1)))
    return f


def _walker(k: int, *, turns=(), every: tuple[int, int] | None = None, hold: int = 1, square: bool = False,
            hood: bool = False, lead: int = 0, staff=(), apart=None) -> Image.Image:
    """The walker's frame `k`: turning over `turns`, or `every` (at, span) frames of each cycle; drawn on `hold`s; square
    from behind (`square`); hooded, its head and shoulders one shape that widens downward with no neck (`hood`); filmed
    smaller over its first `lead` frames, from 0.6 of its size, about its feet (a lead-in); over the frames in `staff`,
    holding a staff beside it that bobs with it, its ornament as high as the head and apart from it, the handle in its
    hand; where `apart` is given, drawn APART px lower on a canvas as much taller, a pennant apart above it at the top
    over the frames in `apart`."""
    t = k - k % hold
    ph = 2 * math.pi * t / CYCLE
    if square:
        f = 0.0
    elif every is not None:
        f = _facing([(every[0] + c * CYCLE, every[0] + c * CYCLE + every[1]) for c in range(-1, FRAMES // CYCLE + 2)], t)
    else:
        f = _facing(turns, t)
    im = Image.new("RGBA", (W, H))
    d = ImageDraw.Draw(im)
    x, y = 150, 40 + round(3 * math.cos(2 * ph))
    sw = 6 * math.sin(ph)
    for sign in (1, -1):
        lx = x + 14 * sign * math.sin(ph)
        d.rectangle((lx - 5, y + 86, lx + 5, y + 132), fill=(60, 60, 70, 255))
    for side in (1, -1):
        tone = round(150 + 75 * (1 + side * f) / 2)
        ay = y + 36 + round(side * sw)
        d.rectangle((x + side * 20 - 4, ay, x + side * 20 + 4, ay + 38), fill=(tone, tone, tone, 255))
    d.rectangle((x - 16, y + 32, x + 16, y + 90), fill=(200, 80, 80, 255))
    px = x + round(9 * f)
    d.rectangle((px - 5, y + 36, px + 5, y + 86), fill=(240, 170, 150, 255))
    fx = x + round(8 * f)
    if hood:
        d.polygon([(x - 13, y), (x + 13, y), (x + 17, y + 34), (x - 17, y + 34)], fill=(70, 90, 140, 255))
        d.rectangle((fx - 5, y + 8, fx + 5, y + 26), fill=(230, 200, 170, 255))
    else:
        d.rectangle((x - 5, y + 26, x + 5, y + 34), fill=(210, 180, 150, 255))
        d.rectangle((x - 13, y, x + 13, y + 28), fill=(90, 60, 40, 255))
        d.rectangle((fx - 5, y + 6, fx + 5, y + 24), fill=(230, 200, 170, 255))
    d.point((x - 10, y + 44 + int(t) % 11), fill=(250, 250, 250, 255))  # no two drawings the same picture
    if k in staff:
        d.rectangle((267, y, 293, y + 28), fill=(90, 60, 40, 255))
        d.rectangle((278, y + 28, 282, y + 90), fill=(140, 90, 40, 255))
        d.rectangle((170, y + 70, 282, y + 74), fill=(140, 90, 40, 255))
    if k < lead:
        s = 0.6 + 0.4 * k / lead
        small = im.resize((round(W * s), round(H * s)), Image.Resampling.LANCZOS)
        im = Image.new("RGBA", (W, H))
        im.paste(small, (round(150 - 150 * s), round(172 - 172 * s)), small)
    if apart is not None:
        tall = Image.new("RGBA", (W, H + APART))
        tall.paste(im, (0, APART))
        if k in apart:
            ImageDraw.Draw(tall).polygon([(130, 0), (170, 0), (130, 20)], fill=(250, 220, 90, 255))
        im = tall
    return im


def _speck(n: int) -> Image.Image:
    """A frame that holds nothing but `n` solid pixels in a row."""
    im = Image.new("RGBA", (W, H))
    ImageDraw.Draw(im).rectangle((150, 100, 149 + n, 100), fill=(200, 80, 80, 255))
    return im


def _keyed(root: Path, **drawn) -> Path:
    keyed = root / "keyed"
    keyed.mkdir(parents=True)
    for k in range(FRAMES):
        _walker(k, **drawn).save(keyed / f"{k:03}.png")
    return keyed


def _frames(**drawn) -> list[Image.Image]:
    return [_walker(k, **drawn) for k in range(FRAMES)]


def _cut(out: Path, keyed: Path, *extra: str) -> dict:
    assert loop.main(["--frames-dir", str(keyed), "--out-dir", str(out), "--state", "walk", "--anchor", "motion-auto",
                      "--repair", "off", *extra]) == 0
    return json.loads((out / "loop.loop.report.json").read_text())


def _refused(out: Path, keyed: Path, capsys, *extra: str) -> tuple[str, dict]:
    capsys.readouterr()
    with pytest.raises(SystemExit) as refused:
        loop.main(["--frames-dir", str(keyed), "--out-dir", str(out), "--state", "walk", "--anchor", "motion-auto",
                   "--repair", "off", *extra])
    report = json.loads((out / "loop.loop.report.json").read_text())
    assert report["status"] == "failed"
    return str(refused.value.code), report


def _same_cut(a: Path, b: Path) -> None:
    for name in ("loop.strip.png", "loop.gif", "loop.webp"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    frames = sorted(p.name for p in (a / "cycle").glob("*.png"))
    assert frames and frames == sorted(p.name for p in (b / "cycle").glob("*.png"))
    assert all((a / "cycle" / f).read_bytes() == (b / "cycle" / f).read_bytes() for f in frames)


def _unread(monkeypatch) -> None:
    """The body turn read as nowhere: the search as it was before the reading."""
    from sprite_gen.video import body_turn

    real = body_turn.read
    monkeypatch.setattr(body_turn, "read", lambda frames: {**real(frames), "turns": []})


def _holds(report: dict, frames: range) -> bool:
    s, L = report["cycle"]["start"], report["cycle"]["length"]
    return bool(set(range(s, s + L)) & set(frames))


# --- the reading ---------------------------------------------------------------------------------------------------


def test_a_turn_is_four_frames_in_a_row_under_zero():
    """TURN_SIDE_MAX 0 and TURN_RUN_MIN 4, each where it turns over: a torso at 0 is not turned, one under it is; three
    frames in a row are the walk's own dip, four a turn; a frame not read ends a run."""
    from sprite_gen.video import body_turn

    assert body_turn.TURN_SIDE_MAX == 0.0 and body_turn.TURN_RUN_MIN == 4
    at = lambda side: [(t["from"], t["to"]) for t in body_turn.turns(side)]
    assert at([0.5, 0.0, 0.0, 0.0, 0.0, 0.5]) == []
    assert at([0.5, -1e-4, -1e-4, -1e-4, -1e-4, 0.5]) == [(1, 4)]
    assert at([0.5, -0.3, -0.3, -0.3, 0.5]) == []
    assert at([0.5, -0.3, -0.3, None, -0.3, -0.3, 0.5]) == []
    turn = body_turn.turns([0.5, -0.1, -0.2, -0.3, -0.1, 0.5])
    assert turn == [{"from": 1, "to": 4, "least": -0.3, "at": 3}]


def test_the_frames_a_cut_holds_of_a_turn_are_named_in_runs():
    from sprite_gen.video import body_turn

    assert body_turn.spans([36, 22, 23, 37, 38]) == "22-23 and 36-38" and body_turn.spans([40]) == "40"


def test_the_turning_walker_reads_a_turn_where_it_turns_and_nowhere_else():
    from sprite_gen.video import body_turn

    read = body_turn.read(_frames(turns=[TURN]))
    assert read["read"] and read["still_side"] >= body_turn.SIDE_SHOWN_MIN and read["side"][0] == 1.0
    (turn,) = read["turns"]
    assert TURN[0] <= turn["from"] and turn["to"] <= TURN[1] and turn["to"] - turn["from"] + 1 >= body_turn.TURN_RUN_MIN
    assert turn["least"] < -0.2
    steady = body_turn.read(_frames())
    assert steady["read"] and steady["turns"] == [] and min(steady["side"]) > 0.5


@pytest.mark.parametrize("turns, frames", [((12, 18), 0), ((12, 19), 4)], ids=["three-frames", "four-frames"])
def test_a_dip_of_three_frames_is_the_walks_own_and_four_are_a_turn(turns, frames):
    """The walker facing past its mirror three frames in a row reads no turn, however deep; four in a row, one."""
    from sprite_gen.video import body_turn

    read = body_turn.read(_frames(turns=[turns]))
    under = [k for k, v in enumerate(read["side"]) if v < body_turn.TURN_SIDE_MAX]
    assert len(under) == (frames or 3) and min(read["side"]) < -0.2
    assert [t["to"] - t["from"] + 1 for t in read["turns"]] == ([frames] if frames else [])


def test_a_square_still_is_not_read_and_the_bound_turns_over_at_the_stills_own_side(monkeypatch):
    """SIDE_SHOWN_MIN: the still's head and shoulders off their mirror under it, the clip is not read (a back walk's far
    step lays its torso nearer its mirror each step); at it, read."""
    from sprite_gen.video import body_turn

    square = body_turn.read(_frames(square=True))
    assert square["read"] is False and square["still_side"] < body_turn.SIDE_SHOWN_MIN and "square" in square["why"]
    shown = body_turn.read(_frames(turns=[TURN]))["still_side"]
    monkeypatch.setattr(body_turn, "SIDE_SHOWN_MIN", shown)
    assert body_turn.read(_frames(turns=[TURN]))["read"] is True
    monkeypatch.setattr(body_turn, "SIDE_SHOWN_MIN", shown + 1e-4)
    assert body_turn.read(_frames(turns=[TURN]))["read"] is False


def test_a_turn_is_counted_in_frames_however_the_walk_is_held():
    """Counted in frames, not drawings: the same turn drawn on twos is the same frames as on ones — counted in drawings
    it would be half as long, and a walk held on twos would need a turn twice as long to be one."""
    from sprite_gen.video import body_turn

    ones = body_turn.read(_frames(turns=[TURN]))["turns"]
    twos = body_turn.read(_frames(turns=[TURN], hold=2))["turns"]
    assert len(ones) == len(twos) == 1
    assert abs(ones[0]["from"] - twos[0]["from"]) <= 1 and abs(ones[0]["to"] - twos[0]["to"]) <= 1
    drawings = len({k - k % 2 for k in range(twos[0]["from"], twos[0]["to"] + 1)})
    assert drawings < twos[0]["to"] - twos[0]["from"] + 1


@pytest.mark.slow
def test_a_lead_in_is_read_against_the_still(tmp_path):
    """The reference is the clip's first frame, the still, lead-in or not (the first 8 frames filmed from 0.6 of the
    size, which the search leaves out): the turn reads where it reads without the lead-in, and the walk where it does
    not turn stays clear of 0. Read from the first frame after the lead-in — a stride's pose, whose mirror the walk
    draws half a cycle on — a walk that does not turn reads a turn every stride."""
    from sprite_gen.video import body_turn

    report = _cut(tmp_path / "out", _keyed(tmp_path, turns=[TURN], lead=8))
    turn = report["body_turn"]
    skip = report["lead_in"]["search_from"]
    assert skip > 0 and turn["reference"] == 0
    (led,), (want,) = turn["turns"], body_turn.read(_frames(turns=[TURN]))["turns"]
    assert (led["from"], led["to"]) == (want["from"], want["to"]) and led["least"] < -0.1
    assert min(v for k, v in enumerate(turn["side"]) if k >= skip and not led["from"] - 2 <= k <= led["to"] + 2) > 0.1
    steady = _frames(lead=8)
    assert body_turn.read(steady)["turns"] == []
    assert len(body_turn.read(steady[skip:])["turns"]) >= 2


def test_a_part_swung_up_beside_the_head_does_not_pull_its_centre(monkeypatch):
    """HEAD_REACH: the side walker with a 60 px tail on a 45-frame beat (tests/video/test_walk_steps.py, the half step),
    its tail swung up into the head's rows every swing. The head's centre read within 0.3 of the body's height of it,
    the walker reads no turn; within 0.4 the tail pulls the centre aside and the torso reads a turn every swing (the
    search then took a cut that does not close)."""
    from sprite_gen.video import body_turn
    from tests.video.test_walk_steps import _side

    frames = [_side(k, cycle=29, tail_cycle=45, tail=60, arms=False) for k in range(FRAMES)]
    assert body_turn.HEAD_REACH == 0.3
    read = body_turn.read(frames)
    assert read["read"] and read["turns"] == [] and min(read["side"]) > 0
    monkeypatch.setattr(body_turn, "HEAD_REACH", 0.4)
    assert len(body_turn.read(frames)["turns"]) == 2


def test_a_body_with_no_neck_reads_its_head_and_chest():
    """A hood widening down to the shoulders has no neck narrower than the head: the narrowest row is the top of the
    range, the crop holds the head and the chest, and the turn reads as on a body with a neck."""
    from sprite_gen.video import body_turn

    hooded = body_turn.read(_frames(turns=[TURN], hood=True))
    assert hooded["neck"] == round(min(float(f) for f in body_turn.ROWS if f >= body_turn.NECK_RANGE[0]), 2)
    assert len(hooded["turns"]) == 1 and TURN[0] <= hooded["turns"][0]["from"] and hooded["turns"][0]["to"] <= TURN[1]
    assert body_turn.read(_frames(hood=True))["turns"] == []


# --- what the reading cannot read ----------------------------------------------------------------------------------


def _gone(read: dict) -> list[int]:
    return [k for k, v in enumerate(read["side"]) if v is None]


def test_a_prop_as_high_as_the_head_and_apart_from_it_leaves_the_still_unread():
    """HEAD_REACH, where nothing lies within it: a staff held beside the walker, its ornament as high as the head and as
    heavy in the top 15 % of the body, puts the middle of those rows between the two, no pixel within 0.3 of the body's
    height of it. The top does not tell which is the head, and the still is not read — not centred on the middle no
    pixel holds (0/0), nor on the larger part (the two weigh the same)."""
    from sprite_gen.video import body_turn

    read = body_turn.read(_frames(staff=range(FRAMES)))
    assert read["read"] is False and read["still_side"] is None and read["turns"] == []
    assert read["why"].startswith("the clip's first frame has no head to centre on: the top 15 % of its body holds "
                                  "parts apart, none within 0.3 of its height of their middle")


def test_a_frame_with_no_head_to_centre_on_is_not_read_and_the_rest_read_as_before():
    """The staff over frames 20-29 only, and frame 40 a sliver of 2 pixels (too small a body for a pixel to lie within
    reach of their middle): those frames are None, `unread` says why, and every other frame reads what it reads without
    them."""
    from sprite_gen.video import body_turn

    frames = _frames(staff=range(20, 30))
    frames[40] = _speck(2)
    read, plain = body_turn.read(frames), body_turn.read(_frames())
    gone = [*range(20, 30), 40]
    assert read["read"] and read["turns"] == [] and _gone(read) == gone
    assert all(read["side"][k] == plain["side"][k] for k in range(FRAMES) if k not in gone)
    (unread,) = read["unread"]
    assert unread["frames"] == gone and unread["why"].startswith("has no head to centre on")


def test_a_still_whose_torso_rows_hold_no_pixel_is_not_read():
    """A pennant apart above the walker tops the still's box, 240 px over its head: the narrowest row in NECK_RANGE is
    the gap's first, and the torso's rows, from there down half the box, hold no solid pixel. No width is measured
    across them; the still is not read."""
    from sprite_gen.video import body_turn

    read = body_turn.read(_frames(apart=range(FRAMES)))
    assert read["read"] is False and read["neck"] == 0.11 and read["turns"] == []
    assert read["why"].startswith("the clip's first frame has no solid pixel in its rows 0.11-")


def test_a_frame_whose_torso_crop_holds_nothing_is_not_read_as_0():
    """The pennant over frames 20-29 only: those frames' boxes run from it, and their torso crops, the still's rows of a
    box three times the body's height, hold nothing — read, they were a torso at 0, as near the still as its mirror.
    They are None, `unread` says why, and every other frame reads what it reads without them."""
    from sprite_gen.video import body_turn

    read, plain = body_turn.read(_frames(apart=range(20, 30))), body_turn.read(_frames(apart=()))
    gone = list(range(20, 30))
    assert read["read"] and _gone(read) == gone
    assert all(read["side"][k] == plain["side"][k] for k in range(FRAMES) if k not in gone)
    (unread,) = read["unread"]
    assert unread["frames"] == gone and unread["why"].startswith("has nothing to read in its crop of rows")


def test_a_crop_of_no_width_is_not_read():
    """A still whose torso's rows hold one column, under its head's centre (a head on a pole): its torso is 0 wide and
    the still is not read. A frame of one solid pixel, its crop as wide as the still's torso but over a body 1 pixel
    high, crops no column either: it is None."""
    from sprite_gen.video import body_turn

    pole = Image.new("RGBA", (W, H))
    d = ImageDraw.Draw(pole)
    d.rectangle((140, 100, 149, 120), fill=(0, 0, 0, 255))
    d.rectangle((150, 100, 160, 120), fill=(255, 255, 255, 255))
    d.line((150, 121, 150, 220), fill=(255, 255, 255, 255))
    read = body_turn.read([pole] * 8)
    assert read["read"] is False and read["still_side"] >= body_turn.SIDE_SHOWN_MIN
    assert read["why"].startswith("the clip's first frame has a crop of no width over its rows")
    frames = _frames()
    frames[40] = _speck(1)
    read = body_turn.read(frames)
    assert read["read"] and _gone(read) == [40] and read["unread"][0]["why"].startswith("has a crop of no width")


def test_a_clip_with_no_frames_is_not_read():
    from sprite_gen.video import body_turn

    read = body_turn.read([])
    assert read["read"] is False and read["why"] == "the clip has no frames" and read["turns"] == []


# --- the cut -------------------------------------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("turned", [(14, 24), (24, 34), TURN], ids=["14-24", "24-34", "30-40"])
def test_a_cut_that_holds_the_turn_gives_way_to_one_that_does_not(tmp_path, monkeypatch, turned):
    """The walker turns over frames 14-24, 24-34 or 30-40: the search took a cut across the turn (11/24, 17/24, 22/24:
    the body swinging round in the loop). Read, the turn's windows are refused and the cut is one that leaves it out."""
    keyed = _keyed(tmp_path, turns=[turned])
    report = _cut(tmp_path / "read", keyed)
    assert not _holds(report, range(*turned))
    turn = report["body_turn"]
    assert turn["verdict"] == "avoided" and turn["refused"] > 0 and turn["cut"]["holds"] == []
    with monkeypatch.context() as m:
        _unread(m)
        before = _cut(tmp_path / "unread", keyed)
    assert _holds(before, range(*turned))  # the cut as the search took it before


@pytest.mark.slow
def test_a_turn_every_cycle_keeps_the_cut_with_a_warning_or_is_refused(tmp_path, monkeypatch, capsys):
    """A walker that turns in every cycle (as a three-quarter walk that turns to the side each stride): no cut leaves
    the turn out. warn keeps the cut the search took before, byte for byte, and a warning line names the turn; refuse
    refuses the clip in a line of its own, before any other."""
    keyed = _keyed(tmp_path, every=(6, 8))
    capsys.readouterr()
    report = _cut(tmp_path / "warn", keyed)
    err = capsys.readouterr().err
    assert "video-loop: warning: the body turns in the walk — " in err and "--body-turn refuse" in err
    assert report["body_turn"]["verdict"] == "kept" and report["body_turn"]["cut"]["holds"]
    with monkeypatch.context() as m:
        _unread(m)
        _cut(tmp_path / "unread", keyed)
    _same_cut(tmp_path / "warn", tmp_path / "unread")
    said, refused = _refused(tmp_path / "refuse", keyed, capsys, "--body-turn", "refuse")
    assert said.startswith("video-loop: the body turns in the walk — its torso faces nearer the still's mirror")
    assert "film it again with the torso and hips kept at the still's angle" in said
    assert refused["body_turn"]["verdict"] == "refused"
    assert capsys.readouterr().err == ""  # the refusal is the first line the run says
    assert not (tmp_path / "refuse" / "loop.strip.png").exists()


@pytest.mark.slow
@pytest.mark.parametrize("drawn", [{}, {"square": True}], ids=["diagonal", "square"])
def test_a_walk_that_does_not_turn_is_cut_as_before(tmp_path, monkeypatch, drawn):
    """No turn — read and none found, or a square still not read — and the cut is every byte the search alone made."""
    keyed = _keyed(tmp_path, **drawn)
    report = _cut(tmp_path / "read", keyed)
    assert report["body_turn"]["verdict"] == ("none" if not drawn else "not-read")
    with monkeypatch.context() as m:
        _unread(m)
        _cut(tmp_path / "unread", keyed)
    _same_cut(tmp_path / "read", tmp_path / "unread")


@pytest.mark.slow
@pytest.mark.parametrize("drawn", [{"staff": range(FRAMES)}, {"apart": (0,)}], ids=["staff", "apart"])
def test_a_walk_whose_still_is_not_read_is_cut_as_before(tmp_path, monkeypatch, drawn):
    """warn, the default, where the still cannot be read — the staff's ornament as high as the head (3.3.0 cut it
    11/24; the reading's 0/0 on it ended the run, `cannot convert float NaN to integer`), or the pennant apart above the
    body in the still alone (over every frame, the box's upper half is the empty gap, where the motion anchor finds no
    textured region to follow): the loop is cut, every byte the search alone cuts, and the report says why the clip
    was not read."""
    keyed = _keyed(tmp_path, **drawn)
    report = _cut(tmp_path / "read", keyed)
    turn = report["body_turn"]
    assert turn["read"] is False and turn["verdict"] == "not-read"
    assert turn["why"].startswith("the clip's first frame has ")
    with monkeypatch.context() as m:
        _unread(m)
        _cut(tmp_path / "unread", keyed)
    _same_cut(tmp_path / "read", tmp_path / "unread")


@pytest.mark.slow
def test_a_walk_with_no_cycle_keeps_its_refusal_and_names_the_turn(tmp_path, monkeypatch, capsys):
    """The walker turns and the search finds no cycle with the turn or without it (every window refused, as on a clip
    that holds no repeat; its feet not read). warn refuses it in the words it was refused in before, after a warning line
    naming the turn; refuse names the turn."""
    def nothing(*args, **kwargs):
        raise ValueError("no periodic cycle found — no supported local two-step repeat after drift analysis")

    keyed = _keyed(tmp_path, turns=[TURN])
    monkeypatch.setattr(local_cycle, "detect", nothing)
    with feet_unseen():
        said, report = _refused(tmp_path / "warn", keyed, capsys)
        err = capsys.readouterr().err
        assert said.startswith("video-loop: no periodic cycle found")
        assert report["body_turn"]["verdict"] == "unfound" and "video-loop: warning: the body turns in the walk" in err
        said, report = _refused(tmp_path / "refuse", keyed, capsys, "--body-turn", "refuse")
    assert said.startswith("video-loop: the body turns in the walk") and report["body_turn"]["verdict"] == "refused"


def test_refuse_needs_motion_auto(tmp_path, capsys):
    keyed = _keyed(tmp_path)
    with pytest.raises(SystemExit, match="--body-turn refuse reads the body's turn"):
        loop.main(["--frames-dir", str(keyed), "--out-dir", str(tmp_path / "out"), "--state", "walk", "--body-turn", "refuse"])
