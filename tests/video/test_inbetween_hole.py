# SPDX-License-Identifier: Apache-2.0
"""A hole through a limb (docs/loop-repair.md section 4): `rife.hole` reads the transparent holes a made frame opened
inside its silhouette that neither source frame beside it has there, in place or carried by the motion the two show, and
the cycle alignment gives such a frame way to the nearer source frame (`--between auto`). RIFE runs a frame's colour and
its coverage apart, and the coverage run can carry the gap one source frame has between two legs into a leg the colour
run keeps whole: a transparent band through the leg with its fill on both sides. Nothing is added, no part coverage is
left, and around the band the leg's own fill is all the line there is — the outline readings judge an outline, not a
hole. A gap the source frames have is theirs wherever the motion between them carries it: kept, moved with the body or
with a part of it, or opened by a step between two legs.

The fixtures are synthetic: a figure of rectangles outlined in black, legs filled dark (under the dark line, 70/255,
as dark tights or boots are), the near leg over the far one; the first source frame with its legs side by side, the
second with a gap between them, as a step opens it; a gap between the arm and the body in every frame. Then a body with
a gap through it, moved whole; a figure whose legs part above boots that close the gap below, still and moving; and a
body that stays with a part on it, a hole through the part, moved."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from sprite_gen.video import align, rife
from sprite_gen.video.interpolation_quality import faults

W, H = 160, 200
LINE, BODY = (0, 0, 0, 255), (250, 240, 235, 255)
FAR, NEAR = (70, 52, 40, 255), (78, 58, 46, 255)  # luma 56 and 62: under the dark line, over the black one by 50/255 and more
SIDE_BY_SIDE, APART, BETWEEN = (50, 80), (45, 85), (47, 82)  # the far and the near leg's left column
BAND = (58, 120, 66, 160)  # inside the far leg between, 8 px wide: 8 px or more from anything under solid in either source
MOVES = (0, 4, 8, 12, 24, 32)  # px the body with a gap moves from one source frame to the next


def _box(d: ImageDraw.ImageDraw, x0: int, y0: int, x1: int, y1: int, fill: tuple[int, ...]) -> None:
    d.rectangle([x0, y0, x1 - 1, y1 - 1], fill=LINE)
    d.rectangle([x0 + 2, y0 + 2, x1 - 3, y1 - 3], fill=fill)


def _figure(legs: tuple[int, int], *, arm_gap: int = 0) -> Image.Image:
    """A body over two legs 30 px wide, the far leg from column legs[0], the near from legs[1] drawn over it; an arm
    on the right joined to the body above and below, the gap between them 8 px wide, `arm_gap` px further right."""
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    far, near = legs
    _box(d, far, 96, far + 30, 180, FAR)
    _box(d, near, 96, near + 30, 185, NEAR)
    x = 128 + arm_gap
    _box(d, x, 30, x + 12, 90, BODY)
    for y in (30, 80):
        _box(d, 110, y, x + 12, y + 10, BODY)
    _box(d, 40, 20, 120, 100, BODY)
    d.rectangle([118, 32, 119, 37], fill=BODY)  # the arm's joins with no line across them
    d.rectangle([118, 82, 119, 87], fill=BODY)
    return im


def _body(dx: int = 0, dy: int = 0) -> Image.Image:
    """A body 120 x 180 px of one colour with a gap 8 x 40 px through it, on a 240 x 220 px canvas, moved dx columns
    right and dy rows down."""
    im = Image.new("RGBA", (240, 220), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle([40 + dx, 20 + dy, 159 + dx, 199 + dy], fill=(45, 55, 60, 255))
    d.rectangle([120 + dx, 60 + dy, 127 + dx, 99 + dy], fill=(0, 0, 0, 0))
    return im


def _parting(x: int, far: int, near: int) -> Image.Image:
    """A body 90 px wide from column x over two legs 24 px wide, the far leg from column x + far and the near from
    x + near, each standing in a boot that reaches under the other leg: legs side by side touch, and a gap a step opens
    between them is closed by the boots below."""
    im = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    fx, nx = x + far, x + near
    _box(d, fx, 90, fx + 24, 170, FAR)
    _box(d, fx, 160, nx + 12, 180, (120, 80, 50, 255))
    _box(d, nx, 90, nx + 24, 172, NEAR)
    _box(d, nx - 6, 162, nx + 30, 184, (130, 90, 60, 255))
    _box(d, x, 20, x + 90, 100, BODY)
    return im


def _ring(dx: int) -> Image.Image:
    """A body 200 x 180 px of one colour that stays, with a part on it 40 px square, a hole 14 px square through the part,
    the part moved dx columns right."""
    im = Image.new("RGBA", (260, 220), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle([30, 20, 229, 199], fill=(45, 55, 60, 255))
    d.rectangle([100 + dx, 80, 139 + dx, 119], fill=(150, 90, 60, 255))
    d.rectangle([113 + dx, 93, 126 + dx, 106], fill=(0, 0, 0, 0))
    return im


def _holed(frame: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    out = frame.copy()
    ImageDraw.Draw(out).rectangle([box[0], box[1], box[2] - 1, box[3] - 1], fill=(0, 0, 0, 0))
    return out


def _measure(made: Image.Image, a: Image.Image, b: Image.Image) -> dict[str, float | None]:
    return {**rife.smear(made, a, b), "outline_shift": rife.outline_shift(made, a, b), "ghost": rife.ghost(made)}


def _shifted(frame: Image.Image, dx: int, dy: int) -> Image.Image:
    out = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    out.paste(frame, (dx, dy))
    return out


# What the alignment does with a frame made between two source frames that it keeps as made: both frames made at t 0.5
# (times .5 and 1.5 of four) delivered as the very frame made, nothing given way, no fault.
KEPT = {"made_at": [1, 3], "nearest_at": [], "faults": [[], []], "delivered": True}


def _resampled(a: Image.Image, made: Image.Image, b: Image.Image) -> dict[str, object]:
    """What the alignment does resampling a and b to four frames, both frames made at t 0.5 being `made`."""
    out, facts = align.resample([a, b], 4, lambda x, y, t: made)
    return {"made_at": facts["made_at"], "nearest_at": facts["nearest_at"], "faults": [m["faults"] for m in facts["smear"]],
            "delivered": out[1] is made and out[3] is made}


def test_a_hole_through_a_leg_is_read_and_a_gap_the_sources_have_is_not():
    """The band reads as a hole over its own pixels; the frame drawn between, the arm's gap all three frames have, and
    that gap moved a few pixels (within rife.HOLE_REACH of theirs) read none. A pinhole thinner than rife.HOLE_THICK is
    opened away."""
    a, b, clean = _figure(SIDE_BY_SIDE), _figure(APART), _figure(BETWEEN)
    alpha = lambda im: np.asarray(im)[..., 3]
    x0, y0, x1, y1 = BAND
    assert (alpha(a)[y0 - 8:y1 + 8, x0 - 8:x1 + 8] == 255).all() and (alpha(b)[y0 - 8:y1 + 8, x0 - 8:x1 + 8] == 255).all()
    holed = _holed(clean, BAND)
    assert rife.hole(holed, a, b, 0.5) == round((x1 - x0) * (y1 - y0) / int((alpha(holed) >= 128).sum()), 5)
    assert rife.hole(clean, a, b, 0.5) == 0
    assert (alpha(clean)[40:80, 120:128] == 0).all()  # the arm's gap, enclosed
    assert rife.hole(_figure(BETWEEN, arm_gap=3), a, b, 0.5) == 0
    assert rife.hole(_holed(clean, (60, 130, 63, 133)), a, b, 0.5) == 0
    assert rife.hole(Image.new("RGBA", (W, H)), a, b, 0.5) == 0


def test_the_outline_readings_pass_a_hole_through_a_dark_leg():
    """Around the band the leg's fill is darker than the dark line, so the dark line counts it as outline; read rank by
    rank against black lines it is lost — but the fault `outline` needs both. Nothing dark is added and nothing is part
    covered: the readings before the hole pass it."""
    a, b = _figure(SIDE_BY_SIDE), _figure(APART)
    m = _measure(_holed(_figure(BETWEEN), BAND), a, b)
    assert m["outline_loss"] <= 0 and m["outline_shift"] > 0.05 and m["dark_excess"] < 0
    assert m["ghost"] == 0 and m["partial_excess"] <= 0 and faults(m) == []
    assert faults({**m, "hole": rife.hole(_holed(_figure(BETWEEN), BAND), a, b, 0.5)}) == ["hole"]


def test_alignment_takes_the_nearer_frame_for_a_hole_through_a_leg():
    """Resampled from two frames to four (times 0, .5, 1, 1.5), both frames made at t 0.5 carry the band: `--between
    auto` takes the nearer source frame at each and names the fault in the report's words; `rife` keeps them, named."""
    a, b = _figure(SIDE_BY_SIDE), _figure(APART)
    holed = _holed(_figure(BETWEEN), BAND)
    out, facts = align.resample([a, b], 4, lambda x, y, t: holed)
    assert facts["made_at"] == [] and facts["nearest_at"] == [1, 3] and out[1] is b and out[3] is a
    assert all(m["method"] == "nearest" and m["faults"] == ["hole"] and m["hole"] > 0 for m in facts["smear"])
    assert "a gap through a limb" in align._fault_line("walk-front", facts["smear"][0])
    out, facts = align.resample([a, b], 4, lambda x, y, t: holed, between="rife")
    assert facts["made_at"] == [1, 3] and out[1] is holed and all(m["faults"] == ["hole"] for m in facts["smear"])


def test_alignment_keeps_a_clean_frame_and_one_with_the_gap_the_sources_have():
    """The frame drawn between, and the same with the arm's gap a few pixels on, are kept as made: the same frame
    delivered, nothing given way."""
    a, b = _figure(SIDE_BY_SIDE), _figure(APART)
    for made in (_figure(BETWEEN), _figure(BETWEEN, arm_gap=3)):
        out, facts = align.resample([a, b], 4, lambda x, y, t: made)
        assert facts["made_at"] == [1, 3] and facts["nearest_at"] == [] and out[1] is made and out[3] is made
        assert all(m["faults"] == [] and m["hole"] == 0 for m in facts["smear"])


def test_a_gap_both_source_frames_have_moved_with_the_body_is_theirs():
    """The body with its gap moved whole from one source frame to the next, across and diagonally, by each of MOVES, the
    frame between drawn halfway: each source frame moved half the way onto it gives it byte for byte. From 24 px on, the
    gap lies 5 px and more from where either source frame has it, but on the way the body moves, so it is theirs: every
    frame is kept as made, and reads no hole."""
    seen = {}
    for move in MOVES:
        for down in (0, 1):
            dx, dy = move // 2, down * move // 2
            a, made, b = _body(-dx, -dy), _body(), _body(dx, dy)
            assert _shifted(a, dx, dy).tobytes() == made.tobytes() == _shifted(b, -dx, -dy).tobytes()
            seen[move, "diagonally" if down else "across"] = (a, made, b, _resampled(a, made, b))
    assert {k: v[3] for k, v in seen.items() if v[3] != KEPT} == {}
    assert {k: rife.hole(made, a, b, 0.5) for k, (a, made, b, _) in seen.items()} == {k: 0.0 for k in seen}


def test_a_gap_one_source_frame_has_half_open_between_is_theirs():
    """Legs side by side in one source frame, touching, and parted in the other; the frame between drawn with them half
    parted. The gap there is the parted frame's, opened halfway: with the figure still, inside where that frame has it;
    with the figure moving 30 px, 8 px and more beside it, where the figure carries it. Both are kept as made."""
    seen = {}
    for name, (xa, fa, na), (xb, fb, nb) in (("still", (40, 10, 34), (40, 4, 40)), ("moving", (20, 10, 34), (50, 10, 50))):
        a, b = _parting(xa, fa, na), _parting(xb, fb, nb)
        x, far, near = (xa + xb) // 2, (fa + fb) // 2, (na + nb) // 2
        made = _parting(x, far, near)
        assert na == fa + 24 and (np.asarray(made)[110:150, x + far + 24:x + near, 3] == 0).all()
        seen[name] = (a, made, b, _resampled(a, made, b))
    assert {k: v[3] for k, v in seen.items() if v[3] != KEPT} == {}
    assert {k: rife.hole(made, a, b, 0.5) for k, (a, made, b, _) in seen.items()} == {"still": 0.0, "moving": 0.0}


def test_a_gap_moved_with_a_part_while_the_body_stays_is_theirs():
    """A part with a hole through it moves 32 px across a body that stays, the frame between drawn with the part
    halfway: the hole lies 5 px and more from where either source frame has it, on the way the part moves. The body
    outweighs the part in the whole figure, which stays; the hole is carried with what lies round it, and the frame is
    kept as made."""
    a, made, b = _ring(-16), _ring(0), _ring(16)
    assert _resampled(a, made, b) == KEPT and rife.hole(made, a, b, 0.5) == 0


def test_a_hole_through_a_moving_body_is_still_read():
    """The parting figure moving 30 px, the frame between with a band 8 px wide through its body: wherever its pixels
    lay, in place or carried, both source frames show the body there, so the band reads as a hole over its own pixels
    and the frame gives way to the nearer source frame. (The body's fill is light, so its edge round the band also reads
    as an outline lost.)"""
    a, b = _parting(20, 10, 34), _parting(50, 10, 50)
    holed = _holed(_parting(35, 10, 42), (70, 40, 78, 80))
    out, facts = align.resample([a, b], 4, lambda x, y, t: holed)
    assert facts["made_at"] == [] and facts["nearest_at"] == [1, 3] and out[1] is b and out[3] is a
    assert all("hole" in m["faults"] for m in facts["smear"])
    assert rife.hole(holed, a, b, 0.5) == rife.hole(holed, b, a, 0.5) == round(8 * 40 / int((np.asarray(holed)[..., 3] >= 128).sum()), 5)
