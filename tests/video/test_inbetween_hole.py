# SPDX-License-Identifier: Apache-2.0
"""A hole through a limb (docs/loop-repair.md section 4): `rife.hole` reads the transparent holes a made frame opened
inside its silhouette that neither source frame beside it has there, and the cycle alignment gives such a frame way to
the nearer source frame (`--between auto`). RIFE runs a frame's colour and its coverage apart, and the coverage run can
carry the gap one source frame has between two legs into a leg the colour run keeps whole: a transparent band through
the leg with its fill on both sides. Nothing is added, no part coverage is left, and around the band the leg's own fill
is all the line there is — the outline readings judge an outline, not a hole.

The fixtures are synthetic: a figure of rectangles outlined in black, legs filled dark (under the dark line, 70/255,
as dark tights or boots are), the near leg over the far one; the first source frame with its legs side by side, the
second with a gap between them, as a step opens it; a gap between the arm and the body in every frame."""

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


def _holed(frame: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    out = frame.copy()
    ImageDraw.Draw(out).rectangle([box[0], box[1], box[2] - 1, box[3] - 1], fill=(0, 0, 0, 0))
    return out


def _measure(made: Image.Image, a: Image.Image, b: Image.Image) -> dict[str, float | None]:
    return {**rife.smear(made, a, b), "outline_shift": rife.outline_shift(made, a, b), "ghost": rife.ghost(made)}


def test_a_hole_through_a_leg_is_read_and_a_gap_the_sources_have_is_not():
    """The band reads as a hole over its own pixels; the frame drawn between, the arm's gap all three frames have, and
    that gap moved a few pixels (within rife.HOLE_REACH of theirs) read none. A pinhole thinner than rife.HOLE_THICK is
    opened away."""
    a, b, clean = _figure(SIDE_BY_SIDE), _figure(APART), _figure(BETWEEN)
    alpha = lambda im: np.asarray(im)[..., 3]
    x0, y0, x1, y1 = BAND
    assert (alpha(a)[y0 - 8:y1 + 8, x0 - 8:x1 + 8] == 255).all() and (alpha(b)[y0 - 8:y1 + 8, x0 - 8:x1 + 8] == 255).all()
    holed = _holed(clean, BAND)
    assert rife.hole(holed, a, b) == round((x1 - x0) * (y1 - y0) / int((alpha(holed) >= 128).sum()), 5)
    assert rife.hole(clean, a, b) == 0
    assert (alpha(clean)[40:80, 120:128] == 0).all()  # the arm's gap, enclosed
    assert rife.hole(_figure(BETWEEN, arm_gap=3), a, b) == 0
    assert rife.hole(_holed(clean, (60, 130, 63, 133)), a, b) == 0
    assert rife.hole(Image.new("RGBA", (W, H)), a, b) == 0


def test_the_outline_readings_pass_a_hole_through_a_dark_leg():
    """Around the band the leg's fill is darker than the dark line, so the dark line counts it as outline; read rank by
    rank against black lines it is lost — but the fault `outline` needs both. Nothing dark is added and nothing is part
    covered: the readings before the hole pass it."""
    a, b = _figure(SIDE_BY_SIDE), _figure(APART)
    m = _measure(_holed(_figure(BETWEEN), BAND), a, b)
    assert m["outline_loss"] <= 0 and m["outline_shift"] > 0.05 and m["dark_excess"] < 0
    assert m["ghost"] == 0 and m["partial_excess"] <= 0 and faults(m) == []
    assert faults({**m, "hole": rife.hole(_holed(_figure(BETWEEN), BAND), a, b)}) == ["hole"]


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
