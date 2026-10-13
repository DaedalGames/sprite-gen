# SPDX-License-Identifier: Apache-2.0
"""Gates G2..G6 and G8 of `pixel-draw` on recorded answers (tests/fixtures/pixel_draw) and on grids
made to break exactly one rule. G2, G3, G6 and G8's canvas edge fail a candidate; G4, G5, the walk
check and G8's body and floor checks only warn. The `*-safe-area-pass` answers were drawn under the
safe-area prompt; the other recorded sprites when the prompt asked to fill the canvas."""
from __future__ import annotations

from pathlib import Path

import pytest

from sprite_gen.gen import pixel_gates, prepare
from sprite_gen.gen import pixel_grid as pg
from sprite_gen.gen.pixel_gates import evaluate, g6_frames, seam_ratio

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "pixel_draw"
WALK = ("walk1", "walk2", "walk3", "walk4")
FOURDIR = ("down", "up", "left", "right")
FREE16 = pg.palette_rule("free")

INK, SKIN, GRASS, MOSS = (40, 30, 46, 255), (240, 192, 152, 255), (78, 144, 56, 255), (92, 158, 66, 255)
CLEAR = (0, 0, 0, 0)


def frames_of(name: str, size: int, frames: tuple[str, ...]):
    grid, issues = pg.parse_answer((FIXTURES / name).read_text(encoding="utf-8"), "chargrid", size, size, frames)
    assert issues == []
    return {f: pg.frame_pixels(grid, f) for f in frames}


def blob(size: int = 8, top: int = 2, bottom: int = 5, fill=INK):
    """A transparent square with an opaque block rows top..bottom, columns 2..size-3."""
    return [[fill if top <= y <= bottom and 2 <= x <= size - 3 else CLEAR for x in range(size)] for y in range(size)]


def test_recorded_sprite_passes_every_gate():
    gates = evaluate(frames_of("sprite-16-safe-area-pass.txt", 16, ("sprite",)), kind="sprite", rule=FREE16,
                     walk_cycle=False)
    assert gates["ok"] and gates["failed"] == [] and gates["warnings"] == []
    assert gates["G2"]["colors_used"] <= 16
    assert gates["G3"]["frames"]["sprite"]["corners_transparent"]
    assert gates["G5"]["isolated_pixels"] == 0 and "G6" not in gates and "G4" not in gates
    assert gates["G8"]["frames"]["sprite"]["bbox"] == [3, 1, 12, 14] and gates["G8"]["floor_row"] == 14


def test_recorded_tile_is_opaque_and_repeats():
    gates = evaluate(frames_of("tile-16-pass.txt", 16, ("tile",)), kind="tile", rule=FREE16, walk_cycle=False)
    assert gates["ok"] and gates["G3"]["frames"]["tile"]["opaque_fraction"] == 1.0
    assert gates["G4"]["seam_ratio"] <= 1.5 and not gates["G4"]["warn"]


def test_g2_counts_colours_against_the_limit_and_the_fixed_set():
    frames = frames_of("sprite-16-safe-area-pass.txt", 16, ("sprite",))
    used = evaluate(frames, kind="sprite", rule=FREE16, walk_cycle=False)["G2"]["colors_used"]
    few = evaluate(frames, kind="sprite", rule=pg.palette_rule("free:4"), walk_cycle=False)
    assert not few["ok"] and few["failed"] == ["G2"] and few["G2"]["colors_used"] == used
    pico = evaluate(frames, kind="sprite", rule=pg.palette_rule("pico8"), walk_cycle=False)
    assert not pico["ok"] and pico["G2"]["off_palette"]
    exact = pg.palette_rule(",".join(sorted({"#{:02X}{:02X}{:02X}".format(*p[:3])
                                             for row in frames["sprite"] for p in row if p[3]})))
    assert evaluate(frames, kind="sprite", rule=exact, walk_cycle=False)["G2"]["ok"]


@pytest.mark.parametrize("frame,kind", [
    ([[INK] + [CLEAR] * 7] + [[CLEAR] * 8 for _ in range(7)], "sprite"),   # an opaque corner
    ([[INK] * 8 for _ in range(3)] + blob()[3:], "icon"),                    # border mostly opaque
    ([[CLEAR] * 8 for _ in range(8)], "sprite"),                            # nothing drawn
    ([[GRASS] * 8 for _ in range(7)] + [[GRASS] * 7 + [CLEAR]], "tile"),    # one hole in a tile
])
def test_g3_alpha_failures(frame, kind):
    gates = evaluate({"f": frame}, kind=kind, rule=FREE16, walk_cycle=False)
    assert not gates["ok"] and "G3" in gates["failed"]


def test_g4_seam_only_warns():
    striped = [[GRASS if x < 2 or x >= 6 else MOSS for x in range(8)] for _ in range(8)]  # wraps cleanly
    seam = [[GRASS if x < 7 else INK for x in range(8)] for _ in range(8)]
    assert seam_ratio(striped) <= 1.5
    gates = evaluate({"tile": seam}, kind="tile", rule=FREE16, walk_cycle=False)
    assert gates["G4"]["warn"] and gates["G4"]["seam_ratio"] > 1.5
    assert gates["ok"] and any(w.startswith("G4 tile seam") for w in gates["warnings"])


def test_g5_stray_pixels_only_warn():
    frame = blob()
    frame[7][7] = CLEAR
    frame[0][4] = INK  # a speck above the block, touching nothing
    gates = evaluate({"f": frame}, kind="icon", rule=FREE16, walk_cycle=False)  # an icon has no floor row
    assert gates["G5"]["isolated_pixels"] == 1 and gates["ok"]
    assert gates["warnings"] == ["G5 1 stray pixel(s) with no opaque neighbour"]


def test_recorded_four_directions_stand_on_one_line():
    gates = evaluate(frames_of("fourdir-16-pass.txt", 16, FOURDIR), kind="sprite", rule=FREE16, walk_cycle=False)
    assert gates["G6"]["ok"] and gates["G6"]["identical_pairs"] == []
    assert len(set(gates["G6"]["baselines"].values())) == 1 and "walk" not in gates["G6"]
    assert gates["failed"] == ["G8"]  # drawn to fill the canvas: the plumes on row 1, the boots on row 16


def test_recorded_walk_passes_with_the_weak_leg_warning():
    gates = evaluate(frames_of("walk-32-safe-area-pass.txt", 32, WALK), kind="sprite", rule=FREE16, walk_cycle=True)
    g6 = gates["G6"]
    assert gates["ok"] and g6["ok"] and g6["height_spread"] == 1  # the passing pose is a pixel higher
    assert g6["walk"]["warn"] and g6["walk"]["ratio"] < 0.4
    assert [w[:30] for w in gates["warnings"]] == ["G6 weak leg alternation: the c"]
    assert set(g6["baselines"].values()) == {gates["G8"]["floor_row"]} == {28}  # row 29 of 32, counted from 1


def test_g6_failures():
    low, high, tall = blob(), blob(top=1, bottom=4), blob(top=0, bottom=5)
    assert not g6_frames({"a": low, "b": [row[:] for row in low]}, walk_cycle=False)["ok"]  # the same frame twice
    shifted = g6_frames({"a": low, "b": high}, walk_cycle=False)
    assert not shifted["ok"] and shifted["baselines"] == {"a": 5, "b": 4}
    spread = g6_frames({"a": low, "b": [r[:] for r in tall]}, walk_cycle=False)
    assert spread["height_spread"] == 2 and not spread["ok"]
    gates = evaluate({"a": low, "b": [row[:] for row in low]}, kind="sprite", rule=FREE16, walk_cycle=False)
    assert gates["failed"] == ["G6"]


# ----------------------------------------------------------------------------- G8 safe area

def canvas(size: int = 32):
    return [[CLEAR] * size for _ in range(size)]


def paint(frame, x0: int, y0: int, x1: int, y1: int, fill=INK):
    """`frame` with columns x0..x1 of rows y0..y1 opaque (inclusive)."""
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            frame[y][x] = fill
    return frame


def figure(size: int = 32):
    """A body filling the safe area's height: at 32 px columns 8..23 of rows 3..28, the 26x26 box's
    rows from top to the floor row."""
    m = int(size * prepare.DEFAULT_SAFE_MARGIN_RATIO)
    return paint(canvas(size), size // 4, m, size - 1 - size // 4, size - 1 - m)


@pytest.mark.parametrize("size,margin,inner", [(8, 0, 8), (16, 1, 14), (24, 2, 20), (32, 3, 26), (48, 4, 40)])
def test_the_safe_area_is_the_engines(size, margin, inner):
    area = pixel_gates.safe_area(size, size)
    cell = prepare.normalize_cell({"width": size, "height": size}, size, None)
    assert area["margin"] == [margin, margin] == [cell["safe_margin_x"], cell["safe_margin_y"]]
    assert margin == int(size * prepare.DEFAULT_SAFE_MARGIN_RATIO)
    assert area["size"] == [inner, inner]
    assert area["box"] == [margin, margin, size - 1 - margin, size - 1 - margin]
    assert area["floor_row"] == size - 1 - margin  # where prepare's fit grounds a frame: the safe area's last row


def test_the_safe_area_follows_prepares_ratio(monkeypatch):
    monkeypatch.setattr(prepare, "DEFAULT_SAFE_MARGIN_RATIO", 0.125)
    assert pixel_gates.safe_area(32, 32)["margin"] == [4, 4]
    assert pixel_gates.safe_area(32, 16)["margin"] == [4, 2]  # an edited grid need not be square


def test_g8_passes_a_body_inside_the_safe_area():
    gates = evaluate({"f": figure()}, kind="sprite", rule=FREE16, walk_cycle=False)
    g8 = gates["G8"]
    assert gates["ok"] and gates["warnings"] == []
    assert g8["ok"] and g8["margin"] == [3, 3] and g8["safe_area"] == [26, 26] and g8["floor_row"] == 28
    frame = g8["frames"]["f"]
    assert frame["edge_pixels"] == 0 and frame["body_box"] == [8, 3, 23, 28] and frame["baseline"] == 28
    assert frame["overflow"] == {"left": 0, "top": 0, "right": 0, "bottom": 0}


def test_g8_fails_a_pixel_on_the_canvas_edge():
    for x, y in ((31, 15), (15, 0), (0, 20), (12, 31)):
        frame = figure()
        frame[y][x] = INK
        gates = evaluate({"f": frame}, kind="sprite", rule=FREE16, walk_cycle=False)
        assert not gates["ok"] and gates["failed"] == ["G8"], (x, y)
        assert gates["G8"]["frames"]["f"]["edge_pixels"] == 1 and gates["G8"]["edge_pixels"] == 1
    standing_on_the_edge = paint(canvas(), 8, 6, 23, 31)
    gates = evaluate({"f": standing_on_the_edge}, kind="icon", rule=FREE16, walk_cycle=False)
    assert gates["failed"] == ["G8"] and gates["G8"]["frames"]["f"]["edge_pixels"] == 16


def test_g8_lets_a_thin_tip_into_the_margin():
    frame = paint(figure(), 24, 15, 30, 15)  # a blade from the body's right side to 2 px past the safe area
    gates = evaluate({"f": frame}, kind="sprite", rule=FREE16, walk_cycle=False)
    measure = gates["G8"]["frames"]["f"]
    assert gates["ok"] and gates["warnings"] == []
    assert measure["bbox"] == [8, 3, 30, 28] and measure["body_box"] == [8, 3, 23, 28]


def test_g8_warns_when_the_body_leaves_the_safe_area_by_more_than_a_pixel():
    one_over = paint(canvas(), 2, 3, 20, 28)
    assert evaluate({"f": one_over}, kind="sprite", rule=FREE16, walk_cycle=False)["warnings"] == []
    two_over = paint(canvas(), 1, 2, 20, 28)
    gates = evaluate({"f": two_over}, kind="sprite", rule=FREE16, walk_cycle=False)
    assert gates["ok"] and gates["G8"]["frames"]["f"]["overflow"] == {"left": 2, "top": 1, "right": 0, "bottom": 0}
    assert gates["warnings"] == ["G8 the body reaches past the 26x26 safe area by more than 1 px "
                                 "(f: 2 px left) — only a thin detail may go into the 3 px margin"]


def test_g8_warns_a_sprite_off_the_floor_row():
    floating = paint(canvas(), 8, 3, 23, 26)
    gates = evaluate({"a": floating, "b": paint(canvas(), 9, 3, 23, 26)}, kind="sprite", rule=FREE16, walk_cycle=False)
    assert gates["ok"] and gates["G6"]["ok"]
    assert gates["warnings"] == ["G8 the feet are not on the floor row (the safe area's bottom row): "
                                 "a 2 px above it, b 2 px above it"]
    below = paint(canvas(), 8, 3, 23, 29)
    assert evaluate({"f": below}, kind="sprite", rule=FREE16, walk_cycle=False)["warnings"] == [
        "G8 the feet are not on the floor row (the safe area's bottom row): f 1 px below it"]
    icon = evaluate({"f": floating}, kind="icon", rule=FREE16, walk_cycle=False)
    assert icon["warnings"] == [] and icon["G8"]["floor_row"] is None


def test_g8_has_no_edge_to_keep_below_a_one_pixel_margin():
    plus = paint(paint(canvas(8), 3, 0, 4, 7), 0, 3, 7, 4)  # touches every side, corners clear
    gates = evaluate({"f": plus}, kind="icon", rule=FREE16, walk_cycle=False)
    assert gates["ok"] and gates["G8"]["margin"] == [0, 0] and gates["G8"]["edge_pixels"] == 0


def test_g8_leaves_tiles_alone():
    gates = evaluate({"tile": [[GRASS] * 32 for _ in range(32)]}, kind="tile", rule=FREE16, walk_cycle=False)
    assert gates["ok"] and "G8" not in gates


def test_a_recorded_blade_reaches_into_the_margin_and_the_body_stays_inside():
    frames = frames_of("attack-32-safe-area-pass.txt", 32, ("attack1", "attack2", "attack3", "attack4"))
    gates = evaluate(frames, kind="sprite", rule=FREE16, walk_cycle=False)
    measures = gates["G8"]["frames"]
    assert gates["ok"] and gates["warnings"] == []
    assert measures["attack1"]["bbox"][0] == 2 and measures["attack3"]["bbox"][2] == 29  # the blade, 1 px past it
    assert all(3 <= m["body_box"][0] and m["body_box"][2] <= 28 and m["baseline"] == 28 for m in measures.values())


def test_g6_measures_the_frames_inside_the_safe_area():
    """What reaches into the margin - a staff raised in one frame, a spark in another - is G8's to
    judge; G6 compares the figures inside the safe area."""
    still, raised = figure(), paint(figure(), 20, 1, 20, 2)  # the staff 2 px into the top margin
    g6 = evaluate({"a": still, "b": raised}, kind="sprite", rule=FREE16, walk_cycle=False)["G6"]
    assert g6["ok"] and g6["heights"] == {"a": 26, "b": 26} and g6["baselines"] == {"a": 28, "b": 28}
    taller = paint(canvas(), 8, 5, 23, 28)  # the body itself 2 px shorter inside the safe area
    assert not evaluate({"a": still, "b": taller}, kind="sprite", rule=FREE16, walk_cycle=False)["G6"]["ok"]


def test_a_recorded_staff_and_spark_in_the_margin_pass():
    """Drawn under the safe-area prompt: the staff pulled back reaches 2 px into the top margin and the
    spark 1 px, the figure is the same height in every frame."""
    frames = frames_of("attack-32-margin-detail.txt", 32, ("attack1", "attack2", "attack3", "attack4"))
    gates = evaluate(frames, kind="sprite", rule=FREE16, walk_cycle=False)
    assert gates["ok"] and gates["warnings"] == [] and gates["G6"]["height_spread"] == 0
    assert [m["bbox"][1] for m in gates["G8"]["frames"].values()] == [3, 1, 2, 3]


def test_recorded_answers_from_the_full_canvas_prompt_fail_g8():
    """Drawn when the prompt asked to fill the canvas: the knight's helmet is on row 1 and its boots
    on row 16, the 16 px canvas's own edge rows."""
    gates = evaluate(frames_of("sprite-16-pass.txt", 16, ("front",)), kind="sprite", rule=FREE16, walk_cycle=False)
    assert gates["failed"] == ["G8"] and gates["G8"]["frames"]["front"]["edge_pixels"] == 14
