# SPDX-License-Identifier: Apache-2.0
"""Gates G2..G6 of `pixel-draw` on recorded answers (tests/fixtures/pixel_draw) and on grids made
to break exactly one rule. G2, G3 and G6 fail a candidate; G4, G5 and the walk check only warn."""
from __future__ import annotations

from pathlib import Path

import pytest

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
    gates = evaluate(frames_of("sprite-16-pass.txt", 16, ("front",)), kind="sprite", rule=FREE16, walk_cycle=False)
    assert gates["ok"] and gates["failed"] == [] and gates["warnings"] == []
    assert gates["G2"]["colors_used"] <= 16
    assert gates["G3"]["frames"]["front"]["corners_transparent"]
    assert gates["G5"]["isolated_pixels"] == 0 and "G6" not in gates and "G4" not in gates


def test_recorded_tile_is_opaque_and_repeats():
    gates = evaluate(frames_of("tile-16-pass.txt", 16, ("tile",)), kind="tile", rule=FREE16, walk_cycle=False)
    assert gates["ok"] and gates["G3"]["frames"]["tile"]["opaque_fraction"] == 1.0
    assert gates["G4"]["seam_ratio"] <= 1.5 and not gates["G4"]["warn"]


def test_g2_counts_colours_against_the_limit_and_the_fixed_set():
    frames = frames_of("sprite-16-pass.txt", 16, ("front",))
    used = evaluate(frames, kind="sprite", rule=FREE16, walk_cycle=False)["G2"]["colors_used"]
    few = evaluate(frames, kind="sprite", rule=pg.palette_rule("free:4"), walk_cycle=False)
    assert not few["ok"] and few["failed"] == ["G2"] and few["G2"]["colors_used"] == used
    pico = evaluate(frames, kind="sprite", rule=pg.palette_rule("pico8"), walk_cycle=False)
    assert not pico["ok"] and pico["G2"]["off_palette"]
    exact = pg.palette_rule(",".join(sorted({"#{:02X}{:02X}{:02X}".format(*p[:3])
                                             for row in frames["front"] for p in row if p[3]})))
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
    gates = evaluate({"f": frame}, kind="sprite", rule=FREE16, walk_cycle=False)
    assert gates["G5"]["isolated_pixels"] == 1 and gates["ok"]
    assert gates["warnings"] == ["G5 1 stray pixel(s) with no opaque neighbour"]


def test_recorded_four_directions_stand_on_one_line():
    gates = evaluate(frames_of("fourdir-16-pass.txt", 16, FOURDIR), kind="sprite", rule=FREE16, walk_cycle=False)
    assert gates["ok"] and gates["G6"]["ok"] and gates["G6"]["identical_pairs"] == []
    assert len(set(gates["G6"]["baselines"].values())) == 1 and "walk" not in gates["G6"]


def test_recorded_walk_passes_with_the_weak_leg_warning():
    gates = evaluate(frames_of("walk-16-pass.txt", 16, WALK), kind="sprite", rule=FREE16, walk_cycle=True)
    g6 = gates["G6"]
    assert gates["ok"] and g6["ok"] and g6["height_spread"] == 1  # the passing pose is a pixel higher
    assert g6["walk"]["warn"] and g6["walk"]["ratio"] < 0.4
    assert any(w.startswith("G6 weak leg alternation") for w in gates["warnings"])


def test_g6_failures():
    low, high, tall = blob(), blob(top=1, bottom=4), blob(top=0, bottom=5)
    assert not g6_frames({"a": low, "b": [row[:] for row in low]}, walk_cycle=False)["ok"]  # the same frame twice
    shifted = g6_frames({"a": low, "b": high}, walk_cycle=False)
    assert not shifted["ok"] and shifted["baselines"] == {"a": 5, "b": 4}
    spread = g6_frames({"a": low, "b": [r[:] for r in tall]}, walk_cycle=False)
    assert spread["height_spread"] == 2 and not spread["ok"]
    gates = evaluate({"a": low, "b": [row[:] for row in low]}, kind="sprite", rule=FREE16, walk_cycle=False)
    assert gates["failed"] == ["G6"]
