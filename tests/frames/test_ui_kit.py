# SPDX-License-Identifier: Apache-2.0
"""`ui-kit`: a transparent sheet drawn on the fixed layout becomes nine-slice elements.

The sheet is synthetic and drawn into the layout's own boxes: frames with known corner
ornaments (one with a long thin flourish along its top edge), a bar frame with an end gem
taller than the bar, a round badge, light colour noise, a faint haze ring around every
element and loose specks. Insets are checked to the pixel: every ornament column and row
lies inside a fixed slice, and nothing more than the 2 px margin does.

A tall (2:3) sheet holds the same seven frames at the same places and, in the band below them,
two bar fills with end caps of their own colour and a top-to-bottom shade, a slot and a counter
plate with corner ornaments, two pips and a joystick base and knob drawn as discs. A fill fits
its bar frame's content area exactly, in the report and in the pack; round parts never stretch.
A square sheet's report, element pixels and packs are frozen from the splitter before the tall
sheet existed.
"""

import hashlib
import json
import math

import numpy as np
import pytest
from PIL import Image

from sprite_gen import cli
from sprite_gen.compose import game_pack
from sprite_gen.frames import ui_kit
from sprite_gen.util.resample import resize_cell

SIZE = 512
TALL = (SIZE, SIZE * 3 // 2)
BORDER = (200, 200, 220, 255)
ORNAMENT = (120, 220, 200, 255)
FLOURISH = (255, 180, 60, 255)
INTERIOR = (20, 30, 60, 210)
TRACK = (10, 10, 20, 180)
GEM = (220, 40, 80, 255)
FILL = (60, 200, 120, 255)
FILL_CAP = (200, 255, 220, 255)
PIP_ON = (255, 220, 90, 255)
PIP_OFF = (70, 60, 50, 255)
KNOB = (180, 190, 210, 255)


def fill(pixels, x0, y0, x1, y1, color):
    pixels[y0:y1, x0:x1] = color


def draw_frame(pixels, x, y, w, h, k, flourish=0):
    fill(pixels, x, y, x + w, y + h, BORDER)
    fill(pixels, x + 3, y + 3, x + w - 3, y + h - 3, INTERIOR)
    for cx, cy in ((x, y), (x + w - k, y), (x, y + h - k), (x + w - k, y + h - k)):
        fill(pixels, cx, cy, cx + k, cy + k, ORNAMENT)
    if flourish:
        fill(pixels, x + k, y + 4, x + k + flourish, y + 6, FLOURISH)


def draw_bar(pixels, x, y, w, h, radius, cap):
    fill(pixels, x + radius, y + 6, x + w, y + h - 6, BORDER)
    fill(pixels, x + radius, y + 9, x + w - 3, y + h - 9, TRACK)
    fill(pixels, x + w - cap, y + 2, x + w, y + h - 2, ORNAMENT)
    yy, xx = np.indices((h, 2 * radius))
    disc = np.hypot(xx - radius + 0.5, yy - h / 2 + 0.5) <= radius
    region = pixels[y:y + h, x:x + 2 * radius]
    region[disc] = GEM


def draw_badge(pixels, x, y, d):
    yy, xx = np.indices((d, d))
    r = np.hypot(xx - d / 2 + 0.5, yy - d / 2 + 0.5)
    region = pixels[y:y + d, x:x + d]
    region[r <= d / 2] = BORDER
    region[r <= d / 2 - 3] = INTERIOR


def draw_fill(pixels, x, y, w, h, cap):
    """A bar fill: one colour along its length, shaded from top to bottom, with an end cap of its own colour."""
    shade = np.linspace(1.0, 0.6, h)[:, None]
    region = pixels[y:y + h, x:x + w]
    region[..., 3] = 255
    for channel in range(3):
        region[..., channel] = (FILL[channel] * shade).round().astype(np.uint8)
    fill(pixels, x, y, x + cap, y + h, FILL_CAP)
    fill(pixels, x + w - cap, y, x + w, y + h, FILL_CAP)


def draw_disc(pixels, x, y, d, color, ring=0):
    yy, xx = np.indices((d, d))
    r = np.hypot(xx - d / 2 + 0.5, yy - d / 2 + 0.5)
    region = pixels[y:y + d, x:x + d]
    region[r <= d / 2] = BORDER if ring else color
    region[r <= d / 2 - ring] = color


# element -> (offset inside its box, drawing parameters); every element sits 4 px inside its box
PLAN = {"panel": dict(k=18, flourish=60), "button": dict(k=12), "button_pressed": dict(k=12),
        "card": dict(k=16), "hp_bar_frame": dict(radius=20, cap=10), "xp_bar_frame": dict(radius=20, cap=10),
        "timer_badge": dict(), "hp_bar_fill": dict(cap=6), "xp_bar_fill": dict(cap=6),
        "slot": dict(k=14), "badge": dict(k=10), "pip_on": dict(color=PIP_ON, ring=3),
        "pip_off": dict(color=PIP_OFF, ring=3), "joystick_base": dict(color=INTERIOR, ring=6),
        "joystick_knob": dict(color=KNOB)}


def synthetic_sheet(*, size=(SIZE, SIZE), haze=True, noise=True, stray=False, missing=None, crest=False):
    pixels = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    drawn = {}
    for name, (bx, by, bw, bh) in ui_kit.boxes(size).items():
        if name == missing:
            continue
        x, y, w, h = bx + 4, by + 4, bw - 8, bh - 8
        params = PLAN[name]
        if name.endswith("bar_frame"):
            draw_bar(pixels, x, y, w, h, **params)
        elif name.endswith("bar_fill"):
            draw_fill(pixels, x, y, w, h, **params)
        elif name == "timer_badge":
            w = h = min(w, h)
            draw_badge(pixels, x, y, w)
        elif name.startswith(("pip_", "joystick_")):
            w = h = min(w, h)
            draw_disc(pixels, x, y, w, **params)
        else:
            draw_frame(pixels, x, y, w, h, **params)
            if crest and name == "card":
                fill(pixels, x + w // 2 - 6, y, x + w // 2 + 6, y + 14, FLOURISH)
        drawn[name] = (x, y, w, h)
    solid = pixels[..., 3] > 0
    if noise:
        rng = np.random.default_rng(3)
        jitter = rng.integers(-4, 5, pixels[..., :3].shape)
        pixels[..., :3] = np.where(solid[..., None], np.clip(pixels[..., :3].astype(int) + jitter, 0, 255), 0)
    if haze:
        near = np.zeros_like(solid)
        for dy in range(-6, 7):
            for dx in range(-6, 7):
                near |= np.roll(np.roll(solid, dy, 0), dx, 1)
        ring = near & ~solid
        pixels[ring] = (150, 150, 170, 22)
        for sx, sy in ((5, 500), (500, 5), (330, 470)):
            pixels[sy, sx] = (200, 200, 200, 40)
    if stray:
        fill(pixels, 250, 410, 270, 430, BORDER)
    return Image.fromarray(pixels), drawn


def split(**kwargs):
    image, drawn = synthetic_sheet(**kwargs)
    elements, report = ui_kit.split_ui_kit(image)
    return image, drawn, elements, report


def offset(report, drawn, name):
    rect = report["elements"][name]["rect"]
    x, y, w, h = drawn[name]
    return x - rect["x"], y - rect["y"], rect["x"] + rect["w"] - (x + w), rect["y"] + rect["h"] - (y + h)


FRAMES = ["panel", "button", "button_pressed", "card", "hp_bar_frame", "xp_bar_frame", "timer_badge"]
BAND = ["hp_bar_fill", "xp_bar_fill", "slot", "badge", "pip_on", "pip_off", "joystick_base", "joystick_knob"]


def test_layout_is_the_documented_table_and_scales_with_the_sheet():
    assert [entry.name for entry in ui_kit.LAYOUT] == FRAMES + BAND
    full = ui_kit.boxes((1024, 1024))
    assert list(full) == FRAMES
    assert full["panel"] == (48, 48, 576, 400)
    assert full["card"] == (704, 400, 272, 576)
    assert full["timer_badge"] == (48, 816, 160, 160)
    assert ui_kit.boxes((512, 512))["card"] == (352, 200, 136, 288)
    assert {e.name: e.stretch for e in ui_kit.LAYOUT} == {
        "panel": "xy", "button": "xy", "button_pressed": "xy", "card": "xy",
        "hp_bar_frame": "x", "xp_bar_frame": "x", "timer_badge": "none",
        "hp_bar_fill": "x", "xp_bar_fill": "x", "slot": "xy", "badge": "xy",
        "pip_on": "none", "pip_off": "none", "joystick_base": "none", "joystick_knob": "none"}
    # the 2:3 sheet keeps the seven frames where the square sheet has them and adds the band below
    tall = ui_kit.boxes((1024, 1536))
    assert list(tall) == FRAMES + BAND
    assert {name: tall[name] for name in FRAMES} == full
    assert tall["hp_bar_fill"] == (48, 1024, 608, 64)
    assert tall["xp_bar_fill"] == (48, 1136, 608, 64)
    assert tall["slot"] == (48, 1248, 192, 192)
    assert tall["badge"] == (288, 1248, 176, 112)
    assert tall["pip_on"] == (512, 1248, 96, 96)
    assert tall["pip_off"] == (512, 1392, 96, 96)
    assert tall["joystick_base"] == (704, 1024, 272, 272)
    assert tall["joystick_knob"] == (768, 1344, 144, 144)
    assert ui_kit.boxes(TALL)["joystick_base"] == (352, 512, 136, 136)
    # boxes never touch: 48 px apart at 1024, and 48 px in from the sheet's edges
    for layout, (width, height) in ((full, (1024, 1024)), (tall, (1024, 1536))):
        names = list(layout)
        for i, a in enumerate(names):
            ax, ay, aw, ah = layout[a]
            assert ax >= 48 and ay >= 48 and ax + aw + 48 <= width and ay + ah + 48 <= height, a
            for b in names[i + 1:]:
                bx, by, bw, bh = layout[b]
                assert ax + aw + 48 <= bx or bx + bw + 48 <= ax or ay + ah + 48 <= by or by + bh + 48 <= ay


def test_frame_insets_cover_every_corner_ornament_and_the_long_flourish():
    _, drawn, elements, report = split()
    assert report["status"] == "pass", report["review_reasons"]
    for name in ("panel", "button", "button_pressed", "card"):
        entry = report["elements"][name]
        left, top, right, bottom = offset(report, drawn, name)
        k = PLAN[name]["k"]
        reach = k + PLAN[name].get("flourish", 0)
        assert entry["stretch"] == "xy"
        assert entry["insets"] == {"left": left + reach + 2, "right": right + k + 2,
                                   "top": top + k + 2, "bottom": bottom + k + 2}, name
        assert elements[name].size == (entry["size"]["w"], entry["size"]["h"])


def test_bar_insets_cover_the_gem_and_the_cap_and_padding_is_the_track():
    _, drawn, elements, report = split()
    for name in ("hp_bar_frame", "xp_bar_frame"):
        entry = report["elements"][name]
        left, top, right, bottom = offset(report, drawn, name)
        radius, cap = PLAN[name]["radius"], PLAN[name]["cap"]
        assert entry["stretch"] == "x"
        # the cap is taller than the bar, so its kept 2 px anti-aliased edge lies left of it, inside the
        # element: that edge belongs to the cap's fixed slice too
        assert entry["insets"] == {"left": left + 2 * radius + 2, "right": right + cap + 2 + 2, "top": 0, "bottom": 0}
        # the track: right of the gem, left of the cap, inside the border rows (9 px from the drawn top/bottom)
        assert entry["padding"] == {"left": left + 2 * radius + 2, "right": right + cap + 2,
                                    "top": top + 9 + 2, "bottom": bottom + 9 + 2}


def test_badge_does_not_stretch_and_its_content_fits_inside_the_ring():
    _, drawn, elements, report = split()
    entry = report["elements"]["timer_badge"]
    assert entry["stretch"] == "none"
    assert entry["insets"] == {"left": 0, "right": 0, "top": 0, "bottom": 0}
    left, top, _, _ = offset(report, drawn, "timer_badge")
    d = drawn["timer_badge"][2]
    centre = (left + d / 2, top + d / 2)
    pad, size = entry["padding"], entry["size"]
    corners = [(pad["left"], pad["top"]), (size["w"] - pad["right"], pad["top"]),
               (pad["left"], size["h"] - pad["bottom"]), (size["w"] - pad["right"], size["h"] - pad["bottom"])]
    for cx, cy in corners:
        assert math.hypot(cx - centre[0], cy - centre[1]) <= d / 2 - 3
    inner = d / 2 - 3
    assert size["w"] - pad["left"] - pad["right"] >= 1.2 * inner


def test_frame_padding_keeps_text_off_the_border_and_the_ornaments():
    _, drawn, elements, report = split()
    for name in ("panel", "card", "button"):
        entry = report["elements"][name]
        left, top, right, bottom = offset(report, drawn, name)
        pad = entry["padding"]
        k = PLAN[name]["k"]
        for side, edge in (("left", left), ("right", right), ("top", top), ("bottom", bottom)):
            assert edge + k <= pad[side] <= edge + k + 4, (name, side, pad)


def test_haze_is_cleared_before_cutting_and_reported():
    _, drawn, elements, report = split()
    cleanup = report["alpha_cleanup"]
    assert cleanup["floor"] == 64 and cleanup["edge_px"] == 2
    assert cleanup["haze_fraction_before"] > 0.01 and cleanup["haze_fraction_after"] == 0
    assert cleanup["changed_solid_pixels"] == 0 and cleanup["changed_edge_pixels"] == 0
    assert report["dropped_parts"] == 0
    assert report["unassigned"] == []
    # the element keeps at most its two-pixel anti-aliased edge of haze, never the six-pixel ring
    for name in drawn:
        assert all(0 <= value <= 2 for value in offset(report, drawn, name)), name


def test_ornament_reaching_the_middle_of_an_edge_is_flagged():
    _, _, _, report = split(crest=True)
    assert report["status"] == "needs-review"
    assert any(reason.startswith("card:") and "x" in reason for reason in report["review_reasons"])


def test_a_large_part_outside_every_box_is_reported():
    _, _, _, report = split(stray=True)
    assert report["status"] == "needs-review"
    assert report["unassigned"] == [{"rect": {"x": 250, "y": 410, "w": 20, "h": 20}, "pixels": 400}]


def test_an_empty_box_fails():
    image, _ = synthetic_sheet(missing="button_pressed")
    with pytest.raises(ValueError, match="button_pressed"):
        ui_kit.split_ui_kit(image)


def test_a_sheet_that_is_neither_square_nor_2_3_fails():
    for size in ((512, 400), (512, 900), (768, 512)):
        with pytest.raises(ValueError, match=r"square \(1:1.*2:3"):
            ui_kit.split_ui_kit(Image.new("RGBA", size))


def test_cli_writes_element_pngs_and_a_report_the_pack_reads(tmp_path):
    image, drawn = synthetic_sheet()
    sheet = tmp_path / "uikit.png"
    image.save(sheet)
    before = sheet.read_bytes()
    out = tmp_path / "ui"
    assert ui_kit.main(["--sheet", str(sheet), "--out-dir", str(out)]) == 0
    assert sheet.read_bytes() == before
    report = json.loads((out / "ui-kit.json").read_text())
    assert report["kind"] == "sprite-gen-ui-kit" and report["version"] == 1
    assert report["sheet"] == str(sheet.resolve()) and report["sheet_size"] == [SIZE, SIZE]
    assert list(report["elements"]) == FRAMES
    assert sorted(p.name for p in out.iterdir()) == sorted([f"{name}.png" for name in FRAMES] + ["ui-kit.json"])
    cleaned, _ = ui_kit.clear_haze(image.convert("RGBA"), floor=64, edge=2)
    for name, entry in report["elements"].items():
        assert entry["file"] == f"{name}.png"
        assert entry["box"] == dict(zip("xywh", ui_kit.boxes((SIZE, SIZE))[name]))
        rect = entry["rect"]
        with Image.open(out / entry["file"]) as element:
            assert element.size == (entry["size"]["w"], entry["size"]["h"]) == (rect["w"], rect["h"])
            crop = cleaned.crop((rect["x"], rect["y"], rect["x"] + rect["w"], rect["y"] + rect["h"]))
            assert element.convert("RGBA").tobytes() == crop.tobytes()
    request = tmp_path / "game.json"
    request.write_text(json.dumps({"version": 2, "ui": {"kit": "ui/ui-kit.json"}}))
    result = game_pack.pack(request, tmp_path / "pack")
    for name, entry in report["elements"].items():
        packed = result["ui"]["nine_slice"][name]
        assert packed["insets"] == entry["insets"] and packed["padding"] == entry["padding"]
        assert packed["size"] == entry["size"] and packed["stretch"] == entry["stretch"]


def test_cli_refuses_to_write_into_the_sheets_directory_over_the_sheet(tmp_path):
    image, _ = synthetic_sheet()
    sheet = tmp_path / "panel.png"
    image.save(sheet)
    with pytest.raises(SystemExit, match="sheet"):
        ui_kit.main(["--sheet", str(sheet), "--out-dir", str(tmp_path)])


# --out-dir, --report (None: the default <out-dir>/ui-kit.json) and the `file` the report gives the
# panel: the PNG's path from the report's own directory, with "/" separators
REPORT_PLACES = [
    ("ui", None, "panel.png"),
    ("elements", "reports/kit.json", "../elements/panel.png"),
    ("ui", "ui/reports/kit.json", "../panel.png"),
    ("reports/elements", "reports/kit.json", "elements/panel.png"),
]


def test_a_report_anywhere_names_the_written_pngs_and_packs_the_same(tmp_path):
    image, _ = synthetic_sheet()
    sheet = tmp_path / "uikit.png"
    image.save(sheet)
    packs = []
    for index, (out_dir, report, panel) in enumerate(REPORT_PLACES):
        root = tmp_path / f"place-{index}"
        args = ["ui-kit", "--sheet", str(sheet), "--out-dir", str(root / out_dir)]
        if report is not None:
            args += ["--report", str(root / report)]
        assert cli.main(args) == 0
        report_path = root / (report or f"{out_dir}/ui-kit.json")
        written = json.loads(report_path.read_text())
        assert written["elements"]["panel"]["file"] == panel
        for name, entry in written["elements"].items():
            assert (report_path.parent / entry["file"]).resolve() == (root / out_dir / f"{name}.png").resolve()
        request = root / "game.json"
        request.write_text(json.dumps({"version": 2, "ui": {"kit": report_path.relative_to(root).as_posix()}}))
        assert cli.main(["game-pack", "--request", str(request), "--out-dir", str(root / "pack")]) == 0
        packs.append({path.name: path.read_bytes() for path in (root / "pack").iterdir()})
    assert len(json.loads(packs[0]["game-pack.json"])["ui"]["nine_slice"]) == len(FRAMES)
    assert packs == [packs[0]] * len(REPORT_PLACES)


def test_a_report_path_that_is_not_json_is_refused_before_anything_is_written(tmp_path):
    image, _ = synthetic_sheet()
    sheet = tmp_path / "uikit.png"
    image.save(sheet)
    out = tmp_path / "ui"
    # the report would replace an element PNG of the same name
    with pytest.raises(SystemExit, match=r"--report must be a \.json path"):
        ui_kit.main(["--sheet", str(sheet), "--out-dir", str(out), "--report", str(out / "panel.png")])
    assert not out.exists()


def test_guide_draws_every_box_and_the_text_names_every_element(tmp_path):
    path = tmp_path / "guide.png"
    cell = ui_kit.draw_guide(path, (1024, 1024))
    with Image.open(path) as guide:
        guide = guide.convert("RGB")
        assert guide.size == (1024, 1024)
        for x, y, w, h in ui_kit.boxes((1024, 1024)).values():
            assert guide.getpixel((x + w // 2, y)) == ui_kit.GUIDE_BOX
            assert guide.getpixel((x, y + h // 2)) == ui_kit.GUIDE_BOX
            assert guide.getpixel((x + w // 2, y + h // 2)) == ui_kit.GUIDE_BACKGROUND
    assert cell == {"kind": "ui-kit", "width": 1024, "height": 1024,
                    "boxes": {name: dict(zip("xywh", box)) for name, box in ui_kit.boxes((1024, 1024)).items()}}
    text = ui_kit.guide_text()
    assert ui_kit.guide_text((1024, 1024)) == text
    positions = [text.index(entry.what) for entry in ui_kit.LAYOUT[:len(FRAMES)]]
    assert positions == sorted(positions)
    assert "last attached image is a layout guide" in text
    assert "no text" in text
    assert not any(entry.what in text for entry in ui_kit.LAYOUT[len(FRAMES):])


def test_tall_guide_draws_the_square_boxes_and_the_band_and_names_every_element(tmp_path):
    path = tmp_path / "guide.png"
    cell = ui_kit.draw_guide(path, (683, 1024))
    assert {slot.name for slot in ui_kit.LAYOUT if slot.round} == set(ROUND)
    with Image.open(path) as guide:
        guide = guide.convert("RGB")
        assert guide.size == (683, 1024)
        for name, (x, y, w, h) in ui_kit.boxes((683, 1024)).items():
            assert on_guide_line(guide, (x + w // 2, y), (0, 1)) and on_guide_line(guide, (x, y + h // 2), (1, 0))
            assert guide.getpixel((x + w // 2, y + h // 2)) == ui_kit.GUIDE_BACKGROUND
            # a round part's place is a circle, so the corners of its box are the guide's grey
            assert (guide.getpixel((x + 2, y + 2)) == ui_kit.GUIDE_BACKGROUND) == (name in ROUND), name
    assert cell == {"kind": "ui-kit", "width": 683, "height": 1024,
                    "boxes": {name: dict(zip("xywh", box)) for name, box in ui_kit.boxes((683, 1024)).items()}}
    assert list(cell["boxes"]) == FRAMES + BAND
    text = ui_kit.guide_text((683, 1024))
    assert text == ui_kit.guide_text((1024, 1536))
    positions = [text.index(entry.what) for entry in ui_kit.LAYOUT]
    assert positions == sorted(positions)
    assert "15 blue boxes and circles" in text and "last attached image is a layout guide" in text
    assert "no text" in text and "no plate, tile or backing" in text and "no frame, rim or end pieces" in text


ROUND = ("pip_on", "pip_off", "joystick_base", "joystick_knob")


def on_guide_line(guide, point, step):
    """The guide's outline passes ``point`` or one of the two pixels inward from it along ``step``."""
    return any(guide.getpixel((point[0] + i * step[0], point[1] + i * step[1])) == ui_kit.GUIDE_BOX for i in range(3))


def test_a_round_part_drawn_on_a_square_plate_is_flagged():
    image, drawn = synthetic_sheet(size=TALL, noise=False, haze=False)
    pixels = np.asarray(image).copy()
    # a square plate behind the lit dot: the box's corners around the disc are painted too
    x, y, w, h = drawn["pip_on"]
    plate = pixels[y:y + h, x:x + w]
    plate[plate[..., 3] == 0] = BORDER
    _, report = ui_kit.split_ui_kit(Image.fromarray(pixels))
    assert report["status"] == "needs-review"
    note = "its corners are filled: it was drawn on a square plate or backing, not as a circle; regenerate it"
    assert [reason for reason in report["review_reasons"] if reason.startswith(ROUND)] == [f"pip_on: {note}"]
    plated = Image.fromarray(np.ascontiguousarray(pixels[y:y + h, x:x + w]))
    assert ui_kit.measure_round(plated) == [note]
    assert ui_kit.measure_round(Image.fromarray(np.ascontiguousarray(np.asarray(image)[y:y + h, x:x + w]))) == []


def tall_split(**kwargs):
    return split(size=TALL, **kwargs)


def test_tall_sheet_cuts_every_element_of_the_whole_kit():
    _, drawn, elements, report = tall_split()
    assert report["status"] == "pass", report["review_reasons"]
    assert report["sheet_size"] == list(TALL)
    assert list(report["elements"]) == list(elements) == FRAMES + BAND
    assert report["unassigned"] == [] and report["dropped_parts"] == 0
    for name in FRAMES + BAND:
        entry = report["elements"][name]
        assert entry["file"] == f"{name}.png" and entry["parts"] == 1
        assert entry["box"] == dict(zip("xywh", ui_kit.boxes(TALL)[name]))
        assert elements[name].size == (entry["size"]["w"], entry["size"]["h"])
        # the cut is the drawn element and at most its kept two-pixel haze edge
        assert all(0 <= value <= 2 for value in offset(report, drawn, name)), name


def test_tall_sheet_cuts_the_seven_frames_as_the_square_sheet_does():
    _, _, square_elements, square = split(noise=False)
    _, _, tall_elements, tall = tall_split(noise=False)
    for name in FRAMES:
        assert tall["elements"][name] == square["elements"][name]
        assert tall_elements[name].tobytes() == square_elements[name].tobytes()


def test_fills_stretch_in_width_and_fit_the_content_area_of_their_bar_frame():
    _, drawn, elements, report = tall_split()
    for fill, frame in (("hp_bar_fill", "hp_bar_frame"), ("xp_bar_fill", "xp_bar_frame")):
        entry, bar = report["elements"][fill], report["elements"][frame]
        left, top, right, bottom = offset(report, drawn, fill)
        cap = PLAN[fill]["cap"]
        assert entry["stretch"] == "x"
        assert entry["insets"] == {"left": left + cap + 2, "right": right + cap + 2, "top": 0, "bottom": 0}
        # a fill holds no content: its padding is 0, as a frame's insets are 0 along an axis it keeps
        assert entry["padding"] == {"left": 0, "right": 0, "top": 0, "bottom": 0}
        pad, size = bar["padding"], bar["size"]
        rect = {"x": pad["left"], "y": pad["top"], "w": size["w"] - pad["left"] - pad["right"],
                "h": size["h"] - pad["top"] - pad["bottom"]}
        assert entry["inside"] == {"element": frame, "rect": rect, "scale": round(rect["h"] / entry["size"]["h"], 6)}
        # scaled by `scale`, the fill is as tall as the content area to the pixel and its two ends fit inside it
        assert abs(entry["size"]["h"] * entry["inside"]["scale"] - rect["h"]) < 0.01
        assert (entry["insets"]["left"] + entry["insets"]["right"]) * entry["inside"]["scale"] < rect["w"]
    for name in FRAMES + BAND:
        if name not in ("hp_bar_fill", "xp_bar_fill"):
            assert "inside" not in report["elements"][name], name


def test_pips_and_joystick_parts_never_stretch_and_keep_their_round_shape():
    _, drawn, elements, report = tall_split()
    for name in ("pip_on", "pip_off", "joystick_base", "joystick_knob"):
        entry = report["elements"][name]
        assert entry["stretch"] == "none"
        assert entry["insets"] == entry["padding"] == {"left": 0, "right": 0, "top": 0, "bottom": 0}
        d = drawn[name][2]
        assert d == drawn[name][3]
        assert d <= entry["size"]["w"] <= d + 4 and d <= entry["size"]["h"] <= d + 4
        assert abs(entry["size"]["w"] - entry["size"]["h"]) <= 2, name
    # the two pips are drawn in boxes of one size, so one dot swaps for the other in place
    assert ui_kit.boxes(TALL)["pip_on"][2:] == ui_kit.boxes(TALL)["pip_off"][2:]


def test_slot_and_badge_insets_cover_their_corner_ornaments_and_padding_keeps_content_off_them():
    _, drawn, elements, report = tall_split()
    for name in ("slot", "badge"):
        entry = report["elements"][name]
        left, top, right, bottom = offset(report, drawn, name)
        k = PLAN[name]["k"]
        assert entry["stretch"] == "xy"
        assert entry["insets"] == {"left": left + k + 2, "right": right + k + 2,
                                   "top": top + k + 2, "bottom": bottom + k + 2}, name
        for side, edge in (("left", left), ("right", right), ("top", top), ("bottom", bottom)):
            assert edge + k <= entry["padding"][side] <= edge + k + 4, (name, side, entry["padding"])


def test_an_empty_box_of_the_band_fails():
    image, _ = synthetic_sheet(size=TALL, missing="joystick_knob")
    with pytest.raises(ValueError, match="joystick_knob"):
        ui_kit.split_ui_kit(image)


def test_a_fill_whose_ends_do_not_fit_its_bar_frame_is_flagged():
    image, drawn = synthetic_sheet(size=TALL, noise=False, haze=False)
    pixels = np.asarray(image).copy()
    # end caps 112 px long on a 296 px fill, short of its middle fifth: scaled to the 26 px track's height
    # (the fill is 24 px tall), the two fixed ends are 247 px wide, and the track is 242 px long
    x, y, w, h = drawn["xp_bar_fill"]
    fill(pixels, x, y, x + 112, y + h, FILL_CAP)
    fill(pixels, x + w - 112, y, x + w, y + h, FILL_CAP)
    _, report = ui_kit.split_ui_kit(Image.fromarray(pixels))
    entry = report["elements"]["xp_bar_fill"]
    assert entry["insets"]["left"] == entry["insets"]["right"] == 114
    rect = entry["inside"]["rect"]
    assert (entry["insets"]["left"] + entry["insets"]["right"]) * entry["inside"]["scale"] >= rect["w"]
    assert report["status"] == "needs-review"
    assert [reason for reason in report["review_reasons"] if reason.startswith("xp_bar_fill:")] == [
        "xp_bar_fill: scaled to the height of xp_bar_frame's content area, its fixed ends are wider than that "
        "area; a full bar would overflow it"]


def pack_tall_kit(tmp_path, scale, **kwargs):
    image, drawn = synthetic_sheet(size=TALL, **kwargs)
    sheet = tmp_path / "uikit.png"
    image.save(sheet)
    assert ui_kit.main(["--sheet", str(sheet), "--out-dir", str(tmp_path / "ui")]) == 0
    request = tmp_path / "game.json"
    request.write_text(json.dumps({"version": 2, "ui": {"kit": "ui/ui-kit.json", "scale": scale}}))
    out = tmp_path / "pack"
    return game_pack.pack(request, out), out


def stored_frame(out, result, frame):
    for page in result["textures"]:
        if frame in page["frames"]:
            box = page["frames"][frame]["frame"]
            with Image.open(out / page["image"]) as atlas:
                return atlas.convert("RGBA").crop((box["x"], box["y"], box["x"] + box["w"], box["y"] + box["h"]))
    raise AssertionError(frame)


@pytest.mark.parametrize("scale", [1, 0.5, 0.37])
def test_a_packed_fill_is_stored_as_its_bar_frames_content_area(tmp_path, scale):
    result, out = pack_tall_kit(tmp_path, scale)
    nine = result["ui"]["nine_slice"]
    assert list(nine) == sorted(FRAMES + BAND)
    for fill_name, frame in (("hp_bar_fill", "hp_bar_frame"), ("xp_bar_fill", "xp_bar_frame")):
        entry, bar = nine[fill_name], nine[frame]
        assert set(entry) == {"frame", "size", "stretch", "insets", "padding", "inside"}
        assert all("inside" not in nine[name] for name in nine if not name.endswith("_fill"))
        pad = bar["padding"]
        rect = {"x": pad["left"], "y": pad["top"], "w": bar["size"]["w"] - pad["left"] - pad["right"],
                "h": bar["size"]["h"] - pad["top"] - pad["bottom"]}
        # the fill is stored at the frame's content area: at full width it covers it to the pixel, never past it
        assert entry["inside"] == {"element": frame, "rect": rect}
        assert entry["size"] == {"w": rect["w"], "h": rect["h"]}
        assert rect["x"] + rect["w"] + pad["right"] == bar["size"]["w"]
        assert rect["y"] + rect["h"] + pad["bottom"] == bar["size"]["h"]
        assert entry["stretch"] == "x" and entry["insets"]["top"] == entry["insets"]["bottom"] == 0
        assert entry["insets"]["left"] + entry["insets"]["right"] < rect["w"]
        assert entry["padding"] == {"left": 0, "right": 0, "top": 0, "bottom": 0}
        stored = stored_frame(out, result, entry["frame"])
        assert stored.size == (rect["w"], rect["h"])
        # its ends are the fill's own ends resized by one factor, never stretched: the factor makes it as tall
        # as the content area
        with Image.open(tmp_path / "ui" / f"{fill_name}.png") as source:
            source = source.convert("RGBA")
        factor = rect["h"] / source.height
        resized = resize_cell(source, (max(1, round(source.width * factor)), rect["h"]))
        left, right = entry["insets"]["left"], entry["insets"]["right"]
        cells = np.asarray(stored)
        ends = np.asarray(resized)
        assert np.array_equal(cells[:, :left], ends[:, :left])
        assert np.array_equal(cells[:, rect["w"] - right:], ends[:, resized.width - right:])


@pytest.mark.parametrize("scale", [1, 0.5, 0.37])
def test_packed_round_parts_keep_their_aspect(tmp_path, scale):
    result, out = pack_tall_kit(tmp_path, scale)
    report = json.loads((tmp_path / "ui" / "ui-kit.json").read_text())
    for name in ("pip_on", "pip_off", "joystick_base", "joystick_knob"):
        entry, source = result["ui"]["nine_slice"][name], report["elements"][name]["size"]
        assert entry["stretch"] == "none" and set(entry["insets"].values()) == {0}
        assert entry["size"] == {"w": max(1, round(source["w"] * scale)), "h": max(1, round(source["h"] * scale))}
        assert stored_frame(out, result, entry["frame"]).size == (entry["size"]["w"], entry["size"]["h"])


@pytest.mark.parametrize("size", [(SIZE, SIZE), TALL])
@pytest.mark.parametrize("scale", [0.75, 0.5, 0.37, 0.25])
def test_packed_kit_keeps_ornament_fringes_out_of_the_stretched_middle(tmp_path, scale, size):
    image, _ = synthetic_sheet(size=size, noise=False, haze=False)
    sheet = tmp_path / "uikit.png"
    image.save(sheet)
    assert ui_kit.main(["--sheet", str(sheet), "--out-dir", str(tmp_path / "ui")]) == 0
    request = tmp_path / "game.json"
    request.write_text(json.dumps({"version": 2, "ui": {"kit": "ui/ui-kit.json", "scale": scale}}))
    out = tmp_path / "pack"
    result = game_pack.pack(request, out)
    frames = {name: (page, record) for page in result["textures"] for name, record in page["frames"].items()}
    for name, entry in result["ui"]["nine_slice"].items():
        page, record = frames[entry["frame"]]
        box = record["frame"]
        with Image.open(out / page["image"]) as atlas:
            stored = np.asarray(atlas.convert("RGBA").crop(
                (box["x"], box["y"], box["x"] + box["w"], box["y"] + box["h"]))).astype(float)
        stored[..., :3] *= stored[..., 3:] / 255
        insets, (w, h) = entry["insets"], (entry["size"]["w"], entry["size"]["h"])
        if entry["stretch"] in ("xy", "x"):
            band = stored[:, insets["left"]:w - insets["right"]]
            # every stretched column looks like the band's own median column, within 8 of 255
            assert np.abs(band - np.median(band, axis=1)[:, None]).max() <= 8, (name, scale, "x")
        if entry["stretch"] == "xy":
            band = stored[insets["top"]:h - insets["bottom"]]
            assert np.abs(band - np.median(band, axis=0)[None]).max() <= 8, (name, scale, "y")


def test_padding_follows_a_smooth_gradient_face_to_its_border():
    # a button face that darkens smoothly from top to bottom (70 units over its height), inside a sharp
    # 4 px silver border: the content area is the whole face, not a band around its centre
    w, h = 160, 64
    pixels = np.zeros((h, w, 4), dtype=np.uint8)
    pixels[:, :] = BORDER
    face = np.linspace(110, 40, h - 8).round().astype(np.uint8)
    pixels[4:h - 4, 4:w - 4, 0] = 20
    pixels[4:h - 4, 4:w - 4, 1] = face[:, None] // 2
    pixels[4:h - 4, 4:w - 4, 2] = face[:, None]
    pixels[4:h - 4, 4:w - 4, 3] = 255
    padding, notes = ui_kit.measure_padding(Image.fromarray(pixels))
    assert padding == {"left": 6, "right": 6, "top": 6, "bottom": 6}
    assert notes == []


# A square sheet, split and packed by `ui-kit` and `game-pack` before the 2:3 sheet existed (base bfcab36):
# the guide, the report (sheet path aside) and every element's pixels, and the pack's manifest bytes and page
# pixels at two scales. Encoded PNG bytes depend on the zlib build, so images are compared as pixels.
SQUARE_FROZEN = {
    "guide_text": "6dccc85d1b063e3d7714435ca629ad0ade3b42b720e01ae2b63bd74723383b43",
    "guide_image": "66fe2622be0544604774c577bf1eb0619abd48f430a0eafe983a4f9132acaf91",
    "report": "56d2b589e613f436fd18c295d74e6031505ffb5114397384929cca9f1e959888",
    "elements": {
        "button.png": "080fcf70864cb593dc41bbce0ca25f53aa98dc0046b376f2244970e130ff3d97",
        "button_pressed.png": "cd789c638d312ad99406db8886c892d3caeb9a2754b31a9a4eb2af7a03f735a2",
        "card.png": "79e53083c231e2cffabad9f96204d907c9fc7e9e0e32ba631aa19a6c14e34b67",
        "hp_bar_frame.png": "d79732398bf77c5cc378a050d4340e7824f8353539a03cc6d5116fb914267f51",
        "panel.png": "e1cb7ffd6c22205d8de77417ceb8d4193a76cb30e06ac2b7186cd61a2e45048d",
        "timer_badge.png": "18e772b599a5c07004eab4c06cf43650b8617690ee151e265d0de9b32d4e9588",
        "xp_bar_frame.png": "9209c935bc75c4dfb465c9060b7f3b8783186f2f69a0ebb85b7ecd161ef9bc00",
    },
    "pack": {
        0.5: ("1c1e4f1f8a844c8ab9eafd2ebbe0adea3a62b79d960d1518f2c55b6f62576a86",
              ["8c107995d96e71ba7693c30a763b8b47df5c497a5041506ec5152fb465e9e640"]),
        0.37: ("698822abe6d887c04cc54e3ed40526366da298f164bda81c77980574c2c94445",
               ["a819f45d7b812e5c420a3f9908dd7bb48e80ba8bac94354859ffd2af09bb40d2"]),
    },
}


def pixel_digest(path):
    with Image.open(path) as image:
        return hashlib.sha256(image.convert("RGBA").tobytes()).hexdigest()


def test_a_square_sheet_is_split_and_packed_as_before_the_tall_sheet(tmp_path):
    ui_kit.draw_guide(tmp_path / "guide.png", (1024, 1024))
    assert hashlib.sha256(ui_kit.guide_text().encode()).hexdigest() == SQUARE_FROZEN["guide_text"]
    assert pixel_digest(tmp_path / "guide.png") == SQUARE_FROZEN["guide_image"]
    image, _ = synthetic_sheet()
    sheet = tmp_path / "uikit.png"
    image.save(sheet)
    assert ui_kit.main(["--sheet", str(sheet), "--out-dir", str(tmp_path / "ui")]) == 0
    text = (tmp_path / "ui" / "ui-kit.json").read_text().replace(json.dumps(str(sheet.resolve())), '"<sheet>"')
    assert hashlib.sha256(text.encode()).hexdigest() == SQUARE_FROZEN["report"]
    assert {png.name: pixel_digest(png) for png in sorted((tmp_path / "ui").glob("*.png"))} == SQUARE_FROZEN["elements"]
    for scale, (manifest, pages) in SQUARE_FROZEN["pack"].items():
        request = tmp_path / f"game-{scale}.json"
        request.write_text(json.dumps({"version": 2, "ui": {"kit": "ui/ui-kit.json", "scale": scale}}))
        out = tmp_path / f"pack-{scale}"
        result = game_pack.pack(request, out)
        assert hashlib.sha256((out / "game-pack.json").read_bytes()).hexdigest() == manifest, scale
        assert [pixel_digest(out / page["image"]) for page in result["textures"]] == pages, scale
