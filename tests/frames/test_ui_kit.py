# SPDX-License-Identifier: Apache-2.0
"""`ui-kit`: a transparent sheet drawn on the fixed layout becomes nine-slice elements.

The sheet is synthetic and drawn into the layout's own boxes: frames with known corner
ornaments (one with a long thin flourish along its top edge), a bar frame with an end gem
taller than the bar, a round badge, light colour noise, a faint haze ring around every
element and loose specks. Insets are checked to the pixel: every ornament column and row
lies inside a fixed slice, and nothing more than the 2 px margin does.
"""

import json
import math

import numpy as np
import pytest
from PIL import Image

from sprite_gen.compose import game_pack
from sprite_gen.frames import ui_kit

SIZE = 512
BORDER = (200, 200, 220, 255)
ORNAMENT = (120, 220, 200, 255)
FLOURISH = (255, 180, 60, 255)
INTERIOR = (20, 30, 60, 210)
TRACK = (10, 10, 20, 180)
GEM = (220, 40, 80, 255)


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


# element -> (offset inside its box, drawing parameters); every element sits 4 px inside its box
PLAN = {"panel": dict(k=18, flourish=60), "button": dict(k=12), "button_pressed": dict(k=12),
        "card": dict(k=16), "hp_bar_frame": dict(radius=20, cap=10), "xp_bar_frame": dict(radius=20, cap=10),
        "timer_badge": dict()}


def synthetic_sheet(*, haze=True, noise=True, stray=False, missing=None, crest=False):
    pixels = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)
    drawn = {}
    for name, (bx, by, bw, bh) in ui_kit.boxes((SIZE, SIZE)).items():
        if name == missing:
            continue
        x, y, w, h = bx + 4, by + 4, bw - 8, bh - 8
        params = PLAN[name]
        if name.endswith("bar_frame"):
            draw_bar(pixels, x, y, w, h, **params)
        elif name == "timer_badge":
            w = h = min(w, h)
            draw_badge(pixels, x, y, w)
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


def test_layout_is_the_documented_table_and_scales_with_the_sheet():
    assert [entry.name for entry in ui_kit.LAYOUT] == [
        "panel", "button", "button_pressed", "card", "hp_bar_frame", "xp_bar_frame", "timer_badge"]
    full = ui_kit.boxes((1024, 1024))
    assert full["panel"] == (48, 48, 576, 400)
    assert full["card"] == (704, 400, 272, 576)
    assert full["timer_badge"] == (48, 816, 160, 160)
    assert ui_kit.boxes((512, 512))["card"] == (352, 200, 136, 288)
    assert {e.name: e.stretch for e in ui_kit.LAYOUT} == {
        "panel": "xy", "button": "xy", "button_pressed": "xy", "card": "xy",
        "hp_bar_frame": "x", "xp_bar_frame": "x", "timer_badge": "none"}
    # boxes never touch: 48 px apart at 1024
    names = list(full)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ax, ay, aw, ah = full[a]
            bx, by, bw, bh = full[b]
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


def test_a_sheet_that_is_not_square_fails():
    with pytest.raises(ValueError, match="square"):
        ui_kit.split_ui_kit(Image.new("RGBA", (512, 400)))


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
    assert list(report["elements"]) == [entry.name for entry in ui_kit.LAYOUT]
    assert sorted(p.name for p in out.iterdir()) == sorted([f"{e.name}.png" for e in ui_kit.LAYOUT] + ["ui-kit.json"])
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
    positions = [text.index(entry.what) for entry in ui_kit.LAYOUT]
    assert positions == sorted(positions)
    assert "last attached image is a layout guide" in text
    assert "no text" in text


@pytest.mark.parametrize("scale", [0.75, 0.5, 0.37, 0.25])
def test_packed_kit_keeps_ornament_fringes_out_of_the_stretched_middle(tmp_path, scale):
    image, _ = synthetic_sheet(noise=False, haze=False)
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
