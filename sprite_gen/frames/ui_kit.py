# SPDX-License-Identifier: Apache-2.0
"""Split a transparent UI kit sheet drawn on the fixed layout into nine-slice elements.

docs/ui-kit.md is the contract. One table (`LAYOUT`) is the layout: `gen --layout-guide ui-kit`
draws it (`draw_guide`) and says it (`guide_text`), and the splitter cuts along it. A sheet's shape
says which part of the table it holds (`SHEETS`): a square sheet the seven frames, a 2:3 sheet the
same frames at the same places and the fills, slot, badge, pips and joystick below them. The sheet's
alpha haze is cleared first (`alpha_floor.clear_haze`), parts are assigned to boxes by centroid,
and each element's insets and content padding are measured on its own pixels; a fill is fitted to its
bar frame's content area. The report names each element PNG by its path from the report's own
directory (`report_file`, `element_path`).
"""

from __future__ import annotations

import argparse
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from sprite_gen._deps import np
from sprite_gen.frames.alpha_floor import clear_haze
from sprite_gen.spec.runio import acquire_run_dir_lock, atomic_write_set, release_run_dir_lock

REFERENCE = 1024
GAP = 48
MIN_PART = 16
MARGIN = 2
# Padding: the interior runs on while each step to the next pixel stays small and the pixel stays near
# the centre, so a smooth gradient face is interior and a border's sharp edge ends it.
PADDING_STEP = 16
PADDING_SPAN = 96
# A round part whose cut is more than this opaque outside the ellipse its bounding box holds was drawn on a
# square plate, not as a circle (docs/ui-kit.md "Layout").
ROUND_CORNER_FILL = 0.25
GUIDE_BACKGROUND = (0xF6, 0xF6, 0xF6)
GUIDE_BOX = (0x2F, 0x80, 0xED)
STRETCHES = ("xy", "x", "none")
SIDES = ("left", "right", "top", "bottom")
REPORT_KIND = "sprite-gen-ui-kit"


@dataclass(frozen=True)
class Slot:
    name: str
    box: tuple[int, int, int, int]
    stretch: str
    where: str
    what: str
    # Padding is measured only for an element that holds content (text, digits, an icon, a bar's fill);
    # the others get 0 on every side, as insets are 0 along an axis the element keeps.
    content: bool = True
    # A fill: the bar frame whose content area it fills (docs/ui-kit.md "Fills").
    inside: str | None = None
    # A round part: the guide draws a circle in its box, and a cut with filled corners is flagged.
    round: bool = False


LAYOUT = (
    Slot("panel", (48, 48, 576, 400), "xy", "the large box at the top left",
         "a large ornate panel frame with an empty centre"),
    Slot("button", (672, 48, 304, 128), "xy", "the upper box at the top right",
         "a wide button in its normal state"),
    Slot("button_pressed", (672, 224, 304, 128), "xy", "the box below it",
         "the same button pressed in"),
    Slot("card", (704, 400, 272, 576), "xy", "the tall box at the right",
         "a tall card frame with an empty centre"),
    Slot("hp_bar_frame", (48, 496, 608, 112), "x", "the long box at the left below the panel",
         "an empty horizontal health bar frame"),
    Slot("xp_bar_frame", (48, 656, 608, 112), "x", "the long box below it",
         "an empty horizontal experience bar frame"),
    Slot("timer_badge", (48, 816, 160, 160), "none", "the small square box at the bottom left",
         "a small round badge with an empty centre"),
    # The band below the square, on the 2:3 sheet only.
    Slot("hp_bar_fill", (48, 1024, 608, 64), "x", "the long thin box at the top left of the band",
         "the health bar's fill alone, a bare strip of colour with no frame, rim or end pieces around it",
         content=False, inside="hp_bar_frame"),
    Slot("xp_bar_fill", (48, 1136, 608, 64), "x", "the long thin box below it",
         "the experience bar's fill alone, the same kind of bare strip in its own colour", content=False,
         inside="xp_bar_frame"),
    Slot("slot", (48, 1248, 192, 192), "xy", "the square box below them at the left",
         "a square item slot frame with an empty centre"),
    Slot("badge", (288, 1248, 176, 112), "xy", "the small wide box to its right",
         "a small counter plate with an empty centre"),
    Slot("pip_on", (512, 1248, 96, 96), "none", "the small circle to the right of that",
         "a small round indicator dot, lit", content=False, round=True),
    Slot("pip_off", (512, 1392, 96, 96), "none", "the small circle below it",
         "the same dot unlit", content=False, round=True),
    Slot("joystick_base", (704, 1024, 272, 272), "none", "the large circle at the right of the band",
         "a round touch joystick base", content=False, round=True),
    Slot("joystick_knob", (768, 1344, 144, 144), "none", "the smaller circle below it",
         "the joystick's round knob", content=False, round=True),
)
FRAME_COUNT = 7
# Sheet shape -> (the size the layout's boxes are given at, the slots it holds). The 2:3 sheet keeps the
# square's seven boxes where they are and adds the band below them, so a square sheet cuts as it always has.
SHEETS = {
    "square": ((REFERENCE, REFERENCE), LAYOUT[:FRAME_COUNT]),
    "tall": ((REFERENCE, REFERENCE * 3 // 2), LAYOUT),
}


def sheet_layout(size: tuple[int, int]) -> tuple[tuple[int, int], tuple[Slot, ...]]:
    """The reference size and the slots of a sheet of ``size``: square, or 2:3, each within 1 %."""
    width, height = size
    for (ref_w, ref_h), slots in SHEETS.values():
        if abs(width * ref_h - height * ref_w) <= 0.01 * max(width * ref_h, height * ref_w):
            return (ref_w, ref_h), slots
    raise ValueError(f"a UI kit sheet must be square (1:1, the frames) or 2:3 (the whole kit); "
                     f"this one is {width}x{height}")


def boxes(size: tuple[int, int]) -> dict[str, tuple[int, int, int, int]]:
    """The boxes a sheet of ``size`` holds, as (x, y, w, h) in its pixels, in layout order."""
    (ref_w, ref_h), slots = sheet_layout(size)
    width, height = size
    sx, sy = width / ref_w, height / ref_h
    return {slot.name: (round(slot.box[0] * sx), round(slot.box[1] * sy),
                        round(slot.box[2] * sx), round(slot.box[3] * sy)) for slot in slots}


def _rect(x: int, y: int, w: int, h: int) -> dict[str, int]:
    return {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}


def _places(slots: tuple[Slot, ...]) -> str:
    return "; ".join(f"in {slot.where}, {slot.what}" for slot in slots)


def guide_text(size: tuple[int, int] = (REFERENCE, REFERENCE)) -> str:
    """The prompt's half of the guide for a sheet of ``size``: what to draw in each box and how, in layout order."""
    _, slots = sheet_layout(size)
    if len(slots) == FRAME_COUNT:
        return (
            "Layout: the last attached image is a layout guide, not part of the artwork. It shows "
            f"{len(slots)} blue boxes on a grey sheet. Draw exactly one user interface element inside each box, "
            "filling most of its box without touching the box's edges, and leave everything outside the boxes "
            f"empty and transparent: {_places(slots)}. Every frame, button and bar has straight, even edges "
            "between its corners, so it can be stretched along them: put ornaments only in the corners, and on "
            "the bars only at their two ends. Leave every centre empty: no text, no letters, no numbers, no icons "
            "and no fill inside the bars. Do not reproduce the layout guide itself: no boxes, guide lines, guide "
            "colors or the guide's grey background may appear in the output."
        )
    frames, band = slots[:FRAME_COUNT], slots[FRAME_COUNT:]
    return (
        "Layout: the last attached image is a layout guide, not part of the artwork. It shows "
        f"{len(slots)} blue boxes and circles on a grey sheet: {len(frames)} in the square at the top and "
        f"{len(band)} in the band below it. Draw exactly one user interface element inside each box and circle, "
        "filling most of it without touching its edges, and leave everything outside them empty and transparent. "
        "In the square "
        f"at the top: {_places(frames)}. In the band below it: {_places(band)}. Every frame, button, bar, fill, "
        "slot and plate has straight, even edges between its corners, so it can be stretched along them: put "
        "ornaments only in the corners, and on the bars and the fills only at their two ends. Each fill is one "
        "solid strip with no frame of its own, the same colour all along its length (it may shade from its top "
        "edge to its bottom edge), with no marks, segments or highlights along it. Leave the centre of every "
        "frame, button, bar frame, slot and plate empty: no text, no letters, no numbers, no icons and no fill "
        "inside the bar frames. The two dots, the joystick base and the knob are circles drawn on the guide's "
        "circles and seen straight from the front, with nothing square around them: no plate, tile or backing, "
        "so the corners around each circle stay empty and transparent; the two dots are the same size. Do not "
        "reproduce the layout guide itself: no boxes, circles, guide lines, guide colors or the guide's grey "
        "background may appear in the output."
    )


def draw_guide(path: Path, size: tuple[int, int] = (REFERENCE, REFERENCE)) -> dict[str, Any]:
    """Draw the layout guide at ``path`` and answer its cell: the kind, the size and every box."""
    width, height = size
    image = Image.new("RGB", (width, height), GUIDE_BACKGROUND)
    draw = ImageDraw.Draw(image)
    line = max(2, round(3 * width / REFERENCE))
    scaled = boxes(size)
    shapes = {slot.name: draw.ellipse if slot.round else draw.rectangle for slot in LAYOUT}
    for name, (x, y, w, h) in scaled.items():
        shapes[name]((x, y, x + w - 1, y + h - 1), outline=GUIDE_BOX, width=line)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return {"kind": "ui-kit", "width": width, "height": height,
            "boxes": {name: _rect(*box) for name, box in scaled.items()}}


def _premultiplied(image: Image.Image) -> Any:
    pixels = np.asarray(image.convert("RGBA")).astype(np.float32)
    pixels[..., :3] *= pixels[..., 3:] / 255
    return pixels


def _stretch_band(pixels: Any) -> tuple[int, int, list[str]]:
    """Fixed slices before and after the stretch band along dimension 1 (docs/ui-kit.md "Insets")."""
    height, width, _ = pixels.shape
    lo = int(width * 0.4)
    hi = max(int(width * 0.6), lo + 1)
    reference = np.median(pixels[:, lo:hi], axis=1)
    error = np.abs(pixels - reference[:, None, :]).max(axis=-1)
    spread = error[:, lo:hi].max(axis=1)
    plain = ~(error > (2 * spread + 8)[:, None]).any(axis=0)
    start = lo
    while start > 0 and plain[start - 1]:
        start -= 1
    end = hi
    while end < width and plain[end]:
        end += 1
    notes = []
    if spread.max() > max(64.0, 4 * float(np.median(spread))):
        notes.append("the middle of the element is not uniform (an ornament in the middle of an edge?); "
                     "stretching would smear it")
    if (start == lo and lo > 0) or (end == hi and hi < width):
        notes.append("an ornament reaches the middle fifth of the element; the insets may not cover it")
    before = min(start + MARGIN, lo) if start > 0 else 0
    after = min(width - end + MARGIN, width - hi) if end < width else 0
    return before, after, notes


def measure_insets(image: Image.Image, stretch: str) -> tuple[dict[str, int], list[str]]:
    """Nine-slice insets in the element's pixels, measured along the axes ``stretch`` allows, and review notes."""
    if stretch not in STRETCHES:
        raise ValueError(f"stretch must be one of {', '.join(STRETCHES)}")
    pixels = _premultiplied(image)
    insets = {"left": 0, "right": 0, "top": 0, "bottom": 0}
    notes: list[str] = []
    if stretch in ("xy", "x"):
        insets["left"], insets["right"], found = _stretch_band(pixels)
        notes += [f"x: {note}" for note in found]
    if stretch == "xy":
        insets["top"], insets["bottom"], found = _stretch_band(pixels.transpose(1, 0, 2))
        notes += [f"y: {note}" for note in found]
    return insets, notes


def _interior_run(line: Any, centre: Any, middle: int) -> tuple[int, int]:
    near = np.abs(line - centre).max(axis=-1) <= PADDING_SPAN
    smooth = np.abs(np.diff(line, axis=0)).max(axis=-1) <= PADDING_STEP
    if not near[middle]:
        return middle, middle + 1
    start, end = middle, middle + 1
    while start > 0 and near[start - 1] and smooth[start - 1]:
        start -= 1
    while end < len(near) and near[end] and smooth[end - 1]:
        end += 1
    return start, end


def measure_padding(image: Image.Image) -> tuple[dict[str, int], list[str]]:
    """Distances from each edge to the element's empty interior (docs/ui-kit.md "Padding"), and review notes."""
    pixels = _premultiplied(image)
    height, width, _ = pixels.shape
    cy, cx = height // 2, width // 2
    rows, cols = slice(max(0, cy - 2), cy + 3), slice(max(0, cx - 2), cx + 3)
    centre = np.median(pixels[rows, cols].reshape(-1, 4), axis=0)
    across, down = np.median(pixels[rows], axis=0), np.median(pixels[:, cols], axis=1)
    x0, x1 = _interior_run(across, centre, cx)
    y0, y1 = _interior_run(down, centre, cy)
    interior_values = np.concatenate((across[x0:x1], down[y0:y1]))
    low, high = interior_values.min(axis=0) - PADDING_STEP, interior_values.max(axis=0) + PADDING_STEP

    def interior(x: int, y: int) -> bool:
        return bool(((pixels[y, x] >= low) & (pixels[y, x] <= high)).all())

    shrink = 0
    while True:
        left, top, right, bottom = x0 + shrink, y0 + shrink, x1 - 1 - shrink, y1 - 1 - shrink
        if left >= right or top >= bottom:
            left, right = min(left, cx), max(right, cx)
            top, bottom = min(top, cy), max(bottom, cy)
            break
        if all(interior(x, y) for x, y in ((left, top), (right, top), (left, bottom), (right, bottom))):
            break
        shrink += 1
    padding = {"left": min(left + MARGIN, cx), "right": min(width - 1 - right + MARGIN, width - 1 - cx),
               "top": min(top + MARGIN, cy), "bottom": min(height - 1 - bottom + MARGIN, height - 1 - cy)}
    notes = []
    if 4 * (width - padding["left"] - padding["right"]) < width or 4 * (height - padding["top"] - padding["bottom"]) < height:
        notes.append("the content area is less than a quarter of the element; its centre is probably not empty")
    return padding, notes


def measure_round(image: Image.Image) -> list[str]:
    """Review notes for a round part: its cut's corners, outside the ellipse its bounding box holds, are empty."""
    alpha = np.asarray(image.convert("RGBA"))[..., 3]
    height, width = alpha.shape
    yy, xx = np.indices((height, width))
    outside = ((xx + 0.5 - width / 2) / (width / 2)) ** 2 + ((yy + 0.5 - height / 2) / (height / 2)) ** 2 > 1
    if outside.any() and (alpha[outside] >= 128).mean() > ROUND_CORNER_FILL:
        return ["its corners are filled: it was drawn on a square plate or backing, not as a circle; regenerate it"]
    return []


def _runs(mask: Any) -> list[tuple[int, int, int]]:
    runs = []
    for y in np.flatnonzero(mask.any(axis=1)):
        edges = np.diff(np.concatenate(([0], mask[y].astype(np.int8), [0])))
        for x0, x1 in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            runs.append((int(y), int(x0), int(x1)))
    return runs


def _parts(mask: Any) -> list[dict[str, Any]]:
    """8-connected parts of ``mask``: their runs, pixel count, bounding box and centroid."""
    runs = _runs(mask)
    parent = list(range(len(runs)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    previous: list[int] = []
    current: list[int] = []
    row = None
    for index, (y, x0, x1) in enumerate(runs):
        if y != row:
            previous = current if row == y - 1 else []
            current, row = [], y
        for other in previous:
            _, ox0, ox1 = runs[other]
            if ox0 <= x1 and x0 <= ox1:  # touching rows; one column apart counts (diagonal)
                parent[find(other)] = find(index)
        current.append(index)
    groups: dict[int, list[int]] = {}
    for index in range(len(runs)):
        groups.setdefault(find(index), []).append(index)
    parts = []
    for members in groups.values():
        spans = [runs[i] for i in members]
        count = sum(x1 - x0 for _, x0, x1 in spans)
        sum_x = sum((x0 + x1 - 1) * (x1 - x0) / 2 for _, x0, x1 in spans)
        sum_y = sum(y * (x1 - x0) for y, x0, x1 in spans)
        x_min, x_max = min(x0 for _, x0, _ in spans), max(x1 for _, _, x1 in spans)
        y_min, y_max = min(y for y, _, _ in spans), max(y for y, _, _ in spans) + 1
        parts.append({"runs": spans, "pixels": count, "centroid": (sum_x / count, sum_y / count),
                      "bbox": (x_min, y_min, x_max, y_max)})
    parts.sort(key=lambda part: (part["bbox"][1], part["bbox"][0]))
    return parts


def split_ui_kit(image: Image.Image, *, alpha_floor: int = 64,
                 alpha_edge: int = 2) -> tuple[dict[str, Image.Image], dict[str, Any]]:
    """Cut ``image`` along the layout: element images by name, and the report as written to its default
    place, ``<out-dir>/ui-kit.json`` (each ``file`` is ``<element>.png``), without the sheet path."""
    rgba = image.convert("RGBA")
    width, height = rgba.size
    _, slots = sheet_layout((width, height))
    cleanup = None
    if alpha_floor:
        rgba, cleanup = clear_haze(rgba, floor=alpha_floor, edge=alpha_edge)
    pixels = np.asarray(rgba)
    scaled = boxes((width, height))
    reach = GAP / 2 * width / REFERENCE
    assigned: dict[str, list[dict[str, Any]]] = {name: [] for name in scaled}
    dropped, unassigned, reasons = 0, [], []
    for part in _parts(pixels[..., 3] > 0):
        if part["pixels"] < MIN_PART:
            dropped += 1
            continue
        cx, cy = part["centroid"]
        owner = next((name for name, (x, y, w, h) in scaled.items()
                      if x - reach <= cx < x + w + reach and y - reach <= cy < y + h + reach), None)
        if owner is None:
            x0, y0, x1, y1 = part["bbox"]
            unassigned.append({"rect": _rect(x0, y0, x1 - x0, y1 - y0), "pixels": part["pixels"]})
            reasons.append(f"a part of {part['pixels']} pixels at ({x0}, {y0}) lies in no box")
            continue
        assigned[owner].append(part)
    elements: dict[str, Image.Image] = {}
    records: dict[str, Any] = {}
    for slot in slots:
        parts = assigned[slot.name]
        if not parts:
            raise ValueError(f"{slot.name}: its box holds no element; generate the sheet again")
        x0 = min(part["bbox"][0] for part in parts)
        y0 = min(part["bbox"][1] for part in parts)
        x1 = max(part["bbox"][2] for part in parts)
        y1 = max(part["bbox"][3] for part in parts)
        cut = np.zeros((y1 - y0, x1 - x0, 4), dtype=np.uint8)
        for part in parts:
            for y, a, b in part["runs"]:
                cut[y - y0, a - x0:b - x0] = pixels[y, a:b]
        element = Image.fromarray(cut)
        insets, inset_notes = measure_insets(element, slot.stretch)
        padding, padding_notes = measure_padding(element) if slot.content else (dict.fromkeys(SIDES, 0), [])
        round_notes = measure_round(element) if slot.round else []
        reasons += [f"{slot.name}: {note}" for note in inset_notes + padding_notes + round_notes]
        elements[slot.name] = element
        records[slot.name] = {
            "file": f"{slot.name}.png",
            "box": _rect(*scaled[slot.name]),
            "rect": _rect(x0, y0, x1 - x0, y1 - y0),
            "size": {"w": x1 - x0, "h": y1 - y0},
            "stretch": slot.stretch,
            "insets": insets,
            "padding": padding,
            "parts": len(parts),
        }
    for slot in slots:
        if slot.inside is not None:
            records[slot.name]["inside"], note = fit_inside(records[slot.name], records[slot.inside], slot.inside)
            reasons += [f"{slot.name}: {note}"] if note else []
    report = {
        "kind": REPORT_KIND,
        "version": 1,
        "sheet_size": [width, height],
        "status": "needs-review" if reasons else "pass",
        "review_reasons": reasons,
        "alpha_cleanup": cleanup,
        "dropped_parts": dropped,
        "unassigned": unassigned,
        "elements": records,
    }
    return elements, report


def content_rect(record: dict[str, Any]) -> dict[str, int]:
    """The content rectangle of an element record (its size less its padding), from its top-left corner."""
    pad, size = record["padding"], record["size"]
    return _rect(pad["left"], pad["top"], size["w"] - pad["left"] - pad["right"],
                 size["h"] - pad["top"] - pad["bottom"])


def fit_inside(fill: dict[str, Any], frame: dict[str, Any], name: str) -> tuple[dict[str, Any], str | None]:
    """A fill's ``inside``: the frame it fills, that frame's content rectangle and the factor that makes the
    fill exactly as tall as it (docs/ui-kit.md "Fills"), and a review note when its fixed ends would not fit."""
    rect = content_rect(frame)
    scale = rect["h"] / fill["size"]["h"]
    note = None
    if (fill["insets"]["left"] + fill["insets"]["right"]) * scale >= rect["w"]:
        note = (f"scaled to the height of {name}'s content area, its fixed ends are wider than that area; "
                "a full bar would overflow it")
    return {"element": name, "rect": rect, "scale": round(scale, 6)}, note


def report_file(report: Path, png: Path) -> str:
    """The ``file`` a report at ``report`` gives ``png``: its path from the report's directory, ``/``-separated."""
    return Path(os.path.relpath(png, Path(report).parent)).as_posix()


def element_path(report: Path, file: str) -> Path:
    """The element PNG that the ``file`` of a report at ``report`` names."""
    return Path(report).parent / file


def _publish(*, sheet: Path, out_dir: Path, alpha_floor: int = 64, alpha_edge: int = 2,
             report: Path | None = None) -> int:
    sheet = Path(sheet).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    report_path = Path(report).expanduser().resolve() if report is not None else out_dir / "ui-kit.json"
    if not sheet.is_file():
        raise ValueError(f"sheet is not a file: {sheet}")
    if out_dir == sheet.parent or out_dir in sheet.parents or report_path == sheet:
        raise ValueError("the output directory must not hold the sheet; choose a new directory")
    if report_path.suffix.lower() != ".json":
        raise ValueError("--report must be a .json path")
    with Image.open(sheet) as image:
        if image.format != "PNG" or getattr(image, "n_frames", 1) != 1:
            raise ValueError("--sheet must be a single-frame PNG")
        elements, measured = split_ui_kit(image, alpha_floor=alpha_floor, alpha_edge=alpha_edge)
    written = {"kind": measured["kind"], "version": measured["version"], "sheet": str(sheet)}
    written.update((key, value) for key, value in measured.items() if key not in written)
    # wherever --report puts the report, each `file` leads from it to the PNG written in out_dir
    written["elements"] = {name: {**record, "file": report_file(report_path, out_dir / record["file"])}
                           for name, record in measured["elements"].items()}
    payloads: dict[Path, bytes | str] = {}
    for name, element in elements.items():
        png = io.BytesIO()
        element.save(png, format="PNG")
        payloads[out_dir / measured["elements"][name]["file"]] = png.getvalue()
    payloads[report_path] = json.dumps(written, indent=2, allow_nan=False) + "\n"
    locked = []
    try:
        for directory in sorted({out_dir, report_path.parent}):
            directory.mkdir(parents=True, exist_ok=True)
            acquire_run_dir_lock(directory, "ui-kit")
            locked.append(directory)
        atomic_write_set(payloads)
    finally:
        for directory in reversed(locked):
            release_run_dir_lock(directory)
    print(f"ui-kit: {written['status']} — {len(elements)} elements in {out_dir} (report: {report_path})")
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sheet", required=True, type=Path,
                        help="transparent square PNG drawn with `gen --layout-guide ui-kit`")
    parser.add_argument("--out-dir", required=True, type=Path,
                        help="directory for <element>.png and ui-kit.json; must not hold the sheet")
    parser.add_argument("--alpha-floor", type=int, default=64,
                        help="alpha haze cleanup floor, 1..255 (0 turns the cleanup off)")
    parser.add_argument("--alpha-edge", type=int, default=2,
                        help="faint pixels this close to a solid pixel are its edge and are kept")
    parser.add_argument("--report", type=Path,
                        help="JSON report (default: <out-dir>/ui-kit.json); its element paths are relative to it")


def run(**kwargs: Any) -> int:
    try:
        return _publish(**kwargs)
    except (OSError, ValueError, TypeError) as exc:
        raise SystemExit(f"ui-kit: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_arguments(parser)
    return run(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
