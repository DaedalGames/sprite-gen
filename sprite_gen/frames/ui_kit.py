# SPDX-License-Identifier: Apache-2.0
"""Split a transparent UI kit sheet drawn on the fixed layout into nine-slice elements.

docs/ui-kit.md is the contract. One table (`LAYOUT`) is the layout: `gen --layout-guide ui-kit`
draws it (`draw_guide`) and says it (`guide_text`), and the splitter cuts along it. The sheet's
alpha haze is cleared first (`alpha_floor.clear_haze`), parts are assigned to boxes by centroid,
and each element's insets and content padding are measured on its own pixels.
"""

from __future__ import annotations

import argparse
import io
import json
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
PADDING_TOLERANCE = 32
GUIDE_BACKGROUND = (0xF6, 0xF6, 0xF6)
GUIDE_BOX = (0x2F, 0x80, 0xED)
STRETCHES = ("xy", "x", "none")


@dataclass(frozen=True)
class Slot:
    name: str
    box: tuple[int, int, int, int]
    stretch: str
    where: str
    what: str


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
)


def boxes(size: tuple[int, int]) -> dict[str, tuple[int, int, int, int]]:
    """The layout's boxes as (x, y, w, h) in pixels of a sheet of ``size``, in layout order."""
    width, height = size
    sx, sy = width / REFERENCE, height / REFERENCE
    return {slot.name: (round(slot.box[0] * sx), round(slot.box[1] * sy),
                        round(slot.box[2] * sx), round(slot.box[3] * sy)) for slot in LAYOUT}


def _rect(x: int, y: int, w: int, h: int) -> dict[str, int]:
    return {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}


def guide_text() -> str:
    """The prompt's half of the guide: what to draw in each box and how, in layout order."""
    places = "; ".join(f"in {slot.where}, {slot.what}" for slot in LAYOUT)
    return (
        "Layout: the last attached image is a layout guide, not part of the artwork. It shows "
        f"{len(LAYOUT)} blue boxes on a grey sheet. Draw exactly one user interface element inside each box, "
        "filling most of its box without touching the box's edges, and leave everything outside the boxes "
        f"empty and transparent: {places}. Every frame, button and bar has straight, even edges between its "
        "corners, so it can be stretched along them: put ornaments only in the corners, and on the bars only "
        "at their two ends. Leave every centre empty: no text, no letters, no numbers, no icons and no fill "
        "inside the bars. Do not reproduce the layout guide itself: no boxes, guide lines, guide colors or the "
        "guide's grey background may appear in the output."
    )


def draw_guide(path: Path, size: tuple[int, int] = (REFERENCE, REFERENCE)) -> dict[str, Any]:
    """Draw the layout guide at ``path`` and answer its cell: the kind, the size and every box."""
    width, height = size
    image = Image.new("RGB", (width, height), GUIDE_BACKGROUND)
    draw = ImageDraw.Draw(image)
    line = max(2, round(3 * width / REFERENCE))
    scaled = boxes(size)
    for x, y, w, h in scaled.values():
        draw.rectangle((x, y, x + w - 1, y + h - 1), outline=GUIDE_BOX, width=line)
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
    close = np.abs(line - centre).max(axis=-1) <= PADDING_TOLERANCE
    if not close[middle]:
        return middle, middle + 1
    start, end = middle, middle + 1
    while start > 0 and close[start - 1]:
        start -= 1
    while end < len(close) and close[end]:
        end += 1
    return start, end


def measure_padding(image: Image.Image) -> tuple[dict[str, int], list[str]]:
    """Distances from each edge to the element's empty interior (docs/ui-kit.md "Padding"), and review notes."""
    pixels = _premultiplied(image)
    height, width, _ = pixels.shape
    cy, cx = height // 2, width // 2
    rows, cols = slice(max(0, cy - 2), cy + 3), slice(max(0, cx - 2), cx + 3)
    centre = np.median(pixels[rows, cols].reshape(-1, 4), axis=0)
    x0, x1 = _interior_run(np.median(pixels[rows], axis=0), centre, cx)
    y0, y1 = _interior_run(np.median(pixels[:, cols], axis=1), centre, cy)

    def interior(x: int, y: int) -> bool:
        return bool(np.abs(pixels[y, x] - centre).max() <= PADDING_TOLERANCE)

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
    """Cut ``image`` along the layout: element images by name, and the report (without the sheet path)."""
    rgba = image.convert("RGBA")
    width, height = rgba.size
    if abs(width - height) > 0.01 * max(width, height):
        raise ValueError(f"a UI kit sheet must be square (1:1); this one is {width}x{height}")
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
    for slot in LAYOUT:
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
        padding, padding_notes = measure_padding(element)
        reasons += [f"{slot.name}: {note}" for note in inset_notes + padding_notes]
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
    report = {
        "kind": "sprite-gen-ui-kit",
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


def _publish(*, sheet: Path, out_dir: Path, alpha_floor: int = 64, alpha_edge: int = 2,
             report: Path | None = None) -> int:
    sheet = Path(sheet).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    report_path = Path(report).expanduser().resolve() if report is not None else out_dir / "ui-kit.json"
    if not sheet.is_file():
        raise ValueError(f"sheet is not a file: {sheet}")
    if out_dir == sheet.parent or out_dir in sheet.parents or report_path == sheet:
        raise ValueError("the output directory must not hold the sheet; choose a new directory")
    with Image.open(sheet) as image:
        if image.format != "PNG" or getattr(image, "n_frames", 1) != 1:
            raise ValueError("--sheet must be a single-frame PNG")
        elements, measured = split_ui_kit(image, alpha_floor=alpha_floor, alpha_edge=alpha_edge)
    written = {"kind": measured["kind"], "version": measured["version"], "sheet": str(sheet)}
    written.update((key, value) for key, value in measured.items() if key not in written)
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
    parser.add_argument("--report", type=Path, help="JSON report (default: <out-dir>/ui-kit.json)")


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
