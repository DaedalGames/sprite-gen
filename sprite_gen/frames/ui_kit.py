# SPDX-License-Identifier: Apache-2.0
"""Split a transparent UI kit sheet drawn on the fixed layout into nine-slice elements.

docs/ui-kit.md is the contract. One table (`LAYOUT`) is the layout: `gen --layout-guide ui-kit`
draws it (`draw_guide`) and says it (`guide_text`), and the splitter cuts along it. A sheet's shape
says which part of the table it holds (`SHEETS`): a square sheet the seven frames, a 2:3 sheet the
same frames at the same places and the fills, slot, badge, pips and joystick below them. The sheet's
alpha haze is cleared first (`alpha_floor.clear_haze`), parts are assigned to boxes by centroid
(when a box is left empty, again to the layout fitted to the drawing, `_fit_layout`), and each
element's insets and content padding are measured on its own pixels; a fill is fitted to its bar
frame's content area. The frames are required; an optional element whose box is empty is left out
and named under `missing`. The report names each element PNG by its path from the report's own
directory (`report_file`, `element_path`). A kit is written whole into a new or empty output
directory (`_write_kit`): staged in a hidden directory beside it and renamed into place in one step.
That rename, or with the report outside the directory the link that names the report, publishes the
kit: a run that fails or is interrupted before it leaves nothing, one after it the whole kit, and a
killed run, or one interrupted again while it cleans up, never part of one. A directory that holds
anything is refused.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import secrets
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw

from sprite_gen._deps import np
from sprite_gen.frames.alpha_floor import clear_haze

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
# Fitting the layout to a drawing that drifted from it (docs/ui-kit.md "Split the sheet"): at most this many
# rounds of fit and assignment, from at least this many boxes that hold parts.
FIT_ROUNDS = 8
FIT_MIN_BOXES = 3
GUIDE_BACKGROUND = (0xF6, 0xF6, 0xF6)
GUIDE_BOX = (0x2F, 0x80, 0xED)
STRETCHES = ("xy", "x", "none")
SIDES = ("left", "right", "top", "bottom")
REPORT_KIND = "sprite-gen-ui-kit"
# A kit's staging directory takes the output directory's name with one rename. POSIX rename(2) takes the name of an
# absent or empty directory and fails on any other; Windows takes no existing directory's name, so an empty output
# directory is removed just before the rename there (Python docs, os.rename).
RENAME_ONTO_EMPTY_DIR = os.name != "nt"


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
    # An optional element: when its box is empty the kit is made without it and the report names it under
    # `missing`; an empty box of a required element refuses the sheet (docs/ui-kit.md "Missing elements").
    optional: bool = False


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
         content=False, inside="hp_bar_frame", optional=True),
    Slot("xp_bar_fill", (48, 1136, 608, 64), "x", "the long thin box below it",
         "the experience bar's fill alone, the same kind of bare strip in its own colour", content=False,
         inside="xp_bar_frame", optional=True),
    Slot("slot", (48, 1248, 192, 192), "xy", "the square box below them at the left",
         "a square item slot frame with an empty centre", optional=True),
    Slot("badge", (288, 1248, 176, 112), "xy", "the small wide box to its right",
         "a small counter plate with an empty centre", optional=True),
    Slot("pip_on", (512, 1248, 96, 96), "none", "the small circle to the right of that",
         "a small round indicator dot, lit", content=False, round=True, optional=True),
    Slot("pip_off", (512, 1392, 96, 96), "none", "the small circle below it",
         "the same dot unlit", content=False, round=True, optional=True),
    Slot("joystick_base", (704, 1024, 272, 272), "none", "the large circle at the right of the band",
         "a round touch joystick base", content=False, round=True, optional=True),
    Slot("joystick_knob", (768, 1344, 144, 144), "none", "the smaller circle below it",
         "the joystick's round knob", content=False, round=True, optional=True),
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


def _element_png(name: str) -> str:
    """The file name of an element's PNG in the output directory."""
    return f"{name}.png"


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


def _assign(parts: list[dict[str, Any]], scaled: dict[str, tuple[float, float, float, float]],
            reach: tuple[float, float]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Each part to the box, widened by ``reach`` along x and y, that holds its centroid (the first in layout
    order): the parts by box, and the parts in no box."""
    assigned: dict[str, list[dict[str, Any]]] = {name: [] for name in scaled}
    loose = []
    rx, ry = reach
    for part in parts:
        cx, cy = part["centroid"]
        owner = next((name for name, (x, y, w, h) in scaled.items()
                      if x - rx <= cx < x + w + rx and y - ry <= cy < y + h + ry), None)
        (assigned[owner] if owner is not None else loose).append(part)
    return assigned, loose


def _fit(assigned: dict[str, list[dict[str, Any]]],
         scaled: dict[str, tuple[float, float, float, float]]) -> tuple[tuple[float, float], ...] | None:
    """The scale and offset along x and y that move the layout's box centres closest (least squares) to the
    centres of what each box holds; None from fewer than FIT_MIN_BOXES boxes that hold parts."""
    centres, drawn = [], []
    for name, parts in assigned.items():
        if parts:
            x, y, w, h = scaled[name]
            x0, y0 = min(part["bbox"][0] for part in parts), min(part["bbox"][1] for part in parts)
            x1, y1 = max(part["bbox"][2] for part in parts), max(part["bbox"][3] for part in parts)
            centres.append((x + w / 2, y + h / 2))
            drawn.append(((x0 + x1) / 2, (y0 + y1) / 2))
    if len(centres) < FIT_MIN_BOXES:
        return None
    centres_array, drawn_array = np.asarray(centres), np.asarray(drawn)
    fit = []
    for axis in (0, 1):
        box, held = centres_array[:, axis], drawn_array[:, axis]
        if np.ptp(box) < 1:  # every box centred on one line along this axis: an offset alone
            fit.append((1.0, float(np.mean(held - box))))
        else:
            scale, offset = np.polyfit(box, held, 1)
            fit.append((float(scale), float(offset)))
    return tuple(fit)


def _fit_layout(parts: list[dict[str, Any]], scaled: dict[str, tuple[float, float, float, float]], reach: float,
                assigned: dict[str, list[dict[str, Any]]], required: set[str]) -> tuple[Any, ...] | None:
    """Fit the layout to a drawing that drifted from it (docs/ui-kit.md "Split the sheet"). Fit the boxes to what
    they hold, assign the parts to the fitted boxes, and again, until the assignment repeats. Returns the fit,
    the fitted boxes, and their assignment and loose parts; None unless the fitted boxes leave fewer ``required``
    boxes empty than the layout's own boxes did, or as many and fewer boxes in all."""

    def empty(held: dict[str, list[dict[str, Any]]]) -> tuple[int, int]:
        return sum(not held[name] for name in required), sum(not parts for parts in held.values())

    current, found = assigned, None
    for _ in range(FIT_ROUNDS):
        fit = _fit(current, scaled)
        if fit is None or min(fit[0][0], fit[1][0]) <= 0:
            break
        (sx, tx), (sy, ty) = fit
        moved = {name: (x * sx + tx, y * sy + ty, w * sx, h * sy) for name, (x, y, w, h) in scaled.items()}
        following, loose = _assign(parts, moved, (reach * sx, reach * sy))
        found = (fit, moved, following, loose)
        repeated = all(list(map(id, following[name])) == list(map(id, current[name])) for name in scaled)
        current = following
        if repeated:
            break
    if found is None or empty(found[2]) >= empty(assigned):
        return None
    return found


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
    scaled: dict[str, tuple[float, float, float, float]] = dict(boxes((width, height)))
    reach = GAP / 2 * width / REFERENCE
    every = _parts(pixels[..., 3] > 0)
    kept = [part for part in every if part["pixels"] >= MIN_PART]
    dropped = len(every) - len(kept)
    assigned, loose = _assign(kept, scaled, (reach, reach))
    required = {slot.name for slot in slots if not slot.optional}
    fitted = (_fit_layout(kept, scaled, reach, assigned, required) if any(not held for held in assigned.values())
              else None)
    if fitted is not None:
        fit, scaled, assigned, loose = fitted
    unassigned, reasons = [], []
    for part in loose:
        x0, y0, x1, y1 = part["bbox"]
        unassigned.append({"rect": _rect(x0, y0, x1 - x0, y1 - y0), "pixels": part["pixels"]})
        reasons.append(f"a part of {part['pixels']} pixels at ({x0}, {y0}) lies in no box")
    missing = [slot.name for slot in slots if not assigned[slot.name]]
    refused = [name for name in missing if name in required]
    if refused:
        holds = "its box holds" if len(refused) == 1 else "their boxes hold"
        raise ValueError(f"{', '.join(refused)}: {holds} no element; generate the sheet again")
    elements: dict[str, Image.Image] = {}
    records: dict[str, Any] = {}
    for slot in slots:
        parts = assigned[slot.name]
        if not parts:
            continue
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
            "file": _element_png(slot.name),
            "box": _rect(*(round(value) for value in scaled[slot.name])),
            "rect": _rect(x0, y0, x1 - x0, y1 - y0),
            "size": {"w": x1 - x0, "h": y1 - y0},
            "stretch": slot.stretch,
            "insets": insets,
            "padding": padding,
            "parts": len(parts),
        }
    for slot in slots:
        if slot.inside is not None and slot.name in records:
            records[slot.name]["inside"], note = fit_inside(records[slot.name], records[slot.inside], slot.inside)
            reasons += [f"{slot.name}: {note}"] if note else []
    report: dict[str, Any] = {
        "kind": REPORT_KIND,
        "version": 1,
        "sheet_size": [width, height],
        "status": "needs-review" if reasons else "pass",
        "review_reasons": reasons,
        "alpha_cleanup": cleanup,
        "dropped_parts": dropped,
        "unassigned": unassigned,
    }
    # a layout with optional elements always says which are missing; a square sheet's elements are all required
    if any(slot.optional for slot in slots):
        report["missing"] = missing
    if fitted is not None:
        (sx, tx), (sy, ty) = fit
        # `+ 0.0`: an offset that rounds to zero is written 0.0, never -0.0
        report["layout_fit"] = {"scale": {"x": round(sx, 4), "y": round(sy, 4)},
                                "offset": {"x": round(tx, 1) + 0.0, "y": round(ty, 1) + 0.0}}
    report["elements"] = records
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


def _taken(out_dir: Path) -> str | None:
    """What keeps a kit out of ``out_dir``, said of it, or None when it is absent or an empty directory."""
    if not os.path.lexists(out_dir):
        return None
    if not out_dir.is_dir():
        return "is a file"
    with os.scandir(out_dir) as entries:
        held = sorted(entry.name for entry in entries)
    if not held:
        return None
    return f"holds {held[0]}" + (f" and {len(held) - 1} more" if len(held) > 1 else "")


def _make_new(path: Path, make: Callable[[Path], object], remove: Callable[[Path], object],
              made: list[tuple[Callable[[Path], object], Path]]) -> None:
    """``make(path)`` for a path that must be new. ``(remove, path)`` goes on ``made`` just before the call, so that a
    run interrupted right after it still removes what it made, and comes back off when the path existed: what another
    process made is not this run's to remove."""
    made.append((remove, path))
    try:
        make(path)
    except FileExistsError:
        made.pop()
        raise


def _make_dirs(directory: Path, made: list[tuple[Callable[[Path], object], Path]]) -> None:
    """Create ``directory`` and its missing parents, putting each one this run creates on ``made`` (`_make_new`)."""
    missing = []
    while not os.path.lexists(directory):
        missing.append(directory)
        directory = directory.parent
    for path in reversed(missing):
        try:
            _make_new(path, os.mkdir, os.rmdir, made)  # removed only while empty: never what another process put there
        except FileExistsError:
            continue  # another process made it


def _write_new(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o666)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _linked(report: Path, temp: Path) -> bool:
    """Whether ``report`` is this run's ``temp`` file under its own name. When that cannot be read and the report
    exists, it may be."""
    try:
        return os.path.samestat(os.lstat(report), os.lstat(temp))
    except OSError:
        return os.path.lexists(report)


def _rename_in(staging: Path, out_dir: Path, given_empty: bool) -> None:
    """Give ``staging`` the name ``out_dir``, removing an empty ``out_dir`` first where a rename cannot take its name.
    A ValueError when another run's kit took the name first. Its try statement is kept out of `_write_kit`'s, which
    holds none (see there)."""
    try:
        if given_empty and not RENAME_ONTO_EMPTY_DIR:
            os.rmdir(out_dir)
        os.rename(staging, out_dir)
    except OSError as exc:
        taken = os.path.lexists(staging) and _taken(out_dir)  # only when the rename did not happen
        if taken:
            raise ValueError(f"--out-dir {out_dir} was filled while this kit was cut: it {taken}; "
                             "nothing was written") from exc
        raise


def _link_report(temp: Path, report: Path) -> None:
    """Give the finished ``temp`` its own name, ``report``; a ValueError when another run's report took it first.
    Its try statement is kept out of `_write_kit`'s, as `_rename_in`'s is."""
    try:
        os.link(temp, report)
    except FileExistsError as exc:
        raise ValueError(f"--report {report} appeared while this kit was cut; nothing was written") from exc


def _write_kit(out_dir: Path, files: dict[str, bytes], report: Path | None, report_data: bytes) -> None:
    """Publish a kit whole into ``out_dir``, absent or empty. ``files`` (paths inside it) are written into a hidden
    staging directory beside it, which then takes its name with one rename. A ``report`` outside it is written to a
    hidden temp file beside it and linked to its own name after the rename; the link fails when that name exists, so
    no run replaces another's report.

    One step publishes the kit: the rename, or with a report outside, the link. Before it, a run that fails or is
    interrupted removes everything it made (the kit it renamed in, the directories it created, its staging directory
    and temp file) and puts back an empty out-dir it was given; from it on, it removes nothing but its temp file.

    Python can raise an interrupt at any instruction: a signal's handler, or a tracer's, runs between any two. So the
    one ``except`` below covers every instruction from the first name the run creates to the end of the publish, and
    no try statement is nested in its own: a nested ``try:`` line compiles to an instruction that no handler covers,
    so an exception raised there would leave without the clean-up. The rename's and the link's own try statements are
    in `_rename_in` and `_link_report`, whose exceptions reach this handler at their call. No step is recorded after
    its call: ``renaming`` and ``linking`` are set before theirs and the filesystem says whether it happened, and
    each name the run creates goes on ``made`` before the call that creates it.

    A killed run, or one interrupted again while it removes what it made, leaves no part of a kit under the out-dir's
    name: at most its hidden staging directory or temp file beside them and the directories it created, or, stopped
    between the rename and the link, the whole kit without its report."""
    given_empty = out_dir.is_dir()
    tag = secrets.token_hex(6)
    staging = out_dir.parent / f".{out_dir.name}.ui-kit-{tag}"
    temp = None if report is None else report.parent / f".{report.name}.{tag}.tmp"
    made: list[tuple[Callable[[Path], object], Path]] = []
    renaming = linking = False
    try:
        _make_dirs(out_dir.parent, made)
        _make_new(staging, os.mkdir, lambda path: shutil.rmtree(path, ignore_errors=True), made)
        for name, data in files.items():
            target = staging / name
            if target.parent != staging:  # a report in a directory of its own inside the out-dir
                target.parent.mkdir(parents=True, exist_ok=True)
            _write_new(target, data)
        if temp is not None:
            _make_dirs(temp.parent, made)
            _make_new(temp, lambda path: _write_new(path, report_data), os.unlink, made)
        renaming = True
        _rename_in(staging, out_dir, given_empty)
        if temp is not None:
            linking = True
            _link_report(temp, report)
            with contextlib.suppress(OSError):
                os.unlink(temp)  # published: the temp is a second name of the report
    except BaseException:
        renamed = renaming and not os.path.lexists(staging)  # the staging directory took the out-dir's name
        published = renamed if temp is None else linking and _linked(report, temp)
        if published:
            if temp is not None:
                with contextlib.suppress(OSError):
                    os.unlink(temp)
            raise
        if renamed:
            with contextlib.suppress(OSError):
                os.rename(out_dir, staging)
        if given_empty and not os.path.lexists(out_dir):
            with contextlib.suppress(OSError):
                os.mkdir(out_dir)
        for remove, path in reversed(made):
            with contextlib.suppress(OSError):
                remove(path)
        raise


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
    taken = _taken(out_dir)
    if taken:
        raise ValueError(f"--out-dir {out_dir} {taken}; a kit is written whole into a new or empty directory")
    inside = report_path.is_relative_to(out_dir)
    if not inside and os.path.lexists(report_path):
        raise ValueError(f"--report {report_path} already exists; a report outside --out-dir is written to a new path")
    with Image.open(sheet) as image:
        if image.format != "PNG" or getattr(image, "n_frames", 1) != 1:
            raise ValueError("--sheet must be a single-frame PNG")
        elements, measured = split_ui_kit(image, alpha_floor=alpha_floor, alpha_edge=alpha_edge)
    written = {"kind": measured["kind"], "version": measured["version"], "sheet": str(sheet)}
    written.update((key, value) for key, value in measured.items() if key not in written)
    # wherever --report puts the report, each `file` leads from it to the PNG written in out_dir
    written["elements"] = {name: {**record, "file": report_file(report_path, out_dir / record["file"])}
                           for name, record in measured["elements"].items()}
    files: dict[str, bytes] = {}
    for name, element in elements.items():
        png = io.BytesIO()
        element.save(png, format="PNG")
        files[measured["elements"][name]["file"]] = png.getvalue()
    report_data = (json.dumps(written, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if inside:
        files[report_path.relative_to(out_dir).as_posix()] = report_data
    _write_kit(out_dir, files, None if inside else report_path, report_data)
    missing = written.get("missing") or []
    left_out = f"; missing, left out of the kit: {', '.join(missing)}" if missing else ""
    print(f"ui-kit: {written['status']} — {len(elements)} elements in {out_dir} (report: {report_path}){left_out}")
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sheet", required=True, type=Path,
                        help="transparent square or 2:3 PNG drawn with `gen --layout-guide ui-kit`")
    parser.add_argument("--out-dir", required=True, type=Path,
                        help="a new or empty directory for <element>.png and ui-kit.json; "
                             "one that holds anything is refused")
    parser.add_argument("--alpha-floor", type=int, default=64,
                        help="alpha haze cleanup floor, 1..255 (0 turns the cleanup off)")
    parser.add_argument("--alpha-edge", type=int, default=2,
                        help="faint pixels this close to a solid pixel are its edge and are kept")
    parser.add_argument("--report", type=Path,
                        help="JSON report (default: <out-dir>/ui-kit.json), a new path when outside --out-dir; "
                             "its element paths are relative to it")


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
