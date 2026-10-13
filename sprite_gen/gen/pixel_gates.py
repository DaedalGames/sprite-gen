# SPDX-License-Identifier: Apache-2.0
"""Gates G2..G6 and G8 of `pixel-draw`, on the pixels of a grid that passed G1 (it parsed whole).

- G2 palette: the opaque colours used are at most the rule's limit; a fixed palette allows only its
  own colours.
- G3 alpha: a sprite or icon frame has opaque pixels, four transparent corners and a border at
  least half transparent; a tile frame is opaque everywhere.
- G4 tile seam (warning): the colour step across the wrap edges over the mean step between
  neighbours inside the tile, at most 1.5. Mortar lines that meet the edge raise it on tiles that
  repeat cleanly, so it warns and never fails until the measure is refined.
- G5 stray pixels (warning): opaque pixels of a sprite or icon with no opaque 4-neighbour.
- G6 frames: every frame stands on the same bottom row, heights differ by at most one pixel and
  no two frames are the same. A sprite's or icon's rows and heights are measured inside its safe
  area: a staff raised or a spark into the margin in one frame is not a taller figure (G8 judges
  the margin). A walk cycle (contact, passing, contact, passing) also warns when
  its two contact frames differ by less than 40% of the mean contact↔passing difference: the legs
  barely change sides.
- G8 safe area, on a sprite or icon: the engine's safe area (`safe_area`, the margin `prepare`
  keeps around every cell). A pixel on the canvas's outermost row or column fails — the margin
  is room for a weapon tip, a strand of hair or an effect, never the canvas edge. It warns when
  the body box — the rows and columns holding at least a quarter as many opaque pixels as the
  fullest one, so a thin tip drops out of it — reaches more than a pixel past the safe area, and
  when a sprite's body box does not end on the floor row, the safe area's own bottom row, where
  `prepare`'s fit grounds a frame. Below 11 px the margin is 0 and there is no edge to keep.

A failed gate fails the candidate; a warning is reported and passes.
"""
from __future__ import annotations

from typing import Any, Sequence

from .pixel_grid import Pixel, PaletteRule

SEAM_RATIO_MAX = 1.5
BORDER_TRANSPARENT_MIN = 0.5
HEIGHT_SPREAD_MAX = 1
WALK_CONTACT_RATIO_MIN = 0.4
BODY_LINE_SHARE = 0.25
SAFE_OVERFLOW_MAX = 1

Frame = Sequence[Sequence[Pixel]]


def _opaque(pixel: Pixel) -> bool:
    return pixel[3] >= 128


def _hex(pixel: Pixel) -> str:
    return "#{:02X}{:02X}{:02X}".format(*pixel[:3])


def g2_palette(frames: dict[str, Frame], rule: PaletteRule) -> dict[str, Any]:
    used = sorted({_hex(p) for frame in frames.values() for row in frame for p in row if _opaque(p)})
    off = [c for c in used if rule.fixed and c not in rule.colors]
    return {"ok": len(used) <= rule.limit and not off, "colors_used": len(used), "limit": rule.limit,
            "palette": rule.name, "off_palette": off}


def _frame_alpha(frame: Frame) -> dict[str, Any]:
    height, width = len(frame), len(frame[0])
    opaque = sum(1 for row in frame for p in row if _opaque(p))
    border = [frame[y][x] for y in range(height) for x in range(width) if x in (0, width - 1) or y in (0, height - 1)]
    corners = (frame[0][0], frame[0][width - 1], frame[height - 1][0], frame[height - 1][width - 1])
    return {
        "opaque_pixels": opaque,
        "opaque_fraction": round(opaque / (width * height), 4),
        "corners_transparent": not any(_opaque(p) for p in corners),
        "border_transparent_fraction": round(sum(1 for p in border if not _opaque(p)) / len(border), 4),
    }


def g3_alpha(frames: dict[str, Frame], kind: str) -> dict[str, Any]:
    per = {name: _frame_alpha(frame) for name, frame in frames.items()}
    if kind == "tile":
        ok = all(m["opaque_fraction"] == 1.0 for m in per.values())
    else:
        ok = all(m["opaque_pixels"] > 0 and m["corners_transparent"]
                 and m["border_transparent_fraction"] >= BORDER_TRANSPARENT_MIN for m in per.values())
    return {"ok": ok, "kind": kind, "frames": per}


def _step(a: Pixel, b: Pixel) -> float:
    return sum(abs(a[i] - b[i]) for i in range(3)) / 3


def seam_ratio(frame: Frame) -> float:
    height, width = len(frame), len(frame[0])
    inner = [_step(frame[y][x], frame[y][x + 1]) for y in range(height) for x in range(width - 1)]
    inner += [_step(frame[y][x], frame[y + 1][x]) for y in range(height - 1) for x in range(width)]
    wrap = [_step(frame[y][width - 1], frame[y][0]) for y in range(height)]
    wrap += [_step(frame[height - 1][x], frame[0][x]) for x in range(width)]
    inner_mean = sum(inner) / max(1, len(inner))
    wrap_mean = sum(wrap) / max(1, len(wrap))
    if inner_mean == 0:
        return 0.0 if wrap_mean == 0 else 99.0
    return round(wrap_mean / inner_mean, 3)


def g4_seam(frames: dict[str, Frame]) -> dict[str, Any]:
    ratios = {name: seam_ratio(frame) for name, frame in frames.items()}
    worst = max(ratios.values())
    return {"warn": worst > SEAM_RATIO_MAX, "seam_ratio": worst, "max": SEAM_RATIO_MAX, "frames": ratios}


def isolated_pixels(frame: Frame) -> int:
    height, width = len(frame), len(frame[0])
    count = 0
    for y in range(height):
        for x in range(width):
            if not _opaque(frame[y][x]):
                continue
            neighbours = ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
            if not any(0 <= nx < width and 0 <= ny < height and _opaque(frame[ny][nx]) for nx, ny in neighbours):
                count += 1
    return count


def g5_isolated(frames: dict[str, Frame]) -> dict[str, Any]:
    per = {name: isolated_pixels(frame) for name, frame in frames.items()}
    total = sum(per.values())
    return {"warn": total > 0, "isolated_pixels": total, "frames": per}


def _bbox_rows(frame: Frame, box: Sequence[int] | None) -> tuple[int, int] | None:
    left, top, right, bottom = box or (0, 0, len(frame[0]) - 1, len(frame) - 1)
    rows = [y for y in range(top, bottom + 1) if any(_opaque(p) for p in frame[y][left:right + 1])]
    return (rows[0], rows[-1]) if rows else None


def _difference(a: Frame, b: Frame) -> int:
    return sum(1 for y in range(len(a)) for x in range(len(a[0])) if a[y][x] != b[y][x])


def g6_frames(frames: dict[str, Frame], *, walk_cycle: bool, box: Sequence[int] | None = None) -> dict[str, Any]:
    """`box` (first and last column and row) is where the rows and heights are measured: a sprite's
    or icon's safe area, the whole frame without one."""
    names = list(frames)
    spans = {name: _bbox_rows(frames[name], box) for name in names}
    baselines = {name: (span[1] if span else None) for name, span in spans.items()}
    heights = {name: (span[1] - span[0] + 1 if span else 0) for name, span in spans.items()}
    identical = [[names[i], names[j]] for i in range(len(names)) for j in range(i + 1, len(names))
                 if frames[names[i]] == frames[names[j]]]
    spread = max(heights.values()) - min(heights.values())
    result: dict[str, Any] = {
        "ok": len(set(baselines.values())) == 1 and None not in baselines.values()
        and spread <= HEIGHT_SPREAD_MAX and not identical,
        "baselines": baselines, "heights": heights, "height_spread": spread, "identical_pairs": identical,
        "measured_in": list(box) if box else None,
    }
    if walk_cycle:
        a, b, c, d = (frames[n] for n in names)
        contact = _difference(a, c)
        crossing = (_difference(a, b) + _difference(b, c) + _difference(c, d) + _difference(d, a)) / 4
        ratio = round(contact / crossing, 3) if crossing else 0.0
        result["walk"] = {"warn": ratio < WALK_CONTACT_RATIO_MIN, "contact_difference": contact,
                          "contact_passing_difference": round(crossing, 1), "ratio": ratio,
                          "min": WALK_CONTACT_RATIO_MIN}
    return result


def safe_area(width: int, height: int) -> dict[str, Any]:
    """The engine's safe area of a width x height canvas, as `prepare` keeps it around a cell
    (`normalize_cell`: DEFAULT_SAFE_MARGIN_RATIO of each side, floored — 32 px keeps 3 and 26
    inside). `box` is its first and last column and row, `floor_row` its bottom row."""
    # Imported here: `prepare` owns the margin and pulls in the row machinery.
    from .prepare import normalize_cell

    cell = normalize_cell({"width": width, "height": height}, width, None)
    mx, my = cell["safe_margin_x"], cell["safe_margin_y"]
    return {"margin": [mx, my], "size": [width - 2 * mx, height - 2 * my],
            "box": [mx, my, width - 1 - mx, height - 1 - my], "floor_row": height - 1 - my}


def _body_lines(counts: list[int]) -> list[int]:
    fullest = max(counts)
    return [i for i, n in enumerate(counts) if n and n >= BODY_LINE_SHARE * fullest]


def _frame_safe(frame: Frame, area: dict[str, Any]) -> dict[str, Any]:
    height, width = len(frame), len(frame[0])
    mx, my = area["margin"]
    left, top, right, bottom = area["box"]
    rows = [sum(1 for p in row if _opaque(p)) for row in frame]
    columns = [sum(1 for y in range(height) if _opaque(frame[y][x])) for x in range(width)]
    edge = {(x, y) for y in range(height) for x in range(width) if _opaque(frame[y][x])
            and ((mx and x in (0, width - 1)) or (my and y in (0, height - 1)))}
    if not any(rows):
        return {"edge_pixels": 0, "bbox": None, "body_box": None, "overflow": None}
    used_rows = [y for y, n in enumerate(rows) if n]
    used_columns = [x for x, n in enumerate(columns) if n]
    body_rows, body_columns = _body_lines(rows), _body_lines(columns)
    body = [body_columns[0], body_rows[0], body_columns[-1], body_rows[-1]]
    return {
        "edge_pixels": len(edge),
        "bbox": [used_columns[0], used_rows[0], used_columns[-1], used_rows[-1]],
        "body_box": body,
        "overflow": {"left": max(0, left - body[0]), "top": max(0, top - body[1]),
                     "right": max(0, body[2] - right), "bottom": max(0, body[3] - bottom)},
    }


def g8_safe_area(frames: dict[str, Frame], kind: str, area: dict[str, Any]) -> dict[str, Any]:
    per = {name: _frame_safe(frame, area) for name, frame in frames.items()}
    edge = sum(m["edge_pixels"] for m in per.values())
    body_overflow = {}
    for name, m in per.items():
        sides = {side: px for side, px in (m["overflow"] or {}).items() if px > SAFE_OVERFLOW_MAX}
        if sides:
            body_overflow[name] = sides
    floor = area["floor_row"] if kind == "sprite" else None
    off_floor = {name: m["body_box"][3] - floor for name, m in per.items()
                 if floor is not None and m["body_box"] and m["body_box"][3] != floor}
    return {"ok": edge == 0, "margin": area["margin"], "safe_area": area["size"], "safe_box": area["box"],
            "floor_row": floor, "edge_pixels": edge, "overflow_max": SAFE_OVERFLOW_MAX,
            "body_overflow": body_overflow, "off_floor": off_floor, "frames": per}


def _g8_warnings(g8: dict[str, Any]) -> list[str]:
    lines = []
    if g8["body_overflow"]:
        (width, height), (mx, my) = g8["safe_area"], g8["margin"]
        where = "; ".join(f"{name}: " + ", ".join(f"{px} px {side}" for side, px in sides.items())
                          for name, sides in g8["body_overflow"].items())
        margin = f"{mx}" if mx == my else f"{mx}x{my}"
        lines.append(f"G8 the body reaches past the {width}x{height} safe area by more than {SAFE_OVERFLOW_MAX} px "
                     f"({where}) — only a thin detail may go into the {margin} px margin")
    if g8["off_floor"]:
        where = ", ".join(f"{name} {abs(d)} px {'below' if d > 0 else 'above'} it" for name, d in g8["off_floor"].items())
        lines.append(f"G8 the feet are not on the floor row (the safe area's bottom row): {where}")
    return lines


def evaluate(frames: dict[str, Frame], *, kind: str, rule: PaletteRule, walk_cycle: bool) -> dict[str, Any]:
    """Every gate of one candidate. `ok` is G2, G3, G6 and G8 together; `warnings` say what G4, G5,
    the walk check and G8 found, one line each."""
    gates: dict[str, Any] = {"G2": g2_palette(frames, rule), "G3": g3_alpha(frames, kind)}
    warnings: list[str] = []
    first = next(iter(frames.values()))
    area = None if kind == "tile" else safe_area(len(first[0]), len(first))
    if kind == "tile":
        gates["G4"] = g4_seam(frames)
        if gates["G4"]["warn"]:
            warnings.append(f"G4 tile seam ratio {gates['G4']['seam_ratio']} > {SEAM_RATIO_MAX} "
                            "(look at tiled-3x3.png: mortar lines on the edge raise it on tiles that repeat cleanly)")
    else:
        gates["G5"] = g5_isolated(frames)
        if gates["G5"]["warn"]:
            warnings.append(f"G5 {gates['G5']['isolated_pixels']} stray pixel(s) with no opaque neighbour")
    if len(frames) > 1:
        gates["G6"] = g6_frames(frames, walk_cycle=walk_cycle, box=area["box"] if area else None)
        walk = gates["G6"].get("walk")
        if walk and walk["warn"]:
            warnings.append(f"G6 weak leg alternation: the contact frames differ by {walk['contact_difference']} "
                            f"pixels, {walk['ratio']:.0%} of the contact-passing difference (under "
                            f"{WALK_CONTACT_RATIO_MIN:.0%}) — touch up the legs")
    if area is not None:
        gates["G8"] = g8_safe_area(frames, kind, area)
        warnings += _g8_warnings(gates["G8"])
    failed = [name for name in ("G2", "G3", "G6", "G8") if name in gates and not gates[name]["ok"]]
    return {"ok": not failed, "failed": failed, "warnings": warnings, **gates}
