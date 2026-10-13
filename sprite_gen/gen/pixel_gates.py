# SPDX-License-Identifier: Apache-2.0
"""Gates G2..G6 of `pixel-draw`, on the pixels of a grid that passed G1 (it parsed whole).

- G2 palette: the opaque colours used are at most the rule's limit; a fixed palette allows only its
  own colours.
- G3 alpha: a sprite or icon frame has opaque pixels, four transparent corners and a border at
  least half transparent; a tile frame is opaque everywhere.
- G4 tile seam (warning): the colour step across the wrap edges over the mean step between
  neighbours inside the tile, at most 1.5. Mortar lines that meet the edge raise it on tiles that
  repeat cleanly, so it warns and never fails until the measure is refined.
- G5 stray pixels (warning): opaque pixels of a sprite or icon with no opaque 4-neighbour.
- G6 frames: every frame stands on the same bottom row, heights differ by at most one pixel and
  no two frames are the same. A walk cycle (contact, passing, contact, passing) also warns when
  its two contact frames differ by less than 40% of the mean contact↔passing difference: the legs
  barely change sides.

A failed gate fails the candidate; a warning is reported and passes.
"""
from __future__ import annotations

from typing import Any, Sequence

from .pixel_grid import Pixel, PaletteRule

SEAM_RATIO_MAX = 1.5
BORDER_TRANSPARENT_MIN = 0.5
HEIGHT_SPREAD_MAX = 1
WALK_CONTACT_RATIO_MIN = 0.4

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


def _bbox_rows(frame: Frame) -> tuple[int, int] | None:
    rows = [y for y, row in enumerate(frame) if any(_opaque(p) for p in row)]
    return (rows[0], rows[-1]) if rows else None


def _difference(a: Frame, b: Frame) -> int:
    return sum(1 for y in range(len(a)) for x in range(len(a[0])) if a[y][x] != b[y][x])


def g6_frames(frames: dict[str, Frame], *, walk_cycle: bool) -> dict[str, Any]:
    names = list(frames)
    spans = {name: _bbox_rows(frames[name]) for name in names}
    baselines = {name: (span[1] if span else None) for name, span in spans.items()}
    heights = {name: (span[1] - span[0] + 1 if span else 0) for name, span in spans.items()}
    identical = [[names[i], names[j]] for i in range(len(names)) for j in range(i + 1, len(names))
                 if frames[names[i]] == frames[names[j]]]
    spread = max(heights.values()) - min(heights.values())
    result: dict[str, Any] = {
        "ok": len(set(baselines.values())) == 1 and None not in baselines.values()
        and spread <= HEIGHT_SPREAD_MAX and not identical,
        "baselines": baselines, "heights": heights, "height_spread": spread, "identical_pairs": identical,
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


def evaluate(frames: dict[str, Frame], *, kind: str, rule: PaletteRule, walk_cycle: bool) -> dict[str, Any]:
    """Every gate of one candidate. `ok` is G2, G3 and G6 together; `warnings` say what G4, G5 and
    the walk check found, one line each."""
    gates: dict[str, Any] = {"G2": g2_palette(frames, rule), "G3": g3_alpha(frames, kind)}
    warnings: list[str] = []
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
        gates["G6"] = g6_frames(frames, walk_cycle=walk_cycle)
        walk = gates["G6"].get("walk")
        if walk and walk["warn"]:
            warnings.append(f"G6 weak leg alternation: the contact frames differ by {walk['contact_difference']} "
                            f"pixels, {walk['ratio']:.0%} of the contact-passing difference (under "
                            f"{WALK_CONTACT_RATIO_MIN:.0%}) — touch up the legs")
    failed = [name for name in ("G2", "G3", "G6") if name in gates and not gates[name]["ok"]]
    return {"ok": not failed, "failed": failed, "warnings": warnings, **gates}
