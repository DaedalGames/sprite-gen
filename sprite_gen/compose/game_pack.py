# SPDX-License-Identifier: Apache-2.0
"""Resize finished assets and publish an immutable game atlas pack."""
from __future__ import annotations

import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from PIL import Image

from sprite_gen.spec.assets import finite, load_asset_states

MAX_PAGE_SIZE = 4096
MAX_BYTES = 50_000_000


def _integer(value, name, *, maximum=None):
    number = finite(value, name, positive=True)
    if number != int(number) or maximum is not None and number > maximum:
        raise ValueError(f"{name} must be an integer between 1 and {maximum or 'its limit'}")
    return int(number)


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError("asset id and state must use letters, digits, underscore, dot or dash")
    return value


def _policy(sequence, entry):
    count = len(sequence.frames)
    metadata = sequence.metadata
    result = {}
    for key, default in (("loop", True), ("hold_last", False), ("end_frame", count - 1),
                         ("blend", "normal"), ("events", []), ("direction", None)):
        result[key] = entry.get(key, metadata.get(key, default))
    for key in ("loop", "hold_last"):
        if type(result[key]) is not bool:
            raise ValueError(f"{key} must be a bool")
    if result["blend"] not in ("normal", "add"):
        raise ValueError("blend must be normal or add")
    if type(result["end_frame"]) is not int or not 0 <= result["end_frame"] < count:
        raise ValueError("end_frame must index a playback frame (zero-based)")
    if result["direction"] is not None and not isinstance(result["direction"], str):
        raise ValueError("direction must be a string or null")
    if not isinstance(result["events"], list):
        raise ValueError("events must be a list")
    for event in result["events"]:
        if (not isinstance(event, dict) or type(event.get("frame")) is not int
                or not 0 <= event["frame"] < count or not isinstance(event.get("name"), str) or not event["name"]):
            raise ValueError("each event needs a zero-based playback frame and a name")
    result["durations_ms"] = [duration * 1000 for duration in sequence.durations]
    return result


def _display(sequence, entry):
    display = entry.get("display")
    if not isinstance(display, dict) or len(display) != 1 or not set(display) <= {"width", "height", "body_height"}:
        raise ValueError("display requires exactly one of width, height or body_height")
    dimension, value = next(iter(display.items()))
    target = _integer(value, f"display {dimension}")
    width, height = sequence.size
    if dimension == "body_height":
        reference = finite(entry.get("source_body_height", sequence.metadata.get("body_h")),
                           "source_body_height (required when the asset has no body_h)", positive=True)
    else:
        reference = width if dimension == "width" else height
    scale = target / reference
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    anchor = [sequence.anchor[0] * size[0] / width, sequence.anchor[1] * size[1] / height]
    sampling = entry.get("resample", "nearest" if sequence.metadata.get("frame_variant") == "pixel" else "lanczos")
    if sampling not in ("nearest", "lanczos"):
        raise ValueError("resample must be nearest or lanczos")
    return size, anchor, Image.Resampling.NEAREST if sampling == "nearest" else Image.Resampling.LANCZOS


def _prepare(request, base, page_size):
    if not isinstance(request, dict) or request.get("version") != 1:
        raise ValueError("game pack request requires version 1")
    entries = request.get("assets")
    if not isinstance(entries, list) or not entries:
        raise ValueError("assets must be a nonempty list")
    ids, animations, frames, sources = set(), {}, [], set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("each asset must be an object")
        asset_id = _name(entry.get("id"))
        if asset_id in ids:
            raise ValueError(f"duplicate asset id: {asset_id}")
        ids.add(asset_id)
        if not isinstance(entry.get("source"), str) or not entry["source"]:
            raise ValueError("each asset requires a source path")
        sequences = load_asset_states(base / entry["source"], state=entry.get("state"), fps=entry.get("fps"), anchor=entry.get("anchor"))
        for state, sequence in sequences.items():
            key = asset_id if state is None else f"{asset_id}/{_name(state)}"
            size, anchor, sampling = _display(sequence, entry)
            # One extruded pixel on each edge keeps filtered textures from bleeding.
            if max(size) + 2 > page_size:
                raise ValueError(f"{key}: frame {size} plus padding exceeds page size {page_size}")
            animation = _policy(sequence, entry)
            animation.update(anchor=anchor, size={"w": size[0], "h": size[1]}, frames=[])
            for index, image in enumerate(sequence.frames):
                name = f"{key}/{index:06d}"
                animation["frames"].append(name)
                frames.append((name, image.resize(size, sampling), anchor))
            animations[key] = animation
            sources.update(sequence.source_files)
    return dict(sorted(animations.items())), sorted(frames, key=lambda f: f[0]), sources


def _layout(frames, limit):
    """Deterministic shelf placement; playback order is independent of texture position."""
    pages, placed = [], []
    x = y = row_height = width = height = 0
    for name, image, anchor in frames:
        w, h = image.width + 2, image.height + 2
        if x + w > limit:
            x, y, row_height = 0, y + row_height, 0
        if y + h > limit:
            pages.append((width, height, placed))
            placed = []
            x = y = row_height = width = height = 0
        placed.append((name, image, anchor, x + 1, y + 1))
        x += w
        row_height = max(row_height, h)
        width, height = max(width, x), max(height, y + h)
    pages.append((width, height, placed))
    return pages


def _paste_extruded(sheet, image, x, y):
    w, h = image.size
    sheet.paste(image, (x, y))
    sheet.paste(image.crop((0, 0, w, 1)), (x, y - 1))
    sheet.paste(image.crop((0, h - 1, w, h)), (x, y + h))
    sheet.paste(image.crop((0, 0, 1, h)), (x - 1, y))
    sheet.paste(image.crop((w - 1, 0, w, h)), (x + w, y))
    for dx, sx in ((-1, 0), (w, w - 1)):
        for dy, sy in ((-1, 0), (h, h - 1)):
            sheet.putpixel((x + dx, y + dy), image.getpixel((sx, sy)))


def _encode(pages, animations, max_bytes):
    payloads, textures = {}, []
    total = 0
    for index, (width, height, placed) in enumerate(pages):
        sheet = Image.new("RGBA", (width, height))
        records = {}
        for name, image, anchor, x, y in placed:
            _paste_extruded(sheet, image, x, y)
            size = {"w": image.width, "h": image.height}
            records[name] = {"frame": {"x": x, "y": y, **size}, "rotated": False, "trimmed": False,
                             "spriteSourceSize": {"x": 0, "y": 0, **size}, "sourceSize": size,
                             "pivot": {"x": anchor[0]/image.width, "y": anchor[1]/image.height}}
        png_name, webp_name = f"atlas-{index:03d}.png", f"atlas-{index:03d}.webp"
        for name, fmt, options in ((png_name, "PNG", {"compress_level": 9}),
                                   (webp_name, "WEBP", {"lossless": True, "exact": True, "method": 6})):
            buffer = BytesIO()
            sheet.save(buffer, format=fmt, **options)
            payloads[name] = buffer.getvalue()
            total += len(payloads[name])
            if total > max_bytes:
                raise ValueError(f"game pack exceeds {max_bytes} bytes (50 MB maximum)")
        textures.append({"image": png_name, "webp": webp_name, "frames": records,
                         "meta": {"image": png_name, "format": "RGBA8888", "size": {"w": width, "h": height}, "scale": "1"}})
    result = {"kind": "sprite-gen-game-pack", "version": 1, "textures": textures, "animations": animations}
    payloads["game-pack.json"] = (json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if total + len(payloads["game-pack.json"]) > max_bytes:
        raise ValueError(f"game pack exceeds {max_bytes} bytes (50 MB maximum, including its manifest)")
    return result, payloads


def _publish(out_dir, payloads):
    """One atomic directory rename; concurrent publishers cannot interleave packs."""
    def identical():
        return (out_dir.is_dir() and {p.name for p in out_dir.iterdir()} == set(payloads)
                and all((out_dir / name).is_file() and (out_dir / name).read_bytes() == data
                        for name, data in payloads.items()))

    if out_dir.exists():
        if identical():
            return
        raise ValueError("output directory already exists with different content; use a new directory")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}-", dir=out_dir.parent))
    try:
        for name, data in payloads.items():
            with (stage / name).open("wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        try:
            stage.rename(out_dir)
        except OSError:
            if not identical():
                raise
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def pack(request: Path, out_dir: Path, *, page_size=MAX_PAGE_SIZE, max_bytes=MAX_BYTES):
    page_size = _integer(page_size, "page size", maximum=MAX_PAGE_SIZE)
    max_bytes = _integer(max_bytes, "max bytes", maximum=MAX_BYTES)
    request, out_dir = Path(request).expanduser().resolve(), Path(out_dir).expanduser().resolve()
    document = json.loads(request.read_text(encoding="utf-8"))
    animations, frames, sources = _prepare(document, request.parent, page_size)
    if any(path == out_dir or out_dir in path.parents for path in sources | {request}):
        raise ValueError("output directory must not contain any source file")
    result, payloads = _encode(_layout(frames, page_size), animations, max_bytes)
    _publish(out_dir, payloads)
    return result


def add_arguments(parser):
    parser.epilog = "events[].frame and end_frame are zero-based indices in output playback order."
    parser.add_argument("--request", type=Path, required=True, help="JSON result list with display sizes")
    parser.add_argument("--out-dir", type=Path, required=True, help="new directory for the immutable pack")
    parser.add_argument("--page-size", type=int, default=MAX_PAGE_SIZE)
    parser.add_argument("--max-bytes", type=int, default=MAX_BYTES)


def run(**kwargs):
    try:
        result = pack(kwargs["request"], kwargs["out_dir"], page_size=kwargs.get("page_size", MAX_PAGE_SIZE),
                      max_bytes=kwargs.get("max_bytes", MAX_BYTES))
    except (ValueError, OSError) as exc:
        raise SystemExit(f"game-pack: {exc}") from exc
    print(json.dumps({"ok": True, "manifest": str(Path(kwargs["out_dir"]) / "game-pack.json"),
                      "pages": len(result["textures"]), "animations": len(result["animations"])}))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return run(**vars(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
