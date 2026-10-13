# SPDX-License-Identifier: Apache-2.0
"""Emissive effects on black: straight-alpha conversion and generation framing.

RGB on black is already multiplied by its coverage. The maximum channel is the
smallest alpha that can reconstruct every channel without clipping. No threshold
is applied: a dim wisp and a detached spark both carry intentional light.
"""
from pathlib import Path
from typing import Any

from PIL import Image

from sprite_gen._deps import np

STATE = "effect"
ALPHA_MODES = ("chroma", "brightness")


def brightness_alpha(image: Image.Image) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    alpha = rgb.max(axis=2)
    color = np.zeros_like(rgb)
    np.divide(rgb * 255, alpha[:, :, None], out=color, where=alpha[:, :, None] > 0)
    return Image.fromarray(np.dstack((np.rint(color), alpha)).astype(np.uint8), "RGBA")


def key_frames(files: list[Path], directory: Path) -> dict[str, Any]:
    if not files:
        raise SystemExit("video-frames: no frames to convert")
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for src in files:
        with Image.open(src) as raw:
            image = brightness_alpha(raw)
        image.save(directory / src.name)
        zero = image.getchannel("A").histogram()[0]
        rows.append({"frame": src.name, "alpha_zero_pct": round(100 * zero / (image.width * image.height), 2),
                     "route": "brightness", "specks": 0})
    return {"alpha_mode": "brightness", "blend": "add", "frames": len(rows), "rows": rows,
            "alpha_zero_pct_min": min(r["alpha_zero_pct"] for r in rows),
            "alpha_zero_pct_max": max(r["alpha_zero_pct"] for r in rows),
            "specks": {"dropped": 0, "frames": 0, "policy": "off"},
            "edge_contacts": [], "edge_policy": "off", "decontam": {"mode": "off", "applied_frames": 0},
            "warnings": ["Brightness alpha represents emitted light. Dark smoke becomes faint; "
                         "black cannot occlude the scene. Use chroma alpha for opaque smoke."]}


def prompt(character: str | None, motion: str | None) -> str:
    subject = character.strip().rstrip(".") if character else "A luminous game effect"
    if motion is not None and not motion.strip():
        raise SystemExit("video: motion description is empty")
    action = f" {' '.join(motion.split())}" if motion else ""
    return (f"2D game effect animation. {subject}.{action} One complete lifetime: the effect appears, "
            "grows to its brightest peak, decays, and disappears completely. Its origin stays at the center "
            "of the image. A perfectly black background (RGB 0,0,0), no scenery, ground, characters, "
            "text or shadows. Only the effect emits light; sparks, glow and wisps are part of it. "
            "Camera locked, no zoom, no pan, no reframing. End on the supplied empty black frame and "
            "hold complete black with no remaining glow. Perform the effect once.")


def canvas(image: Image.Image) -> tuple[Image.Image, dict[str, Any]]:
    """Keep the effect's frame and centre; composite transparent artwork onto black."""
    rgba = image.convert("RGBA")
    black = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
    result = Image.alpha_composite(black, rgba).convert("RGB")
    return result, {"fit": "state", "shape": "source", "canvas": list(result.size), "offset": [0, 0],
                    "key": "black", "key_rgb": [0, 0, 0], "why": "fixed effect origin on black"}


def empty_end(canvas_path: Path, out: Path) -> Path:
    with Image.open(canvas_path) as source:
        blank = Image.new("RGB", source.size, (0, 0, 0))
    blank.save(out)
    return out
