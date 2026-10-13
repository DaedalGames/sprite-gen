# SPDX-License-Identifier: Apache-2.0
"""Black-backed emissions keep their energy, isolated sparks and empty endpoints."""
from PIL import Image
import pytest

from sprite_gen._deps import np
from sprite_gen.video import frames, batch, clip_prompt


def test_brightness_alpha_reconstructs_dark_and_saturated_emission(tmp_path):
    src = tmp_path / "raw.png"
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    rgb[0, 0] = (255, 0, 0)  # an intentional isolated edge spark
    rgb[7, 7] = (3, 3, 3)  # dark smoke cannot be discarded by a threshold
    rgb[8, 8] = (65, 25, 5)  # soft glow: no double multiplication by alpha
    Image.fromarray(rgb).save(src)
    report = frames.key_frames([src], tmp_path / "keyed", alpha_mode="brightness")
    rgba = np.array(Image.open(tmp_path / "keyed" / src.name)).astype(float)
    emitted = rgba[:, :, :3] * rgba[:, :, 3:] / 255
    assert np.max(abs(emitted - rgb)) <= .5
    assert rgba[7, 7, 3] == 3 and rgba[0, 0, 3] == 255
    assert not rgba[1, 1].any()
    # An additive preview on a bright surface must never acquire a dark halo.
    for background in (0, 220, 255):
        composed = np.minimum(background + emitted, 255)
        assert np.min(composed) >= background
    assert report["blend"] == "add" and report["edge_policy"] == "off"
    assert report["specks"]["dropped"] == 0


def test_brightness_run_skips_chroma_cleanup(tmp_path, monkeypatch):
    src = tmp_path / "f0001.png"
    Image.new("RGB", (8, 8), (0, 0, 0)).save(src)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"synthetic")
    monkeypatch.setattr(frames, "probe", lambda _: {"stream_index": 0, "fps": 24})
    monkeypatch.setattr(frames, "extract", lambda *a: [src])
    def forbidden(*a, **k):
        pytest.fail("brightness alpha must not call chroma cleanup")
    for name in ("cutout", "drop_specks", "decide_spill", "edge_contact"):
        monkeypatch.setattr(frames, name, forbidden)
    report = frames.run_frames(clip, tmp_path / "out", key="auto", allow_edge_contact=False,
                              report_path=None, alpha_mode="brightness")
    assert report["alpha_zero_pct_min"] == 100
    assert report["alpha_mode"] == "brightness" and report["blend"] == "add"


def test_effect_prompt_uses_black_end_and_lifetime_only():
    prompt = batch.build_prompt("front", "effect", "A burst of fire")
    assert "black" in prompt and "disappear" in prompt
    assert "no effects" not in prompt and "chroma" not in prompt
    assert "no effects" in batch.build_prompt("front", "attack", None)
    assert batch.pins_last_frame("effect", "front")
    plan = clip_prompt.plan_prompt(direction="front", state="effect")
    assert plan["last_frame"] == "black" and plan["cycle"] == "lifetime"
    assert "--alpha-mode brightness" in plan["commands"]["frames"]
    with pytest.raises(SystemExit, match="end.frame"):
        clip_prompt.plan_prompt(direction="front", state="effect", last_frame=False, unpinned=True)
