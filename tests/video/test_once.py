# SPDX-License-Identifier: Apache-2.0
"""Terminal-pose cuts retain their endpoint and never ask it to match the start."""
import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from sprite_gen.video import loop, batch, canvas, clip_prompt


def keyed(root, *, unsettled=False, clipped=False, count=90):
    root.mkdir()
    for k in range(count):
        t = min(1, max(0, (k - 10) / (count - 25)))
        x = round(28 + 28 * t)
        if unsettled and k >= count - 10:
            x += 8 * (k % 2)
        im = Image.new("RGBA", (128, 96))
        ImageDraw.Draw(im).rectangle((x, 20 + round(35 * t), x + 16 + round(24 * t), 85), fill=(200, 80, 30, 255))
        if clipped and k == count // 2:
            ImageDraw.Draw(im).rectangle((0, 40, 20, 60), fill=(200, 80, 30, 255))
        im.save(root / f"{k:04d}.png")
    return root


def cut(tmp_path, **kwargs):
    return loop.run_loop(tmp_path / "keyed", tmp_path / "out", fps=24, state="death",
                         min_len=None, max_len=None, n_out=7, seam_max=0.001,
                         name="death", report_path=None, cycle_mode="once", **kwargs)


def test_cut_keeps_final_pose_in_strip_and_once_previews(tmp_path):
    keyed(tmp_path / "keyed")
    report = cut(tmp_path)
    meta = report["strip"]
    assert report["cycle"]["kind"] == "once"
    assert meta["loop"] is False and meta["hold_last"] is True
    assert meta["end_frame"] == meta["frames"] - 1
    assert meta["source_cut"]["samples"][-1] == 89
    assert meta["sample_indices"][-1] == report["cycle"]["length"] - 1
    assert 0 <= meta["events"][0]["frame"] <= meta["end_frame"]
    strip = Image.open(meta["path"]).convert("RGBA")
    last = strip.crop((strip.width - meta["w"], 0, strip.width, strip.height))
    for suffix, repeat in (("gif", None), ("webp", 1)):
        with Image.open(tmp_path / "out" / f"death.{suffix}") as im:
            assert im.info.get("loop") == repeat
            im.seek(im.n_frames - 1)
            assert im.convert("RGBA").getchannel("A").tobytes() == last.getchannel("A").tobytes()


@pytest.mark.parametrize("kwargs,reason", [({"unsettled": True}, "stable"), ({"clipped": True}, "edge")])
def test_refuses_unfinished_or_clipped_body(tmp_path, kwargs, reason):
    keyed(tmp_path / "keyed", **kwargs)
    with pytest.raises(SystemExit, match=reason):
        cut(tmp_path)
    report = json.loads((tmp_path / "out/death.loop.report.json").read_text())
    assert report["status"] == "failed"


def test_once_cannot_apply_wrap_correction_or_take_a_loop_state(tmp_path):
    keyed(tmp_path / "keyed")
    with pytest.raises(SystemExit, match="anchor"):
        cut(tmp_path, anchor="body")
    with pytest.raises(SystemExit, match="idle"):
        loop.run_loop(tmp_path / "keyed", tmp_path / "idle", fps=24, state="idle", min_len=None,
                      max_len=None, n_out=None, seam_max=2, name="idle", report_path=None, cycle_mode="once")


@pytest.mark.parametrize("state", ["death", "fall", "land", "cast", "hurt_hold", "transform"])
def test_nonloop_states_have_terminal_prompts_and_route_once(state):
    record = clip_prompt.plan_prompt(direction="side", state=state)
    assert record["cycle"] == "once"
    assert not batch.pins_last_frame(state, "side")
    assert "evenly paced" not in record["prompt"] and "so the animation loops" not in record["prompt"]
    assert "0.4" in record["prompt"]
    assert canvas.profile_for(state) != canvas.profile_for(None)


def test_end_pose_input_is_distinct_from_first_frame_pin():
    record = clip_prompt.plan_prompt(direction="side", state="death", end_frame=Path("final.png"))
    assert record["cycle"] == "once" and record["last_frame"] == "final.png"
    with pytest.raises(SystemExit, match="end frame"):
        clip_prompt.plan_prompt(direction="side", state="death", end_frame=Path("final.png"), last_frame=False)
    with pytest.raises(SystemExit, match="non-loop"):
        clip_prompt.plan_prompt(direction="side", state="idle", end_frame=Path("final.png"))
