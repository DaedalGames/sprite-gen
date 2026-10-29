# SPDX-License-Identifier: Apache-2.0
"""Held cells (sprite_gen/video/held_fill.py, docs/loop-repair.md section 7): neighbour cells of a cut that show the
same picture are found on every cut `video-loop` makes, recorded (`held_fill`: the held runs, the poses and the cells)
and, by default, made again as the frames between the pose before them and the pose after; the cut's first and last
cells, its length, its beat and its wrap stay as they were, and a part standing still in both stays where it stood.
A clip asked for at a higher rate than it was filmed (`--out-fps`) is given frames between, not frames shown again.
With no RIFE the cut is delivered as filmed with a warning naming the motion and the install line; `--fill-held off`
cuts it as before. A cut with no held cell comes out byte for byte as before (the six inputs below, pinned on the
engine before the fill).

The interpolator is a stand-in that draws the synthetic clip again at the time asked for (`Between`), so the tests
need no binary and say which frame comes back."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sprite_gen.video import held_fill
from sprite_gen.video import loop as loop_mod
from tests.video.held_fill_inputs import (N, Between, hop, planter, save, shown_again, slid, swing, topple, walk,
                                          walk_at)


def _no_rife(monkeypatch, tmp_path: Path) -> None:
    """No RIFE anywhere the engine looks: no SPRITE_GEN_RIFE, no directory on PATH holding one, an empty data
    directory."""
    monkeypatch.delenv("SPRITE_GEN_RIFE", raising=False)
    path = [d for d in os.environ.get("PATH", "").split(os.pathsep) if d and not (Path(d) / "rife-ncnn-vulkan").exists()]
    monkeypatch.setenv("PATH", os.pathsep.join(path))
    monkeypatch.setenv("SPRITE_GEN_DATA_DIR", str(tmp_path / "data"))


def _run(tmp_path: Path, frames: list[Image.Image], name: str = "m", **kw) -> dict:
    keyed = save(frames, tmp_path / f"{name}-keyed")
    return loop_mod.run_loop(keyed, tmp_path / f"{name}-out", fps=24.0, min_len=None, max_len=None, n_out=None,
                             seam_max=loop_mod.SEAM_RATIO_MAX, name=name, report_path=None, **kw)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bytes(out: Path, name: str) -> dict[str, str]:
    cycle = hashlib.sha256(b"".join(p.read_bytes() for p in sorted((out / "cycle").glob("frame-*.png")))).hexdigest()
    return {"strip": _sha(out / f"{name}.strip.png"), "gif": _sha(out / f"{name}.gif"), "webp": _sha(out / f"{name}.webp"),
            "cycle": cycle}


def _meta(rep: dict) -> dict:
    return json.loads(Path(rep["strip"]["path"]).with_suffix("").with_suffix(".strip.json").read_text())


def _cycle(rep: dict) -> list[Image.Image]:
    return [Image.open(p).convert("RGBA") for p in sorted((Path(rep["out_dir"]) / "cycle").glob("frame-*.png"))]


# Pinned on the engine before the fill (886936e): a cut with no held cell is delivered as it was, byte for byte.
UNHELD = {
    "walk-fixed": (lambda: walk(), dict(state="walk", cycle_mode="fixed", start=0, length=N, anchor="none", repair="off"), {
        "strip": "2335f3f5da57e08074113d6d30b89a672a3ff1f969ff645c667e07f3d08eaba1",
        "gif": "27f05c499a83e66621c9853efdb61fb591bfa386fbceb159c0a09822f79989c9",
        "webp": "4c9922b16ff73385a87f363d7aac567b5f1885df667fafce91702935fb8451b7",
        "cycle": "26be9c07e34ad0976013e7f5ce5b40af10c071ef78fb6b749ac389deb01e4a8d"}),
    "walk-auto": (lambda: walk(N * 3 + 4), dict(state="walk", repair="off"), {
        "strip": "d9c2143deb3cb3f90d932a4cd47132dcc196a2c83227eea85588c58b3cbc2ca3",
        "gif": "1b53603c32cd6bcca445374eb9403a68d9e1914c6bdf126c49b5312fadd9d62e",
        "webp": "6ae3c4465ef484691deb63883cee6439f8e401ae6d9fc121fd4233a9d8c2215a",
        "cycle": "73de53310593c2792d4ef3b957e4f303befaeb2db5dc79c2e39649a66c0d108a"}),
    "hop-one-shot": (lambda: hop(), dict(state="jump", cycle_mode="one-shot"), {
        "strip": "4c2317151d8ac0f78b2cf05fd00f90d77ec6a84adb418e076693675b92155032",
        "gif": "084867009d141d032adea38edaec5e7277f9f7cc03d59e1498b926c5e4146546",
        "webp": "5d6b8b67c7e515096c002da2f715088b12e1b1f2174de865bda35dd4d2ab8a6f",
        "cycle": "9ccfbada05cf1b964bebea30c67cb92170713f6950dfa348a9d54afb20ddf43a"}),
    "swing-pinned": (lambda: swing(), dict(state="attack", cycle_mode="pinned", anchor="none"), {
        "strip": "74ea7ee259b254744c76ab3d2b5ca3a2a0ee8baa04d7f7beafd1ecf39966d76d",
        "gif": "3944bc371f1b167707a06c069411960d7043f40d784108cb4c231adcfee77fcd",
        "webp": "feea1f26d5dcb5eceb49706440db1c0037b5a95c692ca4ddfa58072c28e3ec70",
        "cycle": "fee8b88afd0ccbbde09c6e14411514bedf030a47dde290a9218fe9c00bbe7457"}),
    "topple-once": (lambda: topple(), dict(state="death", cycle_mode="once"), {
        "strip": "59629dc0b1ba7796e5a7650fc99696e22d84b958012e9d898bb321860118ed0e",
        "gif": "0f71d5df9da9ca3645193481c042dde32cfe6d9d344018835ee09715a08a7809",
        "webp": "6adb35ceaa211abb535e46998df6b513bfd58fa57340d47af652fd28f784a24d",
        "cycle": "6d15da447a24801ec6f4618328e18c7e75dc15a871649d47dd1889db301a8570"}),
    "swing-once-return": (lambda: swing(), dict(state="leap", cycle_mode="once-return", anchor="none"), {
        "strip": "67c113649569ee63f704ba0f4fc9c70a9c26ec50e0e5f25937ebb8c8db3d4638",
        "gif": "af0b2a32648ee4316edbabd5392ea652519a51939e67407c80650859c5fa21da",
        "webp": "ba0ad2de4a5b264ebfc61cd0657f22864340629e842abd2e1ec3929c39362ada",
        "cycle": "c0a3d63b81c139f83566e658959214f965efd136b85e810c86d78bb57551375e"}),
}


@pytest.mark.parametrize("case", sorted(UNHELD))
def test_a_cut_with_no_held_cell_is_delivered_byte_for_byte_as_before(case, tmp_path, monkeypatch):
    _no_rife(monkeypatch, tmp_path)  # nothing is made, so nothing looks for RIFE
    frames, kw, pinned = UNHELD[case]
    rep = _run(tmp_path, frames(), case, **kw)
    assert _bytes(Path(rep["out_dir"]), case) == pinned
    record = _meta(rep)["held_fill"]
    assert record == held_fill.public(rep["held_fill"])
    assert record["mode"] == "auto" and record["made"] == [] and "unmade" not in record
    cells = rep["cycle"]["length"]
    assert record["cells"] == cells
    if case == "topple-once":
        # It lies still for its last five cells: the cut's closing pose, kept.
        assert record["held_runs"] == [{"start": cells - 5, "length": 5}]
        assert record["before"]["held_runs"][0]["filled"] is False
        assert record["poses"] == cells - 4
    else:
        assert record["held_runs"] == [] and record["poses"] == cells


STALLED = {5: 2, 13: 3}  # the walk's frame 5 shown twice, frame 13 three times
STALLED_OFF = {  # the stalled walk cut on the engine before the fill (886936e)
    "strip": "f208b850b2265fd6ce10ce4993998895f70a065f5cea118ef71230581f02ad15",
    "gif": "c25a5d96b123e0237045774d689b9d2205ca51f16e67745e8ea9d290b285e3fb",
    "webp": "117d7b8b18b2e2ff42434890ca58efd0f0855c1db2f9b55515174ac481c749c7",
    "cycle": "5753f68c10942635e9518220423301f04805cbb845abaea7300ba01fb4e1c8cb",
}
STALLED_CUT = dict(state="walk", cycle_mode="fixed", start=0, length=N + 3, anchor="none", repair="off")


def test_a_stalled_walk_has_its_held_cells_made_between_the_poses_beside_them(tmp_path):
    between = Between(walk_at, range(N + 6), period=N)
    rep = _run(tmp_path, shown_again(walk(), STALLED), interpolate=between, **STALLED_CUT)
    record = rep["held_fill"]
    assert record["before"]["cells"] == N + 3 and record["before"]["poses"] == N
    assert [(r["start"], r["length"], r["filled"]) for r in record["before"]["held_runs"]] == [(5, 2, True), (14, 3, True)]
    assert record["made"] == [6, 15, 16]
    # Each made at its own time between the pose before the run and the pose after it: half way, a third and two thirds.
    assert sorted(between.calls) == [(5.0, 6.0, 0.5), (13.0, 14.0, 0.3333), (13.0, 14.0, 0.6667)]
    assert (record["cells"], record["poses"], record["held_runs"]) == (N + 3, N + 3, [])
    cells = _cycle(rep)
    assert cells[6].tobytes() == walk_at(5.5).tobytes()
    assert cells[15].tobytes() == walk_at(13 + 1 / 3).tobytes()
    # Every other cell is the cut's as filmed: the first and the last, the length, the beat.
    filmed = shown_again(walk(), STALLED)[: N + 3]
    assert len(cells) == N + 3
    assert all(cells[k].tobytes() == filmed[k].tobytes() for k in range(N + 3) if k not in (6, 15, 16))
    assert _meta(rep)["held_fill"] == held_fill.public(record)
    assert rep["seam_measurement"] == "rendered-cells"


def test_fill_held_off_cuts_the_stalled_walk_as_before_and_still_names_its_runs(tmp_path):
    rep = _run(tmp_path, shown_again(walk(), STALLED), fill_held="off", **STALLED_CUT)
    assert _bytes(Path(rep["out_dir"]), "m") == STALLED_OFF
    record = rep["held_fill"]
    assert record["mode"] == "off" and record["made"] == []
    assert [(r["start"], r["length"], r["filled"], r["why"]) for r in record["before"]["held_runs"]] == [
        (5, 2, False, "--fill-held off"), (14, 3, False, "--fill-held off")]
    assert (record["cells"], record["poses"]) == (N + 3, N)


def test_no_rife_delivers_the_held_cells_as_filmed_and_says_which_motion(tmp_path, monkeypatch, capsys):
    _no_rife(monkeypatch, tmp_path)
    rep = _run(tmp_path, shown_again(walk(), STALLED), name="stalled-walk", **STALLED_CUT)
    assert rep["status"] == "passed"
    out = Path(rep["out_dir"])
    # The images are the cut as filmed: what --fill-held off delivers.
    off = _run(tmp_path, shown_again(walk(), STALLED), name="off", fill_held="off", **STALLED_CUT)
    assert {k: v for k, v in _bytes(out, "stalled-walk").items()} == _bytes(Path(off["out_dir"]), "off")
    record = rep["held_fill"]
    assert record["made"] == [] and record["unmade"]["cells"] == [6, 15, 16]
    assert record["unmade"]["install"] == loop_mod.rife_mod.install_command()
    assert (record["cells"], record["poses"]) == (N + 3, N)
    err = capsys.readouterr().err
    assert "stalled-walk" in err and "RIFE is not installed" in err and loop_mod.rife_mod.install_command() in err


def test_a_hold_longer_than_three_cells_is_the_motion_s_own_and_kept(tmp_path):
    between = Between(walk_at, range(N + 6), period=N)
    rep = _run(tmp_path, shown_again(walk(), {9: 5}), interpolate=between,
               **{**STALLED_CUT, "length": N + 4})
    run = rep["held_fill"]["before"]["held_runs"]
    assert [(r["start"], r["length"], r["filled"]) for r in run] == [(9, 5, False)]
    assert "longer than" in run[0]["why"]
    assert between.calls == [] and rep["held_fill"]["held_runs"] == [{"start": 9, "length": 5}]


@pytest.mark.slow
def test_a_walk_drawn_on_twos_is_limited_animation_and_kept(tmp_path):
    # Each drawing shown twice, as a clip of 12 drawings a second at 24 fps is (sprite_gen/video/held.py).
    twos = shown_again(walk(), {k: 2 for k in range(N + 6)})
    cut = {**STALLED_CUT, "length": 2 * N}
    between = Between(walk_at, range(N + 6), period=N)
    rep = _run(tmp_path, twos, interpolate=between, **cut)
    record = rep["held_fill"]
    assert rep["cycle_drawings"]["hold"] == 2 and record["drawn"] == 2
    assert between.calls == [] and record["made"] == []
    assert all(not r["filled"] and "drawn on twos" in r["why"] for r in record["before"]["held_runs"])
    assert (record["cells"], record["poses"]) == (2 * N, N)
    off = _run(tmp_path, twos, name="off", fill_held="off", **cut)
    assert _bytes(Path(rep["out_dir"]), "m") == _bytes(Path(off["out_dir"]), "off")


@pytest.mark.slow
def test_a_walk_carried_to_60_fps_by_repeating_frames_reads_a_hold_and_is_filled_to_a_pose_a_cell(tmp_path):
    # 24 fps carried to 60 by the frame under each time: each frame shown three and two times by turns. Its rhythm
    # reads a hold of 2, but its runs are not each two cells: the frames between are made.
    filmed = walk(2 * N + 2)
    carried = [filmed[k * 24 // 60] for k in range(len(filmed) * 60 // 24)]
    between = Between(walk_at, range(2 * N + 2), period=N)
    keyed = save(carried, tmp_path / "carried-keyed")
    rep = loop_mod.run_loop(keyed, tmp_path / "carried-out", fps=60.0, min_len=None, max_len=None, n_out=None,
                            seam_max=loop_mod.SEAM_RATIO_MAX, name="m", report_path=None, interpolate=between,
                            state="walk", anchor="none", repair="off")
    record = rep["held_fill"]
    assert rep["cycle_drawings"]["hold"] == 2 and "drawn" not in record
    assert {r["length"] for r in record["before"]["held_runs"]} == {2, 3}
    assert record["before"]["poses"] < record["before"]["cells"] == rep["cycle"]["length"] == 60
    assert (record["cells"], record["poses"], record["held_runs"]) == (60, 60, [])
    cells, start = _cycle(rep), rep["cycle"]["start"]
    assert cells[0].tobytes() == carried[start].tobytes() and cells[-1].tobytes() == carried[start + 59].tobytes()


def test_a_once_cut_and_a_one_shot_have_their_held_cells_made_too(tmp_path):
    frames = topple()
    stalled = shown_again(frames, {20: 2})
    times = {f.tobytes(): float(k) for k, f in enumerate(frames)}
    between = Between(lambda t: _topple_at(t), [])
    between.known.update(times)
    rep = _run(tmp_path, stalled, "fall", interpolate=between, state="death", cycle_mode="once")
    record = rep["held_fill"]
    start = rep["cycle"]["start"]
    assert [(r["start"], r["filled"]) for r in record["before"]["held_runs"] if r["filled"]] == [(20 - start, True)]
    assert between.calls == [(20.0, 21.0, 0.5)]
    hopped = shown_again(hop(), {18: 2})
    rep = _run(tmp_path, hopped, "hop", interpolate=lambda a, b, t: Image.blend(a, b, t), state="jump", cycle_mode="one-shot")
    assert any(r["start"] + rep["cycle"]["start"] == 18 for r in rep["held_fill"]["before"]["held_runs"])


def _topple_at(time: float) -> Image.Image:
    """`topple`'s body `time` frames in, between its frames as well."""
    t = min(1.0, max(0.0, (time - 6) / 30))
    a = np.zeros((96, 128, 4), dtype=np.uint8)
    height, width = round(64 - 40 * t), round(16 + 36 * t)
    a[86 - height:86, 40:40 + width] = (200, 80, 30, 255)
    return Image.fromarray(a, "RGBA")


# The fill on a cut's frames, as `video-loop` calls it: which runs it makes, which it keeps and why.


def _reach(time: float) -> Image.Image:
    return planter(4 * time)


def _planted(times: list[float]) -> list[Image.Image]:
    return [_reach(t) for t in times]


def test_a_part_standing_still_on_both_sides_stays_where_it_stood():
    frames = _planted([0, 1, 2, 3, 3, 4, 5, 6])
    out, record = held_fill.fill(frames, Between(_reach, range(7)), ring=False)
    assert record["made"] == [4]
    assert out[4].tobytes() == _reach(3.5).tobytes()
    assert record["before"]["held_runs"][0]["filled"] is True
    assert record["contact"][0]["kept"] == 1.0


def test_a_frame_between_that_drags_a_planted_foot_is_not_taken():
    frames = _planted([0, 1, 2, 3, 3, 4, 5, 6])
    out, record = held_fill.fill(frames, Between(_reach, range(7), after=slid), ring=False)
    assert record["made"] == []
    assert out[4].tobytes() == frames[4].tobytes()  # the held cell, as filmed
    run = record["before"]["held_runs"][0]
    assert run["filled"] is False and "contact" in run["faults"]
    assert record["held_runs"] == [{"start": 3, "length": 2}]


def test_a_frame_between_that_cross_fades_two_poses_is_not_taken():
    # Two poses laid over each other at half strength: the double exposure an interpolator gives where it cannot follow.
    frames = shown_again(walk(), STALLED)[: N + 3]
    out, record = held_fill.fill(frames, lambda a, b, t: Image.blend(a, b, t), ring=True)
    assert record["made"] == [] and [o.tobytes() for o in out] == [f.tobytes() for f in frames]
    assert all(not r["filled"] and "ghost" in r["faults"] for r in record["before"]["held_runs"])


def test_the_opening_and_the_closing_pose_of_a_cut_played_once_are_kept():
    frames = _planted([0, 0, 1, 2, 3, 4, 4])
    calls: list = []
    out, record = held_fill.fill(frames, lambda a, b, t: calls.append(t) or a, ring=False)
    whys = [r["why"] for r in record["before"]["held_runs"]]
    assert calls == [] and record["made"] == []
    assert "opening" in whys[0] and "closing" in whys[1]


def test_a_loop_s_held_cells_at_its_wrap_are_kept_and_at_its_start_made():
    frames = _planted([0, 0, 1, 2, 3, 4, 5, 5])
    between = Between(_reach, range(7), period=None)
    # The wrap closes on cell 0 again: a frame between the last pose and the first is the wrap's, not the hold's.
    out, record = held_fill.fill(frames, between, ring=True)
    runs = record["before"]["held_runs"]
    assert (runs[0]["start"], runs[0]["filled"]) == (0, True)
    assert (runs[1]["start"], runs[1]["filled"]) == (6, False) and "wrap" in runs[1]["why"]
    assert record["made"] == [1] and out[0].tobytes() == frames[0].tobytes() and out[7].tobytes() == frames[7].tobytes()


def test_codec_copies_read_as_the_same_picture_and_the_slowest_motion_does_not():
    still = planter(10)
    noisy = np.asarray(still).astype(np.int16)
    noisy[..., :3] += np.random.default_rng(0).integers(-3, 4, size=noisy[..., :3].shape).astype(np.int16)
    noisy = Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8), "RGBA")
    assert held_fill.same_steps([still, noisy, planter(11)]) == [True, False]


def test_a_whole_figure_moving_under_a_thumbnail_pixel_is_not_the_same_picture():
    # A walker going away, a frame 280 px wide: its steps move a thousand of its pixels, and some move none of the
    # analysis thumbnail's. The frame as cut is read too: a cut of it shows no held cell.
    from tests.video.test_gait_fallback import walking_away

    frames = [walking_away(k) for k in range(73)][24:48]
    thumbs = held_fill._thumbs(frames)
    assert any(held_fill._same(a, b) for a, b in zip(thumbs, thumbs[1:]))
    assert held_fill.same_steps(frames) == [False] * 23


# A higher rate than the clip's: frames between, never frames shown again.


@pytest.mark.slow
def test_a_walk_asked_for_at_60_fps_is_made_of_frames_between(tmp_path):
    between = Between(walk_at, range(N + 6), period=N)
    rep = _run(tmp_path, walk(), "walk60", interpolate=between, out_fps=60.0,
               state="walk", cycle_mode="fixed", start=0, length=N, anchor="none", repair="off")
    meta = _meta(rep)
    record = rep["held_fill"]
    assert record["out_fps"]["applied"] is True and record["out_fps"]["cells"] == 60
    assert (record["cells"], record["poses"], record["held_runs"]) == (60, 60, [])
    assert meta["frames"] == 60 and meta["cycle_seconds"] == 1.0 and meta["delay_ms"] == pytest.approx(16.67)
    assert meta["out_fps"] == 60.0
    cells = _cycle(rep)
    assert len(cells) == 60
    assert cells[0].tobytes() == walk_at(0).tobytes() and cells[5].tobytes() == walk_at(2).tobytes()
    assert cells[1].tobytes() == walk_at(0.4).tobytes()
    assert len({c.tobytes() for c in cells}) == 60


@pytest.mark.slow
def test_a_one_shot_asked_for_at_60_fps_keeps_its_first_and_last_frame(tmp_path):
    frames = hop()
    times = {f.tobytes(): float(k) for k, f in enumerate(frames)}
    made: list = []

    def between(a, b, t):
        made.append((times[a.tobytes()], times[b.tobytes()], round(t, 4)))
        return Image.blend(a, b, t)

    rep = _run(tmp_path, frames, "hop60", interpolate=between, out_fps=60.0, state="jump", cycle_mode="one-shot")
    L = rep["cycle"]["length"]
    record = rep["held_fill"]
    cells = _cycle(rep)
    assert len(cells) == round((L - 1) * 2.5) + 1 == record["cells"]
    filmed = hop()[rep["cycle"]["start"]: rep["cycle"]["start"] + L]
    assert cells[0].tobytes() == filmed[0].tobytes() and cells[-1].tobytes() == filmed[-1].tobytes()
    assert record["out_fps"]["applied"] is True


def test_no_rife_delivers_a_clip_asked_for_at_a_higher_rate_at_its_own(tmp_path, monkeypatch, capsys):
    _no_rife(monkeypatch, tmp_path)
    rep = _run(tmp_path, walk(), "walk60", out_fps=60.0, state="walk", cycle_mode="fixed", start=0, length=N,
               anchor="none", repair="off")
    assert rep["status"] == "passed" and _meta(rep)["frames"] == N and "out_fps" not in _meta(rep)
    assert rep["held_fill"]["out_fps"]["applied"] is False
    err = capsys.readouterr().err
    assert "walk60" in err and "--out-fps 60" in err and loop_mod.rife_mod.install_command() in err


def test_a_rate_under_the_clip_s_own_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="--out-fps 12"):
        _run(tmp_path, walk(), out_fps=12.0, state="walk", cycle_mode="fixed", start=0, length=N, anchor="none")


def test_video_loop_and_video_set_take_the_switches():
    import argparse

    from sprite_gen.video import batch

    parser = argparse.ArgumentParser()
    loop_mod.add_arguments(parser)
    args = parser.parse_args(["--frames-dir", "f", "--out-dir", "o", "--fill-held", "off", "--out-fps", "60"])
    assert (args.fill_held, args.out_fps) == ("off", 60.0)
    assert parser.parse_args(["--frames-dir", "f", "--out-dir", "o"]).fill_held == "auto"
    parser = argparse.ArgumentParser()
    batch.add_arguments(parser)
    args = parser.parse_args(["--out-dir", "o", "--fill-held", "off", "--out-fps", "60"])
    assert (args.fill_held, args.out_fps) == ("off", 60.0)
    assert parser.parse_args(["--out-dir", "o"]).fill_held == "auto"


def test_video_set_hands_the_switches_to_each_cut(monkeypatch, tmp_path):
    from sprite_gen.video import batch

    seen: list[dict] = []
    monkeypatch.setattr(batch.frames_mod, "run_frames", lambda clip, out, **kw: {
        "fps": 24, "frames": 10, "alpha_zero_pct_min": 50, "alpha_zero_pct_max": 60, "keyed_dir": str(out)})

    def run_loop(*a, **kw):
        seen.append(kw)
        return {"cycle": {"length": 10, "period_global": 10, "ratio": 0.2}, "resampled_seam_ratio": 0.2, "n_out": 10,
                "gif": {"file": "loop.gif"}, "webp": {"file": "loop.webp"}, "strip": {"path": "strip.png"},
                "held_fill": {"cells": 10, "poses": 9, "held_runs": [{"start": 3, "length": 2}], "made": [],
                              "before": {"cells": 10, "poses": 9, "held_runs": [{"start": 3, "length": 2, "filled": False,
                                                                                  "why": "--fill-held off"}]}}}

    def video(image, prompt, out, report, **kw):
        out.write_bytes(b"test-video")
        report.write_text(json.dumps({"prompt": prompt}))
        return 0

    monkeypatch.setattr(batch.loop_mod, "run_loop", run_loop)
    monkeypatch.setattr(batch, "_staggered_start", lambda gap: None)
    still = tmp_path / "front.png"
    Image.new("RGBA", (64, 64), (200, 80, 30, 255)).save(still)
    result = batch.run_set(bases={"front": still}, states=["idle"], root=tmp_path / "set", character=None, duration=3,
                           resolution="720p", key="green", concurrency=1, force=False, gap=0, video_runner=video,
                           align_cycles="off", fill_held="off", out_fps=60.0)
    assert [(kw["fill_held"], kw["out_fps"]) for kw in seen] == [("off", 60.0)]
    # The set's report says each motion's poses over its cells.
    assert result["items"][0]["loop"]["poses"] == "9/10"


def test_a_cut_with_cells_made_is_not_restored_from_its_source_frames(tmp_path):
    """The source frames do not hold a cell made between two of them: restoration (`video-loop-repair`) reads the cut
    as one it cannot project (`source-processing-unsupported`), as it reads a blended wrap."""
    import shutil
    import subprocess

    from sprite_gen.video import source
    from sprite_gen.video.compare import Loop

    if not shutil.which("ffmpeg") or not loop_mod.img2webp_supports_exact():
        pytest.skip("ffmpeg and img2webp >=1.5 required")
    filmed = shown_again(walk(), STALLED)
    keyed = save(filmed, tmp_path / "keyed")
    out = tmp_path / "cut"
    loop_mod.run_loop(keyed, out, fps=24.0, min_len=None, max_len=None, n_out=None, seam_max=loop_mod.SEAM_RATIO_MAX,
                      name="walk", report_path=None, interpolate=Between(walk_at, range(N + 6), period=N), **STALLED_CUT)
    files = sorted(keyed.glob("*.png"))
    frames_report = tmp_path / "frames.json"
    frames_report.write_text(json.dumps({"kind": "sprite-gen-video-frames-report", "fps": 24.0, "frames": len(files),
                                         "width": filmed[0].width, "height": filmed[0].height, "stream_index": 0,
                                         "key": "green"}))
    clip = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-framerate", "24", "-i", str(keyed / "%04d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(clip)], check=True)
    manifest = tmp_path / "source.json"
    manifest.write_text(json.dumps(source.manifest(clip=clip, canvas=files[0], frames_report=frames_report, files=files,
                                                   timestamps=source.timestamps(clip, 0))))
    loop = Loop.read(out / "walk.strip.png", out / "walk.strip.json", out / "walk.loop.report.json")
    projection, issue = source.project(loop, source.Source.read(
        source_manifest=manifest, source_frames_dir=keyed, source_clip=clip, source_canvas=files[0],
        source_frames_report=frames_report))
    assert (projection, issue) == (None, "source-processing-unsupported")


@pytest.mark.slow
def test_the_alignment_and_a_stop_refuse_a_loop_cut_at_a_rate_of_its_own(tmp_path):
    from sprite_gen.video import align, stop

    rep = _run(tmp_path, walk(), "walk60", interpolate=Between(walk_at, range(N + 6), period=N), out_fps=60.0,
               state="walk", cycle_mode="fixed", start=0, length=N, anchor="none", repair="off")
    out = Path(rep["out_dir"])
    with pytest.raises(ValueError, match="--out-fps 60"):
        align._loop_files(out)
    with pytest.raises(SystemExit, match="--out-fps 60"):
        stop.read_loop(out / "walk60.strip.json")


def test_the_curation_view_marks_the_cells_made_and_the_cells_still_held(tmp_path):
    """A loop's `cycle/` imported into the curation view (`unpack-atlas --pngs-dir`): each card of a cell the fill made
    says `made`, of a cell still showing the picture before it `held`."""
    from sprite_gen import cli

    rep = _run(tmp_path, shown_again(walk(), {**STALLED, 20: 5}), interpolate=Between(walk_at, range(N + 6), period=N),
               **{**STALLED_CUT, "length": N + 7})
    assert rep["held_fill"]["made"] == [6, 15, 16]
    assert cli.main(["unpack-atlas", "--pngs-dir", str(Path(rep["out_dir"]) / "cycle"), "--out-dir", str(tmp_path / "run")]) == 0
    rows = json.loads((tmp_path / "run" / "frames" / "frames-manifest.json").read_text())["rows"]
    labels = rows[0]["labels"]
    assert [k for k, label in enumerate(labels) if label.endswith(" (made)")] == [6, 15, 16]
    assert [k for k, label in enumerate(labels) if label.endswith(" (held)")] == [24, 25, 26, 27]
    assert labels[0] == "frame-000"
