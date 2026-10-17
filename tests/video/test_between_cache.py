# SPDX-License-Identifier: Apache-2.0
"""Frames made between two source frames, kept and taken again (docs/loop-repair.md section 4,
`video-cycle-align --between-cache`). An app aligns a set, then aligns it once more on a copy with
the feet told (or a retake in place): the frames between are the same frames, made from the same
bytes at the same times, and the second pass takes them from the directory the first kept them in.

What must hold: the second pass delivers the very bytes a pass without the directory delivers; a
frame is taken only for the same two frames, the same fraction and the same RIFE on the same
machine, and made again for anything else; and nothing is kept unless a directory is named. The
stand-in RIFE cross-fades, names itself (`identity`) as `rife.Rife` does, and records what it made."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sprite_gen.video import align
from sprite_gen.video import rife
from tests.video.test_cycle_align import _cut
from tests.video.test_rife import _disc, _fake_binary, _legs

VIEWS = ["front", "side@right", "back"]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Blend:
    """A stand-in RIFE: the cross-fade of its two frames at t. Records each frame it made by its two
    frames' sha256 and t."""

    def __init__(self, name: str = "blend") -> None:
        self.name = name
        self.made: list[tuple[str, str, float]] = []

    def __call__(self, a: Image.Image, b: Image.Image, t: float) -> Image.Image:
        self.made.append((_sha(a.tobytes()), _sha(b.tobytes()), t))
        return Image.blend(a, b, t)

    def identity(self) -> str:
        return self.name


def _hashes(root: Path) -> dict[str, str]:
    """Every file under a directory by its sha256."""
    return {p.relative_to(root).as_posix(): _sha(p.read_bytes()) for p in sorted(root.rglob("*")) if p.is_file()}


def _said(report: dict, root: Path, as_root: Path) -> str:
    """A report as it reads with `root` named `as_root`, its cache record left out."""
    text = json.dumps({k: v for k, v in report.items() if k != "between_cache"}, sort_keys=True)
    return text.replace(str(root.resolve()), str(as_root.resolve()))


def _set(tmp_path: Path) -> tuple[Path, list[str]]:
    """Three loops of a walk cut to 20, 24 and 28 frames, as an app keeps them (`loops/<name>`)."""
    root = tmp_path / "loops"
    root.mkdir()
    for name, length in (("S", 20), ("E", 24), ("N", 28)):
        _cut(root, name, length)
    return root, ["S", "E", "N"]


def _pass(root: Path, names: list[str], made: Blend, **kw) -> dict:
    """One alignment of the set under `root`, each foot told as `feet` names it (a foot pass)."""
    feet = {str(root / n): f for n, f in (kw.pop("feet", None) or {}).items()}
    return align.align_set([root / n for n in names], interpolate=made, views=VIEWS, feet=feet or None, **kw)


@pytest.mark.parametrize("between", ["rife", "auto"])
def test_the_foot_pass_takes_every_frame_the_first_pass_made_and_delivers_the_same_bytes(tmp_path, between):
    """The app's two passes: the set aligned, then aligned again on a copy with the feet told. The
    second takes every frame between from the first's directory and makes none, and every file it
    delivers is the file a second pass with no directory delivers, byte for byte."""
    root, names = _set(tmp_path)
    kept = tmp_path / "between-cache"
    first = Blend()
    report = _pass(root, names, first, between=between, between_cache=kept)
    n = len(first.made)
    assert n > 0 and report["between_cache"] == {"dir": str(kept.resolve()), "hits": 0, "misses": n, "unreadable": 0}
    assert len(list(kept.glob("*.png"))) == n
    for label in ("foot", "plain"):
        shutil.copytree(root, tmp_path / f"loops.{label}")
    feet = {"S": "left", "E": "right", "N": "left"}
    reused, fresh = Blend(), Blend()
    taken = _pass(tmp_path / "loops.foot", names, reused, between=between, feet=feet, between_cache=kept)
    plain = _pass(tmp_path / "loops.plain", names, fresh, between=between, feet=feet)
    assert reused.made == [] and len(fresh.made) == n
    assert taken["between_cache"] == {"dir": str(kept.resolve()), "hits": n, "misses": 0, "unreadable": 0}
    assert plain["between_cache"] is None
    assert _hashes(tmp_path / "loops.foot") == _hashes(tmp_path / "loops.plain")
    assert _said(taken, tmp_path / "loops.foot", tmp_path / "loops.plain") == _said(plain, tmp_path / "loops.plain", tmp_path / "loops.plain")


def test_a_frame_whose_inputs_changed_is_made_again_and_never_taken(tmp_path):
    """A source frame changed by one pixel: every frame made from it is made again, every other is
    taken, and the loops come out as a pass with no directory makes them. Another length is other
    times: only a frame kept for the very same two frames and time is taken, and the loops are again
    the plain pass's."""
    root, names = _set(tmp_path)
    kept = tmp_path / "between-cache"
    _pass(root, names, Blend(), between="rife", between_cache=kept)
    for label in ("edit", "plain"):
        shutil.copytree(root, tmp_path / f"loops.{label}")
        path = tmp_path / f"loops.{label}" / "S" / align.SOURCE_DIR / "frame-003.png"
        frame = np.asarray(Image.open(path).convert("RGBA")).copy()
        frame[60, 50] = (255, 0, 0, 255)
        Image.fromarray(frame, "RGBA").save(path)
    changed = _sha(Image.open(tmp_path / "loops.edit" / "S" / align.SOURCE_DIR / "frame-003.png").convert("RGBA").tobytes())
    reused, fresh = Blend(), Blend()
    taken = _pass(tmp_path / "loops.edit", names, reused, between="rife", between_cache=kept)
    _pass(tmp_path / "loops.plain", names, fresh, between="rife")
    remade = [m for m in fresh.made if changed in m[:2]]
    assert remade and reused.made == remade
    assert taken["between_cache"]["misses"] == len(remade) and taken["between_cache"]["hits"] == len(fresh.made) - len(remade)
    assert _hashes(tmp_path / "loops.edit") == _hashes(tmp_path / "loops.plain")

    longer, plain = Blend(), Blend()
    taken = _pass(tmp_path / "loops.edit", names, longer, between="rife", length=22, between_cache=kept)
    _pass(tmp_path / "loops.plain", names, plain, between="rife", length=22)
    assert longer.made == [m for m in plain.made if m not in set(fresh.made)] and longer.made
    assert taken["between_cache"]["hits"] == len(plain.made) - len(longer.made)
    assert _hashes(tmp_path / "loops.edit") == _hashes(tmp_path / "loops.plain")


def _stub_call(monkeypatch) -> list[int]:
    """RIFE's binary call answered by a cross-fade of the two images it is handed."""
    calls: list[int] = []

    def blend(binary, model, x, y, t, tmp):
        calls.append(1)
        return Image.blend(x, y, t)

    monkeypatch.setattr(rife, "_call", blend)
    return calls


def test_a_kept_frame_is_taken_only_for_the_same_frames_fraction_rife_and_machine(monkeypatch, tmp_path):
    calls = _stub_call(monkeypatch)
    binary = _fake_binary(tmp_path / "one")
    made = rife.Rife(binary, binary.parent / rife.MODEL)
    cache = rife.Cache(tmp_path / "kept")
    a, b = _disc(40), _disc(72)
    first = cache.frame(made, a, b, 0.5)
    assert len(calls) == 2 and made.made == 1
    again = cache.frame(made, a, b, 0.5)
    assert len(calls) == 2 and made.made == 1
    assert (again.mode, again.size, again.tobytes()) == (first.mode, first.size, first.tobytes())
    assert (cache.hits, cache.misses, cache.unreadable) == (1, 1, 0)

    moved = np.asarray(a).copy()
    moved[0, 0] = (1, 0, 0, 1)
    others = [(b, a, 0.5), (a, b, 0.25), (a, b, 0.5000001), (Image.fromarray(moved, "RGBA"), b, 0.5)]
    for x, y, t in others:
        assert cache.frame(made, x, y, t).tobytes() == rife.between(x, y, t, binary=binary, model=made.model, tmp=tmp_path).tobytes()
    assert cache.misses == 1 + len(others) and cache.hits == 1

    def remade(changed) -> bool:
        before = cache.misses
        changed()
        cache.frame(rife.Rife(binary, binary.parent / rife.MODEL), a, b, 0.5)
        return cache.misses == before + 1

    assert remade(lambda: (binary.parent / rife.MODEL / "flownet.bin").write_bytes(b"another model"))
    assert remade(lambda: binary.write_text("#!/bin/sh\nexit 0  # another build\n"))
    assert remade(lambda: monkeypatch.setattr(rife.platform, "node", lambda: "another-machine"))
    code = tmp_path / "rife.py"
    code.write_text(Path(rife.__file__).read_text(encoding="utf-8") + "\n# another engine\n", encoding="utf-8")
    assert remade(lambda: monkeypatch.setattr(rife, "__file__", str(code)))
    assert cache.record() == {"dir": str((tmp_path / "kept").resolve()), "hits": 1, "misses": 1 + len(others) + 4, "unreadable": 0}
    if os.name == "posix":  # the directory and its frames are this user's only
        assert (tmp_path / "kept").stat().st_mode & 0o077 == 0
        assert all(p.stat().st_mode & 0o077 == 0 for p in (tmp_path / "kept").iterdir())


def test_a_kept_file_that_does_not_read_back_is_made_again(monkeypatch, tmp_path):
    """A kept file cut short, overwritten with another image — even an RGBA frame of the same size —
    or not an image at all is not taken: the frame is made again and kept over it."""
    _stub_call(monkeypatch)
    binary = _fake_binary(tmp_path)
    made = rife.Rife(binary, binary.parent / rife.MODEL)
    cache = rife.Cache(tmp_path / "kept")
    a, b = _legs(40, 64), _legs(64, 52)
    truth = cache.frame(made, a, b, 0.5).tobytes()
    (path,) = (tmp_path / "kept").glob("*.png")
    whole = path.read_bytes()
    broken = [whole[: len(whole) // 2], b"not a png", _png(Image.new("RGBA", (8, 8))), _png(Image.new("RGB", a.size)),
              _png(Image.new("RGBA", a.size))]
    for data in broken:
        path.write_bytes(data)
        assert cache.frame(made, a, b, 0.5).tobytes() == truth
        assert path.read_bytes() == whole
    assert (cache.hits, cache.misses, cache.unreadable) == (0, 1 + len(broken), len(broken))
    assert cache.frame(made, a, b, 0.5).tobytes() == truth and cache.hits == 1


def _png(image: Image.Image) -> bytes:
    from io import BytesIO

    out = BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def test_nothing_is_kept_without_a_directory_and_a_stand_in_that_names_nothing_is_refused(tmp_path):
    root, names = _set(tmp_path)
    before = sorted(p.name for p in tmp_path.iterdir())
    report = _pass(root, names, Blend(), between="rife")
    assert report["between_cache"] is None and sorted(p.name for p in tmp_path.iterdir()) == before
    files = _hashes(root)

    def anonymous(a: Image.Image, b: Image.Image, t: float) -> Image.Image:
        return Image.blend(a, b, t)

    with pytest.raises(SystemExit, match=r"--between-cache .*names no identity"):
        align.align_set([root / n for n in names], interpolate=anonymous, between="rife", between_cache=tmp_path / "kept")
    assert _hashes(root) == files and not (tmp_path / "kept").exists()


def test_video_cycle_align_between_cache_reports_hits_and_misses(tmp_path, monkeypatch, capsys):
    """The command line: `--between-cache` names the directory; the summary and the report carry the
    hits and misses. RIFE is the stand-in, found as `rife.Rife` is."""

    class Found(Blend):
        def describe(self) -> dict[str, str]:
            return {"binary": "stand-in", "model": "blend"}

    monkeypatch.setattr(align.rife_mod, "Rife", Found)
    root, names = _set(tmp_path)
    argv = [*(a for n in names for a in ("--loop-dir", str(root / n))), "--between", "rife", "--between-cache", str(tmp_path / "kept")]
    for k, label in enumerate(("first", "again")):
        assert align.main([*argv, "--report", str(tmp_path / f"{label}.json")]) == 0
        said = json.loads(capsys.readouterr().out)["between_cache"]
        written = json.loads((tmp_path / f"{label}.json").read_text())["between_cache"]
        assert said == written and said["dir"] == str((tmp_path / "kept").resolve())
        made = said["hits"] + said["misses"]
        assert made > 0 and said["hits"] == (made if k else 0)


@pytest.mark.real_rife
def test_real_rife_a_kept_frame_is_the_frame_made_again(tmp_path):
    """The real binary makes the same bytes for the same frames, so a frame taken from the
    directory is the frame a pass without it makes."""
    made = rife.Rife()
    cache = rife.Cache(tmp_path / "kept")
    a, b = _legs(36, 70), _legs(60, 46)
    kept = cache.frame(made, a, b, 0.5).tobytes()
    taken = cache.frame(made, a, b, 0.5).tobytes()
    fresh = made(a, b, 0.5).tobytes()
    assert (cache.hits, cache.misses) == (1, 1) and made.made == 2
    assert _sha(taken) == _sha(kept) == _sha(fresh)
