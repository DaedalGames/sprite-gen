# SPDX-License-Identifier: Apache-2.0
"""`sprite-gen unpack-atlas` 가 입력을 지키는 계약 — 런 폴더를 만들고 `--force` 로 덮어쓰는 길.

지키는 것 (구현이 아니라 계약):
  1. 입력(`--pngs-dir` 폴더, `--atlas`·`--manifest` 와 그 폴더의 다른 파일)은 한 바이트도 바뀌지 않는다 —
     입력이 옛 임시 폴더 이름(`.<out>.sg-staging`)에 있어도, 링크로 그 이름에 걸려 있어도
  2. 지우는 것은 이 실행이 만든 것과 `--force` 가 갈아 끼우는 런 폴더의 옛 내용뿐이다 — out 옆에 남이 둔
     `.<out>.sg-staging` 은 그대로 남는다
  3. 입력과 out 이 같거나 한쪽이 다른 쪽 안에 들면 `--force` 여도 아무것도 만들기 전에 거절한다 — 실경로와
     폴더 동일성 기준이라 링크로 돌아 들어와도, 대소문자만 다른 철자여도 같은 폴더로 읽는다. 매니페스트가
     가리키는 아틀라스도 입력이다
  4. 내놓다가 못 끝내면 옛 런은 한 바이트도 바뀌지 않고, 이 실행의 임시 폴더만 치운다
  5. out 이 빈 폴더로 가는 링크면 링크가 가리키는 자리에 내놓고 링크는 그대로 둔다
  6. 내놓은 런은 런 폴더의 이름을 단다 — 캐릭터 id 와 매니페스트의 run_dir 은 임시 폴더가 아니라 런 폴더다
"""

from __future__ import annotations

import json
import os
import pathlib
from pathlib import Path

import pytest
from PIL import Image

FRAMES = ["1-a.png", "2-b.png"]
NOTE = b"source notes, not a frame\n"
CELL = 8
RWLOCK = ".run.sg-rwlock"   # 런 폴더 옆 읽기·쓰기 락(runio.publish_guard) — 런 폴더의 영구 동반 파일이다


def _cell(i: int) -> Image.Image:
    return Image.new("RGBA", (CELL, CELL), (40 + 90 * i, 80, 160, 255))


def _pngs(folder: Path, given: Path | None = None) -> list[str]:
    """PNG 둘과 PNG 가 아닌 파일 하나가 든 입력 폴더 — `--pngs-dir` 인자."""
    folder.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(FRAMES):
        _cell(i).save(folder / name)
    (folder / "notes.txt").write_bytes(NOTE)
    return ["--pngs-dir", str(given or folder)]


def _sheet(folder: Path, given: Path | None = None) -> list[str]:
    """두 칸 아틀라스와 PNG 가 아닌 파일 하나가 든 폴더 — `--atlas … --grid 2x1` 인자."""
    folder.mkdir(parents=True, exist_ok=True)
    sheet = Image.new("RGBA", (2 * CELL, CELL), (0, 0, 0, 0))
    for i in range(2):
        sheet.paste(_cell(i), (i * CELL, 0))
    sheet.save(folder / "sheet.png")
    (folder / "notes.txt").write_bytes(NOTE)
    return ["--atlas", str((given or folder) / "sheet.png"), "--grid", "2x1"]


def _manifest(path: Path, atlas: str) -> list[str]:
    rects = [{"x": i * CELL, "y": 0, "w": CELL, "h": CELL} for i in range(2)]
    path.write_text(json.dumps({"cell": {"width": CELL, "height": CELL}, "game_input": atlas,
                                "frame_layout": {"rows": {"idle": rects}}}), encoding="utf-8")
    return ["--manifest", str(path)]


SOURCES = {"pngs": _pngs, "atlas": _sheet}


def _tree(folder: Path) -> dict[str, object]:
    """폴더 안 모든 것 — 상대 경로 → 파일 바이트 · "dir" · 링크 대상. 링크 안으로는 들어가지 않는다."""
    got: dict[str, object] = {}
    for root, dirs, files in os.walk(folder):
        for name in dirs + files:
            path = Path(root) / name
            got[path.relative_to(folder).as_posix()] = (f"link:{os.readlink(path)}" if path.is_symlink()
                                                        else "dir" if path.is_dir() else path.read_bytes())
    return got


def _names(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir())


def _unpack(*argv: str) -> int:
    from sprite_gen import cli

    return cli.main(["unpack-atlas", *argv])


def _published(run: Path) -> None:
    assert (run / "sprite-request.json").is_file() and (run / "unpack-source.json").is_file()
    assert sorted(p.name for p in (run / "frames").rglob("frame-*.png")) == ["frame-0.png", "frame-1.png"]


# ── 1·2. 입력과 남의 폴더는 그대로 ──────────────────────────────────

@pytest.mark.parametrize("kind", SOURCES)
def test_a_source_kept_under_the_old_staging_name_is_kept(tmp_path, kind) -> None:
    src = tmp_path / ".run.sg-staging"
    argv = SOURCES[kind](src)
    before = _tree(src)
    assert _unpack(*argv, "--out-dir", str(tmp_path / "run")) == 0
    assert _tree(src) == before, "입력 폴더(입력과 그 옆 파일)가 바뀌었다"
    _published(tmp_path / "run")
    assert _names(tmp_path) == sorted([".run.sg-staging", RWLOCK, "run"]), "이 실행의 임시 폴더가 남았다"


@pytest.mark.parametrize("kind", SOURCES)
def test_a_source_linked_under_the_old_staging_name_is_kept(tmp_path, kind) -> None:
    real = tmp_path / "source"
    link = tmp_path / ".run.sg-staging"
    argv = SOURCES[kind](real, given=link)
    link.symlink_to(real, target_is_directory=True)
    before = _tree(real)
    assert _unpack(*argv, "--out-dir", str(tmp_path / "run")) == 0
    assert _tree(real) == before, "링크 너머의 입력이 바뀌었다"
    assert link.is_symlink() and link.resolve() == real.resolve()
    _published(tmp_path / "run")


def test_a_staging_folder_someone_else_left_is_not_touched(tmp_path) -> None:
    argv = _pngs(tmp_path / "pngs")
    stale = tmp_path / ".run.sg-staging"
    stale.mkdir()
    (stale / "keep.txt").write_bytes(b"not this run's\n")
    assert _unpack(*argv, "--out-dir", str(tmp_path / "run")) == 0
    assert _tree(stale) == {"keep.txt": b"not this run's\n"}, "이 실행이 만들지 않은 폴더를 지웠다"
    assert _names(tmp_path) == sorted([".run.sg-staging", RWLOCK, "pngs", "run"])


# ── 3. 겹치는 입력은 --force 여도 아무것도 만들기 전에 거절 ───────────

def _prior(run: Path) -> None:
    """--force 가 갈아 끼울 옛 런 폴더."""
    run.mkdir(parents=True, exist_ok=True)
    (run / "old.txt").write_bytes(b"the prior run\n")


def _pngs_is_out(root: Path) -> list[str]:
    return [*_pngs(root / "pngs"), "--out-dir", str(root / "pngs")]


def _out_inside_pngs(root: Path) -> list[str]:
    return [*_pngs(root / "pngs"), "--out-dir", str(root / "pngs" / "run")]


def _pngs_inside_out(root: Path) -> list[str]:
    _prior(root / "run")
    return [*_pngs(root / "run" / "pngs"), "--out-dir", str(root / "run")]


def _out_links_to_pngs(root: Path) -> list[str]:
    argv = _pngs(root / "pngs")
    (root / "run").symlink_to(root / "pngs", target_is_directory=True)
    return [*argv, "--out-dir", str(root / "run")]


def _out_under_a_link_into_pngs(root: Path) -> list[str]:
    argv = _pngs(root / "pngs")
    (root / "alias").symlink_to(root / "pngs", target_is_directory=True)
    return [*argv, "--out-dir", str(root / "alias" / "run")]


def _pngs_through_a_link_inside_out(root: Path) -> list[str]:
    _prior(root / "run")
    argv = _pngs(root / "run" / "pngs", given=root / "in")
    (root / "in").symlink_to(root / "run" / "pngs", target_is_directory=True)
    return [*argv, "--out-dir", str(root / "run")]


def _out_inside_pngs_spelt_in_capitals(root: Path) -> list[str]:
    argv = _pngs(root / "pngs")
    if not (root / "PNGS").exists():
        pytest.skip("대소문자를 가리는 볼륨 — 철자가 다르면 다른 폴더다")
    return [*argv, "--out-dir", str(root / "PNGS" / "run")]


def _atlas_inside_out(root: Path) -> list[str]:
    _prior(root / "run")
    return [*_sheet(root / "run"), "--out-dir", str(root / "run")]


def _atlas_through_a_link_inside_out(root: Path) -> list[str]:
    _prior(root / "run")
    _sheet(root / "run")
    (root / "sheet.png").symlink_to(root / "run" / "sheet.png")
    return ["--atlas", str(root / "sheet.png"), "--grid", "2x1", "--out-dir", str(root / "run")]


def _out_is_the_atlas(root: Path) -> list[str]:
    return [*_sheet(root / "sheets"), "--out-dir", str(root / "sheets" / "sheet.png")]


def _manifest_inside_out(root: Path) -> list[str]:
    _prior(root / "run")
    _sheet(root / "run")
    return [*_manifest(root / "run" / "sheet.json", "sheet.png"), "--out-dir", str(root / "run")]


def _manifest_names_an_atlas_inside_out(root: Path) -> list[str]:
    _prior(root / "run")
    _sheet(root / "run")
    return [*_manifest(root / "sheet.json", "run/sheet.png"), "--out-dir", str(root / "run")]


OVERLAPS = [
    (_pngs_is_out, "--pngs-dir and --out-dir are the same"),
    (_out_inside_pngs, "--out-dir lies inside --pngs-dir"),
    (_pngs_inside_out, "--pngs-dir lies inside --out-dir"),
    (_out_links_to_pngs, "--pngs-dir and --out-dir are the same"),
    (_out_under_a_link_into_pngs, "--out-dir lies inside --pngs-dir"),
    (_pngs_through_a_link_inside_out, "--pngs-dir lies inside --out-dir"),
    (_out_inside_pngs_spelt_in_capitals, "--out-dir lies inside --pngs-dir"),
    (_atlas_inside_out, "--atlas lies inside --out-dir"),
    (_atlas_through_a_link_inside_out, "--atlas lies inside --out-dir"),
    (_out_is_the_atlas, "--atlas and --out-dir are the same"),
    (_manifest_inside_out, "--manifest lies inside --out-dir"),
    (_manifest_names_an_atlas_inside_out, "the atlas the manifest names lies inside --out-dir"),
]


@pytest.mark.parametrize("layout,why", OVERLAPS, ids=[layout.__name__.strip("_") for layout, _ in OVERLAPS])
def test_overlapping_inputs_are_refused_before_anything_is_made(tmp_path, layout, why) -> None:
    argv = layout(tmp_path)
    before = _tree(tmp_path)
    with pytest.raises(SystemExit, match=why):
        _unpack(*argv, "--force")
    assert _tree(tmp_path) == before, "거절했는데 무엇이 생기거나 바뀌었다"


# ── 4. 못 끝낸 내놓기 ───────────────────────────────────────────────

def test_a_force_reimport_that_cannot_publish_leaves_the_whole_prior_run(tmp_path, monkeypatch) -> None:
    argv = _pngs(tmp_path / "pngs")
    run = tmp_path / "run"
    assert _unpack(*argv, "--out-dir", str(run)) == 0
    (run / ".sg-backup").mkdir()                                   # 옛 런에 든, 이 실행이 만들지 않은 폴더
    (run / ".sg-backup" / "keep.txt").write_bytes(b"not this run's\n")
    before = _tree(run)
    _cell(1).save(tmp_path / "pngs" / "3-c.png")                   # 다시 들일 새 내용
    rename = pathlib.Path.rename
    failed = []

    def the_first_move_into_the_run_fails(self, target):
        if Path(target).parent == run.resolve() and not failed:  # 새 내용을 런 폴더로 옮기는 첫 걸음
            failed.append(self)
            raise OSError("injected publish failure")
        return rename(self, target)

    monkeypatch.setattr(pathlib.Path, "rename", the_first_move_into_the_run_fails)
    with pytest.raises(OSError, match="injected"):
        _unpack(*argv, "--out-dir", str(run), "--force")
    assert failed, "내놓기까지 가지 않았다"
    assert _tree(run) == before, "옛 런이 바뀌었다"
    assert _names(tmp_path) == sorted([RWLOCK, "pngs", "run"]), "이 실행의 임시 폴더가 남았다"


# ── 5. 링크 out 은 가리키는 자리에 ───────────────────────────────────

def test_an_output_link_to_an_empty_folder_publishes_where_it_points(tmp_path) -> None:
    argv = _pngs(tmp_path / "pngs")
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "run"
    link.symlink_to(target, target_is_directory=True)
    assert _unpack(*argv, "--out-dir", str(link)) == 0
    assert link.is_symlink() and link.resolve() == target.resolve(), "out 링크가 바뀌었다"
    _published(target)
    assert _names(tmp_path) == sorted([".target.sg-rwlock", "pngs", "run", "target"]), "임시 폴더가 남았다"


# ── 6. 런은 런 폴더의 이름을 단다 ─────────────────────────────────────

@pytest.mark.parametrize("kind", SOURCES)
def test_a_run_is_named_after_its_run_dir_not_its_staging_folder(tmp_path, kind) -> None:
    argv = SOURCES[kind](tmp_path / "source")
    run = tmp_path / "hero-curator"
    assert _unpack(*argv, "--out-dir", str(run)) == 0
    request = json.loads((run / "sprite-request.json").read_text(encoding="utf-8"))
    manifest = json.loads((run / "frames" / "frames-manifest.json").read_text(encoding="utf-8"))
    assert request["character"]["id"] == "hero-curator", "임시 폴더 이름이 캐릭터 id 가 됐다"
    assert manifest["run_dir"] == str(run.resolve()), "매니페스트가 지워진 임시 폴더를 가리킨다"
