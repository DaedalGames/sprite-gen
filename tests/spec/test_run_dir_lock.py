# SPDX-License-Identifier: Apache-2.0
"""런 폴더 단일 writer 락(`runio.acquire_run_dir_lock`)은 락 이름의 항목 그 자체다.

지키는 것 (구현이 아니라 계약):
  1. 락 이름(`.sprite-gen.lock`) 자리가 링크(닿는 곳이 오래된 파일이든 없는 이름이든)나 폴더면, 락을 잡는 동사는
     그 항목을 대며 곧바로 거절하고 링크를 따라가 읽거나 만들거나 지우지 않는다 — 락을 회수해도 링크 너머의
     파일은 지워지지 않는다 (unpack-atlas 의 배치는 tests/frames/test_unpack_atlas_publish.py 7)
  2. 주인을 확인할 수 없는 오래된 락 파일은 회수한다
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from sprite_gen import cli
from sprite_gen.spec import runio


def _tree(folder: Path) -> dict[str, object]:
    """폴더 안 모든 것 — 상대 경로 → 파일 바이트 · "dir" · 링크 대상. 링크 안으로는 들어가지 않는다."""
    got: dict[str, object] = {}
    for root, dirs, files in os.walk(folder):
        for name in dirs + files:
            path = Path(root) / name
            got[path.relative_to(folder).as_posix()] = (f"link:{os.readlink(path)}" if path.is_symlink()
                                                        else "dir" if path.is_dir() else path.read_bytes())
    return got


def _aged(path: Path) -> None:
    """한 시간 전에 바뀐 것 — 락으로 읽히면 주인을 확인할 수 없는 오래된 락이라 회수 대상이다."""
    old = time.time() - 3600
    os.utime(path, (old, old))


@pytest.mark.timeout(10)  # 거절은 곧바로다 — 회수를 되풀이하며 멈추면 이름 붙은 실패가 된다
@pytest.mark.parametrize("entry", ["link-to-an-old-file", "link-to-no-file", "old-folder"])
def test_a_writer_refuses_a_lock_name_that_is_not_a_lock_file_and_never_follows_it(tmp_path, entry) -> None:
    run, outside = tmp_path / "run", tmp_path / "outside"
    run.mkdir()
    outside.mkdir()
    lock = run / runio.LOCK_FILENAME
    if entry == "old-folder":
        lock.mkdir()
        _aged(lock)
    else:
        if entry == "link-to-an-old-file":
            (outside / "keep.txt").write_bytes(b"not a lock\n")
            _aged(outside / "keep.txt")
        lock.symlink_to("../outside/keep.txt")
    before = _tree(tmp_path)
    with pytest.raises(SystemExit, match=r"run[/\\]\.sprite-gen\.lock is not a lock file"):
        cli.main(["export-pngs", "--run-dir", str(run)])
    assert _tree(tmp_path) == before, "락 이름 자리를 따라가 무엇을 지우거나 만들었다"


def test_an_old_lock_file_whose_holder_cannot_be_verified_is_reclaimed(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    lock = run / runio.LOCK_FILENAME
    lock.write_text(json.dumps({"owner": "a writer long gone"}), encoding="utf-8")  # pid 가 없다 — 주인을 확인할 수 없다
    _aged(lock)
    try:
        runio.acquire_run_dir_lock(run, "reclaimer")
        assert json.loads(lock.read_text(encoding="utf-8"))["owner"] == "reclaimer", "오래된 락을 회수하지 않았다"
    finally:
        runio.release_run_dir_lock(run)
    assert not lock.exists()
