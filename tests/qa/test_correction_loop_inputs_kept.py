# SPDX-License-Identifier: Apache-2.0
"""`correction-loop` 이 읽는 런을 지키는 계약 — 출력 폴더가 런과 겹치는 배치.

지키는 것 (구현이 아니라 계약):
  1. 런 폴더(`--run-dir`)와 최선 후보의 복사가 따라가는 링크 하나하나(그 경로가 지나는 링크와 폴더까지)가
     `--out-dir` 와 같거나 그 안에 들면 아무것도 만들기 전에 거절한다 — 루프는 거기에 시도와 최선 후보를 쓰고
     옛 최선 후보를 지운다
  2. `--out-dir` 가 런 폴더 안이면(기본값 `<run-dir>/correction-loop`) 루프는 거기에 더할 뿐이고, 최선 후보의
     복사는 그 폴더를 빼고 런만 복사한다 — 제 출력을 제 안에 복사하지 않는다
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.video.test_loop_inputs_kept import _digest, _state
from sprite_gen.qa.correction_loop import run_loop


def _loop(run_dir: Path, out_dir: Path | None) -> str:
    """dry-run 한 번 — 끝난 모양을 한 줄로(끝나면 rc=0 과 최선 후보, 거절·실패면 그 문구)."""
    try:
        result = run_loop(run_dir, states="walk", out_dir=out_dir, dry_run=True)
    except (SystemExit, OSError) as stopped:
        return f"{type(stopped).__name__}: {str(stopped)[:300]}"
    return f"rc=0, best {result['preserved_best']}"


def _readable(folder: Path) -> dict[str, str]:
    """런 폴더에서 열리는 모든 파일 — 상대 경로 → 바이트의 sha256(링크는 따라가 읽는다)."""
    return {p.relative_to(folder).as_posix(): _digest(p.read_bytes()) for p in sorted(folder.rglob("*")) if p.is_file()}


# ── 1. 출력이 런과 겹치는 배치 — 만들기 전에 거절 ──────────────────────────────

def _run_is_the_best_candidate(root: Path, run: Path):
    (root / "loop").mkdir()
    run.rename(root / "loop" / "best-candidate")
    return root / "loop" / "best-candidate", root / "loop", r"--run-dir lies inside --out-dir"


def _run_is_the_out_dir(root: Path, run: Path):
    return run, run, r"--run-dir and --out-dir are the same folder"


def _a_link_in_the_run_into_the_best_candidate(root: Path, run: Path):
    (root / "loop" / "best-candidate").mkdir(parents=True)
    (root / "loop" / "best-candidate" / "notes.txt").write_bytes(b"what the run's link reads\n")
    (run / "notes.txt").symlink_to(root / "loop" / "best-candidate" / "notes.txt")
    return run, root / "loop", r"notes\.txt in --run-dir lies inside --out-dir"


OVERLAPS = [_run_is_the_best_candidate, _run_is_the_out_dir, _a_link_in_the_run_into_the_best_candidate]


@pytest.mark.parametrize("layout", OVERLAPS, ids=[layout.__name__.strip("_") for layout in OVERLAPS])
def test_the_loop_never_writes_over_or_deletes_the_run_it_reads(tmp_path, fixture_run_dir, layout) -> None:
    run_dir, out_dir, why = layout(tmp_path, fixture_run_dir)
    read, before = _readable(run_dir), _state(tmp_path)
    said = _loop(run_dir, out_dir)
    assert _readable(run_dir) == read, f"correction-loop 이 읽은 런을 지우거나 바꿨다 ({said})"
    assert re.search(why, said), f"런과 겹치는 출력을 거절하지 않았다 ({said})"
    assert _state(tmp_path) == before, f"거절했는데 무엇이 생기거나 바뀌었다 ({said})"


# ── 2. 기본 out 은 런 안 — 더할 뿐, 복사는 그 폴더를 뺀다 ──────────────────────────

def test_the_default_out_dir_inside_the_run_is_left_out_of_the_best_candidates_copy(fixture_run_dir) -> None:
    read = _readable(fixture_run_dir)
    said = _loop(fixture_run_dir, None)
    assert said.startswith("rc=0"), f"기본 out(런 안)으로 돌지 못했다 ({said})"
    best = fixture_run_dir / "correction-loop" / "best-candidate"
    assert _readable(best) == read, "최선 후보가 런 그대로가 아니다 — 제 출력을 제 안에 복사했다"
    kept = {name: data for name, data in _readable(fixture_run_dir).items() if not name.startswith("correction-loop/")}
    assert kept == read, "런의 원래 파일이 바뀌었다"
