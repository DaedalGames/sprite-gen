# SPDX-License-Identifier: Apache-2.0
"""런 폴더 단일 writer 락(`runio.acquire_run_dir_lock`)은 락 이름의 항목 그 자체다.

지키는 것 (구현이 아니라 계약):
  1. 락 이름(`.sprite-gen.lock`) 자리가 링크(닿는 곳이 오래된 파일이든 없는 이름이든)나 폴더면, 락을 잡는 동사는
     그 항목을 대며 곧바로 거절하고 링크를 따라가 읽거나 만들거나 지우지 않는다 — 락을 회수해도 링크 너머의
     파일은 지워지지 않는다 (unpack-atlas 의 배치는 tests/frames/test_unpack_atlas_publish.py 7)
  2. 주인을 확인할 수 없는 오래된 락 파일은 회수한다
  3. 한 프로세스는 자기가 만든 바로 그 락 항목만, 아직 쥐고 있을 때만 지운다 — 명시 해제한 writer 가 끝날 때 그 뒤
     이름에 들어선 것(다음 writer 의 살아 있는 락, 그 자리에 놓인 링크)은 그대로다. 쥔 채 끝나면(해제하지 않았거나,
     해제한 뒤 다시 잡았거나) 그 락을 지운다
  4. 죽은 주인의 락을 회수하는 일과 새 락을 만드는 일은 줄을 선다 — 늦은 회수자가 옛 락을 판정한 동안 다른 writer 는
     기다리고, 회수자가 그 판정한 옛 락만 지우고 잡은 뒤에야 차례가 와 그 살아 있는 락을 대며 거절된다. 두 writer 가
     함께 쥐지 않는다

3·4 는 실제 프로세스로 잰다 — 락은 프로세스 사이의 계약이고, 종료 처리는 프로세스가 끝날 때만 돈다.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from sprite_gen import cli
from sprite_gen.spec import runio

from conftest import symlink_or_skip

# 차례대로 락을 잡고(owner 이름) 놓고(release) 표준 입력 한 줄을 기다린다(wait). 먼저 자기 pid, 잡기 전에 trying, 한 걸음마다
# 그 이름을 찍는다.
_STEPS = r"""
import os
import sys
from pathlib import Path
from sprite_gen.spec import runio
print(os.getpid(), flush=True)
run = Path(sys.argv[1])
for step in sys.argv[2:]:
    if step == "wait":
        sys.stdin.readline()
        continue
    if step == "release":
        runio.release_run_dir_lock(run)
    else:
        print("trying", flush=True)
        runio.acquire_run_dir_lock(run, step)
    print(step, flush=True)
"""

# 옛 락의 주인이 죽었다고 판정한 바로 그 자리에서 멈추는 회수자 — runio._pid_alive 를 감싸 진짜 답을 돌려주기 전에 표준 입력
# 한 줄을 기다린다. 스케줄러가 끼어들 수 있는 자리이고, 멈춤은 그 사이를 매번 같게 만든다. 제품 코드는 그대로다.
_SLOW_RECLAIMER = r"""
import os
import sys
from pathlib import Path
from sprite_gen.spec import runio
print(os.getpid(), flush=True)
judge = runio._pid_alive
paused = []
def judged(pid):
    alive = judge(pid)
    if not alive and not paused:
        paused.append(pid)
        print("judged-stale", flush=True)
        sys.stdin.readline()
    return alive
runio._pid_alive = judged
runio.acquire_run_dir_lock(Path(sys.argv[1]), "slow-reclaimer")
print("slow-reclaimer", flush=True)
sys.stdin.readline()
"""


class _Child:
    """실제 프로세스 하나 — 표준 출력을 한 줄씩 받아, 한 줄을 얼마나 기다릴지 정할 수 있다. `pid` 는 자식이 처음 찍는 자기
    pid 다(Windows venv 의 python.exe 는 진짜 인터프리터를 띄우는 런처라 Popen.pid 와 다르다)."""

    def __init__(self, code: str, *args: object) -> None:
        self.process = subprocess.Popen([sys.executable, "-c", code, *map(str, args)], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._err: list[str] = []
        self._out_pump = threading.Thread(target=self._pump_out, daemon=True)
        self._err_pump = threading.Thread(target=self._err.extend, args=(self.process.stderr,), daemon=True)
        self._out_pump.start()
        self._err_pump.start()
        said = self.line()
        assert said is not None and said.isdigit(), f"child did not start: {said}\n{self.stderr()}"
        self.pid = int(said)

    def _pump_out(self) -> None:
        for line in self.process.stdout:
            self._lines.put(line.strip())
        self._lines.put(None)

    def line(self, timeout: float = 10.0) -> str | None:
        """다음 줄 — 끝났으면 None, `timeout` 초 안에 없으면 "no answer"."""
        try:
            return self._lines.get(timeout=timeout)
        except queue.Empty:
            return "no answer"

    def go_on(self) -> None:
        """기다리는 한 걸음을 넘긴다."""
        with contextlib.suppress(OSError, ValueError):  # 이미 끝났거나 표준 입력을 닫았다
            self.process.stdin.write("\n")
            self.process.stdin.flush()

    def finish(self, timeout: float = 10.0) -> int:
        """남은 기다림을 모두 넘기고 끝날 때까지 기다린다 — 종료 코드."""
        with contextlib.suppress(OSError, ValueError):
            self.process.stdin.write("\n" * 8)
            self.process.stdin.close()
        try:
            return self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()
            return self.process.wait()

    def stderr(self) -> str:
        if self.process.poll() is not None:
            self._err_pump.join(5)
        return "".join(self._err)

    def close(self) -> None:
        """끝까지 기다리고 출력 파이프를 닫는다."""
        self.finish()
        for pump, pipe in ((self._out_pump, self.process.stdout), (self._err_pump, self.process.stderr)):
            pump.join(5)
            pipe.close()


@pytest.fixture
def start():
    """`start(code, *args)` 로 띄운 프로세스는 시험이 어떻게 끝나든 끝까지 기다린다."""
    started: list[_Child] = []

    def start(code: str, *args: object) -> _Child:
        started.append(_Child(code, *args))
        return started[-1]

    yield start
    for child in started:
        child.close()


def _dead_pid() -> int:
    """방금 끝난 프로세스의 pid — 죽은 주인의 자리."""
    gone = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True,
                          encoding="utf-8", check=True)
    pid = int(gone.stdout)
    assert not runio._pid_alive(pid), f"pid {pid} is in use again"
    return pid


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


@pytest.mark.parametrize("after", ["live-writer-lock", "link"])
def test_a_writer_that_released_its_lock_leaves_what_took_the_name_after_it_when_it_exits(tmp_path, start, after) -> None:
    """명시 해제한 writer 가 끝날 때 그 뒤 이름에 들어선 것 — 다음 writer 가 잡은 살아 있는 락, 그 자리에 놓인 링크 — 은
    그대로다. 다음 writer 가 쥔 동안 셋째 writer 는 그 락을 대며 거절된다 (종단 반려 netti-20261023-132314 의 반례 둘)."""
    run, outside = tmp_path / "run", tmp_path / "outside"
    run.mkdir()
    outside.mkdir()
    lock = run / runio.LOCK_FILENAME
    first = start(_STEPS, run, "first-writer", "release", "wait")
    assert [first.line(), first.line(), first.line()] == ["trying", "first-writer", "release"], first.stderr()
    assert not os.path.lexists(lock)
    if after == "live-writer-lock":
        successor = start(_STEPS, run, "successor-writer", "wait")
        assert [successor.line(), successor.line()] == ["trying", "successor-writer"], successor.stderr()
    else:
        (outside / "keep.txt").write_bytes(b"not a lock\n")
        symlink_or_skip(lock, Path("../outside/keep.txt"))
    before = _tree(tmp_path)
    assert first.finish() == 0, first.stderr()
    assert _tree(tmp_path) == before, "락을 놓은 writer 가 끝나면서 그 뒤 이름에 들어선 것을 지웠다"
    if after == "live-writer-lock":
        third = start(_STEPS, run, "third-writer")
        assert third.finish() == 1, "다음 writer 가 쥔 런 폴더를 셋째 writer 가 잡았다"
        assert f"run dir is locked by successor-writer (pid {successor.pid})" in third.stderr()


@pytest.mark.parametrize("steps", [["writer"], ["first-take", "release", "second-take"]],
                         ids=["never-released", "released-and-taken-again"])
def test_a_writer_that_still_holds_its_lock_removes_it_when_it_exits(tmp_path, start, steps) -> None:
    """쥔 채 끝나는 writer 는 자기 락을 지운다 — 해제하지 않았을 때도, 해제한 뒤 같은 프로세스에서 다시 잡았을 때도(그 새
    락). 해제가 종료 처리를 거두어도 다시 잡은 락의 종료 처리는 남는다."""
    run = tmp_path / "run"
    run.mkdir()
    lock = run / runio.LOCK_FILENAME
    writer = start(_STEPS, run, *steps, "wait")
    said = [word for step in steps for word in (["release"] if step == "release" else ["trying", step])]
    assert [writer.line() for _ in said] == said, writer.stderr()
    holder = json.loads(lock.read_text(encoding="utf-8"))
    assert (holder["owner"], holder["pid"]) == (steps[-1], writer.pid)
    assert writer.finish() == 0, writer.stderr()
    assert not os.path.lexists(lock), "쥔 락을 남기고 끝났다"


def test_a_late_reclaimer_and_another_writer_queue_so_only_one_of_them_holds_the_run_dir(tmp_path, start) -> None:
    """죽은 주인의 락을 회수하는 일과 새 락을 만드는 일은 줄을 선다 (회수 경합, 설계 판정 eve-20261023-135225).

    늦은 회수자가 옛 락의 주인이 죽었다고 판정한 자리에서 멈춘 동안, 다른 writer 는 락을 만들지 못하고 기다리며 옛 락은
    그대로다. 회수자가 판정한 그 옛 락을 지우고 잡은 뒤에야 writer 의 차례가 오고, writer 는 살아 있는 회수자의 락을 대며
    거절된다. 두 writer 가 함께 쥐는 일은 없다."""
    run = tmp_path / "run"
    run.mkdir()
    lock = run / runio.LOCK_FILENAME
    lock.write_text(json.dumps({"owner": "a writer that died", "pid": _dead_pid(), "started": 0}), encoding="utf-8")
    stale, stale_bytes = os.lstat(lock), lock.read_bytes()
    slow = start(_SLOW_RECLAIMER, run)
    assert slow.line() == "judged-stale", slow.stderr()
    writer = start(_STEPS, run, "writer", "wait")
    assert writer.line() == "trying", writer.stderr()
    during_pause = writer.line(timeout=3.0)
    try:
        now, now_bytes = os.lstat(lock), lock.read_bytes()
    except FileNotFoundError:  # 멈추지 않은 writer 가 그 사이 지웠다
        now = now_bytes = None
    stale_kept = (now is not None and (now.st_dev, now.st_ino) == (stale.st_dev, stale.st_ino)
                  and now_bytes == stale_bytes)
    slow.go_on()
    said = {"slow-reclaimer": slow.line(), "writer": during_pause if during_pause != "no answer" else writer.line()}
    holders = sorted(name for name, line in said.items() if line == name)
    assert len(holders) == 1, f"두 writer 가 함께 런 폴더를 쥐었다: {holders} (회수자가 판정한 동안 writer: {during_pause})"
    assert during_pause == "no answer" and stale_kept, "회수자가 옛 락을 판정한 동안 다른 writer 가 락을 만들거나 옛 락을 건드렸다"
    assert holders == ["slow-reclaimer"], said
    assert json.loads(lock.read_text(encoding="utf-8"))["pid"] == slow.pid
    assert writer.finish() == 1, writer.stderr()
    assert f"run dir is locked by slow-reclaimer (pid {slow.pid})" in writer.stderr()
    assert slow.finish() == 0, slow.stderr()
    assert not os.path.lexists(lock)
