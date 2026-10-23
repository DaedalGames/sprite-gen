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
  5. 락 항목을 만들고 내용을 다 쓰지 못한 writer 는 그 만들다 만 항목을 거두고 실패한다 — 주인을 확인할 수 없는 락이
     남아 다음 writer 를 막지 않는다
  6. 락은 잡은 프로세스의 것이라 fork 자식은 부모의 락을 지우지도 쥐지도 않는다 — 맨 os.fork 자식, fork 로 띄운 풀 워커,
     파이썬 fork 훅이 돌지 않는 C fork 자식이 끝나거나 놓거나 같은 런 폴더를 잡아도 부모의 살아 있는 락은 그대로고, 같은
     런 폴더 잡기는 그 락을 대며 거절된다. 주인이 런 폴더 커널 락 안에서 죽으면 그 사이 다른 스레드가 fork 한 자식이 살아
     있어도 writer 는 기다리지 않고, 주인이 살아 있으면 그 자식이 끝나도 커널 락은 주인의 것 그대로다. 자식들이 끝난 뒤에도
     부모는 자기 락에 다시 들어오고 놓거나 끝나며 지운다

3–6 은 실제 프로세스로 잰다 — 락은 프로세스 사이의 계약이고, 종료 처리는 프로세스가 끝날 때만 돈다.
"""

from __future__ import annotations

import contextlib
import errno
import json
import os
import queue
import signal
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

# 파일 크기 한도(RLIMIT_FSIZE)를 락 내용보다 작게 둔 writer — 락을 만든 뒤 쓰기가 한도에서 EFBIG 로 끊긴다(디스크가 찬
# 것과 같은 자리). SIGXFSZ 는 무시해 프로세스가 죽지 않고 쓰기가 실패하게 한다. 한도는 import 뒤에 건다.
_CUT_SHORT_WRITER = r"""
import os
import resource
import signal
import sys
from pathlib import Path
from sprite_gen.spec import runio
print(os.getpid(), flush=True)
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
resource.setrlimit(resource.RLIMIT_FSIZE, (16, resource.getrlimit(resource.RLIMIT_FSIZE)[1]))
try:
    runio.acquire_run_dir_lock(Path(sys.argv[1]), "cut-short-writer")
except OSError as exc:
    print("failed", exc.errno, flush=True)
else:
    print("cut-short-writer", flush=True)
"""

# 락을 쥔 부모가 자식 하나를 갈라 낸다 — 맨 os.fork(raw), fork 로 시작하는 ProcessPoolExecutor 워커(pool, 3.12 까지 리눅스의
# 기본 시작 방식), 파이썬 fork 훅이 돌지 않는 libc fork(c-fork — C 코드가 fork 하는 자리, ctypes 로). 자식은 끝나기만 하거나(end —
# raw·c-fork 는 sys.exit 로 보통 끝나 물려받은 종료 처리까지 돌고, 풀 워커는 물려받은 종료 처리 없이 끝난다: 3.12 까지는 os._exit,
# 3.13 부터는 atexit 를 비우고 시작해 자기가 단 것만 돈다) 런 폴더 락을 놓거나(release) 같은 런 폴더를 잡거나(acquire-same) 다른
# 런 폴더를 잡는다(acquire-other). 자식이 끝난 뒤 부모는 본 것을 JSON 한 줄로 찍고 놓는다.
_FORKED = r"""
import ctypes
import json
import multiprocessing
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from sprite_gen.spec import runio


def act(mode, run, other):
    if mode == "release":
        runio.release_run_dir_lock(run)
        return "released"
    if mode.startswith("acquire"):
        try:
            return f"held {runio.acquire_run_dir_lock(run if mode == 'acquire-same' else other, 'forked-child')}"
        except SystemExit as exc:
            return f"refused: {str(exc).splitlines()[0]}"
    return "ended"


def another_writer(run):
    got = subprocess.run([sys.executable, "-c", "import sys; from pathlib import Path; from sprite_gen.spec import runio; "
                          "runio.acquire_run_dir_lock(Path(sys.argv[1]), 'another-writer')", str(run)],
                         capture_output=True, encoding="utf-8", timeout=30)
    return got.returncode, got.stderr


run, other, kind, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
lock = runio.acquire_run_dir_lock(run, "live-parent")
held = lock.read_bytes()
if kind == "pool":
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("fork")) as pool:
        said = pool.submit(act, mode, run, other).result(timeout=30)
    ended = 0  # 풀이 닫혔다 — 워커는 끝났다
else:
    read_end, write_end = os.pipe()
    child = os.fork() if kind == "raw" else ctypes.PyDLL(None).fork()
    if child == 0:
        os.close(read_end)
        os.write(write_end, act(mode, run, other).encode())
        os.close(write_end)
        if mode == "end":
            sys.exit(0)
        os._exit(0)
    os.close(write_end)
    with os.fdopen(read_end, "rb") as pipe:
        said = pipe.read().decode()
    ended = os.waitstatus_to_exitcode(os.waitpid(child, 0)[1])
try:
    kept = lock.read_bytes() == held
except FileNotFoundError:
    kept = False
rc, err = another_writer(run)
seen = {"said": said, "child_exit": ended, "parent_pid": os.getpid(), "parent_lock_kept": kept,
        "another_writer_rc": rc, "another_writer_stderr": err}
if mode == "acquire-other":
    seen["other_lock_left"] = os.path.lexists(other / runio.LOCK_FILENAME)
    seen["other_next_writer_rc"] = another_writer(other)[0]
print(json.dumps(seen), flush=True)
runio.release_run_dir_lock(run)
"""

# 죽은 주인의 락을 판정하는 동안 — 런 폴더 커널 락(`_entry_guard`) 안에서 — SIGKILL 로 죽는 주인. 판정은 한 스레드가 하고,
# fork 면 그 사이 주 스레드가 자식을 갈라 낸다. 자식은 표시 파일 `go` 가 생기거나 6 s 가 지날 때까지 산다.
_KILLED_IN_THE_GUARD = r"""
import os
import signal
import sys
import threading
import time
from pathlib import Path
from sprite_gen.spec import runio
run, marks, dead, mode = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
judge = runio._pid_alive
def judging(pid):
    if pid == dead:
        (marks / "judging").touch()
        time.sleep(60)
    return judge(pid)
runio._pid_alive = judging
threading.Thread(target=runio.acquire_run_dir_lock, args=(run, "killed-holder"), daemon=True).start()
deadline = time.monotonic() + 10
while not (marks / "judging").exists():
    if time.monotonic() > deadline:
        sys.exit("the holder never reached its judgment")
    time.sleep(0.01)
if mode == "fork":
    child = os.fork()
    if child == 0:
        deadline = time.monotonic() + 6
        while not (marks / "go").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        os._exit(0)
    (marks / "child").write_text(str(child))
os.kill(os.getpid(), signal.SIGKILL)
"""

# 락을 잡는 데 걸린 시간(초)을 찍는 writer.
_TIMED_WRITER = r"""
import sys
import time
from pathlib import Path
from sprite_gen.spec import runio
started = time.monotonic()
runio.acquire_run_dir_lock(Path(sys.argv[1]), "writer")
print(f"{time.monotonic() - started:.2f}", flush=True)
"""

# 한 스레드가 죽은 주인의 락을 판정하는 동안 — 런 폴더 커널 락 안에서 — 주 스레드가 자식을 갈라 내고 그 자식이 끝나기를
# 기다린 뒤(forked), 표준 입력 한 줄을 받으면 판정을 이어 락을 잡고(judging-parent) 다시 한 줄을 기다린다.
_JUDGING_PARENT = r"""
import os
import sys
import threading
from pathlib import Path
from sprite_gen.spec import runio
print(os.getpid(), flush=True)
run, dead = Path(sys.argv[1]), int(sys.argv[2])
judge = runio._pid_alive
judging, go_on = threading.Event(), threading.Event()
def judged(pid):
    if pid == dead:
        judging.set()
        go_on.wait()
    return judge(pid)
runio._pid_alive = judged
taker = threading.Thread(target=runio.acquire_run_dir_lock, args=(run, "judging-parent"))
taker.start()
judging.wait()
child = os.fork()
if child == 0:
    os._exit(0)
os.waitpid(child, 0)
print("forked", flush=True)
sys.stdin.readline()
go_on.set()
taker.join()
print("judging-parent", flush=True)
sys.stdin.readline()
"""

# 락을 쥔 부모가 맨 os.fork 자식, libc fork 자식, fork 풀 워커를 차례로 띄우고 끝나기를 기다린 뒤(자식은 아무것도 하지 않고
# os._exit 로 끝난다) 같은 런 폴더를 다시 잡고(재진입) 놓거나(release) 쥔 채 끝난다(exit).
_PARENT_AFTER_CHILDREN = r"""
import ctypes
import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from sprite_gen.spec import runio
run, finish = Path(sys.argv[1]), sys.argv[2]
lock = runio.acquire_run_dir_lock(run, "parent")
for fork in (os.fork, ctypes.PyDLL(None).fork):
    child = fork()
    if child == 0:
        os._exit(0)
    os.waitpid(child, 0)
with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("fork")) as pool:
    pool.submit(os.getpid).result(timeout=30)
print(json.dumps({"reentered": runio.acquire_run_dir_lock(run, "parent") == lock, "kept": lock.is_file()}), flush=True)
if finish == "release":
    runio.release_run_dir_lock(run)
    print(json.dumps({"left_after_release": os.path.lexists(lock)}), flush=True)
"""

_NO_FORK = pytest.mark.skipif(not hasattr(os, "fork"), reason="os.fork and os.register_at_fork are Unix only (Availability: "
                                                            "Unix, not Emscripten, not WASI) — Windows makes no fork child")


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


def _seen(code: str, *args: object) -> list[dict]:
    """실제 프로세스 하나를 끝까지 돌려 그가 찍은 JSON 줄들."""
    got = subprocess.run([sys.executable, "-c", code, *map(str, args)], capture_output=True, encoding="utf-8", timeout=60)
    assert got.returncode == 0, got.stderr
    return [json.loads(line) for line in got.stdout.splitlines() if line.startswith("{")]


def _wait_gone(pid: int | None, timeout: float = 10.0) -> None:
    """시험이 띄운 프로세스의 자식(이제 고아)이 끝날 때까지 — 길어야 `timeout` 초."""
    deadline = time.monotonic() + timeout
    while pid is not None and runio._pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)


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


@pytest.mark.skipif(sys.platform == "win32", reason="runio._pid_alive cannot tell a dead pid on Windows: os.kill(pid, 0) "
                                                    "sends CTRL_C_EVENT there (an existing defect outside this test)")
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


@pytest.mark.skipif(sys.platform == "win32", reason="RLIMIT_FSIZE and SIGXFSZ are POSIX")
def test_a_writer_that_could_not_write_its_lock_takes_back_what_it_made(tmp_path, start) -> None:
    """락 항목을 만들고 내용을 쓰다 끊긴 writer(파일 크기 한도에서 EFBIG — 디스크가 찬 것과 같은 자리)는 그 만들다 만 항목을
    거두고 실패한다. 주인을 확인할 수 없는 그 항목이 남으면 다음 writer 가 15분 동안 거절된다."""
    run = tmp_path / "run"
    run.mkdir()
    lock = run / runio.LOCK_FILENAME
    writer = start(_CUT_SHORT_WRITER, run)
    assert writer.line() == f"failed {errno.EFBIG}", writer.stderr()
    assert writer.finish() == 0, writer.stderr()
    assert not os.path.lexists(lock), f"다 쓰지 못한 락 항목이 남았다: {lock.read_bytes()!r}"
    following = start(_STEPS, run, "following-writer")
    assert following.finish() == 0, following.stderr()


@_NO_FORK
@pytest.mark.parametrize(("kind", "mode"), [(kind, mode) for kind in ("raw", "pool", "c-fork")
                                            for mode in ("end", "release", "acquire-same")] + [("pool", "acquire-other")])
def test_a_forked_child_neither_removes_nor_holds_its_parents_lock(tmp_path, kind, mode) -> None:
    """락은 잡은 프로세스의 것이다 — 락을 쥔 부모가 fork 한 자식이 끝나거나 놓거나 같은 런 폴더를 잡아도 부모의 살아 있는 락은
    그대로고, 자식의 같은 런 폴더 잡기와 다른 writer 는 그 락을 대며 거절된다 (종단 반려 netti-20261023-210333 의 반례 둘이
    raw 의 end·release, 나머지는 설계 판정 eve-20261023-211033 의 fork 갈래). 풀 워커가 잡은 다른 런 폴더의 락은 워커의 것이다
    — 3.12 까지는 워커가 os._exit 로 끝나 남고 죽은 pid 의 락이라 다음 writer 가 회수하며, 3.13 부터는 워커가 끝나며 지운다.
    어느 쪽이든 다음 writer 는 그 런 폴더를 잡는다."""
    run, other = tmp_path / "run", tmp_path / "other"
    run.mkdir()
    other.mkdir()
    [seen] = _seen(_FORKED, run, other, kind, mode)
    refused = f"run dir is locked by live-parent (pid {seen['parent_pid']})"
    assert seen["child_exit"] == 0, seen
    if mode == "acquire-same":
        assert seen["said"] == f"refused: {refused}: {run}", ("fork 자식이 부모가 쥔 런 폴더를 잡았다", seen)
    assert seen["parent_lock_kept"], ("fork 자식이 부모의 살아 있는 락을 지웠다", seen)
    assert seen["another_writer_rc"] == 1 and refused in seen["another_writer_stderr"], seen
    if mode == "acquire-other":
        assert seen["said"] == f"held {other.resolve() / runio.LOCK_FILENAME}", seen
        assert seen["other_next_writer_rc"] == 0, seen


@_NO_FORK
@pytest.mark.parametrize("mode", ["no-child", "fork"])
def test_a_holder_killed_inside_the_run_dirs_kernel_lock_leaves_it_to_no_child(tmp_path, mode) -> None:
    """주인이 런 폴더 커널 락 안에서(죽은 주인의 락을 판정하는 동안) 죽으면 커널이 그 락을 놓는다 — 그 사이 다른 스레드가 fork 한
    자식이 살아 있어도 자식은 물려받은 커널 락을 쥐지 않아, writer 는 자식이 끝나기를 기다리지 않고 죽은 주인의 락을 회수해
    잡는다 (설계 판정 eve-20261023-211033)."""
    run, marks = tmp_path / "run", tmp_path / "marks"
    run.mkdir()
    marks.mkdir()
    dead = _dead_pid()
    (run / runio.LOCK_FILENAME).write_text(json.dumps({"owner": "a writer that died", "pid": dead, "started": 0}),
                                           encoding="utf-8")
    # 주인의 출력은 파이프가 아니라 파일로 — 자식이 물려받은 파이프면 subprocess.run 이 자식이 끝날 때까지 기다린다
    with open(marks / "holder.txt", "w", encoding="utf-8") as said:
        holder = subprocess.run([sys.executable, "-c", _KILLED_IN_THE_GUARD, *map(str, (run, marks, dead, mode))],
                                stdout=said, stderr=said, timeout=60)
    assert holder.returncode == -signal.SIGKILL, (marks / "holder.txt").read_text(encoding="utf-8")
    child = int((marks / "child").read_text()) if mode == "fork" else None
    try:
        assert child is None or runio._pid_alive(child), "주인이 fork 한 자식이 writer 보다 먼저 끝났다"
        writer = subprocess.run([sys.executable, "-c", _TIMED_WRITER, str(run)], capture_output=True, encoding="utf-8",
                                timeout=60)
    finally:
        (marks / "go").touch()
        _wait_gone(child)
    assert writer.returncode == 0, writer.stderr
    assert float(writer.stdout) < 2.0, f"죽은 주인의 fork 자식이 쥔 커널 락을 writer 가 {writer.stdout.strip()} s 기다렸다"


@_NO_FORK
def test_a_fork_child_that_ends_leaves_its_parent_inside_the_run_dirs_kernel_lock(tmp_path, start) -> None:
    """fork 자식은 물려받은 커널 락 사본을 닫기만 하고 풀지 않는다 — 부모의 한 스레드가 런 폴더 커널 락 안에서 판정하는 동안
    다른 스레드가 fork 한 자식이 끝나도 writer 는 기다리고, 부모가 판정한 옛 락을 회수해 잡은 뒤에야 그 락을 대며 거절된다."""
    run = tmp_path / "run"
    run.mkdir()
    dead = _dead_pid()
    (run / runio.LOCK_FILENAME).write_text(json.dumps({"owner": "a writer that died", "pid": dead, "started": 0}),
                                           encoding="utf-8")
    parent = start(_JUDGING_PARENT, run, dead)
    assert parent.line() == "forked", parent.stderr()
    writer = start(_STEPS, run, "writer")
    assert writer.line() == "trying", writer.stderr()
    during = writer.line(timeout=3.0)
    parent.go_on()
    assert parent.line() == "judging-parent", parent.stderr()
    assert during == "no answer", f"부모가 커널 락 안에 있는데 fork 자식이 끝나자 writer 가 들어왔다: {during}"
    assert writer.finish() == 1, writer.stderr()
    assert f"run dir is locked by judging-parent (pid {parent.pid})" in writer.stderr()
    assert parent.finish() == 0, parent.stderr()


@_NO_FORK
@pytest.mark.parametrize("finish", ["release", "exit"])
def test_a_parent_whose_forked_children_ended_still_takes_and_removes_its_own_lock(tmp_path, finish) -> None:
    """자식들(맨 os.fork, libc fork, fork 풀 워커)이 돌고 끝난 뒤에도 부모는 자기 락에 다시 들어오고(재진입), 놓거나 쥔 채
    끝나며 그 락을 지운다 — fork 가 자식의 몫을 지우면서 부모의 보유까지 지우지 않는다."""
    run = tmp_path / "run"
    run.mkdir()
    lock = run / runio.LOCK_FILENAME
    seen = _seen(_PARENT_AFTER_CHILDREN, run, finish)
    assert seen[0] == {"reentered": True, "kept": True}, seen
    if finish == "release":
        assert seen[1] == {"left_after_release": False}, "자식들이 끝난 뒤 부모가 놓았는데 자기 락이 남았다"
    assert not os.path.lexists(lock), "자식들이 끝난 뒤 부모가 끝나면서 자기 락을 남겼다"
