# SPDX-License-Identifier: Apache-2.0
"""Run a bounded inspect -> score -> correction-hint loop."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sprite_gen.util.cli_kwargs import namespace_from_kwargs
from sprite_gen.qa import inspect as inspect_mod
from sprite_gen.qa import score as score_mod
from sprite_gen.spec.runio import LOCK_FILENAME, atomic_write_text, refuse_overlap, relative_posix, same_folder
from sprite_gen.util.console import utf8_streams


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--states", default="all")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--max-passes", type=int, default=3)
    parser.add_argument("--min-attempts", type=int, default=1)
    parser.add_argument("--pass-score", type=float, default=90.0)
    parser.add_argument("--provider-command", help="optional command template; {prompt_file}, {next_run_dir}, {attempt}")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-preserve-best", action="store_true")
    return parser


def _skipped_by_copy(out_dir: Path) -> Callable[[Path], bool]:
    """What the best candidate's copy of a run dir leaves out: its lock, `__pycache__`, and the loop's out dir where
    it lies in the run dir, as the default one does (the copy would copy itself into itself). A link to the out dir
    is not left out: it is an input that reaches the loop's own output, refused before the loop starts."""
    return lambda path: (path.name in (LOCK_FILENAME, "__pycache__")
                         or (not path.is_symlink() and same_folder(path, out_dir)))


def _links_copied(source: Path, skipped: Callable[[Path], bool]) -> list[Path]:
    """Every link in `source` the copy follows (`shutil.copytree` reads a linked file's bytes and a linked folder's
    content), the linked folders walked too, as found there: what the copy reads beside the run dir's own entries."""
    links: list[Path] = []
    todo, walked = [source], set()
    while todo:
        with os.scandir(todo.pop()) as entries:
            for entry in entries:
                path = Path(entry.path)
                if skipped(path):
                    continue
                if entry.is_symlink():
                    links.append(path)
                if entry.is_dir():
                    folder = entry.stat()
                    if (folder.st_dev, folder.st_ino) not in walked:  # a folder linked twice, or a link loop, once
                        walked.add((folder.st_dev, folder.st_ino))
                        todo.append(path)
    return links


def _copy_run_dir(source: Path, target: Path, skipped: Callable[[Path], bool]) -> None:
    if target.exists():
        shutil.rmtree(target)

    def ignore(folder: str, names: list[str]) -> set[str]:
        return {name for name in names if skipped(Path(folder) / name)}

    shutil.copytree(source, target, ignore=ignore)


def _write_hints(path: Path, hints: list[str]) -> None:
    body = "\n".join(f"- {hint}" for hint in hints) if hints else "- No correction hint; candidate passed."
    atomic_write_text(path, body + "\n")


def run_loop(
    run_dir: Path,
    *,
    states: str = "all",
    out_dir: Path | None = None,
    max_passes: int = 3,
    min_attempts: int = 1,
    pass_score: float = 90.0,
    provider_command: str | None = None,
    dry_run: bool = False,
    preserve_best: bool = True,
) -> dict[str, Any]:
    if max_passes < 1:
        raise SystemExit("--max-passes must be at least 1")
    if min_attempts < 1 or min_attempts > max_passes:
        raise SystemExit("--min-attempts must be between 1 and --max-passes")
    given = run_dir
    run_dir = run_dir.expanduser().resolve()
    out_dir = (out_dir or (run_dir / "correction-loop")).expanduser().resolve()
    # The loop never writes over, moves or deletes the run it reads: the run dir, and every link its copy follows
    # (each with every link or folder its path looks up), is never the out dir or inside it. The out dir may lie in
    # the run dir (the default does): it is added to it, and the copy leaves it out.
    skipped = _skipped_by_copy(out_dir)
    found = _links_copied(run_dir, skipped) if run_dir.is_dir() else []
    refuse_overlap({"--run-dir": given, **{f"{relative_posix(p, run_dir)} in --run-dir": p for p in found}},
                   {"--out-dir": out_dir}, replaces="the loop writes its attempts and best candidate there")
    out_dir.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    current = run_dir

    for attempt in range(1, max_passes + 1):
        attempt_dir = out_dir / f"attempt-{attempt}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        inspect_report = inspect_mod.inspect_run(current, states=states)
        scored = score_mod.score_inspection(inspect_report)
        atomic_write_text(attempt_dir / "inspect.json", json.dumps(inspect_report, ensure_ascii=False, indent=2) + "\n")
        atomic_write_text(attempt_dir / "score.json", json.dumps(scored, ensure_ascii=False, indent=2) + "\n")
        _write_hints(attempt_dir / "correction-hints.txt", scored["hints"])
        record = {
            "attempt": attempt,
            "run_dir": str(current),
            "inspect_report": str(attempt_dir / "inspect.json"),
            "score_report": str(attempt_dir / "score.json"),
            "hint_file": str(attempt_dir / "correction-hints.txt"),
            "ok": scored["ok"],
            "overall_score": scored["overall_score"],
            "candidate_rank": scored["candidate_rank"],
        }
        attempts.append(record)
        if best is None or (scored["candidate_rank"], scored["overall_score"]) > (
            best["candidate_rank"],
            best["overall_score"],
        ):
            best = record
        if attempt >= min_attempts and scored["ok"] and float(scored["overall_score"]) >= pass_score:
            break
        if attempt == max_passes:
            break
        if dry_run:
            break
        if not provider_command:
            raise SystemExit("correction loop needs --provider-command, or use --dry-run for an inspect/score/hint demo")
        next_run_dir = out_dir / f"candidate-{attempt + 1}"
        prompt_file = attempt_dir / "correction-hints.txt"
        # The template is split first and each argument filled in after, so a substituted
        # path is never re-read as shell syntax: POSIX splitting would drop the backslashes
        # of a Windows path and break a path that holds a space or a quote.
        argv = [arg.format(prompt_file=str(prompt_file), next_run_dir=str(next_run_dir), attempt=attempt + 1)
                for arg in shlex.split(provider_command)]
        command = shlex.join(argv)
        completed = subprocess.run(argv, capture_output=True, text=True)
        atomic_write_text(
            attempt_dir / "provider.log",
            completed.stdout + ("\n--- stderr ---\n" + completed.stderr if completed.stderr else ""),
        )
        if completed.returncode != 0:
            raise SystemExit(f"provider command failed with exit {completed.returncode}: {command}")
        if not (next_run_dir / "sprite-request.json").is_file():
            raise SystemExit(f"provider command did not create a sprite-gen run dir: {next_run_dir}")
        current = next_run_dir

    assert best is not None
    preserved_best = None
    if preserve_best:
        preserved = out_dir / "best-candidate"
        _copy_run_dir(Path(best["run_dir"]), preserved, skipped)
        preserved_best = str(preserved)
    result = {
        "ok": bool(attempts and attempts[-1]["ok"] and float(attempts[-1]["overall_score"]) >= pass_score),
        "kind": "sprite-gen-correction-loop-report",
        "dry_run": dry_run,
        "min_attempts": min_attempts,
        "max_passes": max_passes,
        "attempts": attempts,
        "best_candidate": best,
        "preserved_best": preserved_best,
    }
    atomic_write_text(out_dir / "correction-loop.report.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def _run(args: argparse.Namespace) -> int:
    result = run_loop(
        args.run_dir,
        states=args.states,
        out_dir=args.out_dir,
        max_passes=args.max_passes,
        min_attempts=args.min_attempts,
        pass_score=args.pass_score,
        provider_command=args.provider_command,
        dry_run=args.dry_run,
        preserve_best=not args.no_preserve_best,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "attempts"}, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


def run(**kwargs: object) -> int:
    args = namespace_from_kwargs(_build_parser(), kwargs)
    return _run(args)


def main() -> int:
    utf8_streams()
    return _run(_build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
