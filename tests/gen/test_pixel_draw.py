# SPDX-License-Identifier: Apache-2.0
"""`sprite-gen pixel-draw` end to end against a fake Claude at the transport seam: what is asked
and in what order, the row repair, the one redraw (G1 still broken, or no answer at max_tokens one
effort lower), refusals and failed calls reported as they came, the G7 pick, Message Batches and
`--resume`, the run folder and its report, and `--grid`. No key, no network: the answers are the
model's own recorded ones (tests/fixtures/pixel_draw)."""
from __future__ import annotations

import base64
import io
import json
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from conftest import help_options
from sprite_gen.gen import anthropic_api as api
from sprite_gen.gen import pixel_draw, refusal
from sprite_gen.gen import pixel_grid as pg
from sprite_gen.gen.anthropic_api import ApiError, Reply

KEY = "sk-ant-api03-" + "Synthetic0Key1For2Tests" * 2
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "pixel_draw"


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


SPRITE16 = fixture("sprite-16-pass.txt")
ICON32 = fixture("icon-32-pass.txt")
ICON_ROW = fixture("icon-32-row-width.txt")
ICON_ROW_FIX = fixture("icon-32-row-width.repair.txt")
TILE_CUT = fixture("tile-32-truncated.txt")
TILE_CUT_FIX = fixture("tile-32-truncated.repair.txt")
WALK16 = fixture("walk-16-pass.txt")
FOURDIR16 = fixture("fourdir-16-pass.txt")
VERDICT = fixture("pick-verdict.json")  # best: "A"


def tile_text(size: int) -> str:
    return "```pixels\nPALETTE\n. transparent\ng #4e9038\nFRAME tile\n" + "\n".join(["g" * size] * size) + "\nEND\n```"


def reply(text: str, *, stop: str = "end_turn", output: int = 1200, thinking: int = 1000, details=None) -> Reply:
    return Reply(text=text, thinking="summary", stop_reason=stop, stop_details=details, model=api.MODEL,
                 usage={"input_tokens": 500, "output_tokens": output, "output_tokens_details": {"thinking_tokens": thinking},
                        "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
                 message_id="msg_x", seconds=1.5)


class FakeClaude:
    """Answers by call id (realtime: the label names it; batches: the custom_id)."""

    def __init__(self, script, polls=None, known=None):
        self.script = dict(script)
        self.asked: list[tuple[str, dict]] = []
        self.submitted: list[tuple[str, list]] = []
        self.polls = dict(polls or {})
        self.batch_ids = dict(known or {})

    def stream(self, token, body, *, sleep, label, timeout=None):
        assert token == KEY
        call_id = label.split()[-1]
        self.asked.append((call_id, body))
        if call_id not in self.script:
            pytest.fail(f"unexpected call {call_id}")
        outcome = self.script[call_id]
        if isinstance(outcome, ApiError):
            raise outcome
        return replace(outcome, request_id=f"req_{call_id}")

    def submit(self, token, requests, *, sleep, label):
        batch_id = f"msgbatch_{len(self.submitted) + 1}"
        self.submitted.append((batch_id, requests))
        self.batch_ids[batch_id] = [cid for cid, _ in requests]
        return {"id": batch_id, "processing_status": "in_progress"}

    def get(self, token, batch_id, *, sleep, label):
        states = self.polls.setdefault(batch_id, ["ended"])
        state = states.pop(0) if len(states) > 1 else states[0]
        return {"id": batch_id, "processing_status": state, "results_url": f"https://results.invalid/{batch_id}",
                "ended_at": "2026-10-12T00:00:00Z" if state == "ended" else None}

    def results(self, token, batch, *, sleep, label):
        return {cid: replace(self.script[cid], mode="batch", batch_id=batch["id"], seconds=None)
                for cid in self.batch_ids[batch["id"]]}


@pytest.fixture
def claude(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    monkeypatch.setattr(pixel_draw, "BATCH_POLL_SECONDS", 0)

    def make(script, **kwargs) -> FakeClaude:
        fake = FakeClaude(script, **kwargs)
        monkeypatch.setattr(api, "stream_message", fake.stream)
        monkeypatch.setattr(api, "submit_batch", fake.submit)
        monkeypatch.setattr(api, "get_batch", fake.get)
        monkeypatch.setattr(api, "batch_results", fake.results)
        return fake

    return make


def draw(out: Path, *args: str) -> int:
    return pixel_draw.main(["--out-dir", str(out), *args])


def report_of(out: Path) -> dict:
    return json.loads((out / "report.json").read_text(encoding="utf-8"))


def asked(fake: FakeClaude) -> list[str]:
    return [call_id for call_id, _ in fake.asked]


def test_one_candidate_is_drawn_checked_and_written(tmp_path, claude, capsys):
    fake = claude({"c1-gen1": reply(SPRITE16, output=2515, thinking=2300)})
    out = tmp_path / "run"
    assert draw(out, "--subject", "a knight", "--size", "16", "--frames", "front", "--candidates", "1") == 0
    report = report_of(out)
    captured = capsys.readouterr()
    assert json.loads(captured.out) == report
    assert "[gen] claude: running on ANTHROPIC_API_KEY — this is a per-call API charge" in captured.err
    assert report["status"] == "ok" and report["chosen"] == "c1" and report["pick"] == {"by": "only", "best": "c1"}
    assert sorted(p.name for p in out.iterdir()) == sorted([
        "report.json", "request.json", "calls", "candidates", "grid.txt", "palette.json", "frame-front.png",
        "preview-x16.png"])
    assert report["files"] == {"grid": "grid.txt", "palette": "palette.json", "frames": {"front": "frame-front.png"},
                               "preview": "preview-x16.png"}
    (call,) = report["calls"]
    assert {k: call[k] for k in ("id", "purpose", "candidate", "outcome", "mode", "model", "effort", "max_tokens",
                                 "request_id", "stop_reason", "thinking_tokens")} == {
        "id": "c1-gen1", "purpose": "generate", "candidate": "c1", "outcome": "reply", "mode": "realtime",
        "model": "claude-opus-5-5", "effort": "high", "max_tokens": 32000, "request_id": "req_c1-gen1",
        "stop_reason": "end_turn", "thinking_tokens": 2300}
    assert call["cost_usd"] == api.cost_usd(call["usage"], batch=False) == report["cost_usd"]
    assert report["usage"]["output_tokens"] == 2515 and report["usage"]["thinking_tokens"] == 2300
    assert report["seconds"] >= 0
    (candidate,) = report["candidates"]
    assert candidate["status"] == "passed" and candidate["G1"] == {"ok": True, "issues": [], "recovered": []}
    assert candidate["gates"]["ok"] and candidate["files"]["grid"] == "candidates/c1/grid.txt"
    grid, _ = pg.parse_answer(SPRITE16, "chargrid", 16, 16, ("front",))
    assert (out / "grid.txt").read_text(encoding="utf-8") == pg.to_text(grid)
    assert (out / "frame-front.png").read_bytes() == pg.png_bytes(pg.frame_pixels(grid, "front"))
    assert (out / "frame-front.png").read_bytes() == (out / "candidates" / "c1" / "frame-front.png").read_bytes()
    record = json.loads((out / "calls" / "c1-gen1.json").read_text(encoding="utf-8"))
    assert record["reply"]["text"] == SPRITE16 and record["prompt"].startswith("Draw: a knight\nCanvas: exactly 16x16")
    body = fake.asked[0][1]
    assert body["model"] == "claude-opus-5-5" and body["system"] == pixel_draw.SYSTEM
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"} and body["output_config"] == {"effort": "high"}
    request = json.loads((out / "request.json").read_text(encoding="utf-8"))
    assert request["palette"] == {"name": "free:16", "limit": 16, "colors": []} and request["frames"] == ["front"]


def test_a_broken_row_is_asked_for_alone_and_spliced(tmp_path, claude):
    fake = claude({"c1-gen1": reply(ICON_ROW), "c1-gen1-fix1": reply(ICON_ROW_FIX, output=336, thinking=318)})
    out = tmp_path / "run"
    assert draw(out, "--subject", "a wooden chest", "--kind", "icon", "--candidates", "1") == 0
    assert asked(fake) == ["c1-gen1", "c1-gen1-fix1"]
    (candidate,) = report_of(out)["candidates"]
    assert candidate["repairs"] == [{"call": "c1-gen1-fix1", "rows_asked": ["icon 16"], "rows_answered": ["icon 16"],
                                     "ignored": [], "rows_still_broken": []}]
    assert candidate["G1"]["ok"] and candidate["generations"] == ["c1-gen1"]
    body = fake.asked[1][1]
    assert body["output_config"] == {"effort": "medium"} and body["max_tokens"] == 16000
    prompt = body["messages"][0]["content"]
    lines = prompt.splitlines()
    broken = next(line for line in lines if "BROKEN" in line)
    assert broken == "  row 16 (BROKEN, 33 characters): .ooooooooooooGGooGgooooooooooooo."
    ruler = lines[lines.index(broken) + 1]
    assert ruler.startswith("  ruler") and ruler.index("123456789") == broken.index(".oooo")
    assert "Rows to write: icon 16" in prompt and "`ROW <frame> <row number>: <exactly 32 characters>`" in prompt
    rows = (out / "grid.txt").read_text(encoding="utf-8").splitlines()
    assert rows[rows.index("FRAME icon") + 16] == ".ooooooooooooGGooGgoooooooooooo."


def test_a_row_left_broken_goes_back_in_a_second_round(tmp_path, claude):
    row25 = "ggggggggggggggdkdggggggggggggggg"
    fake = claude({"c1-gen1": reply(TILE_CUT), "c1-gen1-fix1": reply(TILE_CUT_FIX),
                   "c1-gen1-fix2": reply(f"ROW tile 25: {row25}")})
    out = tmp_path / "run"
    assert draw(out, "--subject", "grass", "--kind", "tile", "--candidates", "1") == 0
    (candidate,) = report_of(out)["candidates"]
    first, second = candidate["repairs"]
    assert first["rows_asked"] == [f"tile {n}" for n in range(16, 33)] and first["rows_still_broken"] == ["tile 25"]
    assert second["rows_asked"] == ["tile 25"] and second["rows_still_broken"] == []
    assert candidate["G1"] == {"ok": True, "issues": [], "recovered": [{"code": "no_end"}]}
    assert "rows 17-32 are MISSING" in fake.asked[1][1]["messages"][0]["content"]
    assert (out / "tiled-3x3.png").is_file()


def test_still_broken_after_two_rounds_is_drawn_once_more(tmp_path, claude, capsys):
    fake = claude({"c1-gen1": reply(TILE_CUT), "c1-gen1-fix1": reply(TILE_CUT_FIX),
                   "c1-gen1-fix2": reply("ROW tile 25: " + "g" * 33), "c1-gen2": reply(tile_text(32))})
    out = tmp_path / "run"
    assert draw(out, "--subject", "grass", "--kind", "tile", "--candidates", "1") == 0
    assert asked(fake) == ["c1-gen1", "c1-gen1-fix1", "c1-gen1-fix2", "c1-gen2"]
    (candidate,) = report_of(out)["candidates"]
    assert candidate["generations"] == ["c1-gen1", "c1-gen2"]
    assert candidate["redraws"] == [{"after": "c1-gen1", "why": "G1", "effort": "high"}]
    assert candidate["status"] == "passed" and "drawing c1 once more" in capsys.readouterr().err


def test_a_structural_failure_is_drawn_again_without_repair_and_a_second_one_ends_it(tmp_path, claude):
    broken = FOURDIR16.replace("FRAME up\n", "")
    fake = claude({"c1-gen1": reply(broken), "c1-gen2": reply(broken)})
    out = tmp_path / "run"
    with pytest.raises(SystemExit) as caught:
        draw(out, "--subject", "a knight", "--size", "16", "--frames", "down,up,left,right", "--candidates", "1")
    assert "no candidate passed the gates (c1: G1: frame_missing up, row_count_over down)" in str(caught.value)
    assert asked(fake) == ["c1-gen1", "c1-gen2"]
    report = report_of(out)
    assert report["status"] == "failed" and report["candidates"][0]["files"] == {}
    assert report["chosen"] is None and not (out / "grid.txt").exists()


def test_no_answer_at_max_tokens_is_drawn_once_more_one_effort_lower(tmp_path, claude, capsys):
    fake = claude({"c1-gen1": reply("", stop="max_tokens", output=32000, thinking=32000), "c1-gen2": reply(SPRITE16)})
    out = tmp_path / "run"
    assert draw(out, "--subject", "a knight", "--size", "16", "--frames", "front", "--candidates", "1",
                "--effort", "xhigh") == 0
    assert [body["output_config"]["effort"] for _, body in fake.asked] == ["xhigh", "high"]
    report = report_of(out)
    (candidate,) = report["candidates"]
    assert candidate["redraws"] == [{"after": "c1-gen1", "why": "max_tokens", "effort": "high"}]
    assert candidate["effort"] == "high" and candidate["status"] == "passed"
    assert report["cost_usd"] == pytest.approx(sum(c["cost_usd"] for c in report["calls"]))
    assert "drawing c1 once more at effort high" in capsys.readouterr().err


def test_an_answer_cut_off_at_max_tokens_keeps_its_rows_and_is_repaired(tmp_path, claude):
    rows = SPRITE16.split("FRAME front\n")[1].split("\nEND")[0].splitlines()
    cut = SPRITE16[:SPRITE16.index("FRAME front\n")] + "FRAME front\n" + "\n".join(rows[:10])
    fix = "\n".join(f"ROW front {n}: {rows[n - 1]}" for n in range(11, 17))
    fake = claude({"c1-gen1": reply(cut, stop="max_tokens", output=32000, thinking=31000), "c1-gen1-fix1": reply(fix)})
    out = tmp_path / "run"
    assert draw(out, "--subject", "a knight", "--size", "16", "--frames", "front", "--candidates", "1") == 0
    assert asked(fake) == ["c1-gen1", "c1-gen1-fix1"]
    (candidate,) = report_of(out)["candidates"]
    assert candidate["redraws"] == [] and candidate["repairs"][0]["rows_asked"] == [f"front {n}" for n in range(11, 17)]
    assert [r["code"] for r in candidate["G1"]["recovered"]] == ["block_unclosed", "no_end"]
    grid, _ = pg.parse_answer(SPRITE16, "chargrid", 16, 16, ("front",))
    assert (out / "grid.txt").read_text(encoding="utf-8") == pg.to_text(grid)


def test_no_answer_twice_ends_the_candidate(tmp_path, claude):
    empty = reply("", stop="max_tokens", output=32000, thinking=32000)
    fake = claude({"c1-gen1": empty, "c1-gen2": empty})
    out = tmp_path / "run"
    with pytest.raises(SystemExit) as caught:
        draw(out, "--subject", "stones", "--kind", "tile", "--candidates", "1", "--effort", "medium")
    assert "c1: max_tokens: c1-gen2 used all 32000 tokens at effort low and wrote nothing" in str(caught.value)
    assert [body["output_config"]["effort"] for _, body in fake.asked] == ["medium", "low"]


def test_a_candidate_is_drawn_twice_at_most(tmp_path, claude):
    fake = claude({"c1-gen1": reply("", stop="max_tokens"), "c1-gen2": reply("no grid here")})
    with pytest.raises(SystemExit, match=r"c1: G1: no_block"):
        draw(tmp_path / "run", "--subject", "a knight", "--candidates", "1")
    assert asked(fake) == ["c1-gen1", "c1-gen2"]


def test_a_refusal_is_reported_as_it_came(tmp_path, claude):
    fake = claude({"c1-gen1": reply("", stop="refusal", details={"type": "refusal", "category": "cyber", "explanation": "x"})})
    out = tmp_path / "run"
    with pytest.raises(SystemExit) as caught:
        draw(out, "--subject", "something", "--candidates", "1")
    assert str(caught.value).startswith(
        f"pixel-draw: {refusal.CONTENT_POLICY} (stop_reason=refusal category=cyber); every candidate was refused")
    assert asked(fake) == ["c1-gen1"]
    report = report_of(out)
    assert report["status"] == "refused" and report["candidates"][0]["status"] == "refused"


def test_a_failed_call_is_not_asked_again(tmp_path, claude):
    error = ApiError("pixel-draw: claude request failed (HTTP 400) code=invalid_request_error; not asked again",
                     status=400, request_id="req_400", billed=False)
    fake = claude({"c1-gen1": error})
    out = tmp_path / "run"
    with pytest.raises(SystemExit, match=r"c1: pixel-draw: claude request failed \(HTTP 400\)"):
        draw(out, "--subject", "a knight", "--candidates", "1")
    assert asked(fake) == ["c1-gen1"]
    (call,) = report_of(out)["calls"]
    assert call["outcome"] == "error" and call["request_id"] == "req_400" and call["cost_usd"] == 0.0
    assert call["error"] == error.line and call["billed"] is False


def test_the_vision_pick_chooses_between_two_that_passed(tmp_path, claude):
    variant = SPRITE16.replace("r #d84040", "r #c03030")
    fake = claude({"c1-gen1": reply(SPRITE16), "c2-gen1": reply(variant), "pick": reply(VERDICT, output=300, thinking=90)})
    out = tmp_path / "run"
    assert draw(out, "--subject", "a knight", "--size", "16", "--frames", "front") == 0
    report = report_of(out)
    pick = report["pick"]
    assert pick["by"] == "vision" and sorted(pick["labels"]) == ["A", "B"] and sorted(pick["labels"].values()) == ["c1", "c2"]
    assert pick["best"] == pick["labels"]["A"] == report["chosen"] and pick["verdict"]["best"] == "A"
    body = dict(fake.asked)["pick"]
    image, text = body["messages"][0]["content"]
    assert image["source"]["media_type"] == "image/png"
    assert base64.standard_b64decode(image["source"]["data"]) == (out / "pick-sheet.png").read_bytes()
    assert text["text"].startswith("You are judging pixel-art game assets") and "candidates labelled A, B" in text["text"]
    assert body["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": pixel_draw.PICK_SCHEMA}}
    assert "system" not in body
    chosen = out / "candidates" / report["chosen"]
    assert (out / "grid.txt").read_bytes() == (chosen / "grid.txt").read_bytes()
    assert (out / "frame-front.png").read_bytes() == (chosen / "frame-front.png").read_bytes()
    assert [c["id"] for c in report["calls"]] == ["c1-gen1", "c2-gen1", "pick"]


def test_a_person_picks_when_asked(tmp_path, claude):
    fake = claude({"c1-gen1": reply(SPRITE16), "c2-gen1": reply(SPRITE16.replace("r #d84040", "r #c03030"))})
    out = tmp_path / "run"
    assert draw(out, "--subject", "a knight", "--size", "16", "--frames", "front", "--pick", "human") == 0
    report = report_of(out)
    assert report["pick"] == {"by": "human", "best": None, "candidates": ["c1", "c2"]}
    assert report["status"] == "ok" and report["chosen"] is None and not (out / "grid.txt").exists()
    assert "pick" not in asked(fake)


def test_a_pick_that_names_no_candidate_fails_the_run(tmp_path, claude):
    verdict = json.loads(VERDICT)
    verdict["best"] = "Z"
    claude({"c1-gen1": reply(SPRITE16), "c2-gen1": reply(SPRITE16.replace("r #d84040", "r #c03030")),
            "pick": reply(json.dumps(verdict))})
    out = tmp_path / "run"
    with pytest.raises(SystemExit, match=r"2 candidates passed but the pick failed \(the pick named 'Z', not one of A, B\)"):
        draw(out, "--subject", "a knight", "--size", "16", "--frames", "front")
    assert report_of(out)["status"] == "pick_failed" and not (out / "grid.txt").exists()


def test_one_that_passed_alone_is_the_pick(tmp_path, claude):
    fake = claude({"c1-gen1": reply(SPRITE16), "c2-gen1": reply(""), "c2-gen2": reply("")})
    out = tmp_path / "run"
    assert draw(out, "--subject", "a knight", "--size", "16", "--frames", "front") == 0
    report = report_of(out)
    assert report["pick"] == {"by": "only", "best": "c1"} and "pick" not in asked(fake)
    assert [c["status"] for c in report["candidates"]] == ["passed", "failed"]


def test_batches_draw_at_half_price_and_repairs_stay_realtime(tmp_path, claude):
    fake = claude({"c1-gen1": reply(ICON_ROW), "c2-gen1": reply(""), "c1-gen1-fix1": reply(ICON_ROW_FIX),
                   "c2-gen2": reply(ICON32), "pick": reply(VERDICT)},
                  polls={"msgbatch_1": ["in_progress", "ended"]})
    out = tmp_path / "run"
    assert draw(out, "--subject", "a chest", "--kind", "icon", "--batch") == 0
    assert [(b, [cid for cid, _ in requests]) for b, requests in fake.submitted] == [
        ("msgbatch_1", ["c1-gen1", "c2-gen1"]), ("msgbatch_2", ["c2-gen2"])]
    assert all("stream" not in params for _, requests in fake.submitted for _, params in requests)
    assert asked(fake) == ["c1-gen1-fix1", "pick"]
    report = report_of(out)
    calls = {c["id"]: c for c in report["calls"]}
    assert calls["c1-gen1"]["mode"] == "batch" and calls["c1-gen1"]["batch_id"] == "msgbatch_1"
    assert calls["c1-gen1"]["cost_usd"] == api.cost_usd(calls["c1-gen1"]["usage"], batch=True)
    assert calls["c1-gen1-fix1"]["mode"] == "realtime"
    assert [(b["id"], b["collected"]) for b in report["batches"]] == [("msgbatch_1", True), ("msgbatch_2", True)]


def test_a_batch_cut_short_is_collected_on_resume_and_not_bought_twice(tmp_path, claude):
    script = {"c1-gen1": reply(SPRITE16)}
    claude(script, polls={"msgbatch_1": ["in_progress"]})
    out = tmp_path / "run"
    with pytest.raises(SystemExit) as caught:
        draw(out, "--subject", "a knight", "--size", "16", "--frames", "front", "--candidates", "1", "--batch",
             "--batch-wait", "0")
    message = str(caught.value)
    assert "batch msgbatch_1 has not ended after 0s; nothing is asked twice" in message
    assert f"`sprite-gen pixel-draw --out-dir {out.resolve()} --resume`" in message
    report = report_of(out)
    assert report["status"] == "unfinished" and report["unfinished"]["batch_id"] == "msgbatch_1"
    record = json.loads((out / "batches" / "msgbatch_1.json").read_text(encoding="utf-8"))
    assert record["collected"] is False and record["custom_ids"] == ["c1-gen1"]
    second = claude(script, known={"msgbatch_1": ["c1-gen1"]})
    assert pixel_draw.main(["--out-dir", str(out), "--resume"]) == 0
    assert second.submitted == [] and second.asked == []
    report = report_of(out)
    assert report["status"] == "ok" and report["batches"][0]["collected"] is True


def test_resume_reads_answered_calls_back(tmp_path, claude):
    claude({"c1-gen1": reply(ICON_ROW), "c1-gen1-fix1": reply(ICON_ROW_FIX)})
    out = tmp_path / "run"
    assert draw(out, "--subject", "a chest", "--kind", "icon", "--candidates", "1") == 0
    before = {name: (out / name).read_bytes() for name in ("grid.txt", "frame-icon.png")}
    again = claude({})
    assert pixel_draw.main(["--out-dir", str(out), "--resume"]) == 0
    assert again.asked == [] and {name: (out / name).read_bytes() for name in before} == before
    assert [c["id"] for c in report_of(out)["calls"]] == ["c1-gen1", "c1-gen1-fix1"]


def test_resume_takes_the_request_from_the_run(tmp_path, claude):
    claude({"c1-gen1": reply(SPRITE16)})
    out = tmp_path / "run"
    assert draw(out, "--subject", "a knight", "--size", "16", "--frames", "front", "--candidates", "1") == 0
    with pytest.raises(SystemExit, match=r"--resume reads the request from .*request.json; drop --size"):
        pixel_draw.main(["--out-dir", str(out), "--resume", "--size", "16"])
    request = json.loads((out / "request.json").read_text(encoding="utf-8"))
    (out / "request.json").write_text(json.dumps({**request, "subject": "a wizard"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="was asked with another request"):
        pixel_draw.main(["--out-dir", str(out), "--resume"])


def test_a_used_folder_is_refused(tmp_path, claude):
    out = tmp_path / "run"
    out.mkdir()
    (out / "notes.txt").write_text("mine", encoding="utf-8")
    claude({})
    with pytest.raises(SystemExit, match="exists and is not an empty folder"):
        draw(out, "--subject", "a knight")


@pytest.mark.parametrize("args,message", [
    (["--size", "7"], "--size is 8..48 pixels, got 7"),
    (["--size", "49"], "--size is 8..48 pixels, got 49"),
    (["--frames", "a,b,c,d,e"], "at most 4 frames in one call, got 5"),
    (["--frames", "a b"], "frame names are letters, digits"),
    (["--frames", "a,a"], "frame names must differ"),
    (["--frames", "a,b", "--walk-cycle"], "--walk-cycle is four frames"),
    (["--candidates", "0"], "--candidates is 1..6, got 0"),
    (["--candidates", "7"], "--candidates is 1..6, got 7"),
    (["--max-tokens", "100"], "--max-tokens is 1024..128000, got 100"),
    (["--max-tokens", "200000"], "--max-tokens is 1024..128000, got 200000"),
    (["--palette", "free:99"], "free:N takes 1..62 colours"),
])
def test_requests_refused_before_anything_is_asked(tmp_path, claude, args, message):
    fake = claude({})
    out = tmp_path / "run"
    with pytest.raises(SystemExit) as caught:
        draw(out, "--subject", "a knight", *args)
    assert message in str(caught.value) and fake.asked == [] and not out.exists()


def test_a_spec_file_holds_the_drawing(tmp_path, claude):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"subject": "a slime", "size": 16, "frames": ["front"], "palette": ["#112233", "#445566"]}),
                    encoding="utf-8")
    claude({})
    with pytest.raises(SystemExit, match=r"--size given both as a flag and in --spec"):
        draw(tmp_path / "a", "--spec", str(spec), "--size", "16")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"subject": "x", "effort": "high"}), encoding="utf-8")
    with pytest.raises(SystemExit, match=r"unknown key\(s\) effort"):
        draw(tmp_path / "b", "--spec", str(bad))
    with pytest.raises(SystemExit, match="nothing to draw"):
        draw(tmp_path / "c")
    two_colors = "```pixels\nPALETTE\n. transparent\na #112233\nb #445566\nFRAME front\n" + "\n".join(
        ["." * 16] + [".." + "ab" * 6 + ".."] * 14 + ["." * 16]) + "\nEND\n```"
    fake = claude({"c1-gen1": reply(two_colors)})
    assert draw(tmp_path / "d", "--spec", str(spec), "--candidates", "1") == 0
    prompt = fake.asked[0][1]["messages"][0]["content"]
    assert "Palette: use only these 2 colors and transparent, no others: #112233 #445566." in prompt
    assert report_of(tmp_path / "d")["candidates"][0]["gates"]["G2"]["ok"]


def test_a_json_answer_is_kept_as_a_chargrid(tmp_path, claude):
    fake = claude({"c1-gen1": reply(fixture("icon-16-json-pass.txt"))})
    out = tmp_path / "run"
    assert draw(out, "--subject", "a potion", "--kind", "icon", "--size", "16", "--format", "json", "--candidates", "1") == 0
    assert '{"palette": ["transparent", "#RRGGBB", ...]' in fake.asked[0][1]["messages"][0]["content"]
    lines = (out / "grid.txt").read_text(encoding="utf-8").splitlines()
    assert lines[:3] == ["PALETTE", ". transparent", "a #2b1b2e"]
    with Image.open(out / "frame-icon.png") as image:
        assert image.getpixel((5, 1)) == (0x2B, 0x1B, 0x2E, 255)


def test_a_walk_cycle_is_one_call_of_four_frames(tmp_path, claude):
    fake = claude({"c1-gen1": reply(WALK16)})
    out = tmp_path / "run"
    assert draw(out, "--subject", "a knight walking right", "--size", "16", "--frames", "walk1,walk2,walk3,walk4",
                "--walk-cycle", "--candidates", "1") == 0
    body = fake.asked[0][1]
    assert body["max_tokens"] == 64000
    assert ("The frames are one looping walk cycle: walk1 = contact pose with the front leg forward"
            in body["messages"][0]["content"])
    report = report_of(out)
    assert any(w.startswith("G6 weak leg alternation") for w in report["warnings"])
    assert report["files"]["gif"] == "anim.gif"
    with Image.open(out / "anim.gif") as gif:
        assert gif.n_frames == 4
    with Image.open(out / "preview-x16.png") as preview:
        assert preview.size == (4 * 256 + 3 * 32, 256)


def test_prompts_say_the_kind_and_the_format():
    def request(**changes):
        base = dict(subject="grass", size=32, kind="tile", frames=("tile",), walk_cycle=False,
                    palette=pg.palette_rule("free:8"), candidates=1, effort="high", answer_format="chargrid",
                    batch=False, max_tokens=32000, pick="vision")
        return pixel_draw.PixelRequest(**{**base, **changes})
    tile = pixel_draw.generation_prompt(request())
    assert "This is a tile: fill every pixel (no transparency) and make it repeat seamlessly" in tile
    assert "Palette: at most 8 colors plus transparent." in tile and "Background: transparent" not in tile
    assert "```pixels\nPALETTE\n. transparent" in tile and "exactly 32 lines of exactly 32 characters" in tile
    sprite = pixel_draw.generation_prompt(request(kind="sprite", frames=("down", "up"), subject="a knight"))
    assert "the feet stand on the same bottom row in every frame." in sprite and "Background: transparent." in sprite
    assert "Frames in this order: down, up." in sprite


def test_an_edited_grid_is_checked_and_rendered_without_a_call(tmp_path, claude, monkeypatch):
    claude({"c1-gen1": reply(SPRITE16)})
    first = tmp_path / "first"
    assert draw(first, "--subject", "a knight", "--size", "16", "--frames", "front", "--candidates", "1") == 0
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    edited = tmp_path / "edited.txt"
    edited.write_text((first / "grid.txt").read_text(encoding="utf-8").replace(
        "FRAME front\n......oooo......", "FRAME front\n......oooo....o."), encoding="utf-8")
    second = tmp_path / "second"
    assert pixel_draw.main(["--grid", str(edited), "--out-dir", str(second)]) == 0
    report = report_of(second)
    assert report["source"] == "grid" and report["calls"] == [] and report["cost_usd"] == 0.0
    assert report["status"] == "ok" and report["warnings"] == ["G5 1 stray pixel(s) with no opaque neighbour"]
    grid, issues = pg.read_grid_text(edited.read_text(encoding="utf-8"))
    assert (second / "frame-front.png").read_bytes() == pg.png_bytes(pg.frame_pixels(grid, "front"))
    broken = tmp_path / "broken.txt"
    broken.write_text(edited.read_text(encoding="utf-8").replace("......oooo....o.\n", "......oooo....o\n"), encoding="utf-8")
    with pytest.raises(SystemExit, match=r"does not parse whole: row_width front row 1"):
        pixel_draw.main(["--grid", str(broken), "--out-dir", str(tmp_path / "third")])
    with pytest.raises(SystemExit, match=r"--grid checks and renders a grid as it is; drop --candidates"):
        pixel_draw.main(["--grid", str(edited), "--out-dir", str(tmp_path / "fourth"), "--candidates", "2"])


def test_no_key_is_refused_before_the_folder_is_made(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out = tmp_path / "run"
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY is not set"):
        draw(out, "--subject", "a knight")
    assert not out.exists()


def test_the_cli_names_the_verb_and_its_flags():
    flags = help_options("-m", "sprite_gen.cli", "pixel-draw")
    assert {"--subject", "--spec", "--grid", "--size", "--kind", "--frames", "--walk-cycle", "--palette", "--candidates",
            "--effort", "--format", "--batch", "--batch-wait", "--max-tokens", "--pick", "--out-dir", "--resume"} <= flags
