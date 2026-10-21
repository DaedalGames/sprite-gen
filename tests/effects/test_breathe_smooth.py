# SPDX-License-Identifier: Apache-2.0
"""Breathe 레이어의 smooth 모드 — 큰 일러·영상 프레임용 부드러운 숨쉬기의 계약.

pixel(기본)은 정수 행·열 사상이라 도트 그림의 격자를 지키지만, 큰 일러에서는 머리가
통째로 오르내리고 행 복제가 칸마다 떨림으로 읽힌다. smooth 는 같은 해부(강체 경계·몸통
축·정규화 기준)를 쓰되 머리를 **자리와 바이트 그대로** 두고, 발바닥(콘텐츠 맨 아래)도
고정한 채 그 사이 몸통만 부드럽게 들어 올린다 — 어깨선이 오르고 목 밴드가 그만큼 준다.

지키는 것 (구현이 아니라 계약):
  1. 위상 0 인 칸은 원본과 바이트 동일 (첫 칸 0 — 루프·묶음 이음이 안 바뀐다)
  2. 강체 경계 위(머리)는 모든 칸에서 자리·바이트 그대로
  3. 어깨선이 depth × 정규화 기준 높이만큼 올랐다 돌아온다 — 한 루프에 breaths 회
  4. 머리·몸통 경계에 이음새·찢김이 없다 (색 기울기가 연속, 몸통 안이 끊기지 않는다)
  5. 큰 캔버스(1000px 높이)에서도 같은 계약
  6. 기본(pixel)은 이전과 바이트 동일, mode 를 안 적은 사이드카의 정규화도 그대로
  7. 사이드카 `mode`·`depth`·`breaths` 를 읽고, 모르는 값은 요란하게 거부
  8. smooth 에 lag(0 아닌 값)는 조용히 버리지 않고 거부한다
  9. 세로로만 움직이므로 화면 변에 닿은(원래 화면 밖으로 잘린) 그림도 거부하지 않는다
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from sprite_gen.curate.curation import state_breathe
from sprite_gen.effects import breathe as B
from sprite_gen.effects.anatomy import analyze
from sprite_gen.frames.extract import solid_alpha_bbox

from tests.effects.test_breathe import _dome, _humanoid, _winged

SKIN = (222, 184, 150, 255)
BODY = (170, 120, 96, 255)
EYE = (24, 18, 22, 255)
MARK = (16, 16, 16, 255)


def _bust(h: int = 120, *, gradient: bool = True, marks: tuple[float, ...] = (),
          margin: float = 0.10, touch_bottom: bool = True) -> Image.Image:
    """머리 + 목 병목 + 화면 아래 변에 닿는 몸통 — 큰 일러 idle 을 줄인 합성 도형.

    `gradient` 면 눈을 뺀 모든 불투명 픽셀의 붉은 채널이 아래로 갈수록 한결같이 오른다
    — 출력의 세로 기울기로 이음새·찢김을 읽는다. `marks` 는 콘텐츠 높이 비율 자리에 그은 1px
    어두운 가로선이다 — 그 무게중심으로 한 행이 얼마나 올랐는지를 소수 px 로 잰다.
    `margin` 은 몸통 옆 여백(캔버스 폭 비율), 0 이면 몸통이 화면 양옆 변에 닿는다."""
    w = int(h * 0.8)
    yy, xx = np.mgrid[0:h, 0:w]
    cx = w / 2.0
    body_bottom = h if touch_bottom else int(h * 0.92)
    head = ((xx + 0.5 - cx) / (h * 0.15)) ** 2 + ((yy + 0.5 - h * 0.25) / (h * 0.17)) ** 2 <= 1.0
    neck = (np.abs(xx + 0.5 - cx) <= h * 0.06) & (yy >= h * 0.38) & (yy < h * 0.48)
    half = cx - margin * w
    body = (np.abs(xx + 0.5 - cx) <= half) & (yy >= h * 0.46) & (yy < body_bottom)
    a = np.zeros((h, w, 4), np.uint8)
    figure = head | neck | body
    if gradient:
        red = (40 + np.round(180.0 * yy / max(1, h - 1))).astype(np.uint8)
        a[figure] = np.stack([red, np.full_like(red, 120), np.full_like(red, 96),
                              np.full_like(red, 255)], axis=-1)[figure]
    else:
        a[head] = SKIN
        a[(neck | body) & ~head] = BODY
    for ex in (cx - h * 0.065, cx + h * 0.065):
        eye = ((xx + 0.5 - ex) / (h * 0.025)) ** 2 + ((yy + 0.5 - h * 0.24) / (h * 0.03)) ** 2 <= 1.0
        a[eye & head] = EYE
    image = Image.fromarray(a, "RGBA")
    top = solid_alpha_bbox(image)[1]
    for frac in marks:
        row = top + int(round(frac))
        a[row, body[row]] = MARK
    return Image.fromarray(a, "RGBA")


def _smooth(**extra) -> dict:
    return {"mode": "smooth", "depth": 0.025, "breaths": 2, **extra}


def _cfg(raw: dict) -> dict:
    """사이드카 값 하나를 굽기와 같은 정규화로 — 굽기가 받는 cfg 그대로."""
    return state_breathe({"states": {"idle": {"breathe": raw}}}, "idle")


def _mark_row(frame: Image.Image, near: float, column: int, reach: int = 4) -> float:
    """`near` 근처 어두운 선의 무게중심 행 (소수) — 한 열 위에서 어두움으로 가중."""
    a = np.asarray(frame.convert("RGBA"), np.float64)
    lo, hi = int(near) - reach, int(near) + reach + 1
    col = a[lo:hi, column, :3].mean(axis=1)
    weight = np.clip(sum(BODY[:3]) / 3.0 - col, 0.0, None)   # 몸통 색 = 0, 표시선 쪽으로 선형
    assert weight.sum() > 0, "표시선을 못 찾았다"
    return float((weight * np.arange(lo, hi)).sum() / weight.sum())


def _phase_frames(src: Image.Image, cfg: dict, count: int):
    frames, phases = B.bake_breathe_sequence([src] * count, cfg)
    return frames, phases


# ── 1. 첫 칸 0 ──────────────────────────────────────────────────────

def test_a_cell_at_phase_zero_is_byte_identical_to_its_source() -> None:
    src = _bust()
    frames, phases = _phase_frames(src, _cfg(_smooth()), 24)
    zero = [i for i, p in enumerate(phases) if p == 0.0]
    assert zero == [0, 12], f"breaths 2 / 24칸이면 위상 0 은 0·12번 칸이다: {zero}"
    for i in zero:
        assert frames[i].tobytes() == src.tobytes(), f"{i}번 칸: 위상 0 인데 원본과 다르다"
    assert any(f.tobytes() != src.tobytes() for f in frames), "숨을 한 번도 안 쉬었다"


# ── 2. 머리 자리·바이트 고정 ───────────────────────────────────────

@pytest.mark.parametrize("h", [120, 1000], ids=["small", "1000px"])
def test_the_head_stays_in_place_byte_for_byte(h: int) -> None:
    src = _bust(h)
    anat = analyze(src)
    top = solid_alpha_bbox(src)[1]
    cut = top + anat.rigid_row                     # 이 위가 강체(머리) 구간
    cfg = _cfg(_smooth())
    expect = np.asarray(src)[:cut].tobytes()
    for phase in (0.1, B.INHALE, 0.55, 0.8):
        got = B.phase_frame(src, cfg, phase, anat)
        assert np.asarray(got)[:cut].tobytes() == expect, \
            f"위상 {phase}: 강체 경계({cut}행) 위가 움직였다 — 머리는 자리·바이트 그대로여야 한다"
        assert got.tobytes() != src.tobytes(), f"위상 {phase}: 몸통이 안 움직였다"


def test_the_head_line_is_the_reference_cells_for_the_whole_loop() -> None:
    """칸마다 머리 윗선이 흔들려도(머리카락 한 올이 솟는 칸) 경계는 줄의 기준 칸(첫 칸)에서 잰
    자리에 있다 — 해부가 줄 한 벌이듯 그 자리도 한 벌이다. 칸마다 그 칸의 bbox 로 다시 놓으면
    경계 바로 위 행이 칸에 따라 움직인다 (영상 루프는 머리 윗선이 칸마다 한두 px 흔들린다)."""
    base = _bust()
    top = solid_alpha_bbox(base)[1]
    cut = top + analyze(base).rigid_row
    strand = np.asarray(base).copy()
    strand[top - 3:top, base.width // 2] = BODY      # 머리 위로 솟은 한 올 — bbox 윗선이 3px 오른다
    frames = [base, Image.fromarray(strand, "RGBA")] * 6
    baked, _ = B.bake_breathe_sequence(frames, _cfg(_smooth()))
    for i, (src, got) in enumerate(zip(frames, baked)):
        assert np.asarray(got)[:cut].tobytes() == np.asarray(src)[:cut].tobytes(), \
            f"{i}번 칸: 기준 칸의 강체 경계({cut}행) 위가 움직였다"


# ── 3. depth 만큼 올랐다 돌아온다, breaths 회 ──────────────────────

def _shoulder_marked(h: int = 120):
    probe = _bust(h, gradient=False)
    anat = analyze(probe)
    shoulder = B.smooth_shoulder_row(anat)
    src = _bust(h, gradient=False, marks=(shoulder,))
    anat = analyze(src)
    top = solid_alpha_bbox(src)[1]
    return src, anat, top + int(round(shoulder))


@pytest.mark.parametrize("h", [120, 1000], ids=["small", "1000px"])
def test_the_shoulder_line_rises_by_depth_and_settles_back(h: int) -> None:
    src, anat, row = _shoulder_marked(h)
    column = solid_alpha_bbox(src)[0] + anat.axis_x
    depth = 0.025
    lift = depth * anat.basis_rows
    reach = int(lift) + 4                           # 선이 오르는 만큼 창을 넓힌다
    rest = _mark_row(src, row, column, reach)
    for phase in (0.0, 0.2, B.INHALE, 0.7, 0.95):
        got = B.phase_frame(src, _cfg(_smooth(depth=depth)), phase, anat)
        rise = rest - _mark_row(got, row, column, reach)
        want = lift * B.smooth_wave(phase)
        assert rise == pytest.approx(want, abs=0.3), \
            f"위상 {phase}: 어깨선이 {rise:.2f}px 올랐다 — {want:.2f}px 여야 한다 (depth {depth} × 기준 {anat.basis_rows}행)"
    assert B.smooth_wave(0.0) == 0.0 and B.smooth_wave(B.INHALE) == 1.0


def test_a_loop_breathes_exactly_breaths_times() -> None:
    src, anat, row = _shoulder_marked()
    column = solid_alpha_bbox(src)[0] + anat.axis_x
    rest = _mark_row(src, row, column)
    for breaths, count in ((1, 18), (2, 24), (3, 30)):
        frames, _ = _phase_frames(src, _cfg(_smooth(breaths=breaths)), count)
        rises = [rest - _mark_row(f, row, column) for f in frames]
        assert rises[0] == 0.0
        peaks = [i for i in range(count)
                 if rises[i] > rises[i - 1] and rises[i] >= rises[(i + 1) % count] and rises[i] > 0.5]
        assert len(peaks) == breaths, f"breaths {breaths}: 정점 {len(peaks)}개 — {np.round(rises, 2)}"
        assert min(rises) >= -0.05, "어깨선이 쉼 자리 아래로 내려갔다 — 숨은 올랐다 돌아올 뿐이다"


# ── 4. 경계에 이음새·찢김 없음 ─────────────────────────────────────

@pytest.mark.parametrize("h", [120, 1000], ids=["small", "1000px"])
def test_no_seam_or_tear_between_the_head_and_the_body(h: int) -> None:
    src = _bust(h)
    anat = analyze(src)
    box = solid_alpha_bbox(src)
    column = box[0] + anat.axis_x
    slope = 180.0 / (h - 1)
    cfg = _cfg(_smooth())
    for phase in (0.25, B.INHALE, 0.6):
        got = np.asarray(B.phase_frame(src, cfg, phase, anat)).astype(int)
        start = box[1]                              # 정수리부터 발끝까지 — 경계를 가로지른다
        red = got[start:box[3], column, 0]
        steps = np.diff(red)
        assert steps.min() >= 0, f"위상 {phase}: 기울기가 거꾸로 섰다 — 행이 접혔다"
        assert steps.max() <= slope * (1 + B.MAX_ROW_STRAIN) + 1.0, \
            f"위상 {phase}: 한 행에 {steps.max()} 뛰었다 — 이음새(행 빠짐)다"
        for y in range(start, box[3]):              # 몸통 안이 끊기지 않는다
            alpha = got[y, :, 3]
            inside = np.nonzero(alpha)[0]
            if inside.size:
                run = alpha[inside[0]:inside[-1] + 1]
                assert (run > 0).all(), f"위상 {phase} {y}행: 몸통 안에 구멍이 났다"


# ── 5. 큰 캔버스 (위 2·3·4 의 1000px 판이 같은 계약을 본다) ─────────
# ── 6. pixel 기본은 바이트 같다 ─────────────────────────────────────

# 이 플랜 전(59b2738)의 굽기 결과 — breaths 2, 12칸. 기본 경로가 한 바이트라도 바뀌면 깨진다.
PIXEL_GOLDEN = {
    "_humanoid": "4f212002c8cec96f4b207b56eec5195b2549069bbc066511385e3a92c74c1fcc",
    "_winged": "5bdf0c0997cf799bc7e3a82374dce16fb65db3545fb3a5f1f1df7b249fcbaf0d",
    "_dome": "492db9827109db3cf48d655437d102f22b1db4843d7121c4f4c65ea3a9fee443",
}
PIXEL_KEYS = ["anatomy", "axis_x", "breaths", "depth", "depth_x", "lag", "rigid_row", "torso_half"]


@pytest.mark.parametrize("build", [_humanoid, _winged, _dome], ids=lambda b: b.__name__)
@pytest.mark.parametrize("mode", [None, "pixel"], ids=["mode-omitted", "mode-pixel"])
def test_the_default_pixel_mode_is_byte_identical_to_before(build, mode) -> None:
    raw = {"depth": 0.06, "breaths": 2, "lag": 0.1}
    if mode:
        raw["mode"] = mode
    cfg = _cfg(raw)
    if mode is None:
        assert sorted(cfg) == PIXEL_KEYS, f"mode 를 안 적은 사이드카의 정규화가 바뀌었다: {sorted(cfg)}"
    frames, _ = B.bake_breathe_sequence([build()] * 12, cfg)
    digest = hashlib.sha256(b"".join(f.tobytes() for f in frames)).hexdigest()
    assert digest == PIXEL_GOLDEN[build.__name__], f"{build.__name__}: pixel 굽기가 바뀌었다"


# ── 7. 사이드카 ─────────────────────────────────────────────────────

def test_the_sidecar_reads_mode_depth_and_breaths() -> None:
    cfg = _cfg({"mode": "smooth", "depth": 0.02, "breaths": 2})
    assert cfg["mode"] == "smooth" and cfg["depth"] == 0.02 and cfg["breaths"] == 2
    assert cfg["depth_x"] == 0.0, "smooth 는 depth_x 를 생략하면 가로 그대로다"
    assert "lag" not in cfg, "smooth 에는 진행파가 없다"
    omitted = _cfg({"mode": "smooth", "breaths": 2})
    assert omitted["depth"] == B.DEFAULT_SMOOTH_DEPTH
    assert _cfg({"mode": "pixel", "breaths": 2})["mode"] == "pixel"


@pytest.mark.parametrize("bad", ["soft", "", 1, ["smooth"]])
def test_an_unknown_mode_is_refused_loudly(bad) -> None:
    with pytest.raises(SystemExit, match="mode"):
        _cfg({"mode": bad, "breaths": 2})


def test_the_sidecar_picks_the_smooth_warp() -> None:
    """사이드카가 smooth 면 굽기가 정말 smooth 로 굽는다 — 원본에 없던 사잇값 색이 생긴다.
    pixel 은 원본 픽셀을 옮기기만 해서 새 색이 없다."""
    src = _bust()
    colors = {tuple(p) for p in np.asarray(src).reshape(-1, 4)}
    smooth, _ = B.bake_breathe_sequence([src] * 8, _cfg(_smooth()))
    pixel, _ = B.bake_breathe_sequence([src] * 8, _cfg({"depth": 0.06, "breaths": 2}))
    new_smooth = {tuple(p) for f in smooth for p in np.asarray(f).reshape(-1, 4)} - colors
    new_pixel = {tuple(p) for f in pixel for p in np.asarray(f).reshape(-1, 4)} - colors
    assert new_smooth, "smooth 사이드카인데 사잇값이 하나도 없다 — pixel 로 구웠다"
    assert not new_pixel


# ── 8. lag ──────────────────────────────────────────────────────────

def test_a_lag_on_smooth_is_refused_not_dropped() -> None:
    with pytest.raises(SystemExit, match="lag"):
        _cfg(_smooth(lag=0.1))
    with pytest.raises(SystemExit, match="lag"):
        B.phase_frame(_bust(), {**_smooth(), "lag": 0.1}, 0.3)
    assert "lag" not in _cfg(_smooth(lag=0))        # 0 은 smooth 가 하는 일 그대로라 받는다


# ── 9. 화면 변 ──────────────────────────────────────────────────────

def test_a_figure_cut_by_the_frame_is_not_refused() -> None:
    """세로로만 움직이므로 아무것도 화면 밖으로 밀리지 않는다 — 머리는 제자리, 발바닥(아래
    변)도 제자리이고 그 사이의 모든 원본 행이 화면 안 어딘가에 그려진다. 그래서 pixel 의
    '셀 밖으로 나가 잘린다' 거부를 물려받지 않는다."""
    src = _bust(margin=0.0)                          # 양옆·아래 변에 닿는다
    anat = analyze(src)
    w, h = src.size
    for phase in (0.2, B.INHALE, 0.7):
        got = np.asarray(B.phase_frame(src, _cfg(_smooth()), phase, anat))
        assert (got[-1, :, 3] == np.asarray(src)[-1, :, 3]).all(), "아래 변이 비었다 — 발바닥이 떴다"
        for col in (0, w - 1):                       # 원본이 변까지 찬 행은 그대로 찬다
            rows = np.nonzero(np.asarray(src)[:, col, 3] == 255)[0]
            assert rows.size and (got[rows, col, 3] == 255).all(), f"{col}열: 옆 변이 비었다"


def test_a_sideways_swell_off_the_canvas_is_still_refused() -> None:
    """가로 부풀림(depth_x)은 옆으로 민다 — 변에 닿은 그림을 거기서 어떻게 할지는 이
    모드가 정하지 않는다. 밀려 나가는 불투명 픽셀이 있으면 pixel 과 같이 멈춘다."""
    src = _bust(margin=0.0)
    with pytest.raises(SystemExit, match="셀 밖"):
        B.phase_frame(src, _cfg(_smooth(depth_x=0.05)), B.INHALE)
    roomy = _bust(margin=0.12)
    got = B.phase_frame(roomy, _cfg(_smooth(depth_x=0.05)), B.INHALE)
    before = solid_alpha_bbox(roomy)
    after = solid_alpha_bbox(got)
    assert after[2] - after[0] > before[2] - before[0], "depth_x 를 줬는데 몸통이 안 넓어졌다"


def test_a_strain_past_the_cap_is_refused() -> None:
    with pytest.raises(SystemExit, match="smooth"):
        B.phase_frame(_bust(), _cfg(_smooth(depth=0.2)), B.INHALE)


# ── CLI: 같은 값, 같은 함수 ────────────────────────────────────────

def _blinking(h: int = 120, count: int = 12) -> list[Image.Image]:
    """칸마다 조금씩 다른 프레임 — 눈을 감았다 뜨는 칸이 섞인 영상 루프의 축소판."""
    frames = []
    for i in range(count):
        im = _bust(h)
        if i in (5, 6):
            a = np.asarray(im).copy()
            a[a[..., 0] == EYE[0]] = SKIN
            im = Image.fromarray(a, "RGBA")
        frames.append(im)
    return frames


@pytest.mark.parametrize("raw,flags", [
    (_smooth(), ["--mode", "smooth", "--depth", "0.025", "--breaths", "2"]),
    ({"depth": 0.06, "breaths": 2, "lag": 0.1},
     ["--depth", "0.06", "--breaths", "2", "--lag", "0.1"]),
], ids=["smooth", "pixel"])
def test_the_cli_bakes_what_the_curation_bake_bakes(tmp_path, raw, flags) -> None:
    from sprite_gen import cli

    frames = _blinking()
    src = tmp_path / "loop"
    src.mkdir()
    for i, im in enumerate(frames):
        im.save(src / f"frame-{i:03d}.png")
    out = tmp_path / "out"
    assert cli.main(["breathe", str(src), str(out), *flags]) == 0
    expect, phases = B.bake_breathe_sequence(frames, _cfg(raw))
    got = sorted(out.glob("frame-*.png"))
    assert [p.name for p in got] == [f"frame-{i:03d}.png" for i in range(len(frames))]
    for i, (path, want) in enumerate(zip(got, expect)):
        with Image.open(path) as im:
            assert im.convert("RGBA").tobytes() == want.tobytes(), f"{i}번 칸이 큐레이션 굽기와 다르다"
    report = json.loads((out / "breathe.json").read_text(encoding="utf-8"))
    assert report["config"] == json.loads(json.dumps(_cfg(raw)))
    assert report["phases"] == phases
    assert report["resolved"]["anatomy"]["rigid_row"] == analyze(frames[0]).rigid_row


def test_the_cli_refuses_what_the_sidecar_refuses(tmp_path) -> None:
    from sprite_gen import cli

    src = tmp_path / "loop"
    src.mkdir()
    _bust().save(src / "frame-000.png")
    with pytest.raises(SystemExit, match="lag"):
        cli.main(["breathe", str(src), str(tmp_path / "a"), "--mode", "smooth", "--lag", "0.1"])
    with pytest.raises(SystemExit, match="depth"):
        cli.main(["breathe", str(src), str(tmp_path / "b"), "--depth", "0.5"])
    with pytest.raises(SystemExit):
        cli.main(["breathe", str(src), str(src), "--mode", "smooth"])   # 원본 위에 덮어쓰지 않는다
