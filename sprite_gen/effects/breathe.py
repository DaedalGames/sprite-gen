# SPDX-License-Identifier: Apache-2.0
"""결정론 호흡(idle breathing) — 봉투 기반 스쿼시&스트레치 (변형 강도 장).

호흡은 프레임 선택(깜빡임)과 직교하는 변조 레이어다 (maintainer 확정 2026-07-18).
설정은 curation.json 사이드카 `states.<state>.breathe` (curation.state_breathe 가
정규화 SSoT), 굽기는 compose/GIF 가 재생 시퀀스 위에 이 모듈의 수학으로 한다.
AI 개입 0 — 같은 입력이면 항상 같은 출력.

## 왜 분할선이 아니라 봉투인가 (2026-07-25 교체)

구 방식은 수평 분할선 위의 행을 정수 px 내리는 것이었다. 실제 출력은 **2상태 토글**
이라(측정: idle GIF 6프레임이 `f0==f2==f4`, `f1==f3==f5`) 호흡이 아니라 1px 진동으로
읽혔다. 머리를 잘라 붙이는 변형으로 바꿔봐도 잘린 단면이 목에 가로선으로 드러났다.

붙이는 연산이 있는 한 이음매는 없앨 수 없다. 그래서 **자르지 않는다.** 스프라이트
전체에 연속 변형장 하나를 걸고 그 강도를 목(또는 얼굴 아래)에서 0 으로 떨군다:

    env(y) = 발끝램프(y) x 강체테이퍼(y)          # 0 = 원본 그대로, 1 = 최대 변형
    g(y,t) = depth * norm * wave(t - lag*u(y)) * env(y)
    sx(y)  = 1 + g                                 # 가로
    sy(y)  = 1 / (1 + g)                           # 세로 (면적 보존)

## 불변식 (테스트가 지키는 것)

1. **강체 구간은 항등이다 — 근사가 아니다.** env=0 인 행은 g=0 -> 가로 사상이 원본
   좌표 그대로, sy=1 -> 누적 높이가 정확히 1씩 증가 -> 행 복제/삭제 0.
   그래서 그 구간은 프레임 간 **비트 동일**하다. 눈·입이 몇 도트뿐인 픽셀아트에서
   3% 세로 신장도 표정을 뭉개므로 이게 핵심 계약이다.
2. **가로 사상은 단조다.** X(x) = ∫ dens(s) ds 이고 dens > 0 이므로 접힘이 구조적으로
   불가능하다. 몸통만 1+g, 부속(날개)은 1 -> 날개는 밖으로 밀리기만 하고 안 늘어난다.
3. **몸통 축 열이 가로 사상의 고정점이다.** 출력 위치를 축의 적분값 기준으로 잡으므로
   축은 언제나 제자리이고, 그래서 (a) 변형이 0 이면 사상이 진짜 항등이 되고
   (b) 배율이 변해도 스프라이트가 통째로 좌우로 튀지 않는다. bbox 중앙을 기준으로
   잡으면 축이 중앙과 다른 캐릭터가 변형 0 에서도 밀린다 (회귀 2026-07-25 validator 검증).
4. **격자를 벗어나는 연산이 없다.** 세로는 행 정수 복제/삭제, 가로는 정수 열 사상.
   보간·블렌딩·비정수 스케일이 한 번도 일어나지 않으므로 출력이 항상 정수 도트다.
   언페이크(pitch 검출)를 프레임마다 다시 할 필요가 없는 이유이자, 다시 하면 안 되는
   이유이기도 하다 (synthetic_fixture_a up_run frame-2, 2026-07-22 하모닉 오검출 붕괴).
5. **루프 길이 불변.** 출력 프레임 수 = 입력 프레임 수. 호흡이 그 안에서 breaths 회
   딱 떨어진다.

(구 테이크 방식 — exhale 프레임을 raw 테이크로 굽고 전체 재추출 — 은 v1.56.31 에서
폐기: 깜빡임×호흡 조합 불가 + 적용마다 수 분 + 팔레트 드리프트. 봉투 방식은 그 셋 중
무엇도 해소하지 않으므로 되돌릴 근거가 없다.)

## smooth 모드 (큰 일러·영상 프레임)

위 불변식은 도트 그림의 것이다. 큰 일러에서는 같은 워프가 머리를 통째로 오르내리게 하고
(발이 고정이라 몸통이 늘면 그 위가 다 뜬다), 정수 행 복제가 칸마다 떨림으로 읽힌다. 가로줄을
통째로 올리는 것도 숨이 아니다 — 어깨와 가슴이 같이 올라 위아래로만 움직이는 것처럼 보인다.
`mode: "smooth"` 는 같은 해부를 쓰되 2차원 변위장 하나를 굽는다:

    가슴: 몸통 축 위 가슴 가운데의 타원 cos² 무게로 depth x 기준 높이만큼 오르고 depth_x 만큼 부푼다
          (몸이 옆을 보면 chest_tilt·chest_bias·chest_dir 로 타원을 눕히고 앞쪽을 키우고 앞-위로 보낸다)
    어깨: 어깨선 좌우가 가슴 오름의 shoulder 배만큼, shoulder_lag 만큼 늦게 오른다
    목 밴드(강체 경계 ~ 어깨선)에서 켜지고 콘텐츠 맨 아래에서 꺼진다

정사상(원본 점의 내용이 가는 자리)의 역을 고정점 반복으로 풀어 쌍선형으로 뜬다(곱한 알파).
강체 경계 위(목·턱 위)와 무게 밖은 원본 바이트 그대로다. 한 호흡은 0 에서 시작해 들숨(INHALE)에
1, 날숨에 0 으로 돌아오는 한쪽 혹이고 어깨의 늦음도 양 끝에서 0 이라, 위상 0 인 칸은 원본과
바이트 같다. 가슴 무게가 몸통 안에서 0 으로 떨어져 화면 변에 닿은 그림도 밖으로 밀리지 않는다.
줄 전체가 기준 칸(첫 칸)에서 잰 해부와 그 자리를 쓴다 — 영상 루프의 머리 윗선이 칸마다 흔들려도
경계는 움직이지 않는다.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from sprite_gen.effects.anatomy import Anatomy, analyze
from sprite_gen.curate.curation import (BREATHE_MODES, BREATHE_SMOOTH_DEPTH, BREATHE_SMOOTH_DEPTH_X,
                                        BREATHE_SMOOTH_SHAPE, normalize_transform, state_breathe)
from sprite_gen.frames.extract import solid_alpha_bbox

# 강체 경계에서 변형 강도가 0 이 되기까지의 테이퍼 반폭 (콘텐츠 높이 비율).
# 이 테이퍼가 이음매를 없앤다 — 경계에서 강도가 계단이 아니라 경사로 떨어진다.
TAPER = 0.055
# 발끝 고정 램프 — 변형 구간 아래쪽 이 비율만큼은 강도가 0 에서 올라온다 (발이 안 뜬다).
FOOT = 0.28
# 행당 변형 상한. 넘으면 조용히 깎지 않고 멈춘다 (No Silent Fallback).
MAX_ROW_STRAIN = 0.25
# 기본 진폭 — 몸통 높이 대비 총 신장량.
DEFAULT_DEPTH = 0.06
# 진행파 위상지연 (호흡 주기 비율) — 몸통 윗부분이 늦게 밀어올려 머리가 한 박자 늦게 온다.
DEFAULT_LAG = 0.10
# 한 호흡이 부드럽게 읽히려면 필요한 최소 프레임 수 (프레임을 찍어낼 때의 권고치).
SMOOTH_CYCLE_FRAMES = 6
# smooth 모드 — 들숨이 한 호흡에서 차지하는 비율 (들숨이 날숨보다 짧다).
INHALE = 0.40
DEFAULT_SMOOTH_DEPTH = BREATHE_SMOOTH_DEPTH
DEFAULT_SMOOTH_DEPTH_X = BREATHE_SMOOTH_DEPTH_X
SMOOTH_SHAPE = dict(BREATHE_SMOOTH_SHAPE)
# smooth 의 어깨 혹 — 몸통 축에서 좌우로 몸통 반폭의 이 배 자리, 가로 반지름도 몸통 반폭 비율.
SHOULDER_X = 0.55
SHOULDER_RX = 0.45


# ── 파형 ────────────────────────────────────────────────────────────

def wave(t: float) -> float:
    """한 호흡의 기본 파형. 2차 하모닉을 살짝 섞어 정점이 뾰족해진다(들숨의 어택)."""
    return 0.86 * math.sin(2 * math.pi * t) + 0.14 * math.sin(4 * math.pi * t)


def smoothstep(a: float, b: float, x: float) -> float:
    if b <= a:
        return 1.0 if x >= b else 0.0
    u = min(1.0, max(0.0, (x - a) / (b - a)))
    return u * u * (3 - 2 * u)


# ── 변형 강도 봉투 ──────────────────────────────────────────────────

def envelope(anat: Anatomy, taper: float = TAPER, foot: float = FOOT):
    """(env(u), 진폭 정규화 계수) — u = 0(발바닥) .. 1(정수리).

    정규화: 테이퍼·발끝램프가 변형 구간을 갉아먹으므로 그대로 두면 강체 경계가 낮은
    캐릭터일수록 호흡이 작아진다. sum(env) 로 나눠 **총 신장량이 항상
    depth × 기준높이**가 되게 한다 — 진폭 숫자가 캐릭터와 무관하게 같은 뜻을 갖는다."""
    height = anat.height
    ru = anat.rigid_u
    band = max(1.5, taper * height) / max(1, height)
    foot_top = foot * ru

    def env(u: float) -> float:
        return smoothstep(0.0, foot_top, u) * (1.0 - smoothstep(ru - band, ru + band, u))

    total = sum(env(j / max(1, height - 1)) for j in range(height))
    norm = anat.basis_rows / total if total > 1e-6 else 0.0
    return env, norm


def protect(anat: Anatomy):
    """부속 보호 가중 p(x) — 1 이면 그 열은 가로로 안 늘어난다 (콘텐츠 bbox 상대 x).

    자동 검출 밴드는 부속(날개·긴 팔)이 실재할 때만 켜지고 램프 끝을 max_half 에
    앵커한다. **수동 밴드(큐레이터 영역 UI)는 무조건 켜지고 램프를 밴드 자체에
    앵커한다** — 블롭(max_half ≈ torso_half)에서 자동 램프는 몸 끝까지 완만해 밴드를
    아무리 좁혀도 보호가 1 에 도달하지 못했다 (실측 2026-07-30 synthetic silhouette 문어:
    밴드 12→4 에 출력 차이가 바이트 22개, maintainer "버그로 보여"). 사람이 밴드를 그렸으면
    그 밖은 밀리기만 해야 한다."""
    cx = anat.axis_x
    if anat.torso_source == "manual":
        t0 = float(anat.torso_half)
        t1 = t0 + 2.0
        return lambda x: smoothstep(t0, t1, abs(x - cx))
    if not anat.has_appendage:
        return lambda x: 0.0
    t0 = anat.torso_half * 1.15
    t1 = max(t0 + 1.0, anat.max_half * 0.95)
    return lambda x: smoothstep(t0, t1, abs(x - cx))


def row_strain(anat: Anatomy, depth: float) -> float:
    """행당 최대 변형 — MAX_ROW_STRAIN 초과 여부 판정용 (관측 가능하게 노출)."""
    env, norm = envelope(anat)
    peak = max(env(j / max(1, anat.height - 1)) for j in range(anat.height))
    return depth * norm * peak


# ── 외곽선 1px 다듬기 ───────────────────────────────────────────────

def _dark(p) -> bool:
    """다듬기의 '외곽선 어두움' 판정 — 외곽선과 어두운 음영을 갈라야 한다.

    실측(문어 idle 왼쪽 볼, 2026-07-30): 외곽선 lum 0~13, 어두운 음영 73~92, 내부 112+.
    구 문턱 96 은 음영까지 외곽선으로 세어 (외곽,외곽,음영) 3연속이 '전부 어두움' 이
    되니 다듬기 패턴이 발화하지 않아 볼의 바깥 검은 열이 잔존했다. 60 이 두 군집
    사이를 가른다."""
    return bool(p[3]) and (0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]) < 60.0


def _thin_outline_1px(out: Image.Image) -> None:
    """안쪽점 기준 다듬기를 고정점까지 반복 — 한 패스의 제거가 새 돌출점을 만들 수 있다.

    실측(문어 idle 위상 0): 2점 패턴의 바깥점 (2,13) 을 지우면 남은 (3,13) 이 이웃 행보다
    1px 튀어나온 새 돌출점이 되는데, 단일 패스 스냅샷은 그걸 못 본다. 각 패스가 불투명
    픽셀을 단조 감소시키므로 종료가 보장된다 (JS 미러 동일)."""
    while _thin_outline_pass(out):
        pass


def _thin_outline_pass(out: Image.Image) -> int:
    """워프된 프레임의 실루엣 외곽선 2px 를 **안쪽점 기준** 1px 로 정규화하는 1패스.

    세로 스쿼시&스트레치는 정수 행 복제/삭제라, 가파른 대각선 외곽선이 국소적으로 2px
    두께가 되며 위상마다 위치를 바꿔 "간헐 2줄 검은점" 으로 읽힌다 (실측 2026-07-30:
    문어 idle 왼쪽 볼, maintainer 발견). 각 열/행의 실루엣 끝에서 '어두움-어두움-내부색'
    3연속을 만나면 **바깥 픽셀을 제거(알파 0)** 해 이웃 행들과 정렬된 **안쪽** 선만
    남긴다 — 초기 구현은 안쪽을 내부색으로 되돌려 바깥점을 선으로 남겼는데, 돌출한
    바깥점이 이웃 행 선과 어긋나 여전히 2줄로 읽혔다 (maintainer 실기기 판정 2026-07-30:
    "안쪽 검은점 기준이 더 정확하다").

    안전 필터 — 제거가 새 결함을 만들면 안 된다:
      · 제거된 자리의 수직(가로 패턴)/수평(세로 패턴) 이웃이 불투명 내부색이면 제거로
        실루엣 끝에 내부색이 노출된다 → 제외.
      · 이웃 양쪽이 제거되지 않는 어두운 픽셀이면 어두운 선 **중간에 구멍**이 난다 → 제외.
      · 이웃이 같은 패스에서 같이 제거되는 후보면 안전(돌출 열/행이 통째로 사라짐).
      제외가 다른 후보의 안전성을 바꾸므로 고정점까지 반복한다(결정론 — 정렬 순회).

    호출자는 **변형이 실제로 일어난 프레임에만** 부른다(_warp 의 deformed 게이트). 변형 0
    프레임은 이 함수를 타지 않아 "변형 0 = 원본 동일" 계약이 유지된다. 원본 스냅샷에서만
    읽고 한 번에 적용하므로 순회 순서와 무관하게 결과가 같다(JS 미러 바이트 동일 성질)."""
    src = out.load()
    w, h = out.size
    op = lambda x, y: 0 <= x < w and 0 <= y < h and bool(src[x, y][3])
    dark = lambda x, y: op(x, y) and _dark(src[x, y])
    # 후보 수집: (바깥픽셀 좌표) -> 패턴 축 ('h' = 행 끝 패턴, 'v' = 열 끝 패턴)
    cand: dict[tuple[int, int], str] = {}
    for x in range(w):
        ys = [y for y in range(h) if op(x, y)]
        if not ys:
            continue
        t0, b0 = ys[0], ys[-1]
        if t0 + 2 <= b0 and dark(x, t0) and dark(x, t0 + 1) and op(x, t0 + 2) and not dark(x, t0 + 2):
            cand[(x, t0)] = "v"
        if b0 - 2 >= t0 and dark(x, b0) and dark(x, b0 - 1) and op(x, b0 - 2) and not dark(x, b0 - 2):
            cand[(x, b0)] = "v"
    # 행 끝 좌표 맵 (돌출점 판정용)
    row_lo: dict[int, int] = {}
    row_hi: dict[int, int] = {}
    for y in range(h):
        xs = [x for x in range(w) if op(x, y)]
        if not xs:
            continue
        row_lo[y], row_hi[y] = xs[0], xs[-1]
        l0, r0 = xs[0], xs[-1]
        if l0 + 2 <= r0 and dark(l0, y) and dark(l0 + 1, y) and op(l0 + 2, y) and not dark(l0 + 2, y):
            cand.setdefault((l0, y), "h")
        if r0 - 2 >= l0 and dark(r0, y) and dark(r0 - 1, y) and op(r0 - 2, y) and not dark(r0 - 2, y):
            cand.setdefault((r0, y), "h")
    # 1px 단독 돌출점 — 그 행의 어두운 끝점이 위아래 행 끝보다 정확히 1px 바깥으로 튀어나온
    # 경우(가로 매핑 반올림이 행마다 갈린 아티팩트). 2점 패턴이 아니라 위 후보에 안 잡힌다.
    # 점을 지우고 **안쪽 자리에 그 색을 그대로 그린다** — 이웃 행들과 정렬된 안쪽 선이 된다.
    # 안전: 돌출 정의상 (x, y±1) 은 투명이라(이웃 행 끝이 더 안쪽) 제거로 노출·구멍이 없다.
    # 가로(좌우) 한정 — 세로(정수리) 1px 계단은 작가 의도(머리카락 스파이크류)와 구분이
    # 안 돼 건드리지 않는다.
    moves: dict[tuple[int, int], tuple[int, int, int, int]] = {}   # 안쪽좌표 -> 칠할 색
    for y in sorted(row_lo):
        up_lo, dn_lo = row_lo.get(y - 1), row_lo.get(y + 1)
        m = row_lo[y]
        if (up_lo is not None and dn_lo is not None and min(up_lo, dn_lo) - m == 1
                and dark(m, y) and op(m + 1, y)):
            cand.setdefault((m, y), "h")
            if not dark(m + 1, y):
                moves[(m + 1, y)] = src[m, y]
        up_hi, dn_hi = row_hi.get(y - 1), row_hi.get(y + 1)
        r = row_hi[y]
        if (up_hi is not None and dn_hi is not None and r - max(up_hi, dn_hi) == 1
                and dark(r, y) and op(r - 1, y)):
            cand.setdefault((r, y), "h")
            if not dark(r - 1, y):
                moves[(r - 1, y)] = src[r, y]
    # 안전 필터 고정점: 제거 대상에서 빠진 픽셀은 이웃 후보의 kept-dark 가 되므로 반복.
    drop = set(cand)
    changed = True
    while changed:
        changed = False
        for p in sorted(drop):
            x, y = p
            neigh = ((x, y - 1), (x, y + 1)) if cand[p] == "h" else ((x - 1, y), (x + 1, y))
            kept_dark = 0
            ok = True
            for nx, ny in neigh:
                if not op(nx, ny) or (nx, ny) in drop:
                    continue
                if dark(nx, ny):
                    kept_dark += 1
                else:
                    ok = False        # 내부색이 실루엣 끝으로 노출된다
                    break
            if not ok or kept_dark >= 2:  # 어두운 선 중간 구멍
                drop.discard(p)
                changed = True
                break
    for x, y in drop:
        out.putpixel((x, y), (0, 0, 0, 0))
    applied = len(drop)
    for (x, y), color in moves.items():
        if (x, y) not in drop:
            out.putpixel((x, y), color)
            applied += 1
    return applied


# ── 워프 ────────────────────────────────────────────────────────────

def _warp(frame: Image.Image, anat: Anatomy, depth: float, lag: float, t: float,
          depth_x: float | None = None) -> Image.Image:
    """한 위상 t 의 프레임 — 캔버스 크기 불변, 발바닥 고정.

    세로는 행 국소 배율을 누적해 각 원본 행이 출력에서 몇 행을 차지하는지 정하고,
    가로는 행 안에서 위치마다 다른 밀도를 적분해 단조 사상을 만든다. 둘 다 정수
    연산이라 출력이 정수 도트다.

    `depth_x` 는 가로 성분의 독립 진폭이다 (maintainer 요청 2026-07-30 — 가로/세로 분리).
    None 이면 depth 를 따른다(레거시와 바이트 동일), 0 이면 가로 사상이 전 위상
    항등이다. 파형·봉투·지연은 두 축이 공유한다 — 축마다 다른 건 스칼라뿐이다."""
    box = solid_alpha_bbox(frame)
    if not box:
        return frame.copy()
    body = frame.convert("RGBA").crop(box)
    src = body.load()
    width, height = body.size
    canvas_w, canvas_h = frame.size
    anchor_x = box[0] + anat.axis_x
    baseline = box[3]

    env, norm = envelope(anat)
    p_of = protect(anat)
    ru = anat.rigid_u
    dx_amp = depth if depth_x is None else depth_x

    def gain(u: float, d: float) -> float:
        e = env(u)
        if e <= 0.0:
            return 0.0
        return d * norm * wave(t - lag * min(1.0, u / max(1e-6, ru))) * e

    # 세로 누적 — j=0 이 정수리
    heights: list[float] = []
    acc = 0.0
    for j in range(height):
        g = gain(1.0 - j / max(1, height - 1), depth)
        acc += 1.0 if g == 0.0 else 1.0 / (1.0 + g)
        heights.append(acc)
    total = max(1, math.floor(acc + 0.5))

    out = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    dst = out.load()
    y_cursor = baseline - total
    prev = 0
    clipped = 0
    deformed = False   # 이 위상에서 변형이 실제로 일어났나 (다듬기 게이트)
    for j in range(height):
        u = 1.0 - j / max(1, height - 1)
        cur = math.floor(heights[j] + 0.5)
        reps = max(0, cur - prev)
        prev = cur
        if reps == 0:
            continue
        g = gain(u, dx_amp)                  # 가로 성분 — 독립 진폭 (depth_x)
        if reps != 1:
            deformed = True                  # 행 복제/삭제 = 세로 변형이 실재
        if g == 0.0:
            # 변형 없음 = 원본 위치 그대로. 축 고정점 사상의 g->0 극한과 정확히 같다.
            row_map = [(box[0] + i, i) for i in range(width)]
        else:
            deformed = True                  # 가로 사상 변형이 실재
            dens = [max(0.05, 1.0 + g * (1.0 - p_of(i))) for i in range(width)]
            edge = [0.0]
            for d in dens:
                edge.append(edge[-1] + d)
            origin = edge[anat.axis_x]       # 축을 고정점으로 — 여기가 anchor_x 에 박힌다
            # half-up 반올림: 파이썬 기본 round() 는 은행가 반올림(round(0.5)==0)이라
            # JS Math.round(half-up) 미러와 .5 경계에서 갈린다. 규칙을 명시해 고정한다.
            lo = math.floor(edge[0] - origin + 0.5)
            hi = math.floor(edge[width] - origin + 0.5)
            row_map = []
            i = 0
            for ox in range(lo, hi):
                while i < width - 1 and (edge[i + 1] - origin) <= ox:
                    i += 1
                row_map.append((anchor_x + ox, i))
            # 외곽선 보존: 가로 축소 위상에서 forward 밀도매핑이 실루엣 양끝 열을 통째로
            # 떨궈 1px 외곽선이 사라진다(발끝 플리커, 실측 2026-07-30: 문어 idle y25~y28
            # 이 날숨 위상마다 소실). 이 행의 최말단 불투명 소스 열(= 외곽선)을 출력의
            # 양끝 불투명 픽셀에 그대로 실어 항상 1px 외곽선이 남게 한다 — 원본 외곽선
            # 픽셀을 복사하므로 색·밝기가 정확히 보존되고(재도색 아님) 위상 간 잔여
            # 깜빡임이 없다. g==0 항등 분기(위)는 손대지 않아 "변형 0 = 원본 동일" 계약이
            # 그대로 산다. 축소가 아니면(양끝이 이미 외곽선을 실으면) no-op 이다.
            op = [k for k in range(width) if src[k, j][3]]
            if op and row_map:
                op_lo, op_hi = op[0], op[-1]
                for k in range(len(row_map) - 1, -1, -1):
                    if src[row_map[k][1], j][3]:
                        row_map[k] = (row_map[k][0], op_hi)
                        break
                for k in range(len(row_map)):
                    if src[row_map[k][1], j][3]:
                        row_map[k] = (row_map[k][0], op_lo)
                        break
        for r in range(reps):
            yy = y_cursor + r
            for ox, si in row_map:
                pixel = src[si, j]
                if not pixel[3]:
                    continue
                if 0 <= yy < canvas_h and 0 <= ox < canvas_w:
                    dst[ox, yy] = pixel
                else:
                    clipped += 1
        y_cursor += reps
    if clipped:
        # 조용히 자르지 않는다 — 정수리나 옆구리가 셀 밖으로 나가면 스프라이트가
        # 망가지고, 그건 여백이나 진폭을 사람이 정해야 하는 문제다 (No Silent Fallback).
        raise SystemExit(
            f"breathe: 늘어난 프레임이 셀 밖으로 나가 불투명 픽셀 {clipped}개가 잘린다 "
            f"(셀 {canvas_w}x{canvas_h}, 콘텐츠 {box[0]},{box[1]}-{box[2]},{box[3]}). "
            f"셀 여백을 늘리거나 depth 를 낮춰라.")
    if deformed:
        _thin_outline_1px(out)
    return out


# ── smooth 모드 (큰 일러·영상 프레임) ───────────────────────────────

def breathe_mode(cfg: dict) -> str:
    """cfg 의 굽기 방식 — 생략은 pixel. 모르는 값을 조용히 pixel 로 굽지 않는다."""
    mode = cfg.get("mode")
    if mode is None:
        return "pixel"
    if not isinstance(mode, str) or mode not in BREATHE_MODES:
        raise SystemExit(f"breathe: mode {mode!r} 를 모른다 — {' | '.join(BREATHE_MODES)} 중 하나다.")
    return mode


def smooth_wave(t: float) -> float:
    """smooth 의 한 호흡 — 위상 0 에서 0, 들숨 끝(INHALE)에 1, 날숨 끝에 다시 0.

    pixel 의 `wave` 는 위아래로 흔드는 사인이고 진행파 지연까지 있어 위상 0 에서도 윗행이
    움직인다. 여기는 쉼(0)에서 올랐다 돌아오기만 하는 한쪽 혹이고, 양 끝과 정점에서 기울기가
    0 이라 루프 이음에서 꺾이지 않는다."""
    t = t % 1.0
    g = 0.5 * t / INHALE if t < INHALE else 0.5 + 0.5 * (t - INHALE) / (1.0 - INHALE)
    return 0.5 - 0.5 * math.cos(2.0 * math.pi * g)


def smooth_shoulder_wave(t: float, lag: float) -> float:
    """어깨의 호흡 — 가슴보다 `lag` 만큼 늦게 정점에 닿되 위상 0 과 1 에서는 가슴과 같이 0.

    `wave(t - lag)` 로 늦추면 위상 0 에서 어깨가 아직 날숨 끝에 걸려 있어 첫 칸이 원본이 아니게
    된다. 시간을 `t - lag·sin(πt)` 로 휘면 양 끝은 그대로 두고 가운데만 늦출 수 있다 (lag < 1/π
    이면 단조라 되감기지 않는다)."""
    t = t % 1.0
    return smooth_wave(t - lag * math.sin(math.pi * t))


def smooth_band(anat: Anatomy) -> tuple[float, float]:
    """(강체 경계, 목 밴드 끝) — 콘텐츠 bbox 기준 행. 그 사이가 목 밴드다: 경계에서 0 이던 움직임이
    여기서 켜진다(목 너비 안은 밴드 끝까지 천천히). 폭은 pixel 테이퍼의 전폭(경계 위아래 각 TAPER)이고
    전부 경계 **아래**에 둔다 — 경계 위(목·턱 위)는 한 픽셀도 움직이지 않아야 한다."""
    top = float(anat.rigid_row)
    return top, top + 2.0 * max(1.5, TAPER * anat.height)


def smooth_shoulder_row(anat: Anatomy) -> float:
    """어깨선 (콘텐츠 bbox 기준 행) — 어깨 혹의 가운데, 목 밴드의 위 4할."""
    top, end = smooth_band(anat)
    return top + 0.4 * (end - top)


def _neck(anat: Anatomy) -> tuple[float, float]:
    """(목 가운데 열, 목 반폭) — 해부가 강체 경계 행에서 잰 구간. 손으로 만든 해부처럼 구간이
    없으면 몸통 축에 몸통 반폭의 1/4 로 둔다."""
    if anat.neck_span is not None:
        lo, hi = anat.neck_span
        return 0.5 * (lo + hi + 1), max(1.0, 0.5 * (hi - lo + 1))
    return anat.axis_x + 0.5, max(1.0, 0.25 * anat.torso_half)


def _shape(cfg: dict) -> dict:
    return {k: (SMOOTH_SHAPE[k] if cfg.get(k) is None else float(cfg[k])) for k in SMOOTH_SHAPE}


def smooth_chest(anat: Anatomy, cfg: dict) -> tuple[float, float]:
    """가슴 가운데 (콘텐츠 bbox 기준 x, y) — 몸통 축 위, 목 아래 몸통의 `chest_y` 비율 높이."""
    top = float(anat.rigid_row)
    return anat.axis_x + 0.5, top + _shape(cfg)["chest_y"] * (anat.height - top)


def _smoothstep_array(a: float, b: float, x: np.ndarray) -> np.ndarray:
    if b <= a:
        return (x >= b).astype(np.float64)
    u = np.clip((x - a) / (b - a), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def smooth_displacement(anat: Anatomy, cfg: dict, x: np.ndarray, y: np.ndarray,
                        chest: float, shoulders: float) -> tuple[np.ndarray, np.ndarray]:
    """콘텐츠 bbox 기준 점 (x, y) 의 내용이 가는 변위 (dx, dy) — 정사상, 위로 오르면 dy < 0.

    가슴: 가운데에 놓인 타원(가로 chest_rx × 몸통 반폭, 세로 chest_ry × 목 아래 몸통, chest_tilt 만큼
    기울임)의 cos² 무게로 `depth × 기준 높이` 만큼 chest_dir 쪽(0 = 바로 위)으로 가고, 그 안이 가운데에서
    바깥으로 `depth_x` 만큼 부푼다. chest_bias 는 무게를 긴 축 따라 한쪽으로 키운다 (몸이 향한 앞쪽).
    어깨: 목 밴드 높이, 몸통 축에서 좌우로 몸통 반폭의 SHOULDER_X 배 자리에 놓인 둥근 혹 둘이
    가슴 오름의 `shoulder` 배만큼 오른다 — 가슴 무게가 이미 덮은 자리는 그만큼 덜 (가운데가 두 번
    오르지 않게). 목과 어깨 사이(옷깃)는 어느 혹에도 안 들어 그대로다.
    고정: 강체 경계 위는 0. 목 너비 안은 목 밴드 전체에 걸쳐 천천히 켜지고(목은 거의 그대로), 그
    밖은 밴드의 0.15 만에(눌림이 행당 상한을 넘지 않는 한) 켜진다 — 목 옆에 높이 붙은 어깨도 숨을
    쉰다. 콘텐츠 맨 아래에서 다시 0 이다 (발바닥 고정)."""
    shape = _shape(cfg)
    top, band_end = smooth_band(anat)
    band = band_end - top
    height = float(anat.height)
    torso = float(anat.torso_half)
    cx, cy = smooth_chest(anat, cfg)
    neck_x, neck_half = _neck(anat)
    rx = shape["chest_rx"] * torso
    ry = shape["chest_ry"] * (height - top)
    # 몸이 향한 쪽: 타원을 기울이고(chest_tilt), 긴 축 따라 한쪽을 키우고(chest_bias), 가슴이 가는 쪽을
    # 바로 위에서 돌린다(chest_dir). 셋 다 0 이면 아래 식은 기울이기 전 식과 비트 같다 (곱 1·0 과 합 0).
    tilt = math.radians(shape["chest_tilt"])
    along = (x - cx) * math.cos(tilt) + (y - cy) * math.sin(tilt)
    across_axis = (y - cy) * math.cos(tilt) - (x - cx) * math.sin(tilt)
    r = np.sqrt((along / rx) ** 2 + (across_axis / ry) ** 2)
    w_round = np.where(r < 1.0, np.cos(0.5 * np.pi * np.minimum(r, 1.0)) ** 2, 0.0)
    w_chest = w_round * (1.0 + shape["chest_bias"] * np.clip(along / rx, -1.0, 1.0))
    heading = math.radians(shape["chest_dir"])
    neckness = 1.0 - _smoothstep_array(neck_half, 1.5 * neck_half, np.abs(x - neck_x))
    across = np.minimum(1.0, np.abs(np.abs(x - cx) - SHOULDER_X * torso) / (SHOULDER_RX * torso))
    down = np.minimum(1.0, np.abs(y - smooth_shoulder_row(anat)) / (1.4 * band))   # 양 어깨 높이가 달라도
    w_shoulder = np.cos(0.5 * np.pi * down) ** 2 * np.cos(0.5 * np.pi * across) ** 2
    lift = float(cfg.get("depth") if cfg.get("depth") is not None else DEFAULT_SMOOTH_DEPTH) * anat.basis_rows
    swell = float(cfg.get("depth_x") if cfg.get("depth_x") is not None else DEFAULT_SMOOTH_DEPTH_X)
    # 어깨가 켜지는 구간은 오르는 양의 8배 이상 — 더 짧으면 그 사이가 행당 상한 넘게 눌린다.
    quick = max(0.15 * band, 8.0 * shape["shoulder"] * lift)
    u = np.clip((y - top) / (quick + (band - quick) * neckness), 0.0, 1.0) if band > quick else \
        np.clip((y - top) / quick, 0.0, 1.0)
    guard = u * u * (3.0 - 2.0 * u) * (1.0 - _smoothstep_array(height - 0.5 * band, height, y))
    rise = lift * (chest * w_chest * math.cos(heading) + shape["shoulder"] * shoulders * w_shoulder * (1.0 - w_round))
    forward = lift * chest * w_chest * math.sin(heading)
    puff = swell * chest * w_chest
    return (puff * (x - cx) + forward) * guard, (puff * (y - cy) - rise) * guard


def smooth_strain(anat: Anatomy, cfg: dict) -> tuple[float, float]:
    """(행당 최대 변형, 최소 면적비) — 가슴과 어깨가 함께 정점일 때의 변위장 기울기에서.

    행당 변형은 늘고 주는 성분(∂dx/∂x, ∂dy/∂y)의 최대다 — pixel 의 행당 변형과 같은 뜻. 목 옆에서
    어깨가 오르며 생기는 비틀림(옆 칸끼리 오르는 양의 차)은 늘고 줄지 않으니 여기 세지 않고,
    대신 사상이 접히지 않는지를 면적비 det(I + ∇D) 로 본다."""
    ys = np.arange(anat.height, dtype=np.float64) + 0.5
    xs = np.arange(anat.width, dtype=np.float64) + 0.5
    if ys.size < 2 or xs.size < 2:
        return 0.0, 1.0
    gx, gy = np.meshgrid(xs, ys)
    dx, dy = smooth_displacement(anat, cfg, gx, gy, 1.0, 1.0)
    dxy, dxx = np.gradient(dx)
    dyy, dyx = np.gradient(dy)
    stretch = float(max(np.abs(dxx).max(), np.abs(dyy).max()))
    area = float(((1.0 + dxx) * (1.0 + dyy) - dxy * dyx).min())
    return stretch, area


def _smooth_warp(frame: Image.Image, anat: Anatomy, cfg: dict, t: float) -> Image.Image:
    """smooth 의 한 위상 — 캔버스 크기 불변, 강체 경계 위와 발바닥 고정.

    정사상(원본 점 p 의 내용이 p + D(p) 로 간다)의 역을 고정점 반복(q = o - D(q))으로 풀고, 그
    자리를 곱한 알파에서 쌍선형으로 뜬다. 변위가 행당 상한(MAX_ROW_STRAIN) 아래라 반복은
    수축이다. 움직이지 않는 픽셀(경계 위, 무게 밖)은 원본 바이트를 그대로 둔다 — 곱하고 나누는
    왕복조차 거치지 않는다."""
    rgba = frame.convert("RGBA")
    chest = smooth_wave(t)
    shoulders = smooth_shoulder_wave(t, _shape(cfg)["shoulder_lag"])
    box = solid_alpha_bbox(rgba)
    if not box or (chest == 0.0 and shoulders == 0.0):
        return rgba                                  # 위상 0 — 원본 그대로 (바이트)
    if anat.origin is not None:
        # 줄의 기준 칸에서 잰 자리 — 해부가 한 벌이듯 경계·축·발바닥의 자리도 한 벌이다.
        # 영상 루프는 머리 윗선이 칸마다 흔들려, 칸마다 다시 놓으면 경계 위 행이 움직인다.
        box = (anat.origin[0], anat.origin[1],
               anat.origin[0] + anat.width, anat.origin[1] + anat.height)
    src = np.asarray(rgba)
    canvas_h, canvas_w = src.shape[:2]
    # 움직일 수 있는 행: 강체 경계 아래 ~ 콘텐츠 맨 아래 (그 밖은 변위가 0 이다).
    y_lo = max(0, int(math.floor(box[1] + anat.rigid_row)))
    y_hi = min(canvas_h, box[1] + anat.height)
    if y_hi <= y_lo:
        return rgba
    oy, ox = np.mgrid[y_lo:y_hi, 0:canvas_w].astype(np.float64)
    oy += 0.5 - box[1]                               # 콘텐츠 bbox 기준 픽셀 중심
    ox += 0.5 - box[0]

    def field(x, y):
        return smooth_displacement(anat, cfg, x, y, chest, shoulders)

    # 밀려 나가는 내용 — 원본 픽셀 중심이 가는 자리가 캔버스 밖이면 조용히 자르지 않는다.
    fx, fy = field(ox, oy)
    land_x, land_y = ox + fx + box[0], oy + fy + box[1]
    gone = (land_x < 0.0) | (land_x >= canvas_w) | (land_y < 0.0) | (land_y >= canvas_h)
    clipped = int(np.count_nonzero(src[y_lo:y_hi, :, 3][gone]))
    if clipped:
        raise SystemExit(
            f"breathe(smooth): 가슴 부풂이 불투명 픽셀 {clipped}개를 셀 밖으로 민다 "
            f"(셀 {canvas_w}x{canvas_h}, 콘텐츠 {box[0]},{box[1]}-{box[2]},{box[3]}). "
            f"chest_rx 나 depth_x 를 줄여라 — 기본 모양은 가슴 무게가 몸통 안에서 0 이 된다.")
    qx, qy = ox.copy(), oy.copy()
    for _ in range(8):
        dx, dy = field(qx, qy)
        qx, qy = ox - dx, oy - dy
    moved = (qx != ox) | (qy != oy)
    if not moved.any():
        return rgba
    pm = src.astype(np.float64) / 255.0
    pm[..., :3] *= pm[..., 3:4]
    sx = qx[moved] + box[0] - 0.5                    # 캔버스 인덱스 좌표
    sy = qy[moved] + box[1] - 0.5
    x0, y0 = np.floor(sx).astype(np.int64), np.floor(sy).astype(np.int64)
    ax, ay = (sx - x0)[:, None], (sy - y0)[:, None]
    x0c, x1c = np.clip(x0, 0, canvas_w - 1), np.clip(x0 + 1, 0, canvas_w - 1)
    y0c, y1c = np.clip(y0, 0, canvas_h - 1), np.clip(y0 + 1, 0, canvas_h - 1)
    val = ((pm[y0c, x0c] * (1.0 - ax) + pm[y0c, x1c] * ax) * (1.0 - ay)
           + (pm[y1c, x0c] * (1.0 - ax) + pm[y1c, x1c] * ax) * ay)
    alpha = val[:, 3:4]
    rgb = np.where(alpha > 0.0, val[:, :3] / np.maximum(alpha, 1e-12), 0.0)
    out = src.copy()
    region = out[y_lo:y_hi]
    region[moved] = np.clip(np.floor(np.concatenate([rgb, alpha], axis=1) * 255.0 + 0.5), 0, 255).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def _smooth_phase_frame(frame: Image.Image, cfg: dict, phase: float, anat: Anatomy) -> Image.Image:
    lag = cfg.get("lag")
    if lag is not None and float(lag) != 0.0:
        # 진행파는 윗행 위상을 늦춰 위상 0 인 칸도 움직인다 — 조용히 버리지 않는다.
        raise SystemExit(f"breathe(smooth): lag {lag} — smooth 에는 진행파가 없다. 위상 0 인 칸이 원본 "
                         f"그대로여야 루프·묶음 이음이 안 바뀐다. lag 를 지워라 (어깨가 늦는 건 shoulder_lag).")
    strain, area = smooth_strain(anat, cfg)
    if strain > MAX_ROW_STRAIN or area < 1.0 - MAX_ROW_STRAIN:
        raise SystemExit(
            f"breathe(smooth): 행당 변형 {strain:.3f}·최소 면적비 {area:.3f} — 상한 {MAX_ROW_STRAIN} 을 넘는다. "
            f"가슴·어깨가 오르는 만큼을 목 밴드와 가슴 무게가 못 받는다 (강체 경계 {anat.rigid_row}/{anat.height}, "
            f"기준 {anat.basis_rows}행). depth·depth_x 를 낮추거나 chest_rx·chest_ry 를 넓혀라. 조용히 깎지 않는다.")
    return _smooth_warp(frame, anat, cfg, phase)


# ── 해부 결과 얼리기 / 자가 복구 ────────────────────────────────────

def _fnv1a(data: bytes) -> int:
    """FNV-1a 32비트 — **JS 미러가 똑같이 계산할 수 있어야 해서** 이 해시를 쓴다.

    SHA-256 은 브라우저에서 동기로 못 구한다(SubtleCrypto 는 async 라 렌더 루프에서 못 쓴다).
    미러가 지문을 못 구하면 프리뷰는 자기가 그리는 프레임이 얼린 해부와 맞는지 확인할
    방법이 없고, 그러면 굽기만 자가 복구하고 프리뷰는 낡은 숫자로 계속 그린다
    (validator 실측 2026-07-25: 픽셀 편집 후 최대 617바이트, 불투명 픽셀 수까지 불일치).
    충돌 위험은 "프레임이 바뀌었나" 판정에는 무시할 수준이다."""
    h = 2166136261
    for b in data:
        h = ((h ^ b) * 16777619) & 0xFFFFFFFF
    return h


def _num(value) -> str:
    """키에 들어갈 수의 정규형 — 파이썬과 JS 가 **같은 문자열**을 내야 한다.

    `f"{v:.6f}"` 와 `v.toFixed(6)` 는 6번째 자리가 정확히 반올림 경계일 때 갈린다(파이썬은
    십진 표현 기준 round-half-even, JS 는 이진값 기준). `repr` 계열도 지수 표기가 다르다
    (`1e-07` vs `1e-7`). 그래서 **부동소수 곱·합·floor 만** 쓴다 — 양쪽 다 IEEE754 double 이라
    같은 입력에 같은 결과가 나오는 연산이다."""
    if value is None:
        return "~"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return str(math.floor(float(value) * 1000000.0 + 0.5))
    return str(value)


def _canon(value) -> str:
    """키에 들어갈 값의 정규 직렬화 — dict 는 키 정렬, 수는 `_num`."""
    if isinstance(value, dict):
        return "{" + ",".join(f"{k}={_canon(value[k])}" for k in sorted(value, key=str)) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_canon(v) for v in value) + "]"
    return _num(value)


def reference_key(*, state: str, variant: str, request_stamp: str, source_index: int,
                  source_stamp: str, pixel_ops, transform) -> str:
    """기준 프레임의 **정체성** — 그 프레임을 만드는 입력들로만 만든다. 픽셀은 안 읽는다.

    예전엔 이게 *변형 결과 RGBA* 의 해시였다. 그런데 굽기는 `apply_transform` 에서
    부드럽게 리샘플하고(당시 BICUBIC, 지금 `transform_cell`; `snap_scale` 없는 런 = 기본 런 전부) 웹뷰 캔버스는
    `imageSmoothingEnabled=false`, 즉 **NEAREST** 다. 같은 원본에 같은 변형을 걸어도 두 쪽이
    만드는 그림이 다르니 지문은 **영구 불일치**였다 — 회전·확대가 걸린 줄은 프리뷰가 영원히
    원본으로 떨어지고 영상 내보내기가 영구 차단됐으며, 안내대로 "해부를 갱신" 해도 라우트가
    또 BICUBIC 프레임으로 같은 숫자를 만들어 **절대 안 풀렸다** (validator 실측 2026-07-26:
    rotate 3°에서 555px 상이, 정수 이동만 걸린 줄은 우연히 일치).

    입력을 해시하면 리샘플러가 식에서 아예 빠진다. 여기 들어가는 값은 전부 사이드카(사람의
    의도)와 서버가 준 파일 스탬프라, 웹뷰가 **동기로 정확히** 아는 것들이다. 덤으로 웹뷰가
    신선도 판정에 캔버스를 읽을 이유가 없어져서 "어느 캔버스를 쟀나" 계열 결함(9·10라운드)이
    구조적으로 사라진다.

    실패 방향도 안전하다: 런을 복사해 mtime 이 바뀌면 키가 달라져 **시끄럽게 거부**하고 한 번
    다시 잰다 — 조용히 낡은 숫자로 굽는 반대 방향이 아니다.

    ## 왜 이 재료들인가 (그리고 셀·snap 이 왜 없는가)

    재료는 전부 웹뷰가 **동기로 정확히** 아는 것이어야 한다. 사이드카 값(변종·픽셀편집·변형)은
    웹뷰가 소유한 canon 이고, 파일 스탬프 둘은 서버가 상태 응답에 실어 보낸다.

    셀 크기와 `snap_scale` 은 뺐다 — 둘 다 `sprite-request.json` 과 변종의 순수 함수인데,
    변종은 이미 키에 있고 request 는 `request_stamp` 로 통째로 덮인다. 파생값을 또 넣으면
    양쪽이 그 파생을 각자 재구현해야 하고(웹뷰의 측정 k 는 `pixel_snap_scale` 과 다른
    검출기다) 거기서 드리프트가 난다. 원인을 덮는 편이 결과를 흉내내는 것보다 안전하다."""
    # 정규화를 **키 안에서** 한다. 호출부에 맡기면 한쪽은 사이드카 원문(`{"rotate": 3}`)을,
    # 다른 쪽은 항등 머지본(7키 전부)을 넘겨 같은 프레임이 다른 키를 갖는다 — 변형이 없는
    # 프레임(대부분)이 정확히 그 모양이었다.
    return "|".join((
        "breathe-ref-v1", str(state), str(variant), str(request_stamp),
        str(int(source_index)), str(source_stamp),
        _canon(pixel_ops or {}), _canon(normalize_transform(transform)),
    ))


def anatomy_fingerprint(key: str) -> str:
    """기준 프레임 키의 지문 — 변종·소스 인덱스는 읽을 수 있게 남기고 나머지는 해시한다.

    키를 그대로 쓰면 거부 메시지는 친절해지지만 사이드카가 픽셀 편집량만큼 부푼다(칠한
    점 하나가 키 한 조각이다). 앞머리 두 조각만 남겨 "어느 변종의 몇 번 프레임" 은
    보이게 하고, 입력 전체는 해시로 접는다."""
    parts = key.split("|")
    variant, index = (parts[2], parts[4]) if len(parts) > 5 else ("?", "?")
    return f"{variant}:{index}:{_fnv1a(key.encode('utf-8')):08x}"


def resolve_anatomy(reference: Image.Image, cfg: dict) -> Anatomy:
    """**줄 전체가 공유하는 한 벌**을 기준 프레임에서 확정한다.

    해부는 **캐릭터의 속성이지 프레임의 속성이 아니다.** 깜빡임 프레임이라고 목이 옮겨
    가지 않는다. 프레임마다 다시 재면 검출 지터로 `rigid_row` 가 흔들리고(실측 2↔3),
    그러면 "강체 구간" 이 프레임 간 **같은 구간이 아니게 된다** — 이 모듈의 핵심 계약이
    프레임별 재검출 때문에 깨지는 것이다. 게다가 사이드카에는 해부가 한 벌뿐이라
    프리뷰(미러)는 그 한 벌로 그리고 굽기는 프레임마다 다른 값으로 구워 갈렸다
    (validator 실측 2026-07-25: 골든 픽스처 idle 4프레임에서 최대 220바이트, 불투명 픽셀
    **수**까지 불일치 — 위치가 아니라 그림이 달랐다).

    그래서 지문은 **기준 프레임 하나**를 지킨다: 스프라이트가 바뀌면 여기서 잡혀 한 번
    다시 재고 그 사실을 보고한다. 나머지 프레임이 서로 다른 것은 정상이며 재검출 사유가
    아니다.

    `rigid_row` 는 **사람의 의도(입력)** 이고 `anatomy` 는 거기서 파생된 **캐시**다.
    그래서 얼린 값이 override 와 어긋나면 캐시가 낡은 것이므로 다시 잰다 — 얼린 값을
    그대로 쓰면 사람이 고친 숫자가 조용히 버려진다 (validator 검증 2026-07-25: cfg 33 을
    줘도 얼린 23 이 구워졌고 경고도 없었다).

    **굽기는 얼린 해부를 쓰지 않는다 — 매번 자기 기준 프레임에서 다시 잰다.** 굽기는
    진짜 프레임을 손에 들고 있으니 재는 게 언제나 옳고, 얼린 값은 웹뷰가 미리보기를
    그리려고 들고 있는 **캐시**일 뿐이다 (C11: `rigid_row` 는 의도, `anatomy` 는 캐시).
    예전엔 여기서 얼린 값을 신뢰할지 판정하려고 프레임 RGBA 를 해시했는데, 그 해시가
    웹뷰와 영구 불일치를 만드는 원흉이었다 — `reference_key` 주석 참조.

    사이드카 캐시가 실제로 구운 값과 어긋났는지는 `anatomy_report` 가 대조해서 보고한다.
    "재검출했나(proxy)" 보다 "**사이드카와 다른 값을 구웠나**" 가 원칙 6이 요구하는 바로
    그 관측이다."""
    return analyze(reference, rigid_row=cfg.get("rigid_row"),
                   axis_x=cfg.get("axis_x"), torso_half=cfg.get("torso_half"))


def anatomy_report(images: list[Image.Image], cfg: dict) -> dict:
    """굽기가 실제로 쓴 해부를 manifest 에 실을 형태로 — 자가 복구를 관측 가능하게.

    줄 전체가 **한 벌**을 쓰므로 보고도 한 벌이다. 예전엔 프레임별로 재서
    `rigid_row_varies` 를 실어 보냈는데, 그건 관측이 아니라 결함의 증상이었다 —
    경계가 프레임마다 움직이면 강체 구간이 프레임 간 같은 구간이 아니다."""
    if not images:
        return {"anatomy": None, "matches_sidecar": True, "warnings": []}
    anat = resolve_anatomy(images[0], cfg)
    baked = anat.as_dict()
    frozen = cfg.get("anatomy")
    # 사이드카가 들고 있던 캐시와 실제로 구운 값의 **대조**. 캐시가 없으면 대조할 게 없다.
    # 지문 비교가 아니라 값 비교인 게 핵심이다 — 지문은 웹뷰가 자기 미리보기의 신선도를
    # 판정하는 도구이고, 여기서 궁금한 건 "사이드카 숫자와 다른 그림을 구웠나" 다.
    stale = None
    if isinstance(frozen, dict):
        stale = {k: [frozen.get(k), v] for k, v in baked.items()
                 if k in frozen and frozen.get(k) != v}
    return {
        "anatomy": baked,
        "matches_sidecar": not stale,
        "sidecar_drift": stale or None,
        "warnings": sorted(anat.warnings),
    }


def freeze_anatomy(frame: Image.Image, cfg: dict, key: str) -> dict:
    """큐레이션 시점에 해부를 1회 돌려 사이드카에 얼릴 dict 를 만든다.

    이걸 사이드카에 넣어두면 굽기와 큐레이터 웹뷰가 같은 숫자를 읽는다 — 웹뷰는
    검출을 재구현하지 않고 warp 만 미러링하면 된다 (검출 중복 제거)."""
    anat = analyze(frame, rigid_row=cfg.get("rigid_row"),
                   axis_x=cfg.get("axis_x"), torso_half=cfg.get("torso_half"))
    return {**anat.as_dict(), "fingerprint": anatomy_fingerprint(key)}


# ── 재생 시퀀스 계약 (compose/GIF 진입점) ───────────────────────────

def fit_breathe_pattern(seq_len: int, cfg: dict) -> list[float]:
    """시퀀스 길이에 딱 맞는 호흡 위상 시퀀스 (길이 == seq_len, 루프 불변).

    위상은 [0, 1) 의 연속 값이다 — 구 방식의 정수 분할선 단계가 아니다. breaths 회가
    시퀀스 안에서 정확히 반복되므로 루프 이음매가 없고, 등분 보정도 필요 없다.

    **정수 나머지를 먼저 취한다.** `(i*breaths/seq_len) % 1.0` 로 쓰면 수학적으로 같은
    위상이 서로 다른 double 이 되어(18슬롯 3호흡: 유니크 6 -> 14) 아틀라스 칸 재사용이
    표현 노이즈로 깨진다. 분자를 정수로 접고 한 번만 나누면 반복 위상이 비트 동일하다
    (validator 검증 2026-07-25: 시트 1344x192 -> 576x192)."""
    if seq_len <= 0:
        return []
    breaths = max(1, int(cfg.get("breaths", 1)))
    return [((i * breaths) % seq_len) / seq_len for i in range(seq_len)]


def fitted_breath_count(seq_len: int, cfg: dict) -> int:
    """실제 적용되는 호흡 횟수 — 연속 위상이라 요청값이 그대로 성립한다."""
    if seq_len <= 0:
        return 0
    return max(1, int(cfg.get("breaths", 1)))


def recommended_breathe_frames(cfg: dict, per_cycle: int = SMOOTH_CYCLE_FRAMES) -> int:
    """호흡이 부드럽게 읽히려면 필요한 최소 재생-시퀀스 길이.

    정지 1컷 + 링크 복제로 프레임을 찍어내는 레시피(정지자세)는 이 값으로 복제 수를
    정한다: `clones = recommended_breathe_frames(cfg) - 1`. 유저가 이미 가진 프레임
    수를 줄이거나 호흡 횟수를 바꾸지 않는다 — 오직 프레임을 새로 만들 때의 목표다."""
    return max(1, int(cfg.get("breaths", 1))) * max(2, per_cycle)


def breathe_reads_smoothly(seq_len: int, cfg: dict, per_cycle: int = SMOOTH_CYCLE_FRAMES) -> bool:
    """요청한 호흡 횟수가 이 시퀀스 길이에서 부드럽게 렌더되는가 (관측용)."""
    if seq_len <= 0:
        return False
    return seq_len // max(1, int(cfg.get("breaths", 1))) >= max(2, per_cycle)


def phase_frame(frame: Image.Image, cfg: dict, phase: float,
                anat: Anatomy | None = None) -> Image.Image:
    """프레임에 호흡 위상 하나(0 <= phase < 1)를 적용.

    `anat` 를 주면 그것을 쓴다 — 줄 전체가 한 벌을 공유해야 하므로 호출자가 한 번
    확정해 넘기는 게 정석이다. 생략하면 이 프레임을 기준으로 확정한다 (단발 호출용).
    `cfg["mode"]` 가 smooth 면 smooth 로 굽는다 — GIF·아틀라스·레이어·CLI 가 모두 여기를 탄다."""
    if anat is None:
        anat = resolve_anatomy(frame, cfg)
    if breathe_mode(cfg) == "smooth":
        return _smooth_phase_frame(frame, cfg, phase, anat)
    depth = float(cfg.get("depth", DEFAULT_DEPTH))
    raw_dx = cfg.get("depth_x")
    depth_x = None if raw_dx is None else float(raw_dx)
    for axis, d in (("depth", depth), ("depth_x", depth_x)):
        if d is None:
            continue
        strain = row_strain(anat, d)
        if strain > MAX_ROW_STRAIN:
            raise SystemExit(
                f"breathe: 행당 변형({axis}) {strain:.3f} > 상한 {MAX_ROW_STRAIN} — 변형 구간이 너무 "
                f"좁다 (강체 경계 {anat.rigid_row}/{anat.height}, 정규화 기준 {anat.basis_row}). "
                f"{axis} 를 낮추거나 rigid_row 를 올려라. 조용히 깎지 않는다.")
    return _warp(frame, anat, depth, float(cfg.get("lag", DEFAULT_LAG)), phase, depth_x=depth_x)


def bake_breathe_sequence(images: list[Image.Image], cfg: dict) -> tuple[list[Image.Image], list[float]]:
    """재생 시퀀스에 호흡 레이어를 굽는다 -> (프레임들, 적용 위상들).

    출력 길이 = 입력 길이 (루프 불변 — 루프는 기존 프레임 그대로, 호흡이 그 안에서
    breaths 회 딱 떨어진다)."""
    if not images:
        return images, []
    phases = fit_breathe_pattern(len(images), cfg)
    anat = resolve_anatomy(images[0], cfg)        # 줄 전체가 한 벌을 공유한다
    return [phase_frame(images[i], cfg, phases[i], anat) for i in range(len(images))], phases


# ── CLI: 프레임 폴더에 굽기 (`sprite-gen breathe`) ─────────────────

def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("frames", help="folder of the loop's frames (*.png, played in name order)")
    parser.add_argument("out", help="new folder for the breathing frames (same names) and breathe.json")
    parser.add_argument("--mode", choices=BREATHE_MODES, default=None,
                        help="pixel (default): integer rows and columns, for pixel art; smooth: the head stays "
                             "in place, the body rises smoothly, for large illustrations and video frames")
    parser.add_argument("--depth", type=float, default=None,
                        help="how far the body rises, as a share of its height below the neck "
                             f"(default {DEFAULT_DEPTH} pixel, {DEFAULT_SMOOTH_DEPTH} smooth)")
    parser.add_argument("--depth-x", type=float, default=None,
                        help=f"sideways swell (pixel: follows --depth; smooth: the chest puffs out by this share, "
                             f"default {DEFAULT_SMOOTH_DEPTH_X})")
    parser.add_argument("--breaths", type=int, default=None, help="breaths per loop (default 1)")
    parser.add_argument("--lag", type=float, default=None, help=f"pixel only: travelling-wave lag (default {DEFAULT_LAG})")
    parser.add_argument("--rigid-row", type=int, default=None,
                        help="the row the head starts below, counted from the top of the figure (default: measured)")
    shape = parser.add_argument_group("smooth shape (defaults in parentheses)")
    shape.add_argument("--chest-y", type=float, default=None,
                       help=f"chest centre height, a share of the body below the neck ({SMOOTH_SHAPE['chest_y']})")
    shape.add_argument("--chest-rx", type=float, default=None,
                       help=f"chest weight's half width, times the torso half width ({SMOOTH_SHAPE['chest_rx']})")
    shape.add_argument("--chest-ry", type=float, default=None,
                       help=f"chest weight's half height, a share of the body below the neck ({SMOOTH_SHAPE['chest_ry']})")
    shape.add_argument("--shoulder", type=float, default=None,
                       help=f"how far the shoulders rise, times the chest's ({SMOOTH_SHAPE['shoulder']})")
    shape.add_argument("--shoulder-lag", type=float, default=None,
                       help=f"how late the shoulders peak, a share of one breath ({SMOOTH_SHAPE['shoulder_lag']})")
    shape.add_argument("--chest-tilt", type=float, default=None,
                       help="the chest weight turned clockwise on screen, degrees, for a body turned aside (0)")
    shape.add_argument("--chest-bias", type=float, default=None,
                       help="one side of the chest breathes more, -0.9..0.9, negative the screen's left (0)")
    shape.add_argument("--chest-dir", type=float, default=None,
                       help="where the chest moves, degrees from straight up, negative up-left (0)")


def run(*, frames: str, out: str, mode: str | None = None, depth: float | None = None,
        depth_x: float | None = None, breaths: int | None = None, lag: float | None = None,
        rigid_row: int | None = None, chest_y: float | None = None, chest_rx: float | None = None,
        chest_ry: float | None = None, shoulder: float | None = None, shoulder_lag: float | None = None,
        chest_tilt: float | None = None, chest_bias: float | None = None, chest_dir: float | None = None) -> int:
    """프레임 폴더에 Breathe 레이어를 굽는다 — 큐레이션 사이드카 `breathe` 와 같은 값, 같은 정규화,
    같은 굽기(`bake_breathe_sequence`). 영상 루프의 칸처럼 큐레이션 런이 아닌 프레임이 이 길로 온다."""
    src_dir, out_dir = Path(frames), Path(out)
    raw = {k: v for k, v in (("mode", mode), ("depth", depth), ("depth_x", depth_x), ("breaths", breaths),
                             ("lag", lag), ("rigid_row", rigid_row), ("chest_y", chest_y), ("chest_rx", chest_rx),
                             ("chest_ry", chest_ry), ("shoulder", shoulder), ("shoulder_lag", shoulder_lag),
                             ("chest_tilt", chest_tilt), ("chest_bias", chest_bias), ("chest_dir", chest_dir))
           if v is not None}
    try:
        cfg = state_breathe({"states": {"frames": {"breathe": raw}}}, "frames")
    except SystemExit as exc:
        raise SystemExit(f"breathe: 사이드카와 같은 검사에서 거부됐다 — {exc}")
    paths = sorted(src_dir.glob("*.png"))
    if not paths:
        raise SystemExit(f"breathe: {src_dir} 에 PNG 프레임이 없다")
    if out_dir.resolve() == src_dir.resolve():
        raise SystemExit("breathe: out 이 frames 와 같은 폴더다 — 원본 프레임 위에 덮어쓰지 않는다")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"breathe: {out_dir} 가 비어 있지 않다 — 새 폴더를 주라")
    images = []
    for path in paths:
        with Image.open(path) as opened:
            images.append(opened.convert("RGBA"))
    if len({im.size for im in images}) != 1:
        raise SystemExit(f"breathe: 프레임 크기가 서로 다르다 — {sorted({im.size for im in images})}")
    resolved = anatomy_report(images, cfg)
    baked, phases = bake_breathe_sequence(images, cfg)
    # 다 굽고 나서 한 번에 내놓는다 — 굽기가 중간에 거부되면 반쯤 찬 폴더를 남기지 않는다.
    staging = out_dir.parent / f".{out_dir.name}.partial"
    if staging.exists():
        for stale in staging.iterdir():
            stale.unlink()
        staging.rmdir()
    staging.mkdir(parents=True)
    for path, frame in zip(paths, baked):
        frame.save(staging / path.name)
    report = {"frames": len(paths), "source": str(src_dir), "config": cfg, "phases": phases,
              "resolved": resolved}
    (staging / "breathe.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                                          encoding="utf-8")
    if out_dir.exists():
        out_dir.rmdir()
    staging.rename(out_dir)
    print(f"breathe: {len(paths)} frames, {breathe_mode(cfg)}, depth {cfg['depth']}, "
          f"breaths {cfg['breaths']} -> {out_dir}")
    return 0
