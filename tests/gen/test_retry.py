# SPDX-License-Identifier: Apache-2.0
"""When an answered provider request is sent again is one rule, `gen.retry`: a POST only when its
provider documents the answer as a request not taken (Anthropic 429 but its spend cap, 500, 529;
OpenAI 429 but its billing codes, 503; xAI 429), a GET on any of 429/5xx it may be answered with;
the wait is the answer's retry-after when this run waits that long, else the doubling `backoff` the
video polls wait by; twice at most. Every direct provider call asks it — the Anthropic transport, the
OpenAI and Grok image calls, the facing check and the clip request."""
from __future__ import annotations

import io
import json
import urllib.error

import pytest
from PIL import Image

from sprite_gen.gen import anthropic_api, facing_vision, grok_provider, openai_provider, retry, video, xai
from sprite_gen.gen.base import GenRequest


def test_the_wait_doubles_to_a_ceiling():
    assert [retry.backoff(n) for n in (1, 2, 3, 4, 5)] == [8.0, 16.0, 32.0, 60.0, 60.0]


@pytest.mark.parametrize(("headers", "seconds"), [
    ({"retry-after": "3"}, 3.0), ({"retry-after": "0"}, 0.0), ({"retry-after": "soon"}, None),
    ({"retry-after": "-1"}, None), ({}, None), (None, None),
])
def test_retry_after_is_read_when_it_is_a_number(headers, seconds):
    assert retry.retry_after(headers) == seconds


SPEND_CAP = {"type": "error", "error": {"type": "rate_limit_error", "message": "usage limits",
                                        "details": {"error_code": "enforced_spend_limit_reached"}}}


@pytest.mark.parametrize(("provider", "status", "body", "again"), [
    ("anthropic", 429, {"error": {"type": "rate_limit_error"}}, True),
    ("anthropic", 529, {"error": {"type": "overloaded_error"}}, True),
    ("anthropic", 500, {"error": {"type": "api_error"}}, True),
    ("anthropic", 504, {"error": {"type": "timeout_error"}}, False),  # "timed out while processing"
    ("anthropic", 502, {}, False),
    ("anthropic", 503, {}, False),
    ("anthropic", 429, SPEND_CAP, False),
    ("anthropic", 400, {"error": {"type": "invalid_request_error"}}, False),
    ("openai", 429, {"error": {"code": "rate_limit_exceeded"}}, True),
    ("openai", 429, {"error": {"type": "rate_limit_error", "code": "slow_down"}}, True),
    ("openai", 503, {"error": {"code": "server_is_overloaded"}}, True),
    ("openai", 500, {}, False),  # "The server had an error while processing your request"
    ("openai", 429, {"error": {"code": "credit_balance_exhausted"}}, False),
    ("openai", 429, {"error": {"code": "organization_spend_limit_exceeded"}}, False),
    ("openai", 429, {"error": {"code": "project_spend_limit_exceeded"}}, False),
    ("openai", 429, {"error": {"code": "organization_usage_limit_exceeded"}}, False),
    ("openai", 429, {"error": {"type": "insufficient_quota"}}, False),
    ("xai", 429, {"code": "resource-exhausted"}, True),
    ("xai", 500, {}, False),
    ("xai", 503, {}, False),
])
def test_a_post_is_sent_again_only_when_its_provider_documents_the_answer_as_not_taken(provider, status, body, again):
    wait = retry.again_after(provider, "POST", status, body, None, 1)
    assert (wait == 8.0) if again else wait is None


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504, 529])
def test_a_get_buys_nothing_and_is_asked_again(status):
    assert retry.again_after("anthropic", "GET", status, {}, None, 1) == 8.0
    assert retry.again_after("anthropic", "GET", 404, {}, None, 1) is None


def test_twice_at_most_and_the_answers_own_wait_when_this_run_waits_that_long():
    assert retry.again_after("xai", "POST", 429, {}, None, 2) == 16.0
    assert retry.again_after("xai", "POST", 429, {}, None, 3) is None
    assert retry.again_after("openai", "POST", 429, {}, {"retry-after": "2"}, 1) == 2.0
    # A longer wait than the run takes is not cut short: the request is not sent again.
    assert retry.again_after("openai", "POST", 429, {}, {"retry-after": "600"}, 1) is None


def test_ask_sends_again_says_so_and_returns_the_last_answer(capsys):
    answers = [(429, {"error": {"code": "rate_limit_exceeded"}}, {}), (503, {}, {"retry-after": "1"}), (200, {"ok": 1}, {})]
    slept = []
    status, body, _headers, asks = retry.ask(lambda: answers.pop(0), provider="openai", who="openai", label="gen",
                                             sleep=slept.append, kind=lambda b: (b.get("error") or {}).get("code"))
    assert (status, body, asks, slept) == (200, {"ok": 1}, 3, [8.0, 1.0])
    err = capsys.readouterr().err.splitlines()
    assert err == ["[gen] openai refused the request (429 rate_limit_exceeded); nothing was made — asking again in 8s (1 of 2)",
                   "[gen] openai refused the request (503); nothing was made — asking again in 1s (2 of 2)"]


# --- every direct provider call asks it -------------------------------------------------------------


@pytest.fixture
def asked(monkeypatch):
    """Every `retry.ask`: the provider and method it was asked for; the answer is the real one's."""
    real, seen = retry.ask, []

    def recording(send, **kwargs):
        seen.append((kwargs["provider"], kwargs.get("method", "POST")))
        return real(send, **kwargs)

    monkeypatch.setattr(retry, "ask", recording)
    return seen


class _Answer:
    status = 200
    headers: dict = {}

    def __init__(self, body: dict):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self):
        return json.dumps(self.body).encode()


def _png_b64() -> str:
    import base64

    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), (20, 90, 200)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def test_the_openai_image_call_asks_it(monkeypatch, tmp_path, asked):
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic")
    monkeypatch.setattr(openai_provider.urllib.request, "urlopen",
                        lambda request, timeout: _Answer({"data": [{"b64_json": _png_b64()}]}))
    openai_provider.OpenAIProvider().generate(GenRequest("x", tmp_path / "raw.png"), tmp_path)
    assert asked == [("openai", "POST")]


def test_the_grok_image_call_and_the_facing_check_ask_it(monkeypatch, tmp_path, asked):
    monkeypatch.setenv("XAI_API_KEY", "synthetic")
    monkeypatch.setenv("GROK_HOME", str(tmp_path / "no-login"))
    monkeypatch.setattr(xai.urllib.request, "urlopen", lambda request, timeout: _Answer({"data": [{"b64_json": _png_b64()}]}))
    grok_provider.GrokProvider().generate(GenRequest("x", tmp_path / "raw.png"), tmp_path)
    facing_vision.grok_inspect(tmp_path / "raw.png", credential=xai.Credential("t", xai.AUTH_SOURCE_GROK_LOGIN),
                               call=lambda *a, **k: (200, {"output": []}))
    assert asked == [("xai", "POST"), ("xai", "POST")]


def test_the_clip_request_asks_it(tmp_path, asked):
    still = tmp_path / "still.png"
    Image.new("RGB", (8, 8), (1, 2, 3)).save(still)
    calls = []

    def call(method, url, token, body, *, deadline=None):
        calls.append(method)
        return (200, {"request_id": "r"}) if method == "POST" else (
            200, {"status": "done", "video": {"url": "https://vidgen.x.ai/v/1.mp4"}})

    video.generate_video(video.VideoRequest(image=still, prompt="sway", out=tmp_path / "c.mp4"),
                         credential=video.Credential("t", video.AUTH_SOURCE_GROK_LOGIN), call=call,
                         download=lambda url, token, **_: b"\x00\x00\x00\x18ftypisom" + b"\x00" * 64,
                         sleep=lambda s: None)
    assert asked == [("xai", "POST")] and calls == ["POST", "GET"]


def test_the_anthropic_transport_asks_it_for_its_posts_and_its_reads(monkeypatch, asked):
    answers = iter([{"id": "msgbatch_1"}, {"id": "msgbatch_1", "processing_status": "in_progress"}])
    monkeypatch.setattr(anthropic_api.urllib.request, "urlopen", lambda request, timeout: _Answer(next(answers)))
    anthropic_api.submit_batch("k", [("c1", {})], sleep=lambda s: None)
    anthropic_api.get_batch("k", "msgbatch_1", sleep=lambda s: None)
    assert asked == [("anthropic", "POST"), ("anthropic", "GET")]


# --- a refusal's body is read by one function ---------------------------------------------------------


class _Cut(io.RawIOBase):
    """A refusal body that breaks on the way."""

    def readinto(self, buffer):
        import http.client

        raise http.client.IncompleteRead(b'{"error": ', 40)


def _refused_cut(request, timeout=None):
    raise urllib.error.HTTPError(request.full_url, 429, "Too Many Requests", {}, io.BufferedReader(_Cut()))


def test_a_refusal_whose_body_broke_is_no_answer_on_every_transport(monkeypatch, tmp_path):
    """Read whole or not at all (`xai.error_body`): its status alone is not an answer read whole, so none
    of the three takes it as a refusal to read or to send again."""
    monkeypatch.setattr(xai.urllib.request, "urlopen", _refused_cut)
    with pytest.raises(xai.TransportError):
        xai.http_json("POST", "https://api.x.ai/v1/images/generations", "t", {})
    monkeypatch.setattr(openai_provider.urllib.request, "urlopen", _refused_cut)
    with pytest.raises(SystemExit, match="request failed or timed out; no automatic retry"):
        openai_provider.http_image("https://api.openai.com/v1/images/generations", "t", b"{}", "application/json", timeout=5)
    monkeypatch.setattr(anthropic_api.urllib.request, "urlopen", _refused_cut)
    with pytest.raises(anthropic_api.ApiError, match="got no answer") as caught:
        anthropic_api.submit_batch("k", [("c1", {})], sleep=lambda s: pytest.fail("no answer is never asked again"))
    assert caught.value.status is None


def test_every_transport_reads_a_refusals_body_through_it(monkeypatch):
    seen = []
    real = xai.error_body

    def recording(refused, *args, **kwargs):
        seen.append(refused.code)
        return real(refused, *args, **kwargs)

    def refused(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", {}, io.BytesIO(b'{"error": "no"}'))

    for module in (xai, openai_provider, anthropic_api):
        monkeypatch.setattr(module, "error_body", recording)
        monkeypatch.setattr(module.urllib.request, "urlopen", refused)
    assert xai.http_json("POST", "https://api.x.ai/v1/x", "t", {}) == (400, {"error": "no"})
    assert openai_provider.http_image("https://api.openai.com/v1/x", "t", b"{}", "application/json", timeout=5)[:2] == (400, {"error": "no"})
    with pytest.raises(anthropic_api.ApiError, match=r"\(HTTP 400\)"):
        anthropic_api.submit_batch("k", [("c1", {})], sleep=lambda s: None)
    assert seen == [400, 400, 400]
