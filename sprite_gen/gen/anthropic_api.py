# SPDX-License-Identifier: Apache-2.0
"""Anthropic Messages API transport for `pixel-draw`: the credential, one streamed message,
Message Batches, and what a call cost.

Raw HTTP over urllib, like `openai_provider` and `xai`: the runtime dependency surface is Pillow
and NumPy (SECURITY.md), and one more SDK with its own HTTP stack would be a second transport for
the same job. Field names and behaviour are the documented Messages API surface
(platform.claude.com, 2026-10-12 확인):

- `POST /v1/messages` with `stream: true` answers Server-Sent Events: `message_start` (a message
  with empty content and the input usage), per content block a `content_block_start`, its deltas
  (`text_delta`, `thinking_delta`, `signature_delta`) and a `content_block_stop`, then
  `message_delta` (stop reason, cumulative usage) and `message_stop`; `ping` at any time; an
  `error` event may arrive mid-stream (`overloaded_error`). The response header `request-id`
  names the request.
- `claude-opus-5-5` takes `thinking: {type: adaptive}` only (`disabled` and a token budget are
  400s) and `output_config.effort` low|medium|high|xhigh|max, default medium — so the effort is
  always sent. Thinking tokens are billed as output and reported under
  `usage.output_tokens_details.thinking_tokens`.
- `POST /v1/messages/batches` takes `{requests: [{custom_id, params}]}` (custom_id
  `^[a-zA-Z0-9_-]{1,64}$`, no `stream` in params); `GET /v1/messages/batches/{id}` reports
  `processing_status` in_progress|canceling|ended and, once ended, `results_url`, a JSONL file of
  `{custom_id, result: {type: succeeded|errored|canceled|expired, ...}}`. Only `succeeded` is
  billed, at half the price of the same call made directly.

A refusal of the request itself — 429, 500, 502, 503, 504, 529 before any answer began — made
nothing, so it is asked again, twice at most, each time said on stderr. Anything after the answer
began (a broken stream, an `error` event) may have been billed and is never asked again here.
There is no refusal fallback to another model: `pixel-draw` reports a refusal as it came.
"""
from __future__ import annotations

import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from . import refusal

API_BASE = "https://api.anthropic.com/v1"
API_VERSION = "2023-06-01"
AUTH_ENV = "ANTHROPIC_API_KEY"
MODEL = "claude-opus-5-5"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
# Output cap of claude-opus-5-5 (Models API `max_tokens`, measured 2026-10-12).
MODEL_MAX_TOKENS = 128000

# USD per million tokens, platform.claude.com/docs/en/about-claude/pricing (2026-10-12 확인).
PRICE_PER_MTOK = {
    MODEL: {"input": 4.00, "cache_write_5m": 5.00, "cache_write_1h": 8.00, "cache_read": 0.20, "output": 20.00},
}
BATCH_DISCOUNT = 0.5

RETRY_STATUSES = (429, 500, 502, 503, 504, 529)
RETRY_WAITS = (10.0, 30.0)  # before the second and the third ask
RETRY_AFTER_MAX = 120.0
# One socket read may wait this long. Thinking streams its summary (`display: summarized`) and the
# API sends `ping`s, so a stream that says nothing for minutes is stalled, not thinking.
READ_TIMEOUT_SECONDS = float(os.environ.get("SPRITE_GEN_CLAUDE_READ_TIMEOUT_SECONDS", "300"))

# Everything that ends an exchange before its answer is read whole (as in `xai.NO_ANSWER`).
NO_ANSWER = (OSError, http.client.HTTPException)

Sleep = Callable[[float], None]


def resolve_credential(env: dict[str, str] | None = None) -> str:
    """Return the Anthropic API key, or fail loud naming the variable that is missing."""
    env = os.environ if env is None else env
    key = (env.get(AUTH_ENV) or "").strip()
    if key:
        return key
    if AUTH_ENV in env:
        raise SystemExit(f"pixel-draw: {AUTH_ENV} is set but empty; give it a value")
    raise SystemExit(
        f"pixel-draw: no Anthropic credential — {AUTH_ENV} is not set.\n"
        f"  export {AUTH_ENV}=<your Claude Console API key> (platform.claude.com/settings/keys)."
    )


@dataclass
class Reply:
    """One answered message: its text, its thinking summary and what it used."""

    text: str
    thinking: str
    stop_reason: str | None
    usage: dict[str, Any]
    model: str | None = None
    message_id: str | None = None
    stop_details: dict[str, Any] | None = None
    request_id: str | None = None
    batch_id: str | None = None
    mode: str = "realtime"
    seconds: float | None = None


class ApiError(Exception):
    """The request was refused, got no answer, or its answer broke. `line` says which, masked.

    `partial` is what had arrived when a stream broke (its usage may be billed); `billed` is False
    only when the API says nothing was made (a refused request, a batch item that errored)."""

    def __init__(self, line: str, *, status: int | None = None, request_id: str | None = None,
                 partial: Reply | None = None, billed: bool | None = None) -> None:
        super().__init__(line)
        self.line = line
        self.status = status
        self.request_id = request_id
        self.partial = partial
        self.billed = billed


def message_body(*, system: str | None, content: str | list[dict[str, Any]], effort: str, max_tokens: int,
                 schema: dict[str, Any] | None = None) -> dict[str, Any]:
    """A Messages API request body. Effort is always explicit (the model's default is medium)."""
    if effort not in EFFORTS:
        raise ValueError(f"effort must be one of {', '.join(EFFORTS)}, got {effort!r}")
    output_config: dict[str, Any] = {"effort": effort}
    if schema is not None:
        output_config["format"] = {"type": "json_schema", "schema": schema}
    body: dict[str, Any] = {"model": MODEL, "max_tokens": max_tokens}
    if system is not None:
        body["system"] = system
    body.update(thinking={"type": "adaptive", "display": "summarized"}, output_config=output_config,
                messages=[{"role": "user", "content": content}])
    return body


def thinking_tokens(usage: dict[str, Any]) -> int | None:
    details = usage.get("output_tokens_details")
    value = details.get("thinking_tokens") if isinstance(details, dict) else None
    return value if isinstance(value, int) else None


def cost_usd(usage: dict[str, Any], *, batch: bool, model: str = MODEL) -> float:
    """What `usage` costs at the published prices; Message Batches bill half."""
    price = PRICE_PER_MTOK[model]
    creation = usage.get("cache_creation") if isinstance(usage.get("cache_creation"), dict) else {}
    write_5m = creation.get("ephemeral_5m_input_tokens")
    write_1h = creation.get("ephemeral_1h_input_tokens")
    if write_5m is None and write_1h is None:
        write_5m, write_1h = usage.get("cache_creation_input_tokens") or 0, 0
    total = (
        (usage.get("input_tokens") or 0) * price["input"]
        + (write_5m or 0) * price["cache_write_5m"]
        + (write_1h or 0) * price["cache_write_1h"]
        + (usage.get("cache_read_input_tokens") or 0) * price["cache_read"]
        + (usage.get("output_tokens") or 0) * price["output"]
    ) / 1_000_000
    return round(total * (BATCH_DISCOUNT if batch else 1.0), 6)


def _request(method: str, url: str, token: str, data: bytes | None, *, accept: str) -> urllib.request.Request:
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("x-api-key", token)
    request.add_header("anthropic-version", API_VERSION)
    request.add_header("Accept", accept)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    return request


def _error_body(exc: urllib.error.HTTPError) -> Any:
    try:
        raw = exc.read().decode("utf-8", "replace")
    except NO_ANSWER:
        return refusal.RawBody("")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return refusal.RawBody(raw)


def _error_type(body: Any) -> str | None:
    """`error.type` of an Anthropic error body (`{"type": "error", "error": {"type", "message"}}`)."""
    error = body.get("error") if isinstance(body, dict) else None
    return refusal.identifier(error.get("type")) if isinstance(error, dict) else None


def refused_line(status: int, body: Any, token: str | None) -> str:
    code = _error_type(body)
    suffix = f" code={code}" if code else ""
    if status in (401, 403):
        return (f"pixel-draw: credential {AUTH_ENV} rejected (HTTP {status}){suffix}; check the key's value and "
                f"that its workspace may call {MODEL}{refusal.said(body, token)}")
    return f"pixel-draw: claude request failed (HTTP {status}){suffix}; not asked again{refusal.said(body, token)}"


def _no_answer(exc: BaseException) -> str:
    said = exc.reason if isinstance(exc, urllib.error.URLError) else exc
    if isinstance(said, http.client.HTTPException) and not isinstance(said, OSError):
        return repr(said)[:200]
    return (str(said) or type(said).__name__)[:200]


def _retry_after(headers: Any) -> float | None:
    value = headers.get("retry-after") if headers is not None else None
    try:
        seconds = float(value) if value is not None else None
    except ValueError:
        return None
    return min(seconds, RETRY_AFTER_MAX) if seconds is not None and seconds >= 0 else None


def _open(request_for: Callable[[], urllib.request.Request], token: str, *, timeout: float, sleep: Sleep,
          label: str) -> http.client.HTTPResponse:
    """Send a request; a refusal that made nothing is asked again (RETRY_STATUSES), anything else raises."""
    for attempt in range(len(RETRY_WAITS) + 1):
        try:
            return urllib.request.urlopen(request_for(), timeout=timeout)
        except urllib.error.HTTPError as exc:
            body = _error_body(exc)
            if exc.code in RETRY_STATUSES and attempt < len(RETRY_WAITS):
                wait = _retry_after(exc.headers)
                wait = RETRY_WAITS[attempt] if wait is None else wait
                kind = _error_type(body)
                print(f"[{label}] claude refused the request ({exc.code}{' ' + kind if kind else ''}); "
                      f"nothing was made — asking again in {wait:g}s ({attempt + 1} of {len(RETRY_WAITS)})",
                      file=sys.stderr)
                sleep(wait)
                continue
            raise ApiError(refused_line(exc.code, body, token), status=exc.code,
                           request_id=exc.headers.get("request-id") if exc.headers else None, billed=False) from exc
        except NO_ANSWER as exc:
            raise ApiError(f"pixel-draw: claude request got no answer ({_no_answer(exc)}); not asked again "
                           "(the API may have accepted it)") from exc
    raise AssertionError("unreachable")


class _Stream:
    """The state of one streamed message, fed one SSE event at a time."""

    def __init__(self) -> None:
        self.message: dict[str, Any] = {}
        self.usage: dict[str, Any] = {}
        self.blocks: dict[int, dict[str, Any]] = {}
        self.stop_reason: str | None = None
        self.stop_details: dict[str, Any] | None = None
        self.done = False
        self.error: Any = None

    def feed(self, payload: dict[str, Any]) -> None:
        kind = payload.get("type")
        if kind == "message_start":
            self.message = payload.get("message") or {}
            self.usage.update(self.message.get("usage") or {})
        elif kind == "content_block_start":
            block = payload.get("content_block") or {}
            self.blocks[payload.get("index", len(self.blocks))] = {
                "type": block.get("type"), "text": block.get("text") or "", "thinking": block.get("thinking") or ""}
        elif kind == "content_block_delta":
            block = self.blocks.setdefault(payload.get("index", 0), {"type": None, "text": "", "thinking": ""})
            delta = payload.get("delta") or {}
            if delta.get("type") == "text_delta":
                block["text"] += delta.get("text") or ""
            elif delta.get("type") == "thinking_delta":
                block["thinking"] += delta.get("thinking") or ""
        elif kind == "message_delta":
            delta = payload.get("delta") or {}
            self.stop_reason = delta.get("stop_reason", self.stop_reason)
            if isinstance(delta.get("stop_details"), dict):
                self.stop_details = delta["stop_details"]
            self.usage.update(payload.get("usage") or {})  # cumulative counts
        elif kind == "message_stop":
            self.done = True
        elif kind == "error":
            self.error = payload
        # ping and event types added later are not ours to read

    def reply(self, *, request_id: str | None, seconds: float) -> Reply:
        ordered = [self.blocks[i] for i in sorted(self.blocks)]
        return Reply(
            text="".join(b["text"] for b in ordered if b["type"] == "text"),
            thinking="\n".join(b["thinking"] for b in ordered if b["type"] == "thinking" and b["thinking"]),
            stop_reason=self.stop_reason,
            stop_details=self.stop_details,
            usage=dict(self.usage),
            model=self.message.get("model"),
            message_id=self.message.get("id"),
            request_id=request_id,
            mode="realtime",
            seconds=round(seconds, 3),
        )


def stream_message(token: str, body: dict[str, Any], *, sleep: Sleep = time.sleep,
                   timeout: float | None = None, label: str = "pixel-draw") -> Reply:
    """Send one message with `stream: true` and read its events to the end."""
    data = json.dumps({**body, "stream": True}).encode("utf-8")
    timeout = READ_TIMEOUT_SECONDS if timeout is None else timeout
    started = time.monotonic()
    response = _open(lambda: _request("POST", API_BASE + "/messages", token, data, accept="text/event-stream"),
                     token, timeout=timeout, sleep=sleep, label=label)
    request_id = response.headers.get("request-id")
    state = _Stream()
    event_data: list[str] = []
    try:
        with response:
            for raw in response:
                line = raw.decode("utf-8").rstrip("\r\n")
                if line:
                    field_name, _, value = line.partition(":")
                    if field_name == "data":
                        event_data.append(value[1:] if value.startswith(" ") else value)
                    continue
                if event_data:
                    state.feed(json.loads("\n".join(event_data)))
                    event_data = []
                if state.error is not None or state.done:
                    break
            if event_data and state.error is None and not state.done:
                state.feed(json.loads("\n".join(event_data)))
    except NO_ANSWER as exc:
        partial = state.reply(request_id=request_id, seconds=time.monotonic() - started)
        raise ApiError(f"pixel-draw: the answer stream broke ({_no_answer(exc)}); not asked again "
                       "(what was written may be billed)", request_id=request_id, partial=partial) from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        partial = state.reply(request_id=request_id, seconds=time.monotonic() - started)
        raise ApiError("pixel-draw: the answer stream carried an event that is not JSON; not asked again",
                       request_id=request_id, partial=partial) from exc
    reply = state.reply(request_id=request_id, seconds=time.monotonic() - started)
    if state.error is not None:
        error = state.error.get("error") if isinstance(state.error, dict) else None
        code = refusal.identifier(error.get("type")) if isinstance(error, dict) else None
        raise ApiError(f"pixel-draw: the answer stream ended in an error{' code=' + code if code else ''}; "
                       f"not asked again (what was written may be billed){refusal.said(state.error, token)}",
                       request_id=request_id, partial=reply)
    if not state.done:
        raise ApiError("pixel-draw: the answer stream ended before message_stop; not asked again "
                       "(what was written may be billed)", request_id=request_id, partial=reply)
    return reply


def _json_call(method: str, url: str, token: str, body: dict[str, Any] | None, *, sleep: Sleep,
               label: str, timeout: float = 120.0) -> Any:
    data = None if body is None else json.dumps(body).encode("utf-8")
    response = _open(lambda: _request(method, url, token, data, accept="application/json"),
                     token, timeout=timeout, sleep=sleep, label=label)
    try:
        with response:
            raw = response.read().decode("utf-8")
    except NO_ANSWER as exc:
        raise ApiError(f"pixel-draw: claude answer broke ({_no_answer(exc)})") from exc
    try:
        return json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        raise ApiError("pixel-draw: claude answer was not JSON") from exc


def submit_batch(token: str, requests: list[tuple[str, dict[str, Any]]], *, sleep: Sleep = time.sleep,
                 label: str = "pixel-draw") -> dict[str, Any]:
    """Create one Message Batch of `(custom_id, params)` and return the batch object."""
    body = {"requests": [{"custom_id": custom_id, "params": params} for custom_id, params in requests]}
    batch = _json_call("POST", API_BASE + "/messages/batches", token, body, sleep=sleep, label=label)
    if not isinstance(batch, dict) or not isinstance(batch.get("id"), str):
        raise ApiError("pixel-draw: the batch answer names no batch id")
    return batch


def get_batch(token: str, batch_id: str, *, sleep: Sleep = time.sleep, label: str = "pixel-draw") -> dict[str, Any]:
    batch = _json_call("GET", f"{API_BASE}/messages/batches/{batch_id}", token, None, sleep=sleep, label=label)
    if not isinstance(batch, dict):
        raise ApiError("pixel-draw: the batch status answer is not an object")
    return batch


def _message_reply(message: dict[str, Any], batch_id: str) -> Reply:
    content = message.get("content") if isinstance(message.get("content"), list) else []
    return Reply(
        text="".join(b.get("text") or "" for b in content if isinstance(b, dict) and b.get("type") == "text"),
        thinking="\n".join(b.get("thinking") or "" for b in content
                           if isinstance(b, dict) and b.get("type") == "thinking" and b.get("thinking")),
        stop_reason=message.get("stop_reason"),
        stop_details=message.get("stop_details") if isinstance(message.get("stop_details"), dict) else None,
        usage=message.get("usage") if isinstance(message.get("usage"), dict) else {},
        model=message.get("model"),
        message_id=message.get("id"),
        batch_id=batch_id,
        mode="batch",
    )


def batch_results(token: str, batch: dict[str, Any], *, sleep: Sleep = time.sleep,
                  label: str = "pixel-draw") -> dict[str, Reply | ApiError]:
    """Every result of an ended batch by custom_id; an item that was not answered is an ApiError."""
    url = batch.get("results_url")
    if not isinstance(url, str) or not url:
        raise ApiError(f"pixel-draw: batch {batch.get('id')} has ended but names no results_url")
    response = _open(lambda: _request("GET", url, token, None, accept="application/binary"),
                     token, timeout=120.0, sleep=sleep, label=label)
    try:
        with response:
            lines = response.read().decode("utf-8").splitlines()
    except NO_ANSWER as exc:
        raise ApiError(f"pixel-draw: batch results broke ({_no_answer(exc)})") from exc
    results: dict[str, Reply | ApiError] = {}
    for line in lines:
        if not line.strip():
            continue
        item = json.loads(line)
        custom_id, result = item.get("custom_id"), item.get("result") or {}
        kind = result.get("type")
        if kind == "succeeded" and isinstance(result.get("message"), dict):
            results[custom_id] = _message_reply(result["message"], batch["id"])
        elif kind == "errored":
            error = result.get("error") or {}
            results[custom_id] = ApiError(
                f"pixel-draw: batch item {custom_id} errored{' code=' + str(_error_type(error)) if _error_type(error) else ''}"
                f" (not billed){refusal.said(error, token)}", billed=False)
        else:
            results[custom_id] = ApiError(f"pixel-draw: batch item {custom_id} {kind or 'has no result'} (not billed)",
                                          billed=False)
    return results
