# SPDX-License-Identifier: Apache-2.0
"""`sprite-gen video` contract: credential resolution, the api.x.ai call shape,
verified mp4 publishing, and loud failures. No network — the HTTP layer is a fake."""

from __future__ import annotations

import errno
import json
import os
import re
import socket
import threading
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from PIL import Image

from sprite_gen.gen import video, xai

NOW = datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)
MP4 = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 64


def _still(tmp_path: Path) -> Path:
    path = tmp_path / "still.png"
    Image.new("RGBA", (8, 8), (255, 120, 0, 255)).save(path)
    return path


def _login_file(tmp_path: Path, *, expires_at: str | None = "2026-09-08T18:00:00.000000Z", key: str = "tok") -> Path:
    home = tmp_path / "grokhome"
    home.mkdir(exist_ok=True)
    entry = {"key": key, "auth_mode": "oidc"}
    if expires_at is not None:
        entry["expires_at"] = expires_at
    (home / "auth.json").write_text(json.dumps({"https://auth.x.ai::client": entry}), encoding="utf-8")
    return home


class _FakeApi:
    """Scripted xAI: one POST reply, then a sequence of poll replies."""

    def __init__(self, post=(200, {"request_id": "req-1"}), polls=None):
        self.post = post
        self.polls = list(polls or [(200, {"status": "done", "video": {"url": "https://vidgen.x.ai/v/1.mp4", "duration": 6.0}, "model": "grok-imagine-video-1.5"})])
        self.calls: list[tuple[str, str, str, dict | None]] = []
        self.downloads: list[str] = []
        self.payload = MP4

    def call(self, method, url, token, body):
        self.calls.append((method, url, token, body))
        if method == "POST":
            return self.post
        return self.polls.pop(0) if len(self.polls) > 1 else self.polls[0]

    def download(self, url, token):
        self.downloads.append(url)
        return self.payload


def _request(tmp_path: Path, **overrides) -> video.VideoRequest:
    kwargs = dict(image=_still(tmp_path), prompt="camera locked, gentle idle sway", out=tmp_path / "clip.mp4")
    kwargs.update(overrides)
    return video.VideoRequest(**kwargs)


# --- credential resolution -------------------------------------------------


@pytest.mark.parametrize("api_key", ["xai-console", "", "  "])
def test_grok_login_wins_over_api_key_env(tmp_path: Path, monkeypatch, api_key) -> None:
    monkeypatch.setenv("GROK_HOME", str(_login_file(tmp_path)))
    cred = video.resolve_credential(env={"XAI_API_KEY": api_key}, now=NOW)
    assert cred.token == "tok" and cred.source == video.AUTH_SOURCE_GROK_LOGIN


def test_api_key_is_used_only_without_a_login_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("GROK_HOME", str(tmp_path / "no-login"))
    cred = video.resolve_credential(env={"XAI_API_KEY": "xai-console"}, now=NOW)
    assert cred == video.Credential(token="xai-console", source=video.AUTH_SOURCE_API_KEY)


def test_grok_login_is_used_when_no_api_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("GROK_HOME", str(_login_file(tmp_path, key="oidc-token")))
    cred = video.resolve_credential(env={}, now=NOW)
    assert cred.source == video.AUTH_SOURCE_GROK_LOGIN
    assert cred.token == "oidc-token"
    assert cred.expires_at == "2026-09-08T18:00:00.000000Z"


def test_expired_grok_login_fails_before_upload_with_refresh_prescription(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("GROK_HOME", str(_login_file(tmp_path, expires_at="2026-09-08T10:56:34.899829Z")))
    with pytest.raises(SystemExit, match="expired at 2026-09-08T10:56:34") as excinfo:
        video.resolve_credential(env={}, now=NOW)
    message = str(excinfo.value)
    assert video.GROK_REFRESH_COMMAND in message
    assert video.GROK_LOGIN_COMMAND in message
    assert "never rewrites" in message


def test_missing_login_and_key_prescribes_both_setups(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("GROK_HOME", str(tmp_path / "nowhere"))
    with pytest.raises(SystemExit, match="no xAI credential") as excinfo:
        video.resolve_credential(env={}, now=NOW)
    assert "grok login" in str(excinfo.value) and "XAI_API_KEY" in str(excinfo.value)


def test_empty_api_key_env_is_refused_not_ignored(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("GROK_HOME", str(tmp_path / "no-login"))
    with pytest.raises(SystemExit, match="XAI_API_KEY is set but empty"):
        video.resolve_credential(env={"XAI_API_KEY": "  "}, now=NOW)


def test_login_file_with_two_accounts_is_refused(tmp_path: Path, monkeypatch) -> None:
    home = _login_file(tmp_path)
    (home / "auth.json").write_text(json.dumps({"a": {"key": "1"}, "b": {"key": "2"}}), encoding="utf-8")
    monkeypatch.setenv("GROK_HOME", str(home))
    with pytest.raises(SystemExit, match="holds 2 account entries"):
        video.resolve_credential(env={}, now=NOW)


def test_login_without_expiry_is_accepted(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("GROK_HOME", str(_login_file(tmp_path, expires_at=None)))
    assert video.resolve_credential(env={}, now=NOW).expires_at is None


# --- generation ------------------------------------------------------------


def test_generate_video_posts_data_url_polls_and_publishes_verified_mp4(tmp_path: Path) -> None:
    api = _FakeApi(polls=[(200, {"status": "pending", "progress": 10}), (200, {"status": "done", "video": {"url": "https://vidgen.x.ai/v/1.mp4", "duration": 6.0}, "model": "grok-imagine-video-1.5"})])
    slept: list[float] = []
    cred = video.Credential(token="tok", source=video.AUTH_SOURCE_GROK_LOGIN)

    result = video.generate_video(
        _request(tmp_path, aspect_ratio="1:1", generate_audio=False),
        credential=cred, call=api.call, download=api.download, sleep=slept.append,
    )

    method, url, token, body = api.calls[0]
    assert (method, url, token) == ("POST", f"{video.API_BASE}/videos/generations", "tok")
    assert body["model"] == video.DEFAULT_MODEL
    assert body["duration"] == 6 and body["resolution"] == "720p" and body["aspect_ratio"] == "1:1"
    assert body["generate_audio"] is False
    assert body["image"]["url"].startswith("data:image/png;base64,")
    assert "output" not in body  # no upload_url: the direct call is what works on ZDR teams
    assert api.calls[1][:2] == ("GET", f"{video.API_BASE}/videos/req-1")
    assert slept == [video.POLL_INTERVAL_SECONDS]
    assert api.downloads == ["https://vidgen.x.ai/v/1.mp4"]
    assert result.out.read_bytes() == MP4
    assert not result.out.with_name("clip.mp4.part").exists()
    payload = result.to_dict()
    assert payload["kind"] == "sprite-gen-video-report"
    assert payload["auth_source"] == "grok-login"
    assert payload["request_id"] == "req-1"
    assert payload["host"] == "vidgen.x.ai"
    assert payload["duration_reported"] == 6.0 and payload["duration_requested"] == 6
    assert payload["polls"] == 2
    # Secrets never reach the report: no token, no full download URL.
    dumped = json.dumps(payload)
    assert "tok" not in dumped.replace("token", "") and "vidgen.x.ai/v/1.mp4" not in dumped


def test_audio_default_is_left_to_the_api(tmp_path: Path) -> None:
    api = _FakeApi()
    video.generate_video(_request(tmp_path), credential=video.Credential("t", "XAI_API_KEY"), call=api.call, download=api.download, sleep=lambda s: None)
    assert "generate_audio" not in api.calls[0][3] and "aspect_ratio" not in api.calls[0][3]


@pytest.mark.parametrize(
    ("post", "polls", "expected"),
    [
        ((200, {"error": "quota"}), None, "generation request refused"),
        ((400, {"code": "invalid-argument", "error": "Zero Data Retention teams must provide output.upload_url"}), None, "refused \\(HTTP 400\\)"),
        ((403, {"code": "unauthenticated:bad-credentials", "error": "The OAuth2 access token could not be validated."}), None, "rejected the credential \\(grok-login\\) with HTTP 403"),
        ((200, {"request_id": "r"}), [(200, {"status": "failed", "error": "nsfw"})], "status='failed'"),
        ((200, {"request_id": "r"}), [(200, {"status": "expired"})], "status='expired'"),
        ((200, {"request_id": "r"}), [(404, {"raw": "boom"})], "HTTP 404"),
        ((200, {"request_id": "r"}), [(200, {"status": "done", "video": {}})], "reported no video url"),
    ],
)
def test_api_failures_are_named_and_write_nothing(tmp_path: Path, post, polls, expected) -> None:
    api = _FakeApi(post=post, polls=polls)
    request = _request(tmp_path)
    with pytest.raises(SystemExit, match=expected):
        video.generate_video(request, credential=video.Credential("t", video.AUTH_SOURCE_GROK_LOGIN), call=api.call, download=api.download, sleep=lambda s: None)
    assert not request.out.exists()
    assert not list(tmp_path.glob("*.part"))


def test_bad_credential_prescribes_refresh_for_login_and_check_for_key(tmp_path: Path) -> None:
    api = _FakeApi(post=(403, {"error": "bad"}))
    with pytest.raises(SystemExit, match=video.GROK_REFRESH_COMMAND.split()[0]):
        video.generate_video(_request(tmp_path), credential=video.Credential("t", video.AUTH_SOURCE_GROK_LOGIN), call=api.call, download=api.download, sleep=lambda s: None)
    with pytest.raises(SystemExit, match="check the XAI_API_KEY value"):
        video.generate_video(_request(tmp_path), credential=video.Credential("t", video.AUTH_SOURCE_API_KEY), call=api.call, download=api.download, sleep=lambda s: None)


def test_poll_timeout_fails_loud(tmp_path: Path, monkeypatch) -> None:
    api = _FakeApi(polls=[(200, {"status": "pending"})])
    clock = {"t": 0.0}
    monkeypatch.setattr(video.time, "monotonic", lambda: clock["t"])

    def sleep(seconds: float) -> None:
        clock["t"] += seconds

    request = _request(tmp_path)
    with pytest.raises(SystemExit, match="poll timeout"):
        video.generate_video(request, credential=video.Credential("t", "XAI_API_KEY"), call=api.call, download=api.download, sleep=sleep, poll_timeout=10)
    assert not request.out.exists()


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"duration": 0}, "--duration must be 1..15"),
        ({"duration": 16}, "--duration must be 1..15"),
        ({"resolution": "4k"}, "--resolution must be one of"),
        ({"aspect_ratio": "21:9"}, "--aspect-ratio must be one of"),
        ({"prompt": "   "}, "empty prompt"),
    ],
)
def test_request_validation_runs_before_any_call(tmp_path: Path, overrides, expected) -> None:
    api = _FakeApi()
    with pytest.raises(SystemExit, match=expected):
        video.generate_video(_request(tmp_path, **overrides), credential=video.Credential("t", "XAI_API_KEY"), call=api.call, download=api.download)
    assert api.calls == []


def test_missing_still_fails_before_any_call(tmp_path: Path) -> None:
    api = _FakeApi()
    with pytest.raises(SystemExit, match="still image not found"):
        video.generate_video(video.VideoRequest(image=tmp_path / "nope.png", prompt="p", out=tmp_path / "o.mp4"), credential=video.Credential("t", "XAI_API_KEY"), call=api.call, download=api.download)
    assert api.calls == []


# --- CLI ---------------------------------------------------------------------


def test_cli_run_writes_report_and_defaults(tmp_path: Path, monkeypatch) -> None:
    api = _FakeApi()
    monkeypatch.setattr(video, "resolve_credential", lambda **_: video.Credential("t", video.AUTH_SOURCE_API_KEY))
    monkeypatch.setattr(video, "http_json", api.call)
    monkeypatch.setattr(video, "http_download", api.download)
    monkeypatch.setattr(video.time, "sleep", lambda s: None)
    out = tmp_path / "clip.mp4"
    report = tmp_path / "report.json"

    rc = video.run(image=_still(tmp_path), prompt="sway", prompt_file=None, out=out, duration=6, resolution="720p", aspect_ratio=None, model=video.DEFAULT_MODEL, generate_audio=None, report=report)

    assert rc == 0
    assert out.read_bytes() == MP4
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["auth_source"] == "XAI_API_KEY" and payload["out"] == str(out.resolve())


def test_cli_parser_shape() -> None:
    parser = video._build_parser()
    args = parser.parse_args(["--image", "a.png", "--prompt", "p", "--out", "o.mp4"])
    assert args.duration == 6 and args.resolution == "720p" and args.aspect_ratio is None and args.generate_audio is None
    assert parser.parse_args(["--image", "a.png", "--out", "o.mp4", "--no-audio"]).generate_audio is False
    assert parser.parse_args(["--image", "a.png", "--out", "o.mp4", "--audio"]).generate_audio is True
    with pytest.raises(SystemExit):
        parser.parse_args(["--image", "a.png", "--out", "o.mp4", "--resolution", "4k"])


def test_unified_cli_exposes_video() -> None:
    from sprite_gen import cli

    assert "video" in cli.COMMANDS
    parser = cli._build_parser()
    args = parser.parse_args(["video", "--image", "a.png", "--prompt", "p", "--out", "o.mp4"])
    assert args.command == "video"


@pytest.mark.parametrize("contents, message", [
    ('{"account":{"key":"subscription","expires_at":"2000-01-01T00:00:00Z"}}', "expired"),
    ('{', "cannot read grok login"),
    ('{"account":{}}', "no access token"),
    ('{"a":{"key":"one"},"b":{"key":"two"}}', "holds 2 account"),
])
def test_invalid_login_never_falls_back_to_configured_key(tmp_path, monkeypatch, contents, message):
    monkeypatch.setenv("GROK_HOME", str(tmp_path))
    auth = tmp_path / "auth.json"
    auth.write_text(contents)
    with pytest.raises(SystemExit, match=message):
        video.resolve_credential(env={"XAI_API_KEY": "console-key"}, now=NOW)
    assert auth.read_text() == contents


def test_login_directory_never_permits_api_credit(tmp_path, monkeypatch):
    monkeypatch.setenv("GROK_HOME", str(tmp_path))
    (tmp_path / "auth.json").mkdir()
    with pytest.raises(SystemExit, match="not a readable file"):
        video.resolve_credential(env={"XAI_API_KEY": "console-key"}, now=NOW)


def test_uninspectable_login_never_permits_api_credit(tmp_path, monkeypatch):
    monkeypatch.setenv("GROK_HOME", str(tmp_path))
    original = Path.lstat
    def denied(path, *args, **kwargs):
        if path == tmp_path / "auth.json":
            raise PermissionError("test")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "lstat", denied)
    with pytest.raises(SystemExit, match="refusing API-credit fallback"):
        video.resolve_credential(env={"XAI_API_KEY": "console-key"}, now=NOW)


@pytest.mark.parametrize("status", [200, 403, 429])
def test_video_uses_subscription_with_api_key_present(tmp_path, monkeypatch, status):
    monkeypatch.setenv("GROK_HOME", str(_login_file(tmp_path, key="subscription-token", expires_at="2100-01-01T00:00:00Z")))
    monkeypatch.setenv("XAI_API_KEY", "console-key")
    api = _FakeApi(post=(status, {"error": "rejected"}) if status != 200 else (200, {"request_id": "req-1"}))
    request = _request(tmp_path)
    if status != 200:
        with pytest.raises(SystemExit, match=f"HTTP {status}"):
            video.generate_video(request, call=api.call, download=api.download, sleep=lambda _: None)
        assert len(api.calls) == 1 and not request.out.exists()
    else:
        result = video.generate_video(request, call=api.call, download=api.download, sleep=lambda _: None)
        assert result.auth_source == "grok-login"
    assert all(call[2] == "subscription-token" for call in api.calls)


def test_refresh_prescription_is_a_non_agent_command_run_outside_the_repo(tmp_path: Path, monkeypatch) -> None:
    # 2026-09-13 incident: `grok -p ok` started the coding agent in the worker's cwd,
    # overwrote a script and spent a paid video call. The prescription must never be
    # a prompt, and must say where to run it.
    assert video.GROK_REFRESH_COMMAND == "grok models"
    assert " -p " not in f" {video.GROK_REFRESH_COMMAND} " and "--single" not in video.GROK_REFRESH_COMMAND
    assert "empty directory" in video.GROK_REFRESH_WHERE
    monkeypatch.setenv("GROK_HOME", str(_login_file(tmp_path, expires_at="2026-09-08T10:56:34.899829Z")))
    with pytest.raises(SystemExit) as excinfo:
        video.resolve_credential(env={}, now=NOW)
    message = str(excinfo.value)
    assert video.GROK_REFRESH_WHERE in message
    assert "grok -p ok" not in message.replace("`grok -p …`", "")
    assert "coding agent" in message


# --- the API-credit notice (구독 우선 불변식 5) ---------------------------------


def _billed_run(verb: str, tmp_path: Path, **transport):
    """Submit one job of `verb` through the fake transport."""
    if verb == "video":
        return video.generate_video(_request(tmp_path, duration=3, resolution="480p"), **transport)
    clip = tmp_path / "in.mp4"
    clip.write_bytes(MP4)
    transport = {**transport, "probe": lambda path: 6.0}  # ffprobe stand-in: input and staged alike
    if verb == "video-extend":
        return video.extend_video(video.ExtendRequest(video=clip, prompt="she presses on", out=tmp_path / "longer.mp4", duration=4), **transport)
    return video.edit_video(video.EditRequest(video=clip, prompt="make the hakama black", out=tmp_path / "edited.mp4"), **transport)


@pytest.mark.parametrize("verb, priced", [
    ("video", " (duration=3s, resolution=480p)"),
    ("video-extend", " (duration=4s)"),
    ("video-edit", ""),  # the edit body sends neither knob, so the line names neither
])
def test_every_verb_announces_the_charge_before_the_request_leaves(tmp_path: Path, capsys, verb, priced) -> None:
    """A clip on XAI_API_KEY is metered API credit; the payer hears it before the upload,
    not on the invoice. Imagine prices a clip by length x output size, so the knobs that
    set the amount ride along with the charge."""
    api = _FakeApi()
    heard: list[str] = []

    def call(method, url, token, body):
        if method == "POST":
            heard.append(capsys.readouterr().err)  # everything said before the job was submitted
        return api.call(method, url, token, body)

    _billed_run(verb, tmp_path, credential=video.Credential("console-key", video.AUTH_SOURCE_API_KEY),
                call=call, download=api.download, sleep=lambda s: None)

    notice = [line for line in heard[0].splitlines() if "per-call API charge" in line]
    assert len(notice) == 1
    assert notice[0].startswith(f"[gen] {verb}: running on {video.AUTH_ENV} ")
    assert f"not a subscription{priced} — " in notice[0]
    # The shared prefix denies the subscription once; the detail carries the remedy only
    # (the grok image notice read "not a subscription (...), not your Grok subscription").
    assert notice[0].count("not ") == 1
    assert f"`{video.GROK_LOGIN_COMMAND}` signs your Grok subscription in." in notice[0]


@pytest.mark.parametrize("verb", ["video", "video-extend", "video-edit"])
def test_the_subscription_route_stays_silent_about_billing(tmp_path: Path, capsys, verb) -> None:
    api = _FakeApi()
    _billed_run(verb, tmp_path, credential=video.Credential("tok", video.AUTH_SOURCE_GROK_LOGIN),
                call=api.call, download=api.download, sleep=lambda s: None)
    assert "per-call API charge" not in capsys.readouterr().err


def test_a_request_that_never_leaves_never_announces_a_charge(tmp_path: Path, monkeypatch, capsys) -> None:
    """The line tracks charges, not intentions: a body refused locally costs nothing."""
    monkeypatch.setenv("GROK_HOME", str(tmp_path / "no-login"))
    monkeypatch.setenv("XAI_API_KEY", "console-key")
    api = _FakeApi()
    with pytest.raises(SystemExit, match="--duration must be"):
        video.generate_video(_request(tmp_path, duration=99), call=api.call, download=api.download, sleep=lambda s: None)
    assert api.calls == []
    assert "per-call API charge" not in capsys.readouterr().err


def test_the_cli_announces_on_stderr_leaving_the_report_on_stdout(tmp_path: Path, monkeypatch, capsys) -> None:
    """The credential the environment actually hands the CLI is what decides, and the
    notice goes to stderr so a caller parsing stdout still reads one JSON report."""
    monkeypatch.setenv("GROK_HOME", str(tmp_path / "no-login"))
    monkeypatch.setenv("XAI_API_KEY", "console-key")
    api = _FakeApi()
    monkeypatch.setattr(video, "http_json", api.call)
    monkeypatch.setattr(video, "http_download", api.download)
    monkeypatch.setattr(video.time, "sleep", lambda s: None)

    rc = video.run(image=_still(tmp_path), prompt="sway", prompt_file=None, out=tmp_path / "clip.mp4",
                   duration=6, resolution="720p", aspect_ratio=None, model=video.DEFAULT_MODEL,
                   generate_audio=None, report=None)

    captured = capsys.readouterr()
    assert rc == 0
    assert len([line for line in captured.err.splitlines() if "per-call API charge" in line]) == 1
    assert json.loads(captured.out)["auth_source"] == "XAI_API_KEY"
    assert "console-key" not in captured.err + captured.out  # the key itself is never printed


# --- refusal lines: the provider code, and xAI's one content-policy signal ---------

HTTP_STATUS = re.compile(r"\(HTTP (\d{3})\)")  # what a caller reads the status with


def _refused(tmp_path: Path, api: _FakeApi) -> tuple[str, Path]:
    request = _request(tmp_path)
    with pytest.raises(SystemExit) as refused:
        video.generate_video(request, credential=video.Credential("t", video.AUTH_SOURCE_GROK_LOGIN), call=api.call, download=api.download, sleep=lambda s: None)
    assert not request.out.exists() and not list(tmp_path.glob("*.part"))
    assert api.downloads == []
    return str(refused.value), request.out


def test_a_moderated_clip_is_a_content_policy_refusal(tmp_path: Path) -> None:
    api = _FakeApi(polls=[(200, {"status": "done", "video": {"url": "", "respect_moderation": False}})])
    message, _ = _refused(tmp_path, api)
    assert message == ("video: generation req-1 refused by the provider's content policy (HTTP 200) "
                       "respect_moderation=false; nothing was written")
    assert HTTP_STATUS.search(message).group(1) == "200"


def test_a_moderated_clip_is_refused_even_if_a_url_came_back(tmp_path: Path) -> None:
    api = _FakeApi(polls=[(200, {"status": "done", "video": {"url": "https://vidgen.x.ai/v/1.mp4", "respect_moderation": False}})])
    assert "refused by the provider's content policy" in _refused(tmp_path, api)[0]


def test_a_clip_that_respects_moderation_publishes(tmp_path: Path) -> None:
    api = _FakeApi(polls=[(200, {"status": "done", "video": {"url": "https://vidgen.x.ai/v/1.mp4", "duration": 6.0, "respect_moderation": True}})])
    request = _request(tmp_path)
    video.generate_video(request, credential=video.Credential("t", video.AUTH_SOURCE_GROK_LOGIN), call=api.call, download=api.download, sleep=lambda s: None)
    assert request.out.read_bytes() == MP4


def test_a_failed_poll_names_the_code_inside_its_error_object(tmp_path: Path) -> None:
    """`invalid_argument` covers moderation and bad parameters alike, so it is a code, not a policy verdict."""
    api = _FakeApi(polls=[(200, {"status": "failed", "error": {"code": "invalid_argument", "message": "synthetic"}})])
    message, _ = _refused(tmp_path, api)
    assert message.startswith("video: generation req-1 ended with status='failed' (HTTP 200) code=invalid_argument: ")
    assert "content policy" not in message


@pytest.mark.parametrize("scripted", ["post", "poll"])
def test_what_xai_said_is_shown_with_keys_and_signatures_masked(tmp_path: Path, scripted) -> None:
    """The image lines' mask (refusal.masked) on the clip lines, which already print the body."""
    token = "synthetic-login-token"
    said = (f"echo {token}, key xai-{'Synthetic0Key1For2Tests3Only4' * 3}, "
            "url https://vidgen.x.ai/v/1.mp4?sig=abc, (HTTP 999) code=decoy")
    error = {"code": "invalid_argument", "message": said}
    api = _FakeApi(post=(400, {"error": error})) if scripted == "post" else _FakeApi(polls=[(200, {"status": "failed", "error": error})])
    request = _request(tmp_path)
    with pytest.raises(SystemExit) as refused:
        video.generate_video(request, credential=video.Credential(token, video.AUTH_SOURCE_GROK_LOGIN),
                             call=api.call, download=api.download, sleep=lambda s: None)
    message = str(refused.value)
    front = ("video: generation request refused (HTTP 400) code=invalid_argument: " if scripted == "post"
             else "video: generation req-1 ended with status='failed' (HTTP 200) code=invalid_argument: ")
    # A query runs to the next space or quote, so the comma after it goes too.
    assert message == front + ("error={'code': 'invalid_argument', 'message': 'echo [redacted], key xai-[redacted], "
                               "url https://vidgen.x.ai/v/1.mp4?[redacted] (HTTP 999) code=decoy'}"
                               + ("" if scripted == "post" else ", status='failed'"))
    assert HTTP_STATUS.search(message).group(1) == ("400" if scripted == "post" else "200")


@pytest.mark.parametrize("body", [
    "x" * 285 + " xai-" + "K" * 60,
    {"detail": "x" * 270 + " xai-" + "K" * 60},
], ids=["text", "json"])
def test_a_key_the_300_character_cut_would_split_is_masked_before_the_cut(body) -> None:
    """Masked, then cut: cut first, the key's tail was shorter than a key and slipped through."""
    detail = video._error_detail(body)
    assert "KKKK" not in detail
    assert "xai-[redacted]" in detail and len(detail) <= 300


@pytest.mark.parametrize(("said", "masked"), [
    ("api_key_sk-" + "Synthetic0Key1For2Tests" * 2, "api_key_sk-[redacted]"),
    ("token-xai-" + "Synthetic0Key1For2Tests" * 2, "token-xai-[redacted]"),
    ("task-" + "Synthetic0Key1For2Tests" * 2, "task-" + "Synthetic0Key1For2Tests" * 2),
], ids=["underscore", "hyphen", "word"])
def test_a_key_after_an_underscore_or_hyphen_is_masked_and_a_word_ending_in_sk_is_not(said, masked) -> None:
    assert video._error_detail({"message": said}) == f"message={masked!r}"


@pytest.mark.parametrize(("said", "masked"), [
    ("Authorization: Bearer " + "synthetic.token.value-0123456789", "Authorization: Bearer [redacted]"),
    ("x_Bearer " + "synthetic.token.value-0123456789", "x_Bearer [redacted]"),
    ("x-bearer " + "synthetic.token.value-0123456789", "x-bearer [redacted]"),
    ("xbearer " + "synthetic.token.value-0123456789", "xbearer " + "synthetic.token.value-0123456789"),
], ids=["header", "underscore", "hyphen", "word"])
def test_a_bearer_value_after_an_underscore_or_hyphen_is_masked_and_a_word_ending_in_bearer_is_not(said, masked) -> None:
    assert video._error_detail({"message": said}) == f"message={masked!r}"


@pytest.mark.parametrize(("post", "prefix"), [
    ((400, {"code": "invalid-argument", "error": "synthetic"}), "video: generation request refused (HTTP 400) code=invalid-argument: "),
    ((400, {"error": {"code": "invalid_argument", "message": "synthetic"}}), "video: generation request refused (HTTP 400) code=invalid_argument: "),
    ((400, {"error": "synthetic"}), "video: generation request refused (HTTP 400): "),
    ((400, {"code": "not a code", "error": "synthetic"}), "video: generation request refused (HTTP 400): "),
    ((429, {"error": {"code": "x" * 65}}), "video: generation request refused (HTTP 429): "),
])
def test_a_refused_request_names_a_well_formed_code(tmp_path: Path, post, prefix) -> None:
    message, _ = _refused(tmp_path, _FakeApi(post=post))
    assert message.startswith(prefix)
    assert HTTP_STATUS.search(message).group(1) == str(post[0])


# --- a poll answered 429 or 5xx ----------------------------------------------
# The POST was accepted, so xAI may bill the clip whatever its polls answer. A poll the
# provider throttles (429) or fails on its side (5xx) asks the same request again; ending
# the run there sends a caller that retries on `(HTTP 429)` (video-set, a product's
# backoff) to submit a second clip and pay for both.

DONE = (200, {"status": "done", "video": {"url": "https://vidgen.x.ai/v/1.mp4", "duration": 6.0}, "model": "grok-imagine-video-1.5"})


def _clock(monkeypatch) -> tuple[dict, list[float], object]:
    clock = {"t": 0.0}
    slept: list[float] = []
    monkeypatch.setattr(video.time, "monotonic", lambda: clock["t"])

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        clock["t"] += seconds

    return clock, slept, sleep


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_a_throttled_poll_asks_the_same_request_again_and_publishes(tmp_path: Path, status) -> None:
    api = _FakeApi(polls=[(status, {"code": "synthetic", "error": "slow down"}), (status, {"error": "slow down"}),
                          (200, {"status": "pending"}), DONE])
    slept: list[float] = []
    result = video.generate_video(_request(tmp_path), credential=video.Credential("t", video.AUTH_SOURCE_GROK_LOGIN),
                                  call=api.call, download=api.download, sleep=slept.append)
    assert [call[0] for call in api.calls] == ["POST", "GET", "GET", "GET", "GET"]
    assert {call[1] for call in api.calls[1:]} == {f"{video.API_BASE}/videos/req-1"}
    # The wait doubles from the poll interval while the polls are refused; an answer puts it back.
    assert slept == [video.POLL_INTERVAL_SECONDS * 2, video.POLL_INTERVAL_SECONDS * 4, video.POLL_INTERVAL_SECONDS]
    assert result.out.read_bytes() == MP4
    payload = result.to_dict()
    assert payload["request_id"] == "req-1" and payload["polls"] == 4 and payload["polls_refused"] == 2


def test_a_clip_whose_polls_were_never_refused_reports_no_retry(tmp_path: Path) -> None:
    api = _FakeApi()
    payload = video.generate_video(_request(tmp_path), credential=video.Credential("t", "XAI_API_KEY"),
                                   call=api.call, download=api.download, sleep=lambda s: None).to_dict()
    assert payload["polls"] == 1 and payload["polls_refused"] == 0


def test_polls_refused_until_the_poll_timeout_end_naming_the_request_to_ask_again(tmp_path: Path, monkeypatch, capsys) -> None:
    api = _FakeApi(polls=[(429, {"code": "synthetic", "error": "slow down"})])
    clock, slept, sleep = _clock(monkeypatch)
    request = _request(tmp_path)
    with pytest.raises(video.GenerationUnfinished) as unfinished:
        video.generate_video(request, credential=video.Credential("t", video.AUTH_SOURCE_GROK_LOGIN),
                             call=api.call, download=api.download, sleep=sleep, poll_timeout=300)
    assert isinstance(unfinished.value, SystemExit)
    assert [call[0] for call in api.calls].count("POST") == 1
    assert {call[1] for call in api.calls[1:]} == {f"{video.API_BASE}/videos/req-1"}
    # Doubling, held at the ceiling, the last wait cut to the timeout — and one more poll asked there.
    assert slept == [8.0, 16.0, 32.0, 60.0, 60.0, 60.0, 60.0, 4.0]
    assert clock["t"] == 300 and len(api.calls) == 1 + 9
    message = str(unfinished.value)
    assert message == ("video: generation req-1 was accepted, but its polls were refused (last answer 429) until the "
                       "poll timeout (300s, 9 polls); it may still finish and be billed — ask for request_id=req-1 "
                       f"again with `sprite-gen video-resume --request-id req-1 --out {request.out.resolve()}` "
                       "instead of submitting a new request; nothing was written")
    # No `(HTTP n)`, a refusal line's status: a caller reading `(HTTP 429)` sends the request again.
    assert HTTP_STATUS.search(message) is None
    assert "HTTP 429" not in message + capsys.readouterr().err
    assert (unfinished.value.request_id, unfinished.value.last_http, unfinished.value.polls) == ("req-1", 429, 9)
    assert not request.out.exists() and not list(tmp_path.glob("*.part")) and api.downloads == []


def test_a_poll_timeout_while_pending_names_the_request_to_ask_again(tmp_path: Path, monkeypatch) -> None:
    api = _FakeApi(polls=[(200, {"status": "pending"})])
    _clock_state, _slept, sleep = _clock(monkeypatch)
    with pytest.raises(video.GenerationUnfinished) as unfinished:
        video.generate_video(_request(tmp_path), credential=video.Credential("t", "XAI_API_KEY"),
                             call=api.call, download=api.download, sleep=sleep, poll_timeout=10)
    message = str(unfinished.value)
    assert message.startswith("video: generation req-1 still 'pending' after 4 polls — poll timeout; ")
    assert "request_id=req-1" in message and HTTP_STATUS.search(message) is None
    assert (unfinished.value.request_id, unfinished.value.last_http, unfinished.value.last_status) == ("req-1", 200, "pending")


@pytest.mark.parametrize("status", [400, 404])
def test_a_poll_refused_for_good_ends_at_once_with_its_status(tmp_path: Path, status) -> None:
    api = _FakeApi(polls=[(status, {"error": "synthetic"})])
    message, _ = _refused(tmp_path, api)
    assert len(api.calls) == 2
    assert HTTP_STATUS.search(message).group(1) == str(status)


def test_a_refused_submission_is_never_sent_again(tmp_path: Path) -> None:
    """The POST stays unretried (the provider may have taken it); its 429 keeps `(HTTP 429)` for the caller."""
    api = _FakeApi(post=(429, {"error": "slow down"}))
    message, _ = _refused(tmp_path, api)
    assert [call[0] for call in api.calls] == ["POST"]
    assert HTTP_STATUS.search(message).group(1) == "429"


def test_the_cli_leaves_an_unfinished_report_naming_the_request(tmp_path: Path, monkeypatch, capsys) -> None:
    api = _FakeApi(polls=[(503, {"error": "unavailable"})])
    _clock_state, _slept, sleep = _clock(monkeypatch)
    monkeypatch.setattr(video, "resolve_credential", lambda **_: video.Credential("t", video.AUTH_SOURCE_GROK_LOGIN))
    monkeypatch.setattr(video, "http_json", api.call)
    monkeypatch.setattr(video, "http_download", api.download)
    monkeypatch.setattr(video.time, "sleep", sleep)
    monkeypatch.setattr(video, "POLL_TIMEOUT_SECONDS", 30)
    out, report = tmp_path / "clip.mp4", tmp_path / "clip.report.json"
    with pytest.raises(video.GenerationUnfinished):
        video.run(image=_still(tmp_path), prompt="sway", out=out, report=report)
    assert json.loads(report.read_text(encoding="utf-8")) == {
        "kind": "sprite-gen-video-unfinished",
        "provider": "grok-imagine",
        "verb": "video",
        "auth_source": "grok-login",
        "model": video.DEFAULT_MODEL,
        "request_id": "req-1",
        "out": str(out.resolve()),
        "polls": 4,
        "polls_refused": 4,
        "polls_unanswered": 0,
        "last_poll": {"http": 503, "status": None},
        "elapsed_seconds": 30.0,
    }
    assert capsys.readouterr().out == "" and not out.exists()


# --- a poll that gets no answer, or whose credential xAI rejects ----------------------
# The POST was accepted either way, so the run must leave the request's id. A poll lost on the
# way (the connection failed or timed out) is a GET, safe to ask again like a refused one; a
# poll whose credential is rejected ends unfinished at once — that token will not mend itself.

LOST = "lost"  # a scripted poll that gets no HTTP answer


class _LossyApi(_FakeApi):
    def call(self, method, url, token, body):
        reply = super().call(method, url, token, body)
        if reply == LOST:
            raise video.TransportError("timed out")
        return reply


@pytest.mark.parametrize(("raised", "reason"), [
    (urllib.error.URLError("[Errno 8] nodename nor servname provided"), "[Errno 8] nodename nor servname provided"),
    (urllib.error.URLError(TimeoutError("timed out")), "timed out"),
    (TimeoutError("timed out"), "timed out"),
    (TimeoutError(), "TimeoutError"),
    (ConnectionResetError(54, "Connection reset by peer"), "[Errno 54] Connection reset by peer"),
], ids=["dns", "url-timeout", "timeout", "bare-timeout", "reset"])
def test_http_json_names_a_request_that_got_no_answer(monkeypatch, raised, reason) -> None:
    """Still the SystemExit and line it always was; the type lets a caller that may ask again catch it."""
    def urlopen(request, *, timeout):
        raise raised

    monkeypatch.setattr(xai.urllib.request, "urlopen", urlopen)
    with pytest.raises(xai.TransportError) as lost:
        xai.http_json("GET", f"{xai.API_BASE}/videos/req-1", "t")
    assert isinstance(lost.value, SystemExit) and video.TransportError is xai.TransportError
    assert str(lost.value) == "xai: request failed or timed out; no automatic retry (the server may have accepted it)"
    assert lost.value.reason == reason


def test_a_poll_that_gets_no_answer_is_asked_again_and_publishes(tmp_path: Path, capsys) -> None:
    api = _LossyApi(polls=[LOST, (200, {"status": "pending"}), (503, {"error": "unavailable"}), LOST, DONE])
    slept: list[float] = []
    result = video.generate_video(_request(tmp_path), credential=video.Credential("t", video.AUTH_SOURCE_GROK_LOGIN),
                                  call=api.call, download=api.download, sleep=slept.append)
    assert [call[0] for call in api.calls] == ["POST", "GET", "GET", "GET", "GET", "GET"]
    assert {call[1] for call in api.calls[1:]} == {f"{video.API_BASE}/videos/req-1"}
    # A lost poll waits like a refused one, and a refused poll then a lost one are two in a row.
    assert slept == [8.0, 4.0, 8.0, 16.0]
    assert result.out.read_bytes() == MP4
    payload = result.to_dict()
    assert (payload["polls"], payload["polls_refused"], payload["polls_unanswered"]) == (5, 1, 2)
    assert "[video] poll 1 of generation req-1 got no answer (timed out); asking again in 8s" in capsys.readouterr().err


def test_a_clip_whose_polls_were_all_answered_reports_none_unanswered(tmp_path: Path) -> None:
    api = _FakeApi()
    payload = video.generate_video(_request(tmp_path), credential=video.Credential("t", "XAI_API_KEY"),
                                   call=api.call, download=api.download, sleep=lambda s: None).to_dict()
    assert payload["polls_unanswered"] == 0


def test_polls_that_get_no_answer_until_the_poll_timeout_end_unfinished_naming_the_request(tmp_path: Path, monkeypatch) -> None:
    api = _LossyApi(polls=[LOST])
    clock, slept, sleep = _clock(monkeypatch)
    request = _request(tmp_path)
    with pytest.raises(video.GenerationUnfinished) as unfinished:
        video.generate_video(request, credential=video.Credential("t", video.AUTH_SOURCE_GROK_LOGIN),
                             call=api.call, download=api.download, sleep=sleep, poll_timeout=30)
    assert [call[0] for call in api.calls].count("POST") == 1
    assert slept == [8.0, 16.0, 6.0] and clock["t"] == 30 and len(api.calls) == 1 + 4
    message = str(unfinished.value)
    assert message == ("video: generation req-1 was accepted, but its polls got no answer (last: timed out) until the "
                       "poll timeout (30s, 4 polls); it may still finish and be billed — ask for request_id=req-1 "
                       f"again with `sprite-gen video-resume --request-id req-1 --out {request.out.resolve()}` "
                       "instead of submitting a new request; nothing was written")
    assert HTTP_STATUS.search(message) is None
    report = unfinished.value.to_dict()
    assert (report["polls"], report["polls_refused"], report["polls_unanswered"]) == (4, 0, 4)
    assert report["last_poll"] == {"http": None, "status": None, "error": "timed out"}
    assert not request.out.exists() and not list(tmp_path.glob("*.part")) and api.downloads == []


def test_a_submission_that_gets_no_answer_is_not_unfinished_and_is_never_sent_again(tmp_path: Path) -> None:
    """No request_id came back, so there is nothing to ask for: the POST's line stays as it was."""
    calls: list[str] = []

    def call(method, url, token, body):
        calls.append(method)
        raise video.TransportError("timed out")

    with pytest.raises(SystemExit) as failed:
        video.generate_video(_request(tmp_path), credential=video.Credential("t", video.AUTH_SOURCE_GROK_LOGIN),
                             call=call, download=lambda url, token: MP4, sleep=lambda s: None)
    assert not isinstance(failed.value, video.GenerationUnfinished)
    assert calls == ["POST"] and "no automatic retry" in str(failed.value)


@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.parametrize(("source", "fix"), [
    (video.AUTH_SOURCE_GROK_LOGIN, f"refresh the login with `{video.GROK_REFRESH_COMMAND}` run {video.GROK_REFRESH_WHERE} "
                                   f"or `{video.GROK_LOGIN_COMMAND}`"),
    (video.AUTH_SOURCE_API_KEY, f"check the {video.AUTH_ENV} value"),
], ids=["grok-login", "api-key"])
def test_a_poll_whose_credential_is_rejected_ends_unfinished_at_once(tmp_path: Path, status, source, fix) -> None:
    api = _FakeApi(polls=[(200, {"status": "pending"}), (status, {"error": "token expired"})])
    request = _request(tmp_path)
    with pytest.raises(video.GenerationUnfinished) as unfinished:
        video.generate_video(request, credential=video.Credential("t", source),
                             call=api.call, download=api.download, sleep=lambda s: None)
    assert [call[0] for call in api.calls] == ["POST", "GET", "GET"]
    message = str(unfinished.value)
    assert message == (f"video: generation req-1 was accepted, but xAI rejected the credential ({source}) on poll 2 "
                       f"(answer {status}); it may still finish and be billed — {fix}, then ask for request_id=req-1 "
                       f"again with `sprite-gen video-resume --request-id req-1 --out {request.out.resolve()}` "
                       "instead of submitting a new request; nothing was written")
    assert HTTP_STATUS.search(message) is None and f"HTTP {status}" not in message
    report = unfinished.value.to_dict()
    assert (report["request_id"], report["polls"], report["last_poll"]) == ("req-1", 2, {"http": status, "status": None})
    assert not request.out.exists() and api.downloads == []


# --- video-resume: an accepted request asked for again by its id ----------------------
# The request an unfinished run named is polled, downloaded and published — no POST, so nothing
# is bought twice. A resume cut short again is unfinished on the same id.

RESUMED = video.Credential("t", video.AUTH_SOURCE_GROK_LOGIN)


def _resume(tmp_path: Path, api: _FakeApi, request_id: str = "req-7", **kwargs) -> video.VideoResult:
    return video.resume_video(video.ResumeRequest(request_id=request_id, out=tmp_path / "clip.mp4"),
                              call=api.call, download=api.download, **{"credential": RESUMED, "sleep": lambda s: None, **kwargs})


def test_resume_polls_the_request_without_a_submission_and_publishes(tmp_path: Path, capsys) -> None:
    api = _FakeApi(polls=[(429, {"error": "slow down"}), (200, {"status": "pending"}), DONE])
    result = _resume(tmp_path, api, credential=video.Credential("xai-console", video.AUTH_SOURCE_API_KEY))
    assert [call[0] for call in api.calls] == ["GET", "GET", "GET"]
    assert {call[1] for call in api.calls} == {f"{video.API_BASE}/videos/req-7"}
    assert result.out == (tmp_path / "clip.mp4").resolve() and result.out.read_bytes() == MP4
    payload = result.to_dict()
    assert payload["kind"] == "sprite-gen-video-report" and payload["mode"] == "resume"
    assert (payload["request_id"], payload["polls"], payload["polls_refused"]) == ("req-7", 3, 1)
    assert payload["model"] == "grok-imagine-video-1.5" and payload["duration_reported"] == 6.0
    assert payload["duration_requested"] is None and payload["inputs"] == {} and payload["auth_source"] == "XAI_API_KEY"
    # Nothing new is bought, so the API-key route announces no charge.
    assert "per-call API charge" not in capsys.readouterr().err


@pytest.mark.parametrize("scripted", ["refused", "lost", "pending"])
def test_a_resume_cut_short_again_is_unfinished_on_the_same_request(tmp_path: Path, monkeypatch, scripted) -> None:
    poll = {"refused": (429, {"error": "slow down"}), "lost": LOST, "pending": (200, {"status": "pending"})}[scripted]
    api = _LossyApi(polls=[poll])
    _clock_state, _slept, sleep = _clock(monkeypatch)
    with pytest.raises(video.GenerationUnfinished) as unfinished:
        _resume(tmp_path, api, sleep=sleep, poll_timeout=30)
    assert "POST" not in [call[0] for call in api.calls]
    message = str(unfinished.value)
    assert message.startswith("video-resume: generation req-7 ")
    assert message.endswith("it may still finish and be billed — ask for request_id=req-7 again with "
                            f"`sprite-gen video-resume --request-id req-7 --out {(tmp_path / 'clip.mp4').resolve()}` "
                            "instead of submitting a new request; nothing was written")
    report = unfinished.value.to_dict()
    assert (report["verb"], report["request_id"], report["model"]) == ("video-resume", "req-7", None)
    assert not (tmp_path / "clip.mp4").exists()


@pytest.mark.parametrize("status", [401, 403])
def test_a_resume_whose_credential_is_rejected_is_unfinished_on_the_same_request(tmp_path: Path, status) -> None:
    api = _FakeApi(polls=[(status, {"error": "token expired"})])
    with pytest.raises(video.GenerationUnfinished) as unfinished:
        _resume(tmp_path, api)
    assert [call[0] for call in api.calls] == ["GET"]
    assert str(unfinished.value).startswith(
        f"video-resume: generation req-7 was accepted, but xAI rejected the credential (grok-login) on poll 1 (answer {status}); ")


def test_a_resume_of_a_request_xai_does_not_know_ends_with_its_status(tmp_path: Path) -> None:
    api = _FakeApi(polls=[(404, {"error": "not found"})])
    with pytest.raises(SystemExit) as refused:
        _resume(tmp_path, api)
    assert not isinstance(refused.value, video.GenerationUnfinished)
    assert str(refused.value).startswith("video-resume: generation req-7 ")
    assert HTTP_STATUS.search(str(refused.value)).group(1) == "404"
    assert not (tmp_path / "clip.mp4").exists()


@pytest.mark.parametrize("request_id", ["", "  ", "req/../1", "req-1?x=1", "req 1", "req#1", "%2e%2e", "-req"])
def test_resume_refuses_what_cannot_be_a_request_id_before_any_call(tmp_path: Path, request_id) -> None:
    api = _FakeApi()
    with pytest.raises(SystemExit, match="video-resume: --request-id"):
        _resume(tmp_path, api, request_id=request_id)
    assert api.calls == []


def test_the_resume_cli_writes_the_report_and_prints_it(tmp_path: Path, monkeypatch, capsys) -> None:
    api = _FakeApi()
    monkeypatch.setattr(video, "resolve_credential", lambda **_: RESUMED)
    monkeypatch.setattr(video, "http_json", api.call)
    monkeypatch.setattr(video, "http_download", api.download)
    monkeypatch.setattr(video.time, "sleep", lambda s: None)
    out, report = tmp_path / "clip.mp4", tmp_path / "clip.report.json"
    assert video.run_resume(request_id="req-7", out=out, report=report) == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert (payload["mode"], payload["request_id"], payload["out"]) == ("resume", "req-7", str(out.resolve()))
    assert json.loads(capsys.readouterr().out)["report"] == str(report.resolve())
    assert [call[0] for call in api.calls] == ["GET"] and out.read_bytes() == MP4
    with pytest.raises(TypeError, match="unexpected keyword"):
        video.run_resume(request_id="req-7", out=out, prompt="p")


def test_the_resume_cli_leaves_an_unfinished_report_on_the_same_request(tmp_path: Path, monkeypatch, capsys) -> None:
    api = _FakeApi(polls=[(200, {"status": "pending"})])
    _clock_state, _slept, sleep = _clock(monkeypatch)
    monkeypatch.setattr(video, "resolve_credential", lambda **_: RESUMED)
    monkeypatch.setattr(video, "http_json", api.call)
    monkeypatch.setattr(video, "http_download", api.download)
    monkeypatch.setattr(video.time, "sleep", sleep)
    monkeypatch.setattr(video, "POLL_TIMEOUT_SECONDS", 10)
    out, report = tmp_path / "clip.mp4", tmp_path / "clip.report.json"
    with pytest.raises(video.GenerationUnfinished):
        video.run_resume(request_id="req-7", out=out, report=report)
    assert json.loads(report.read_text(encoding="utf-8")) == {
        "kind": "sprite-gen-video-unfinished",
        "provider": "grok-imagine",
        "verb": "video-resume",
        "auth_source": "grok-login",
        "model": None,
        "request_id": "req-7",
        "out": str(out.resolve()),
        "polls": 4,
        "polls_refused": 0,
        "polls_unanswered": 0,
        "last_poll": {"http": 200, "status": "pending"},
        "elapsed_seconds": 10.0,
    }
    assert capsys.readouterr().out == "" and not out.exists()


def test_the_unified_cli_routes_video_resume_under_gen() -> None:
    from sprite_gen import cli

    assert "video-resume" in cli.command_domains()["gen"]
    parser = cli._build_parser()
    args = parser.parse_args(["video-resume", "--request-id", "req-7", "--out", "o.mp4", "--report", "r.json"])
    assert (args.command, args.request_id, args.out, args.report) == ("video-resume", "req-7", Path("o.mp4"), Path("r.json"))
    for missing in (["--out", "o.mp4"], ["--request-id", "req-7"]):
        with pytest.raises(SystemExit):
            parser.parse_args(["video-resume", *missing])


# --- an answer cut short, garbled or not JSON, over a real socket ---------------------
# A poll whose answer began and broke (the body ends before its Content-Length, the status line is
# not one) or came whole but is not JSON gave the run nothing to read: it is asked again on the same
# request like a lost one, and at the poll timeout the run is unfinished on that id — never an
# exception that leaves no id behind. A local server answers over a real socket, so http.client
# raises what it raises on the wire.


def _answer(status: str, body: bytes, *, length: int | None = None, content_type: str = "application/json") -> bytes:
    """A raw HTTP/1.1 answer whose Content-Length says `length` (by default the body's own)."""
    head = (f"HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\n"
            f"Content-Length: {len(body) if length is None else length}\r\nConnection: close\r\n\r\n")
    return head.encode("latin-1") + body


ACCEPTED = _answer("200 OK", b'{"request_id": "req-1"}')
CUT = _answer("200 OK", b'{"status":', length=200)  # 10 of 200 bytes, then the connection closes
CUT_REFUSAL = _answer("503 Service Unavailable", b'{"error":', length=200)  # a refusal's body cut the same way
GARBLED = b"garbage\r\n"  # where the status line should be
NOT_JSON = _answer("200 OK", b"<html>upstream busy</html>")  # whole, and not JSON
PENDING_ON_WIRE = _answer("200 OK", b'{"status": "pending"}')
DONE_ON_WIRE = _answer("200 OK", json.dumps(DONE[1]).encode("utf-8"))
CUT_SAID = "IncompleteRead(10 bytes read, 190 more expected)"


class _Hold:
    """Scripted bytes after which the server holds the connection open, until the client gives up and closes it."""

    def __init__(self, data: bytes) -> None:
        self.data = data


class _Wire:
    """A local xAI on 127.0.0.1: each request, read whole, gets the next scripted bytes (the last one
    repeats), then the server closes the connection whatever its Content-Length promised — or, for a
    `_Hold`, keeps it open until the client closes it."""

    def __init__(self, answers: list[bytes | _Hold]) -> None:
        self.answers = list(answers)
        self.seen: list[tuple[str, str]] = []
        self.listener = socket.create_server(("127.0.0.1", 0))
        self.listener.settimeout(0.1)
        self.root = f"http://127.0.0.1:{self.listener.getsockname()[1]}"
        self.base = f"{self.root}/v1"
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self) -> None:
        while not self.stopped.is_set():
            try:
                conn, _ = self.listener.accept()
            except TimeoutError:
                continue
            with conn:
                conn.settimeout(5)
                received = b""
                while b"\r\n\r\n" not in received:
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    received += chunk
                head, _, body = received.partition(b"\r\n\r\n")
                lines = head.decode("latin-1").split("\r\n")
                length = next((int(line.split(":", 1)[1]) for line in lines[1:]
                               if line.lower().startswith("content-length:")), 0)
                while len(body) < length:  # a body left unread would make the close a reset
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    body += chunk
                method, path = lines[0].split(" ")[:2]
                self.seen.append((method, path))
                answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
                if isinstance(answer, _Hold):
                    conn.sendall(answer.data)
                    try:
                        while conn.recv(65536):
                            pass
                    except OSError:
                        pass
                else:
                    conn.sendall(answer)
                    conn.shutdown(socket.SHUT_WR)

    def close(self) -> None:
        self.stopped.set()
        self.thread.join(5)
        self.listener.close()


@pytest.fixture
def wire(monkeypatch):
    """`wire(*answers)` starts a _Wire and points every xAI URL of the video module at it."""
    started: list[_Wire] = []

    def serve(*answers: bytes | _Hold) -> _Wire:
        server = _Wire(list(answers))
        started.append(server)
        monkeypatch.setattr(video, "API_BASE", server.base)
        return server

    yield serve
    for server in started:
        server.close()


def _on_wire(method, url, token, body):
    # The real transport, with a timeout a test can wait out if the server ever stalls.
    return xai.http_json(method, url, token, body, timeout=5)


@pytest.mark.parametrize(("answer", "line", "reason"), [
    (CUT, xai.TRANSPORT_FAILED, CUT_SAID),
    (CUT_REFUSAL, xai.TRANSPORT_FAILED, "IncompleteRead(9 bytes read, 191 more expected)"),
    (GARBLED, xai.TRANSPORT_FAILED, "BadStatusLine('garbage\\r\\n')"),
    (NOT_JSON, "xai: response was not valid JSON", "response was not valid JSON"),
], ids=["cut", "cut-refusal", "garbled", "not-json"])
def test_http_json_reads_an_answer_it_cannot_read_whole_as_no_answer(wire, answer, line, reason) -> None:
    """The line each always had — the transport's, or the not-JSON one — on the type a poll asks again on."""
    server = wire(answer)
    with pytest.raises(xai.TransportError) as lost:
        _on_wire("GET", f"{server.base}/videos/req-1", "t", None)
    assert isinstance(lost.value, SystemExit)
    assert (str(lost.value), lost.value.reason) == (line, reason)
    assert server.seen == [("GET", "/v1/videos/req-1")]


def test_a_poll_cut_short_garbled_or_not_json_is_asked_again_and_publishes(tmp_path: Path, wire, capsys) -> None:
    server = wire(ACCEPTED, CUT, GARBLED, NOT_JSON, CUT_REFUSAL, PENDING_ON_WIRE, DONE_ON_WIRE)
    slept: list[float] = []
    result = video.generate_video(_request(tmp_path), credential=video.Credential("t", video.AUTH_SOURCE_GROK_LOGIN),
                                  call=_on_wire, download=lambda url, token: MP4, sleep=slept.append)
    assert server.seen == [("POST", "/v1/videos/generations")] + [("GET", "/v1/videos/req-1")] * 6
    # Four unreadable answers in a row wait like four lost polls; the pending answer puts the wait back.
    assert slept == [8.0, 16.0, 32.0, 60.0, 4.0]
    assert result.out.read_bytes() == MP4
    payload = result.to_dict()
    assert (payload["request_id"], payload["polls"], payload["polls_refused"], payload["polls_unanswered"]) == ("req-1", 6, 0, 4)
    err = capsys.readouterr().err
    assert f"[video] poll 1 of generation req-1 got no answer ({CUT_SAID}); asking again in 8s" in err
    assert "[video] poll 3 of generation req-1 got no answer (response was not valid JSON); asking again in 32s" in err


def test_a_resume_whose_poll_is_cut_short_asks_the_same_request_again(tmp_path: Path, wire) -> None:
    """The validator's counterexample: a resumed poll cut 10 bytes into its 200-byte body."""
    server = wire(CUT, DONE_ON_WIRE)
    result = video.resume_video(video.ResumeRequest(request_id="req-1", out=tmp_path / "clip.mp4"), credential=RESUMED,
                                call=_on_wire, download=lambda url, token: MP4, sleep=lambda s: None)
    assert server.seen == [("GET", "/v1/videos/req-1")] * 2
    assert result.out.read_bytes() == MP4 and result.to_dict()["polls_unanswered"] == 1


def test_polls_cut_short_until_the_poll_timeout_leave_an_unfinished_report_naming_the_request(
        tmp_path: Path, wire, monkeypatch, capsys) -> None:
    server = wire(ACCEPTED, CUT)
    clock, slept, sleep = _clock(monkeypatch)
    monkeypatch.setattr(video, "resolve_credential", lambda **_: video.Credential("t", video.AUTH_SOURCE_GROK_LOGIN))
    monkeypatch.setattr(video.time, "sleep", sleep)
    monkeypatch.setattr(video, "POLL_TIMEOUT_SECONDS", 30)
    out, report = tmp_path / "clip.mp4", tmp_path / "clip.report.json"
    with pytest.raises(video.GenerationUnfinished) as unfinished:
        video.run(image=_still(tmp_path), prompt="sway", out=out, report=report)
    assert server.seen == [("POST", "/v1/videos/generations")] + [("GET", "/v1/videos/req-1")] * 4
    assert slept == [8.0, 16.0, 6.0] and clock["t"] == 30
    message = str(unfinished.value)
    assert message == (f"video: generation req-1 was accepted, but its polls got no answer (last: {CUT_SAID}) until the "
                       "poll timeout (30s, 4 polls); it may still finish and be billed — ask for request_id=req-1 "
                       f"again with `sprite-gen video-resume --request-id req-1 --out {out.resolve()}` "
                       "instead of submitting a new request; nothing was written")
    assert HTTP_STATUS.search(message) is None
    assert json.loads(report.read_text(encoding="utf-8")) == {
        "kind": "sprite-gen-video-unfinished",
        "provider": "grok-imagine",
        "verb": "video",
        "auth_source": "grok-login",
        "model": video.DEFAULT_MODEL,
        "request_id": "req-1",
        "out": str(out.resolve()),
        "polls": 4,
        "polls_refused": 0,
        "polls_unanswered": 4,
        "last_poll": {"http": None, "status": None, "error": CUT_SAID},
        "elapsed_seconds": 30.0,
    }
    assert capsys.readouterr().out == "" and not out.exists() and not list(tmp_path.glob("*.part"))


def test_a_submission_cut_short_is_not_unfinished_and_is_never_sent_again(tmp_path: Path, wire) -> None:
    """No request_id could be read, so there is nothing to ask for: the POST ends at once on the transport's line."""
    server = wire(CUT)
    with pytest.raises(SystemExit) as failed:
        video.generate_video(_request(tmp_path), credential=video.Credential("t", video.AUTH_SOURCE_GROK_LOGIN),
                             call=_on_wire, download=lambda url, token: MP4, sleep=lambda s: None)
    assert isinstance(failed.value, xai.TransportError) and not isinstance(failed.value, video.GenerationUnfinished)
    assert str(failed.value) == xai.TRANSPORT_FAILED and failed.value.reason == CUT_SAID
    assert server.seen == [("POST", "/v1/videos/generations")]


# --- a clip that is done and could not be downloaded, over a real socket ---------------
# The poll said done: the clip is made and may be billed, and its request_id is the one way back to
# it. A download that gets no answer it can read (the connection refused, the body cut short, a read
# that timed out) or that the clip's host refuses with 429 or 5xx is downloaded again, up to three
# downloads; one still failing — or refused with another status, or bytes that are not an mp4 —
# leaves the run unfinished on that request_id, never an exception that names no request, and
# `video-resume` polls it again for a fresh URL.

CLIP = _answer("200 OK", MP4, content_type="video/mp4")
CLIP_CUT = _answer("200 OK", MP4[:10], length=len(MP4), content_type="video/mp4")  # closes 10 bytes in
CLIP_HELD = _Hold(CLIP_CUT)  # the same 10 bytes, then the connection stays open: the read times out
CLIP_REFUSED = _answer("503 Service Unavailable", b"busy", content_type="text/plain")
CLIP_GONE = _answer("404 Not Found", b"gone", content_type="text/plain")
NOT_A_CLIP_BODY = b"<html>this link has expired</html>"
NOT_A_CLIP = _answer("200 OK", NOT_A_CLIP_BODY, content_type="text/html")  # whole, and not an mp4
CLIP_CUT_SAID = f"IncompleteRead(10 bytes read, {len(MP4) - 10} more expected)"
REFUSED_SAID = f"[Errno {errno.ECONNREFUSED}] {os.strerror(errno.ECONNREFUSED)}"
READ_TIMEOUT_SECONDS = 1.0
SUBMITTED = [("POST", "/v1/videos/generations"), ("GET", "/v1/videos/req-1")]


def _done_at(url: str) -> bytes:
    """The done poll, naming the clip at `url`."""
    return _answer("200 OK", json.dumps({**DONE[1], "video": {**DONE[1]["video"], "url": url}}).encode("utf-8"))


def _clip_wire(wire, *downloads: bytes | _Hold, clip_url: str | None = None) -> _Wire:
    """The POST accepted, one poll done naming the clip at `clip_url` (by default on the wire itself,
    `/clips/1.mp4`), then `downloads` answer the clip's downloads in turn."""
    server = wire()
    server.answers = [ACCEPTED, _done_at(clip_url or f"{server.root}/clips/1.mp4"), *downloads]
    return server


def _refused_url() -> str:
    """A clip URL on a port nothing holds: a free port is taken and let go, so a connection to it is refused.
    (A port held bound and not listening is no such port on macOS: its SYN is dropped and the connect times out.)"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    return f"http://127.0.0.1:{port}/clips/1.mp4"


def _cli_on_wire(monkeypatch) -> list[float]:
    """The CLI on its own transports, a read that waits READ_TIMEOUT_SECONDS for the clip, a clock its waits move."""
    _clock_state, slept, sleep = _clock(monkeypatch)
    monkeypatch.setattr(video, "resolve_credential", lambda **_: video.Credential("t", video.AUTH_SOURCE_GROK_LOGIN))
    monkeypatch.setattr(video.time, "sleep", sleep)
    monkeypatch.setattr(video, "HTTP_TIMEOUT_SECONDS", READ_TIMEOUT_SECONDS)
    return slept


@pytest.mark.parametrize(("failure", "said"), [
    ("cut", CLIP_CUT_SAID),
    ("read-timeout", "timed out"),
    ("refused-connection", REFUSED_SAID),
], ids=["cut", "read-timeout", "refused-connection"])
def test_a_clip_whose_downloads_get_no_answer_leaves_an_unfinished_report_naming_the_request(
        tmp_path: Path, wire, monkeypatch, capsys, failure, said) -> None:
    if failure == "refused-connection":
        server, downloads = _clip_wire(wire, clip_url=_refused_url()), []
    else:
        server = _clip_wire(wire, CLIP_CUT if failure == "cut" else CLIP_HELD)
        downloads = [("GET", "/clips/1.mp4")] * 3
    slept = _cli_on_wire(monkeypatch)
    out, report = tmp_path / "clip.mp4", tmp_path / "clip.report.json"
    with pytest.raises(video.GenerationUnfinished) as unfinished:
        video.run(image=_still(tmp_path), prompt="sway", out=out, report=report)
    # One POST, one poll; the clip downloaded three times with waits 8 and 16 between.
    assert server.seen == SUBMITTED + downloads
    assert slept == [8.0, 16.0]
    message = str(unfinished.value)
    assert message == (f"video: generation req-1 is done, but its download got no answer (last: {said}) after 3 "
                       "attempts; the clip is made and may be billed — ask for request_id=req-1 again with "
                       f"`sprite-gen video-resume --request-id req-1 --out {out.resolve()}` instead of submitting a "
                       "new request; nothing was written")
    assert HTTP_STATUS.search(message) is None
    assert json.loads(report.read_text(encoding="utf-8")) == {
        "kind": "sprite-gen-video-unfinished",
        "provider": "grok-imagine",
        "verb": "video",
        "auth_source": "grok-login",
        "model": video.DEFAULT_MODEL,
        "request_id": "req-1",
        "out": str(out.resolve()),
        "polls": 1,
        "polls_refused": 0,
        "polls_unanswered": 0,
        "last_poll": {"http": 200, "status": "done"},
        "downloads": 3,
        "last_download": {"http": None, "error": said},
        "elapsed_seconds": 24.0,
    }
    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"[video] download 1 of generation req-1 got no answer ({said}); downloading again in 8s" in captured.err
    assert f"[video] download 2 of generation req-1 got no answer ({said}); downloading again in 16s" in captured.err
    assert not out.exists() and not list(tmp_path.glob("*.part"))


def test_a_download_cut_short_or_refused_is_downloaded_again_and_publishes(tmp_path: Path, wire, monkeypatch, capsys) -> None:
    server = _clip_wire(wire, CLIP_CUT, CLIP_REFUSED, CLIP)
    monkeypatch.setattr(video, "HTTP_TIMEOUT_SECONDS", READ_TIMEOUT_SECONDS)
    slept: list[float] = []
    result = video.generate_video(_request(tmp_path), credential=video.Credential("t", video.AUTH_SOURCE_GROK_LOGIN),
                                  call=_on_wire, sleep=slept.append)
    assert server.seen == SUBMITTED + [("GET", "/clips/1.mp4")] * 3
    assert slept == [8.0, 16.0]
    assert result.out.read_bytes() == MP4
    payload = result.to_dict()
    assert (payload["request_id"], payload["polls"], payload["downloads"]) == ("req-1", 1, 3)
    err = capsys.readouterr().err
    assert f"[video] download 1 of generation req-1 got no answer ({CLIP_CUT_SAID}); downloading again in 8s" in err
    assert "[video] download 2 of generation req-1 was refused (503); downloading again in 16s" in err


def test_a_clip_downloaded_whole_at_once_reports_one_download(tmp_path: Path) -> None:
    api = _FakeApi()
    payload = video.generate_video(_request(tmp_path), credential=video.Credential("t", "XAI_API_KEY"),
                                   call=api.call, download=api.download, sleep=lambda s: None).to_dict()
    assert payload["downloads"] == 1


@pytest.mark.parametrize(("answer", "downloads", "said", "status"), [
    (CLIP_REFUSED, 3, "was refused (last answer 503) after 3 attempts", 503),
    (CLIP_GONE, 1, "was refused (answer 404)", 404),
], ids=["refused-503", "gone-404"])
def test_a_download_refused_ends_unfinished_on_the_same_request(tmp_path: Path, wire, answer, downloads, said, status) -> None:
    """429 and 5xx are downloaded again; any other status ends at once — `video-resume` asks for a fresh URL."""
    server = _clip_wire(wire, answer)
    slept: list[float] = []
    request = _request(tmp_path)
    with pytest.raises(video.GenerationUnfinished) as unfinished:
        video.generate_video(request, credential=video.Credential("t", video.AUTH_SOURCE_GROK_LOGIN),
                             call=_on_wire, sleep=slept.append)
    assert server.seen == SUBMITTED + [("GET", "/clips/1.mp4")] * downloads
    assert slept == [8.0, 16.0][:downloads - 1]
    message = str(unfinished.value)
    assert message == (f"video: generation req-1 is done, but its download {said}; the clip is made and may be billed — "
                       f"ask for request_id=req-1 again with `sprite-gen video-resume --request-id req-1 --out "
                       f"{request.out.resolve()}` instead of submitting a new request; nothing was written")
    # No `(HTTP n)`: a caller reading `(HTTP 503)` sends the request again and buys the clip twice.
    assert HTTP_STATUS.search(message) is None
    report = unfinished.value.to_dict()
    assert (report["request_id"], report["downloads"], report["last_download"]) == ("req-1", downloads, {"http": status})
    assert not request.out.exists() and not list(tmp_path.glob("*.part"))


def test_a_download_that_is_not_an_mp4_is_unfinished_on_the_same_request(tmp_path: Path, wire) -> None:
    """The ruling on a failed mp4 check: bytes that came whole and are not an mp4 are not xAI's verdict on
    the request (that is `respect_moderation`, read before the download), so the run is unfinished on its
    id — `video-resume` polls for a fresh URL — and never a refusal that leaves only a second purchase.
    The same URL would bring the same bytes, so it is not downloaded again."""
    server = _clip_wire(wire, NOT_A_CLIP)
    request = _request(tmp_path)
    with pytest.raises(video.GenerationUnfinished) as unfinished:
        video.generate_video(request, credential=video.Credential("t", video.AUTH_SOURCE_GROK_LOGIN),
                             call=_on_wire, sleep=lambda s: None)
    assert server.seen == SUBMITTED + [("GET", "/clips/1.mp4")]
    size = len(NOT_A_CLIP_BODY)
    assert str(unfinished.value) == (
        f"video: generation req-1 is done, but its download brought {size} bytes that are not an mp4 (no ftyp box); "
        "the clip is made and may be billed — ask for request_id=req-1 again with `sprite-gen video-resume "
        f"--request-id req-1 --out {request.out.resolve()}` instead of submitting a new request; nothing was written")
    report = unfinished.value.to_dict()
    assert (report["downloads"], report["last_download"]) == (
        1, {"http": None, "error": "not an mp4 (no ftyp box)", "bytes": size})
    assert not request.out.exists() and not list(tmp_path.glob("*.part"))


def test_a_clip_left_unfinished_by_its_download_is_published_by_video_resume(tmp_path: Path, wire, monkeypatch, capsys) -> None:
    """The run's report names the request; `video-resume` on that id polls without a POST, gets the clip's
    fresh URL and publishes it."""
    server = _clip_wire(wire, CLIP_CUT, CLIP_CUT, CLIP_CUT)
    server.answers += [_done_at(f"{server.root}/clips/2.mp4"), CLIP]
    _cli_on_wire(monkeypatch)
    out, report = tmp_path / "clip.mp4", tmp_path / "clip.report.json"
    with pytest.raises(video.GenerationUnfinished):
        video.run(image=_still(tmp_path), prompt="sway", out=out, report=report)
    left = json.loads(report.read_text(encoding="utf-8"))
    assert (left["kind"], left["request_id"], left["downloads"]) == ("sprite-gen-video-unfinished", "req-1", 3)
    capsys.readouterr()

    assert video.run_resume(request_id=left["request_id"], out=out, report=report) == 0
    assert server.seen == SUBMITTED + [("GET", "/clips/1.mp4")] * 3 + [("GET", "/v1/videos/req-1"), ("GET", "/clips/2.mp4")]
    resumed = json.loads(report.read_text(encoding="utf-8"))
    assert (resumed["kind"], resumed["mode"], resumed["request_id"]) == ("sprite-gen-video-report", "resume", "req-1")
    assert (resumed["polls"], resumed["downloads"]) == (1, 1)
    assert json.loads(capsys.readouterr().out)["report"] == str(report.resolve())
    assert out.read_bytes() == MP4 and not list(tmp_path.glob("*.part"))
