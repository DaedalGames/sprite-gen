# Audio Asset Reference: `sprite-gen sound`

## Overview and CLI Usage

```bash
sprite-gen sound --kind sfx|bgm --prompt TEXT (or --prompt-file PATH) --duration SECONDS --event NAME --out-dir DIR [--cache-dir DIR] [--loop|--no-loop] [--target-lufs FLOAT] [--crossfade SECONDS]
```

### Options and System Requirements
- `--kind`: `sfx` (default `--no-loop`, -18 LUFS) or `bgm` (default `--loop`, -20 LUFS, instrumental).
- `--prompt` / `--prompt-file`: Prompt text or file. The provider may reject prompts containing artist names, franchises, or lyrics; the CLI omits a dialogue verb.
- `--duration`: Requested duration in seconds.
- `--event`: Safe identifier (1..128 characters); first character must be alphanumeric, remaining characters can be alphanumeric, dot, underscore, dash, or colon.
- `--out-dir`: Destination; must be new or match existing recipe and output file digests.
- `--cache-dir`: Defaults to `$XDG_CACHE_HOME/sprite-gen/audio` or `~/.cache/sprite-gen/audio` (never inside `--out-dir`; local by default but can be copied; no cross-cache-directory or cross-machine duplicate-request guarantee).
- `--target-lufs`: Target integrated loudness within `-36..-9` LUFS.
- `--crossfade`: Overlap duration in seconds.
- **Environment & PATH**: `ELEVENLABS_API_KEY` is required only for new paid requests (offline work needs none). `ffmpeg` with `libopus` and `aac` encoders is required on `PATH` and checked before paid calls (`ffprobe` is not required).

## Provider APIs and Cost Accounting

Requests use ElevenLabs (`output_format=mp3_44100_128`):
- **SFX**: `POST /v1/sound-generation` (`text`, `duration_seconds` 0.5..30 [API reference takes precedence over 0.1s overview bound], `loop`, `model_id="eleven_text_to_sound_v2"`, `prompt_influence=0.3`).
- **BGM**: `POST /v1/music` (`prompt` max 4100 characters, `music_length_ms` 3000..600000, `model_id="music_v1"`, `force_instrumental=true`).

### Billing and Usage Accounting
Checked 2026-10-13: [ElevenLabs Pricing](https://elevenlabs.io/pricing/api) quotes SFX at $0.12/min and music at $0.15/min (USD), FAQ notes generation metering, and legacy [Capabilities](https://elevenlabs.io/docs/overview/capabilities/sound-effects) states 40 credits/sec for fixed-duration SFX. Because plans vary, no currency conversion is hardcoded; the runtime records provider `character_cost` directly from responses when present (never guessed currency). Check quotas via [User Subscription](https://elevenlabs.io/docs/api-reference/user/subscription/get) (`character_count`/`character_limit`).

## Cache Idempotency and Retries

Raw paid audio is stored in SQLite keyed by provider parameter digest. Tuning `--event`, loudness, or exports reuses raw cache without new purchases:
- **Locking & Recovery**: Atomic claims precede requests; raw blobs and receipts commit on success. Duplicates, crashes, or uncertain results refuse resubmission. Do not delete unresolved cache entries blindly; inspect provider billing/history first. All errors remain unresolved.
- **Offline Processing**: Downstream processing runs offline. Identical runs require no network.
- **Retries**: Uses `sprite_gen.gen.retry` exclusively. ElevenLabs 429 retries only when `detail.status` is `system_busy` or `too_many_concurrent_requests` (max 2 retries via `Retry-After` or engine backoff; see [ElevenLabs 429 Errors](https://elevenlabs.io/docs/help-center/technical/api-error-code-429)). No other 429, 5xx, transport retries, or provider fallbacks occur. Emits only bounded status, code, and request IDs without error prose.

## Audio DSP, Looping, and Quality Gates

1. **Cosine Looping**: Blends tail and head with complementary cosine crossfade, subtracting overlap from provider length (default `min(0.25, requested_duration/8)`, under half source duration; 20s BGM yields ~19.75s). Cosine tail/head overlap is followed by a 5ms cosine taper to zero at both outer boundaries to suppress codec priming discontinuities; duration is not shortened further.
2. **Normalization**: Two-pass [FFmpeg Loudnorm](https://ffmpeg.org/ffmpeg-filters.html#loudnorm) targets -2 dBTP with repeating loop context. At most three per-format encode/measure passes correct codec reconstruction overshoot using measured gain and a tighter oversampled [FFmpeg Alimiter](https://ffmpeg.org/ffmpeg-filters.html#alimiter) (192kHz, auto-gain off, latency compensated); all passes are offline and never trigger provider retries.
3. **Quality Gates**: Output formats must decode to target +/-1 LU and true peak <=-1 dBTP. Silence or unmeasurable audio fails. Numeric checks do not judge musical beat alignment.
4. **Export Formats**: Outputs 48kHz stereo Opus in OGG (`sound.ogg`, `libopus`, 160 kbps) and AAC in M4A (`sound.m4a`, `aac`, 192 kbps).
5. **Playback**: Web Audio uses `loopStart` (`0`) and `loopEnd` (`processed PCM duration`) on decoded `AudioBuffer` to exclude padding; HTML audio (such as `<audio loop>`) is not guaranteed gapless.

## Manifest Schema and Pack Integration

Outputs `sound.json` (v1, `kind="sprite-gen-sound"`): `event`, `audio_kind`, `loop`, `duration_seconds`, `target_lufs`, `codec_taper_seconds` (0.005 for loops, 0 for nonloops), `loop_start_seconds` (0), `loop_end_seconds` (processed PCM duration), and `files` (`ogg` and `m4a` entries with `path`, decoded `duration_seconds`, `lufs`, `true_peak_dbfs` [true peak dBTP], `sha256`, `encode_passes`, `pcm_peak_limit_dbtp`).

Game pack recipes reference audio via `request.audio = [{"id": "hit", "source": "../hit/sound.json"}]`:
- Files copy flat as `audio-<id>.ogg` and `audio-<id>.m4a`; `manifest.audio[id]` preserves properties and metadata.
- Consumers match `event` strings against animation `events[].name` (packs require no animation for UI/music and generate no runtime code).
- Audio sizes count toward the 50,000,000-byte pack limit; audio-free packages remain unchanged.

## Examples

```bash
sprite-gen sound --kind sfx --prompt "laser" --duration 0.8 --event laser:shot --out-dir ./dist/laser
sprite-gen sound --kind bgm --prompt "synth cave" --duration 20.0 --event cave:bgm --out-dir ./dist/cave
```

### Manifest Audio Fragment (Partial)
```json
{
  "audio": {
    "hit": {
      "event": "hit",
      "audio_kind": "sfx",
      "loop": false,
      "duration_seconds": 1.25,
      "target_lufs": -18.0,
      "loop_start_seconds": 0.0,
      "loop_end_seconds": 1.25,
      "files": {
        "ogg": { "path": "audio-hit.ogg" },
        "m4a": { "path": "audio-hit.m4a" }
      }
    }
  }
}
```

## References (Checked 2026-10-13)
- [ElevenLabs SFX](https://elevenlabs.io/docs/api-reference/text-to-sound-effects/convert)
- [ElevenLabs Music](https://elevenlabs.io/docs/api-reference/music/compose)
- [ElevenLabs 429 Errors](https://elevenlabs.io/docs/help-center/technical/api-error-code-429)
- [ElevenLabs Capabilities](https://elevenlabs.io/docs/overview/capabilities/sound-effects)
- [ElevenLabs Pricing](https://elevenlabs.io/pricing/api)
- [ElevenLabs Subscription](https://elevenlabs.io/docs/api-reference/user/subscription/get)
- [FFmpeg Loudnorm](https://ffmpeg.org/ffmpeg-filters.html#loudnorm)
- [FFmpeg Alimiter](https://ffmpeg.org/ffmpeg-filters.html#alimiter)
