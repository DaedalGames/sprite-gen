# SPDX-License-Identifier: Apache-2.0
"""PCM loop overlap, EBU R128 normalization and measured browser audio encodes."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np

RATE = 48000
CODEC_TAPER_SECONDS = .005


def require_tools():
    for name in ('ffmpeg',):
        if not shutil.which(name):
            raise ValueError(f'{name} is required on PATH for sound processing')
    encoders = _ffmpeg(['-encoders']).stdout.decode(errors='replace')
    for codec in ('libopus', 'aac'):
        if not re.search(r'(?m)^\s*A\S*\s+'+codec+r'\s', encoders):
            raise ValueError(f'ffmpeg requires the {codec} encoder before a paid sound request')


def _ffmpeg(args, data=None):
    result = subprocess.run(['ffmpeg', '-hide_banner', '-nostdin', '-y', *args],
                            input=data, capture_output=True, timeout=180)
    if result.returncode:
        # Source metadata can contain untrusted prose. Keep diagnostics local and bounded.
        raise ValueError('ffmpeg audio processing failed; check input audio and codec availability')
    return result


def decode(path):
    result = _ffmpeg(['-v', 'error', '-i', str(path), '-map', '0:a:0', '-ac', '2', '-ar', str(RATE),
                      '-f', 'f32le', 'pipe:1'])
    samples = np.frombuffer(result.stdout, dtype='<f4').reshape(-1, 2).copy()
    if not len(samples) or not np.isfinite(samples).all():
        raise ValueError('source has no finite audio samples')
    return samples


def make_loop(samples, seconds):
    count = round(seconds * RATE)
    if count < 2 or count * 2 >= len(samples):
        raise ValueError('loop crossfade must be positive and shorter than half the source')
    weight = (.5 - .5 * np.cos(np.linspace(0, np.pi, count)))[:, None]
    overlap = samples[-count:] * (1 - weight) + samples[:count] * weight
    output = np.concatenate((samples[count:-count], overlap)).astype('<f4')
    # Lossy encoders have no preceding signal at the beginning of a standalone file.
    # A short, smooth zero boundary prevents that priming error becoming a wrap impulse.
    guard = min(round(CODEC_TAPER_SECONDS * RATE), len(output)//4)
    taper = (.5 - .5*np.cos(np.linspace(0, np.pi, guard)))[:, None]
    output[:guard] *= taper
    output[-guard:] *= taper[::-1]
    return output


def _pcm_args():
    return ['-f', 'f32le', '-ar', str(RATE), '-ac', '2', '-i', 'pipe:0']


def measure(samples, target=-18):
    result = _ffmpeg([*_pcm_args(), '-af', f'loudnorm=I={target}:TP=-2:LRA=11:print_format=json',
                     '-f', 'null', '-'], samples.astype('<f4').tobytes())
    matches = re.findall(r'\{\s*"input_i".*?\}', result.stderr.decode(errors='replace'), re.S)
    if not matches:
        raise ValueError('ffmpeg did not report loudness')
    values = json.loads(matches[-1])
    return {k: float(v) for k, v in values.items() if k != 'normalization_type'}


def limit_gain(samples, gain, ceiling, loop):
    context = np.tile(samples, (3, 1)) if loop else samples
    limit = 10 ** (ceiling / 20)
    filt = (f'aresample=192000,volume={gain}dB,'
            f'alimiter=limit={limit}:level=false:latency=true:attack=1:release=20')
    result = _ffmpeg([*_pcm_args(), '-af', filt, '-ar', str(RATE), '-f', 'f32le', 'pipe:1'],
                     context.astype('<f4').tobytes())
    pcm = np.frombuffer(result.stdout, dtype='<f4').reshape(-1, 2).copy()
    return pcm[len(samples):2*len(samples)] if loop else pcm[:len(samples)]


def normalize(samples, target, loop):
    # Repeated context avoids loudnorm's startup/lookahead envelope at the loop boundary.
    source = np.tile(samples, (3, 1)) if loop else samples
    measured = measure(source, target)
    if not all(math.isfinite(measured[k]) for k in ('input_i', 'input_tp', 'input_thresh', 'target_offset')):
        raise ValueError('silent or unmeasurable audio cannot be loudness normalized')
    filt = (f'loudnorm=I={target}:TP=-2:LRA=11:measured_I={measured["input_i"]}:'
            f'measured_TP={measured["input_tp"]}:measured_LRA={measured["input_lra"]}:'
            f'measured_thresh={measured["input_thresh"]}:offset={measured["target_offset"]}:linear=true')
    result = _ffmpeg([*_pcm_args(), '-af', filt, '-ar', str(RATE), '-f', 'f32le', 'pipe:1'],
                     source.astype('<f4').tobytes())
    output = np.frombuffer(result.stdout, dtype='<f4').reshape(-1, 2).copy()
    output = output[len(samples):2*len(samples)] if loop else output[:len(samples)]
    corrections = 0
    # A short transient can make loudnorm stop at the peak ceiling below the LUFS target.
    # Measure the actual playback, then apply gain with an oversampled lookahead limiter.
    # These are local DSP passes over the paid source, never provider retries.
    for _ in range(3):
        context = np.tile(output, (3, 1)) if loop else output
        stats = measure(context, target)
        if abs(stats['input_i'] - target) <= .2 and stats['input_tp'] <= -1.8:
            break
        gain = target - stats['input_i']
        output = limit_gain(output, gain, -2, loop)
        corrections += 1
    return output, corrections


def export(source: Path, out_dir: Path, *, target_lufs, loop, crossfade):
    samples = decode(source)
    source_seconds = len(samples) / RATE
    if loop:
        samples = make_loop(samples, crossfade)
    samples, corrections = normalize(samples, target_lufs, loop)
    duration = len(samples) / RATE
    files = {}
    for fmt, codec in (('ogg', ['-c:a', 'libopus', '-b:a', '160k']), ('m4a', ['-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart'])):
        path = out_dir / f'sound.{fmt}'
        encoded_samples, ceiling = samples, -2.0
        for attempt in range(3):
            _ffmpeg([*_pcm_args(), '-map_metadata', '-1', *codec, str(path)], encoded_samples.astype('<f4').tobytes())
            decoded = decode(path)
            if len(decoded) < len(samples) - 2:
                raise ValueError(f'{fmt} truncated the audio')
            playback = decoded[:len(samples)]  # AAC may retain a final partial codec frame.
            measured = measure(np.tile(playback, (3, 1)) if loop else playback, target_lufs)
            lufs, peak = measured['input_i'], measured['input_tp']
            if math.isfinite(lufs) and abs(lufs - target_lufs) <= 1 and peak <= -1:
                break
            if not math.isfinite(lufs) or attempt == 2:
                raise ValueError(f'{fmt} failed loudness/peak check ({lufs:g} LUFS, {peak:g} dBTP); cached source retained')
            # Lossy codec reconstruction can overshoot PCM peaks. Tighten the PCM limiter
            # by the measured overshoot plus headroom, then encode and measure again.
            ceiling -= max(0, peak + 1.5) if peak > -1 else 0
            encoded_samples = limit_gain(encoded_samples, target_lufs - lufs, ceiling, loop)
        files[fmt] = {'path': path.name, 'duration_seconds': len(decoded)/RATE,
                      'lufs': lufs, 'true_peak_dbfs': peak, 'encode_passes': attempt + 1,
                      'pcm_peak_limit_dbtp': ceiling,
                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        if loop:
            files[fmt]['seam_delta'] = float(np.max(np.abs(playback[-1] - playback[0])))
    return {'files': files, 'duration_seconds': duration, 'source_duration_seconds': source_seconds,
            'sample_rate': RATE, 'loop_start_seconds': 0, 'loop_end_seconds': duration,
            'crossfade_seconds': crossfade if loop else 0,
            'codec_taper_seconds': CODEC_TAPER_SECONDS if loop else 0,
            'normalization': {'method': 'loudnorm-two-pass', 'peak_limit_dbtp': -2, 'gain_limiter_passes': corrections}}
