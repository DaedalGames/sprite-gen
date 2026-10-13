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
    return np.concatenate((samples[count:-count], overlap)).astype('<f4')


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
    return output[len(samples):2*len(samples)] if loop else output[:len(samples)]


def export(source: Path, out_dir: Path, *, target_lufs, loop, crossfade):
    samples = decode(source)
    source_seconds = len(samples) / RATE
    if loop:
        samples = make_loop(samples, crossfade)
    samples = normalize(samples, target_lufs, loop)
    duration = len(samples) / RATE
    files = {}
    for fmt, codec in (('ogg', ['-c:a', 'libopus', '-b:a', '160k']), ('m4a', ['-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart'])):
        path = out_dir / f'sound.{fmt}'
        _ffmpeg([*_pcm_args(), '-map_metadata', '-1', *codec, str(path)], samples.astype('<f4').tobytes())
        decoded = decode(path)
        if len(decoded) < len(samples) - 2:
            raise ValueError(f'{fmt} truncated the audio')
        playback = decoded[:len(samples)]  # AAC may retain a final partial codec frame.
        measured = measure(np.tile(playback, (3, 1)) if loop else playback, target_lufs)
        lufs, peak = measured['input_i'], measured['input_tp']
        if not math.isfinite(lufs) or abs(lufs - target_lufs) > 1 or peak > -1:
            raise ValueError(f'{fmt} failed loudness/peak check ({lufs:g} LUFS, {peak:g} dBTP); cached source retained')
        files[fmt] = {'path': path.name, 'duration_seconds': len(decoded)/RATE,
                      'lufs': lufs, 'true_peak_dbfs': peak,
                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        if loop:
            files[fmt]['seam_delta'] = float(np.max(np.abs(playback[-1] - playback[0])))
    return {'files': files, 'duration_seconds': duration, 'source_duration_seconds': source_seconds,
            'sample_rate': RATE, 'loop_start_seconds': 0, 'loop_end_seconds': duration,
            'crossfade_seconds': crossfade if loop else 0}
