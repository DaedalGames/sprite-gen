# SPDX-License-Identifier: Apache-2.0
"""EBU R128 normalization and unspliced browser audio encodes."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np

from sprite_gen.audio import playback

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
    frames = playback.validate({'version': playback.VERSION, 'sample_rate': RATE,
                                'start_frame': 0, 'end_frame': len(samples),
                                'overlap_frames': round(crossfade*RATE) if loop else 0}, loop)
    samples, corrections = normalize(samples, target_lufs, loop)
    files = {}
    for fmt, codec in (('ogg', ['-c:a', 'libopus', '-b:a', '160k']), ('m4a', ['-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart'])):
        path = out_dir / f'sound.{fmt}'
        encoded_samples, ceiling = samples, -2.0
        for attempt in range(3):
            _ffmpeg([*_pcm_args(), '-map_metadata', '-1', *codec, str(path)], encoded_samples.astype('<f4').tobytes())
            decoded = decode(path)
            if len(decoded) < len(samples) - 2:
                raise ValueError(f'{fmt} truncated the audio')
            interval = decoded[:len(samples)]  # AAC may retain a final partial codec frame.
            measured = measure(np.tile(interval, (3, 1)) if loop else interval, target_lufs)
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
    return {'files': files, 'duration_seconds': playback.duration(frames), 'playback': frames,
            'normalization': {'method': 'loudnorm-two-pass', 'peak_limit_dbtp': -2, 'gain_limiter_passes': corrections}}
