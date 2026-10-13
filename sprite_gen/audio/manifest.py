# SPDX-License-Identifier: Apache-2.0
"""Validated playback metadata and file digests for a sound bundle."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

from sprite_gen.audio import playback


def event_name(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}', value):
        raise ValueError('event requires 1 to 128 letters, digits, dots, colons, dashes or underscores')
    return value


def _number(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'invalid audio {name}')
    return value


def load(path, max_bytes):
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get('kind') != 'sprite-gen-sound' or value.get('version') != playback.VERSION:
        raise ValueError('audio source must be a sprite-gen-sound version 2 manifest')
    if value.get('audio_kind') not in ('sfx', 'bgm') or type(value.get('loop')) is not bool:
        raise ValueError('invalid audio kind or loop flag')
    frames = playback.validate(value.get('playback'), value['loop'])
    duration = playback.duration(frames)
    if value.get('duration_seconds') != duration:
        raise ValueError('audio duration must be derived from playback frames')
    result = {'event': event_name(value.get('event')), 'audio_kind': value['audio_kind'], 'loop': value['loop'],
              'duration_seconds': duration, 'target_lufs': _number(value.get('target_lufs'), 'target LUFS', -36, -9),
              'playback': frames, 'files': {}}
    files = value.get('files')
    if not isinstance(files, dict) or set(files) != {'ogg', 'm4a'}:
        raise ValueError('audio source requires ogg and m4a files')
    payloads, sources, size = {}, {path}, 0
    for fmt in ('ogg', 'm4a'):
        file = files[fmt]
        name = file.get('path') if isinstance(file, dict) else None
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9_.-]*\.'+fmt, name):
            raise ValueError('audio file must be a local filename of the declared format')
        source = (path.parent/name).resolve()
        if source.parent != path.parent:
            raise ValueError('audio source must stay within its bundle')
        size += source.stat().st_size
        if size > max_bytes:
            raise ValueError(f'audio exceeds {max_bytes} bytes')
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != file.get('sha256'):
            raise ValueError('audio file digest does not match its manifest')
        record = {'path': name, 'sha256': file['sha256'],
                  'duration_seconds': _number(file.get('duration_seconds'), 'file duration', frames['end_frame']/frames['sample_rate']-2/frames['sample_rate'], 602),
                  'lufs': _number(file.get('lufs'), 'LUFS', -70, 0),
                  'true_peak_dbfs': _number(file.get('true_peak_dbfs'), 'true peak', -100, -1)}
        if abs(record['lufs'] - result['target_lufs']) > 1:
            raise ValueError('audio loudness does not match target')
        result['files'][fmt] = record
        payloads[fmt] = data
        sources.add(source)
    return result, payloads, sources
