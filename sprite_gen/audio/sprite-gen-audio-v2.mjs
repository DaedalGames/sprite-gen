// SPDX-License-Identifier: Apache-2.0
// Frame contract v2. No processing is applied before decoding or on each repeat.
const prepared = new WeakMap();
const cached = new WeakMap();

export function prepareAudioBuffer(decoded, playback) {
  const { version, sample_rate: rate, start_frame: start, end_frame: end,
    overlap_frames: overlap } = playback;
  if (version !== 2 || ![rate, start, end, overlap].every(Number.isSafeInteger) ||
      rate < 8000 || rate > 192000 || start < 0 || end <= start || end > 601 * rate ||
      overlap < 0 || (overlap !== 0 && (overlap < 2 || overlap * 2 >= end - start))) {
    throw new RangeError('Invalid audio playback frame contract');
  }
  const key = [version, rate, start, end, overlap].join(':');
  if (prepared.has(decoded)) {
    if (prepared.get(decoded) !== key) throw new RangeError('AudioBuffer is already prepared');
    return decoded;
  }
  if (cached.get(decoded)?.has(key)) return cached.get(decoded).get(key);
  const scale = decoded.sampleRate / rate;
  const first = Math.round(start * scale);
  const last = first + Math.round((end - start) * scale);
  // Round the declared final duration once. Independent rounding of N and M can
  // change N-M by a frame; derive the overlap from the mapped interval instead.
  const length = Math.round((end - start - overlap) * scale);
  const count = last - first - length;
  if (last > decoded.length || length < 1 || count < 0 ||
      (overlap > 0 && (count < 2 || count * 2 >= last - first))) {
    throw new RangeError('Decoded audio does not contain the playback interval');
  }
  const output = new AudioBuffer({ numberOfChannels: decoded.numberOfChannels,
    length, sampleRate: decoded.sampleRate });
  for (let channel = 0; channel < decoded.numberOfChannels; channel++) {
    const source = decoded.getChannelData(channel);
    const target = output.getChannelData(channel);
    if (!overlap) {
      target.set(source.subarray(first, first + length));
      continue;
    }
    const middle = last - first - 2 * count;
    target.set(source.subarray(first + count, last - count));
    for (let i = 0; i < count; i++) {
      const headWeight = 0.5 - 0.5 * Math.cos(Math.PI * i / (count - 1));
      target[middle + i] = source[last - count + i] * (1 - headWeight) +
        source[first + i] * headWeight;
    }
  }
  prepared.set(output, key);
  if (!cached.has(decoded)) cached.set(decoded, new Map());
  cached.get(decoded).set(key, output);
  return output;
}

export function createAudioSource(context, decoded, audio) {
  if (typeof audio.loop !== 'boolean' || audio.loop !== (audio.playback.overlap_frames > 0)) {
    throw new RangeError('Audio loop flag does not match playback overlap');
  }
  const source = context.createBufferSource();
  source.buffer = prepareAudioBuffer(decoded, audio.playback);
  source.loop = audio.loop;
  source.loopStart = 0;
  source.loopEnd = source.buffer.duration;
  return source;
}
