# SPDX-License-Identifier: Apache-2.0
from concurrent.futures import ThreadPoolExecutor
import io
import http.client
import json
import threading
import urllib.error
import wave

import numpy as np
import pytest

from sprite_gen.audio import cache, provider, processing, sound
from sprite_gen.gen import retry


@pytest.fixture(autouse=True)
def fake_credential(monkeypatch):
    monkeypatch.setenv('ELEVENLABS_API_KEY', 'synthetic-key')


def wav_bytes(seconds=3):
    t = np.arange(round(seconds * 48000)) / 48000
    x = .08 * np.sin(2 * np.pi * 321.3 * t)
    x[100:110] = .95  # sparse peak: gain alone would clip
    data = io.BytesIO()
    with wave.open(data, 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(48000)
        f.writeframes((x * 32767).astype('<i2').tobytes())
    return data.getvalue()


def test_provider_contract_and_safe_retry(monkeypatch, capsys):
    requests = []
    class Answer:
        status = 200
        headers = {'character-cost': '120', 'request-id': 'req-synthetic'}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return b'audio'
    def send(req, **kw):
        requests.append(req)
        if len(requests) == 1:
            raise urllib.error.HTTPError(req.full_url, 429, 'busy', {'retry-after':'0'},
                io.BytesIO(b'{"detail":{"status":"system_busy","message":"secret-key"}}'))
        return Answer()
    monkeypatch.setenv('ELEVENLABS_API_KEY', 'secret-key')
    monkeypatch.setattr(provider, 'urlopen', send)
    monkeypatch.setattr(retry, '_sleep', lambda _: None)
    req = provider.request('sfx', 'Impact', 3, True)
    audio, meta = provider.generate(req)
    assert audio == b'audio' and len(requests) == 2
    assert requests[0].full_url.endswith('/sound-generation?output_format=mp3_44100_128')
    assert json.loads(requests[0].data) == {'text':'Impact', 'duration_seconds':3.0,
        'loop':True, 'model_id':'eleven_text_to_sound_v2', 'prompt_influence':.3}
    assert meta['character_cost'] == 120
    assert 'secret-key' not in capsys.readouterr().err
    bgm = provider.request('bgm', 'Adventure', 8, True)
    assert bgm['body'] == {'prompt':'Adventure', 'music_length_ms':8000,
                          'model_id':'music_v1', 'force_instrumental':True}


@pytest.mark.parametrize('status,body', [(500, {}), (429, {'detail':{'status':'quota_exceeded'}}),
                                         (429, {'detail':{'status':'unknown'}}), (422, {})])
def test_unproven_refusals_are_not_retried(status, body):
    assert retry.again_after('elevenlabs', 'POST', status, body, {}, 1) is None


def test_cache_serializes_paid_requests_and_survives_uncertain_results(tmp_path):
    db = tmp_path / 'cache.sqlite3'
    entered, release = threading.Event(), threading.Event()
    def make():
        entered.set()
        assert release.wait(5)
        return b'audio', {'request_id':'synthetic'}
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(cache.obtain, db, {'a':1}, make)
        assert entered.wait(5)
        with pytest.raises(ValueError, match='unresolved'):
            cache.obtain(db, {'a':1}, make)
        release.set()
        assert first.result()[0] == b'audio'
    assert cache.obtain(db, {'a':1}, lambda: pytest.fail('charged twice'))[2] is True
    def broken(): raise OSError('connection closed')
    with pytest.raises(OSError): cache.obtain(db, {'a':2}, broken)
    with pytest.raises(ValueError, match='unresolved'):
        cache.obtain(db, {'a':2}, lambda: pytest.fail('ambiguous POST resent'))


def test_loop_overlaps_ends_without_discontinuity():
    t = np.arange(48000 * 4) / 48000
    x = np.stack([.1*np.sin(2*np.pi*317.25*t) + .2*t/4]*2, axis=1)
    y = processing.make_loop(x, .2)
    assert len(y) == len(x) - 9600
    assert np.max(np.abs(y[-1] - y[0])) < .01
    assert np.max(np.abs(np.diff(y, axis=0))) < .01


def test_actual_exports_are_normalized_and_reused(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(provider, 'generate', lambda req: (called.append(req) or wav_bytes(), {}))
    kw = dict(kind='sfx', prompt='synthetic', duration=3, event='combat.hit',
              cache_dir=tmp_path/'cache', out_dir=tmp_path/'out')
    first = sound.generate(**kw)
    second = sound.generate(**{**kw, 'out_dir':tmp_path/'out2', 'event':'combat.other'})
    assert len(called) == 1 and second['cache_hit']
    assert first['event'] == 'combat.hit'
    for fmt, metadata in first['files'].items():
        path = tmp_path/'out'/metadata['path']
        assert path.suffix == '.' + fmt and path.stat().st_size > 100
        assert abs(metadata['lufs'] - first['target_lufs']) <= 1
        assert metadata['true_peak_dbfs'] <= -1
        assert abs(metadata['duration_seconds']-3) < .05


def test_processing_failure_reuses_paid_audio(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(provider, 'generate', lambda req: (called.append(req) or wav_bytes(), {}))
    original = processing.export
    monkeypatch.setattr(processing, 'export', lambda *a, **kw: (_ for _ in ()).throw(ValueError('processing failed')))
    kw = dict(kind='sfx', prompt='synthetic', duration=3, event='hit', cache_dir=tmp_path/'cache', out_dir=tmp_path/'out')
    with pytest.raises(ValueError, match='processing failed'): sound.generate(**kw)
    assert not kw['out_dir'].exists()
    monkeypatch.setattr(processing, 'export', original)
    assert sound.generate(**kw)['cache_hit'] and len(called) == 1


@pytest.mark.parametrize('kind,duration,loop', [('sfx', .5, False), ('bgm', 3, True)])
def test_short_effect_and_loop_encoded_playback(tmp_path, monkeypatch, kind, duration, loop):
    monkeypatch.setattr(provider, 'generate', lambda req: (wav_bytes(duration), {}))
    result = sound.generate(kind=kind, prompt='synthetic', duration=duration, event='sample',
                            cache_dir=tmp_path/'cache', out_dir=tmp_path/'out')
    assert result['loop'] is loop
    assert abs(result['duration_seconds'] - (duration-result['crossfade_seconds'])) < .001
    if loop:
        for file in result['files'].values():
            decoded = processing.decode(tmp_path/'out'/file['path'])[:round(result['loop_end_seconds']*48000)]
            assert np.max(np.abs(decoded[-1]-decoded[0])) < .025


def test_preflight_does_not_claim_and_cached_source_works_without_key(tmp_path, monkeypatch):
    db = tmp_path/'cache.sqlite3'
    def no_key(): raise ValueError('missing key')
    with pytest.raises(ValueError, match='missing key'):
        cache.obtain(db, {'request':1}, lambda: pytest.fail('must preflight'), prepare=no_key)
    assert cache.obtain(db, {'request':1}, lambda: (b'audio', {}))[2] is False
    assert cache.obtain(db, {'request':1}, lambda: pytest.fail('cache must suffice'), prepare=no_key)[2] is True


def test_error_body_and_transport_cannot_disclose_key(monkeypatch, capsys):
    key = 'secret-api-key-should-never-appear'
    monkeypatch.setenv('ELEVENLABS_API_KEY', key)
    def refuse(req, **kwargs):
        raise urllib.error.HTTPError(req.full_url, 422, key, {}, io.BytesIO(json.dumps({'detail':{'message':key,'status':key}}).encode()))
    monkeypatch.setattr(provider, 'urlopen', refuse)
    with pytest.raises(ValueError) as err: provider.generate(provider.request('sfx','impact',1,False))
    assert key not in str(err.value) + capsys.readouterr().err
    def broken(*a, **kw): raise OSError(key)
    monkeypatch.setattr(provider, 'urlopen', broken)
    with pytest.raises(ValueError) as err: provider.generate(provider.request('sfx','impact',1,False))
    assert key not in str(err.value) + capsys.readouterr().err


def test_broken_refusal_body_is_not_retried_or_printed(monkeypatch, capsys):
    calls = []
    class Broken(io.BytesIO):
        def read(self, *args): raise http.client.IncompleteRead(b'secret-key', 100)
    def send(req, **kw):
        calls.append(req)
        raise urllib.error.HTTPError(req.full_url, 429, 'busy', {}, Broken())
    monkeypatch.setattr(provider, 'urlopen', send)
    with pytest.raises(ValueError, match='unresolved') as err:
        provider.generate(provider.request('sfx', 'impact', 1, False))
    assert len(calls) == 1 and 'secret-key' not in str(err.value) + capsys.readouterr().err


def test_cli_and_python_entry_publish_the_same_bundle(tmp_path, monkeypatch, capsys):
    from sprite_gen.cli import main
    monkeypatch.setattr(provider, 'generate', lambda req: (wav_bytes(1), {}))
    args = ['sound', '--kind', 'sfx', '--prompt', 'synthetic', '--duration', '1', '--event', 'hit',
            '--out-dir', str(tmp_path/'out'), '--cache-dir', str(tmp_path/'cache')]
    assert main(args) == 0
    monkeypatch.delenv('ELEVENLABS_API_KEY')
    assert sound.run(kind='sfx',prompt='synthetic',duration=1,event='hit',out_dir=str(tmp_path/'out'),
                     cache_dir=tmp_path/'cache') == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert lines[1]['cache_hit'] is True
    assert lines[0]['manifest'] == lines[1]['manifest']


def test_codec_reconstruction_keeps_a_broadband_transient_below_peak_ceiling(tmp_path, monkeypatch):
    t = np.arange(24000)/48000
    x = .8*np.sign(np.sin(2*np.pi*2200*t))*np.exp(-t*32)
    stream=io.BytesIO()
    with wave.open(stream,'wb') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(48000)
        f.writeframes((x*32767).astype('<i2').tobytes())
    monkeypatch.setattr(provider,'generate',lambda _: (stream.getvalue(), {}))
    result=sound.generate(kind='sfx',prompt='synthetic transient',duration=.5,event='hit',
                          out_dir=tmp_path/'out',cache_dir=tmp_path/'cache')
    for file in result['files'].values():
        assert file['true_peak_dbfs'] <= -1
        assert abs(file['lufs']+18) <= 1


def test_loop_codec_boundary_at_nonzero_stereo_phase(tmp_path, monkeypatch):
    t = np.arange(48000*3)/48000
    x = np.stack([.1*np.sin(2*np.pi*121*t+.5), .4*np.cos(2*np.pi*73*t)], axis=1)
    stream = io.BytesIO()
    with wave.open(stream, 'wb') as f:
        f.setnchannels(2); f.setsampwidth(2); f.setframerate(48000)
        f.writeframes((x*32767).astype('<i2').tobytes())
    monkeypatch.setattr(provider, 'generate', lambda _: (stream.getvalue(), {}))
    result = sound.generate(kind='bgm', prompt='synthetic stereo loop', duration=3, event='music',
                            out_dir=tmp_path/'out', cache_dir=tmp_path/'cache')
    for file in result['files'].values():
        decoded = processing.decode(tmp_path/'out'/file['path'])[:round(result['duration_seconds']*48000)]
        # Codec priming must not turn a large nonzero start into a wrap impulse.
        assert np.max(np.abs(decoded[-1]-decoded[0])) < .003
        assert np.max(np.abs(decoded[[0,-1]])) < .003
