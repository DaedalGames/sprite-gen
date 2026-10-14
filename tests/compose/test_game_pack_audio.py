# SPDX-License-Identifier: Apache-2.0
import hashlib
import json

import pytest
from PIL import Image

from sprite_gen.compose.game_pack import pack


def inputs(tmp_path):
    Image.new('RGBA', (8, 8), 'red').save(tmp_path/'image.png')
    request = {'version':1, 'assets':[{'id':'hero', 'source':'image.png', 'display':{'width':8},
                                     'events':[{'frame':0, 'name':'combat.hit'}]}]}
    source = tmp_path/'audio'
    source.mkdir()
    files = {}
    for fmt in ('ogg', 'm4a'):
        data = b'synthetic-'+fmt.encode()
        (source/f'sound.{fmt}').write_bytes(data)
        files[fmt] = {'path':f'sound.{fmt}', 'duration_seconds':1, 'lufs':-18,
                      'true_peak_dbfs':-2, 'sha256':hashlib.sha256(data).hexdigest()}
    manifest = {'kind':'sprite-gen-sound','version':2,'event':'combat.hit', 'audio_kind':'sfx',
                'loop':False,'duration_seconds':1,'target_lufs':-18,'files':files,
                'playback': {'version':2, 'sample_rate':48000, 'start_frame':0,
                             'end_frame':48000, 'overlap_frames':0}}
    (source/'sound.json').write_text(json.dumps(manifest))
    path = tmp_path/'request.json'
    path.write_text(json.dumps(request))
    return path, request, manifest


def test_optional_audio_preserves_image_only_bytes_and_packs_events(tmp_path):
    path, request, original = inputs(tmp_path)
    pack(path, tmp_path/'before')
    request['audio'] = []
    path.write_text(json.dumps(request))
    pack(path, tmp_path/'empty')
    assert {p.name:p.read_bytes() for p in (tmp_path/'before').iterdir()} == {
        p.name:p.read_bytes() for p in (tmp_path/'empty').iterdir()}
    request['audio'] = [{'id':'hit','source':'audio/sound.json'}]
    path.write_text(json.dumps(request))
    result = pack(path, tmp_path/'with')
    audio = result['audio']['hit']
    assert audio['event'] == result['animations']['hero']['events'][0]['name']
    assert (tmp_path/'with'/audio['files']['ogg']['path']).read_bytes() == b'synthetic-ogg'
    assert audio['files']['m4a']['path'] == 'audio-hit.m4a'
    assert audio['playback'] == original['playback']
    helper = audio['player']
    assert helper['path'] == 'sprite-gen-audio-v2.mjs'
    assert helper['export'] == 'createAudioSource'
    assert hashlib.sha256((tmp_path/'with'/helper['path']).read_bytes()).hexdigest() == helper['sha256']
    assert pack(path, tmp_path/'with') == result
    total = sum(p.stat().st_size for p in (tmp_path/'with').iterdir())
    with pytest.raises(ValueError, match='bytes'):
        pack(path, tmp_path/'small', max_bytes=total-1)
    assert not (tmp_path/'small').exists()


@pytest.mark.parametrize('defect', ['duplicate', 'traversal', 'digest', 'nonfinite', 'badloop', 'old', 'fraction', 'overlap'])
def test_broken_audio_cannot_publish_a_partial_pack(tmp_path, defect):
    path, request, manifest = inputs(tmp_path)
    request['audio'] = [{'id':'hit','source':'audio/sound.json'}]
    if defect == 'duplicate': request['audio'] *= 2
    if defect == 'traversal': manifest['files']['ogg']['path'] = '../image.png'
    if defect == 'digest': manifest['files']['ogg']['sha256'] = '0'*64
    if defect == 'nonfinite': manifest['target_lufs'] = float('nan')
    if defect == 'badloop': manifest['loop'] = 'false'
    if defect == 'old': manifest['version'] = 1
    if defect == 'fraction': manifest['playback']['end_frame'] = 48000.5
    if defect == 'overlap': manifest['playback']['overlap_frames'] = 24000
    path.write_text(json.dumps(request))
    (tmp_path/'audio'/'sound.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError): pack(path, tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


# A version 1 pack with sound, packed by 2.44.0 (before request version 2): its file names, the
# manifest's exact bytes and the page's decoded pixels (encoded PNG/WebP bytes depend on the zlib and
# libwebp build). The sound files and the helper are compared byte for byte below.
V1_AUDIO_FILES = ['atlas-000.png', 'atlas-000.webp', 'audio-hit.m4a', 'audio-hit.ogg', 'game-pack.json',
                  'sprite-gen-audio-v2.mjs']
V1_AUDIO_MANIFEST_SHA256 = 'b79d31a4f38982f2c4a97b2aa5ddb4e77b545acee29210dac6a0e6649f9f5b67'
V1_AUDIO_PAGE_PIXELS_SHA256 = '834f7685ed723160d26784bb8ba1153e5f8ae56f5ddc8f51ebb0c54b1a63e067'


def with_sound(tmp_path):
    path, request, _ = inputs(tmp_path)
    request['audio'] = [{'id': 'hit', 'source': 'audio/sound.json'}]
    path.write_text(json.dumps(request))
    return path, request


def test_version_1_pack_with_sound_keeps_its_files_and_bytes(tmp_path):
    path, _ = with_sound(tmp_path)
    out = tmp_path/'out'
    result = pack(path, out)
    assert sorted(p.name for p in out.iterdir()) == V1_AUDIO_FILES
    manifest = (out/'game-pack.json').read_bytes()
    assert hashlib.sha256(manifest).hexdigest() == V1_AUDIO_MANIFEST_SHA256
    assert json.loads(manifest)['version'] == 1
    with Image.open(out/'atlas-000.png') as page:
        assert hashlib.sha256(page.convert('RGBA').tobytes()).hexdigest() == V1_AUDIO_PAGE_PIXELS_SHA256
    for fmt in ('ogg', 'm4a'):
        assert (out/f'audio-hit.{fmt}').read_bytes() == f'synthetic-{fmt}'.encode()
    assert hashlib.sha256((out/'sprite-gen-audio-v2.mjs').read_bytes()).hexdigest() == result['audio']['hit']['player']['sha256']


def test_version_2_pack_with_sound_adds_only_the_tile_files(tmp_path):
    path, request = with_sound(tmp_path)
    pack(path, tmp_path/'v1')
    Image.new('RGBA', (8, 8), 'blue').save(tmp_path/'ground.png')
    request.update(version=2, tiles=[{'id': 'ground', 'source': 'ground.png', 'display': {'width': 16}}])
    path.write_text(json.dumps(request))
    result = pack(path, tmp_path/'v2')
    v1 = {p.name: p.read_bytes() for p in (tmp_path/'v1').iterdir()}
    v2 = {p.name: p.read_bytes() for p in (tmp_path/'v2').iterdir()}
    assert sorted(v2) == sorted([*v1, 'tile-ground.png', 'tile-ground.webp'])
    for name in ('audio-hit.ogg', 'audio-hit.m4a', 'sprite-gen-audio-v2.mjs', 'atlas-000.png', 'atlas-000.webp'):
        assert v2[name] == v1[name], name
    assert result['version'] == 2 and list(result['audio']) == ['hit']
    assert result['tiles']['ground']['image'] == 'tile-ground.png'
