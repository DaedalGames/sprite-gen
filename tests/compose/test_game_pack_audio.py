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
    manifest = {'kind':'sprite-gen-sound','version':1,'event':'combat.hit', 'audio_kind':'sfx',
                'loop':False,'duration_seconds':1,'target_lufs':-18,'files':files,
                'loop_start_seconds':0, 'loop_end_seconds':1}
    (source/'sound.json').write_text(json.dumps(manifest))
    path = tmp_path/'request.json'
    path.write_text(json.dumps(request))
    return path, request, manifest


def test_optional_audio_preserves_image_only_bytes_and_packs_events(tmp_path):
    path, request, _ = inputs(tmp_path)
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
    assert pack(path, tmp_path/'with') == result
    total = sum(p.stat().st_size for p in (tmp_path/'with').iterdir())
    with pytest.raises(ValueError, match='bytes'):
        pack(path, tmp_path/'small', max_bytes=total-1)
    assert not (tmp_path/'small').exists()


@pytest.mark.parametrize('defect', ['duplicate', 'traversal', 'digest', 'nonfinite', 'badloop'])
def test_broken_audio_cannot_publish_a_partial_pack(tmp_path, defect):
    path, request, manifest = inputs(tmp_path)
    request['audio'] = [{'id':'hit','source':'audio/sound.json'}]
    if defect == 'duplicate': request['audio'] *= 2
    if defect == 'traversal': manifest['files']['ogg']['path'] = '../image.png'
    if defect == 'digest': manifest['files']['ogg']['sha256'] = '0'*64
    if defect == 'nonfinite': manifest['target_lufs'] = float('nan')
    if defect == 'badloop': manifest['loop'] = 'false'
    path.write_text(json.dumps(request))
    (tmp_path/'audio'/'sound.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError): pack(path, tmp_path/'bad')
    assert not (tmp_path/'bad').exists()
