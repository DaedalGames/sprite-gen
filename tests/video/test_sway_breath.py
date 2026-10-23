"""`video-sway --breathe <breathe.json>`: the Sway layer on the breath, not the body's outline.

A breathing idle's head stays where it is and its body moves a pixel or two, so the damped mass on the outline answers
whole-pixel steps, not the breath. On the breath the part follows the chest's rise late: the chest rises lift x w, the
part as far late by --lag of a breath (the time warp of the Breathe layer's shoulders), and it lies the difference behind.
The first cell, every rest and the loop's close are 0, so a cell there is the breathed cell byte for byte. On a
synthetic bust breathed by `sprite-gen breathe` (smooth, two breaths in 48 cells).
"""
import hashlib
import json
import math

import numpy as np
import pytest
from PIL import Image
from sprite_gen import cli
from sprite_gen.effects import breathe
from sprite_gen.video import sway, sway_inspect

from tests.effects.test_breathe_smooth import _bust

N, BREATHS, DEPTH = 48, 2, 0.025
INHALE = breathe.SMOOTH_SHAPE['inhale']  # the default in-breath share
CHEST = (48.0, 80.0, 30.0, 20.0)  # over the bust's chest, in frame pixels
FILES = [f'frame-{k:03d}.png' for k in range(N)]


@pytest.fixture(scope='module')
def breathed(tmp_path_factory):
    """A still bust, 48 frames, and those frames breathed: `plain/`, `breathed/` (with breathe.json)."""
    root = tmp_path_factory.mktemp('breath')
    plain = root/'plain'
    plain.mkdir()
    for name in FILES:
        _bust().save(plain/name)
    assert cli.main(['breathe', str(plain), str(root/'breathed'), '--mode', 'smooth', '--depth', str(DEPTH),
                     '--breaths', str(BREATHS)]) == 0
    return root


def rgba(path):
    with Image.open(path) as im:
        return np.asarray(im.convert('RGBA'))


def lift(record):
    """The chest's rise at the top of an in-breath, px: depth times the body below the neck, as the record says it."""
    anatomy = record['resolved']['anatomy']
    return record['config']['depth']*(anatomy['height']-anatomy['basis_row'])


def answer(breathed, **kw):
    cells = [Image.fromarray(rgba(breathed/'breathed'/name)) for name in FILES]
    return sway.solve_cells(sway.read_ask([CHEST], breathe=breathed/'breathed'/'breathe.json', **kw), cells, None)


def region(values=CHEST):
    return ','.join(f'{v:g}' for v in values)


def test_the_first_cell_every_rest_and_the_close_are_the_breathed_cells(breathed, tmp_path):
    out = tmp_path/'swayed'
    src = breathed/'breathed'
    assert cli.main(['video-sway', '--frames', str(src), '--out', str(out), '--breathe', str(src/'breathe.json'),
                     '--region', region(), '--gain', '3']) == 0
    record = json.loads((out/'sway.json').read_text())
    rests = [k for k in range(N) if (k*BREATHS) % N == 0]
    assert rests == [0, N//BREATHS]
    for k in rests:
        assert record['dy_px'][k] == record['dx_px'][k] == 0
        assert np.array_equal(rgba(out/FILES[k]), rgba(src/FILES[k])), k
    assert sum(not np.array_equal(rgba(out/name), rgba(src/name)) for name in FILES) > N//2
    dy = np.asarray(record['dy_px'])
    assert abs(dy[0]-dy[-1]) <= np.abs(np.diff(dy)).max() / 4, 'the loop does not close at rest'
    report = json.loads((src/'breathe.json').read_text())
    assert record['drive'] == {'kind': 'breath', 'breathe_sha256': hashlib.sha256((src/'breathe.json').read_bytes()).hexdigest(),
                               'lag': sway.LAG_DEFAULT, 'lift_px': round(lift(report), 4), 'heading_deg': 0.0, 'inhale': INHALE,
                               'breaths': BREATHS}
    assert 'freq_hz' not in record and record['frame_ms'] is None


def late_peak(lag, inhale=INHALE):
    """Where in a breath the part peaks: the phase t whose warped time t - lag sin(pi t) is the chest's peak."""
    t = inhale
    for _ in range(60):
        t = inhale + lag*math.sin(math.pi*t)
    return t


@pytest.mark.parametrize('lag', [0.1, 0.15, 0.2])
def test_the_part_peaks_late_by_the_lag_and_reaches_gain_times_the_lift(breathed, lag):
    report = json.loads((breathed/'breathed'/'breathe.json').read_text())
    got = answer(breathed, lag=lag, gain=1.0)
    w = np.asarray([breathe.smooth_wave(p, INHALE) for p in report['phases']])
    rise = lift(report)
    part = w - got.dy/rise  # the part's own rise, as a share of the chest's
    per = N//BREATHS
    assert int(np.argmax(w[:per])) == round(INHALE*per)
    assert int(np.argmax(part[:per])) == round(late_peak(lag)*per)
    late = np.asarray([breathe.smooth_shoulder_wave(p, lag, INHALE) for p in report['phases']])
    assert np.allclose(part, late, atol=1e-12)
    for gain in (1.0, 2.5):
        assert answer(breathed, lag=lag, gain=gain).reach == pytest.approx(gain*rise*float(np.max(np.abs(w-late))), rel=1e-12)
    assert np.all(got.dx == 0) and got.dy[0] == 0


def wave(t, inhale):
    """The breath as the Breathe layer defines it, written out: 0 at rest, 1 at the in-breath's share of the breath, a
    cosine half up and a cosine half down."""
    t %= 1.0
    g = 0.5*t/inhale if t < inhale else 0.5 + 0.5*(t-inhale)/(1-inhale)
    return 0.5 - 0.5*math.cos(2*math.pi*g)


@pytest.mark.parametrize('beat', [['--breaths', str(BREATHS)], ['--breath-peaks', '10,34']], ids=['breaths', 'peaks'])
def test_the_part_follows_the_in_breath_share_the_record_says(breathed, tmp_path, beat):
    """A breath whose in-breath takes 0.42 of it, as a breath made elsewhere may: `sprite-gen breathe --inhale 0.42`
    writes the share into its record, and the sway on that record follows that breath — gain x lift x (w - w late) on
    the 0.42 curve, the default's curve a visible way off."""
    rec, out = tmp_path/'breathed', tmp_path/'swayed'
    assert cli.main(['breathe', str(breathed/'plain'), str(rec), '--mode', 'smooth', '--depth', str(DEPTH),
                     '--inhale', '0.42', *beat]) == 0
    assert cli.main(['video-sway', '--frames', str(rec), '--out', str(out), '--breathe', str(rec/'breathe.json'),
                     '--region', region(), '--lag', '0.15', '--gain', '3']) == 0
    report = json.loads((rec/'breathe.json').read_text())
    record = json.loads((out/'sway.json').read_text())
    assert report['config']['inhale'] == record['drive']['inhale'] == 0.42

    def offset(inhale):
        return np.asarray([3*lift(report)*(wave(p, inhale) - wave(p - 0.15*math.sin(math.pi*p), inhale)) for p in report['phases']])
    assert np.allclose(record['dy_px'], offset(0.42), rtol=0, atol=0.005+1e-9)
    assert np.abs(offset(0.42) - offset(INHALE)).max() > 0.1


def test_a_record_written_before_the_share_was_there_is_the_default_breath(breathed, tmp_path):
    """A record that does not say its in-breath share, as none did before the key was there, says the default, as a
    sidecar leaving it out does: the part follows the same breath to the bit."""
    src = breathed/'breathed'
    report = json.loads((src/'breathe.json').read_text())
    assert report['config']['inhale'] == INHALE
    unsaid = tmp_path/'unsaid.json'
    unsaid.write_text(json.dumps(report | {'config': {k: v for k, v in report['config'].items() if k != 'inhale'}}))
    cells = [Image.fromarray(rgba(src/name)) for name in FILES]
    a, b = (sway.solve_cells(sway.read_ask([CHEST], breathe=path), cells, None) for path in (src/'breathe.json', unsaid))
    assert np.array_equal(a.dy, b.dy) and np.array_equal(a.dx, b.dx) and np.abs(a.dy).max() > 0


# The cells the default breath swayed into before a record could say its in-breath share (e18c905): the 48 cells of
# `breathed`, swayed at gain 3, their RGBA bytes in name order.
SWAY_GOLDEN = '3ca59f0ac832805ba2045a6299bb9cf66c7fe5a971fc4e041e9e54a41dd1d03a'


def test_the_default_breath_sways_the_bytes_it_swayed_before(breathed, tmp_path):
    src, out = breathed/'breathed', tmp_path/'swayed'
    assert cli.main(['video-sway', '--frames', str(src), '--out', str(out), '--breathe', str(src/'breathe.json'),
                     '--region', region(), '--gain', '3']) == 0
    assert hashlib.sha256(b''.join(rgba(out/name).tobytes() for name in FILES)).hexdigest() == SWAY_GOLDEN


def test_a_heading_moves_the_part_against_the_way_the_chest_rises(breathed, tmp_path):
    plain = breathed/'plain'
    assert cli.main(['breathe', str(plain), str(tmp_path/'turned'), '--mode', 'smooth', '--depth', str(DEPTH),
                     '--breaths', str(BREATHS), '--chest-dir', '-30']) == 0
    cells = [Image.fromarray(rgba(tmp_path/'turned'/name)) for name in FILES]
    got = sway.solve_cells(sway.read_ask([CHEST], breathe=tmp_path/'turned'/'breathe.json'), cells, None)
    k = int(np.argmax(got.dy))
    assert got.dy[k] > 0 and got.dx[k] > 0  # the chest rises up and to the left; the part lies behind, down and right
    assert math.degrees(math.atan2(got.dx[k], got.dy[k])) == pytest.approx(30)


def test_a_record_that_does_not_say_the_breath_is_refused(breathed, tmp_path):
    src = breathed/'breathed'
    report = json.loads((src/'breathe.json').read_text())
    cases = {
        'changed': report | {'phases': [0.0, *report['phases'][1:-1], 0.5]},
        'counted': report | {'frames': N-1},
        'pixel': report | {'config': {k: v for k, v in report['config'].items() if k != 'mode'} | {'mode': 'pixel'}},
        'not one': {'phases': report['phases']},
    }
    whys = {'changed': 'changed after it was written', 'counted': f'says {N-1} frames', 'pixel': 'a pixel breath',
            'not one': "not a Breathe layer's record"}
    for name, bad in cases.items():
        path = tmp_path/f'{name}.json'
        path.write_text(json.dumps(bad))
        with pytest.raises(SystemExit, match=whys[name]):
            cli.main(['video-sway', '--frames', str(src), '--out', str(tmp_path/name), '--breathe', str(path), '--region', region()])
        assert not (tmp_path/name).exists()
    fewer = tmp_path/'fewer'
    fewer.mkdir()
    for name in FILES[:-1]:
        (fewer/name).write_bytes((src/name).read_bytes())
    with pytest.raises(SystemExit, match=f'breathes {N} cells and there are {N-1} to sway'):
        cli.main(['video-sway', '--frames', str(fewer), '--out', str(tmp_path/'f'), '--breathe', str(src/'breathe.json'),
                  '--region', region()])


def test_what_belongs_to_one_drive_is_refused_on_the_other(breathed, tmp_path):
    src = breathed/'breathed'
    asked = ['video-sway', '--frames', str(src), '--out', str(tmp_path/'out'), '--region', region()]
    on_breath = [*asked, '--breathe', str(src/'breathe.json')]
    for extra, why in ((['--freq', '1'], '--freq and --zeta'), (['--zeta', '0.5'], '--freq and --zeta'),
                       (['--fps', '24'], '--fps is the damped mass'), (['--lag', '0.4'], 'under 1/π'), (['--lag', '0'], 'above 0')):
        with pytest.raises(SystemExit, match=why):
            cli.main([*on_breath, *extra])
    with pytest.raises(SystemExit, match='--lag .* needs --breathe'):
        cli.main([*asked, '--fps', '24', '--lag', '0.1'])
    assert not (tmp_path/'out').exists()


def test_the_inspection_records_the_drive_and_its_way_by_name(breathed, tmp_path):
    src = breathed/'breathed'
    cells = [rgba(src/name) for name in FILES]
    h, w = cells[0].shape[:2]
    loop_dir = tmp_path/'loop'
    loop_dir.mkdir()
    Image.fromarray(np.concatenate(cells, axis=1)).save(loop_dir/'idle.strip.png')
    (loop_dir/'idle.strip.json').write_text(json.dumps({'kind': 'periodic', 'frames': N, 'w': w, 'h': h, 'delay_ms': 41.67}))
    record = sway_inspect.inspect_loop(loop_dir, [CHEST], tmp_path/'out', breathe=src/'breathe.json')
    digest = hashlib.sha256((src/'breathe.json').read_bytes()).hexdigest()
    assert record['follow']['drive']['kind'] == 'breath' and record['follow']['drive']['breathe_sha256'] == digest
    assert record['inputs']['policy']['name'] == sway.BREATH_POLICY != sway.POLICY
    assert record['inputs']['settings'] | {} == {'gain': sway.BREATH_GAIN_DEFAULT, 'on_fold': 'refuse', 'freq_hz': None, 'zeta': None,
                                                'read_on': None, 'stretch_floor': None, 'breathe_sha256': digest,
                                                'lag': sway.LAG_DEFAULT}
    assert record['cells'][0]['changed_px'] == 0 and any(cell['changed_px'] for cell in record['cells'])
    on_body = sway_inspect.inspect_loop(loop_dir, [CHEST], tmp_path/'body')
    assert on_body['inputs']['policy']['name'] == sway.POLICY and 'drive' not in on_body['follow']
    assert on_body['input_id'] != record['input_id']


def test_a_cell_no_region_moves_is_the_cell_as_it_was():
    """A move of nothing keeps even the colour under a clear pixel, which a move clears (the WebP check refuses it)."""
    cell = np.zeros((40, 40, 4), np.uint8)
    cell[..., :3] = np.random.default_rng(1).integers(0, 256, (40, 40, 3))
    cell[10:30, 10:30, 3] = 255
    picture = Image.fromarray(cell)
    still = sway.move_regions(picture, [([(20.0, 20.0, 15.0, 15.0)], (0.0, 0.0))], (2.0, 1.0))
    assert still.tobytes() == picture.tobytes()
    moved = np.asarray(sway.move_regions(picture, [([(20.0, 20.0, 15.0, 15.0)], (0.5, 0.0))], (2.0, 1.0)))
    yy, xx = np.mgrid[0:40, 0:40]
    clear_inside = (((xx-22)/15)**2+((yy-21)/15)**2 < 1) & (cell[..., 3] == 0) & (moved[..., 3] == 0)
    assert cell[clear_inside][:, :3].any() and not moved[clear_inside][:, :3].any()
