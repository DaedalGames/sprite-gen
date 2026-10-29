"""`video-sway --read-on` with `--read-on-cut`: regions read on one cut of a clip, moved to another cut of the same clip.

A clip cut twice (a cut, and a cut again from the same clip at another start and crop — not a new take) holds the same
source frames in other cells and at other places. The regions read on the first cut's cell 0, with the region mask drawn
there, are paired by the source frame that cell is with the cell of the loop's cut that is that frame, moved onto it by
the whole px the two crops set that frame apart, the mask pixel for pixel, and carried from there; on a loop aligned from
its cut, through the alignment's record of where each cell is from. Each crop is taken on the frame as filmed: less where
its cut's anchor moved that frame (the body anchor's ramp, the motion anchor's correction), as its record gives it. A cut
whose frame the loop's cut does not hold, a cut whose record names no source frame, a cut of another clip, a record that
does not give where its anchor moved the frame and a loop aligned from its cut are refused no-match, saying which, and
leave the loop as it was. Synthetic figures only: the read-on test's runner, cut from a clip of two strides at two starts
and two crops, by hand or by video-loop.

Each case first reads what a sway writes with the new key (its cells, its record, whether it refuses and what it says),
so on a release without it the same input fails on what comes out.
"""
import json
import math

import numpy as np
import pytest
from PIL import Image
from sprite_gen import cli
from sprite_gen.video import loop, sway

from tests.video.test_follow_read_on import H, N, PATCH, SHIFT, W, aligned, cells_of, cut, files, meta_of, runner, sha
from tests.video.test_sway_region import mask_png

CLIP = 2*N  # the clip's frames: the runner's stride twice
FPS = 24


def clip():
    return [runner(k) for k in range(CLIP)]


def bbox(frames):
    """The union of the frames' visible pixels, as video-loop crops a cut: left, top, right, bottom (last excluded)."""
    seen = np.zeros(np.asarray(frames[0]).shape[:2], bool)
    for frame in frames:
        seen |= np.asarray(frame)[..., 3] > 0
    ys, xs = np.nonzero(seen)
    return int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1


def cut_of(path, start, rect, *, source='clip-a', record=True, name='run', anchor='none'):
    """A loop directory as video-loop leaves one, cut from the clip at `start` with the crop `rect` at scale 1: its strip,
    and its record with how its frames were anchored (`foot_anchor`; None: not said) and the source frame each cell is
    (`source_cut`) where `record`."""
    frames = clip()
    x0, y0, x1, y1 = rect
    w, h = x1-x0, y1-y0
    strip = Image.new('RGBA', (w*N, h))
    for k in range(N):
        strip.alpha_composite(frames[(start+k) % CLIP].crop(rect), (k*w, 0))
    meta = {'frames': N, 'sample_indices': list(range(N)), 'source_rect': list(rect), 'w': w, 'h': h, 'kind': 'periodic',
            'loop': True, 'scale': 1.0, 'delay_ms': round(1000/FPS, 2), 'cycle_frames': N, 'cycle_seconds': N/FPS,
            'subsampled': False} | ({} if anchor is None else {'foot_anchor': anchor})
    if record:
        meta['source_cut'] = {'source': {'sha256': source, 'frames': CLIP, 'fps': FPS}, 'start': start, 'length': N,
                              'samples': [start+k for k in range(N)]}
    path.mkdir(parents=True)
    strip.save(path/f'{name}.strip.png')
    (path/f'{name}.strip.json').write_text(json.dumps(meta, indent=2)+'\n')
    return path


def patch_at(frame_index, rect):
    """The runner's chest patch in the cell of source frame `frame_index` cut with `rect`: its centre, and its pixels."""
    phase = 2*math.pi*(frame_index+SHIFT)/N
    cx, cy = 80+round(3*math.sin(phase))-rect[0], 92+round(5*math.cos(2*phase))-rect[1]
    h, w = rect[3]-rect[1], rect[2]-rect[0]
    yy, xx = np.mgrid[0:h, 0:w]
    return (cx, cy), (xx-cx)**2+(yy-cy)**2 <= 17**2


def moved(mask, shift, rect):
    """`mask` moved by the whole px `shift` onto a cell cut with `rect`."""
    out = np.zeros((rect[3]-rect[1], rect[2]-rect[0]), bool)
    ys, xs = np.nonzero(mask)
    out[ys+shift[1], xs+shift[0]] = True
    return out


def walked(path, step, size):
    """The clip keyed as `video-frames` leaves it, one PNG a frame: the runner of frame k laid `step` px further (across,
    down) each frame, on a canvas of `size`."""
    path.mkdir(parents=True)
    for k in range(CLIP):
        frame = Image.new('RGBA', size)
        frame.paste(runner(k), (step[0]*k, step[1]*k))
        frame.save(path/f'frame-{k:03d}.png')
    return path


def loop_cut(clip_dir, path, start, anchor, **kw):
    """`video-loop` on the keyed clip: the cut given (--cycle fixed from `start`, N frames long), anchored by `anchor`."""
    loop.run_loop(clip_dir, path, fps=FPS, state=None, min_len=None, max_len=None, n_out=None, seam_max=100, name='run',
                  report_path=None, cycle_mode='fixed', start=start, length=N, anchor=anchor, repair='off', size_hold='off',
                  seam_pop_max=None, wrap_blend='off', strip_height=520, **kw)
    return path


def patch_in(loop_dir, cell):
    """The runner's chest patch, its own colour's pixels, in a cell of a loop's strip as drawn."""
    return np.all(cells_of(loop_dir)[cell] == PATCH, axis=2)


def read_on_cut(a, b):
    """A region on the chest patch of `a`'s cell 0 and its mask, those pixels, asked on `b` with `--read-on-cut a`: the
    region, the request's keywords, and the answer solved on `b`'s strip as cut."""
    drawn = patch_in(a, 0)
    ys, xs = np.nonzero(drawn)
    region = (float(xs.mean()), float(ys.mean()), 18.0, 18.0)
    kw = dict(region_mask=mask_png(a.parent/f'{a.name}-mask.png', drawn), read_on=sha(a/'run.strip.png'),
              read_on_cut=a/'run.strip.png', **KW)
    answer = sway.solve(sway.read_request(b, [region], **kw), Image.open(b/'run.strip.png').convert('RGBA'),
                        strip_name='run.strip.png', strip_sha256=sha(b/'run.strip.png'))
    return region, kw, answer


@pytest.fixture
def cuts(tmp_path):
    """The first cut (`a`, at frame 4, cropped to the clip's frames) and its region mask on its cell 0, the patch."""
    frames = clip()
    tight = bbox(frames)
    a = cut_of(tmp_path/'a', 4, tight)
    (cx, cy), patch = patch_at(4, tight)
    return a, tight, (cx, cy, 17.0, 17.0), mask_png(tmp_path/'mask.png', patch)


KW = dict(gain=1.0, on_fold='lower', region_shrink=0)


# ---------------------------------------------------------------- moved by the source frame

def test_on_a_cut_again_at_the_same_frame_the_regions_are_the_ones_drawn_there_moved_by_the_crop(cuts, tmp_path):
    a, tight, region, mask = cuts
    wide = (tight[0]-3, tight[1]-2, tight[2]+2, tight[3]+1)  # cut again, cropped 3 px wider left, 2 px higher
    loop_dir = cut_of(tmp_path/'b', 4, wide)
    record = sway.sway_loop(loop_dir, [region], region_mask=mask, read_on=sha(a/'run.strip.png'),
                            read_on_cut=a/'run.strip.png', **KW)
    read_on = record['read_on']
    assert read_on['relation'] == 'recut' and read_on['policy'] == read_on['pair_policy'] == sway.RECUT_POLICY
    assert (read_on['source_frame'], read_on['cut_cell'], read_on['base_cell'], read_on['shift_px']) == (4, 0, 0, [3, 2])
    assert read_on['cut']['sha256'] == sha(a/'run.strip.png') and read_on['drift_px'] == [0.0, 0.0]
    # The same sway asked on the loop's own cell 0 with the mask and the region moved there by hand: byte for byte.
    by_hand = cut_of(tmp_path/'by-hand', 4, wide)
    shifted = mask_png(tmp_path/'shifted.png', moved(np.asarray(Image.open(mask).getchannel('A')) >= 128, [3, 2], wide))
    plain = sway.sway_loop(by_hand, [(region[0]+3, region[1]+2, *region[2:])], region_mask=shifted, **KW)
    assert (loop_dir/'run.strip.png').read_bytes() == (by_hand/'run.strip.png').read_bytes()
    assert record['dx_px'] == plain['dx_px'] and record['dy_px'] == plain['dy_px']


def test_on_a_cut_again_at_another_start_the_mask_is_moved_onto_the_cell_that_is_that_frame(cuts, tmp_path):
    a, tight, region, mask = cuts
    wide = (tight[0]-3, tight[1]-2, tight[2]+2, tight[3]+1)
    loop_dir = cut_of(tmp_path/'b', 1, wide)  # holds source frames 1 to N: frame 4 is its cell 3
    source = cells_of(loop_dir)
    request = sway.read_request(loop_dir, [region], region_mask=mask, read_on=sha(a/'run.strip.png'),
                                read_on_cut=a/'run.strip.png', **KW)
    data = (loop_dir/'run.strip.png').read_bytes()
    answer = sway.solve(request, Image.open(loop_dir/'run.strip.png').convert('RGBA'), strip_name='run.strip.png',
                        strip_sha256=sha(loop_dir/'run.strip.png'))
    assert (answer.read_on['cut_cell'], answer.read_on['base_cell'], answer.read_on['shift_px']) == (3, 3, [3, 2])
    drawn = np.asarray(Image.open(mask).getchannel('A')) >= 128
    # The mask on the cell it was moved to: the pixels drawn, moved by the whole shift, none grown or lost.
    assert np.array_equal(answer.mask.worn[0], moved(drawn, [3, 2], wide))
    assert answer.carry(3) == (0.0, 0.0) and answer.ask.regions[0][:2] == (region[0]+3, region[1]+2)
    record = sway.sway_loop(loop_dir, [region], region_mask=mask, read_on=sha(a/'run.strip.png'),
                            read_on_cut=a/'run.strip.png', **KW)
    assert record['read_on']['base_cell'] == 3 and (loop_dir/'run.strip.png').read_bytes() != data
    # Outside the region and its soft edge every visible pixel keeps its bytes, in every cell.
    out = cells_of(loop_dir)
    for k in range(N):
        _, read = answer.masked_cell(k)
        outside = (source[k][..., 3] > 0) & ~read['region'] & ~read['soft']
        assert np.array_equal(out[k][outside], source[k][outside]), k


def test_on_a_loop_aligned_from_the_cut_again_the_pair_is_carried_through_the_alignment(tmp_path):
    # The loop's cut as video-loop cuts one (its own crop), from frame 1; the regions read on a cut from frame 4 cropped
    # 3 px in from its left and 2 px from its top.
    b = cut(tmp_path/'b', clip()[1:1+N])
    meta = meta_of(b)
    assert meta['scale'] == 1.0 and meta['sample_indices'] == list(range(N))
    meta['source_cut'] = {'source': {'sha256': 'clip-a', 'frames': CLIP, 'fps': FPS}, 'start': 1, 'length': N,
                          'samples': [1+k for k in range(N)]}
    (b/'run.strip.json').write_text(json.dumps(meta, indent=2)+'\n')
    left, top, right, bottom = meta['source_rect']
    inner = (left+3, top+2, right, bottom)
    a = cut_of(tmp_path/'a', 4, inner)
    (cx, cy), patch = patch_at(4, inner)
    region, mask = (cx, cy, 17.0, 17.0), mask_png(tmp_path/'mask.png', patch)
    loop_dir = aligned(b, tmp_path/'turned')
    record = meta_of(loop_dir)['cycle_align']
    assert record['origin']['strip_sha256'] == sha(b/'run.strip.png') and record['turned_by'] != 0
    result = sway.sway_loop(loop_dir, [region], region_mask=mask, read_on=sha(a/'run.strip.png'),
                            read_on_cut=a/'run.strip.png', **KW)
    read_on = result['read_on']
    assert read_on['shift_px'] == [3, 2]
    cut_cells = [entry['source'] for entry in record['cells_from']]
    assert read_on['relation'] == 'recut' and read_on['cut_cells'] == cut_cells and read_on['policy'] == sway.TRANSPORT_POLICY
    assert read_on['pair_policy'] == sway.RECUT_POLICY
    assert read_on['cut_cell'] == 3 and read_on['base_cell'] == cut_cells.index(3) and read_on['source_frame'] == 4


# ---------------------------------------------------------------- the frame where each cut's anchor moved it

def test_two_cuts_at_anchor_body_pair_the_frame_where_each_ramp_moved_it(tmp_path):
    # The runner walking 2 px a frame, cut by video-loop at --anchor body from frame 4 and from frame 1: each cut's ramp
    # moves its frame k by round(wrap_dx·k/N), so frame 4 — the first cut's cell 0 — is moved by nothing there, and by
    # round(-24·3/13) = -6 px as the second cut's cell 3. The crops alone set it 6 px apart; on the frames as filmed, none.
    clip_dir = walked(tmp_path/'clip', (2, 0), (220, 220))
    a, b = loop_cut(clip_dir, tmp_path/'a', 4, 'body'), loop_cut(clip_dir, tmp_path/'b', 1, 'body')
    ma, mb = meta_of(a), meta_of(b)
    assert ma['wrap_dx_px'] == mb['wrap_dx_px'] == -24 and ma['source_cut']['samples'][0] == mb['source_cut']['samples'][3] == 4
    assert ma['source_rect'][0] - mb['source_rect'][0] == 6 and ma['source_rect'][1] == mb['source_rect'][1]
    there = patch_in(b, 3)  # frame 4's patch where the second cut drew it
    region, kw, answer = read_on_cut(a, b)
    pair = answer.read_on
    assert (pair['relation'], pair['source_frame'], pair['cut_cell'], pair['base_cell']) == ('recut', 4, 3, 3)
    assert pair['shift_px'] == [0, 0] and pair['drift_px'] == [0.0, 0.0]
    assert there.sum() == patch_in(a, 0).sum() > 100 and np.array_equal(answer.mask.worn[0], there)
    record = sway.sway_loop(b, [region], **kw)
    assert (record['read_on']['relation'], record['read_on']['shift_px']) == ('recut', [0, 0])


def test_two_cuts_at_anchor_body_whose_ramp_moves_nothing_pair_by_their_crops_alone(tmp_path):
    # The runner laid 1 px lower each frame, cut at --anchor body from frame 12 and from frame 5, where the last frame's
    # head and torso lie over the first's (the across of their sway is the same): both wrap offsets are nothing, the ramp
    # moves no frame, and frame 12 — the first cut's cell 0, the second's cell 7 — is moved by the crops alone, as ever.
    clip_dir = walked(tmp_path/'clip', (0, 1), (W, H+CLIP))
    a, b = loop_cut(clip_dir, tmp_path/'a', 12, 'body'), loop_cut(clip_dir, tmp_path/'b', 5, 'body')
    ma, mb = meta_of(a), meta_of(b)
    assert ma['wrap_dx_px'] == mb['wrap_dx_px'] == 0 and ma['source_cut']['samples'][0] == mb['source_cut']['samples'][7] == 12
    crops = [ma['source_rect'][axis] - mb['source_rect'][axis] for axis in (0, 1)]
    assert crops[0] == 0 and crops[1] > 0
    there = patch_in(b, 7)
    region, kw, answer = read_on_cut(a, b)
    assert (answer.read_on['cut_cell'], answer.read_on['base_cell'], answer.read_on['shift_px']) == (7, 7, crops)
    assert there.sum() > 100 and np.array_equal(answer.mask.worn[0], there)


@pytest.mark.parametrize('anchor', ['motion', 'motion-auto'])
def test_two_cuts_at_anchor_motion_pair_the_frame_where_each_correction_laid_it(tmp_path, anchor):
    # The runner walking 2 px a frame, cut by video-loop at --anchor motion from frame 4 and from frame 1 on its head and
    # torso in each cut's first frame: each correction lays its frame k on a canvas padded for every frame, at the padding
    # plus that frame's shift, so the crops alone do not say where frame 4 lies in the second cut; the corrections do.
    clip_dir = walked(tmp_path/'clip', (2, 0), (220, 220))

    def regions(start):
        phase = 2*math.pi*(start+SHIFT)/N
        x, y = round(3*math.sin(phase)) + 2*start, round(5*math.cos(2*phase))
        return [(60+x, 12+y, 101+x, 52+y), (46+x, 52+y, 115+x, 136+y)]  # head, torso

    a = loop_cut(clip_dir, tmp_path/'a', 4, 'motion', anchor_regions=regions(4))
    b = loop_cut(clip_dir, tmp_path/'b', 1, 'motion', anchor_regions=regions(1))
    if anchor == 'motion-auto':
        # --anchor motion-auto finds its regions and its cut itself (it takes no --cycle fixed), and records its
        # correction as --anchor motion does, the same `motion_anchor` under its own `foot_anchor`: these two cuts' records.
        for loop_dir in (a, b):
            (loop_dir/'run.strip.json').write_text(json.dumps(meta_of(loop_dir) | {'foot_anchor': anchor}, indent=2)+'\n')
    ma, mb = meta_of(a), meta_of(b)
    assert ma['foot_anchor'] == mb['foot_anchor'] == anchor and ma['source_cut']['samples'][0] == mb['source_cut']['samples'][3] == 4
    laid = [m['motion_anchor']['padding_ltrb'][axis] + m['motion_anchor']['shifts_xy'][k][axis]
            for m, k in ((ma, 0), (mb, 3)) for axis in (0, 1)]
    crops = [ma['source_rect'][axis] - mb['source_rect'][axis] for axis in (0, 1)]
    assert laid[:2] != laid[2:]  # the two corrections lay frame 4 apart
    there = patch_in(b, 3)
    region, kw, answer = read_on_cut(a, b)
    pair = answer.read_on
    assert (pair['relation'], pair['cut_cell'], pair['base_cell']) == ('recut', 3, 3)
    assert pair['shift_px'] == [crops[axis] - laid[axis] + laid[2+axis] for axis in (0, 1)] != crops
    assert there.sum() == patch_in(a, 0).sum() > 100 and np.array_equal(answer.mask.worn[0], there)


# ---------------------------------------------------------------- refused, saying why

def test_a_cut_whose_frame_the_loop_does_not_hold_or_that_names_none_is_refused_no_match(cuts, tmp_path):
    a, tight, region, mask = cuts
    later = cut_of(tmp_path/'later', 7, tight)  # source frames 7 to N+6: frame 4 is not among them
    before = files(later)
    with pytest.raises(SystemExit, match=r'no-match — the regions were read on another cut, whose cell 0 is source frame 4, '
                                         r'which this loop\'s cut does not hold \(it holds source frames 7 to 19\)'):
        sway.sway_loop(later, [region], region_mask=mask, read_on=sha(a/'run.strip.png'), read_on_cut=a/'run.strip.png', **KW)
    assert files(later) == before
    bare = cut_of(tmp_path/'bare', 5, tight, record=False, name='walk')
    with pytest.raises(SystemExit, match=r"no-match — .*the cut's \(--read-on-cut\) record names no source frame"):
        sway.sway_loop(cut_of(tmp_path/'b', 4, tight), [region], region_mask=mask, read_on=sha(bare/'walk.strip.png'),
                       read_on_cut=bare/'walk.strip.png', **KW)
    # Without the cut's record nothing says which frame its cell 0 is: refused as ever, saying so.
    with pytest.raises(SystemExit, match=r'no-match — .*no record of it says which source frame its cell 0 is \(--read-on-cut'):
        sway.sway_loop(later, [region], region_mask=mask, read_on=sha(a/'run.strip.png'), **KW)
    assert files(later) == before


UNREAD = {  # how the cut the regions were read on and this loop's cut are recorded, and what the refusal says of them
    'cut-says-no-anchor': (dict(name='walk', anchor=None), {},
                           r"the cut's \(--read-on-cut\) record: it does not say how its frames were anchored \(foot_anchor\)"),
    'body-no-wrap': ({}, dict(anchor='body'),
                     r"this loop's record: it was cut at --anchor body and records no whole wrap offset \(wrap_dx_px\)"),
    'motion-no-correction': ({}, dict(anchor='motion'),
                             r"this loop's record: it was cut at --anchor motion and records no whole shift of its frame 3 of 13 "
                             r"and no padding \(motion_anchor\)")}


@pytest.mark.parametrize('case', list(UNREAD))
def test_a_record_that_does_not_give_where_its_anchor_moved_the_frame_is_refused_no_match(cuts, tmp_path, case):
    a, tight, region, mask = cuts
    read_kw, loop_kw, said = UNREAD[case]
    read = cut_of(tmp_path/'read', 4, tight, **read_kw) if read_kw else a
    loop_dir = cut_of(tmp_path/'b', 1, (tight[0]-3, tight[1]-2, tight[2]+2, tight[3]+1), **loop_kw)
    strip = next(read.glob('*.strip.png'))
    before = files(loop_dir)
    with pytest.raises(SystemExit, match=r"no-match — the regions were read on another cut, whose cell 0 is source frame 4 "
                                         r"\(this loop's cut's cell 3\), but where the anchor moved that frame before the crop "
                                         r"cannot be read off " + said + r"; locate them again on this strip$"):
        sway.sway_loop(loop_dir, [region], region_mask=mask, read_on=sha(strip), read_on_cut=strip, **KW)
    assert files(loop_dir) == before


def test_a_loop_aligned_from_its_cut_named_by_read_on_cut_is_refused_no_match(tmp_path):
    # The cut from frame 1 aligned on its own (video-cycle-align): its record keeps the cut's, source frames 1 to N in
    # order, but its cell 0 is another of them, so the frame the record's cut starts on is not where the regions were read.
    o = cut(tmp_path/'o', clip()[1:1+N])
    (o/'run.strip.json').write_text(json.dumps(meta_of(o) | {'source_cut': {
        'source': {'sha256': 'clip-a', 'frames': CLIP, 'fps': FPS}, 'start': 1, 'length': N, 'samples': [1+k for k in range(N)]}},
        indent=2)+'\n')
    turned = aligned(o, tmp_path/'turned')
    assert meta_of(turned)['source_cut']['samples'][0] == 1 and meta_of(turned)['cycle_align']['cells_from'][0]['source'] != 0
    drawn = patch_in(turned, 0)
    ys, xs = np.nonzero(drawn)
    loop_dir = cut_of(tmp_path/'b', 1, bbox(clip()))
    before = files(loop_dir)
    with pytest.raises(SystemExit, match=r'no-match — the regions were read on another cut, but the strip --read-on-cut names '
                                         r'is a loop aligned from its cut \(cycle_align\), whose cells are not the cut.s frames '
                                         r'in order; locate them again on this strip$'):
        sway.sway_loop(loop_dir, [(float(xs.mean()), float(ys.mean()), 18.0, 18.0)],
                       region_mask=mask_png(tmp_path/'mask.png', drawn), read_on=sha(turned/'run.strip.png'),
                       read_on_cut=turned/'run.strip.png', **KW)
    assert files(loop_dir) == before


def test_a_cut_of_another_clip_is_refused_no_match(cuts, tmp_path):
    a, tight, region, mask = cuts
    other = cut_of(tmp_path/'other', 4, (tight[0]-3, tight[1]-2, tight[2]+2, tight[3]+1), source='clip-b')
    before = files(other)
    with pytest.raises(SystemExit, match=r'no-match — the regions were read on another cut, of another clip \(clip-a; this '
                                         r"loop's is clip-b\)"):
        sway.sway_loop(other, [region], region_mask=mask, read_on=sha(a/'run.strip.png'), read_on_cut=a/'run.strip.png', **KW)
    assert files(other) == before


def test_what_the_key_is_refused_for_leaves_the_loop_as_it_was(cuts, tmp_path):
    a, tight, region, mask = cuts
    loop_dir = cut_of(tmp_path/'b', 1, (tight[0]-3, tight[1]-2, tight[2]+2, tight[3]+1))
    before = files(loop_dir)
    text = ','.join(f'{v:g}' for v in region)
    for args, said in (
            (['--read-on-cut', str(a/'run.strip.png')], r'--read-on-cut names the strip --read-on names by its sha256; it needs --read-on$'),
            (['--read-on', 'ab'*32, '--read-on-cut', str(a/'run.strip.png')], r'is [0-9a-f]{64}, not the strip --read-on names'),
            (['--read-on', sha(a/'run.strip.png'), '--read-on-cut', str(tmp_path/'none.png')], r'wants the strip.s PNG and its record')):
        with pytest.raises(SystemExit, match=said) as refused:
            cli.main(['video-sway', '--loop-dir', str(loop_dir), '--region', text, '--region-mask', str(mask), *args])
        assert refused.value.code not in (0, None), args
        assert files(loop_dir) == before
    # And on the command line it is taken as the call takes it.
    assert cli.main(['video-sway', '--loop-dir', str(loop_dir), '--region', text, '--region-mask', str(mask), '--region-shrink',
                     '0', '--gain', '1', '--on-fold', 'lower', '--read-on', sha(a/'run.strip.png'), '--read-on-cut',
                     str(a/'run.strip.png')]) == 0
    assert meta_of(loop_dir)['follow']['read_on']['relation'] == 'recut'
