"""`video-sway --region-mask` with `--region-over`, `--region-shear`, and the crease the wear leaves where it is.

`--region-over` lays the region over the figure under it: what lies under its lower edge is no anchor of the rim, so the
lower edge moves with the rest; the band of it within the air's reach is drawn over by the region's own pixels by their
alpha and keeps its bytes where they do not land; where the region lifts, its pixels show the figure under it; its
outline against the air goes with it and leaves the air behind, its outline against the figure stays; `--hold-mask` names
the pixels it leaves out (an arm), never the region's, never drawn over. `--region-shear S` widens the rim so a move one
way rises across it the other way by at most S px a px. With either, the ledger reads the steps the drawn move takes, the
strip a region drawn short of its side leaves standing, and its tip. `--region-crease` with `--region-shrink`: the wear
leaves the lower edge on the crease. Synthetic figures only: the region test's body and its striped part over the torso's
right side and past it over the air, a dark crease under it on the torso; an arm block beside the part's lower left; a dark
outline under the part over the air.

Each case first reads what a sway writes with the new keys (its cells, its record), so on a release without them the
same input fails on what comes out.
"""
import json
import math

import numpy as np
import pytest
from PIL import Image
from sprite_gen import cli
from sprite_gen.video import sway

from tests.video.test_follow_inspect import N, W, H, strip_dir, sway as body_sway
from tests.video.test_sway_region import BOTTOM, CREASE, PART, SOFT, answer_of, cells_of, mask_png, part_at, part_figure, text

ARM = (60, 200, 90, 255)
ARM_BOX = (42, 78, 49, 104)  # x0, y0, x1, y1 in cell 0, last included: beside the part's lower left, on the torso
OUTLINE = (40, 20, 30, 255)


def arm_at(k):
    """The arm's pixels in cell k: carried with the body."""
    sx, sy = body_sway(k)
    yy, xx = np.mgrid[0:H, 0:W]
    x0, y0, x1, y1 = ARM_BOX
    return (xx >= x0+sx) & (xx <= x1+sx) & (yy >= y0+sy) & (yy <= y1+sy)


def arm_figure(k, n=N):
    """The region test's body and part, and an arm block beside the part's lower left, over the torso."""
    a = np.asarray(part_figure(k, n)).copy()
    a[arm_at(k)] = ARM
    return Image.fromarray(a, 'RGBA')


def outline_at(k):
    """A dark outline row under the part where it hangs over the air (past the torso's soft side)."""
    sx, sy = body_sway(k)
    yy, xx = np.mgrid[0:H, 0:W]
    below = np.roll(part_at(k), 1, axis=0) & ~part_at(k)
    return below & (yy == BOTTOM+1+sy) & (xx > 86+sx)


def outlined_figure(k, n=N):
    a = np.asarray(arm_figure(k, n)).copy()
    a[outline_at(k)] = OUTLINE
    return Image.fromarray(a, 'RGBA')


def arm_masks(path):
    """The arm in every cell, a folder of one PNG per cell, held as given."""
    path.mkdir()
    for k in range(N):
        mask_png(path/f'cell-{k:03d}.png', arm_at(k))
    return path


def lower_rows(region, solid, columns, rows=3):
    """The region's lowest `rows` pixels in each of `columns` whose lowest one has a solid pixel under it (on the figure)."""
    out = np.zeros_like(region)
    for x in np.nonzero(region.any(axis=0))[0]:
        if x not in columns:
            continue
        ys = np.nonzero(region[:, x])[0]
        if ys.max()+1 < H and solid[ys.max()+1, x]:
            out[ys[-rows:], x] = True
    return out


def share_of(answer, k, where):
    """The mean move over `where` in cell k over the cell's offset."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    field, _ = answer.mask_field(k, xx, yy)
    offset = math.hypot(answer.dx[k], answer.dy[k])
    return float(np.hypot(field[1], field[2])[where].mean()/offset)


RISE = 8  # rows over the lower rows the part's own move is read for them to be set against


def lifting(answer, least=1.0):
    """The cells whose part lifts by `least` px or more."""
    return [k for k in range(N) if answer.dy[k] <= -least]


# ---------------------------------------------------------------- off, nothing of them

def test_without_the_new_keys_nothing_of_them_is_read_or_written(tmp_path):
    mask = mask_png(tmp_path/'mask.png', part_at(0))
    as_ever, asked_off = strip_dir(tmp_path/'as-ever', arm_figure), strip_dir(tmp_path/'off', arm_figure)
    record = sway.sway_loop(as_ever, [PART], region_mask=mask, band_up=(0.4, 'auto'), on_fold='lower')
    off = sway.sway_loop(asked_off, [PART], region_mask=mask, band_up=(0.4, 'auto'), on_fold='lower', region_over=False,
                         region_shear=None)
    assert (as_ever/'walk.strip.png').read_bytes() == (asked_off/'walk.strip.png').read_bytes()
    assert off['region_mask'] == record['region_mask'] and off['fold'] == record['fold']
    masked = record['region_mask']
    assert 'over' not in masked and 'over' not in masked['asked'] and 'shear' not in masked['asked']
    assert 'shear_px' not in masked['rim']
    assert not {'band_px', 'step_across_most', 'still_strip', 'tip', 'edge_folded_px'} & set(masked['ledger'])


# ---------------------------------------------------------------- the crease the wear leaves where it is

def test_the_wear_leaves_the_lower_edge_on_the_crease_and_wears_in_the_rest(tmp_path):
    mask = mask_png(tmp_path/'mask.png', part_at(0))
    loop_dir = strip_dir(tmp_path/'loop', part_figure)
    record = sway.sway_loop(loop_dir, [PART], region_mask=mask, region_crease=True)
    answer = answer_of(strip_dir(tmp_path/'again', part_figure), [PART], region_mask=mask, region_crease=True)
    worn, part = answer.mask.worn[0], part_at(0)
    on_torso = [x for x in range(W) if part[:, x].any() and x <= 85]
    over_air = [x for x in range(W) if part[:, x].any() and x > 85]
    lowest = {x: int(np.nonzero(worn[:, x])[0].max()) for x in range(W) if worn[:, x].any()}
    kept = [x for x in on_torso if x in lowest]
    assert kept and all(lowest[x] == BOTTOM+1 for x in kept)  # on the crease, as crease_of ended it
    assert all(lowest[x] == BOTTOM-2 for x in over_air if x in lowest)  # over the air worn in by 2 as ever
    xs = sorted(lowest)
    assert xs[0] == min(on_torso)+2 and xs[-1] == max(over_air)-2  # the sides worn in by 2 as ever
    assert record['region_mask']['mask']['box'][3] == BOTTOM+1 and record['region_mask']['crease'][0]['rows'] == \
        {str(x): BOTTOM+1 for x in on_torso}
    # Nothing under the crease moves: the torso keeps its bytes there in every cell.
    source, moved = cells_of(strip_dir(tmp_path/'source', part_figure)), cells_of(loop_dir)
    yy = np.mgrid[0:H, 0:W][0]
    for k in range(N):
        under = (source[k][..., 3] > 0) & (yy > BOTTOM+1+body_sway(k)[1]) & ~part_at(k)
        assert np.array_equal(moved[k][under], source[k][under]), k


# ---------------------------------------------------------------- the layer

@pytest.mark.parametrize('gain', [2.5, 1.25])
def test_laid_over_the_figure_its_lower_edge_moves_with_the_rest(tmp_path, gain):
    mask = mask_png(tmp_path/'mask.png', part_at(0))
    kw = dict(gain=gain, region_mask=mask, region_shrink=0, region_crease=True)
    loop_dir = strip_dir(tmp_path/'loop', part_figure)
    record = sway.sway_loop(loop_dir, [PART], region_over=True, **kw)
    over = answer_of(strip_dir(tmp_path/'over', part_figure), [PART], region_over=True, **kw)
    pinned = answer_of(strip_dir(tmp_path/'pinned', part_figure), [PART], **kw)
    read = 0
    for k in range(N):
        if math.hypot(over.dx[k], over.dy[k]) < 0.5:
            continue
        solid = np.asarray(over.cells[k])[..., 3] >= 128
        region = over.mask.region_in(k, solid)
        # well inside the torso, away from its soft side and the part's corner over the air
        sx = body_sway(k)[0]
        lower = lower_rows(region, solid, range(60+sx, 76+sx))
        above = np.roll(lower, -RISE, axis=0) & region
        # Laid over it, the lower edge moves as the part over it does; held by the figure under it, it does not.
        assert share_of(over, k, lower) >= share_of(over, k, above) - 0.05, k
        assert share_of(pinned, k, lower) <= 0.5*share_of(pinned, k, above), k
        read += 1
    assert read > 0
    assert record['region_mask']['over']['policy'] == sway.OVER_POLICY
    assert record['region_mask']['over']['band_px'] == math.ceil(over.reach)+sway.REGION_AIR == record['region_mask']['air_px']
    assert record['region_mask']['asked']['over'] is True and record['region_mask']['asked']['hold'] is None


def test_the_band_is_drawn_over_where_the_layer_lands_and_keeps_its_bytes_elsewhere(tmp_path):
    mask = mask_png(tmp_path/'mask.png', part_at(0))
    kw = dict(region_mask=mask, region_shrink=0, region_crease=True, region_over=True)
    loop_dir = strip_dir(tmp_path/'loop', part_figure)
    record = sway.sway_loop(loop_dir, [PART], **kw)
    answer = answer_of(strip_dir(tmp_path/'again', part_figure), [PART], **kw)
    source, moved = cells_of(strip_dir(tmp_path/'source', part_figure)), cells_of(loop_dir)
    air = math.ceil(answer.reach)+2
    covered = 0
    for k in range(N):
        drawn, read = answer.masked_cell(k)
        assert np.array_equal(np.asarray(drawn), moved[k]), k
        band, laid = read['band'], read['laid']
        # the layer draws the band and the air; no read of the move's is taken there
        assert not (read['readers'] & (band | read['air'])).any(), k
        changed = np.any(moved[k] != source[k], axis=-1)
        visible = source[k][..., 3] > 0
        # Outside the region, its soft edge, the air and the band nothing changes; in the band only where the layer lands.
        outside = visible & ~read['region'] & ~read['soft'] & ~band
        assert not (changed & outside).any(), k
        assert not (changed & band & ~laid).any(), k
        covered += int((changed & band).sum())
        # The band is the torso under the lower edge (the crease, the region's last row; beside it, the torso's soft side),
        # within the air's reach of it.
        ys, _ = np.nonzero(band)
        assert len(ys) and not (band & ~visible).any(), k
        assert np.nonzero(band & (source[k][..., 3] >= 128))[0].min() >= BOTTOM+2+body_sway(k)[1], k
        assert ys.min() >= BOTTOM+1+body_sway(k)[1] and ys.max() <= BOTTOM+1+body_sway(k)[1]+sway.REGION_SOFT+air, k
    assert covered > 0
    ledger = record['region_mask']['ledger']
    assert ledger['cover_depth_px_max'] <= air and ledger['over_px_max'] > 0 and ledger['held_changed_px_max'] == 0
    assert ledger['folded_px_max'] == 0 and min(ledger['jacobian_min']) > 0 and ledger['walked_px_max'] == 0


def test_where_the_part_lifts_it_shows_the_figure_under_it(tmp_path):
    mask = mask_png(tmp_path/'mask.png', part_at(0))
    kw = dict(region_mask=mask, region_shrink=0, region_crease=True, region_over=True)
    loop_dir = strip_dir(tmp_path/'loop', part_figure)
    record = sway.sway_loop(loop_dir, [PART], **kw)
    answer = answer_of(strip_dir(tmp_path/'again', part_figure), [PART], **kw)
    moved = cells_of(loop_dir)
    cells = lifting(answer)
    assert cells
    shown = 0
    for k in cells:
        _, read = answer.masked_cell(k)
        lifted = read['lifted']
        assert lifted.any(), k
        sx, sy = body_sway(k)
        side = np.mgrid[0:H, 0:W][1] == 86+sx  # the torso's soft side
        # the torso and the crease read up into the part's lower rows, each in its own column: no hole (the torso's soft
        # side shows as soft as it is), the torso's and the crease's red, none of the stripes' (200 or more)
        assert (moved[k][lifted & ~side][:, 3] >= 128).all(), k
        assert (moved[k][lifted & side][:, 3] >= SOFT-1).all(), k
        assert float(moved[k][lifted][:, 0].mean()) < 100, k
        # the torso's soft side under the part stays where it is as the part lifts off it, no notch: in the rows the part
        # leaves, it is as soft as it was under them
        rows = slice(BOTTOM+1+sy+math.floor(answer.dy[k]), BOTTOM+1+sy)
        assert (moved[k][rows, 86+sx, 3] >= SOFT-1).all(), k
        shown += int(lifted.sum())
    assert shown > 0 and record['region_mask']['ledger']['lifted_px_max'] > 0


def test_its_outline_against_the_air_goes_with_it_and_against_the_figure_stays(tmp_path):
    mask = mask_png(tmp_path/'mask.png', part_at(0))
    kw = dict(region_mask=mask, region_shrink=0, region_over=True, hold_mask=arm_masks(tmp_path/'arm'))
    loop_dir = strip_dir(tmp_path/'loop', outlined_figure)
    sway.sway_loop(loop_dir, [PART], **kw)
    answer = answer_of(strip_dir(tmp_path/'again', outlined_figure), [PART], **kw)
    source, moved = cells_of(strip_dir(tmp_path/'source', outlined_figure)), cells_of(loop_dir)
    for k in range(N):
        _, read = answer.masked_cell(k)
        line = outline_at(k)
        assert line.any() and np.array_equal(read['edge'] & line, line), k  # the outline over the air: the part's
        crease = np.all(source[k][..., :3] == CREASE, axis=-1) & (source[k][..., 3] == 255)
        assert crease.any() and not (read['edge'] & crease).any(), k  # on the torso, the figure's
        assert (read['band'] & crease).any(), k
    cells = lifting(answer)
    assert cells
    for k in cells:
        line = outline_at(k)
        # lifted, the outline over the air leaves the air behind it, no speck: none of it stays where it was, at its end
        # beside the torso's soft side too (under the lower edge, no anchor of the rim: `layer_sets`)
        assert (moved[k][line][:, 3] == 0).all(), k
        # and goes up with the part: over its source row, in its columns, dark where the stripes were
        sy, columns = body_sway(k)[1], np.nonzero(line.any(axis=0))[0]
        rows = slice(BOTTOM+1+sy+math.floor(answer.dy[k]), BOTTOM+1+sy)
        dark = moved[k][rows, columns.min():columns.max()+1]
        assert ((dark[..., :3].astype(int).sum(axis=-1) < 300) & (dark[..., 3] >= 128)).any(), k
    # where it is not lifted the crease keeps its bytes but where the part lands on it
    for k in cells:
        crease = np.all(source[k][..., :3] == CREASE, axis=-1) & (source[k][..., 3] == 255)
        assert np.array_equal(moved[k][crease], source[k][crease]), k


def test_the_hold_mask_leaves_the_arm_out_of_the_region_and_the_band(tmp_path):
    over_arm = part_at(0) | (arm_at(0) & (np.mgrid[0:H, 0:W][0] >= 86))  # the mask runs onto the arm's lower rows
    mask = mask_png(tmp_path/'mask.png', over_arm)
    source = cells_of(strip_dir(tmp_path/'source', arm_figure))
    for name, hold in (('folder', arm_masks(tmp_path/'arms')), ('one', mask_png(tmp_path/'arm.png', arm_at(0)))):
        loop_dir = strip_dir(tmp_path/name, arm_figure)
        record = sway.sway_loop(loop_dir, [PART], region_mask=mask, region_shrink=0, region_over=True, hold_mask=hold)
        moved = cells_of(loop_dir)
        for k in range(N):  # the arm keeps its bytes in every cell
            assert np.array_equal(moved[k][arm_at(k)], source[k][arm_at(k)]), (name, k)
        masked = record['region_mask']
        assert masked['ledger']['left_out_px_max'] > 0 and masked['asked']['hold']['per_cell'] == (name == 'folder')
        left_out = masked['over']['left_out']
        if name == 'folder':
            assert left_out['carry'] is None and left_out['grow_px'] is None
        else:
            assert left_out['grow_px'] == 3 and len(left_out['carry']['cells']) == N and left_out['lay'] == []
            assert left_out['pieces'] == [{'px': int(arm_at(0).sum()), 'box': [42, 78, 49, 104]}]
    # Without it the arm's lower rows are the region's, and under its lower edge the band: they move.
    loop_dir = strip_dir(tmp_path/'no-hold', arm_figure)
    sway.sway_loop(loop_dir, [PART], region_mask=mask, region_shrink=0, region_over=True)
    moved = cells_of(loop_dir)
    assert any(not np.array_equal(moved[k][arm_at(k)], source[k][arm_at(k)]) for k in range(N))


# ---------------------------------------------------------------- the shear

@pytest.mark.parametrize('shear', [0.5, 0.25])
def test_the_shear_widens_the_rim_so_a_move_rises_across_it_no_steeper(tmp_path, shear):
    mask = mask_png(tmp_path/'mask.png', part_at(0))
    kw = dict(gain=1.25, region_mask=mask, region_shrink=0)
    loop_dir, plain_dir = strip_dir(tmp_path/'loop', part_figure), strip_dir(tmp_path/'plain', part_figure)
    record = sway.sway_loop(loop_dir, [PART], region_shear=shear, **kw)
    plain = sway.sway_loop(plain_dir, [PART], region_shear=1e6, **kw)  # the ledger reads the edges; the rim as ever
    answer = answer_of(strip_dir(tmp_path/'again', part_figure), [PART], region_shear=shear, **kw)
    rim, was = record['region_mask']['rim'], plain['region_mask']['rim']
    floor = [math.ceil(0.5*math.pi*float(np.max(np.abs(answer.dy)))/shear), math.ceil(0.5*math.pi*float(np.max(np.abs(answer.dx)))/shear)]
    assert rim['shear_px'] == floor and record['region_mask']['asked']['shear'] == shear
    assert (rim['across_px'], rim['down_px']) == (max(was['across_px'], floor[0]), max(was['down_px'], floor[1]))
    assert rim['across_px'] > was['across_px'] and rim['least_px'] == was['least_px'] and was['shear_px'] == [1, 1]
    lx, ly = rim['across_px'], rim['down_px']
    reach = max(math.hypot(x, y) for x, y in zip(answer.dx, answer.dy))
    ledger, before = record['region_mask']['ledger'], plain['region_mask']['ledger']
    # sin² rises at most π/2 a rim's width: a step across at most π/2 · |move| / Lx, a move down within it at most shear
    assert ledger['step_across_max'] <= 0.5*math.pi*reach/lx + 1e-6
    assert ledger['step_down_max'] <= 0.5*math.pi*reach/ly + 1e-6
    for k in range(N):
        spot = ledger['step_across_most'][k]
        if spot is not None:
            assert spot[0] <= math.hypot(shear, 0.5*math.pi*abs(answer.dx[k])/lx) + 1e-6, k
    assert ledger['step_across_max'] < before['step_across_max']
    assert ledger['folded_px_max'] == 0 and ledger['walked_px_max'] == 0


# ---------------------------------------------------------------- the edges the ledger reads

def test_the_still_strip_is_read_from_the_region_to_its_side(tmp_path):
    short = part_at(0) & (np.mgrid[0:H, 0:W][1] >= 53)  # drawn 3 px short of the part's left side
    mask = mask_png(tmp_path/'mask.png', short)
    cell = np.asarray(arm_figure(0))
    arm = arm_at(0)
    widths = sway.still_strip(short, cell[..., 3], arm)
    by_row = {}
    for y in np.nonzero(short.any(axis=1))[0]:
        xs = np.nonzero(short[y])[0]
        left = xs.min()-1-(max(x for x in range(xs.min()) if arm[y, x] or cell[y, x, 3] < 128))
        right = (min(x for x in range(xs.max()+1, W) if arm[y, x] or cell[y, x, 3] < 128))-xs.max()-1
        by_row[int(y)] = (int(left), int(right))
    assert sorted(widths) == sorted(w for pair in by_row.values() for w in pair)
    assert all(by_row[y][0] == 3 for y in range(ARM_BOX[1], BOTTOM+1))  # beside the arm, the 3 px the mask left
    assert all(by_row[y][1] == 0 for y in range(int(PART[1]), BOTTOM+1))  # over the air, it reaches its side
    loop_dir = strip_dir(tmp_path/'loop', arm_figure)
    record = sway.sway_loop(loop_dir, [PART], region_mask=mask, region_shrink=0, region_over=True,
                            hold_mask=arm_masks(tmp_path/'arms'))
    ledger, rim = record['region_mask']['ledger'], record['region_mask']['rim']['across_px']
    first = ledger['still_strip'][0]
    assert first == {'median': float(np.median(widths)), 'most': max(widths), 'ends': len(widths),
                     'covered': sum(w <= rim for w in widths)}
    assert len(ledger['still_strip']) == N and all(entry is not None for entry in ledger['still_strip'])


def test_the_tip_is_read_on_the_side_over_the_air(tmp_path):
    # A tip moved whole is its own shape: 0 px off it once the move is taken away, the move found.
    cell = np.asarray(part_figure(0))
    mine = part_at(0)
    shifted = np.zeros_like(cell)
    shifted[1:, 2:] = cell[:-1, :-2]
    move = (np.full((H, W), 2.0), np.full((H, W), 1.0))
    tip = sway.tip_shape(cell, shifted, mine, move)
    assert tip['side'] == 'right' and tip['shape_px'] == 0.0 and tip['shift'] == [2.0, 1.0] and tip['move_mean'] == 2.2361
    xs = np.nonzero(mine.any(axis=0))[0]
    rows = np.nonzero(mine[:, xs.max()-9:].any(axis=1))[0]
    assert tip['box'] == [int(xs.max())-9, int(rows.min())-8, int(xs.max())+9, int(rows.max())+9] and tip['ring_px'] > 0
    assert sway.tip_shape(cell, cell, mine, (np.zeros((H, W)), np.zeros((H, W))))['shape_px'] == 0.0
    mask = mask_png(tmp_path/'mask.png', part_at(0))
    record = sway.sway_loop(strip_dir(tmp_path/'loop', part_figure), [PART], region_mask=mask, region_shear=0.5)
    ledger = record['region_mask']['ledger']
    tips = ledger['tip']
    assert len(tips) == N and all(t['side'] == 'right' for t in tips)
    assert ledger['tip_shape_max'] == list(max(((t['shape_px'], k) for k, t in enumerate(tips)), key=lambda q: q[0]))
    assert ledger['tip_move_max'][0] > 0.5 and all(t['shape_px'] >= 0 for t in tips)


def test_move_steps_reads_the_most_step_both_ways_where_both_are_shown():
    mx, my = np.zeros((6, 8)), np.zeros((6, 8))
    my[:, 5:] = 0.75  # a step down of 0.75 between columns 4 and 5
    mx[3:, :] = 0.5   # a step across of 0.5 between rows 2 and 3
    shown = np.ones((6, 8), bool)
    across, down = sway.move_steps(mx, my, shown)
    assert across == [0.75, 4, 0] and down == [0.5, 0, 2]
    shown[:, 5] = False
    assert sway.move_steps(mx, my, shown)[0][0] == 0.0


# ---------------------------------------------------------------- the same on every path, and refused

def test_the_frames_are_moved_as_the_strip(tmp_path):
    mask = mask_png(tmp_path/'mask.png', part_at(0))
    arms = arm_masks(tmp_path/'arms')
    frames = tmp_path/'frames'
    frames.mkdir()
    names = [f'frame-{k:03d}.png' for k in range(N)]
    for k, name in enumerate(names):
        arm_figure(k).save(frames/name)
    loop_dir = strip_dir(tmp_path/'loop', arm_figure, delay_ms=41.67)
    asked = ['--region-mask', str(mask), '--region-crease', '--region-over', '--hold-mask', str(arms), '--region-shear', '0.5']
    assert cli.main(['video-sway', '--frames', str(frames), '--out', str(tmp_path/'out'), '--fps', '24', '--region', text(PART),
                     *asked]) == 0
    record = sway.sway_loop(loop_dir, [PART], region_mask=mask, region_crease=True, region_over=True, hold_mask=arms,
                            region_shear=0.5)
    strip = cells_of(loop_dir)
    out = [np.asarray(Image.open(tmp_path/'out'/name).convert('RGBA')) for name in names]
    assert all(np.array_equal(out[k], strip[k]) for k in range(N))
    assert json.loads((tmp_path/'out'/'sway.json').read_text())['region_mask'] == record['region_mask']


def test_what_the_new_keys_are_refused_for_leaves_the_loop_as_it_was(tmp_path):
    loop_dir = strip_dir(tmp_path/'loop', part_figure)
    as_cut = (loop_dir/'walk.strip.png').read_bytes()
    mask = mask_png(tmp_path/'mask.png', part_at(0))
    for args, said in (
            (['--region-over'], r'--region-over and --region-shear shape a region mask; they need --region-mask$'),
            (['--region-shear', '0.5'], r'they need --region-mask$'),
            (['--region-mask', str(mask), '--region-shear', '0'], r'--region-shear is the most a move one way may rise across'),
            (['--region-mask', str(mask), '--region-shear', '-1'], r'over 0; got -1$'),
            (['--region-mask', str(mask), '--region-shear', 'nan'], r'over 0; got nan$'),
            (['--region-mask', str(mask), '--hold-mask', str(mask)],
             r"--region-mask holds every visible pixel outside the region already; --hold-mask is a band's, or with --region-over")):
        with pytest.raises(SystemExit, match=said) as refused:
            cli.main(['video-sway', '--loop-dir', str(loop_dir), '--region', text(PART), *args])
        assert refused.value.code not in (0, None), args
        assert (loop_dir/'walk.strip.png').read_bytes() == as_cut and 'follow' not in json.loads((loop_dir/'walk.strip.json').read_text())
