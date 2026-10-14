# `ui-kit` — one painted sheet to nine-slice UI elements

> Owns: The UI kit layout guide, splitting a transparent UI kit sheet into elements, nine-slice insets and content padding · Index: [docs/README.md](README.md)

A game UI kit is one transparent image with every frame on it: a panel, a button and its pressed state, a card frame, two bar frames and a timer badge. `gen --layout-guide ui-kit` asks the image model to draw each element in a fixed box. `ui-kit` cuts the sheet along those boxes, measures where each element may stretch, and writes one PNG per element with a JSON report. [`game-pack`](game-pack.md#version-2-images-tiles-and-ui) reads that report.

## Generate the sheet

```bash
sprite-gen gen --provider openai --quality high --transparent --aspect-ratio 1:1 \
  --layout-guide ui-kit --prompt-file uikit-prompt.txt --out uikit.png
```

`--layout-guide ui-kit` draws the [layout](#layout) as a guide image, attaches it after any `--ref`, and adds a sentence to the prompt that names every box and what to draw in it. The caller's prompt describes the style: material, palette, trim, mood. It does not need to list the elements or their positions. The guide asks for straight, even edges between corner ornaments, empty element centres, no text and empty space between elements, because only those can be nine-sliced. The sheet must be square; `--aspect-ratio 1:1` gives that on every provider that takes a ratio.

## Layout

The guide and the splitter read one table (`sprite_gen.frames.ui_kit.LAYOUT`). Boxes are in pixels of a 1024 × 1024 sheet and scale with the sheet's size; neighbouring boxes are 48 px apart.

| Element | Box (x, y, w, h) | Stretch | Content |
|---|---|---|---|
| `panel` | 48, 48, 576, 400 | `xy` | a large frame with an empty centre |
| `button` | 672, 48, 304, 128 | `xy` | a wide button, normal state |
| `button_pressed` | 672, 224, 304, 128 | `xy` | the same button, pressed |
| `card` | 704, 400, 272, 576 | `xy` | a tall card frame with an empty centre |
| `hp_bar_frame` | 48, 496, 608, 112 | `x` | an empty horizontal health bar frame |
| `xp_bar_frame` | 48, 656, 608, 112 | `x` | an empty horizontal experience bar frame |
| `timer_badge` | 48, 816, 160, 160 | `none` | a small round badge with an empty centre |

`stretch` is how a game may resize the element: `xy` as a nine-slice, `x` in width only (a three-slice), `none` not at all. It belongs to the layout, not to the measurement.

## Split the sheet

```bash
sprite-gen ui-kit --sheet uikit.png --out-dir ui/
```

The command writes `ui/<element>.png` for every element of the layout and `ui/ui-kit.json`. It never changes the sheet and refuses an output directory that holds the sheet. Options:

* `--alpha-floor N` (default `64`, `0` turns it off) and `--alpha-edge N` (default `2`): the [alpha cleanup](#alpha-cleanup) that runs before anything is cut.
* `--report PATH`: write the JSON report elsewhere (default `<out-dir>/ui-kit.json`).

1. **Alpha cleanup.** Faint haze between elements would join them, so it is cleared first (below).
2. **Parts.** Pixels with alpha above zero form 8-connected parts. A part smaller than 16 pixels is dropped and counted.
3. **Assignment.** Each part goes to the box that contains its centroid, with every box widened by half the gap (24 px at 1024). A part whose centroid lies in no widened box is reported under `unassigned`; it makes the report `needs-review` when it has at least 16 pixels.
4. **Crop.** An element is the bounding box of its box's parts, so ornaments that reach past the box stay whole. A box with no part fails the command: the sheet has to be generated again.
5. **Insets** and **padding** are measured on the cropped element (below).

## Insets

Insets are measured only along the axes the layout lets the element stretch; the others are `0`. Along x (columns; y is the same with rows):

1. Colours are taken premultiplied by alpha, with alpha as a fourth channel, in 8-bit units.
2. The reference band is the middle fifth of the columns (from 40 % to 60 % of the width). The reference column is the per-row median over that band.
3. A pixel deviates when its largest channel difference from the reference column, in its row, exceeds twice the largest difference found in the reference band in that row, plus 8.
4. A column with no deviating pixel is plain. The stretch band is the run of plain columns that contains the reference band.
5. `left` is the band's first column plus a 2 px margin, and `right` is the distance from the band's end to the right edge plus 2 px; a side with no deviating column at all gets `0`. An inset never reaches into the reference band.

So every column with an ornament, a cap, a border line or their anti-aliased edge lies inside a fixed slice, and the stretched middle holds only columns that look like the middle of the element. Two cases make the report `needs-review`, naming the element and the axis: an ornament that reaches the reference band (the band could not grow on that side), and a reference band that is not uniform itself, so an ornament probably sits in the middle of an edge (in some row its largest difference exceeds both 64 and four times the median over the rows). A frame with an ornament in the middle of an edge cannot be nine-sliced; regenerate it.

## Padding

Padding places the content (text, a card's contents, a bar's fill, a badge's digits). On the element's middle five rows, take the per-column median. The interior is the run of columns around the centre that continues while each step to the next column changes no channel by more than 16 and every column stays within 96 of the centre (the median of the central 5 × 5 pixels): a smooth gradient on a button face stays interior, a border's sharp edge ends it. The same on the middle five columns gives the interior rows. A pixel counts as interior when every channel lies within 16 of the range those two runs cover; when a corner of the rectangle is not interior (a round badge, an ornament reaching inward), the rectangle shrinks evenly toward the centre until all four corners are. Each padding is the distance from an element edge to that rectangle, plus 2 px. A content area narrower or shorter than a quarter of its element makes the report `needs-review`: the centre was probably not empty.

## Alpha cleanup

Transparent sheets from image models often carry a faint alpha haze: barely visible pixels around and between the drawn objects. The cleanup is shared by `ui-kit` and [`slice-sheet --alpha-floor`](sheet-slicing.md#transparent-sheets) (`sprite_gen.frames.alpha_floor.clear_haze`).

* A pixel at or above the floor (`--alpha-floor`) is solid. A pixel above zero and below the floor is faint.
* A faint pixel within `--alpha-edge` pixels (8-connected, so diagonals count) of a solid pixel is part of an object's anti-aliased edge and is kept unchanged.
* Every other faint pixel is haze and becomes fully transparent (`0, 0, 0, 0`). A pixel that was already fully transparent is left as it was.

Solid pixels and edge pixels are never changed, so a stroke stays whole as long as some of its pixels reach the floor. A stroke drawn entirely below the floor is indistinguishable from haze and is removed: lower the floor for very faint art. The report measures instead of asserting: it counts what actually changed.

| Field | Meaning |
|---|---|
| `floor`, `edge_px` | the settings |
| `pixels` | pixels in the image |
| `faint_fraction_before`, `faint_fraction_after` | share of all pixels with alpha from 1 to `floor - 1`, before and after |
| `haze_fraction_before`, `haze_fraction_after` | share of all pixels that are faint and farther than `edge_px` from a solid pixel |
| `cleared_pixels` | pixels made transparent |
| `changed_solid_pixels`, `changed_edge_pixels` | solid and kept-edge pixels whose value changed; `0` unless the cleanup is wrong |

## Report

`ui-kit.json`:

```json
{
  "kind": "sprite-gen-ui-kit",
  "version": 1,
  "sheet": "/abs/path/uikit.png",
  "sheet_size": [1024, 1024],
  "status": "pass",
  "review_reasons": [],
  "alpha_cleanup": { "floor": 64, "edge_px": 2, "...": "..." },
  "dropped_parts": 3,
  "unassigned": [],
  "elements": {
    "panel": {
      "file": "panel.png",
      "box": { "x": 48, "y": 48, "w": 576, "h": 400 },
      "rect": { "x": 52, "y": 50, "w": 570, "h": 396 },
      "size": { "w": 570, "h": 396 },
      "stretch": "xy",
      "insets": { "left": 112, "right": 110, "top": 92, "bottom": 90 },
      "padding": { "left": 40, "right": 40, "top": 38, "bottom": 38 },
      "parts": 1
    }
  }
}
```

`box` is the layout box scaled to this sheet and `rect` the element's bounding box, both in sheet pixels. `file` is relative to the report. `size`, `insets` and `padding` are in pixels of the element PNG. Elements appear in layout order. `status` is `needs-review` when any `review_reasons` apply; the PNGs are written either way, so the elements can be inspected.

Like every measurement here, `pass` means the numbers found nothing to flag. Look at the elements stretched to a few sizes before shipping them.

## Python API

* `sprite_gen.frames.ui_kit.split_ui_kit(image, *, alpha_floor=64, alpha_edge=2) -> (elements, report)`: the elements as images by name, and the report as written without `sheet`.
* `sprite_gen.frames.ui_kit.measure_insets(image, stretch) -> (insets, notes)` and `measure_padding(image) -> (padding, notes)`: the measures above on one element image, with the review notes they raise.
* `sprite_gen.frames.ui_kit.boxes(size)`, `draw_guide(path, size)` and `guide_text()`: the layout, the guide image and its prompt paragraph.
* `sprite_gen.frames.alpha_floor.clear_haze(image, *, floor=64, edge=2) -> (image, report)`.
