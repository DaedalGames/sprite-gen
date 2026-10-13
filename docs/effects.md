# Effects

`sprite-gen` creates non-looping additive visual effects (bursts, flashes, impacts) using pixel drawing or video workflows.

## Pixel Effects

Claude draws pixel grids (8–48 px) using palette brightness steps:

```bash
sprite-gen pixel-draw --kind effect --subject 'a small orange fire burst' \
  --size 16 --frames birth,peak,decay,gone --candidates 1 --out-dir fire
```

- **Frames**: 2–8 frames (default: `birth,peak,decay,gone`; other kinds retain at most four). Empty frames are allowed if content changes and at least one frame is visible.
- **Constraints**: No fixed height, feet, or stray spark checks. Origin is centered across all frames.
- **Outputs**: Generates `effect.strip.png`, `effect.strip.json` (mapped in `report.json`), preview, and individual frames.
- **Dark Smoke**: Dark occluding smoke is not additive light; use `sprite`, `icon`, or chroma modes for normal blending.

## Video Pipeline

1. **Prepare canvas**:
   ```bash
   sprite-gen video-canvas --still effect.png --state effect --out canvas.png
   ```
   Accepts RGBA or artwork on black. Retains canvas dimensions and center, composites alpha over black, and writes empty black `canvas.end.png`. Framing overrides are disabled for effects.

2. **Generate prompt**:
   ```bash
   sprite-gen video-prompt --direction front --state effect --character 'A fiery explosion' > prompt.txt
   ```

3. **Generate video**:
   ```bash
   sprite-gen video --image canvas.png --last-frame canvas.end.png \
     --prompt-file prompt.txt --duration 3 --resolution 480p --out clip.mp4
   ```
   `canvas.end.png` can be passed alone without `--image` for text-driven content. Standard provider billing applies.

4. **Extract frames**:
   ```bash
   sprite-gen video-frames --clip clip.mp4 --out-dir frames --alpha-mode brightness
   ```

5. **Extract lifetime sequence**:
   ```bash
   sprite-gen video-loop --frames-dir frames/keyed --fps 24 --state effect \
     --cycle lifetime --out-dir result --name fire
   ```
   Replace `24` with the FPS from `frames/frames.report.json`. State `effect` automatically selects `lifetime`. An empty ending is required; incomplete clips fail without return-to-start conditions and never convert into periodic loops. No particle cleanup or edge failure checks occur. Origin is fixed across all frames; brightest and darkest motion changes are preserved in PNG/WebP.

Automate all stages using `video-set`:
```bash
sprite-gen video-set --base front=effect.png --states effect \
  --out-dir result --character 'A fiery explosion'
```
No per-direction variations are needed; rotate a single front effect at runtime.

## Brightness Alpha & Rendering

Calculates alpha as $a = \max(R, G, B) / 255$. Unpremultiplied color is $U = C / a$ (where $C$ is source-on-black, $0$ when $a = 0$). Output additive rendering in 0–255 units is:
$$\text{output} = \min(255, B + a \cdot U)$$
This preserves saturated colors and faint glows without double-multiplying brightness. There is no threshold: black disappears, dark smoke is faint and cannot occlude, and light saturates to white over pure white backgrounds. Use normal or chroma modes for opaque dark smoke (chroma defaults unchanged).

GIF cannot represent partial transparency, providing only a preview of emission against black with limited alpha. Use PNG or WebP with additive blending for true alpha.

## Metadata

Shared attributes:
- `loop: false`, `hold_last: false`, `blend: add`
- `end_frame`: final delivered strip index
- `anchor`: canvas center (relative to crop in video)

Pipeline differences:
- **Pixel**: `events: []`
- **Video**: `kind: lifetime`, `events: [{frame: <int>, name: "motion_peak"}]`, and `source_cut.samples` naming source frames.

GIF and WebP play once and retain the empty endpoint. Strips can subsample while retaining both endpoints.
