# Video → Phosphene → Cortical Map: Minimal Prototype Demo

## Quick start
```
python3 -m venv .venv && source .venv/bin/activate
pip install numpy opencv-python matplotlib
python phosphene_demo.py --camera 0   # or: --image sample.jpg
```
Press `q` to quit.

## What it does
1) Grabs a webcam frame (or loads an image).
2) Downsamples to an N×N grid (default 55).
3) Renders phosphenes as Gaussian blobs on a visual field.
4) Applies a log-polar retinotopy transform to visualize cortical coordinates.
5) Shows side-by-side: original, downsampled grid, phosphene field, cortical map.

## Tuning
- `--grid 16|32|64` controls electrode count.
- `--sigma` controls phosphene spread.
- `--gamma` controls contrast compression.
- `--cortex_w` and `--cortex_h` set cortical map resolution.

## Files
- `phosphene_demo.py` main loop.
- `retinotopy.py` log-polar mapping utilities.
- `utils.py` drawing and Gaussian renderer.

## Notes
- Retinotopy is an approximate log-polar model. Replace with your map once you have electrode positions.
- For VR/Unity, mirror the phosphene renderer to a texture. A C# sketch is included in `Unity_PhopheneShader_Sketch.cs`.
