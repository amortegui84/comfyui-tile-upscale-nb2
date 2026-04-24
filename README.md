# ComfyUI-Tile-Upscale-NB2

## Overview
Tile-based upscaling nodes for ComfyUI, designed for use with Nano Banana 2. Splits large images into four overlapping tiles, scales each one to an exact NB2 resolution without aspect-ratio deformation, and blends the processed results back into a single seamless upscaled image.

## Purpose
Nano Banana 2 works best at its native resolutions. Feeding a full high-resolution image directly often produces inconsistent quality. These nodes solve that by dividing the image into four overlapping tiles, each scaled to a proper NB2 resolution so the model processes every region at peak quality. After generation, the tiles are blended back together with smooth feathering — no visible seams.

## Key Nodes

**Tile Crop (NB2)**: Splits the input image into four overlapping tiles (top-left, top-right, bottom-left, bottom-right). Each tile is cropped and scaled to the selected NB2 resolution while preserving the correct aspect ratio — no stretching. Also outputs the resolved `aspect_ratio` as a STRING that can be wired directly into the Nano Banana node, so the aspect ratio is set automatically without configuring it twice.

**Tile Stitch (NB2)**: Receives the four processed tiles and blends them back into a single upscaled image. Uses smoothstep feathering on all interior edges — paired tile weights always sum to exactly 1, producing seamless results across the full image including the center overlap zone.

## Supported Aspect Ratios and Resolutions

| Aspect Ratio | 1K | 2K | 4K |
|---|---|---|---|
| 16:9 | 1376 × 768 | 2752 × 1536 | 5504 × 3072 |
| 9:16 | 768 × 1376 | 1536 × 2752 | 3072 × 5504 |
| 1:1 | 1024 × 1024 | 2048 × 2048 | 4096 × 4096 |
| 4:5 | 928 × 1152 | 1856 × 2304 | 3712 × 4608 |
| 5:4 | 1152 × 928 | 2304 × 1856 | 4608 × 3712 |

Set `aspect_ratio` to **auto** and the node detects the closest NB2 ratio automatically. The resolved ratio is exposed as an output so it can be connected directly to the Nano Banana aspect ratio input.

## Workflow
1. Connect your image to **Tile Crop (NB2)** and choose resolution and overlap.
2. Wire the `aspect_ratio` STRING output to the Nano Banana aspect ratio input.
3. Send each of the four tile outputs through Nano Banana for processing.
4. Connect the `tile_stitcher` and all four processed tiles to **Tile Stitch (NB2)**.
5. The stitched output is your full upscaled image.

## Installation
Clone into `ComfyUI/custom_nodes` and restart ComfyUI. No additional Python packages required beyond PyTorch, which ComfyUI already provides.

```
git clone https://github.com/amortegui84/comfyui-tile-upscale-nb2
```

## License
Apache 2.0 — © 2025 amortegui84
