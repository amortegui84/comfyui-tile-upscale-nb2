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

## Upscale Service — Nano Banana or Any Dedicated Upscaler

The workflow is built around Nano Banana 2, but the upscale step at the tile level is **fully interchangeable**. The four tiles output by **Tile Crop (NB2)** are standard `IMAGE` types — wire them into whatever processor you prefer, then connect the four results to **Tile Stitch (NB2)**.

**Compatible upscalers:**
- **Nano Banana 2** (Gemini) — multimodal, prompt-guided, API-based
- **Magnific / Krea / other dedicated upscale APIs** — drop-in replacements via their respective ComfyUI nodes
- **Local models** — Real-ESRGAN, ESRGAN, SwinIR, or any model loaded via ComfyUI's built-in upscale nodes
- **Any node that accepts `IMAGE` and returns `IMAGE`** — the stitcher does not care what processed the tile

This makes it straightforward to swap services based on budget, speed, or quality requirements without changing the crop/stitch logic.

### Controlling Output Quality with a Prompt

When using an AI model as the upscaler, a text prompt controls how the model interprets each tile. The included example workflow uses a **COPY MODE** prompt that instructs the model to:

- Only increase pixel resolution and clarity — no new detail invented
- Preserve the exact camera angle, subject, pose, lighting, and background
- Treat the image as a faithful copy operation, not a creative reinterpretation

Adjusting this prompt is the primary lever for quality control: a strict COPY prompt keeps output pixel-faithful, while a more open prompt lets the model add texture and sharpness at the cost of structural fidelity. Tune it to match the use case.

## Example Workflow

[![Watch the demo](https://img.youtube.com/vi/3A-F3N-qy1w/maxresdefault.jpg)](https://www.youtube.com/watch?v=3A-F3N-qy1w)

The example workflow file `tile_upscale_nb2.json` demonstrates a complete upscale pipeline using Nano Banana 2 (Gemini 3.1 Flash Image):

| Step | Node | Settings |
|---|---|---|
| 1 | **Load Image** | Source image to upscale |
| 2 | **Tile Crop (NB2)** | `4:5` · `2K` · overlap `0.15` · `gpu` |
| 3 | **ImageBatchMulti** | Pairs each tile with a reference image for fidelity anchoring |
| 4 | **GeminiNanoBanana2** × 4 | One per tile · `4:5` · `4K` · COPY MODE prompt |
| 5 | **Tile Stitch (NB2)** | Blends the four upscaled tiles into the final image |
| 6 | **Save Image** | Exports the final upscaled result |

The `aspect_ratio` STRING from **Tile Crop** routes through a single Reroute node into all four upscaler nodes simultaneously — the ratio is set once and propagates automatically.

## Workflow

1. Connect your image to **Tile Crop (NB2)** and choose resolution and overlap.
2. Wire the `aspect_ratio` STRING output to the upscaler's aspect ratio input.
3. Send each of the four tile outputs through your upscale service of choice.
4. Connect the `tile_stitcher` and all four processed tiles to **Tile Stitch (NB2)**.
5. The stitched output is your full upscaled image.

## Installation
Clone into `ComfyUI/custom_nodes` and restart ComfyUI. No additional Python packages required beyond PyTorch, which ComfyUI already provides.

```
git clone https://github.com/amortegui84/comfyui-tile-upscale-nb2
```

## License
Apache 2.0 — © 2025 amortegui84
