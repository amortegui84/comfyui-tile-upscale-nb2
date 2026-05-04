# ComfyUI Tile Upscale AM

Method-aware tile upscale nodes for ComfyUI.

Split an image into overlapping tiles, process each tile with your upscaler of choice (NB2, GPT-Image-2, Topaz, SeedVR2, ESRGAN, or any other), then stitch them back into one seamless image. Feathering and optional color matching are applied automatically based on the selected method preset.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/amortegui84/comfyui-tile-upscale-AM
```

Restart ComfyUI after installing or updating.

## Nodes

Nodes appear in the **AM/TileUpscale** category, numbered in pipeline order.

### 1. Tile Crop (AM)

Splits the source image into a row-major IMAGE batch and emits JSON metadata that carries the tile geometry through the rest of the pipeline.

| Input | Description |
|---|---|
| `image` | Source image |
| `method` | Blend preset: `nb2`, `image_2`, `topaz`, `seedv2`, `passthrough`, `custom` |
| `grid_preset` | `method default` auto-selects the recommended grid for the chosen method. Pick any fixed preset (e.g. `2×3 — portrait tiles`) to override, or `manual` to use `grid_cols`/`grid_rows` directly. |
| `grid_cols`, `grid_rows` | Active only when `grid_preset = manual` |
| `overlap_percent` | `-1` uses the method preset default |
| `target_tile_width/height` | Optional: force a uniform output tile size |

| Output | Description |
|---|---|
| `tiles` | IMAGE batch (N tiles) |
| `tile_metadata` | JSON string — wire to Tile Collect (AM) |
| `tile_count` | Total number of tiles |

### 2. Tile Extract (AM)

Extracts one tile from the batch by row-major index. Use this when your upscaler processes one image per call (e.g. NB2, GPT-Image-2 API).

| Index layout | 2×2 | 3×3 |
|---|---|---|
| Row 0 | 0 1 | 0 1 2 |
| Row 1 | 2 3 | 3 4 5 |
| Row 2 | — | 6 7 8 |

### 3. Tile Scale By / Placeholder (AM)

A runnable stand-in for your real upscaler. Use it to confirm that tiling and stitching work correctly before connecting your actual model. When ready, replace each `TileScaleByAM` node with your upscaler — the rest of the workflow stays identical.

| Input | Description |
|---|---|
| `image` | Tile image |
| `scale_factor` | Multiply dimensions by this factor (e.g. `2.0` = double resolution) |
| `upscale_method` | `lanczos`, `bicubic`, `bilinear`, `nearest` |

### 4. Tile Collect (AM)

Collects processed per-tile images back into one IMAGE batch. Connect tiles in the same row-major order emitted by Tile Crop. The `tile_metadata` input is optional but enables count validation.

**Tip:** Wire `tile_metadata` from Tile Collect's output (slot 3) to Tile Stitch — this keeps the pipeline linear and avoids a second wire from Tile Crop.

| Output | Description |
|---|---|
| `tiles` | Collected tile batch |
| `tile_count` | Number of connected tiles |
| `info` | JSON summary and warnings |
| `tile_metadata` | Passthrough — wire directly to Tile Stitch (AM) |

### 5. Tile Stitch (AM)

Stitches the processed tile batch into one seamless image. The upscale factor is detected automatically from the processed tile dimensions — no manual scale input required.

| Input | Description |
|---|---|
| `tiles` | Processed tile batch from Tile Collect (AM) |
| `tile_metadata` | From Tile Collect (AM) output slot `tile_metadata` |
| `color_match_override` | `auto` / `on` / `off` — overrides the method preset |
| `feather_mode_override` | `auto` / `strong` / `medium` / `minimal` |

### Tile Info / Debug (AM)

Inspect tile metadata for a specific index. Shows method, grid layout, source region, tile size, overlap, feather mode, color matching, and warnings.

### Save Image With DPI (AM)

Save the stitched image with embedded DPI metadata. DPI is metadata only — it does not add pixel detail.

DPI options: `72` (screen/web), `150` (draft print), `300` (quality print), `600` (high-res print).

| Format | DPI support |
|---|---|
| PNG | pHYs chunk (lossless) |
| TIFF | Resolution tags (lossless with LZW) |
| JPEG | APP0/JFIF fields (lossy) |

---

## Method Presets

The `method` in Tile Crop selects blend geometry — not which upscaler to use. The upscaler is whatever node you place between Tile Extract and Tile Collect.

| Method | Category | Preset overlap | Default grid | Stitch behavior |
|---|---|---|---|---|
| `nb2` | regenerative | 20% | 2×2 | strong feather, color match |
| `image_2` | regenerative | 20% | 2×2 | strong feather, color match |
| `topaz` | faithful | 8% | 2×2 | minimal feather, no color match |
| `seedv2` | faithful | 10% | 2×3 | strong feather, color match — square-ish tiles suit the model |
| `passthrough` | passthrough | 4% | 2×2 | near-exact placement |
| `custom` | custom | 12% | 2×2 | medium feather, user-controlled |

---

## Upscaler Quick Reference

Choose the row that matches your upscaler. All settings can be adjusted in the nodes listed in the **Where to set it** column.

| Upscaler | Method | Grid | Overlap | Feather | Color match | Prompt / reference |
|---|---|---|---|---|---|---|
| **NB2** (Nano Banana 2) | `nb2` | 2×2 | 20% | strong | on | **Yes — required.** Connect the same text prompt to every NB2 node. Describe the subject, style, and level of detail you want. A consistent prompt across all tiles prevents content drift between them. |
| **GPT-Image-2** | `image_2` | 2×2 | 20% | strong | on | **Yes — recommended.** Connect the same prompt to every Image-2 node. Include a description of the scene plus any style or quality keywords. You can also pass the original image as a reference input to anchor the regeneration. |
| **Topaz** (Photo AI / Sharpen AI) | `topaz` | 2×2 | 8% | minimal | off | **No prompt.** Topaz is a faithful upscaler — it does not accept text input. If seams appear, raise overlap to 15% and switch `feather_mode_override` to `medium` in Tile Stitch. |
| **SeedVR2** | `seedv2` | 2×3 | 10–20% | strong | on | **No prompt.** SeedVR2 was trained on video frames (landscape aspect ratio), so the 2×3 default grid produces square-ish tiles that fit the model better. Color match is on by default because SeedVR2 can introduce subtle per-tile brightness drift. If seams are still visible, raise overlap to 20%. |
| **ESRGAN / RealESRGAN** | `topaz` or `passthrough` | 2×2 or batch | 8% | minimal | off | **No prompt.** These models accept a whole batch — you can skip Tile Extract and Tile Collect entirely and wire the tile batch directly to your ESRGAN node, then straight to Tile Stitch. |

### About prompts for regenerative upscalers (NB2, GPT-Image-2)

Regenerative upscalers re-draw each tile guided by your prompt. A few tips:

- **Same prompt to all tiles.** Use one prompt node and wire it to every upscaler node in the workflow. Diverging prompts cause visible seams that no amount of feathering can fix.
- **Describe the output, not the input.** Write what you want the final result to look like — `sharp portrait, fine skin texture, soft studio light` — not a description of degradation (`blurry`, `low res`).
- **Add a reference image when available.** NB2 and GPT-Image-2 both accept an optional reference or style image. Passing the original (or a crop of it) as reference keeps the regeneration anchored to the source content.
- **Keep style consistent.** If tiles look inconsistent in color or lighting despite a shared prompt, enable `color_match_override = on` in Tile Stitch — it normalizes each tile to match its neighbors before blending.

---

## Pipeline

### Per-tile upscaler (NB2, GPT-Image-2, Topaz, etc.)

Each tile is processed independently. Scale factor is inferred automatically at stitch time.

```text
Load Image
  └─ Tile Crop (AM)  [method=nb2, grid=2×2]
       ├─ tiles ──► Tile Extract (AM) [0] ──► [your upscaler] ──► Tile Collect (AM) tile_0
       ├─ tiles ──► Tile Extract (AM) [1] ──► [your upscaler] ──► Tile Collect (AM) tile_1
       ├─ tiles ──► Tile Extract (AM) [2] ──► [your upscaler] ──► Tile Collect (AM) tile_2
       ├─ tiles ──► Tile Extract (AM) [3] ──► [your upscaler] ──► Tile Collect (AM) tile_3
       └─ tile_metadata ──────────────────────────────────────► Tile Collect (AM)

Tile Collect (AM)
  ├─ tiles ────────────────────────────────────────────────────► Tile Stitch (AM)
  └─ tile_metadata ────────────────────────────────────────────► Tile Stitch (AM)

Tile Stitch (AM)
  ├─► Preview Image
  └─► Save Image With DPI (AM)
```

### Batch upscaler (ESRGAN, RealESRGAN, etc.)

If your upscaler accepts a batch of images, skip Extract/Collect entirely.

```text
Load Image
  └─ Tile Crop (AM)
       ├─ tiles ─────────────────────────────────────────────────► [batch upscaler]
       └─ tile_metadata ──────────────────────────────────────────► Tile Stitch (AM)

[batch upscaler] ──► Tile Stitch (AM) ──► Save Image With DPI (AM)
```

---

## Example Workflows

All three workflows ship with **Tile Scale By / Placeholder (AM)** as the upscaler. Load any of them, confirm the tiling and stitching look correct, then replace each placeholder node with your real upscaler.

| File | Method | Grid | Tiles | Upscaler slot |
|---|---|---|---|---|
| `tile_upscale_01_nb2_2x2_4_tiles.json` | `nb2` | 2×2 | 4 | 4× TileScaleByAM |
| `tile_upscale_02_image2_3x2_6_tiles.json` | `image_2` | 3×2 | 6 | 6× TileScaleByAM |
| `tile_upscale_03_faithful_2x2_4_tiles.json` | `topaz` | 2×2 | 4 | 4× TileScaleByAM |

Each workflow includes a preview after the crop (see all tiles) and a preview after the collect (see all upscaled tiles before stitching). To use more tiles, increase `grid_cols`/`grid_rows` in Tile Crop and add the matching Extract + ScaleBy nodes.

---

## Repository Separation

This project is separate from:

```
https://github.com/amortegui84/comfyui-inpaint-cropstitch-nb2
```

That repository handles regional inpaint / crop-stitch workflows. This repository is the tile upscale system.
