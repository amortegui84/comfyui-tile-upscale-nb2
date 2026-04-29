# ComfyUI Tile Upscale — Universal Edition

> Tile-based upscaling nodes for ComfyUI.  
> Supports **NB2, ChatGPT Image 2, DALL-E 3, SDXL, SD15, Topaz, SeedV2** and any custom resolution.  
> Grid sizes from **2×2 to 4×4** (up to 16 tiles per image).  
> Full backward-compatible with v1 NB2 workflows.

---

## What it does

Split a large image into overlapping tiles → process each tile through any AI model → stitch back into a seamless upscaled image.

```
Original 4K image
      ↓
 TileCrop (3×3)          → 9 tiles at SDXL resolution
      ↓
 [AI model per tile]     → 9 enhanced tiles
      ↓
 TileStitch (3×3)        → seamless 8K+ result
```

Tiles overlap with smootherstep (C² quintic) feathering, so there are no visible seams between tiles, including interior tiles in 3×3 and 4×4 grids.

---

## Installation

### Option A — ComfyUI Manager (recommended)
Search for **comfyui-tile-upscale-nb2** in the Manager and click Install. Restart ComfyUI.

### Option B — Git clone
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/amortegui84/comfyui-tile-upscale-nb2
```

### Updating
```bash
cd ComfyUI/custom_nodes/comfyui-tile-upscale-nb2
git pull
```

Restart ComfyUI after installing or updating.

Find nodes under **TileUpscale** (universal) or **NanoBanana2/Tiles** (legacy NB2).

### Optional — Florence2 workflows
Required only for `florence_kijai_mask_crop.json` and `florence_kijai_tile_upscale.json`.
Runs fully **locally** — no API key needed.

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/kijai/ComfyUI-Florence2
```

Or search **ComfyUI-Florence2** in the Manager.

---

## Nodes

### TileCrop (Universal)
Splits an image into N×M tiles.

| Parameter | Description |
|---|---|
| `model_preset` | Resolution target: NB2, ChatGPT-Image-2, DALL-E-3, SDXL, SD15, custom, passthrough |
| `grid` | 2×2 / 3×3 / 4×4 etc. (cols × rows) |
| `aspect_ratio` | auto-detect or manual: 1:1, 16:9, 9:16, 4:5, 5:4, 4:3, 3:4, 2:3, 3:2, 21:9 |
| `resolution_tier` | 1K / 2K / 4K / HD / standard (depends on preset) |
| `overlap` | Fraction of tile width shared between adjacent tiles (0.05 – 0.45) |
| `scale_algo` | bicubic (recommended) / bilinear / nearest |
| `blend_mode` | smootherstep (recommended) / smoothstep / cosine / linear |
| `custom_w/h` | Only used when model_preset = custom |

**Outputs:**
- `tile_stitcher` → connect to TileStitch
- `tiles` → IMAGE batch [N×M, H, W, C] — connect to batch model or TileExtract
- `aspect_ratio` → detected AR string
- `tile_info` → debug info (connect to ShowText to inspect)

---

### TileStitch (Universal)
Blends processed tiles back into one image.

| Parameter | Description |
|---|---|
| `tile_stitcher` | From TileCrop |
| `tiles` | IMAGE batch [N×M, H, W, C] from processed tiles |

Resolution-agnostic: if the model upscaled tiles 2× or 4×, the output canvas scales automatically. No manual factor needed.

---

### TileExtract
Extracts one tile from the batch by index (0-based, row-major).

```
2×2:  0=TL  1=TR
      2=BL  3=BR

3×3:  0  1  2
      3  4  5
      6  7  8
```

---

### TileCollect
Collects individually processed tiles back into a batch.
Connect `tile_0` … `tile_N` in the same row-major order as above.

---

### TileInfo
Shows grid geometry, scale factors, tile positions. Useful for debugging.
Connect output to a **ShowText** node.

---

### Florence Mask Align
Normalises a Florence-generated `MASK` so it matches the real `IMAGE` size exactly before cropping.

Florence2 internally resizes images so the longest side = 1024 px, then pads them to a square.
The output mask is in that padded-square coordinate space.  Without correction, naively resizing
the square mask to the original image dimensions stretches the letterbox area and shifts the bbox —
causing the crop to land at the wrong location (typically offset toward a corner or edge).

`depad_florence=True` (default) detects and removes this letterbox padding before resize, so
the bbox coordinates correctly target the segmented subject in the original image.

| Parameter | Description |
|---|---|
| `resize_mode` | nearest (hard masks) / bilinear / bicubic |
| `threshold` | Binarisation cutoff (default 0.5) |
| `binarize` | Convert mask to binary 0/1 |
| `invert_mask` | Flip mask values |
| `bbox_padding` | Extra pixels added around the detected bbox |
| `depad_florence` | **Remove Florence's square letterbox padding before resize** (default True) |

**Outputs:**
- `image` → passthrough original image
- `mask` → aligned mask at the exact image size
- `masked_preview` → image multiplied by the aligned mask (preview)
- `bbox_x / bbox_y / bbox_w / bbox_h` → bounding box extracted from the aligned mask
- `info` → debug summary: detected mask layout, depad action taken, final bbox (connect to ShowText)

Recommended flow:
`LoadImage → Florence2Run (kijai) → Florence Mask Align → Mask BBox Crop`

---

### Mask BBox Crop
Crops the image and the aligned mask using `bbox_x / bbox_y / bbox_w / bbox_h`, typically straight from `Florence Mask Align`.

Use it when you want a real cut of the detected region instead of only an aligned mask preview.

---

### Tile Crop (NB2) / Tile Stitch (NB2)
Legacy v1 nodes. **Existing workflows load without changes.**
Outputs at positions 0–5 are identical to v1. A new `tiles_batch` output is added at position 6.

---

## Example Workflows

Ready-to-load workflows are in the `workflows/` folder:

| File | Description |
|---|---|
| `nb2_2x2_per_tile.json` | 2×2 per-tile skeleton — TileExtract × 4 → insert your model → TileCollect → TileStitch |
| `universal_3x3_batch.json` | 3×3 batch skeleton — tiles batch direct to TileStitch (replace middle with batch model) |
| `passthrough_topaz_seedv2.json` | 4×4 passthrough skeleton — for Topaz, SeedV2, or any external tool |
| `florence_kijai_mask_crop.json` | Florence2 (kijai) → aligned mask → bbox crop skeleton |
| `florence_kijai_tile_upscale.json` | Florence2 detect subject → aligned crop → 2×2 tile upscale skeleton *(requires kijai/ComfyUI-Florence2)* |

Load any `.json` via **ComfyUI → Load** (drag & drop or File > Open).

---

## Workflow Patterns

### Pattern 1 — NB2 2×2 (existing workflow, no changes needed)
```
LoadImage
  └─ TileCropNB2
        ├── tile_stitcher ──────────────────────────┐
        ├── tile_tl → GeminiNanoBanana2 → tile_tl ──┤
        ├── tile_tr → GeminiNanoBanana2 → tile_tr ──┤
        ├── tile_bl → GeminiNanoBanana2 → tile_bl ──┤
        └── tile_br → GeminiNanoBanana2 → tile_br ──┤
                                                     │
                                        TileStitchNB2┘
```

### Pattern 2 — NB2 3×3 (9 tiles, larger upscale)
```
LoadImage
  └─ TileCrop (NB2, 3×3, 2K)
        ├── tile_stitcher ─────────────────────────────────────────────┐
        └── tiles                                                       │
              ├─ TileExtract(0) → GeminiNanoBanana2 → ┐                │
              ├─ TileExtract(1) → GeminiNanoBanana2 → ┤                │
              ├─ TileExtract(2) → GeminiNanoBanana2 → ┤                │
              ├─ TileExtract(3) → GeminiNanoBanana2 → ┤ TileCollect    │
              ├─ TileExtract(4) → GeminiNanoBanana2 → ┤ (tile_0…8) → TileStitch
              ├─ TileExtract(5) → GeminiNanoBanana2 → ┤                │
              ├─ TileExtract(6) → GeminiNanoBanana2 → ┤                │
              ├─ TileExtract(7) → GeminiNanoBanana2 → ┤                │
              └─ TileExtract(8) → GeminiNanoBanana2 → ┘                │
                                                                        │
                                                         TileStitch ───-┘
```

### Pattern 3 — ChatGPT Image 2 tiles
Tiles are resized to 1024×1024 (or 1536×1024 for 16:9).
Send each tile to the OpenAI images/edits API via a custom API node, then stitch.
```
TileCrop (ChatGPT-Image-2, 2×2, 1K)
  └─ tiles → [OpenAI API node or batch API node]
               └─ processed tiles → TileStitch
```

### Pattern 4 — Topaz / SeedV2 / local upscaler (passthrough)
Tiles are output at their native crop size. The model scales them; TileStitch reads the actual output size automatically.
```
TileCrop (passthrough, 4×4)           ← 16 tiles at native crop size
  └─ tiles → [Topaz / SeedV2 node]    ← model upscales 4× internally
               └─ processed tiles → TileStitch   ← auto-detects 4× scale
```

### Pattern 5 — SDXL ControlNet Tile upscale
```
TileCrop (SDXL, 3×3, 1K)
  └─ tiles → SDXL KSampler (img2img, Tile ControlNet)
               └─ processed tiles → TileStitch
```

---

### Pattern 6 â€” Florence mask -> aligned crop
If Florence returns a mask that looks shifted, tiny, or scaled wrong, insert the alignment node before any crop step:
```
LoadImage
  -> image --------------------------.
  -> Florence mask node -> Florence Mask Align -> aligned MASK
                                            -> masked_preview
                                            -> bbox_x/y/w/h -> Mask BBox Crop
```
Use `nearest` for hard segmentation masks. Raise `bbox_padding` a little if the crop feels too tight.

---

## Grid Size Guide

| Grid | Tiles | Use case |
|---|---|---|
| 2×2 | 4 | Standard upscale. Good for 2–3× on typical images. |
| 3×3 | 9 | More detail, better coverage. Good for 3–5× upscale. |
| 4×4 | 16 | Maximum detail. Very high VRAM. Use for 8K+ output. |
| 3×4 or 4×3 | 12 | Portrait or landscape-biased coverage. |

**Rule of thumb:** A 3×3 grid with 4K NB2 tiles on a 2K source produces ~8–10K equivalent detail.

---

## Model Preset Guide

| Preset | AR / Sizes | Use when |
|---|---|---|
| NB2 | 11 ARs, 1K/2K/4K | Using Nano Banana 2 or Gemini-based upscalers |
| ChatGPT-Image-2 | 1:1 1024, 16:9 1536×1024, 9:16 1024×1536 | Sending tiles to gpt-image-1 API |
| DALL-E-3 | 1:1 / 16:9 / 9:16 | Sending tiles to DALL-E 3 API |
| SDXL | SDXL bucket sizes | SDXL img2img or ControlNet Tile |
| SD15 | 512 / 768 | SD 1.5 img2img |
| custom | Any W×H | Any other model with specific size requirements |
| passthrough | Native crop size | Topaz, SeedV2, local ESRGAN, any external tool |

---

## Overlap Guide

`overlap` = fraction of tile width shared between adjacent tiles.

| Value | Overlap | Use when |
|---|---|---|
| 0.10 | 10% | Fast, minimal blending. Acceptable for simple scenes. |
| 0.15 | 15% | **Default. Good balance.** |
| 0.25 | 25% | Better for scenes with sharp edges near midpoints. |
| 0.35 | 35% | Very smooth transitions. More tiles to process per image. |

Higher overlap = more tiles per image + more compute, but smoother seams.

---

## Blend Mode Guide

| Mode | Continuity | Notes |
|---|---|---|
| `smootherstep` | C² (quintic) | **Recommended.** Perlin's formula — no visible ramp artifacts. |
| `smoothstep` | C¹ (cubic) | Slightly harder edge at ramp start/end. |
| `cosine` | C¹ | Similar to smoothstep, different shape. |
| `linear` | C⁰ | Linear fade. Visible if tiles have colour differences. |

---

## Migration from v1

**Existing NB2 workflows load without changes.** The `TileCropNB2` and `TileStitchNB2` nodes remain in the `NanoBanana2/Tiles` menu and have identical output positions.

New in v1→v2 on the NB2 node:
- Output slot 6 (`tiles_batch`) is new — ignore it if you don't need it.
- More aspect ratios available (3:4, 4:3, 2:3, 3:2, 21:9).

To upgrade a workflow to use the universal nodes:
1. Replace `TileCropNB2` with `TileCrop` (set model_preset=NB2, grid=2×2).
2. Replace `TileStitchNB2` with `TileStitch`.
3. Connect `tiles_batch` output directly to TileStitch for simple batch workflows,
   or use TileExtract to pull individual tiles for per-tile processing.

---

## Changelog

### v2.1.0
- **FlorenceMaskAlign: fix Florence2 letterbox offset** — added `depad_florence` parameter (default True).  Detects and removes Florence's internal square padding before resize so bbox coordinates land on the correct region in non-square images.  Previously the crop would be offset sideways or vertically on landscape / portrait images.
- Updated `florence_kijai_mask_crop.json` workflow with the fix enabled, plus aligned-mask preview via MaskToImage.

### v2.0.0
- Universal `TileCrop` and `TileStitch` nodes with N×M grid support
- Model presets: ChatGPT-Image-2, DALL-E-3, SDXL, SD15, custom, passthrough
- Smootherstep (C²) as default blend mode — better seam quality
- Bilateral feathering for interior tiles (3×3 / 4×4 middle tiles)
- `TileExtract` and `TileCollect` utility nodes
- `TileInfo` debug node
- 11 aspect ratios in NB2 preset (added 3:4, 4:3, 2:3, 3:2, 21:9, 9:21)
- Full backward compatibility: v1 NB2 workflow positions unchanged

### v1.0.0
- Initial release: TileCropNB2 / TileStitchNB2 (2×2, NB2 only)
