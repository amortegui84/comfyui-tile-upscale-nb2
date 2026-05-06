# ComfyUI Tile Upscale AM

Method-aware tile upscale nodes for ComfyUI.

The nodes split one image into overlapping tiles, let you process each tile with
an upscaler such as SeedVR2, NB2, GPT-Image-2, Topaz, ESRGAN, or another image
upscaler, then stitch the processed tiles back into one image using the saved
tile geometry.

## Nodes

Nodes appear in the `AM/TileUpscale` category.

| Node | Purpose |
|---|---|
| `Tile Crop (AM)` | Splits the source image into a row-major IMAGE batch and emits JSON tile metadata. |
| `Tile Extract (AM)` | Extracts one tile from the batch by zero-based row-major index. |
| `Tile Scale By / Placeholder (AM)` | Test upscaler that scales an image by a decimal factor. Replace it with your real upscaler. |
| `SeedVR2 Factor / Noise Controls (AM)` | Reusable decimal outputs for SeedVR2 `upscale_factor` and `noise_scale`. |
| `Tile Collect (AM)` | Collects processed tiles back into one IMAGE batch and validates the count when metadata is connected. |
| `Tile Stitch (AM)` | Reconstructs the final image from processed tiles and metadata. |
| `Tile Info / Debug (AM)` | Shows metadata for one tile. |
| `Save Image With DPI (AM)` | Saves PNG, TIFF, or JPEG with DPI metadata. |
| `Tile Cost & Runtime Info (AM)` | Reports estimated API cost and timing at the end of the workflow. |

## Supported Tile Layouts

`Tile Crop (AM)` supports these presets:

| Preset | Tiles | Order |
|---|---:|---|
| `fixed 2x2` / `fixed 2×2` | 4 | `0 1 / 2 3` |
| `3x2 horizontal` | 6 | `0 1 2 / 3 4 5` |
| `2x3 vertical` | 6 | `0 1 / 2 3 / 4 5` |
| `3x3` | 9 | `0 1 2 / 3 4 5 / 6 7 8` |
| `method default` | Method dependent | SeedVR2 defaults to `3x2 horizontal`; most other methods default to `2x2`. |

Tile order is always row-major: left to right across the first row, then the
next row, until the bottom-right tile.

Use `3x2 horizontal` for most landscape or wide images, and for the default
SeedVR2 workflow. It gives six tiles with more horizontal coverage while keeping
the workflow manageable.

Use `2x3 vertical` for portrait images where a horizontal grid would produce
overly wide tile crops.

Use `3x3` when you need more local detail, larger output sizes, or better
coverage for difficult images. It costs more processing time because every
upscaler node runs nine times.

## Aspect Ratio

`Tile Crop (AM)` computes uniform source crop windows for the selected grid and
stores exact source coordinates in metadata. `Tile Stitch (AM)` uses those
coordinates to reconstruct the final canvas, so 3x2, 2x3, and 3x3 layouts
preserve the original image aspect ratio. The final size is inferred from the
processed tile size relative to the original tile size.

When `Tile Stitch (AM)` receives an explicit `upscale_factor`, final dimensions
are deterministic:

```text
final width  = input width  x upscale_factor
final height = input height x upscale_factor
```

If `upscale_factor` is left at `0`, Tile Stitch infers the scale from the
processed tile dimensions. That is useful for fixed-size external upscalers, but
decimal factors can otherwise introduce one-pixel rounding differences.

## SeedVR2 Workflows

Example workflows are in `workflows/`:

| File | Purpose |
|---|---|
| `workflows/workflow example.json` | Existing 6-tile SeedVR2 workflow using `3x2 horizontal`. |
| `workflows/workflow_example_seedvr2_9_tiles.json` | New 9-tile SeedVR2 workflow using `3x3`. |
| `workflows/tile_upscale_01_nb2_2x2_4_tiles.json` | Placeholder 4-tile NB2 workflow. |
| `workflows/tile_upscale_02_image2_2x2_4_tiles.json` | Placeholder 4-tile GPT-Image-2 workflow. |
| `workflows/tile_upscale_03_faithful_2x2_4_tiles.json` | Placeholder 4-tile faithful upscale workflow. |

For SeedVR2 workflows:

1. Load an image.
2. Use `Tile Crop (AM)` with `method=seedv2`.
3. Use `Tile Extract (AM)` nodes for each tile index.
4. Connect each extracted tile to a SeedVR2 upscaler node.
5. Connect all SeedVR2 outputs to `Tile Collect (AM)` in row-major order.
6. Connect `Tile Collect (AM)` `tile_metadata` to `Tile Stitch (AM)`.
7. Preview or save the stitched image.

## Factor And Noise Controls

Use `SeedVR2 Factor / Noise Controls (AM)` when multiple SeedVR2 nodes need the
same values:

| Output | Default | Connect to |
|---|---:|---|
| `upscale_factor` | `2.0` | Every SeedVR2 `upscale_factor` input |
| `noise_scale` | `0.15` | Every SeedVR2 `noise_scale` input |

Also connect `upscale_factor` to `Tile Stitch (AM)` when you want exact final
dimensions from a decimal factor.

The placeholder `Tile Scale By / Placeholder (AM)` also has a decimal
`scale_factor` input defaulting to `2.0`, useful for validating crop and stitch
geometry before using SeedVR2.

## SeedVR2 / Fal Cost Estimation

`Tile Cost & Runtime Info (AM)` calculates the estimated API cost for a tiled
SeedVR2 run and displays a summary report at the end of the workflow.

### How Fal billing works

Fal charges **$0.001 per output megapixel** per API call to SeedVR2.  Each tile
in a tile workflow is a separate API call, so tile count directly multiplies your
cost.

| Grid | Tiles | API calls |
|---:|---:|---:|
| 2×2 | 4 | 4 |
| 3×2 | 6 | 6 |
| 3×3 | 9 | 9 |

### Why billed megapixels exceed the final stitched image

Tiles overlap their neighbours.  For a 3×3 grid with 10% overlap, each tile
includes extra pixels on every shared edge.  Those overlap pixels are processed
and billed in each adjacent tile's API call, so the total billed megapixels are
always higher than the final stitched image megapixels.

```text
cost formula:
  mp_per_tile     = (tile_source_px × upscale_factor)² ÷ 1,000,000
  total_billed_mp = mp_per_tile × tile_count
  fal_cost        = total_billed_mp × price_per_megapixel
```

### Example — 3×3 SeedVR2 workflow

```text
Source image:          4096 × 2896 px
Upscale factor:        2×
Grid:                  3×3 = 9 tiles
Overlap:               10%

Uniform tile (source): ~1502 × ~1062 px  (includes overlap)
Tile output:           ~3004 × ~2124 px
MP per tile:           3004 × 2124 ÷ 1,000,000 = 6.38 MP

Total billed MP:       6.38 × 9 = 57.4 MP
Fal cost ($0.001/MP):  57.4 × 0.001 = $0.057

Final stitched image:  8192 × 5792 px = 47.45 MP
Billed MP vs final:    57.4 MP vs 47.45 MP  (overlap causes ~21% extra billing)
```

### Server compute estimate (optional)

Server-side stitching and local GPU time are separate from Fal API cost.
Enable `include_server_estimate` and set `server_cost_per_hour_usd` to add a
rough estimate based on elapsed time:

```text
server_cost = elapsed_seconds ÷ 3600 × server_cost_per_hour_usd
```

Connect `elapsed_seconds` manually or leave it at `0` to show "Not provided."

### Connecting the node

Add `Tile Cost & Runtime Info (AM)` after `Tile Stitch (AM)` and connect:

| Input | Source |
|---|---|
| `tile_metadata` | `Tile Collect (AM)` → `tile_metadata` |
| `stitched_image` | `Tile Stitch (AM)` → `stitched_image` |
| `tile_count` | `Tile Collect (AM)` → `tile_count` |
| `upscale_factor` | `SeedVR2 Factor / Noise Controls (AM)` → `upscale_factor` |

The `9-tile` example workflow already includes the node wired up.

## Large Print Calculation

For a large print target, convert physical size to inches and multiply by DPI:

```text
61 cm / 2.54 = 24.016 in
24.016 in x 600 DPI = 14,409 px
```

So a 61 cm print at 600 DPI needs about `14,409 px` on that side. Increase tile
count and validate the final pixel dimensions before printing.

## Limitations And Assumptions

- Tile stitching assumes each processed tile preserves its tile aspect ratio.
- If an upscaler returns slightly different tile sizes, `Tile Collect (AM)`
  normalizes all tiles to match `tile_0`.
- Connect `Tile Collect (AM)` inputs without gaps: `tile_0`, `tile_1`,
  `tile_2`, and so on.
- Regenerative upscalers can redraw tile content differently. Higher overlap,
  strong feathering, and color matching can reduce visible seams but cannot
  fully correct content drift.
- DPI metadata does not create pixels. Large-format print targets, such as
  61 cm at 600 DPI, may require increasing tile count and validating the final
  pixel dimensions before printing.
