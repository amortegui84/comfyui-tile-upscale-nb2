# ComfyUI Tile Upscale AM

Method-aware tile upscale nodes for ComfyUI.

This repository is for full-image tile upscaling: split an image into overlapping tiles, process the tiles with the selected method, stitch them back into one integrated image, and optionally export the final image with DPI metadata.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/amortegui84/comfyui-tile-upscale-AM
```

Restart ComfyUI after installing or updating.

## Nodes

### Tile Crop (AM)

Splits the source image into a row-major IMAGE batch and emits JSON metadata for stitching.

Inputs:

- `method`: `nb2`, `image_2`, `topaz`, `seedv2`, `passthrough`, `custom`
- `grid_cols`, `grid_rows`: tile grid size
- `overlap_percent`: `-1` uses the method preset
- `target_tile_width`, `target_tile_height`: optional uniform tile output size

Outputs:

- `tiles`: IMAGE batch
- `tile_metadata`: JSON metadata for stitching/debug
- `tile_count`: total tile count

### Tile Stitch (AM)

Stitches processed tiles back into one image. It reads the processed tile size automatically and builds the final canvas from the detected scale.

Method behavior:

- `nb2`, `image_2`: regenerative methods, stronger feathering and color matching.
- `topaz`, `seedv2`: faithful methods, minimal feathering and sharper structure preservation.
- `passthrough`: near-exact placement for external or scripted processing.
- `custom`: user-controlled middle ground.

### Tile Info / Debug (AM)

Shows method, grid, source region, tile size, overlap, source canvas size, tile count, feather mode, color matching, and warnings for a selected tile index.

### Save Image With DPI (AM)

Saves the final image with embedded DPI metadata.

Supported formats:

- PNG
- TIFF
- JPEG

Common DPI values:

- 72: screen/web
- 150: draft print
- 300: quality print
- 600: high-resolution print metadata

DPI is metadata only. It changes the intended physical display/print size, not the pixel count and not the real image detail.

## Method Presets

| Method | Category | Preset overlap | Stitch behavior |
|---|---:|---:|---|
| `nb2` | regenerative | 20% | strong feather, color match |
| `image_2` | regenerative | 20% | strong feather, color match |
| `topaz` | faithful | 8% | minimal feather, no color match |
| `seedv2` | faithful | 10% | minimal feather, no color match |
| `passthrough` | passthrough | 4% | near-exact placement |
| `custom` | custom | 12% | medium feather |

## Example Workflows

Current method-aware workflows:

- `workflows/tile_upscale_01_nb2_regenerative_2x2.json`
- `workflows/tile_upscale_02_gpt_image2_regenerative_3x3.json`
- `workflows/tile_upscale_03_faithful_topaz_seedv2_2x2.json`

The workflow folder intentionally contains only the current method-aware tile upscale examples.

## Basic Pipeline

```text
Load Image
  -> Tile Crop (AM)
      tiles -> upscaler node -> Tile Stitch (AM)
      tile_metadata ---------> Tile Stitch (AM)
      tile_metadata ---------> Tile Info / Debug (AM)

Tile Stitch (AM)
  -> Preview Image
  -> Save Image With DPI (AM)
```

## Repository Separation

This project is separate from:

```text
https://github.com/amortegui84/comfyui-inpaint-cropstitch-nb2
```

The inpainting repository should remain focused on regional inpaint/crop-stitch workflows. This repository should contain the tile upscale system.
