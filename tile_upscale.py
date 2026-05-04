"""
tile_upscale.py — Method-aware tile upscale nodes for ComfyUI.

Nodes:
  TileCropAM        — Split image into tiles with method-aware overlap
  TileStitchAM      — Stitch processed tiles back into one seamless image
  TileInfoAM        — Inspect tile metadata for debugging
  SaveImageWithDPI  — Save image with embedded DPI metadata

Method presets define how crop and stitch behave per upscale type.
"""

import json
import os
import struct
import zlib

import numpy as np
import torch
from PIL import Image

# ── Method presets ─────────────────────────────────────────────────────────────
# Each preset defines crop/stitch behavior for a given upscale method.

METHOD_PRESETS: dict[str, dict] = {
    "nb2": {
        "category": "regenerative",
        "changes_content": True,
        "recommended_overlap_pct": 20.0,
        "feather_mode": "strong",
        "color_match": True,
        "description": "Nano Banana 2 — generative reinterpretation, strong blending",
    },
    "image_2": {
        "category": "regenerative",
        "changes_content": True,
        "recommended_overlap_pct": 20.0,
        "feather_mode": "strong",
        "color_match": True,
        "description": "GPT Image 2 — generative reinterpretation, strong blending",
    },
    "topaz": {
        "category": "faithful",
        "changes_content": False,
        "recommended_overlap_pct": 8.0,
        "feather_mode": "minimal",
        "color_match": False,
        "description": "Topaz-style faithful upscale — minimal blending, preserve structure",
    },
    "seedv2": {
        "category": "faithful",
        "changes_content": False,
        "recommended_overlap_pct": 10.0,
        "feather_mode": "minimal",
        "color_match": False,
        "description": "SeedV2-style faithful upscale — minimal blending, preserve structure",
    },
    "passthrough": {
        "category": "passthrough",
        "changes_content": False,
        "recommended_overlap_pct": 4.0,
        "feather_mode": "minimal",
        "color_match": False,
        "description": "Passthrough / external upscale — exact alignment, near-zero blending",
    },
    "custom": {
        "category": "custom",
        "changes_content": False,
        "recommended_overlap_pct": 12.0,
        "feather_mode": "medium",
        "color_match": False,
        "description": "Custom — user controls all parameters",
    },
}

# Smoothstep power per feather mode.
# Higher power = sharper transition concentrated near the boundary center.
# Lower power (1.0 = linear) = softer, wider transition.
_FEATHER_POWER = {"strong": 1.0, "medium": 2.0, "minimal": 4.0}


# ── Image helpers ──────────────────────────────────────────────────────────────

def _t2np(t: torch.Tensor) -> np.ndarray:
    """(H,W,C) float32 tensor → float32 numpy [0,1]."""
    return t.cpu().float().numpy()


def _np2t(a: np.ndarray) -> torch.Tensor:
    """float32 numpy [0,1] → (H,W,C) float32 tensor."""
    return torch.from_numpy(a.astype(np.float32))


def _pil_to_np(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB")).astype(np.float32) / 255.0


def _np_to_pil(a: np.ndarray) -> Image.Image:
    return Image.fromarray((a * 255).clip(0, 255).astype(np.uint8), "RGB")


def _resize_np(a: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resize (H,W,C) float32 array to (h, w, C) via LANCZOS."""
    if a.shape[1] == w and a.shape[0] == h:
        return a
    return _pil_to_np(_np_to_pil(a).resize((w, h), Image.LANCZOS))


# ── Feather mask ───────────────────────────────────────────────────────────────

def _smoothstep_ramp(n: int, overlap: int, power: float) -> np.ndarray:
    """
    1-D weight array of length n.
    Rises from 0 to 1 over the first `overlap` samples,
    falls from 1 to 0 over the last `overlap` samples,
    stays 1 in between.
    `power` controls the ramp curve (1=linear, 2=smoothstep, 4=steep).
    Complementary ramps from two adjacent tiles sum to (overlap-1)/overlap per
    pixel; the canvas weighted-average normalisation absorbs this exactly.
    """
    ramp = np.ones(n, dtype=np.float64)
    if overlap <= 0:
        return ramp
    # Guard: if overlap > half the tile the two ramp regions would overwrite
    # each other.  Clamp so rising and falling never share indices.
    overlap = min(overlap, n // 2)
    t = np.linspace(0.0, 1.0, overlap, endpoint=False)
    fade = np.power(t, 1.0 / power)   # rising edge: 0 → ~1
    ramp[:overlap] = fade
    ramp[-overlap:] = fade[::-1]      # falling edge: ~1 → 0
    return ramp


def _feather_mask(h: int, w: int, ov_h: int, ov_w: int, power: float) -> np.ndarray:
    """
    2-D weight mask (H, W) float64.
    Falls off smoothly in the `ov_w` / `ov_h` overlap zones at every edge.
    Edge tiles: normalization handles the missing neighbour — no special casing needed.
    """
    row_w = _smoothstep_ramp(h, ov_h, power)
    col_w = _smoothstep_ramp(w, ov_w, power)
    return np.outer(row_w, col_w)


# ── Color matching ─────────────────────────────────────────────────────────────

def _color_match_to_canvas(
    tile: np.ndarray,
    canvas: np.ndarray,
    canvas_weights: np.ndarray,
    feather: np.ndarray,
) -> np.ndarray:
    """
    Adjust tile mean/std to match the already-composited canvas in the overlap zone.

    tile, canvas  — (H, W, 3) float32
    canvas_weights — (H, W, 1) accumulated weight (0 where nothing placed yet)
    feather        — (H, W) weight for this tile
    Returns corrected tile (H, W, 3) float32.
    """
    # Overlap = where feather < 1 AND canvas already has content.
    placed = canvas_weights[:, :, 0] > 1e-6
    in_overlap = (feather < 0.999) & placed
    if not in_overlap.any():
        return tile

    # Normalised canvas values in overlap zone.
    ref = (canvas / np.where(canvas_weights > 0, canvas_weights, 1)).astype(np.float64)

    mask = in_overlap.astype(np.float64)
    eps = 1e-6

    ref_mean = (ref * mask[:, :, None]).sum((0, 1)) / (mask.sum() + eps)
    tile_mean = (tile * mask[:, :, None]).sum((0, 1)) / (mask.sum() + eps)

    ref_std = np.sqrt(
        ((ref - ref_mean) ** 2 * mask[:, :, None]).sum((0, 1)) / (mask.sum() + eps) + eps
    )
    tile_std = np.sqrt(
        ((tile - tile_mean) ** 2 * mask[:, :, None]).sum((0, 1)) / (mask.sum() + eps) + eps
    )

    corrected = (tile - tile_mean) * (ref_std / tile_std) + ref_mean
    # Blend correction proportional to in_overlap (full correction in overlap, none elsewhere).
    blend = np.clip(mask[:, :, None], 0, 1)
    return (tile * (1 - blend) + corrected * blend).clip(0, 1).astype(np.float32)


# ── TileCropAM ─────────────────────────────────────────────────────────────────

class TileCropAM:
    """
    Split an image into overlapping tiles with method-aware defaults.

    The node outputs:
    - tiles          — a batch IMAGE tensor (N, H, W, C).  All tiles are the
                       same pixel dimensions (the "uniform tile size"), achieved by
                       resizing each crop with LANCZOS.  Store the original crop
                       coordinates in tile_metadata so TileStitchAM can reconstruct
                       the canvas at any scale.
    - tile_metadata  — JSON string consumed by TileStitchAM / TileInfoAM.
    - tile_count     — convenience INT output (= grid_cols × grid_rows).
    """

    CATEGORY = "AM/TileUpscale"
    RETURN_TYPES = ("IMAGE", "STRING", "INT")
    RETURN_NAMES = ("tiles", "tile_metadata", "tile_count")
    FUNCTION = "crop_tiles"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "method": (list(METHOD_PRESETS.keys()), {"default": "nb2"}),
                "grid_cols": ("INT", {"default": 2, "min": 1, "max": 16, "step": 1}),
                "grid_rows": ("INT", {"default": 2, "min": 1, "max": 16, "step": 1}),
                "overlap_percent": (
                    "FLOAT",
                    {
                        "default": -1.0,
                        "min": -1.0,
                        "max": 50.0,
                        "step": 1.0,
                        "tooltip": "Overlap as % of base tile size. -1 = use method preset.",
                    },
                ),
            },
            "optional": {
                "target_tile_width": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 32768,
                        "step": 8,
                        "tooltip": "Resize each tile to this width before output. 0 = keep natural tile size (tile 0 size).",
                    },
                ),
                "target_tile_height": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 32768,
                        "step": 8,
                        "tooltip": "Resize each tile to this height before output. 0 = keep natural tile size (tile 0 size).",
                    },
                ),
            },
        }

    # ------------------------------------------------------------------ #

    def crop_tiles(
        self,
        image: torch.Tensor,
        method: str,
        grid_cols: int,
        grid_rows: int,
        overlap_percent: float,
        target_tile_width: int = 0,
        target_tile_height: int = 0,
    ):
        # image: (B, H, W, C) — take the first frame; drop alpha if present.
        src_np = _t2np(image[0])           # (H, W, C)
        if src_np.shape[2] == 4:
            src_np = src_np[:, :, :3]
        src_h, src_w = src_np.shape[:2]

        preset = METHOD_PRESETS[method]
        eff_overlap = overlap_percent if overlap_percent >= 0.0 else preset["recommended_overlap_pct"]

        # Base tile size (integer division — image need not divide evenly).
        base_w = src_w // grid_cols
        base_h = src_h // grid_rows

        # Overlap in pixels (full overlap zone; each tile extends by half on each side).
        ov_w = int(base_w * eff_overlap / 100.0)
        ov_h = int(base_h * eff_overlap / 100.0)
        # Force even so half-overlap is exact integers.
        ov_w = (ov_w // 2) * 2
        ov_h = (ov_h // 2) * 2

        # Build tile crop regions.
        tile_infos: list[dict] = []
        crops: list[np.ndarray] = []

        for row in range(grid_rows):
            for col in range(grid_cols):
                x0 = col * base_w - (ov_w // 2 if col > 0 else 0)
                y0 = row * base_h - (ov_h // 2 if row > 0 else 0)
                x1 = (col + 1) * base_w + (ov_w // 2 if col < grid_cols - 1 else 0)
                y1 = (row + 1) * base_h + (ov_h // 2 if row < grid_rows - 1 else 0)

                # Last column / row: snap to image edge to cover any remainder pixels.
                if col == grid_cols - 1:
                    x1 = src_w
                if row == grid_rows - 1:
                    y1 = src_h

                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(src_w, x1), min(src_h, y1)

                crop = src_np[y0:y1, x0:x1]

                tile_infos.append(
                    {
                        "index": row * grid_cols + col,
                        "row": row,
                        "col": col,
                        "src_x0": x0,
                        "src_y0": y0,
                        "src_x1": x1,
                        "src_y1": y1,
                        "src_w": x1 - x0,
                        "src_h": y1 - y0,
                    }
                )
                crops.append(crop)

        # Determine uniform tile output dimensions.
        # If target not specified, use the natural size of the first tile.
        unif_w = target_tile_width if target_tile_width > 0 else crops[0].shape[1]
        unif_h = target_tile_height if target_tile_height > 0 else crops[0].shape[0]

        tensors: list[torch.Tensor] = []
        for crop in crops:
            resized = _resize_np(crop, unif_w, unif_h)
            tensors.append(_np2t(resized))

        batch = torch.stack(tensors, dim=0)   # (N, unif_h, unif_w, C)

        metadata = {
            "method": method,
            "preset": preset,
            "grid_cols": grid_cols,
            "grid_rows": grid_rows,
            "overlap_pct": eff_overlap,
            "overlap_w": ov_w,
            "overlap_h": ov_h,
            "base_tile_w": base_w,
            "base_tile_h": base_h,
            "src_w": src_w,
            "src_h": src_h,
            "uniform_tile_w": unif_w,
            "uniform_tile_h": unif_h,
            "tiles": tile_infos,
        }

        return (batch, json.dumps(metadata, indent=2), len(tensors))


# ── TileStitchAM ───────────────────────────────────────────────────────────────

class TileStitchAM:
    """
    Reconstruct a seamless image from processed tiles using method-aware blending.

    Accepts ANY processed tile size (the upscaler can change pixel dimensions).
    The upscale factor is inferred automatically from the processed tile size
    vs the uniform_tile_w / uniform_tile_h stored in the metadata, and the
    final canvas is sized accordingly.

    Blending strategy per method category:
    - regenerative (nb2, image_2):  wide smooth feather + optional colour match.
    - faithful (topaz, seedv2):     narrow steep feather, no colour match.
    - passthrough:                  near-zero feather, exact alignment.
    - custom:                       medium feather, user overrides available.
    """

    CATEGORY = "AM/TileUpscale"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("stitched_image",)
    FUNCTION = "stitch_tiles"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tiles": ("IMAGE",),
                "tile_metadata": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "color_match_override": (
                    ["auto", "on", "off"],
                    {
                        "default": "auto",
                        "tooltip": "auto = use method preset. on/off override.",
                    },
                ),
                "feather_mode_override": (
                    ["auto", "strong", "medium", "minimal"],
                    {
                        "default": "auto",
                        "tooltip": "auto = use method preset feather mode.",
                    },
                ),
            },
        }

    # ------------------------------------------------------------------ #

    def stitch_tiles(
        self,
        tiles: torch.Tensor,
        tile_metadata: str,
        color_match_override: str = "auto",
        feather_mode_override: str = "auto",
    ):
        meta = json.loads(tile_metadata)
        preset: dict = meta["preset"]
        tile_infos: list[dict] = meta["tiles"]

        src_w: int = meta["src_w"]
        src_h: int = meta["src_h"]
        ov_w: int = meta["overlap_w"]
        ov_h: int = meta["overlap_h"]
        unif_w: int = meta["uniform_tile_w"]
        unif_h: int = meta["uniform_tile_h"]

        n_tiles = tiles.shape[0]
        if n_tiles != len(tile_infos):
            raise ValueError(
                f"TileStitchAM: received {n_tiles} tiles but metadata describes "
                f"{len(tile_infos)}. Check your tile count."
            )

        # Determine upscale factor from processed tile size vs uniform tile size.
        proc_h, proc_w = tiles.shape[1], tiles.shape[2]
        scale_x = proc_w / unif_w
        scale_y = proc_h / unif_h

        canvas_w = round(src_w * scale_x)
        canvas_h = round(src_h * scale_y)

        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float64)
        weight_acc = np.zeros((canvas_h, canvas_w, 1), dtype=np.float64)

        # Resolve blend settings.
        do_color_match = (
            preset["color_match"]
            if color_match_override == "auto"
            else (color_match_override == "on")
        )
        feather_mode = (
            preset["feather_mode"]
            if feather_mode_override == "auto"
            else feather_mode_override
        )
        power = _FEATHER_POWER[feather_mode]

        # Scaled overlap sizes.
        ov_w_s = round(ov_w * scale_x)
        ov_h_s = round(ov_h * scale_y)

        for idx, tinfo in enumerate(tile_infos):
            tile_np = _t2np(tiles[idx])      # (proc_h, proc_w, C)
            if tile_np.shape[2] == 4:        # drop alpha if upscaler returned RGBA
                tile_np = tile_np[:, :, :3]

            # Destination canvas region (scaled from source coordinates).
            dst_x0 = round(tinfo["src_x0"] * scale_x)
            dst_y0 = round(tinfo["src_y0"] * scale_y)
            dst_w = round(tinfo["src_w"] * scale_x)
            dst_h = round(tinfo["src_h"] * scale_y)
            dst_x1 = min(dst_x0 + dst_w, canvas_w)
            dst_y1 = min(dst_y0 + dst_h, canvas_h)
            dst_w = dst_x1 - dst_x0
            dst_h = dst_y1 - dst_y0

            # Resize processed tile to exactly fit destination region.
            tile_r = _resize_np(tile_np, dst_w, dst_h)

            # Build feather weight mask for this tile.
            # Overlap is applied symmetrically; normalisation handles edge tiles.
            feather = _feather_mask(dst_h, dst_w, ov_h_s, ov_w_s, power)

            # Optional: colour-match tile to already-placed canvas in overlap zone.
            if do_color_match:
                tile_r = _color_match_to_canvas(
                    tile_r,
                    canvas[dst_y0:dst_y1, dst_x0:dst_x1],
                    weight_acc[dst_y0:dst_y1, dst_x0:dst_x1],
                    feather,
                )

            canvas[dst_y0:dst_y1, dst_x0:dst_x1] += tile_r * feather[:, :, None]
            weight_acc[dst_y0:dst_y1, dst_x0:dst_x1] += feather[:, :, None]

        # Normalise — safe even where weight==0 (black where no tile placed).
        safe_w = np.where(weight_acc > 0, weight_acc, 1.0)
        result = (canvas / safe_w).clip(0, 1).astype(np.float32)
        return (_np2t(result).unsqueeze(0),)


# ── TileInfoAM ─────────────────────────────────────────────────────────────────

class TileInfoAM:
    """
    Inspect a single tile's metadata for debugging and workflow validation.

    Connect tile_metadata from TileCropAM and set tile_index to examine
    any specific tile's source position, expected output size, and method info.
    Warnings are emitted if tile_index is out of range or if parameters look
    inconsistent (e.g. overlap > 50% of tile size).
    """

    CATEGORY = "AM/TileUpscale"
    RETURN_TYPES = ("STRING", "INT", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = (
        "info",
        "tile_index",
        "pos_x",
        "pos_y",
        "tile_width",
        "tile_height",
        "tile_count",
    )
    FUNCTION = "inspect"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tile_metadata": ("STRING", {"forceInput": True}),
                "tile_index": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
            },
        }

    def inspect(self, tile_metadata: str, tile_index: int):
        meta = json.loads(tile_metadata)
        tiles: list[dict] = meta["tiles"]
        n = len(tiles)
        warnings: list[str] = []

        # Clamp out-of-range index with a warning.
        if tile_index >= n:
            warnings.append(
                f"tile_index {tile_index} is out of range (tile_count={n}), clamped to {n - 1}"
            )
            tile_index = n - 1

        t = tiles[tile_index]
        preset = meta["preset"]

        # Sanity checks.
        ov_pct = meta["overlap_pct"]
        if ov_pct > 40:
            warnings.append(f"Overlap {ov_pct:.1f}% is very high — may cause artefacts.")
        if meta["grid_cols"] * meta["grid_rows"] > 64:
            warnings.append("Grid has more than 64 tiles — consider reducing for performance.")

        info = {
            "tile_index": tile_index,
            "row": t["row"],
            "col": t["col"],
            "source_region": {
                "x0": t["src_x0"],
                "y0": t["src_y0"],
                "x1": t["src_x1"],
                "y1": t["src_y1"],
            },
            "source_size_px": {"w": t["src_w"], "h": t["src_h"]},
            "uniform_tile_size_px": {
                "w": meta["uniform_tile_w"],
                "h": meta["uniform_tile_h"],
            },
            "method": meta["method"],
            "method_category": preset["category"],
            "method_changes_content": preset["changes_content"],
            "overlap_pct": ov_pct,
            "overlap_px": {"w": meta["overlap_w"], "h": meta["overlap_h"]},
            "grid": {"cols": meta["grid_cols"], "rows": meta["grid_rows"]},
            "source_canvas_px": {"w": meta["src_w"], "h": meta["src_h"]},
            "tile_count": n,
            "feather_mode": preset["feather_mode"],
            "color_match": preset["color_match"],
            "warnings": warnings,
        }

        return (
            json.dumps(info, indent=2),
            tile_index,
            t["src_x0"],
            t["src_y0"],
            t["src_w"],
            t["src_h"],
            n,
        )


# ── SaveImageWithDPI ───────────────────────────────────────────────────────────

def _write_png_dpi(path: str, img_pil: Image.Image, dpi: int) -> None:
    """
    Write a PNG file with a pHYs chunk encoding the given DPI.

    DPI metadata tells viewers / printers how to scale the image for display
    or print.  It does NOT add image detail — pixel count is unchanged.
    We inject the pHYs chunk manually after IHDR to guarantee it is present
    regardless of PIL version.
    """
    import io

    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    raw = buf.getvalue()

    # PNG signature is 8 bytes; IHDR chunk follows immediately.
    # Chunk format: [4-byte length][4-byte type][data][4-byte CRC]
    ihdr_data_len = struct.unpack(">I", raw[8:12])[0]
    ihdr_end = 8 + 4 + 4 + ihdr_data_len + 4   # sig + len + type + data + crc

    # pHYs payload: x_ppm (4), y_ppm (4), unit=1/metre (1) = 9 bytes.
    ppm = round(dpi / 0.0254)
    phys_payload = struct.pack(">IIB", ppm, ppm, 1)
    phys_crc = zlib.crc32(b"pHYs" + phys_payload) & 0xFFFFFFFF
    phys_chunk = (
        struct.pack(">I", len(phys_payload))
        + b"pHYs"
        + phys_payload
        + struct.pack(">I", phys_crc)
    )

    out = raw[:ihdr_end] + phys_chunk + raw[ihdr_end:]
    with open(path, "wb") as f:
        f.write(out)


class SaveImageWithDPI:
    """
    Save an image with embedded DPI metadata.

    IMPORTANT: DPI is metadata that tells viewers/printers the intended display
    size.  Changing DPI does NOT create new image detail.  A 1000×1000 image
    saved at 300 DPI will print at ~8.5cm × 8.5cm; saved at 72 DPI it prints
    at ~35cm × 35cm — same pixel count, different physical size interpretation.

    Supported formats and DPI support:
    - PNG:  pHYs chunk (lossless, always embedded).
    - TIFF: ResolutionUnit / XResolution / YResolution tags (lossless with LZW).
    - JPEG: APP0/JFIF DPI fields (lossy, quality-controlled).
    """

    CATEGORY = "AM/TileUpscale"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    OUTPUT_NODE = True
    FUNCTION = "save_image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "tile_upscale"}),
                "dpi": (
                    "INT",
                    {
                        "default": 300,
                        "min": 1,
                        "max": 2400,
                        "step": 1,
                        "tooltip": (
                            "DPI metadata only — does not resize or add detail. "
                            "Common values: 72 (screen), 150 (draft print), "
                            "300 (quality print), 600 (high-res print)."
                        ),
                    },
                ),
                "format": (["png", "tiff", "jpeg"], {"default": "png"}),
            },
            "optional": {
                "jpeg_quality": (
                    "INT",
                    {
                        "default": 95,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "tooltip": "JPEG only. Higher = better quality, larger file.",
                    },
                ),
                "output_subfolder": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Subfolder inside ComfyUI output directory. Empty = root output dir.",
                    },
                ),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def save_image(
        self,
        image: torch.Tensor,
        filename_prefix: str,
        dpi: int,
        format: str,
        jpeg_quality: int = 95,
        output_subfolder: str = "",
        prompt=None,
        extra_pnginfo=None,
    ):
        fmt = format  # avoid shadowing the built-in

        # Resolve output directory — fall back gracefully if folder_paths unavailable.
        try:
            import folder_paths
            base_dir = folder_paths.get_output_directory()
        except ImportError:
            base_dir = os.path.join(os.path.dirname(__file__), "output")

        out_dir = os.path.join(base_dir, output_subfolder) if output_subfolder else base_dir
        os.makedirs(out_dir, exist_ok=True)

        ext = {"png": ".png", "tiff": ".tif", "jpeg": ".jpg"}[fmt]

        # Find next available filename (non-overwriting).
        counter = 1
        while True:
            fname = f"{filename_prefix}_{counter:04d}{ext}"
            path = os.path.join(out_dir, fname)
            if not os.path.exists(path):
                break
            counter += 1

        # Convert tensor to PIL — first frame only; strip alpha if present.
        img_t = _t2np(image[0])
        if img_t.shape[2] == 4:
            img_t = img_t[:, :, :3]
        img_np = (img_t * 255).clip(0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np, "RGB")
        w, h = img_pil.size

        if fmt == "png":
            _write_png_dpi(path, img_pil, dpi)
        elif fmt == "tiff":
            img_pil.save(path, format="TIFF", dpi=(dpi, dpi), compression="lzw")
        elif fmt == "jpeg":
            img_pil.save(path, format="JPEG", quality=jpeg_quality, dpi=(dpi, dpi))

        print(f"[SaveImageWithDPI] {w}x{h}px @ {dpi} DPI -> {path}")

        # ComfyUI preview output.
        results = [{"filename": fname, "subfolder": output_subfolder, "type": "output"}]
        return {"ui": {"images": results}, "result": (path,)}


# ── Node registration (imported by __init__.py) ────────────────────────────────

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "TileCropAM": TileCropAM,
    "TileStitchAM": TileStitchAM,
    "TileInfoAM": TileInfoAM,
    "SaveImageWithDPI": SaveImageWithDPI,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "TileCropAM": "Tile Crop (AM)",
    "TileStitchAM": "Tile Stitch (AM)",
    "TileInfoAM": "Tile Info / Debug (AM)",
    "SaveImageWithDPI": "Save Image With DPI (AM)",
}
