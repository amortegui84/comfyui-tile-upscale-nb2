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
        "recommended_cols": 2,
        "recommended_rows": 2,
        "description": "Nano Banana 2 — generative reinterpretation, strong blending",
    },
    "image_2": {
        "category": "regenerative",
        "changes_content": True,
        "recommended_overlap_pct": 20.0,
        "feather_mode": "strong",
        "color_match": True,
        "recommended_cols": 2,
        "recommended_rows": 2,
        "description": "GPT Image 2 — generative reinterpretation, strong blending",
    },
    "topaz": {
        "category": "faithful",
        "changes_content": False,
        "recommended_overlap_pct": 8.0,
        "feather_mode": "minimal",
        "color_match": False,
        "recommended_cols": 2,
        "recommended_rows": 2,
        "description": "Topaz-style faithful upscale — minimal blending, preserve structure",
    },
    "seedv2": {
        "category": "faithful",
        "changes_content": False,
        "recommended_overlap_pct": 10.0,
        "feather_mode": "strong",
        "color_match": True,
        "recommended_cols": 3,
        "recommended_rows": 2,
        "description": "SeedVR2-style faithful upscale - 3x2 tiles, color match on",
    },
    "passthrough": {
        "category": "passthrough",
        "changes_content": False,
        "recommended_overlap_pct": 4.0,
        "feather_mode": "minimal",
        "color_match": False,
        "recommended_cols": 2,
        "recommended_rows": 2,
        "description": "Passthrough / external upscale — exact alignment, near-zero blending",
    },
    "custom": {
        "category": "custom",
        "changes_content": False,
        "recommended_overlap_pct": 12.0,
        "feather_mode": "medium",
        "color_match": False,
        "recommended_cols": 2,
        "recommended_rows": 2,
        "description": "Custom — user controls all parameters",
    },
}

# Keep old input names for saved workflow compatibility, but allow the layouts
# used by SeedVR2 tile workflows.
GRID_PRESETS: dict[str, tuple[int, int] | None] = {
    "method default": None,
    "fixed 2x2": (2, 2),
    "fixed 2×2": (2, 2),
    "3x2 horizontal": (3, 2),
    "2x3 vertical": (2, 3),
    "3x3": (3, 3),
}

# Smoothstep power per feather mode.
# Higher power = sharper transition concentrated near the boundary center.
# Lower power (1.0 = linear) = softer, wider transition.
_FEATHER_POWER = {"strong": 1.0, "medium": 2.0, "minimal": 4.0}
MAX_COLLECT_TILES = 24


def _normalize_grid_label(label: str) -> str:
    return str(label).strip().lower().replace("×", "x")


def _resolve_grid(method: str, grid_preset: str, grid_cols: int, grid_rows: int) -> tuple[int, int, str]:
    preset = METHOD_PRESETS[method]
    label = _normalize_grid_label(grid_preset)

    if label in ("method default", "auto"):
        cols = int(preset["recommended_cols"])
        rows = int(preset["recommended_rows"])
        policy = "method_default"
    elif label in ("fixed 2x2", "2x2"):
        cols, rows = 2, 2
        policy = "preset"
    elif label in ("3x2 horizontal", "3x2"):
        cols, rows = 3, 2
        policy = "preset"
    elif label in ("2x3 vertical", "2x3"):
        cols, rows = 2, 3
        policy = "preset"
    elif label in ("3x3", "9 tiles", "9_tiles"):
        cols, rows = 3, 3
        policy = "preset"
    else:
        cols, rows = int(grid_cols), int(grid_rows)
        policy = "custom"

    cols = max(1, min(cols, 8))
    rows = max(1, min(rows, 8))
    return cols, rows, policy


def _uniform_tile_size(src_size: int, count: int, overlap_fraction: float) -> int:
    if count <= 1:
        return src_size
    coverage = 1.0 + (count - 1) * (1.0 - overlap_fraction)
    return max(1, min(src_size, int(np.ceil(src_size / coverage))))


def _tile_starts(src_size: int, tile_size: int, count: int) -> list[int]:
    if count <= 1:
        return [0]
    max_start = max(0, src_size - tile_size)
    return [round(i * max_start / (count - 1)) for i in range(count)]


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


def _edge_feather_axis(
    n: int,
    overlap: int,
    power: float,
    fade_start: bool,
    fade_end: bool,
) -> np.ndarray:
    ramp = np.ones(n, dtype=np.float64)
    if overlap <= 0:
        return ramp
    overlap = min(overlap, n // 2)
    t = np.linspace(0.0, 1.0, overlap, endpoint=False)
    fade = np.power(t, 1.0 / power)
    if fade_start:
        ramp[:overlap] *= fade
    if fade_end:
        ramp[-overlap:] *= fade[::-1]
    return ramp


def _tile_feather_mask(
    h: int,
    w: int,
    ov_h: int,
    ov_w: int,
    power: float,
    row: int,
    col: int,
    rows: int,
    cols: int,
) -> np.ndarray:
    row_w = _edge_feather_axis(h, ov_h, power, row > 0, row < rows - 1)
    col_w = _edge_feather_axis(w, ov_w, power, col > 0, col < cols - 1)
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
                "grid_preset": (
                    list(GRID_PRESETS.keys()),
                    {
                        "default": "method default",
                        "tooltip": "Grid layout. Method default uses each method preset recommendation.",
                    },
                ),
                "grid_cols": ("INT", {"default": 3, "min": 1, "max": 8, "step": 1,
                                      "tooltip": "Used when grid_preset is custom/unknown."}),
                "grid_rows": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1,
                                      "tooltip": "Used when grid_preset is custom/unknown."}),
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
        grid_preset: str,
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

        requested_grid_cols = int(grid_cols)
        requested_grid_rows = int(grid_rows)
        requested_grid_preset = grid_preset

        grid_cols, grid_rows, tile_count_policy = _resolve_grid(
            method, grid_preset, grid_cols, grid_rows
        )

        eff_overlap = overlap_percent if overlap_percent >= 0.0 else preset["recommended_overlap_pct"]
        overlap_fraction = max(0.0, min(float(eff_overlap) / 100.0, 0.75))

        tile_w = _uniform_tile_size(src_w, grid_cols, overlap_fraction)
        tile_h = _uniform_tile_size(src_h, grid_rows, overlap_fraction)
        xs = _tile_starts(src_w, tile_w, grid_cols)
        ys = _tile_starts(src_h, tile_h, grid_rows)

        # Nominal overlap in pixels for feathering. Actual start positions are
        # stored per tile and used for exact placement during stitch.
        ov_w = max(0, round(tile_w * overlap_fraction))
        ov_h = max(0, round(tile_h * overlap_fraction))

        base_w = tile_w
        base_h = tile_h

        # Build tile crop regions.
        tile_infos: list[dict] = []
        crops: list[np.ndarray] = []

        for row in range(grid_rows):
            for col in range(grid_cols):
                x0 = xs[col]
                y0 = ys[row]
                x1 = min(src_w, x0 + tile_w)
                y1 = min(src_h, y0 + tile_h)

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
            "requested_grid_preset": requested_grid_preset,
            "requested_grid_cols": requested_grid_cols,
            "requested_grid_rows": requested_grid_rows,
            "grid_cols": grid_cols,
            "grid_rows": grid_rows,
            "tile_count_policy": tile_count_policy,
            "grid_preset": grid_preset,
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

class TileExtractAM:
    """
    Extract a single tile from the TileCropAM batch by row-major index.

    This is the bridge for one-image-per-call APIs such as NB2 or Image 2:
    TileCropAM -> TileExtractAM(index=N) -> API/upscaler -> TileCollectAM.
    """

    CATEGORY = "AM/TileUpscale"
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("tile", "tile_index")
    FUNCTION = "extract"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tiles": ("IMAGE",),
                "tile_index": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
            },
        }

    def extract(self, tiles: torch.Tensor, tile_index: int):
        if tiles.ndim != 4:
            raise ValueError(f"TileExtractAM expected IMAGE batch [N,H,W,C], got {tuple(tiles.shape)}")
        if tiles.shape[0] < 1:
            raise ValueError("TileExtractAM received an empty tile batch")
        idx = max(0, min(int(tile_index), tiles.shape[0] - 1))
        return (tiles[idx: idx + 1], idx)


class TileScaleByAM:
    """
    Scale an image by a fixed factor — placeholder for a real tile upscaler.

    Use this while setting up and testing workflow geometry. When the tiling and
    stitching look correct, replace this node with your actual upscaler (Topaz,
    SeedV2, NB2, GPT-Image-2, ESRGAN, etc.) — the rest of the workflow stays
    identical because TileStitchAM infers the scale factor automatically from
    the processed tile size.
    """

    CATEGORY = "AM/TileUpscale"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "scale_by"

    _PIL_METHODS = {
        "lanczos": Image.LANCZOS,
        "bicubic": Image.BICUBIC,
        "bilinear": Image.BILINEAR,
        "nearest": Image.NEAREST,
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale_factor": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.1,
                        "max": 16.0,
                        "step": 0.05,
                        "tooltip": (
                            "Multiply width and height by this factor. "
                            "2.0 = double resolution. Replace this node with "
                            "your real upscaler once the workflow geometry is confirmed."
                        ),
                    },
                ),
                "upscale_method": (
                    ["lanczos", "bicubic", "bilinear", "nearest"],
                    {"default": "lanczos"},
                ),
            },
        }

    def scale_by(self, image: torch.Tensor, scale_factor: float, upscale_method: str):
        pil_method = self._PIL_METHODS[upscale_method]
        results: list[torch.Tensor] = []
        for i in range(image.shape[0]):
            np_img = _t2np(image[i])
            H, W = np_img.shape[:2]
            new_w = max(1, round(W * scale_factor))
            new_h = max(1, round(H * scale_factor))
            pil_resized = _np_to_pil(np_img).resize((new_w, new_h), pil_method)
            results.append(_np2t(_pil_to_np(pil_resized)))
        return (torch.stack(results, dim=0),)


class TileSeedVR2ControlsAM:
    """
    Reusable decimal controls for SeedVR2 tile workflows.

    Connect upscale_factor and noise_scale to every SeedVR2 tile node so all
    tiles use the same values.
    """

    CATEGORY = "AM/TileUpscale"
    RETURN_TYPES = ("FLOAT", "FLOAT")
    RETURN_NAMES = ("upscale_factor", "noise_scale")
    FUNCTION = "controls"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscale_factor": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.1,
                        "max": 16.0,
                        "step": 0.05,
                        "tooltip": "SeedVR2 upscale factor shared by all tile nodes.",
                    },
                ),
                "noise_scale": (
                    "FLOAT",
                    {
                        "default": 0.15,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "SeedVR2 noise scale shared by all tile nodes.",
                    },
                ),
            },
        }

    def controls(self, upscale_factor: float, noise_scale: float):
        return (float(upscale_factor), float(noise_scale))


class TileCollectAM:
    """
    Collect individually processed tiles back into one IMAGE batch.

    Connect tiles in the same row-major order emitted by TileCropAM:
    2x2 -> 0,1,2,3; 3x3 -> 0..8. All connected tiles are resized to tile_0's
    processed size so TileStitchAM receives a valid batch tensor.

    tile_metadata is passed through unchanged so TileStitchAM can be wired
    from this node instead of directly from TileCropAM, keeping the pipeline
    linear: CropAM -> ... -> CollectAM -> StitchAM.
    """

    CATEGORY = "AM/TileUpscale"
    RETURN_TYPES = ("IMAGE", "INT", "STRING", "STRING")
    RETURN_NAMES = ("tiles", "tile_count", "info", "tile_metadata")
    FUNCTION = "collect"

    @classmethod
    def INPUT_TYPES(cls):
        optional = {"tile_metadata": ("STRING", {"forceInput": True})}
        optional.update({f"tile_{i}": ("IMAGE",) for i in range(1, MAX_COLLECT_TILES)})
        return {
            "required": {
                "tile_0": ("IMAGE",),
            },
            "optional": optional,
        }

    def collect(self, tile_0: torch.Tensor, tile_metadata: str = "", **kwargs):
        if tile_0.ndim != 4 or tile_0.shape[0] < 1:
            raise ValueError(f"TileCollectAM expected tile_0 as IMAGE [B,H,W,C], got {tuple(tile_0.shape)}")

        connected: list[torch.Tensor] = [tile_0[:1]]
        missing_before_connected = False
        seen_gap = False

        for idx in range(1, MAX_COLLECT_TILES):
            tile = kwargs.get(f"tile_{idx}")
            if tile is None:
                seen_gap = True
                continue
            if seen_gap:
                missing_before_connected = True
            if tile.ndim != 4 or tile.shape[0] < 1:
                raise ValueError(f"TileCollectAM expected tile_{idx} as IMAGE [B,H,W,C], got {tuple(tile.shape)}")
            connected.append(tile[:1])

        expected = None
        if tile_metadata:
            try:
                meta = json.loads(tile_metadata)
                expected = len(meta.get("tiles", []))
            except Exception:
                expected = None

        warnings: list[str] = []
        if expected is not None and len(connected) > expected:
            warnings.append(f"ignored {len(connected) - expected} extra tile input(s) beyond metadata count")
            connected = connected[:expected]

        ref_h, ref_w = connected[0].shape[1], connected[0].shape[2]
        normalized: list[torch.Tensor] = []
        resized_count = 0
        for tile in connected:
            one = tile[0]
            if one.shape[0] != ref_h or one.shape[1] != ref_w:
                one = _np2t(_resize_np(_t2np(one), ref_w, ref_h))
                resized_count += 1
            normalized.append(one.unsqueeze(0))

        if expected is not None and expected != len(normalized):
            warnings.append(f"metadata expects {expected} tiles but TileCollectAM received {len(normalized)}")
        if missing_before_connected:
            warnings.append("non-contiguous tile inputs detected; connect tiles in row-major order without gaps")
        if resized_count:
            warnings.append(f"resized {resized_count} tile(s) to match tile_0 processed size")

        info = {
            "tile_count": len(normalized),
            "expected_tile_count": expected,
            "tile_size": {"w": ref_w, "h": ref_h},
            "warnings": warnings,
        }

        return (torch.cat(normalized, dim=0), len(normalized), json.dumps(info, indent=2), tile_metadata)


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
                "upscale_factor": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 16.0,
                        "step": 0.05,
                        "tooltip": "Optional explicit output scale. 0 = infer from processed tile size.",
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
        upscale_factor: float = 0.0,
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
        grid_cols: int = meta.get("grid_cols", 1)
        grid_rows: int = meta.get("grid_rows", 1)

        n_tiles = tiles.shape[0]
        if n_tiles != len(tile_infos):
            raise ValueError(
                f"TileStitchAM: received {n_tiles} tiles but metadata describes "
                f"{len(tile_infos)}. Check your tile count."
            )

        # Determine upscale factor from processed tile size vs uniform tile size,
        # unless an explicit workflow factor is connected for deterministic sizing.
        proc_h, proc_w = tiles.shape[1], tiles.shape[2]
        if upscale_factor > 0.0:
            scale_x = scale_y = float(upscale_factor)
        else:
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

            # Build an edge-aware feather mask. Image-boundary edges stay fully
            # weighted; only edges with neighboring tiles fade into overlaps.
            feather = _tile_feather_mask(
                dst_h,
                dst_w,
                ov_h_s,
                ov_w_s,
                power,
                int(tinfo.get("row", 0)),
                int(tinfo.get("col", 0)),
                grid_rows,
                grid_cols,
            )

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
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("saved_path", "print_size_info")
    OUTPUT_NODE = True
    FUNCTION = "save_image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "tile_upscale"}),
                "dpi": (
                    ["72", "150", "300", "600"],
                    {
                        "default": "300",
                        "tooltip": (
                            "DPI metadata only — does not resize or add detail. "
                            "72=screen/web, 150=draft print, 300=quality print, 600=high-res print."
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
        dpi: str,
        format: str,
        jpeg_quality: int = 95,
        output_subfolder: str = "",
        prompt=None,
        extra_pnginfo=None,
    ):
        dpi_val = int(dpi)
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
            _write_png_dpi(path, img_pil, dpi_val)
        elif fmt == "tiff":
            img_pil.save(path, format="TIFF", dpi=(dpi_val, dpi_val), compression="lzw")
        elif fmt == "jpeg":
            img_pil.save(path, format="JPEG", quality=jpeg_quality, dpi=(dpi_val, dpi_val))

        # Physical print size (DPI is metadata only — pixel count is unchanged).
        w_in = w / dpi_val
        h_in = h / dpi_val
        w_cm = w_in * 2.54
        h_cm = h_in * 2.54
        size_info = (
            f"{w}×{h} px  |  {dpi_val} DPI\n"
            f"Print size: {w_in:.1f}\" × {h_in:.1f}\"  ({w_cm:.1f} cm × {h_cm:.1f} cm)\n"
            f"Saved: {path}"
        )
        print(f"[SaveImageWithDPI] {size_info}")

        # ComfyUI preview output.
        results = [{"filename": fname, "subfolder": output_subfolder, "type": "output"}]
        return {"ui": {"images": results}, "result": (path, size_info)}


# ── TileCostReporterAM ─────────────────────────────────────────────────────────

class TileCostReporterAM:
    """
    Reports estimated API cost and timing for tiled upscale workflows.

    Fal/SeedVR2 charges per output megapixel per API call.  In a tile workflow
    each tile is a separate API call, so a 3×3 grid = 9 calls.  Overlap means
    the total billed megapixels exceed the final stitched image megapixels.
    """

    CATEGORY = "AM/TileUpscale"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "generate_report"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tile_metadata": ("STRING", {"forceInput": True}),
                "stitched_image": ("IMAGE",),
                "tile_count": ("INT", {"default": 0, "min": 0, "max": 256, "forceInput": True}),
            },
            "optional": {
                "upscale_factor": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 16.0,
                        "step": 0.1,
                        "tooltip": "0 = infer from stitched image vs source dimensions.",
                    },
                ),
                "price_per_megapixel": (
                    "FLOAT",
                    {
                        "default": 0.001,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.0001,
                        "tooltip": "Provider cost per output megapixel. Fal/SeedVR2 default is $0.001.",
                    },
                ),
                "provider_name": (
                    "STRING",
                    {"default": "fal.ai / SeedVR2", "multiline": False},
                ),
                "include_server_estimate": (
                    ["false", "true"],
                    {
                        "default": "false",
                        "tooltip": "Include local server/GPU compute cost in the total estimate.",
                    },
                ),
                "server_cost_per_hour_usd": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1000.0,
                        "step": 0.01,
                        "tooltip": "Your server/GPU cost per hour in USD.",
                    },
                ),
                "elapsed_seconds": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 86400.0,
                        "step": 1.0,
                        "tooltip": "Total generation time in seconds. 0 = not provided.",
                    },
                ),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def generate_report(
        self,
        tile_metadata: str,
        stitched_image: torch.Tensor,
        tile_count: int,
        upscale_factor: float = 0.0,
        price_per_megapixel: float = 0.001,
        provider_name: str = "fal.ai / SeedVR2",
        include_server_estimate: str = "false",
        server_cost_per_hour_usd: float = 0.0,
        elapsed_seconds: float = 0.0,
    ) -> dict:
        include_server = include_server_estimate == "true"
        warn: list[str] = []

        # ── Parse metadata ──────────────────────────────────────────────────────
        try:
            meta = json.loads(tile_metadata)
        except (json.JSONDecodeError, TypeError):
            err = "ERROR: Could not parse tile_metadata JSON."
            return {"ui": {"text": [err]}, "result": (err,)}

        grid_cols: int = meta.get("grid_cols", 0)
        grid_rows: int = meta.get("grid_rows", 0)
        meta_tile_count: int = len(meta.get("tiles", []))
        src_w: int = meta.get("src_w", 0)
        uniform_tile_w: int = meta.get("uniform_tile_w", 0)
        uniform_tile_h: int = meta.get("uniform_tile_h", 0)
        overlap_pct: float = meta.get("overlap_pct", 0.0)

        # ── Stitched image dimensions ──────────────────────────────────────────
        # Tensor shape: [B, H, W, C]
        _, stitch_h, stitch_w, _ = stitched_image.shape
        final_mp = (stitch_w * stitch_h) / 1_000_000.0

        # ── Resolve upscale factor ─────────────────────────────────────────────
        upscale_inferred = False
        if upscale_factor <= 0.0:
            if src_w > 0 and stitch_w > 0:
                upscale_factor = stitch_w / src_w
                upscale_inferred = True
            else:
                upscale_factor = 1.0
                warn.append("Could not infer upscale factor — defaulted to 1.0.")

        # ── Tile count validation ─────────────────────────────────────────────
        if meta_tile_count > 0 and tile_count != meta_tile_count:
            warn.append(
                f"tile_count ({tile_count}) differs from metadata grid count "
                f"({meta_tile_count}). Using tile_count for cost."
            )
        effective_calls = tile_count if tile_count > 0 else meta_tile_count

        # ── Per-tile billed megapixels ─────────────────────────────────────────
        # Fal bills per OUTPUT megapixel per API call.
        # Each tile output = source_tile_px × upscale_factor (per axis).
        tile_size_note = "from metadata"
        if uniform_tile_w > 0 and uniform_tile_h > 0:
            proc_tile_w = int(uniform_tile_w * upscale_factor)
            proc_tile_h = int(uniform_tile_h * upscale_factor)
            mp_per_tile = (proc_tile_w * proc_tile_h) / 1_000_000.0
        elif grid_cols > 0 and grid_rows > 0 and stitch_w > 0 and stitch_h > 0:
            # Fallback: divide final image by grid without overlap correction.
            mp_per_tile = final_mp / (grid_cols * grid_rows)
            proc_tile_w = int(stitch_w / grid_cols)
            proc_tile_h = int(stitch_h / grid_rows)
            tile_size_note = "estimated (no tile dims in metadata)"
            warn.append(
                "Tile dimensions not in metadata — tile MP estimated from final "
                "image size. Actual billed MP may be higher due to overlap."
            )
        else:
            mp_per_tile = final_mp
            proc_tile_w = stitch_w
            proc_tile_h = stitch_h
            tile_size_note = "estimated (fallback)"
            warn.append("Could not determine tile dimensions — cost is approximate.")

        total_billed_mp = mp_per_tile * effective_calls

        # ── Validate price ───────────────────────────────────────────────────
        if price_per_megapixel <= 0.0:
            warn.append(
                f"price_per_megapixel ({price_per_megapixel}) is invalid — using $0.001."
            )
            price_per_megapixel = 0.001

        # ── Cost calculation ──────────────────────────────────────────────────
        fal_cost = total_billed_mp * price_per_megapixel

        server_cost = 0.0
        if include_server and server_cost_per_hour_usd > 0.0 and elapsed_seconds > 0.0:
            server_cost = (elapsed_seconds / 3600.0) * server_cost_per_hour_usd

        total_cost = fal_cost + server_cost

        # ── Build report ──────────────────────────────────────────────────────
        sep = "─" * 50
        uf_tag = f"{upscale_factor:.2f}x" + (" (inferred)" if upscale_inferred else "")
        provider_short = provider_name.split("/")[0].strip()

        lines: list[str] = [sep, "  Tile Cost & Runtime Report", sep]
        lines += [
            f"Final output:           {stitch_w} x {stitch_h} px",
            f"Final megapixels:       {final_mp:.2f} MP",
            f"Grid:                   {grid_cols} x {grid_rows}",
            f"External calls:         {effective_calls}",
            f"Upscale factor:         {uf_tag}",
            f"Overlap:                {overlap_pct:.1f}%",
            "",
            f"Provider:               {provider_name}",
            f"Price:                  ${price_per_megapixel:.4f} / MP",
            "",
            f"Tile size (source):     {uniform_tile_w} x {uniform_tile_h} px",
            f"Tile size (processed):  {proc_tile_w} x {proc_tile_h} px  [{tile_size_note}]",
            f"MP per tile (billed):   {mp_per_tile:.4f} MP",
            f"Total billed MP:        {total_billed_mp:.4f} MP",
            f"  = {mp_per_tile:.4f} MP/tile × {effective_calls} calls",
            "",
            f"Estimated {provider_short} cost:  ${fal_cost:.4f}",
        ]

        if include_server:
            if server_cost > 0.0:
                lines.append(
                    f"Server compute est.:    ${server_cost:.4f}"
                    f"  ({elapsed_seconds:.0f}s ÷ 3600 × ${server_cost_per_hour_usd:.2f}/hr)"
                )
            else:
                lines.append("Server compute est.:    Not provided")

        lines.append(f"Total estimated cost:   ${total_cost:.4f}")
        lines.append("")

        if elapsed_seconds > 0.0:
            mins = int(elapsed_seconds // 60)
            secs = elapsed_seconds - mins * 60
            lines.append(f"Elapsed time:           {mins}m {secs:.1f}s ({elapsed_seconds:.1f}s total)")
            if effective_calls > 0:
                lines.append(f"Avg time / tile:        {elapsed_seconds / effective_calls:.1f}s")
        else:
            lines.append("Elapsed time:           Not provided")
            lines.append("Avg time / tile:        Not provided")

        if warn:
            lines.append("")
            lines.append("Warnings:")
            for w in warn:
                lines.append(f"  ! {w}")

        lines += [
            "",
            "Note: Cost is based on processed tile outputs (including overlap),",
            "      not only final stitched dimensions. Overlap increases billed MP.",
            sep,
        ]

        report = "\n".join(lines)
        print(f"[TileCostReporterAM]\n{report}")
        return {"ui": {"text": [report]}, "result": (report,)}


# ── ShowTextAM ─────────────────────────────────────────────────────────────────

class ShowTextAM:
    """
    Display a text string inside the node — useful for showing print_size_info
    from SaveImageWithDPI at the end of the workflow.
    """

    CATEGORY = "AM/TileUpscale"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "show_text"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def show_text(self, text: str):
        return {"ui": {"text": [text]}, "result": (text,)}


# ── Node registration (imported by __init__.py) ────────────────────────────────

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "TileCropAM": TileCropAM,
    "TileExtractAM": TileExtractAM,
    "TileScaleByAM": TileScaleByAM,
    "TileSeedVR2ControlsAM": TileSeedVR2ControlsAM,
    "TileCollectAM": TileCollectAM,
    "TileStitchAM": TileStitchAM,
    "TileInfoAM": TileInfoAM,
    "SaveImageWithDPI": SaveImageWithDPI,
    "ShowTextAM": ShowTextAM,
    "TileCostReporterAM": TileCostReporterAM,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "TileCropAM": "1. Tile Crop (AM)",
    "TileExtractAM": "2. Tile Extract (AM)",
    "TileScaleByAM": "3. Tile Scale By / Placeholder (AM)",
    "TileSeedVR2ControlsAM": "SeedVR2 Factor / Noise Controls (AM)",
    "TileCollectAM": "4. Tile Collect (AM)",
    "TileStitchAM": "5. Tile Stitch (AM)",
    "TileInfoAM": "Tile Info / Debug (AM)",
    "ShowTextAM": "Show Text (AM)",
    "SaveImageWithDPI": "Save Image With DPI (AM)",
    "TileCostReporterAM": "Tile Cost & Runtime Info (AM)",
}
