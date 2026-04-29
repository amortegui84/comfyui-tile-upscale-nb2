"""
ComfyUI Tile Upscale — Universal Edition v2.0

Multi-model (NB2, ChatGPT-Image-2, DALL-E-3, SDXL, SD15, custom, passthrough)
N×M grid (1×1 … 4×4), partition-of-unity smootherstep blending.
Full backward-compatible: TileCropNB2 / TileStitchNB2 preserve v1 output positions.

Bugs fixed vs first draft:
  • TileCropNB2 output order restored (tiles_batch moved to last slot, not slot 1)
  • _weight_1d uses multiplicative feathering → no silent overwrite for middle tiles
  • _tile_positions always returns exactly `count` positions (fixes 3×3, 4×4)
  • _get_target_res finds closest AR rather than blindly falling back to 1:1
  • NanoBanana2/Tiles category preserved on legacy nodes for search compat
"""
import math
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple

# ─── Resolution presets ───────────────────────────────────────────────────────
# (width, height).  All values divisible by 8 so every diffusion model accepts them.

RESOLUTION_PRESETS: Dict[str, Dict] = {

    # ── Nano Banana 2 ─────────────────────────────────────────────────────────
    "NB2": {
        "1:1":  {"1K": (1024, 1024), "2K": (2048, 2048), "4K": (4096, 4096)},
        "16:9": {"1K": (1376, 768),  "2K": (2752, 1536), "4K": (5504, 3072)},
        "9:16": {"1K": (768,  1376), "2K": (1536, 2752), "4K": (3072, 5504)},
        "4:5":  {"1K": (928,  1152), "2K": (1856, 2304), "4K": (3712, 4608)},
        "5:4":  {"1K": (1152, 928),  "2K": (2304, 1856), "4K": (4608, 3712)},
        "3:4":  {"1K": (768,  1024), "2K": (1536, 2048), "4K": (3072, 4096)},
        "4:3":  {"1K": (1024, 768),  "2K": (2048, 1536), "4K": (4096, 3072)},
        "2:3":  {"1K": (832,  1216), "2K": (1664, 2432), "4K": (3328, 4864)},
        "3:2":  {"1K": (1216, 832),  "2K": (2432, 1664), "4K": (4864, 3328)},
        "21:9": {"1K": (1536, 640),  "2K": (3072, 1280), "4K": (6144, 2560)},
        "9:21": {"1K": (640,  1536), "2K": (1280, 3072), "4K": (2560, 6144)},
    },

    # ── ChatGPT Image 2  (gpt-image-1 via OpenAI images/edits API) ────────────
    # Tiles sent to this model must use one of these exact sizes.
    "ChatGPT-Image-2": {
        "1:1":  {"1K": (1024, 1024)},
        "16:9": {"HD": (1536, 1024)},
        "9:16": {"HD": (1024, 1536)},
    },

    # ── DALL-E 3 ──────────────────────────────────────────────────────────────
    "DALL-E-3": {
        "1:1":  {"standard": (1024, 1024)},
        "16:9": {"standard": (1792, 1024)},
        "9:16": {"standard": (1024, 1792)},
    },

    # ── SDXL training-bucket resolutions ──────────────────────────────────────
    "SDXL": {
        "1:1":  {"1K": (1024, 1024)},
        "16:9": {"std": (1344, 768)},
        "9:16": {"std": (768,  1344)},
        "4:3":  {"std": (1152, 896)},
        "3:4":  {"std": (896,  1152)},
        "4:5":  {"std": (896,  1152)},
        "5:4":  {"std": (1152, 896)},
        "3:2":  {"std": (1216, 832)},
        "2:3":  {"std": (832,  1216)},
        "21:9": {"std": (1536, 640)},
        "9:21": {"std": (640,  1536)},
    },

    # ── Stable Diffusion 1.5 ──────────────────────────────────────────────────
    "SD15": {
        "1:1":  {"512": (512, 512),  "768": (768, 768)},
        "16:9": {"512": (768, 448),  "768": (912, 512)},
        "9:16": {"512": (448, 768),  "768": (512, 912)},
        "4:3":  {"512": (680, 512),  "768": (768, 576)},
        "3:4":  {"512": (512, 680),  "768": (576, 768)},
        "3:2":  {"512": (768, 512),  "768": (768, 512)},
        "2:3":  {"512": (512, 768),  "768": (512, 768)},
    },
}

# Canonical w/h ratios — used for auto-detect and closest-match fallback.
_AR_VALUES: Dict[str, float] = {
    "1:1":  1.0,
    "16:9": 16 / 9,
    "9:16": 9  / 16,
    "4:5":  4  / 5,
    "5:4":  5  / 4,
    "4:3":  4  / 3,
    "3:4":  3  / 4,
    "3:2":  3  / 2,
    "2:3":  2  / 3,
    "21:9": 21 / 9,
    "9:21": 9  / 21,
}

_INTERP = {"bilinear": "bilinear", "bicubic": "bicubic", "nearest": "nearest"}

GRID_OPTIONS = [
    "1×1",
    "1×2", "2×1",
    "2×2",
    "2×3", "3×2",
    "3×3",
    "3×4", "4×3",
    "4×4",
    "2×4", "4×2",
]
BLEND_MODES = ["smootherstep", "smoothstep", "cosine", "linear"]


# ─── Blend / ramp functions ───────────────────────────────────────────────────

def _smoothstep(t: torch.Tensor) -> torch.Tensor:
    """C1 continuous (Ken Perlin original)."""
    return t * t * (3.0 - 2.0 * t)

def _smootherstep(t: torch.Tensor) -> torch.Tensor:
    """C2 continuous quintic — better seam suppression, recommended default."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

def _cosine_ramp(t: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 - torch.cos(math.pi * t))

def _linear_ramp(t: torch.Tensor) -> torch.Tensor:
    return t

_BLEND_FN = {
    "smootherstep": _smootherstep,
    "smoothstep":   _smoothstep,
    "cosine":       _cosine_ramp,
    "linear":       _linear_ramp,
}


# ─── Core tensor helpers ──────────────────────────────────────────────────────

def _resize(img: torch.Tensor, th: int, tw: int, mode: str) -> torch.Tensor:
    """[H, W, C]  →  [th, tw, C]"""
    x  = img.permute(2, 0, 1).unsqueeze(0).float()
    kw = {} if mode == "nearest" else {"align_corners": False}
    x  = F.interpolate(x, size=(th, tw), mode=mode, **kw)
    return x.squeeze(0).permute(1, 2, 0)


def _resize_mask(mask: torch.Tensor, th: int, tw: int, mode: str = "nearest") -> torch.Tensor:
    """[B, H, W] -> [B, th, tw]"""
    x = mask.unsqueeze(1).float()
    kw = {} if mode == "nearest" else {"align_corners": False}
    x = F.interpolate(x, size=(th, tw), mode=mode, **kw)
    return x.squeeze(1)


def _coerce_mask_batch(mask: torch.Tensor, batch_size: int, target_h: int, target_w: int) -> Tuple[torch.Tensor, str]:
    """
    Normalise common Comfy/Florence mask layouts to [B, H, W].
    """
    shape = tuple(mask.shape)
    note = f"input mask shape {shape}"

    if mask.ndim == 2:
        out = mask.unsqueeze(0)
    elif mask.ndim == 3:
        if mask.shape[-1] in (1, 3, 4) and (mask.shape[0] != batch_size or mask.shape[1] != target_h):
            out = mask[..., 0].unsqueeze(0)
            note += " -> interpreted as [H,W,C]"
        else:
            out = mask
            note += " -> interpreted as [B,H,W]"
    elif mask.ndim == 4:
        if mask.shape[-1] in (1, 3, 4):
            out = mask[..., 0]
            note += " -> interpreted as [B,H,W,C]"
        elif mask.shape[1] in (1, 3, 4):
            out = mask[:, 0, :, :]
            note += " -> interpreted as [B,C,H,W]"
        else:
            raise ValueError(f"Unsupported MASK tensor layout: {shape}")
    else:
        raise ValueError(f"Unsupported MASK tensor rank: {mask.ndim}")

    source_batch = out.shape[0]
    if source_batch == 1 and batch_size > 1:
        out = out.repeat(batch_size, 1, 1)
        note += f" | repeated to image batch {batch_size}"
    elif source_batch != batch_size:
        out = out[:1].repeat(batch_size, 1, 1)
        note += f" | batch mismatch fixed from {source_batch} to {batch_size}"

    return out.float(), note


def _weight_1d(length: int,
               has_interior_start: bool,
               has_interior_end: bool,
               overlap_px: int,
               device: torch.device,
               blend_mode: str = "smootherstep") -> torch.Tensor:
    """
    1-D feather weight for one tile axis.

    Logic:
      • has_interior_start → left/top edge touches a neighbour  → ramp 0→1
      • has_interior_end   → right/bottom edge touches a neighbour → ramp 1→0
      • Image-boundary edges (non-interior) always get weight = 1.

    Implementation: compute left-weight and right-weight independently, then
    MULTIPLY them.  This handles the edge case where both feather zones overlap
    (heavy overlap on a narrow tile) without silent overwrites.

    Partition-of-unity property is maintained via the canvas/w_acc normalisation
    in TileStitch, so absolute weight magnitudes don't need to be exactly 1.
    """
    ramp_fn = _BLEND_FN.get(blend_mode, _smootherstep)

    w_left = torch.ones(length, device=device)
    if has_interior_start and overlap_px > 0:
        t = torch.linspace(0.0, 1.0, overlap_px, device=device)
        w_left[:overlap_px] = ramp_fn(t)            # 0 → 1

    w_right = torch.ones(length, device=device)
    if has_interior_end and overlap_px > 0:
        t = torch.linspace(0.0, 1.0, overlap_px, device=device)
        w_right[length - overlap_px:] = 1.0 - ramp_fn(t)   # 1 → 0

    return w_left * w_right


# ─── Geometry helpers ─────────────────────────────────────────────────────────

def _detect_ar(w: int, h: int) -> str:
    r = w / h
    return min(_AR_VALUES, key=lambda ar: abs(_AR_VALUES[ar] - r))


def _parse_grid(grid_str: str) -> Tuple[int, int]:
    for sep in ("×", "x", "X"):
        if sep in grid_str:
            a, b = grid_str.split(sep)
            return int(a), int(b)
    return 2, 2


def _compute_tile_dims(orig_w: int, orig_h: int,
                        target_w: int, target_h: int,
                        cols: int, rows: int,
                        overlap: float) -> Tuple[int, int]:
    """
    Find tile_w × tile_h such that:
      1. tile_w / tile_h  ==  target_w / target_h   (isotropic, no deformation)
      2. N tiles across orig_w with `overlap` fraction of tile_w shared between pairs
      3. M tiles down   orig_h with `overlap` fraction of tile_h shared between pairs

    Derivation for axis X with N tiles:
      stride = tile_w * (1 - overlap)
      total  = tile_w + (N-1)*stride = tile_w * (1 + (N-1)*(1-overlap))  ≥  orig_w
      → tile_w ≥ orig_w / (1 + (N-1)*(1-overlap))
    """
    k = target_w / target_h

    tw_from_x = orig_w / (1.0 + (cols - 1) * (1.0 - overlap)) if cols > 1 else float(orig_w)
    tw_from_y = (orig_h / (1.0 + (rows - 1) * (1.0 - overlap))) * k if rows > 1 else float(orig_h) * k

    tw = max(tw_from_x, tw_from_y)
    th = tw / k

    # Clamp to image bounds while preserving ratio
    if tw > orig_w:
        tw, th = float(orig_w), float(orig_w) / k
    if th > orig_h:
        th, tw = float(orig_h), float(orig_h) * k

    return round(tw), round(th)


def _tile_positions(orig_size: int, tile_size: int, count: int) -> List[int]:
    """
    Returns exactly `count` evenly-distributed start positions.
      pos[0]  = 0
      pos[-1] = max(0, orig_size - tile_size)   ← last tile ends at orig_size
    Always returns `count` items — even when count=1 or tile_size >= orig_size.
    """
    if count == 1:
        return [0]
    max_start = max(0, orig_size - tile_size)
    return [round(i * max_start / (count - 1)) for i in range(count)]


def _get_target_res(model_preset: str, ar: str, tier: str,
                    custom_w: int, custom_h: int) -> Tuple[int, int]:
    """
    Resolve (target_w, target_h) from preset + AR + tier.
    Falls back gracefully:
      1. Exact AR + tier match
      2. Exact AR, different tier
      3. Closest AR in preset (by ratio distance)
      4. custom_w × custom_h
    """
    if model_preset == "custom":
        return custom_w, custom_h
    if model_preset == "passthrough":
        return 0, 0          # sentinel: no scaling

    preset = RESOLUTION_PRESETS.get(model_preset)
    if not preset:
        return custom_w, custom_h

    ar_tbl = preset.get(ar)

    if not ar_tbl:
        # Find the closest aspect ratio available in this preset
        target_ratio = _AR_VALUES.get(ar, 1.0)
        closest_ar   = min(preset.keys(),
                           key=lambda k: abs(_AR_VALUES.get(k, 1.0) - target_ratio))
        ar_tbl = preset[closest_ar]

    if tier in ar_tbl:
        return ar_tbl[tier]
    # Tier not found — return first available tier for this AR
    return next(iter(ar_tbl.values()))


# ─── Universal crop node ──────────────────────────────────────────────────────

class TileCrop:
    """
    Splits an image into N×M overlapping tiles, each optionally scaled to a
    model-specific target resolution.

    Output `tiles` is an IMAGE batch of shape [N*M, H, W, C] in row-major
    order (left→right, top→bottom).  Connect directly to any model that accepts
    an IMAGE batch, OR use TileExtract to pull out individual tiles for
    workflows that process one tile at a time (e.g. NB2 with a reference image).

    Model presets
    ─────────────
    NB2            → Nano Banana 2 resolutions
    ChatGPT-Image-2→ gpt-image-1 (1024×1024 / 1536×1024 / 1024×1536)
    DALL-E-3       → 1024×1024 / 1792×1024 / 1024×1792
    SDXL           → SDXL training bucket sizes
    SD15           → SD 1.5 base resolutions
    custom         → enter custom_w × custom_h directly
    passthrough    → no scaling; tiles out at native crop size
                     (for Topaz AI, SeedV2, or any local upscaler)

    Grid guide
    ──────────
    2×2 (4 tiles)  → good for 2-4× upscale of typical images
    3×3 (9 tiles)  → more detail coverage, larger effective upscale
    4×4 (16 tiles) → maximum detail, high VRAM requirement
    """

    CATEGORY      = "TileUpscale"
    RETURN_TYPES  = ("TILE_STITCHER", "IMAGE", "STRING", "STRING")
    RETURN_NAMES  = ("tile_stitcher", "tiles", "aspect_ratio", "tile_info")
    FUNCTION      = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        presets = list(RESOLUTION_PRESETS.keys()) + ["custom", "passthrough"]
        tiers   = ["1K", "2K", "4K", "HD", "standard", "std", "512", "768"]
        ars     = ["auto"] + list(_AR_VALUES.keys())
        return {
            "required": {
                "image":           ("IMAGE",),
                "model_preset":    (presets,      {"default": "NB2"}),
                "grid":            (GRID_OPTIONS, {"default": "2×2"}),
                "aspect_ratio":    (ars,           {"default": "auto"}),
                "resolution_tier": (tiers,         {"default": "2K"}),
                "overlap":         ("FLOAT", {"default": 0.15, "min": 0.05,
                                              "max": 0.45, "step": 0.01}),
                "scale_algo":      (list(_INTERP), {"default": "bicubic"}),
                "blend_mode":      (BLEND_MODES,   {"default": "smootherstep"}),
                "device_mode":     (["gpu (much faster)", "cpu"],
                                    {"default": "gpu (much faster)"}),
            },
            "optional": {
                "custom_w": ("INT", {"default": 1024, "min": 64,
                                     "max": 16384,    "step": 8}),
                "custom_h": ("INT", {"default": 1024, "min": 64,
                                     "max": 16384,    "step": 8}),
            },
        }

    def execute(self, image, model_preset, grid, aspect_ratio, resolution_tier,
                overlap, scale_algo, blend_mode, device_mode,
                custom_w=1024, custom_h=1024):

        use_gpu = device_mode.startswith("gpu") and torch.cuda.is_available()
        device  = torch.device("cuda" if use_gpu else "cpu")
        mode    = _INTERP[scale_algo]

        img    = image[0].to(device)
        orig_h, orig_w, C = img.shape

        cols, rows  = _parse_grid(grid)
        ar          = _detect_ar(orig_w, orig_h) if aspect_ratio == "auto" else aspect_ratio
        passthrough = model_preset == "passthrough"
        target_w, target_h = _get_target_res(model_preset, ar, resolution_tier,
                                              custom_w, custom_h)

        # ── Tile crop dimensions ──────────────────────────────────────────────
        if passthrough:
            # No AR constraint; guarantee coverage with ceil.
            tw = math.ceil(orig_w / (1.0 + (cols - 1) * (1.0 - overlap))) if cols > 1 else orig_w
            th = math.ceil(orig_h / (1.0 + (rows - 1) * (1.0 - overlap))) if rows > 1 else orig_h
        else:
            tw, th = _compute_tile_dims(orig_w, orig_h, target_w, target_h,
                                         cols, rows, overlap)

        xs = _tile_positions(orig_w, tw, cols)
        ys = _tile_positions(orig_h, th, rows)

        tiles_list: List[torch.Tensor] = []
        for y0 in ys:
            for x0 in xs:
                crop = img[y0: y0 + th, x0: x0 + tw, :]
                if passthrough:
                    tile = crop.clamp(0.0, 1.0).cpu()
                else:
                    tile = _resize(crop, target_h, target_w, mode).clamp(0.0, 1.0).cpu()
                tiles_list.append(tile)

        tiles_batch = torch.stack(tiles_list, dim=0)   # [N*M, H, W, C]

        # Compute actual overlap % for info display
        if cols > 1 and orig_w > tw:
            stride_x    = (orig_w - tw) / (cols - 1)
            ov_x_pct    = (tw - stride_x) / tw * 100
        else:
            ov_x_pct    = 0.0
        if rows > 1 and orig_h > th:
            stride_y    = (orig_h - th) / (rows - 1)
            ov_y_pct    = (th - stride_y) / th * 100
        else:
            ov_y_pct    = 0.0

        if passthrough:
            scale_str   = "passthrough"
            out_res_str = f"{tw}×{th} (native)"
        else:
            sx          = target_w / tw  if tw  else 0
            sy          = target_h / th  if th  else 0
            scale_str   = f"{sx:.2f}× W / {sy:.2f}× H"
            out_res_str = f"{target_w}×{target_h}"

        info = (
            f"Grid {cols}×{rows} | {cols * rows} tiles | AR {ar}\n"
            f"Source {orig_w}×{orig_h} → crop {tw}×{th} → {out_res_str}\n"
            f"Scale {scale_str} | Overlap X {ov_x_pct:.1f}% Y {ov_y_pct:.1f}%\n"
            f"Blend {blend_mode} | Device {'CUDA' if use_gpu else 'CPU'}"
        )

        stitcher = {
            "orig_w":     orig_w,    "orig_h":    orig_h,
            "tile_w":     tw,        "tile_h":    th,
            "target_w":   target_w if not passthrough else tw,
            "target_h":   target_h if not passthrough else th,
            "cols":       cols,      "rows":      rows,
            "xs":         xs,        "ys":        ys,
            "scale_algo": scale_algo,
            "blend_mode": blend_mode,
            "device":     "cuda" if use_gpu else "cpu",
            "C":          C,
            "passthrough": passthrough,
        }

        return (stitcher, tiles_batch, ar, info)


# ─── Universal stitch node ────────────────────────────────────────────────────

class TileStitch:
    """
    Blends N×M processed tiles back into a single upscaled image.

    Resolution-agnostic: reads actual tile dimensions from the incoming batch.
    If the model upscaled tiles 2× or 4×, the output canvas is scaled
    proportionally — no manual scale factor required.

    Uses smootherstep feathering (or whatever blend_mode the crop node stored)
    with proper bilateral weighting for interior tiles (tiles with neighbours on
    both sides in 3×3 / 4×4 grids).
    """

    CATEGORY      = "TileUpscale"
    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("image",)
    FUNCTION      = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tile_stitcher": ("TILE_STITCHER",),
                "tiles":         ("IMAGE",),    # [N*M, H, W, C]
            }
        }

    def execute(self, tile_stitcher: dict, tiles: torch.Tensor):
        s      = tile_stitcher
        dev_s  = s["device"]
        device = torch.device(dev_s if (dev_s == "cuda" and torch.cuda.is_available()) else "cpu")
        mode   = _INTERP[s["scale_algo"]]
        bmode  = s.get("blend_mode", "smootherstep")

        orig_w, orig_h = s["orig_w"], s["orig_h"]
        tile_w, tile_h = s["tile_w"], s["tile_h"]
        cols, rows     = s["cols"],   s["rows"]
        xs, ys         = s["xs"],     s["ys"]
        C              = s["C"]

        # Actual tile size after model processing (may be larger than crop)
        th_proc, tw_proc = tiles.shape[1], tiles.shape[2]

        # Scale factors: how much did the model upscale each tile?
        scale_x = tw_proc / tile_w
        scale_y = th_proc / tile_h
        out_w   = round(orig_w * scale_x)
        out_h   = round(orig_h * scale_y)

        xs_out = [round(x * scale_x) for x in xs]
        ys_out = [round(y * scale_y) for y in ys]

        # Interior overlap in output pixels (uniform stride → same for all pairs)
        ov_x = max(0, tw_proc - (xs_out[1] - xs_out[0])) if cols > 1 else 0
        ov_y = max(0, th_proc - (ys_out[1] - ys_out[0])) if rows > 1 else 0

        canvas = torch.zeros(out_h, out_w, C, device=device)
        w_acc  = torch.zeros(out_h, out_w, 1, device=device)

        idx = 0
        for ri, cy in enumerate(ys_out):
            for ci, cx in enumerate(xs_out):
                t_img = tiles[idx].to(device).float()

                # Normalise size — guards ±1-pixel rounding drift between tiles
                if t_img.shape[0] != th_proc or t_img.shape[1] != tw_proc:
                    t_img = _resize(t_img, th_proc, tw_proc, mode)

                wx  = _weight_1d(tw_proc, ci > 0, ci < cols - 1, ov_x, device, bmode)
                wy  = _weight_1d(th_proc, ri > 0, ri < rows - 1, ov_y, device, bmode)
                w2d = (wy.unsqueeze(1) * wx.unsqueeze(0)).unsqueeze(-1)  # [th, tw, 1]

                # Clamp paste region to canvas bounds (rounding can push 1px over)
                x1, y1 = min(cx + tw_proc, out_w), min(cy + th_proc, out_h)
                pw, ph = x1 - cx, y1 - cy

                canvas[cy:y1, cx:x1, :] += t_img[:ph, :pw, :] * w2d[:ph, :pw, :]
                w_acc[cy:y1,  cx:x1, :] += w2d[:ph, :pw, :]

                idx += 1

        result = (canvas / w_acc.clamp(min=1e-6)).clamp(0.0, 1.0)
        return (result.unsqueeze(0).cpu(),)


# ─── Utility nodes ────────────────────────────────────────────────────────────

class TileExtract:
    """
    Extracts one tile from the tiles batch by 0-based index (row-major order).

    Index mapping for 2×2:  0=TL  1=TR  2=BL  3=BR
    Index mapping for 3×3:  0=TL  1=TM  2=TR  3=ML  4=MM  5=MR  6=BL  7=BM  8=BR

    Connect the output to any model that processes a single image
    (e.g. GeminiNanoBanana2 with a reference image).
    """
    CATEGORY      = "TileUpscale"
    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("tile",)
    FUNCTION      = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tiles": ("IMAGE",),
                "index": ("INT", {"default": 0, "min": 0, "max": 63, "step": 1}),
            }
        }

    def execute(self, tiles: torch.Tensor, index: int):
        idx = index % tiles.shape[0]
        return (tiles[idx: idx + 1],)        # [1, H, W, C]


class TileCollect:
    """
    Collects individually processed tiles back into a batch for TileStitch.

    Connect tile_0 … tile_N in row-major order matching the grid used in TileCrop.
    For a 2×2 grid: tile_0=TL, tile_1=TR, tile_2=BL, tile_3=BR.
    For a 3×3 grid: tile_0…tile_8 left→right, top→bottom.
    Up to 16 tiles (4×4 grid).  Only connect the slots your grid requires.

    All tiles are normalised to the size of tile_0 if they differ by a pixel.
    """
    CATEGORY      = "TileUpscale"
    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("tiles",)
    FUNCTION      = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"tile_0": ("IMAGE",)},
            "optional": {f"tile_{i}": ("IMAGE",) for i in range(1, 16)},
        }

    def execute(self, tile_0: torch.Tensor, **kwargs):
        collected = [tile_0]
        for i in range(1, 16):
            t = kwargs.get(f"tile_{i}")
            if t is not None:
                collected.append(t)

        ref_h, ref_w = tile_0.shape[1], tile_0.shape[2]
        out: List[torch.Tensor] = []
        for t in collected:
            if t.shape[1] != ref_h or t.shape[2] != ref_w:
                t = _resize(t[0], ref_h, ref_w, "bicubic").unsqueeze(0)
            out.append(t)
        return (torch.cat(out, dim=0),)      # [N, H, W, C]


class TileInfo:
    """
    Outputs a human-readable summary of the TILE_STITCHER for debugging.
    Connect to a ShowText node to inspect grid geometry and scale factors.
    """
    CATEGORY      = "TileUpscale"
    RETURN_TYPES  = ("STRING",)
    RETURN_NAMES  = ("info",)
    FUNCTION      = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"tile_stitcher": ("TILE_STITCHER",)}}

    def execute(self, tile_stitcher: dict):
        s    = tile_stitcher
        cols, rows = s["cols"], s["rows"]
        tw, th     = s["tile_w"], s["tile_h"]
        ttw, tth   = s.get("target_w", tw), s.get("target_h", th)

        if s.get("passthrough"):
            scale_str = "passthrough (scale determined by external model)"
        else:
            sx = ttw / tw if tw else 0
            sy = tth / th if th else 0
            scale_str = f"{sx:.3f}× W  /  {sy:.3f}× H"

        lines = [
            f"Grid:        {cols}×{rows}  ({cols * rows} tiles)",
            f"Source:      {s['orig_w']} × {s['orig_h']}",
            f"Crop size:   {tw} × {th}",
            f"Target size: {ttw} × {tth}",
            f"Scale:       {scale_str}",
            f"X positions: {s['xs']}",
            f"Y positions: {s['ys']}",
            f"Blend mode:  {s.get('blend_mode', 'smootherstep')}",
            f"Device:      {s['device']}",
        ]
        return ("\n".join(lines),)


# ─── Legacy NB2 nodes  (v1 backward compatibility) ────────────────────────────
#
# Output POSITIONS are identical to v1 — ComfyUI saves connections by index.
# tiles_batch is appended at the END (position 6) so existing connections to
# positions 0-5 load without any changes.

class FlorenceMaskAlign:
    """
    Align a Florence mask to the exact image size before crop / bbox nodes use it.

    Florence2 internally pads images to a square (1024×1024) before processing.
    The output mask is in that padded-square coordinate space.  When the original
    image is non-square, naively resizing the square mask to image dimensions
    stretches the letterbox padding into the image area and shifts the bbox.

    depad_florence=True (default) detects and removes this padding before
    resizing, so bbox coordinates land on the correct region in the original image.
    """

    CATEGORY      = "TileUpscale"
    RETURN_TYPES  = ("IMAGE", "MASK", "IMAGE", "INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES  = ("image", "mask", "masked_preview", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "info")
    FUNCTION      = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":           ("IMAGE",),
                "mask":            ("MASK",),
                "resize_mode":     (["nearest", "bilinear", "bicubic"], {"default": "nearest"}),
                "threshold":       ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "binarize":        ("BOOLEAN", {"default": True}),
                "invert_mask":     ("BOOLEAN", {"default": False}),
                "bbox_padding":    ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "depad_florence":  ("BOOLEAN", {"default": True}),
            }
        }

    def execute(self, image, mask, resize_mode, threshold, binarize, invert_mask,
                bbox_padding, depad_florence=True):
        if image.ndim != 4:
            raise ValueError(f"Expected IMAGE tensor [B,H,W,C], got shape {tuple(image.shape)}")

        batch_size, target_h, target_w, _ = image.shape
        mask_batch, note = _coerce_mask_batch(mask, batch_size, target_h, target_w)

        # ── Florence2 letterbox correction ────────────────────────────────────
        # Florence2 resizes images so the longest side = its internal resolution,
        # then pads to a square.  The output mask lives in that padded-square space.
        # We detect this by checking: mask is approximately square AND its AR
        # differs from the target (original image) AR.  If so, crop out the
        # letterbox padding before resizing so the bbox lands in the right place.
        mask_h, mask_w = mask_batch.shape[1], mask_batch.shape[2]
        if (depad_florence
                and (mask_h != target_h or mask_w != target_w)
                and mask_h > 0 and mask_w > 0):
            mask_ar   = mask_w / mask_h
            target_ar = target_w / target_h
            ar_diff   = abs(mask_ar - target_ar)
            if ar_diff > 0.05 and abs(mask_ar - 1.0) < 0.05:
                if target_ar > 1.0:
                    # Landscape original → Florence padded top/bottom
                    content_h = max(1, round(mask_h * target_h / target_w))
                    pad_y     = max(0, (mask_h - content_h) // 2)
                    content_h = min(content_h, mask_h - pad_y)
                    mask_batch = mask_batch[:, pad_y : pad_y + content_h, :]
                    note += (f" | Florence depad top/bottom:"
                             f" y[{pad_y}:{pad_y+content_h}] of {mask_h}")
                else:
                    # Portrait original → Florence padded left/right
                    content_w = max(1, round(mask_w * target_w / target_h))
                    pad_x     = max(0, (mask_w - content_w) // 2)
                    content_w = min(content_w, mask_w - pad_x)
                    mask_batch = mask_batch[:, :, pad_x : pad_x + content_w]
                    note += (f" | Florence depad left/right:"
                             f" x[{pad_x}:{pad_x+content_w}] of {mask_w}")

        mask_resized = _resize_mask(mask_batch, target_h, target_w, resize_mode).clamp(0.0, 1.0)

        if invert_mask:
            mask_resized = 1.0 - mask_resized

        if binarize:
            mask_resized = (mask_resized >= threshold).float()

        mask_for_bbox = mask_resized[0] >= threshold if binarize else mask_resized[0] > threshold
        coords = torch.nonzero(mask_for_bbox, as_tuple=False)
        if coords.numel() == 0:
            x0, y0, x1, y1 = 0, 0, target_w - 1, target_h - 1
            bbox_note = "no active mask pixels; bbox fell back to full image"
        else:
            y0 = max(0, int(coords[:, 0].min().item()) - bbox_padding)
            y1 = min(target_h - 1, int(coords[:, 0].max().item()) + bbox_padding)
            x0 = max(0, int(coords[:, 1].min().item()) - bbox_padding)
            x1 = min(target_w - 1, int(coords[:, 1].max().item()) + bbox_padding)
            bbox_note = "bbox extracted from aligned mask"

        bbox_w = x1 - x0 + 1
        bbox_h = y1 - y0 + 1

        preview = image * mask_resized.unsqueeze(-1)
        info = (
            f"Image {target_w}x{target_h} | {note}\n"
            f"Aligned mask -> [{batch_size}, {target_h}, {target_w}] via {resize_mode}\n"
            f"threshold={threshold:.2f} | binarize={binarize} | invert={invert_mask}\n"
            f"BBox x={x0} y={y0} w={bbox_w} h={bbox_h} | {bbox_note}"
        )
        return (image, mask_resized, preview, x0, y0, bbox_w, bbox_h, info)


class MaskBBoxCrop:
    """
    Crop image + mask using bbox coordinates, typically from FlorenceMaskAlign.
    """

    CATEGORY      = "TileUpscale"
    RETURN_TYPES  = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES  = ("image", "mask", "info")
    FUNCTION      = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":        ("IMAGE",),
                "mask":         ("MASK",),
                "bbox_x":       ("INT", {"default": 0, "min": 0, "max": 16384, "step": 1}),
                "bbox_y":       ("INT", {"default": 0, "min": 0, "max": 16384, "step": 1}),
                "bbox_w":       ("INT", {"default": 512, "min": 1, "max": 16384, "step": 1}),
                "bbox_h":       ("INT", {"default": 512, "min": 1, "max": 16384, "step": 1}),
                "extra_padding": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            }
        }

    def execute(self, image, mask, bbox_x, bbox_y, bbox_w, bbox_h, extra_padding):
        if image.ndim != 4:
            raise ValueError(f"Expected IMAGE tensor [B,H,W,C], got shape {tuple(image.shape)}")

        batch_size, target_h, target_w, _ = image.shape
        mask_batch, note = _coerce_mask_batch(mask, batch_size, target_h, target_w)
        mask_resized = _resize_mask(mask_batch, target_h, target_w, "nearest").clamp(0.0, 1.0)

        x0 = max(0, bbox_x - extra_padding)
        y0 = max(0, bbox_y - extra_padding)
        x1 = min(target_w, bbox_x + bbox_w + extra_padding)
        y1 = min(target_h, bbox_y + bbox_h + extra_padding)

        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"Invalid bbox after padding: x={x0}:{x1}, y={y0}:{y1}")

        cropped_image = image[:, y0:y1, x0:x1, :]
        cropped_mask = mask_resized[:, y0:y1, x0:x1]
        info = (
            f"Crop x={x0} y={y0} w={x1 - x0} h={y1 - y0} | extra_padding={extra_padding}\n"
            f"Source image {target_w}x{target_h} | {note}"
        )
        return (cropped_image, cropped_mask, info)


class TileCropNB2:
    """
    2×2 NB2 tile crop — v1 compatible.

    Outputs (in order):
      0  tile_stitcher   — connect to TileStitchNB2 or TileStitch
      1  tile_tl         — top-left  tile
      2  tile_tr         — top-right tile
      3  tile_bl         — bottom-left  tile
      4  tile_br         — bottom-right tile
      5  aspect_ratio    — detected or selected AR string
      6  tiles_batch     — all 4 tiles as a single [4,H,W,C] batch (NEW in v2)

    To access 3×3 / 4×4 grids or other model presets, use TileCrop (Universal).
    """

    CATEGORY      = "NanoBanana2/Tiles"
    RETURN_TYPES  = ("TILE_STITCHER", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING", "IMAGE")
    RETURN_NAMES  = ("tile_stitcher", "tile_tl", "tile_tr", "tile_bl", "tile_br",
                     "aspect_ratio", "tiles_batch")
    FUNCTION      = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        ars = ["auto", "16:9", "9:16", "1:1", "4:5", "5:4",
               "3:4",  "4:3",  "2:3",  "3:2", "21:9", "9:21"]
        return {
            "required": {
                "image":        ("IMAGE",),
                "aspect_ratio": (ars, {"default": "auto"}),
                "resolution":   (["1K", "2K", "4K"], {"default": "2K"}),
                "overlap":      ("FLOAT", {"default": 0.15, "min": 0.05,
                                           "max": 0.40,     "step": 0.01}),
                "scale_algo":   (list(_INTERP), {"default": "bicubic"}),
                "device_mode":  (["gpu (much faster)", "cpu"],
                                  {"default": "gpu (much faster)"}),
            }
        }

    def execute(self, image, aspect_ratio, resolution, overlap, scale_algo, device_mode):
        stitcher, tiles, ar, _ = TileCrop().execute(
            image=image,
            model_preset="NB2",
            grid="2×2",
            aspect_ratio=aspect_ratio,
            resolution_tier=resolution,
            overlap=overlap,
            scale_algo=scale_algo,
            blend_mode="smootherstep",
            device_mode=device_mode,
        )
        # Positions 0-5 match v1 exactly; tiles_batch at position 6 is new.
        return (stitcher,
                tiles[0:1], tiles[1:2], tiles[2:3], tiles[3:4],
                ar, tiles)


class TileStitchNB2:
    """
    NB2 tile stitch — v1 compatible.
    Accepts 4 individually-processed tiles (TL, TR, BL, BR).
    Delegates to TileStitch for consistent blending with TileCrop.
    """

    CATEGORY      = "NanoBanana2/Tiles"
    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("image",)
    FUNCTION      = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tile_stitcher": ("TILE_STITCHER",),
                "tile_tl":       ("IMAGE",),
                "tile_tr":       ("IMAGE",),
                "tile_bl":       ("IMAGE",),
                "tile_br":       ("IMAGE",),
            }
        }

    def execute(self, tile_stitcher, tile_tl, tile_tr, tile_bl, tile_br):
        tiles = torch.cat([tile_tl, tile_tr, tile_bl, tile_br], dim=0)
        return TileStitch().execute(tile_stitcher, tiles)


# ─── Registration ─────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    # Universal
    "TileCrop":        TileCrop,
    "TileStitch":      TileStitch,
    "TileExtract":     TileExtract,
    "TileCollect":     TileCollect,
    "TileInfo":        TileInfo,
    "FlorenceMaskAlign": FlorenceMaskAlign,
    "MaskBBoxCrop":    MaskBBoxCrop,
    # Legacy NB2
    "TileCropNB2":     TileCropNB2,
    "TileStitchNB2":   TileStitchNB2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TileCrop":        "Tile Crop (Universal)",
    "TileStitch":      "Tile Stitch (Universal)",
    "TileExtract":     "Tile Extract",
    "TileCollect":     "Tile Collect",
    "TileInfo":        "Tile Info",
    "FlorenceMaskAlign": "Florence Mask Align",
    "MaskBBoxCrop":    "Mask BBox Crop",
    "TileCropNB2":     "Tile Crop (NB2)",
    "TileStitchNB2":   "Tile Stitch (NB2)",
}
