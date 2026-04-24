"""
TileCropNB2 / TileStitchNB2
Tile-based upscaling nodes for ComfyUI, compatible with Nano Banana 2 resolutions.
"""
import torch
import torch.nn.functional as F

# ─── NB2 resolution table ─────────────────────────────────────────────────────
NB2_RESOLUTIONS = {
    "16:9": {"1K": (1376, 768),  "2K": (2752, 1536), "4K": (5504, 3072)},
    "9:16": {"1K": (768, 1376),  "2K": (1536, 2752), "4K": (3072, 5504)},
    "1:1":  {"1K": (1024, 1024), "2K": (2048, 2048), "4K": (4096, 4096)},
    "4:5":  {"1K": (928, 1152),  "2K": (1856, 2304), "4K": (3712, 4608)},
    "5:4":  {"1K": (1152, 928),  "2K": (2304, 1856), "4K": (4608, 3712)},
}

_NB2_RATIOS = {ar: NB2_RESOLUTIONS[ar]["1K"][0] / NB2_RESOLUTIONS[ar]["1K"][1]
               for ar in NB2_RESOLUTIONS}

_INTERP = {"bilinear": "bilinear", "bicubic": "bicubic", "nearest": "nearest"}


# ─── helpers ──────────────────────────────────────────────────────────────────

def _detect_ar(w: int, h: int) -> str:
    """Pick the NB2 aspect ratio closest to the image's actual ratio."""
    r = w / h
    return min(_NB2_RATIOS.items(), key=lambda kv: abs(kv[1] - r))[0]


def _tile_dims(orig_w: int, orig_h: int, nb2_w: int, nb2_h: int, overlap: float):
    """
    Compute tile (width, height) such that:
      1. tile_w / tile_h == nb2_w / nb2_h  → isotropic scale, NO deformation.
      2. tile_w >= orig_w * (0.5 + overlap) AND tile_h >= orig_h * (0.5 + overlap)
         → guaranteed overlap on both interior axes.
      3. tile fits inside the original image.

    Strategy: the NB2 ratio k = nb2_w/nb2_h may differ from orig_w/orig_h.
    We independently compute the minimum tile_w needed to satisfy each axis
    constraint, take the maximum (so both are satisfied), then derive the
    other dimension from k. Finally clamp to image bounds.
    """
    k = nb2_w / nb2_h

    # Minimum tile_w to honour the overlap requirement on each axis separately
    tw_from_x = orig_w * (0.5 + overlap)          # X axis drives tile_w directly
    tw_from_y = orig_h * (0.5 + overlap) * k      # Y axis drives tile_h → convert to tile_w via k

    tw = max(tw_from_x, tw_from_y)
    th = tw / k

    # Clamp to image bounds while preserving the ratio
    if tw > orig_w:
        tw = float(orig_w)
        th = tw / k
    if th > orig_h:
        th = float(orig_h)
        tw = th * k

    return round(tw), round(th)


def _smoothstep(t: torch.Tensor) -> torch.Tensor:
    return t * t * (3.0 - 2.0 * t)


def _resize(img: torch.Tensor, th: int, tw: int, mode: str) -> torch.Tensor:
    """[H, W, C] → [th, tw, C]"""
    x = img.permute(2, 0, 1).unsqueeze(0).float()
    kw = {} if mode == "nearest" else {"align_corners": False}
    x = F.interpolate(x, size=(th, tw), mode=mode, **kw)
    return x.squeeze(0).permute(1, 2, 0)


def _weight_1d(length: int, is_far_side: bool, overlap_px: int,
               device: torch.device) -> torch.Tensor:
    """
    1D feather weight vector of `length` pixels.

    is_far_side=False  (near / outer side — TL,BL in X; TL,TR in Y):
        weight = 1 everywhere except the last `overlap_px` pixels which
        feather smoothly from 1 → 0 toward the interior edge.
    is_far_side=True   (far / outer side — TR,BR in X; BL,BR in Y):
        first `overlap_px` pixels feather 0 → 1 from the interior edge,
        rest = 1.

    Property: smoothstep(t) + smoothstep(1-t) = 1, so paired weights for
    any two adjacent tiles always sum to exactly 1 — no normalization needed.
    Outer image boundaries (non-overlapping edges) always receive weight 1.
    """
    w = torch.ones(length, device=device)
    if overlap_px <= 0:
        return w
    t    = torch.linspace(0.0, 1.0, overlap_px, device=device)
    ramp = _smoothstep(t)
    if is_far_side:
        w[:overlap_px] = ramp           # interior edge: 0 → 1
    else:
        w[length - overlap_px:] = 1.0 - ramp   # interior edge: 1 → 0
    return w


# ─── nodes ────────────────────────────────────────────────────────────────────

class TileCropNB2:
    """
    Splits an image into 4 overlapping tiles, each scaled to NB2 resolution
    with NO aspect-ratio deformation.

    Key fix vs naive approach:
      Naive: tile_w = orig_w * 0.65, tile_h = orig_h * 0.65
             → tile ratio = orig ratio ≠ NB2 ratio → stretch on scale.
      Fixed: tile dimensions derived from NB2 ratio so that scaling
             to nb2_w × nb2_h is purely isotropic.

    Overlap guarantees:
      Both interior axes always have ≥ `overlap` extension beyond the midpoint.
      One axis will match exactly; the other gets a slightly larger overlap
      to satisfy the NB2 ratio constraint — still seamless after stitching.

    Example — 2712×4608 portrait → auto detects 9:16, resolution 4K:
      NB2 target: 3072×5504, ratio k = 0.5581
      tile_w = max(2712×0.65, 4608×0.65×k) = max(1763, 1677) = 1763
      tile_h = 1763 / k = 3159
      X interior overlap: (2×1763 - 2712) / 2712 = 30.0%
      Y interior overlap: (2×3159 - 4608) / 4608 = 37.1%   ← both ≥ 15% ✓
      Scale: 3072 / 1763 = 1.74×
      Output canvas: 4727×8031 (preserves original 2712/4608 ratio)
    """

    CATEGORY = "NanoBanana2/Tiles"
    RETURN_TYPES  = ("TILE_STITCHER", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES  = ("tile_stitcher", "tile_tl", "tile_tr", "tile_bl", "tile_br", "aspect_ratio")
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image":        ("IMAGE",),
                "aspect_ratio": (["auto", "16:9", "9:16", "1:1", "4:5", "5:4"], {"default": "auto"}),
                "resolution":   (["1K", "2K", "4K"],              {"default": "2K"}),
                "overlap":      ("FLOAT",  {"default": 0.15, "min": 0.05,
                                            "max": 0.40,  "step": 0.01}),
                "scale_algo":   (["bicubic", "bilinear", "nearest"], {"default": "bicubic"}),
                "device_mode":  (["gpu (much faster)", "cpu"],       {"default": "gpu (much faster)"}),
            }
        }

    def execute(self, image, aspect_ratio, resolution, overlap, scale_algo, device_mode):
        use_gpu = device_mode.startswith("gpu") and torch.cuda.is_available()
        device  = torch.device("cuda" if use_gpu else "cpu")
        mode    = _INTERP[scale_algo]

        img = image[0].to(device)
        orig_h, orig_w, C = img.shape

        ar           = _detect_ar(orig_w, orig_h) if aspect_ratio == "auto" else aspect_ratio
        nb2_w, nb2_h = NB2_RESOLUTIONS[ar][resolution]

        tile_w, tile_h = _tile_dims(orig_w, orig_h, nb2_w, nb2_h, overlap)

        # Top-left corner of the two right / bottom tiles
        x1 = orig_w - tile_w
        y1 = orig_h - tile_h

        regions = {
            "tl": (0,  0,  tile_w,  tile_h),
            "tr": (x1, 0,  orig_w,  tile_h),
            "bl": (0,  y1, tile_w,  orig_h),
            "br": (x1, y1, orig_w,  orig_h),
        }

        def crop_scale(x0, cy0, x_end, y_end):
            crop   = img[cy0:y_end, x0:x_end, :]          # [tile_h, tile_w, C]
            scaled = _resize(crop, nb2_h, nb2_w, mode)    # [nb2_h, nb2_w, C] — isotropic
            return scaled.clamp(0.0, 1.0).unsqueeze(0).cpu()

        stitcher = {
            "orig_w":     orig_w,  "orig_h":     orig_h,
            "tile_w":     tile_w,  "tile_h":     tile_h,
            "nb2_w":      nb2_w,   "nb2_h":      nb2_h,
            "scale_algo": scale_algo,
            "device":     "cuda" if use_gpu else "cpu",
            "C":          C,
        }

        return (
            stitcher,
            crop_scale(*regions["tl"]),
            crop_scale(*regions["tr"]),
            crop_scale(*regions["bl"]),
            crop_scale(*regions["br"]),
            ar,
        )


class TileStitchNB2:
    """
    Blends 4 NB2-processed tiles back into a single upscaled image.

    Resolution-agnostic: the actual tile dimensions are read from the incoming
    images, so it works regardless of whether NB2 returns tiles at the same
    resolution it received (1:1), at 2× (e.g. 2K in → 4K out), or any other
    factor.  The output canvas is derived from orig_w/orig_h × scale.

    Each tile receives a 2D smoothstep weight mask:
      - weight = 1 on outer (image-boundary) edges — no feathering there
      - feathers smoothly 1→0 toward each interior (shared) edge
    Paired masks from adjacent tiles sum to exactly 1 (partition of unity),
    so blending is seamless across both edge strips and the centre quad.
    """

    CATEGORY = "NanoBanana2/Tiles"
    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("image",)
    FUNCTION = "execute"

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
        s      = tile_stitcher
        dev_s  = s["device"]
        device = torch.device(dev_s if (dev_s == "cuda" and torch.cuda.is_available()) else "cpu")
        mode   = _INTERP[s["scale_algo"]]

        orig_w, orig_h = s["orig_w"], s["orig_h"]
        tile_w, tile_h = s["tile_w"], s["tile_h"]   # original crop size (pre-NB2)
        C = s["C"]

        # Read actual output dimensions from the processed tiles.
        # NB2 may upscale (e.g. send 2K → receive 4K), so we cannot assume
        # the tiles come back at the same resolution they were sent at.
        # All four tiles must be the same size; we read from TL as reference.
        th, tw = tile_tl.shape[1], tile_tl.shape[2]   # [B, H, W, C]

        # How much did NB2 scale each tile relative to the original crop?
        scale_x = tw / tile_w
        scale_y = th / tile_h

        # Output canvas: original image at the NB2-upscaled size.
        # Because tile_w/tile_h == nb2_w/nb2_h (enforced in TileCropNB2),
        # scale_x == scale_y and the original aspect ratio is preserved.
        out_w = round(orig_w * scale_x)
        out_h = round(orig_h * scale_y)

        # Each tile lands on the canvas counting from the nearest corner.
        # Interior overlap zones are derived from the same "border inward"
        # logic used at crop time, but expressed in output (NB2) pixels.
        x1_out = out_w - tw          # left edge of TR / BR
        y1_out = out_h - th          # top  edge of BL / BR
        ov_x   = tw - x1_out         # interior horizontal overlap band
        ov_y   = th - y1_out         # interior vertical   overlap band

        # (tile_batch, canvas_x0, canvas_y0, is_far_x, is_far_y)
        tiles = [
            (tile_tl, 0,      0,      False, False),
            (tile_tr, x1_out, 0,      True,  False),
            (tile_bl, 0,      y1_out, False, True),
            (tile_br, x1_out, y1_out, True,  True),
        ]

        canvas = torch.zeros(out_h, out_w, C, device=device)
        w_acc  = torch.zeros(out_h, out_w, 1, device=device)

        for batch, cx, cy, far_x, far_y in tiles:
            img = batch[0].to(device).float()
            # Normalise to the reference size if tiles differ by a pixel
            if img.shape[0] != th or img.shape[1] != tw:
                img = _resize(img, th, tw, mode)

            wx  = _weight_1d(tw, far_x, ov_x, device)               # [tw]
            wy  = _weight_1d(th, far_y, ov_y, device)               # [th]
            w2d = (wy.unsqueeze(1) * wx.unsqueeze(0)).unsqueeze(-1)  # [th, tw, 1]

            canvas[cy:cy+th, cx:cx+tw, :] += img * w2d
            w_acc[cy:cy+th, cx:cx+tw, :]  += w2d

        # Weights sum to 1 by construction; clamp guards float drift at edges
        result = (canvas / w_acc.clamp(min=1e-6)).clamp(0.0, 1.0)
        return (result.unsqueeze(0).cpu(),)


# ─── registration ─────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "TileCropNB2":   TileCropNB2,
    "TileStitchNB2": TileStitchNB2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TileCropNB2":   "Tile Crop (NB2)",
    "TileStitchNB2": "Tile Stitch (NB2)",
}
