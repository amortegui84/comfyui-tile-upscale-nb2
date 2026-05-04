import json, os

def build_workflow(method, grid_cols, grid_rows, filename_prefix, note_text, out_path):
    N = grid_cols * grid_rows

    _nid = [0]; _lid = [0]
    def nn(): _nid[0] += 1; return _nid[0]
    def nl(fn, fs, tn, ts, t): _lid[0] += 1; return [_lid[0], fn, fs, tn, ts, t]

    NOTE_ID      = nn()
    REF_NOTE_ID  = nn()
    LOAD_ID      = nn()
    CROP_ID      = nn()
    PREV_CROP_ID = nn()
    EX_IDS  = [nn() for _ in range(N)]
    SB_IDS  = [nn() for _ in range(N)]
    COLL_ID      = nn()
    PREV_COLL_ID = nn()
    STIT_ID      = nn()
    PREV_STIT_ID = nn()
    SAVE_ID      = nn()
    TEXT_ID      = nn()

    L_load_crop  = nl(LOAD_ID, 0, CROP_ID,      0, "IMAGE")
    L_crop_prev  = nl(CROP_ID, 0, PREV_CROP_ID, 0, "IMAGE")
    L_meta_coll  = nl(CROP_ID, 1, COLL_ID,      1, "STRING")
    L_crop_ex    = [nl(CROP_ID,  0, EX_IDS[i], 0, "IMAGE") for i in range(N)]
    L_ex_sb      = [nl(EX_IDS[i], 0, SB_IDS[i], 0, "IMAGE") for i in range(N)]
    L_sb_coll    = [nl(SB_IDS[i], 0, COLL_ID, 0 if i == 0 else i + 1, "IMAGE") for i in range(N)]
    L_coll_prev  = nl(COLL_ID, 0, PREV_COLL_ID, 0, "IMAGE")
    L_coll_stit  = nl(COLL_ID, 0, STIT_ID,       0, "IMAGE")
    L_meta_stit  = nl(COLL_ID, 3, STIT_ID,       1, "STRING")
    L_stit_prev  = nl(STIT_ID, 0, PREV_STIT_ID,  0, "IMAGE")
    L_stit_save  = nl(STIT_ID, 0, SAVE_ID,        0, "IMAGE")
    L_save_text  = nl(SAVE_ID, 1, TEXT_ID,         0, "STRING")

    links = ([L_load_crop, L_crop_prev, L_meta_coll] +
             L_crop_ex + L_ex_sb + L_sb_coll +
             [L_coll_prev, L_coll_stit, L_meta_stit, L_stit_prev, L_stit_save, L_save_text])

    def lid(lnk): return lnk[0]

    X_LOAD, X_CROP, X_EX, X_SB, X_COLL, X_STIT, X_OUT = 40, 380, 740, 1040, 1380, 1760, 2110
    Y_MAIN = 250
    Y_STEP = 120
    Y_TILES = [Y_MAIN - 40 + i * Y_STEP for i in range(N)]
    COLL_H = 200 + N * 40
    COLL_Y = Y_MAIN

    nodes = []

    REF_NOTE_TEXT = (
        "UPSCALER QUICK REFERENCE\n"
        "──────────────────────────────────────────────────────────────\n"
        "NB2 (Nano Banana 2)\n"
        "  Method: nb2 | Grid: 2×2 | Overlap: 20% | Feather: strong | Color match: ON\n"
        "  Prompt: YES — connect the same prompt to every NB2 node. Describe the output style and detail level.\n"
        "  Tip: pass the original image as reference to anchor the regeneration.\n"
        "\n"
        "GPT-Image-2\n"
        "  Method: image_2 | Grid: 2×2 | Overlap: 20% | Feather: strong | Color match: ON\n"
        "  Prompt: YES — same prompt to all tiles + optional reference image input.\n"
        "\n"
        "Topaz (Photo AI / Sharpen AI)\n"
        "  Method: topaz | Grid: 2×2 | Overlap: 8% | Feather: minimal | Color match: OFF\n"
        "  Prompt: none. If seams appear → raise overlap to 15%, set feather_mode_override = medium in Tile Stitch.\n"
        "\n"
        "SeedVR2\n"
        "  Method: seedv2 | Grid: 2×3 | Overlap: 10–20% | Feather: strong | Color match: ON\n"
        "  Prompt: none. 2×3 grid gives square-ish tiles (model was trained on landscape video frames).\n"
        "  If seams persist → raise overlap to 20%.\n"
        "\n"
        "ESRGAN / RealESRGAN\n"
        "  Method: topaz or passthrough | Grid: 2×2 (or batch mode) | Feather: minimal | Color match: OFF\n"
        "  Prompt: none. These models accept a full batch — you can skip Extract/Collect entirely.\n"
        "──────────────────────────────────────────────────────────────\n"
        "PROMPT TIPS (regenerative models — NB2, GPT-Image-2)\n"
        "  • Same prompt to ALL tiles — diverging prompts cause seams feathering cannot fix\n"
        "  • Describe the output, not the input  (e.g. 'sharp portrait, fine skin texture, soft light')\n"
        "  • Pass the original image as a reference / style input to keep regeneration anchored\n"
        "  • If color looks inconsistent between tiles → color_match_override = on in Tile Stitch"
    )

    nodes.append({
        "id": NOTE_ID, "type": "Note",
        "pos": [X_LOAD, 30], "size": {"0": 980, "1": 170},
        "flags": {}, "order": 0, "mode": 0, "inputs": [], "outputs": [],
        "properties": {}, "widgets_values": [note_text]
    })

    nodes.append({
        "id": REF_NOTE_ID, "type": "Note",
        "pos": [X_LOAD + 1010, 30], "size": {"0": 780, "1": 560},
        "flags": {}, "order": 0, "mode": 0, "inputs": [], "outputs": [],
        "properties": {}, "widgets_values": [REF_NOTE_TEXT]
    })

    nodes.append({
        "id": LOAD_ID, "type": "LoadImage",
        "pos": [X_LOAD, Y_MAIN], "size": {"0": 290, "1": 314},
        "flags": {}, "order": 0, "mode": 0,
        "inputs": [],
        "outputs": [
            {"name": "IMAGE", "type": "IMAGE", "links": [lid(L_load_crop)]},
            {"name": "MASK",  "type": "MASK",  "links": None}
        ],
        "properties": {"Node name for S&R": "LoadImage"},
        "widgets_values": ["example.png", "image"]
    })

    nodes.append({
        "id": CROP_ID, "type": "TileCropAM",
        "pos": [X_CROP, Y_MAIN], "size": {"0": 310, "1": 250},
        "flags": {}, "order": 1, "mode": 0,
        "inputs": [{"name": "image", "type": "IMAGE", "link": lid(L_load_crop)}],
        "outputs": [
            {"name": "tiles",         "type": "IMAGE",  "links": [lid(L_crop_prev)] + [lid(l) for l in L_crop_ex]},
            {"name": "tile_metadata", "type": "STRING", "links": [lid(L_meta_coll)]},
            {"name": "tile_count",    "type": "INT",    "links": None}
        ],
        "properties": {"Node name for S&R": "TileCropAM"},
        "widgets_values": [method, "method default", grid_cols, grid_rows, -1.0, 0, 0]
    })

    crop_bottom = Y_MAIN + 250 + 40
    nodes.append({
        "id": PREV_CROP_ID, "type": "PreviewImage",
        "pos": [X_CROP, crop_bottom], "size": {"0": 310, "1": 260},
        "flags": {}, "order": 2, "mode": 0,
        "inputs": [{"name": "images", "type": "IMAGE", "link": lid(L_crop_prev)}],
        "outputs": [],
        "properties": {"Node name for S&R": "PreviewImage"},
        "widgets_values": []
    })

    for i in range(N):
        nodes.append({
            "id": EX_IDS[i], "type": "TileExtractAM",
            "pos": [X_EX, Y_TILES[i]], "size": {"0": 240, "1": 90},
            "flags": {}, "order": 2, "mode": 0,
            "inputs": [{"name": "tiles", "type": "IMAGE", "link": lid(L_crop_ex[i])}],
            "outputs": [
                {"name": "tile",       "type": "IMAGE", "links": [lid(L_ex_sb[i])]},
                {"name": "tile_index", "type": "INT",   "links": None}
            ],
            "properties": {"Node name for S&R": "TileExtractAM"},
            "widgets_values": [i]
        })

    for i in range(N):
        nodes.append({
            "id": SB_IDS[i], "type": "TileScaleByAM",
            "pos": [X_SB, Y_TILES[i]], "size": {"0": 280, "1": 100},
            "flags": {}, "order": 3, "mode": 0,
            "inputs": [{"name": "image", "type": "IMAGE", "link": lid(L_ex_sb[i])}],
            "outputs": [{"name": "image", "type": "IMAGE", "links": [lid(L_sb_coll[i])]}],
            "properties": {"Node name for S&R": "TileScaleByAM"},
            "widgets_values": [2.0, "lanczos"]
        })

    coll_inputs = [{"name": "tile_0",        "type": "IMAGE",  "link": lid(L_sb_coll[0])}]
    coll_inputs.append({"name": "tile_metadata", "type": "STRING", "link": lid(L_meta_coll)})
    for i in range(1, N):
        coll_inputs.append({"name": f"tile_{i}", "type": "IMAGE", "link": lid(L_sb_coll[i])})

    nodes.append({
        "id": COLL_ID, "type": "TileCollectAM",
        "pos": [X_COLL, COLL_Y], "size": {"0": 330, "1": COLL_H},
        "flags": {}, "order": 4, "mode": 0,
        "inputs": coll_inputs,
        "outputs": [
            {"name": "tiles",         "type": "IMAGE",  "links": [lid(L_coll_prev), lid(L_coll_stit)]},
            {"name": "tile_count",    "type": "INT",    "links": None},
            {"name": "info",          "type": "STRING", "links": None},
            {"name": "tile_metadata", "type": "STRING", "links": [lid(L_meta_stit)]}
        ],
        "properties": {"Node name for S&R": "TileCollectAM"},
        "widgets_values": []
    })

    coll_bottom = COLL_Y + COLL_H + 40
    nodes.append({
        "id": PREV_COLL_ID, "type": "PreviewImage",
        "pos": [X_COLL, coll_bottom], "size": {"0": 330, "1": 260},
        "flags": {}, "order": 5, "mode": 0,
        "inputs": [{"name": "images", "type": "IMAGE", "link": lid(L_coll_prev)}],
        "outputs": [],
        "properties": {"Node name for S&R": "PreviewImage"},
        "widgets_values": []
    })

    nodes.append({
        "id": STIT_ID, "type": "TileStitchAM",
        "pos": [X_STIT, COLL_Y], "size": {"0": 310, "1": 160},
        "flags": {}, "order": 5, "mode": 0,
        "inputs": [
            {"name": "tiles",         "type": "IMAGE",  "link": lid(L_coll_stit)},
            {"name": "tile_metadata", "type": "STRING", "link": lid(L_meta_stit)}
        ],
        "outputs": [{"name": "stitched_image", "type": "IMAGE",
                     "links": [lid(L_stit_prev), lid(L_stit_save)]}],
        "properties": {"Node name for S&R": "TileStitchAM"},
        "widgets_values": ["auto", "auto"]
    })

    nodes.append({
        "id": PREV_STIT_ID, "type": "PreviewImage",
        "pos": [X_OUT, COLL_Y], "size": {"0": 310, "1": 310},
        "flags": {}, "order": 6, "mode": 0,
        "inputs": [{"name": "images", "type": "IMAGE", "link": lid(L_stit_prev)}],
        "outputs": [],
        "properties": {"Node name for S&R": "PreviewImage"},
        "widgets_values": []
    })

    nodes.append({
        "id": SAVE_ID, "type": "SaveImageWithDPI",
        "pos": [X_OUT, COLL_Y + 360], "size": {"0": 310, "1": 210},
        "flags": {}, "order": 6, "mode": 0,
        "inputs": [{"name": "image", "type": "IMAGE", "link": lid(L_stit_save)}],
        "outputs": [
            {"name": "saved_path",      "type": "STRING", "links": None},
            {"name": "print_size_info", "type": "STRING", "links": [lid(L_save_text)]}
        ],
        "properties": {"Node name for S&R": "SaveImageWithDPI"},
        "widgets_values": [filename_prefix, "300", "png", 95, ""]
    })

    nodes.append({
        "id": TEXT_ID, "type": "ShowTextAM",
        "pos": [X_OUT, COLL_Y + 620], "size": {"0": 310, "1": 120},
        "flags": {}, "order": 7, "mode": 0,
        "inputs": [{"name": "text", "type": "STRING", "link": lid(L_save_text)}],
        "outputs": [{"name": "text", "type": "STRING", "links": None}],
        "properties": {"Node name for S&R": "ShowTextAM"},
        "widgets_values": []
    })

    workflow = {
        "id": f"tile-upscale-{method}-{grid_cols}x{grid_rows}",
        "revision": 1,
        "last_node_id": _nid[0],
        "last_link_id": _lid[0],
        "nodes": nodes,
        "links": links,
        "groups": [], "config": {}, "extra": {}, "version": 0.4
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(workflow, f, indent=2)
    print(f"Generated: {out_path}  (grid={grid_cols}x{grid_rows}, N={N}, nodes={_nid[0]}, links={_lid[0]})")


build_workflow(
    method="nb2", grid_cols=2, grid_rows=2,
    filename_prefix="nb2_tile_upscale",
    note_text="NB2 Tile Upscale - Regenerative 2x2 (4 tiles)\n\nPipeline: Tile Crop -> preview tiles -> Extract (x4) -> Scale By Placeholder (x4) -> Tile Collect -> preview upscaled -> Tile Stitch -> preview + Save + print size.\n\nReplace each Tile Scale By / Placeholder with your NB2 or GPT-Image-2 upscaler node.\ntile_metadata flows linearly: TileCropAM -> TileCollectAM (passthrough) -> TileStitchAM.\nGrid preset 'method default' auto-selects the recommended grid for the chosen method.",
    out_path="workflows/tile_upscale_01_nb2_2x2_4_tiles.json"
)

build_workflow(
    method="image_2", grid_cols=3, grid_rows=2,
    filename_prefix="image2_tile_upscale",
    note_text="GPT Image 2 Tile Upscale - Regenerative 3x2 (6 tiles)\n\nPipeline: Tile Crop -> preview tiles -> Extract (x6) -> Scale By Placeholder (x6) -> Tile Collect -> preview upscaled -> Tile Stitch -> preview + Save + print size.\n\nReplace each Tile Scale By / Placeholder with your GPT-Image-2 or NB2 upscaler node.\ntile_metadata flows linearly: TileCropAM -> TileCollectAM (passthrough) -> TileStitchAM.\nGrid preset 'method default' auto-selects the recommended grid for the chosen method.",
    out_path="workflows/tile_upscale_02_image2_3x2_6_tiles.json"
)

build_workflow(
    method="topaz", grid_cols=2, grid_rows=2,
    filename_prefix="topaz_tile_upscale",
    note_text="Faithful Tile Upscale - Topaz / SeedV2 2x2 (4 tiles)\n\nPipeline: Tile Crop -> preview tiles -> Extract (x4) -> Scale By Placeholder (x4) -> Tile Collect -> preview upscaled -> Tile Stitch -> preview + Save + print size.\n\nReplace each Tile Scale By / Placeholder with Topaz, SeedV2, ESRGAN, or another faithful upscaler.\ntile_metadata flows linearly: TileCropAM -> TileCollectAM (passthrough) -> TileStitchAM.\nFor SeedVR2: change method to 'seedv2' — grid_preset 'method default' will switch to 2x3 (square-ish tiles).",
    out_path="workflows/tile_upscale_03_faithful_2x2_4_tiles.json"
)
