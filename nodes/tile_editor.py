"""MMH3 Spatial Tile Editor node: visual tile configuration for MMH3 Spatial Extend Video.

Provides a dock-panel UI (via the companion JS file) for configuring per-tile
prompts, reference images, dimensions, overlap and fade values. Outputs a
tile_config dict consumed by the modified MMH3 Spatial Extend Video node, which creates
per-tile conditioning internally.

Design: this node is a PURE INFORMATION node. It never loads images nor emits
image tensors. Each tile independently picks its conditioning mode (FL2VA
or Ref2VA) and its reference source (the compose split crop, or its own image
list). For whole-image compose, the JS front-end computes the exact
source-pixel crop box per tile (``compose_crops``) and ships it as plain
numbers alongside the tile; the consuming Extend Video node performs the
actual file crop at execution time. This keeps crop geometry in one place
(front-end) and makes it inspectable instead of opaque tensors.
"""

import json

from comfy_api.latest import io

VAE_DOWNSAMPLE = 16
CANVAS_MULTIPLE = 32

# Defaults for new tiles (used by JS when creating tiles)
DEFAULT_TILE_WIDTH = 512
DEFAULT_TILE_HEIGHT = 768
DEFAULT_OVERLAP_W = 128
DEFAULT_OVERLAP_H = 128
DEFAULT_FADE_W = 32
DEFAULT_FADE_H = 32


# ---------------------------------------------------------------------------
# Layout geometry helpers
# ---------------------------------------------------------------------------

def _alternating_offsets(widths, ol_ws):
    n = len(widths)
    offs = [0] * n
    lo_edge = 0
    hi_edge = widths[0]
    for i in range(1, n):
        if i % 2 == 1:
            offs[i] = hi_edge - ol_ws[i]
            hi_edge = offs[i] + widths[i]
        else:
            offs[i] = lo_edge + ol_ws[i] - widths[i]
            lo_edge = offs[i]
    return offs


def _total_size(tiles, scheme, axis):
    if scheme == "4_quadrants_expand" and len(tiles) == 5:
        return _quadrant_total_size(tiles)
    if axis == "horizontal":
        widths = [t["width"] for t in tiles]
        ol_ws = [t["overlap_w"] for t in tiles]
        offs = _alternating_offsets(widths, ol_ws)
        lo = min(offs)
        hi = max(offs[i] + widths[i] for i in range(len(tiles)))
        th = tiles[0]["height"]
        return (hi - lo, th)
    else:
        heights = [t["height"] for t in tiles]
        ol_hs = [t["overlap_h"] for t in tiles]
        offs = _alternating_offsets(heights, ol_hs)
        lo = min(offs)
        hi = max(offs[i] + heights[i] for i in range(len(tiles)))
        tw = tiles[0]["width"]
        return (tw, hi - lo)


def _quadrant_total_size(tiles):
    (h0, w0) = (tiles[0]["height"], tiles[0]["width"])
    (h1, w1) = (tiles[1]["height"], tiles[1]["width"])
    (h2, w2) = (tiles[2]["height"], tiles[2]["width"])
    (h3, w3) = (tiles[3]["height"], tiles[3]["width"])
    (h4, w4) = (tiles[4]["height"], tiles[4]["width"])
    oh1, oh2, oh3, oh4 = [t["overlap_h"] for t in tiles[1:]]
    ow1, ow2, ow3, ow4 = [t["overlap_w"] for t in tiles[1:]]

    top = oh1 - h1
    left = ow1 - w1
    right = max(w0 + w2 - ow2, w0 + w4 - ow4)
    bot = max(h0 + h3 - oh3, h0 + h4 - oh4)
    return (right - left, bot - top)


def _tile_placements(tiles, scheme, axis):
    if scheme == "4_quadrants_expand" and len(tiles) == 5:
        return _quadrant_placements(tiles)
    placements = []
    if axis == "horizontal":
        widths = [t["width"] for t in tiles]
        ol_ws = [t["overlap_w"] for t in tiles]
        heights = [t["height"] for t in tiles]
        offs = _alternating_offsets(widths, ol_ws)
        lo = min(offs)
        for i, t in enumerate(tiles):
            placements.append((offs[i] - lo, 0, widths[i], heights[i]))
    else:
        heights = [t["height"] for t in tiles]
        ol_hs = [t["overlap_h"] for t in tiles]
        widths = [t["width"] for t in tiles]
        offs = _alternating_offsets(heights, ol_hs)
        lo = min(offs)
        for i, t in enumerate(tiles):
            placements.append((0, offs[i] - lo, widths[i], heights[i]))
    return placements


def _quadrant_placements(tiles):
    (h0, w0) = (tiles[0]["height"], tiles[0]["width"])
    (h1, w1) = (tiles[1]["height"], tiles[1]["width"])
    (h2, w2) = (tiles[2]["height"], tiles[2]["width"])
    (h3, w3) = (tiles[3]["height"], tiles[3]["width"])
    (h4, w4) = (tiles[4]["height"], tiles[4]["width"])
    oh1, oh2, oh3, oh4 = [t["overlap_h"] for t in tiles[1:]]
    ow1, ow2, ow3, ow4 = [t["overlap_w"] for t in tiles[1:]]

    origins = [
        (0, 0),
        (oh1 - h1, ow1 - w1),
        (oh2 - h2, w0 - ow2),
        (h0 - oh3, ow3 - w3),
        (h0 - oh4, w0 - ow4),
    ]
    top = origins[1][0]
    left = origins[1][1]
    return [(co - left, ro - top, tiles[i]["width"], tiles[i]["height"])
            for i, (ro, co) in enumerate(origins)]


# ---------------------------------------------------------------------------
# Plan <-> (scheme, axis) mapping
# ---------------------------------------------------------------------------

PLAN_OPTIONS = ["horizontal_alternating", "vertical_alternating", "4_quadrants_expand"]


def _plan_to_scheme_axis(plan):
    if plan == "4_quadrants_expand":
        return "4_quadrants_expand", "horizontal"
    if plan == "vertical_alternating":
        return "alternating", "vertical"
    return "alternating", "horizontal"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_tiles(tiles, scheme, axis):
    for t in tiles:
        for dim in ("width", "height", "overlap_w", "overlap_h", "fade_w", "fade_h"):
            v = t[dim]
            if v % 32 != 0:
                t[dim] = max(32, round(v / 32) * 32)
        if t["overlap_w"] >= t["width"]:
            t["overlap_w"] = max(0, (t["width"] // 32 - 1) * 32)
        if t["overlap_h"] >= t["height"]:
            t["overlap_h"] = max(0, (t["height"] // 32 - 1) * 32)
        if t["fade_w"] > t["overlap_w"]:
            t["fade_w"] = t["overlap_w"]
        if t["fade_h"] > t["overlap_h"]:
            t["fade_h"] = t["overlap_h"]

    if scheme == "alternating":
        if axis == "horizontal":
            hs = set(t["height"] for t in tiles)
            if len(hs) > 1:
                most = max(hs, key=lambda v: sum(1 for t in tiles if t["height"] == v))
                for t in tiles:
                    t["height"] = most
        else:
            ws = set(t["width"] for t in tiles)
            if len(ws) > 1:
                most = max(ws, key=lambda v: sum(1 for t in tiles if t["width"] == v))
                for t in tiles:
                    t["width"] = most
    elif scheme == "4_quadrants_expand":
        if len(tiles) != 5:
            raise ValueError("4_quadrants_expand requires exactly 5 tiles")


# ---------------------------------------------------------------------------
# Node definition
# ---------------------------------------------------------------------------

class MMH3SpatialTileEditor(io.ComfyNode):
    """Visual tile configuration for MMH3 Spatial Extend Video."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3SpatialTileEditor",
            display_name="MMH3 Spatial Tile Editor",
            category="model/conditioning/minimax",
            description=(
                "Pure-information tile configuration for 'MMH3 Spatial Extend Video'. "
                "Provides a dock-panel editor for arranging tiles, setting "
                "per-tile prompts and conditioning mode (each tile is "
                "independently FL2VA or Ref2VA), and auto-calculating "
                "overlap/fade values. No images are loaded or tensors emitted. "
                "Compose supplies a whole image that is split by the tile plan; "
                "each tile may use its split crop (as FL2VA first/last frames "
                "or Ref2VA reference blocks) or load its own images. The JS "
                "ships each tile's exact crop box as plain numbers "
                "(compose_crops). Connect tile_config to MMH3 Spatial Extend Video."
            ),
            search_aliases=["h3 tile editor", "h3 tile config", "h3 extend editor"],
            inputs=[
                io.Boolean.Input("show_editor", default=True,
                                 tooltip="Toggle the visual dock editor panel on/off."),
                io.String.Input("base_prompt", default="",
                                multiline=True, dynamic_prompts=True,
                                tooltip="Shared positive prompt applied to all tiles."),
                io.String.Input("base_negative", default="",
                                multiline=True, dynamic_prompts=True,
                                tooltip="Shared negative prompt applied to all tiles."),
                io.String.Input("tile_data", default="{}",
                                tooltip="Internal JSON managed by the dock editor."),
            ],
            outputs=[
                io.Dict.Output("tile_config",
                               tooltip="Complete tile configuration (pure dict/text) for MMH3 Spatial Extend Video."),
                io.Dict.Output("segments_info",
                               tooltip="Inspectable geometry summary: tile count, plan, dimensions, and per-tile compose crop boxes."),
            ],
        )

    @classmethod
    def execute(cls, show_editor=True,
                base_prompt="", base_negative="",
                tile_data="{}") -> io.NodeOutput:
        raw = {}
        if tile_data:
            try:
                raw = json.loads(tile_data)
            except (json.JSONDecodeError, TypeError):
                raw = {}

        # Plan and tile count are JS-managed state serialized inside tile_data
        # (top-level "plan"/"tile_count" keys) - the python node exposes no
        # widgets for them anymore. Older saves without the keys fall back to
        # the default plan and the highest tile index present.
        plan = raw.get("plan", "horizontal_alternating")
        if plan not in PLAN_OPTIONS:
            plan = "horizontal_alternating"
        tile_count = raw.get("tile_count")
        if not isinstance(tile_count, int) or tile_count < 1:
            idxs = [int(k) for k in raw if isinstance(k, str) and k.isdigit()]
            tile_count = (max(idxs) + 1) if idxs else 2
        tile_count = max(1, min(12, tile_count))
        if plan == "4_quadrants_expand":
            tile_count = 5
        scheme, axis = _plan_to_scheme_axis(plan)

        # Compose is mode-agnostic: it merely supplies whole images that the
        # tile plan splits into per-tile crops. Each tile independently decides
        # (via ref_source) whether it consumes its split crop or its own image
        # list, and (via cond_mode) whether that reference feeds FL2VA frames
        # or Ref2VA blocks. Accept the historical "fl2va_compose" key from
        # older saved workflows.
        compose = raw.get("compose", raw.get("fl2va_compose"))
        # Compose is on whenever a First or Last source image is loaded
        # (the front-end no longer ships an "enabled" switch).
        compose_on = isinstance(compose, dict) and bool(
            (compose.get("first") or {}).get("name")
            or (compose.get("last") or {}).get("name"))

        tiles = []
        for i in range(tile_count):
            t = raw.get(str(i), raw.get(i, {}))
            if not isinstance(t, dict):
                t = {}
            cond = t.get("cond_mode", "FL2VA")
            if cond not in ("FL2VA", "Ref2VA"):
                cond = "FL2VA"
            # Default ref_source matches the JS migration rule: tiles that
            # already carry manual refs keep using them ("own"), empty tiles
            # default to their compose split crop ("crop").
            has_own = bool(t.get("ref_images"))
            ref_source = t.get("ref_source") or ("own" if has_own else "crop")
            if ref_source not in ("crop", "own"):
                ref_source = "own" if has_own else "crop"
            # Compose crop boxes computed by the JS front-end as plain
            # numbers: [{kind:"first"|"last", name, box:[L,T,R,B]}]. Passed
            # through verbatim; Extend Video performs the file crop at run
            # time (this node never loads images / emits tensors). Honoured
            # only while compose is on AND the tile opts into its crop
            # (stale boxes left in a previously-saved tile_data must not
            # leak through).
            crops = t.get("compose_crops", t.get("fl2va_crops", [])) \
                if (compose_on and ref_source == "crop") else []
            tiles.append({
                "index": i,
                "prompt": t.get("prompt", ""),
                "negative": t.get("negative", ""),
                "cond_mode": cond,
                "ref_source": ref_source,
                "ref_images": t.get("ref_images", []),
                "compose_crops": crops,
                "width": t.get("width", DEFAULT_TILE_WIDTH),
                "height": t.get("height", DEFAULT_TILE_HEIGHT),
                "overlap_w": t.get("overlap_w", DEFAULT_OVERLAP_W),
                "overlap_h": t.get("overlap_h", DEFAULT_OVERLAP_H),
                "fade_w": t.get("fade_w", DEFAULT_FADE_W),
                "fade_h": t.get("fade_h", DEFAULT_FADE_H),
            })

        _validate_tiles(tiles, scheme, axis)

        total_w, total_h = _total_size(tiles, scheme, axis)
        placements = _tile_placements(tiles, scheme, axis)

        # Pure-information payload. placement (canvas-pixel rect) and the total
        # canvas size are included so the consuming Extend Video node can
        # derive a fit/centred crop purely from numbers if a front-end box is
        # ever absent (async decode races) -- no layout re-derivation needed.
        for i, t in enumerate(tiles):
            t["placement"] = list(placements[i])
        config = {
            "scheme": scheme,
            "axis": axis,
            "base_prompt": base_prompt,
            "base_negative": base_negative,
            "compose_enabled": compose_on,
            "total_width": total_w,
            "total_height": total_h,
            "tiles": tiles,
        }

        segments = {
            "tile_count": tile_count,
            "plan": plan,
            "scheme": scheme,
            "axis": axis,
            "total_width": total_w,
            "total_height": total_h,
            # Inspectable per-tile geometry (plain numbers only), so crop
            # discrepancies like "starts at the top-left of the ref image"
            # can be diagnosed by reading this dict directly: `placement` is
            # the tile's rect on the layout canvas (x,y,w,h), `compose_crops`
            # is the exact source-pixel crop box the JS front-end computed.
            "tiles": [{
                "index": t["index"],
                "width": t["width"],
                "height": t["height"],
                "cond_mode": t["cond_mode"],
                "ref_source": t["ref_source"],
                "placement": t["placement"],
                "compose_crops": t["compose_crops"],
            } for t in tiles],
        }
        return io.NodeOutput(config, segments)


def align_frame_count(n):
    while n % 17 != 5:
        n += 1
    return n
