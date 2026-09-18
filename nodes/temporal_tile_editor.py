"""MMH3 Temporal Tile Editor node: per-segment configuration for multi-segment
H3 video generation (each segment continues from the previous one's tail).

Pure-information node (same design as the Spatial Tile Editor): the JS dock
panel serializes all state into the hidden ``tile_data`` JSON string widget;
this node parses and normalizes it into the ``temporal_tile_config`` dict
consumed by MMH3 Temporal Extend Video.

Reference images can also arrive as plain Autogrow IMAGE sockets
('First_or_Ref_Image_0', 'Last_or_Ref_Image_1', 'Ref_Image_2', ...) for users
who prefer wiring LoadImage nodes over picking files in the dock panel. Every
connected socket is available to every segment as a 'load images' reference
('First_or_Ref_Image_0' doubles as segment 0's FL2VA first frame,
'Last_or_Ref_Image_1' as the FL2VA last frame); a segment can rule individual
connected sockets out (``ref_socket_off``).

The sockets reach the Extend node through the custom ``MMH3_REF_IMAGES`` link
as ONE RECORD PER SOCKET (name, slot, batch size, tensor), each at its NATIVE
resolution - they are never merged into a single IMAGE batch. Merging would
force every image to one common size (stretching whatever did not already
match that aspect ratio) and erase the socket boundaries that the '<Picture
i>' numbering and the per-segment ``ref_socket_off`` depend on. Following
MiniMaxH3ReferenceToVideo, ONE SOCKET IS ONE '<Picture i>' (a socket fed by a
batch of images contributes its FIRST image only).

The slot names are kept identifier-safe (``_or_`` rather than ``/``). Only the
GROUP id ('ref_images') reaches ``execute`` as a keyword argument, and the slot
labels become keys inside that dict, so a slash would in fact survive - but
graph-visible names that are valid identifiers avoid a whole family of
surprises (compare the '2nd_sample_params' socket in temporal_extend.py, which
needs ``**kwargs`` precisely because its id is not a Python identifier).

Only ONE socket is created up front (``min=0``: the Autogrow frontend starts
with ``min + 1`` slots, so ``min=1`` would already show two and the old
``min=2`` showed three); the rest appear as the user connects, up to
``max=16``. The frontend always keeps one empty slot available, so connecting
the first image immediately reveals 'Last_or_Ref_Image_1'.

The fun-control assets (MiniMax H3 Fun ControlNet) follow the same split:
an OPTIONAL chain-wide video is wired here ('fun_control_video', plus the
optional 'fun_control_mask' / 'fun_control_source_video' inpaint pair) and
each segment decides whether it is controlled, how strongly and over which
sigma range. They travel to the Extend node over their own typed link
('control_image_slots', MMH3_CONTROL_SLOTS) rather than inside the config,
because control images and reference pictures must never be interchangeable -
a reference picture becomes a '<Picture i>' the prompt can name, a control
image drives the Fun ControlNet. Nothing is decoded here either: the Extend
node slices each controlled segment's window out of the VIDEO and encodes it.

Concepts:
- Segment 0 ("first") starts a fresh generation: FL2VA (first frame + optional
  last frame) or Ref2VA. Its width/height define the whole chain's resolution
  and are LOCKED once a session has stored latents/previews.
- Segments >= 1 continue the previous segment: the previous merged latent's
  tail becomes the new segment's head (frozen + fade, see Temporal Extend
  Video). Conditioning per segment: Ref2VA (manual images or a frame taken
  from the previous segment), FL2VA with ONLY a last-frame keyframe (the
  frame-0 keyframe slot is reserved for the seam anchor), or no reference at
  all (prompt-only; the frozen tail still anchors the content).
- Frame counts: the FIRST segment's length must be 17n+5 (the model's grid);
  later segments specify the NEW frames to generate (a multiple of 17) - the
  segment's total length (carried tail 17m+5 + new 17k) automatically lands
  back on the 17n+5 grid.
"""

import json

from comfy_api.latest import io

FPS = 24  # MiniMax H3 output fps (comfy_extras/nodes_minimax_h3.py)

# Canonical names of the Autogrow reference-image sockets (see the module
# docstring). Slot 0 is ALSO segment 0's FL2VA first frame and slot 1 its
# FL2VA last frame, hence the 'First_or_' / 'Last_or_' prefixes. These are
# the authoritative names: MMH3 Temporal Extend Video imports this list
# instead of repeating the strings, and the JS editor reads the names straight
# off the node's own sockets, so nothing can drift out of sync.
SOCKET_COUNT = 16
REF_SOCKET_NAMES = (
    ["First_or_Ref_Image_0", "Last_or_Ref_Image_1"]
    + [f"Ref_Image_{i}" for i in range(2, SOCKET_COUNT)]
)
REF_SOCKET_RANK = {name: i for i, name in enumerate(REF_SOCKET_NAMES)}

# Custom link type carrying the wired sockets to the Extend node (see the
# module docstring). Plain IMAGE would either stretch the images or lose the
# per-socket boundaries - and io.Custom links are already the house style for
# structured payloads here (MMH3_SAMPLE_PARAMS, MMH3_OVERLAP_PARAMS).
REF_IMAGES = io.Custom("MMH3_REF_IMAGES")
REF_IMAGE_SLOTS_VERSION = 1

# Custom link carrying the FUN CONTROL assets (the control video plus the
# optional inpaint mask / source video) from this node to the Extend node.
# Same versioned-payload mechanics as MMH3_REF_IMAGES - and deliberately NOT
# the same TYPE: a reference picture becomes a '<Picture i>' the prompt can
# name, a control image drives MiniMaxH3FunControlNet. Sharing one link type
# would let the two be cross-wired and silently reinterpreted, so they travel
# on their own socket pair. The Extend node also checks the payload's 'kind',
# which turns a hand-made / mixed-up link into a clear error instead of a
# confusing KeyError.
CONTROL_SLOTS = io.Custom("MMH3_CONTROL_SLOTS")
CONTROL_BUNDLE_VERSION = 1
CONTROL_BUNDLE_KIND = "mmh3_fun_control"

# Per-segment fun-control mode:
#   'off'       - this segment is not controlled at all
#   'auto_crop' - the slice of the wired 'fun_control_video' covering this
#                 segment's ABSOLUTE span on the chain's timeline: it starts
#                 one CARRIED TAIL before the segment's own boundary (the
#                 piece re-covers that tail, so the control slice must too)
#                 and runs for the whole piece (carried tail + new frames).
#                 Abutting slices therefore reassemble the control video
#                 continuously instead of every segment restarting at its
#                 first frame (which is what the native Apply node does).
#   'whole'     - the control video from its FIRST frame, clamped/truncated
#                 to this piece - the native node's behavior, and the right
#                 choice for a still image / a looping control signal.
CONTROL_MODES = ["off", "auto_crop", "whole"]
DEFAULT_CONTROL_STRENGTH = 1.0
DEFAULT_CONTROL_START = 0.0
DEFAULT_CONTROL_END = 1.0

# ── "direct reference in frame" (a second video spliced onto a segment's
#    sampling canvas, then cut back off) ──
# A segment's SAMPLING canvas can carry a strip of another video spliced along
# one edge ("material" in the code, "direct reference in frame" in the UI). The
# strip is written straight into the video latent and pinned by the video
# noise-mask, so the model only GENERATES the remaining area while SEEING the
# strip as
# pixel-exact conditioning - H3's attention is dense and takes no mask, so the
# strip's K/V rows are exactly what carries the source motion into the
# generated part. That makes this the most stable way to drive a generated
# character with a reference performance (the strip is literal pixels, not a
# reference block re-encoded at its own resolution), at the price of paying for
# the strip's rows in every sampling step.
#
# The record is PER SEGMENT ('side' == 'off' is that segment's switch), so a
# segment can turn the mode on whether or not the one before it did: its piece
# is spliced over the carried tail too, and the frames the previous segment
# generated carry the strip from this segment's overlap onwards. The canvas is
# per segment too - each one splices, samples and cuts its own - so the sides
# may differ between segments; only the GENERATION AREA is chain-wide. See
# _material_chain and the Extend node's layout.
MATERIAL_SIDES = ["off", "left", "right", "top", "bottom"]
DEFAULT_MATERIAL_SIDE = "off"
# The seam is never a hard edge: a band of the strip at the boundary is left
# FREE so the model can synthesize the transition into the generated area
# (an unprotected butt seam shows as a line / a torn edge after decode).
# Quantized to 32 px because the DiT pools the pixel mask through a 2x2 patch
# on a 16x-downsampled latent - ONE token column IS 32 px (see
# _pool_masks_to_token_grid), so every value in 1..32 is the same column and
# an arbitrary number would just lie about what it does.
DEFAULT_MATERIAL_EXPOSE = 32

CONDITION_MODES_FIRST = ["FL2VA", "Ref2VA"]
# Later segments: FL2VA = last-frame-only keyframe (seam owns frame 0).
CONDITION_MODES_LATER = ["Ref2VA", "FL2VA"]
REF_SOURCES_FIRST = ["manual", "none"]
REF_SOURCES_LATER = ["manual", "prev_frame", "none"]
# per-segment audio reference mode: 'load' = picked ref_audio file,
# 'prev' = previous segment's soundtrack (latent-level, no VAE),
# 'initial' = first segment's soundtrack (latent-level, no VAE),
# 'bgm' = the segment's OWN soundtrack is cut out of the 'audio_BGM' socket:
#   each segment takes the slice covering its absolute position on the chain's
#   timeline (the sum of the PRECEDING segments' frame counts decides where it
#   starts), frozen by the audio noise-mask. The slice is encoded on demand
#   (one segment at a time - the track is never encoded in one piece), so peak
#   VRAM tracks a single segment instead of the whole song. A chain whose
#   segments all use 'bgm' therefore ends up with the BGM's head as its
#   soundtrack - not a model rendition of it. When the music ends mid-chain
#   the audio noise-mask simply drops to 1 past the BGM's end: the model has
#   been hearing the music up to that frame, so the free tail sounds like a
#   natural continuation rather than a hard cut.
AUDIO_REF_MODES = ["none", "load", "prev", "initial", "bgm"]

DEFAULT_WIDTH = 768
DEFAULT_HEIGHT = 768
DEFAULT_FIRST_FRAMES = 124      # ~5.2 s, 17*7+5
DEFAULT_NEW_FRAMES = 102        # ~4.25 s per later segment, 17*6

DEFAULT_TAIL_FRAMES = 39
DEFAULT_FADE_FRAMES = 0
DEFAULT_TAIL_MODE = "freeze_fade"
DEFAULT_FADE_VALUE = 0.5
DEFAULT_FADE_MODE = "flat"
DEFAULT_ANCHOR_SEAM = True
DEFAULT_ANCHOR_STRENGTH = 0.999
DEFAULT_KEYFRAMES_MODE = "reanchor"
DEFAULT_REF_IMAGE_SIZE = "match"


def snap_17n5(n):
    """Snap up to the nearest 17n+5 frame count (min 5)."""
    n = max(5, int(n))
    return n + ((5 - n) % 17)


def snap_17(n):
    """Snap up to the nearest multiple of 17 (min 17)."""
    n = max(17, int(n))
    return n + ((-n) % 17)


def frames_to_seconds(frames):
    return round(int(frames) / FPS, 3)


def seconds_to_frames_hint(seconds):
    return max(1, round(float(seconds) * FPS))


def _norm_seed(v):
    try:
        s = int(v)
    except (TypeError, ValueError):
        return 0
    return s & 0xFFFFFFFFFFFFFFFF


def _norm_bool(v, default):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    return default


def _norm_float(v, default, lo=None, hi=None):
    """Float with a fallback; NaN/inf and anything unparsable fall back too.
    ``lo``/``hi`` (when given) clamp instead of reject, so a slider that was
    dragged out of range in an older build still loads."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        f = float(default)
    if f != f or f in (float("inf"), float("-inf")):
        f = float(default)
    if lo is not None:
        f = max(float(lo), f)
    if hi is not None:
        f = min(float(hi), f)
    return f


def _mute_stop(raw, count):
    """The chain's stop point (the editor's 'mute' button).

    ONE stop point per editor, not a flag per segment: every segment from the
    stop on is CLOSED - the run samples nothing there and the output is the
    merged latent through the last active segment. Returns -1 when the whole
    chain runs, else the index the chain stops BEFORE (0..count-1). A value
    that is missing, of the wrong type, or past the end of a shorter chain
    reads as -1, exactly like the editor's muteStopIndex().
    """
    m = raw.get("mute_from_segment", -1) if isinstance(raw, dict) else -1
    if isinstance(m, bool) or not isinstance(m, int):
        return -1
    if m < 0 or m >= count:
        return -1
    return m


def _sanitize_control(rs):
    """Normalize one segment's fun-control settings.

    The mode decides whether the segment is controlled at all; the three
    numbers are handed to ``MiniMaxH3FunControlPatch`` verbatim (strength is
    a multiplier, start/end are SIGMA percentages of the model's own table).
    ``end < start`` collapses to an empty window - never an error, and never
    a silently INVERTED window (which would control the opposite range)."""
    mode = rs.get("control_mode", "off")
    if mode not in CONTROL_MODES:
        mode = "off"
    start = _norm_float(rs.get("control_start"), DEFAULT_CONTROL_START,
                        lo=0.0, hi=1.0)
    end = _norm_float(rs.get("control_end"), DEFAULT_CONTROL_END,
                      lo=0.0, hi=1.0)
    if end < start:
        end = start
    return {
        "control_mode": mode,
        "control_strength": _norm_float(rs.get("control_strength"),
                                        DEFAULT_CONTROL_STRENGTH, lo=0.0),
        "control_start": start,
        "control_end": end,
    }


def _sanitize_material(rs):
    """Normalize ONE segment's 'direct reference in frame' record.

    The whole record is per segment: whether this segment uses the mode at all
    ('side' == 'off' IS the switch), which edge its strip is spliced along, how
    wide the free band at its seam is, and 'cut_at_split' (drop the strip at the
    HIGH -> LOW handoff instead of keeping it for the whole schedule - the
    second switch, and the only one that changes WHEN the strip is there).
    Segments are independent - turning the mode on here needs nothing from the
    segment before it (the carried tail is spliced over as well, see the Extend
    node's _freeze_material), and a chain where only SOME segments use it is
    legal.

    The canvas is per segment too: the strip is spliced onto THIS segment's
    sampling canvas and cut back off once it has been sampled (the Extend node's
    _expand_canvas / _crop_canvas), so two segments may splice along different
    edges without any conflict - see _material_chain, which derives the chain
    view this record is aggregated into."""
    m = rs.get("material")
    if not isinstance(m, dict):
        m = {}
    side = m.get("side", DEFAULT_MATERIAL_SIDE)
    if side not in MATERIAL_SIDES:
        side = DEFAULT_MATERIAL_SIDE
    try:
        expose = int(m.get("expose", DEFAULT_MATERIAL_EXPOSE))
    except (TypeError, ValueError):
        expose = DEFAULT_MATERIAL_EXPOSE
    expose = max(0, round(expose / 32) * 32)
    cut = m.get("cut_at_split", False)
    if isinstance(cut, str):
        cut = cut.strip().lower() in ("1", "true", "yes", "on")
    else:
        cut = cut is True
    return {"material": {"side": side, "expose": expose,
                         "cut_at_split": cut}}


def _material_chain(segments):
    """Derived chain-wide view of the per-segment records (observability).

    There is no single answer to 'the chain's side' any more: every segment
    splices its own strip onto its OWN sampling canvas and gets it cut back off
    before the next segment runs, so 'sides' is the inventory (side -> segments)
    and 'side' is only the first enabling segment's, kept as a convenience for
    callers that want one label. The Extend node reads the per-segment records
    themselves - this mirror is for the config / segments_info, not a second
    source of truth: a caller can see what the chain will do without running the
    model."""
    on = [(s.get("index", i), s.get("material") or {})
          for i, s in enumerate(segments)
          if (s.get("material") or {}).get("side", "off") != "off"]
    if not on:
        return {"side": "off", "expose": DEFAULT_MATERIAL_EXPOSE,
                "segments": [], "sides": {}, "cut_at_split": []}
    sides = {}
    for i, r in on:
        sides.setdefault(r.get("side", "off"), []).append(i)
    first = on[0][1]
    return {"side": first.get("side", "off"),
            "expose": first.get("expose", DEFAULT_MATERIAL_EXPOSE),
            "segments": [i for i, _ in on], "sides": sides,
            # per segment like everything else: the segments that drop their
            # strip at the split, whatever the segment next to them does
            "cut_at_split": [i for i, r in on if r.get("cut_at_split")]}


# ── reference image sockets (Autogrow) ──
# ComfyUI hands the whole group over as ONE dict, keyed by the bare socket
# label: {"First_or_Ref_Image_0": tensor, "Ref_Image_2": tensor, ...}
# (unconnected slots are absent or None).
def _socket_slot_names(ref_images):
    """Connected sockets in canonical slot order (e.g.
    ['First_or_Ref_Image_0', 'Ref_Image_2']).

    Order comes from REF_SOCKET_NAMES rather than from the numeric suffix, so
    a key the schema does not know about is ignored instead of mis-slotted."""
    if not isinstance(ref_images, dict):
        return []
    return sorted([n for n, v in ref_images.items()
                   if v is not None and isinstance(n, str)
                   and n in REF_SOCKET_RANK],
                  key=REF_SOCKET_RANK.__getitem__)


def _collect_socket_slots(ref_images, names):
    """Bundle the connected sockets for the Extend node.

    Returns ``{"version": 1, "slots": [{"name", "slot", "count", "images"},
    ...]}`` in canonical slot order (empty ``slots`` when nothing is wired).

    The tensors travel exactly as they arrived - native resolution, native
    batch size, unmodified - so the consumer can apply each segment's
    ``ref_image_size`` (aspect-preserving, same code path as the picked
    files). ``count`` exposes the socket's batch size; per
    MiniMaxH3ReferenceToVideo only the FIRST image of a socket is used as a
    reference (one socket == one '<Picture i>')."""
    slots = []
    if isinstance(ref_images, dict):
        for name in names:
            t = ref_images.get(name)
            if t is None or not hasattr(t, "shape") or not int(t.shape[0]):
                continue
            slots.append({"name": name,
                          "slot": REF_SOCKET_RANK[name],
                          "count": int(t.shape[0]),
                          "images": t})
    return {"version": REF_IMAGE_SLOTS_VERSION, "slots": slots}


# Tail/fade/anchor/overlap parameters. These are PER-SEGMENT from the JS
# editor (segments >= 1); the global dict is only the fallback for segments
# that carry no per-segment extend_params (old workflows / JS defaults).
DEFAULT_EXTEND = {
    "tail_frames": DEFAULT_TAIL_FRAMES,
    "tail_mode": DEFAULT_TAIL_MODE,
    "fade_frames": DEFAULT_FADE_FRAMES,
    "fade_value": DEFAULT_FADE_VALUE,
    "fade_mode": DEFAULT_FADE_MODE,
    "keyframes_mode": DEFAULT_KEYFRAMES_MODE,
    "anchor_seam": DEFAULT_ANCHOR_SEAM,
    "anchor_strength": DEFAULT_ANCHOR_STRENGTH,
    "overlap_mode": "later",        # 'later' = new segment wins the overlap band
    "overlap_blend": "linear",      # linear | smoothstep | midpoint | overwrite
}


def _sanitize_extend(g, fb):
    """Normalize an extend-params dict; missing/invalid keys fall back to `fb`."""
    g = g if isinstance(g, dict) else {}
    tail_frames = snap_17n5(g.get("tail_frames", fb["tail_frames"]))
    # 0 = no fade: the whole carried tail stays frozen and the new content
    # hard-cuts in. snap_17() has a min of 17, so keep 0 explicit.
    ff = int(g.get("fade_frames", fb["fade_frames"]))
    fade_frames = 0 if ff <= 0 else min(snap_17(ff), tail_frames)
    return {
        "tail_frames": tail_frames,
        "tail_mode": g.get("tail_mode", fb["tail_mode"])
                     if g.get("tail_mode") in ("freeze_fade", "free_resample")
                     else fb["tail_mode"],
        "fade_frames": fade_frames,
        "fade_value": float(g.get("fade_value", fb["fade_value"])),
        "fade_mode": g.get("fade_mode", fb["fade_mode"])
                     if g.get("fade_mode") in ("flat", "gradient", "smoothstep")
                     else fb["fade_mode"],
        "keyframes_mode": g.get("keyframes_mode", fb["keyframes_mode"])
                          if g.get("keyframes_mode") in ("reanchor", "drop", "keep")
                          else fb["keyframes_mode"],
        "anchor_seam": _norm_bool(g.get("anchor_seam"), fb["anchor_seam"]),
        "anchor_strength": float(g.get("anchor_strength", fb["anchor_strength"])),
        "overlap_mode": g.get("overlap_mode", fb["overlap_mode"])
                        if g.get("overlap_mode") in ("earlier", "later")
                        else fb["overlap_mode"],
        "overlap_blend": g.get("overlap_blend", fb["overlap_blend"])
                         if g.get("overlap_blend") in ("linear", "smoothstep",
                                                       "midpoint", "overwrite")
                         else fb["overlap_blend"],
    }


class MMH3TemporalTileEditor(io.ComfyNode):
    """Per-segment configuration for MMH3 Temporal Extend Video."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3TemporalTileEditor",
            display_name="MMH3 Temporal Tile Editor",
            category="model/conditioning/minimax",
            description=(
                "Per-segment configuration for 'MMH3 Temporal Extend Video'. "
                "Dock-panel editor: each segment carries its own conditioning "
                "mode, prompt, seed, reference images and duration. Segment 0 "
                "generates a fresh video (FL2VA or Ref2VA) and its resolution "
                "is locked for the whole chain; later segments continue the "
                "previous tail (Ref2VA / FL2VA last-frame-only / no reference) "
                "and may take a frame of the previous segment as reference. "
                "Previews of past runs let you pick a segment to re-generate "
                "from without redoing the earlier ones. A segment may also "
                "take its whole soundtrack out of the optional 'audio_BGM' "
                "socket ('bgm' audio reference mode): the music is then cut "
                "per segment and frozen, so the finished chain's audio is the "
                "music itself. Pure-information node: no images are loaded and "
                "nothing is encoded here - the 'audio_BGM' audio is forwarded "
                "to the Extend node verbatim. When a run "
                "contains multiple segments to sample and none of them "
                "references the previous segment's frames, all statically "
                "encodable CLIP/VAE conditioning (prompts, reference images, "
                "first/last frames) is encoded up front before any H3 "
                "sampling starts, so the run does not ping-pong between the "
                "text encoder and the DiT mid-run. Segments referencing the "
                "previous segment's output ('prev_frame' reference, or the "
                "'prev_tail' seam reference beyond the first sampled "
                "segment) are still encoded inline - their inputs only "
                "exist once the earlier segments have been sampled."
            ),
            search_aliases=["h3 temporal tile editor", "temporal editor",
                            "multi segment video", "video continuation"],
            inputs=[
                io.Boolean.Input("show_editor", default=True,
                                 tooltip="Toggle the visual dock editor panel on/off."),
                io.String.Input("session_name", default="session1",
                                tooltip="Session directory name: <temp|output/latents>/mmh3_temporal/<name>/ holding the 'latents' and 'previews' subdirectories. Every session of a storage location shares one reference/conditioning cache in the sibling 'cache' directory."),
                io.Combo.Input("storage_location", options=["temp", "output/latents"], default="temp",
                               tooltip="Where per-segment latents (.h3latent) and preview WebPs go: <location>/mmh3_temporal/<session>/{latents,previews}, beside a shared 'cache' of reference latents and encoded conditioning reused by every session. 'temp' is cleared on restart; 'output/latents' persists."),
                io.String.Input("base_prompt", default="",
                                multiline=True, dynamic_prompts=True,
                                tooltip="Shared prompt prefix applied to all segments."),
                io.String.Input("base_negative", default="",
                                multiline=True, dynamic_prompts=True,
                                tooltip="Shared negative prompt applied to all segments."),
                io.String.Input("tile_data", default="{}",
                                tooltip="Internal JSON managed by the dock editor."),
                io.Autogrow.Input(
                    "ref_images",
                    # TemplateNames (not TemplatePrefix) because the display
                    # names are not a uniform prefix: slot 0 is the FL2VA
                    # first frame, slot 1 the last frame, the rest are plain
                    # references. min=0 so exactly ONE empty socket shows up
                    # (the frontend renders min + 1 slots).
                    template=io.Autogrow.TemplateNames(
                        input=io.Image.Input("ref", optional=True),
                        names=REF_SOCKET_NAMES, min=0),
                    optional=True,
                    tooltip="Direct IMAGE sockets: 'First_or_Ref_Image_0', "
                            "'Last_or_Ref_Image_1', then 'Ref_Image_2'... "
                            "Wiring LoadImage here is an alternative to "
                            "picking files in the dock panel. Every segment "
                            "uses the connected images as 'load images' "
                            "references by default; slot 0 doubles as segment "
                            "0's FL2VA first frame and slot 1 as the FL2VA "
                            "last frame (when no file is picked for them). "
                            "Remove an image from one segment only in the "
                            "editor's reference row (it stays available to "
                            "the other segments). One socket is one "
                            "reference picture: an input carrying a batch "
                            "contributes its FIRST image only (the same "
                            "convention as MiniMaxH3 Reference to Video), so "
                            "the prompt's '<Picture i>' numbering is one per "
                            "socket. Images are never resized or merged "
                            "here - the Extend node sizes each one by the "
                            "segment's 'reference image size'. The row order "
                            "IS the prompt "
                            "order: wired sockets come first (slot 0, slot 1, "
                            "...), the picked files after them, left to right "
                            "matching the '<Picture 1>', '<Picture 2>', ... "
                            "the Ref2VA prompt has to name (the editor badges "
                            "each chip with its index)."),
                io.Video.Input("ref_video_input", optional=True,
                               tooltip="OPTIONAL reference video as a ComfyUI "
                                       "VIDEO (a file, not a decoded frame "
                                       "tensor) - the same socket type the "
                                       "native 'MiniMaxH3 Reference to Video' "
                                       "family and the video Load nodes speak. "
                                       "By keeping it a file, the Extend node "
                                       "only DECODES the frames each segment "
                                       "actually needs (no huge frame tensor "
                                       "ever materializes in memory). Set a "
                                       "segment's ref-video mode to 'auto crop "
                                       "input ref': each such segment then "
                                       "takes the slice covering its ABSOLUTE "
                                       "span on the chain's timeline (the sum "
                                       "of the PRECEDING segments' frames "
                                       "decides where it starts, its own "
                                       "new-frames decide the length), so the "
                                       "segment's reference video is that slice "
                                       "of the input. If the video carries an "
                                       "audio track and the 'audio_BGM' input is "
                                       "NOT connected, the video's own audio is "
                                       "used as the background music. "
                                       "SECOND ROLE - direct reference in "
                                       "frame: on any segment whose "
                                       "'direct reference in frame' side is "
                                       "set to left/right/top/bottom, this "
                                       "same video is ALSO the material an "
                                       "edge strip of that segment's "
                                       "SAMPLING canvas is spliced from "
                                       "(resized to the strip, never "
                                       "stretched out of shape) and frozen by "
                                       "the video noise mask, so the model "
                                       "generates only the remaining area "
                                       "while seeing the strip as literal "
                                       "pixels. The strip is scaffolding: it "
                                       "is cut back off once the segment has "
                                       "been sampled, so the output stays the "
                                       "generation area and the next segment "
                                       "inherits no material. "
                                       "A segment cannot do both at "
                                       "once: 'auto crop input ref' plus its "
                                       "own direct-reference side is "
                                       "rejected. "
                                       "Unconnected = 'auto crop input ref' "
                                       "segments, and any segment with a "
                                       "direct-reference side set, are "
                                       "rejected."),
                io.Audio.Input("audio_bgm", optional=True,
                               tooltip="OPTIONAL background music for the whole "
                                       "chain - a song, roughly as long as the "
                                       "finished video. Wire it here and set a "
                                       "segment's audio reference mode to "
                                       "'bgm': that segment's soundtrack is cut "
                                       "out of this audio at its own position on "
                                       "the chain's timeline (the PRECEDING "
                                       "segments' frame counts decide the "
                                       "offset), encoded through the Extend "
                                       "node's audio VAE and FROZEN by the audio "
                                       "noise-mask - the segment's audio IS this "
                                       "music and the model only generates video "
                                       "on top of it (it still hears the music). "
                                       "The music is encoded ONE SEGMENT AT A "
                                       "TIME (only the span that segment needs "
                                       "reaches the audio VAE), so peak VRAM "
                                       "tracks a single segment instead of the "
                                       "whole track. "
                                       "Segments that do not use 'bgm' keep "
                                       "their own soundtrack, so the mode can be "
                                       "switched per segment. When the BGM ends "
                                       "before the finished video, the rest is "
                                       "generated freely (model continues the "
                                       "music). The Extend node still errors "
                                       "out before sampling when 'bgm' is used "
                                       "without an audio VAE. Unconnected = "
                                       "'bgm' segments are rejected."),
                io.Video.Input("fun_control_video", optional=True,
                               tooltip="OPTIONAL fun-control video (MiniMax H3 "
                                       "Fun ControlNet) for the whole chain - a "
                                       "ComfyUI VIDEO (a file, not a decoded "
                                       "frame tensor), the same socket type the "
                                       "reference-video input and the video "
                                       "Load nodes speak. By keeping it a file, "
                                       "the Extend node only DECODES the frames "
                                       "each segment actually needs (no huge "
                                       "frame tensor ever materializes in "
                                       "memory). Set a segment's fun-control "
                                       "mode to 'auto crop input control': that "
                                       "segment's control slice is the part of "
                                       "this video covering its ABSOLUTE span on "
                                       "the chain's timeline, extended backwards "
                                       "by its carried tail - so abutting slices "
                                       "reassemble the control video "
                                       "continuously. 'whole control video' "
                                       "instead always starts at the video's "
                                       "first frame (the native node's "
                                       "behavior; right for a still image). "
                                       "This node does NOT decode, resize or "
                                       "encode anything here - the raw VIDEO "
                                       "object travels to the Extend node, "
                                       "which slices/encodes per segment. "
                                       "Unconnected = segments whose mode is not "
                                       "'off' are rejected."),
                io.Mask.Input("fun_control_mask", optional=True,
                              tooltip="OPTIONAL inpaint mask for the fun-control "
                                      "pass - the native Apply node's 'mask' "
                                      "input (1 = regenerate this region). One "
                                      "mask frame per chain frame, aligned to "
                                      "the chain's timeline EXACTLY like the "
                                      "control video: each controlled segment "
                                      "takes the frames covering its own span "
                                      "(its carried tail included), and a mask "
                                      "with fewer frames than the chain freezes "
                                      "its LAST frame from there on - a single "
                                      "frame therefore means the same static "
                                      "mask for the whole chain, exactly like "
                                      "the native node. Only read on segments "
                                      "whose fun-control mode is not 'off', and "
                                      "only together with 'fun_control_source_"
                                      "video' is the masked-out content fed back "
                                      "as the inpaint source."),
                io.Video.Input("fun_control_source_video", optional=True,
                               tooltip="OPTIONAL source video behind the "
                                       "'fun_control_mask' (what gets preserved "
                                       "where the mask is 0) - the native Apply "
                                       "node's 'source_video'. A ComfyUI VIDEO "
                                       "(a file, decoded per segment just like "
                                       "the control video, on the SAME window). "
                                       "Only read when a mask is connected: "
                                       "without one nothing is masked out, so "
                                       "there is nothing to source. Unconnected "
                                       "with a mask connected = the masked-out "
                                       "region is inpainted from scratch."),
            ],
            outputs=[
                io.Dict.Output("temporal_tile_config",
                               tooltip="Complete per-segment configuration for MMH3 Temporal Extend Video."),
                io.Dict.Output("segments_info",
                               tooltip="Inspectable summary: session, storage, segment count and per-segment parameters."),
                REF_IMAGES.Output("ref_image_slots",
                                  tooltip="The connected reference sockets, one record per socket (name, slot, batch size, image tensor at its own resolution) in slot order - connect to 'MMH3 Temporal Extend Video' (its 'ref_image_slots' input). Images are NOT resized or merged here: the Extend node sizes each one per segment ('match' / 'max'), exactly like the picked files. Empty when nothing is wired."),
                CONTROL_SLOTS.Output("control_image_slots",
                                     tooltip="The fun-control assets wired into this node ('fun_control_video' + optional 'fun_control_mask' / 'fun_control_source_video') bundled for 'MMH3 Temporal Extend Video' (its 'control_image_slots' input). Nothing is decoded, resized or encoded here: the Extend node slices each controlled segment's window out of the video and encodes it as that segment's control hint. The bundle also carries an 'absent' payload when only the mask is wired. Left unconnected = controlled segments are rejected with a clear error."),
            ],
        )

    @classmethod
    def validate_inputs(cls, **kwargs) -> bool | str:
        # Runs at QUEUE time (before any model loads): reject a resume point
        # that goes BEYOND the chain, so the user gets an immediate error
        # instead of one after the models have been loaded. resume ==
        # len(segs) (every segment locked) is VALID on purpose: the executor
        # then outputs the stored merged latent without sampling anything.
        tile_data = kwargs.get("tile_data") or "{}"
        try:
            raw = json.loads(tile_data)
        except (json.JSONDecodeError, TypeError):
            return True  # execute() tolerates malformed JSON the same way
        if not isinstance(raw, dict):
            return True
        segs = raw.get("segments")
        resume = raw.get("resume_from_segment", 0)
        if (isinstance(segs, list) and segs
                and isinstance(resume, int) and resume > len(segs)):
            return (f"resume_from_segment={resume} goes beyond the last "
                    f"segment ({len(segs)}) - nothing to sample. Unlock one "
                    "or add a new segment in the Tile Editor.")
        # a stop point ON segment 0 closes the whole chain: not a single
        # segment is left to sample, so this is the same "nothing to run"
        # state - caught here, before the models are loaded
        if isinstance(segs, list) and segs and _mute_stop(raw, len(segs)) == 0:
            return ("every segment is closed (mute): segment 0 is the chain's "
                    "stop point, so there is nothing to sample - unmute a "
                    "segment (or add one) in the Tile Editor.")
        return True

    @classmethod
    def execute(cls, show_editor=True, session_name="session1",
                storage_location="temp",
                base_prompt="", base_negative="",
                tile_data="{}", ref_images=None, ref_video_input=None,
                audio_bgm=None, fun_control_video=None, fun_control_mask=None,
                fun_control_source_video=None) -> io.NodeOutput:
        raw = {}
        if tile_data:
            try:
                raw = json.loads(tile_data)
            except (json.JSONDecodeError, TypeError):
                raw = {}
        if not isinstance(raw, dict):
            raw = {}

        # ── session / storage (widget wins over stale JS state) ──
        storage = storage_location if storage_location in ("temp", "output/latents") \
            else raw.get("storage", "temp")
        session_name = (session_name or raw.get("session_name") or "session1").strip() or "session1"
        # resume point is JS-managed (preview click); sanitize
        resume = raw.get("resume_from_segment", 0)
        resume = resume if isinstance(resume, int) and resume > 0 else 0

        # ── global extend parameters: FALLBACK for segments that do not
        #    carry their own per-segment extend_params ──
        g = raw.get("extend_params", {})
        extend_params = _sanitize_extend(g, DEFAULT_EXTEND)

        # ── segments ──
        # reference image sockets are shared by every segment; each segment
        # may rule individual sockets out (ref_socket_off) in the dock panel
        sock_names = _socket_slot_names(ref_images)
        raw_segs = raw.get("segments")
        if not isinstance(raw_segs, list) or not raw_segs:
            raw_segs = [{}]
        segments = []
        for i, rs in enumerate(raw_segs):
            rs = rs if isinstance(rs, dict) else {}
            first = (i == 0)
            mode = rs.get("mode", "FL2VA" if first else "Ref2VA")
            allowed = CONDITION_MODES_FIRST if first else CONDITION_MODES_LATER
            if mode not in allowed:
                mode = allowed[0]
            if first:
                new_frames = snap_17n5(rs.get("new_frames", DEFAULT_FIRST_FRAMES))
            else:
                new_frames = snap_17(rs.get("new_frames", DEFAULT_NEW_FRAMES))
            ref_source = rs.get("ref_source")
            allowed_rs = REF_SOURCES_FIRST if first else REF_SOURCES_LATER
            if ref_source not in allowed_rs:
                # degrade: keep 'manual' only when something can actually be
                # referenced - picked files or connected image sockets
                ref_source = "manual" if (rs.get("ref_images") or sock_names) \
                    else "none"
            if ref_source not in allowed_rs:
                ref_source = "none"
            ref_files = [str(x) for x in (rs.get("ref_images") or []) if x]
            # sockets this segment rules out (the 'x' on a dummy chip in the
            # reference row); stale names are silently dropped
            off = set(str(x) for x in (rs.get("ref_socket_off") or []))
            ref_socket_off = [n for n in sock_names if n in off]
            if ref_source == "manual" and not ref_files \
                    and not [n for n in sock_names if n not in off]:
                ref_source = "none"
            # audio reference mode: legacy configs without an explicit mode
            # but with a picked file degrade to 'load' (the pre-modes
            # behavior). seg0 has no previous/first audio to reference, so
            # 'prev'/'initial' degrade there. Only 'load' forwards the
            # picked file - 'none' must not resurrect a stale file, and
            # 'prev'/'initial' reference the latent directly (a file would
            # become an EXTRA standalone audio reference). The JS keeps the
            # file remembered in its own state for when the user switches
            # back to 'load'.
            ref_audio = str(rs.get("ref_audio", "") or "")
            arm = rs.get("ref_audio_mode") or ("load" if ref_audio else "none")
            if arm not in AUDIO_REF_MODES:
                arm = "load" if ref_audio else "none"
            if first and arm in ("prev", "initial"):
                arm = "load" if ref_audio else "none"
            if arm != "load":
                ref_audio = ""
            prev_frame_index = rs.get("prev_frame_index", -1)
            try:
                prev_frame_index = int(prev_frame_index)
            except (TypeError, ValueError):
                prev_frame_index = -1
            seg = {
                "index": i,
                "first": first,
                "mode": mode,
                "seed": _norm_seed(rs.get("seed", 0)),
                "prompt": str(rs.get("prompt", "")),
                "negative": str(rs.get("negative", "")),
                "ref_source": ref_source,
                "ref_images": ref_files,
                "ref_image_size": rs.get("ref_image_size", DEFAULT_REF_IMAGE_SIZE)
                                  if rs.get("ref_image_size") in ("match", "max")
                                  else DEFAULT_REF_IMAGE_SIZE,
                "fl2va_first_image": str(rs.get("fl2va_first_image", "")),
                "fl2va_last_image": str(rs.get("fl2va_last_image", "")),
                "prev_frame_index": prev_frame_index,
                # MiniMaxH3ReferenceToVideo-style extra references (file names
                # in the ComfyUI input directory, loaded at execution time)
                # ref_video_mode: 'load' = the picked ref_video file (see
                # below); 'auto_crop' = take a slice of the wired
                # ref_video_input socket covering this segment's absolute span
                # on the chain's timeline.
                "ref_video": str(rs.get("ref_video", "") or ""),
                "ref_video_mode": rs.get("ref_video_mode", "load")
                                   if rs.get("ref_video_mode") in ("load", "auto_crop")
                                   else "load",
                "ref_video_audio": str(rs.get("ref_video_audio", "") or ""),
                "ref_audio": ref_audio,
                "ref_audio_mode": arm,
                # connected image sockets this segment does NOT use as a
                # reference (slot names, e.g. ['Ref_Image_2'])
                "ref_socket_off": ref_socket_off,
                "new_frames": new_frames,
                "new_seconds": frames_to_seconds(new_frames),
            }
            # fun control is per-segment on a chain-wide asset: the segment
            # only decides whether it is controlled, how strongly and over
            # which sigma range - the video itself comes from the node input
            # (see _sanitize_control / CONTROL_MODES).
            seg.update(_sanitize_control(rs))
            # direct reference in frame: the WHOLE record is per segment -
            # 'side' == 'off' is this segment's switch (see _sanitize_material
            # / MATERIAL_SIDES)
            seg.update(_sanitize_material(rs))
            if first:
                w = int(rs.get("width", DEFAULT_WIDTH) or DEFAULT_WIDTH)
                h = int(rs.get("height", DEFAULT_HEIGHT) or DEFAULT_HEIGHT)
                seg["width"] = max(32, round(w / 32) * 32)
                seg["height"] = max(32, round(h / 32) * 32)
            else:
                # the ONLY per-segment continuation parameter: the carried-
                # tail length ("overlap frames"). Everything else is global
                # (MMH3 Temporal Overlap Params sub-node). Legacy per-seg
                # extend_params.tail_frames is still accepted.
                sp = rs.get("extend_params")
                raw_of = rs.get("overlap_frames")
                if raw_of is None and isinstance(sp, dict):
                    raw_of = sp.get("overlap_frames", sp.get("tail_frames"))
                if raw_of is not None:
                    try:
                        of = int(raw_of)
                        # 0 (or anything below it) = HARD CUT: no fade and no
                        # blend across the seam, the new content cutting in at
                        # the boundary. It is NOT a lattice point - the lattice
                        # is 17n+5 with a floor of 5 - so snap_17n5() would turn
                        # that sentinel into a real 5-frame tail and the flag
                        # would be gone before the Extend node ever read it.
                        # Keep 0 explicit here, the same way fade_frames is kept
                        # explicit in _sanitize_extend, and mirror both the JS
                        # editor (overlapFramesValue) and the Extend node
                        # (_seg_ep). The sentinel travels as 0; realizing it is
                        # the Extend node's business, and it realizes the
                        # smallest legal seam (seam_split) - the token-group
                        # grid cannot host a boundary at the raw end of the
                        # chain, and a piece spliced in off that grid decodes on
                        # the wrong group phase.
                        seg["overlap_frames"] = 0 if of <= 0 else snap_17n5(of)
                    except (TypeError, ValueError):
                        pass
            segments.append(seg)

        # ── the chain's stop point ('mute', see _mute_stop) ──
        # Realized HERE, by shortening the chain: everything downstream (the
        # cumulative frame counts, the auto-crop / bgm / fun-control windows,
        # the per-segment material canvas records, the resume point) is derived
        # from this list, so a closed tail cannot leak into a slice or a canvas
        # and the run agrees with the editor's "running N" line by construction.
        stop = _mute_stop(raw, len(segments))
        if stop == 0:
            raise ValueError(
                "every segment is closed (mute): segment 0 is the chain's stop "
                "point, so this run has nothing to sample. Unmute a segment in "
                "the Tile Editor.")
        if stop > 0:
            segments = segments[:stop]
            print(f"[MMH3-TemporalTileEditor] mute: the chain stops before "
                  f"segment {stop} - segment(s) {stop}..{len(raw_segs) - 1} are "
                  "closed and are not sampled in this run.")
        # a stop point at or before the resume point leaves nothing to sample:
        # the Extend node then passes the stored merged latent through (its
        # resume == len(segs) branch), which is what the editor shows
        resume = min(resume, len(segments))

        # ── connected reference sockets -> MMH3_REF_IMAGES payload ──
        # (one record per socket, untouched resolution/aspect/batch: the
        # Extend node slices the first image per socket and resizes it with
        # the segment's ref_image_size, exactly like a picked file)
        ref_slots = _collect_socket_slots(ref_images, sock_names)

        config = {
            "session_name": session_name,
            "storage": storage,
            "resume_from_segment": resume,
            "mute_from_segment": stop,
            "fps": FPS,
            "base_prompt": base_prompt,
            "base_negative": base_negative,
            "extend_params": extend_params,
            "segments": segments,
        }
        # ── direct reference in frame (per-segment records) ──
        # Every segment carries its own record and the Extend node reads them
        # from there; 'material' below is the DERIVED chain view (the inventory
        # of sides plus the first enabling segment's, kept for observability) -
        # it is never read back as authoritative. Each segment splices its own
        # canvas and has it cut back off after sampling, so nothing has to agree
        # between segments. The asset itself is the node's existing
        # 'ref_video_input' socket, which is only valid because the strip and an
        # 'auto crop input ref' reference video cannot both claim that socket -
        # the Extend node rejects the combination rather than silently picking
        # one.
        config["material"] = _material_chain(segments)
        config["material_segments"] = list(config["material"]["segments"])
        # 'bgm' segments cut their soundtrack out of this audio. It is NOT
        # encoded here (this node owns no audio VAE) - the {waveform,
        # sample_rate} dict travels in the config and the Extend node encodes
        # it once, caches it and slices each segment's span out of the latent.
        if audio_bgm is not None:
            config["bgm_audio"] = audio_bgm
        # ref video input for 'auto crop input ref' segments. It is NOT sliced
        # or encoded here (this node owns no VAE): the raw tensor travels in
        # the config and the Extend node slices each auto_crop segment's span
        # out of it (continuous concatenation on the chain's timeline) and
        # encodes that slice as that segment's reference video on demand.
        if ref_video_input is not None:
            config["ref_video_input"] = ref_video_input
        # ── fun control assets -> MMH3_CONTROL_SLOTS payload ──
        # Like the reference sockets, these travel over their own typed link
        # instead of inside the config dict: the Extend node needs a WIRED
        # socket to tell "the user wants control but forgot to wire it" apart
        # from "nothing to control", and a link keeps the (large) video
        # objects out of the per-segment JSON snapshot the editor persists.
        # Nothing is decoded here - the VIDEO objects are forwarded verbatim.
        control_bundle = {
            "version": CONTROL_BUNDLE_VERSION,
            "kind": CONTROL_BUNDLE_KIND,
            "video": fun_control_video,
            "mask": fun_control_mask,
            "source_video": fun_control_source_video,
            "connected": (fun_control_video is not None
                          or fun_control_mask is not None
                          or fun_control_source_video is not None),
        }
        segments_info = {
            "session_name": session_name,
            "storage": storage,
            "resume_from_segment": resume,
            "mute_from_segment": stop,
            "segment_count": len(segments),
            "closed_segment_count": max(0, len(raw_segs) - len(segments)),
            "bgm_connected": audio_bgm is not None,
            "bgm_segments": [s["index"] for s in segments
                             if s.get("ref_audio_mode") == "bgm"],
            "ref_video_connected": ref_video_input is not None,
            "auto_crop_video_segments": [s["index"] for s in segments
                                         if s.get("ref_video_mode") == "auto_crop"],
            # DERIVED from the per-segment records (see _material_chain): the
            # segments themselves are the source of truth
            "material": dict(config["material"]),
            "control_video_connected": fun_control_video is not None,
            "control_mask_connected": fun_control_mask is not None,
            "control_source_video_connected": fun_control_source_video is not None,
            "control_segments": [s["index"] for s in segments
                                 if s.get("control_mode", "off") != "off"],
            # wired sockets (name + batch size) - informational: the images
            # themselves travel through the MMH3_REF_IMAGES link
            "ref_socket_slots": [{"name": s["name"], "count": s["count"]}
                                 for s in ref_slots["slots"]],
            "segments": [{
                "index": s["index"], "first": s["first"], "mode": s["mode"],
                "ref_source": s["ref_source"], "new_frames": s["new_frames"],
                "new_seconds": s["new_seconds"], "seed": s["seed"],
                "ref_socket_off": s.get("ref_socket_off", []),
                "control_mode": s.get("control_mode", "off"),
                "control_strength": s.get("control_strength"),
                "control_start": s.get("control_start"),
                "control_end": s.get("control_end"),
                **({"width": s["width"], "height": s["height"]} if s["first"] else {}),
            } for s in segments],
        }
        return io.NodeOutput(config, segments_info, ref_slots, control_bundle)
