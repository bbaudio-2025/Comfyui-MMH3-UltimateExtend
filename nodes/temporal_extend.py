"""MMH3 Temporal Extend Video (experimental) node.

Takes a fully-sampled MiniMax H3 AV latent, splits off its TAIL and uses it as
the HEAD of a fresh, longer sampling latent: the carried-over tail anchors the
model (frozen zone + fade zone via the latent noise_mask, plus an optional
frame-0 keyframe pin) while the appended tail of the new latent is sampled
freely - the temporal counterpart of the spatial Extend Video node.

Frame/token mapping (mirrors comfy.ldm.minimax.model):
  * video latent token k covers FRAME_PER_TOKEN[k % 5] = (1, 4, 4, 4, 4) pixel
    frames (periodic grid, 17 frames per 5 tokens)
  * audio latent frames run at FRAME_RESCALE = 5/3 per pixel frame (40 vs 24 Hz)

Every split boundary is snapped to a video-token boundary on the keyframe grid
(token index % 5 == 0), so the carried-over head starts on the same phase the
H3 model expects for a standalone video. When the source latent has the model's
standard length (17n + 5 frames), the realized tail is also 17m + 5 frames and
head + extend lands back on the 17n + 5 grid automatically.

Temporal split/stitch/anchor logic mirrors the Comfyui-MiniMax-H3-LatentSplit
plugin (compute_segments / trim_keyframe / reanchor / crossfade append) and the
temporal chunking of MMH3 Ultimate Upscale (anchor_conditioning /
temporal_append); the frozen+fade zone philosophy follows the spatial Extend
Video node's noise-mask design (per-token LABEL-INPUT CONSISTENCY: the mask
value becomes each 2x2-pooled token's timestep label AND mixes that token's
input, so a per-token constant band reads as "partially preserved content").
"""

import math
import os

import torch

import comfy.nested_tensor

from comfy_api.latest import io

from .helpers import blend_weights, is_h3_av_latent, sample_piece
from .temporal_tile_editor import REF_SOCKET_NAMES, snap_17, snap_17n5

# typed link between the MMH3 Sample Params sub-node and the Temporal Extend node
SAMPLE_PARAMS = io.Custom("MMH3_SAMPLE_PARAMS")
OVERLAP_PARAMS = io.Custom("MMH3_OVERLAP_PARAMS")

# Typed link carrying the Tile Editor's wired reference sockets: ONE RECORD PER
# SOCKET (name, slot, batch size, image tensor) at native resolution. NOT a
# plain IMAGE batch - flattening the sockets would stretch mismatched aspect
# ratios and erase the socket boundaries (prompt <Picture i> numbering and the
# per-segment ref_socket_off).
REF_IMAGES = io.Custom("MMH3_REF_IMAGES")

# The two Tile Editor sockets that carry a double role: slot 0 is also
# segment 0's FL2VA first frame, slot 1 its last frame. Imported from the Tile
# Editor module so the names exist in exactly one place (they must not have a
# prefix relationship, hence TemplateNames over there).
SOCK_SLOT_FIRST = REF_SOCKET_NAMES[0]
SOCK_SLOT_LAST = REF_SOCKET_NAMES[1]

try:
    from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE
except Exception:
    FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
    FRAME_RESCALE = 5.0 / 3.0

try:
    from comfy_extras.nodes_custom_sampler import Noise_RandomNoise
except Exception:  # pragma: no cover - mirrors comfy_extras/nodes_custom_sampler.py
    class Noise_RandomNoise:
        def __init__(self, seed):
            self.seed = seed

        def generate_noise(self, input_latent):
            import comfy.sample
            latent_image = input_latent["samples"]
            batch_inds = input_latent["batch_index"] if "batch_index" in input_latent else None
            return comfy.sample.prepare_noise(latent_image, self.seed, batch_inds)


# ---------------------------------------------------------------------------
# frame <-> token helpers (same mapping as Comfyui-MiniMax-H3-LatentSplit)
# ---------------------------------------------------------------------------

def frames_for_tokens(n):
    """Pixel frames covered by the first `n` video latent tokens."""
    return sum(FRAME_PER_TOKEN[i % 5] for i in range(n))


def tokens_for_frames(f):
    """Smallest token count whose cumulative frames reach at least `f`."""
    n, acc = 0, 0
    while acc < f:
        acc += FRAME_PER_TOKEN[n % 5]
        n += 1
    return n


def audio_range(f0, f1):
    """Audio latent token range [a0, a1) for the pixel-frame span [f0, f1)."""
    return round(f0 * FRAME_RESCALE), round(f1 * FRAME_RESCALE)


def _broadcast_segment_ready(seq, seg_index, files, total):
    """Push a 'segment N done' event to the frontend via the ComfyUI WebSocket.

    Lets the Tile Editor refresh its per-segment preview as soon as a segment
    finishes sampling, instead of waiting for the whole multi-segment chain
    (one synchronous onExec run) to finish. Best-effort: if the server
    handle/instance isn't reachable we just skip, the file writes above are
    the source of truth."""
    try:
        from server import PromptServer
        prompt_server = getattr(PromptServer, "instance", None)
        if prompt_server is None or not getattr(prompt_server, "send_sync", None):
            return
        prompt_server.send_sync("mmh3te:segment_ready", {
            "seq": int(seq),
            "seg_index": int(seg_index),
            "total": int(total),
            "files": files,
        })
    except Exception as exc:
        print(f"[MMH3-TemporalExtend] broadcast segment-ready failed: {exc}")


# ── background music ('bgm' audio reference mode) ──
# The music is encoded ONE SEGMENT AT A TIME: only the span a 'bgm' segment
# needs is ever handed to the audio VAE, so the peak allocation tracks ONE
# segment's length instead of the whole track (the VAE's encoder cost is linear
# in its input: ~1 GB per 20 s of 32 kHz stereo, i.e. gigabytes for a song).
# Each slice is cut with this much context on both sides and the context frames
# are dropped after the encode. The encoder is convolution + CAUSAL attention,
# so with the receptive field (~0.6 s) covered the slice's latents are the ones
# a full-track encode would have produced - what the truncated history leaves
# out is only the attention term's slow-moving contribution.
# A BGM shorter than the chain is NOT rejected: the slice is silently
# truncated to whatever the music still has, and the audio noise-mask freezes
# only the BGM-covered frames (mask = 0 there, i.e. pinned to the music) and
# leaves the rest at 1 so the model continues on its own - by that point the
# segment's conditioning carries the music it heard up to its last frame, so
# the free tail sounds like a natural continuation rather than a hard cut.
BGM_SLICE_CTX_FRAMES = 40          # 1 s at 40 audio latent frames per second


def _bgm_vae_waveform(audio_vae, audio):
    """The BGM waveform as [1, 2, L] on the CPU at the audio VAE's own rate.

    Resampled ONCE per run (a full-track resample is a cheap linear pass, ~50 MB
    for a song), so every per-segment slice can be cut at exact latent-frame
    boundaries and only the slice ever reaches the VAE."""
    wf = audio["waveform"]
    if wf.dim() == 2:
        wf = wf.unsqueeze(0)
    wf = wf[:1].detach().to("cpu").float()
    sr = int(audio.get("sample_rate") or 0)
    vae_sr = int(getattr(audio_vae, "audio_sample_rate", 32000) or 32000)
    if sr and sr != vae_sr:
        import torchaudio
        wf = torchaudio.functional.resample(wf, sr, vae_sr)
    return wf


def _bgm_total_frames(audio_vae, wave):
    """Audio latent frame count of the whole BGM (what one big encode of the
    track would have returned): the encoder zero-pads to a whole hop."""
    hop = int(getattr(audio_vae, "downscale_ratio", 800) or 800)
    return -(-int(wave.shape[-1]) // hop)


def _encode_bgm_slice(audio_vae, wave, a0, n):
    """Encode the BGM's audio latent span [a0, a0 + n) out of `wave` (already at
    the VAE's rate - see _bgm_vae_waveform). Returns [1, 32, 2, m] with m <= n;
    m < n only when the music ends inside the span. Returns None when the
    slice would have zero BGM frames (a0 past the music's end, or n == 0). The
    caller is responsible for turning the truncation into a free tail via the
    noise-mask; a short BGM is NOT a hard error."""
    hop = int(getattr(audio_vae, "downscale_ratio", 800) or 800)
    a0 = max(0, int(a0))
    n = min(max(0, int(n)), _bgm_total_frames(audio_vae, wave) - a0)
    if n == 0:
        return None
    ctx = BGM_SLICE_CTX_FRAMES * hop
    s0 = max(0, a0 * hop - ctx)
    s1 = min(int(wave.shape[-1]), (a0 + n) * hop + ctx)
    z = audio_vae.encode(wave[..., s0:s1].movedim(1, -1))   # [1, 32, 2, T]
    k = (a0 * hop - s0) // hop          # context frames to drop
    return z[..., k:k + n].contiguous()


def snap_split_frame(frame_count, tail_frames):
    """Keyframe-grid split point nearest to `frame_count - tail_frames`.

    Only boundaries at token indices that are a multiple of 5 (one keyframe
    grid step = 17 frames) are considered, and never at/after the latent's end,
    so the realized tail is always at least the final keyframe token. Returns
    (video_token_index, exact_pixel_frames)."""
    target = frame_count - max(5, int(tail_frames))
    best_k, best_f, best_d = 0, 0, abs(0 - target)
    k = 5
    while True:
        f = frames_for_tokens(k)
        if f >= frame_count:
            break
        d = abs(f - target)
        if d < best_d:
            best_k, best_f, best_d = k, f, d
        k += 5
    return best_k, best_f


# ---------------------------------------------------------------------------
# conditioning re-anchoring (mirrors Comfyui-MiniMax-H3-LatentSplit)
# ---------------------------------------------------------------------------

def trim_keyframe(kf, f0, f1):
    """Copy a keyframe cut to the portion fully inside pixel frames [f0, f1).

    The video latent is trimmed to whole tokens strictly inside the segment and
    resolved_frame_index is re-anchored to the segment origin. Returns None
    when nothing of the keyframe survives."""
    idx = kf["resolved_frame_index"]
    latent = kf.get("latent")
    audio_latent = kf.get("audio_latent")
    has_v = latent is not None
    has_a = audio_latent is not None

    if not has_v and not has_a:
        if idx < f0 or idx >= f1:
            return None
        return {"resolved_frame_index": idx - f0}

    out = {}
    if has_v:
        t_start = t_end = None
        pos = idx
        for k in range(latent.shape[2]):
            span = FRAME_PER_TOKEN[k % 5]
            if f0 <= pos and pos + span <= f1:
                if t_start is None:
                    t_start = k
                t_end = k + 1
            pos += span
        if t_start is None:
            return None
        out["latent"] = latent[:, :, t_start:t_end].contiguous()
        out["resolved_frame_index"] = idx + frames_for_tokens(t_start) - f0
    if has_a:
        rt = audio_latent.shape[-1]
        a_start = max(0, math.ceil((f0 - idx) * FRAME_RESCALE))
        a_end = min(rt, math.floor((f1 - idx) / FRAME_RESCALE))
        if a_end > a_start:
            out["audio_latent"] = audio_latent[..., a_start:a_end].contiguous()
            if "resolved_frame_index" not in out:
                out["resolved_frame_index"] = max(0, idx - f0)
    if "latent" not in out and "audio_latent" not in out:
        return None
    return out


def reanchor_conditioning(cond, f0, f1):
    """Return a conditioning list whose minimax_keyframes are cut/re-anchored to
    the pixel-frame segment [f0, f1). minimax_refs are left untouched (they are
    positioned before the target timeline and need no re-anchoring)."""
    out = []
    for tensor, d in cond:
        nd = dict(d)
        kfs = nd.get("minimax_keyframes")
        if kfs:
            trimmed = [trim_keyframe(kf, f0, f1) for kf in kfs]
            trimmed = [kf for kf in trimmed if kf is not None]
            if trimmed:
                nd["minimax_keyframes"] = trimmed
            else:
                nd.pop("minimax_keyframes", None)
        out.append([tensor, nd])
    return out


def anchor_conditioning(cond, prev_video, token_index, strength):
    """Replace the frame-0 keyframe with the carried-over tail's first frame.

    Keyframes are frozen rows in the H3 packed sequence, so pinning frame 0 to
    the actual boundary content removes the detail mismatch at the seam.
    `strength` becomes minimax_visual_cond_noise_aug (0.999 = model default)."""
    anchor_kf = {"resolved_frame_index": 0,
                 "latent": prev_video[:, :, token_index:token_index + 1].contiguous()}
    aug = max(0.0, min(1.0, float(strength)))
    out = []
    for tensor, d in cond:
        nd = dict(d)
        kfs = nd.get("minimax_keyframes")
        if kfs:
            kept = [kf for kf in kfs if kf.get("resolved_frame_index") != 0 or "latent" not in kf]
            nd["minimax_keyframes"] = [anchor_kf] + kept
        else:
            nd["minimax_keyframes"] = [anchor_kf]
        nd["minimax_visual_cond_noise_aug"] = aug
        out.append([tensor, nd])
    return out


def strip_keyframes(cond):
    """Remove every minimax_keyframes entry (fresh continuation guided only by
    the frozen latent head and the prompt)."""
    out = []
    for tensor, d in cond:
        nd = dict(d)
        nd.pop("minimax_keyframes", None)
        out.append([tensor, nd])
    return out


# ---------------------------------------------------------------------------
# stitching (mirrors Ultimate Upscale's temporal_append)
# ---------------------------------------------------------------------------

def _crossfade(a, b, dim, weights):
    n = a.shape[dim]
    shape = [1] * a.ndim
    shape[dim] = n
    w = weights.view(shape).to(a.dtype)
    return a + (b - a) * w


def temporal_stitch(src_v, src_a, seg_v, seg_a, k_split, a_split,
                    overlap_mode, overlap_blend, blend_len_v=None,
                    blend_len_a=None):
    """Stitch the re-sampled segment onto the source timeline (cross-fade over
    the overlap). `overlap_mode` selects who wins the band ('earlier' = source,
    'later' = new segment); `overlap_blend` shapes the transition. Returns
    (result_v, result_a).

    `blend_len_v` / `blend_len_a` (token counts) confine the cross-fade RAMP
    to the frozen zone - the part of the overlap where both sides carry the
    SAME pristine content, so any blend shape is invisible there and only
    smooths micro-differences (e.g. from a refinement pass). The remainder of
    the overlap - the re-developed fade band - is assigned WHOLLY to the new
    segment: the new content was sampled to follow the re-developed band, so
    blending it against the previous rendition would only dissolve two
    genuinely different takes (ghosting)."""
    total_v = max(src_v.shape[2], k_split + seg_v.shape[2])
    total_a = max(src_a.shape[-1], a_split + seg_a.shape[-1])
    result_v = torch.zeros((1, src_v.shape[1], total_v, src_v.shape[3], src_v.shape[4]),
                           device=src_v.device, dtype=src_v.dtype)
    result_a = torch.zeros((1, 32, 2, total_a), device=src_a.device, dtype=src_a.dtype)
    result_v[:, :, :src_v.shape[2]] = src_v
    result_a[:, :, :, :src_a.shape[-1]] = src_a

    ov = src_v.shape[2] - k_split
    if ov > 0:
        ov = min(ov, seg_v.shape[2])
        bl = ov if blend_len_v is None else max(0, min(int(blend_len_v), ov))
        t = torch.linspace(0.0, 1.0, max(bl, 1),
                           device=result_v.device,
                           dtype=torch.float32)[:bl]
        w = blend_weights(t, overlap_blend, overlap_mode)
        if bl < ov:  # fade band: wholly to the new segment
            w = torch.cat([w, torch.ones(ov - bl, device=w.device,
                                         dtype=w.dtype)])
        tail = result_v[:, :, k_split:k_split + ov].clone()
        result_v[:, :, k_split:k_split + ov] = _crossfade(tail, seg_v[:, :, :ov], dim=2, weights=w)
    write_v = k_split + max(ov, 0)
    if seg_v.shape[2] - max(ov, 0) > 0:
        result_v[:, :, write_v:write_v + seg_v.shape[2] - max(ov, 0)] = seg_v[:, :, max(ov, 0):]

    ova = src_a.shape[-1] - a_split
    if ova > 0:
        ova = min(ova, seg_a.shape[-1])
        bla = ova if blend_len_a is None else max(0, min(int(blend_len_a), ova))
        t = torch.linspace(0.0, 1.0, max(bla, 1),
                           device=result_a.device,
                           dtype=torch.float32)[:bla]
        w = blend_weights(t, overlap_blend, overlap_mode)
        if bla < ova:  # fade band: wholly to the new segment
            w = torch.cat([w, torch.ones(ova - bla, device=w.device,
                                         dtype=w.dtype)])
        tail = result_a[:, :, :, a_split:a_split + ova].clone()
        result_a[:, :, :, a_split:a_split + ova] = _crossfade(tail, seg_a[:, :, :, :ova], dim=3, weights=w)
    write_a = a_split + max(ova, 0)
    if seg_a.shape[-1] - max(ova, 0) > 0:
        result_a[:, :, :, write_a:write_a + seg_a.shape[-1] - max(ova, 0)] = seg_a[:, :, :, max(ova, 0):]

    return result_v, result_a


# ---------------------------------------------------------------------------
# per-segment conditioning / storage / preview infrastructure
# ---------------------------------------------------------------------------

def _frame_to_token(frame_idx, total_tokens):
    """Video latent token that covers pixel frame `frame_idx`."""
    f = 0
    for k in range(total_tokens):
        span = FRAME_PER_TOKEN[k % 5]
        if f <= frame_idx < f + span:
            return k
        f += span
    return max(total_tokens - 1, 0)


def _load_input_image(filename):
    """Load an image from the ComfyUI input folder into a [1, H, W, 3] tensor."""
    from PIL import Image as PILImage
    from PIL import ImageOps as PILImageOps
    import numpy as np
    import folder_paths
    path = folder_paths.get_annotated_filepath(filename)
    img = PILImage.open(path).convert("RGB")
    img = PILImageOps.exif_transpose(img)
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)  # [1, H, W, 3]


def _load_input_audio(filename):
    """Load an audio file from the input folder into the native Audio dict
    ({"waveform": [B, C, L], "sample_rate": sr}) via PyAV (same backend as
    comfy_extras/nodes_audio.py)."""
    from comfy_extras.nodes_audio import load as _av_audio_load
    import folder_paths
    wav, sr = _av_audio_load(folder_paths.get_annotated_filepath(filename))
    return {"waveform": wav.unsqueeze(0), "sample_rate": int(sr)}


def _load_input_video_frames(filename, max_seconds=15.0):
    """Load a reference video's frames at 24 fps into a [N, H, W, 3] tensor
    (0..1). Mirrors the native MiniMaxH3ReferenceToVideo expectations: 2-15 s
    clips; longer inputs are truncated, other frame rates are resampled to
    24 fps by nearest index."""
    import av
    import numpy as np
    import folder_paths
    path = folder_paths.get_annotated_filepath(filename)
    with av.open(path) as container:
        vstream = container.streams.video[0]
        src_fps = float(vstream.average_rate or 24.0)
        if src_fps <= 0 or math.isnan(src_fps):
            src_fps = 24.0
        tb = vstream.time_base
        picked = []
        for frame in container.decode(video=0):
            if frame.pts is not None:
                t = float(frame.pts * tb)
            else:
                t = len(picked) / src_fps
            if t > max_seconds:
                break
            picked.append(frame)
    if not picked:
        raise ValueError(f"ref video '{filename}' has no decodable video frames")
    arrs = []
    for f in picked:
        a = f.to_ndarray(format="rgb24").astype(np.float32) / 255.0
        arrs.append(a)
    img = torch.from_numpy(np.stack(arrs))  # [N, H, W, 3]
    if abs(src_fps - 24.0) > 0.5:
        idx = torch.round(torch.arange(0, img.shape[0], src_fps / 24.0)).long()
        idx = idx[idx < img.shape[0]]
        img = img[idx]
    return img


def _save_h3latent(path, samples):
    """Save an H3 AV latent (NestedTensor video+audio) in the
    comfyui-minimax-h3-latent '.h3latent' safetensors format."""
    import os
    import safetensors.torch
    members = list(samples.tensors) if getattr(samples, "is_nested", False) \
        else [samples]
    out = {
        "format_version": torch.tensor([1], dtype=torch.int32),
        "tensor_count": torch.tensor([len(members)], dtype=torch.int32),
    }
    for i, t in enumerate(members):
        out[f"latent_{i}"] = t.detach().to("cpu").contiguous()
    tmp = f"{path}.tmp"
    safetensors.torch.save_file(out, tmp)
    os.replace(tmp, path)


def _load_h3latent(path):
    """Load a '.h3latent' file (or any saved latent) into a LATENT dict whose
    'samples' is a NestedTensor (video+audio) when the file has 2 tensors."""
    import safetensors.torch
    data = safetensors.torch.load_file(path, device="cpu")
    count = int(data["tensor_count"].item())
    tensors = [data[f"latent_{i}"].float() for i in range(count)]
    samples = tensors[0] if count == 1 else comfy.nested_tensor.NestedTensor(tensors)
    return {"samples": samples}


# ---------------------------------------------------------------------------
# storage layout
#
#   <base>/mmh3_temporal/
#       <session>/                    one directory per Tile Editor session
#           session.json              attempt ledger (authoritative mapping)
#           latents/                  segment_<seq>.h3latent, merged_<seq>.h3latent
#           previews/                 <kind>_preview_<seq>.webp, merged_lastframe_<seq>.png
#       cache/                        SHARED by every session of this storage root
#           refs/<kind>_<key>.safetensors   VAE latents of the reference assets
#           cond/<key>.safetensors          encoded conditioning (prompt + refs)
#
# The session directory is the GENERATED result (its own latents + previews);
# cache/ holds what is expensive to RE-derive and identical across sessions:
# a reference image's VAE latent and the conditioning that a prompt + a set of
# references encodes to. Splitting them keeps a session folder self-contained
# (copy it alone and the chain still resumes) while the cache survives every
# session that reuses the same assets.
#
# Before 2026-09-11 everything lived flat in the session directory; the
# resolver below still finds those files, so old sessions keep resuming.
# ---------------------------------------------------------------------------

LATENTS_SUBDIR = "latents"
PREVIEWS_SUBDIR = "previews"
CACHE_DIRNAME = "cache"

# a session literally named 'cache' would collide with the shared cache
_RESERVED_SESSION_NAMES = frozenset((CACHE_DIRNAME,))


def _temporal_root(storage):
    """The mmh3_temporal root of a storage location ('temp' |
    'output/latents')."""
    import os
    import folder_paths
    base = (folder_paths.get_temp_directory() if storage == "temp"
            else os.path.join(folder_paths.get_output_directory(), "latents"))
    return os.path.join(base, "mmh3_temporal")


def _session_dir(storage, session_name, create=True):
    """Resolve a session's directory. Its subdirectories are created lazily by
    their writers, so a session that never stored a preview does not grow an
    empty 'previews' folder.

    `create=False` resolves the SAME path without touching the disk - the read
    paths (the HTTP route that lists a session) need the name -> path mapping
    without making a session look like it exists."""
    import os
    safe = "".join(c if c.isalnum() or c in "-_" else "_"
                   for c in (session_name or "")) or "session1"
    if safe in _RESERVED_SESSION_NAMES:
        safe = f"{safe}_session"
    path = os.path.join(_temporal_root(storage), safe)
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def _session_subdir(sdir, sub, create=True):
    """A session subdirectory ('latents' / 'previews'). Writes go through
    this; the ledger only ever stores the relative name
    ('previews/merged_preview_3.webp'), never an absolute path."""
    import os
    path = os.path.join(sdir, sub)
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def _find_session_file(sdir, name, fallback_sub=None):
    """Resolve a ledger file name to a path.

    New ledgers store a subdirectory-qualified name
    ('latents/merged_3.h3latent'); ledgers written before 2026-09-11 store a
    bare name with the file flat in the session directory. A bare name is
    looked up in the session directory, then in every known subdirectory, so
    both layouts - and a half-migrated one - resolve. Returns None when
    nothing is on disk; `fallback_sub` asks for the path a missing file WOULD
    be written to, for error messages."""
    import os
    if not name:
        return None
    rel = name.replace("\\", "/")
    direct = os.path.join(sdir, rel)
    if os.path.isfile(direct):
        return direct
    base = os.path.basename(rel)
    for sub in (LATENTS_SUBDIR, PREVIEWS_SUBDIR, ""):
        cand = os.path.join(sdir, sub, base) if sub else os.path.join(sdir, base)
        if os.path.isfile(cand):
            return cand
    if fallback_sub:
        return os.path.join(sdir, fallback_sub, base)
    return None


# ---------------------------------------------------------------------------
# reference / conditioning cache
#
# Both caches are CONTENT addressed: the key contains the source identity
# (path+mtime+size for a file in the input folder, a blake2b of the pixels for
# a wired socket image) plus every parameter that changes what gets encoded.
# Nothing is ever invalidated by hand - a changed input simply hashes to a new
# entry. CACHE_FORMAT is the escape hatch for changes in the PLUGIN's own
# behaviour (how a reference is resized, how the presentation is assembled,
# what a block carries): bump it and every earlier entry stops matching.
#
# 'refs'  - a reference asset -> its VAE latent. Pays off when the prompt
#           changed but the assets did not (the conditioning must be
#           re-encoded, the VAE passes need not run again).
# 'cond'  - the full encoded conditioning (prompt embedding, token tags,
#           keyframes, reference blocks). Paying it off means the tokenizer
#           and the text encoder are skipped ENTIRELY - the single most
#           expensive step of a reference-conditioned segment.
# ---------------------------------------------------------------------------

# v2: the FL2VA first frame is centre cover-cropped instead of stretched, so
# the same source image now produces a different latent.
CACHE_FORMAT = 2

_COND_META_KEY = "mmh3_cond"
_TENSOR_TAG = "__tensor__"


def _file_source_id(name):
    """Load-free identity of an input-folder asset: absolute path + mtime +
    size. Replacing a file (or editing it) yields a new id, so a cached latent
    can never be replayed onto different pixels."""
    import os
    import folder_paths
    try:
        path = folder_paths.get_annotated_filepath(name)
    except Exception:
        return f"missing:{name}"
    try:
        st = os.stat(path)
        return "f|%s|%d|%d" % (os.path.abspath(path), int(st.st_mtime_ns),
                               int(st.st_size))
    except OSError:
        return f"missing:{name}"


def _tensor_source_id(t):
    """Content identity of an in-memory tensor - a wired socket image has no
    path, so the pixels themselves are the identity. A 1024x1024 RGB frame
    hashes in single-digit milliseconds, against seconds for the encode it
    guards."""
    import hashlib
    tt = t.detach()
    if tt.device.type != "cpu":
        tt = tt.cpu()
    h = hashlib.blake2b(digest_size=16)
    h.update(repr((tuple(tt.shape), str(tt.dtype))).encode())
    data = tt.contiguous()
    try:
        h.update(memoryview(data.numpy().data))
    except Exception:
        # bfloat16 and friends have no numpy dtype
        h.update(data.to(torch.float32).contiguous().numpy().tobytes())
    return "t|" + h.hexdigest()


def _cache_key(*parts):
    import hashlib
    h = hashlib.blake2b(digest_size=20)
    for p in parts:
        h.update(("%s" % (p,)).encode("utf-8"))
        h.update(b"\x1e")
    return h.hexdigest()


def _clip_fingerprint(clip):
    """Best-effort identity of the text encoder behind `clip`, so a cached
    conditioning is never replayed onto a different encoder. Built from stable
    strings only (class names, tokenizer paths) - never object ids, which
    would change on every restart."""
    bits = [type(clip).__name__, str(getattr(clip, "layer_idx", None))]
    for attr in ("cond_stage_model", "tokenizer"):
        obj = getattr(clip, attr, None)
        if obj is None:
            continue
        bits.append(type(obj).__name__)
        inner = getattr(obj, "tokenizer", None)
        for probe in ("name_or_path",):
            v = getattr(inner, probe, None) or getattr(obj, probe, None)
            if v:
                bits.append(str(v))
    return _cache_key("clip", *bits)[:16]


def _vae_fingerprint(vae):
    """Structural identity of a VAE: class names plus its parameter count.
    Reads shapes only, so it costs nothing measurable, and it is stable across
    restarts. Enough to keep a cached latent from being replayed onto a
    different VAE."""
    if vae is None:
        return "none"
    bits = [type(vae).__name__]
    inner = getattr(vae, "first_stage_model", None)
    if inner is not None:
        bits.append(type(inner).__name__)
        try:
            bits.append(str(sum(int(p.numel()) for p in inner.parameters())))
        except Exception:
            pass
    return _cache_key("vae", *bits)[:12]


def _cached_ref_latent(cache, kind, source_id, params, build):
    """The cached VAE latent of one reference asset: `build()` unless the
    cache already holds it. Without a cache (or without a content identity)
    this is a plain `build()`."""
    if cache is None:
        return build()
    return cache.ref(kind, source_id, params, build)


def _spill(obj, flat):
    """Replace every tensor of a conditioning structure with a name
    placeholder. Raises TypeError for a value that cannot round-trip, which
    the caller reads as 'this conditioning is not cacheable'."""
    if isinstance(obj, torch.Tensor):
        name = "t%d" % len(flat)
        flat[name] = obj.detach().to("cpu").contiguous()
        return {_TENSOR_TAG: name}
    if isinstance(obj, dict):
        return {str(k): _spill(v, flat) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_spill(v, flat) for v in obj]
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    raise TypeError(f"not cacheable: {type(obj).__name__}")


def _unspill(obj, tensors):
    if isinstance(obj, dict):
        if len(obj) == 1 and _TENSOR_TAG in obj:
            return tensors[obj[_TENSOR_TAG]]
        return {k: _unspill(v, tensors) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_unspill(v, tensors) for v in obj]
    return obj


class _ContentCache:
    """Content-addressed store for the two expensive halves of reference
    conditioning (see the block comment above).

    Lives at <mmh3_temporal>/cache, i.e. it is shared by every session of the
    storage root. Every method is failure-tolerant: a cache that cannot be
    read or written degrades to 'no cache' and the run continues."""

    def __init__(self, storage, clip=None, vae=None, audio_vae=None):
        self.storage = storage
        # Everything cached depends on the models that produced it. Fold their
        # fingerprints into EVERY key: a swapped text encoder or VAE simply
        # stops matching instead of silently replaying a stale conditioning.
        # (Deliberately coarse - over-invalidating costs one re-encode,
        # under-invalidating costs a wrong result.)
        self.env = _cache_key(_clip_fingerprint(clip) if clip is not None else "no-clip",
                              _vae_fingerprint(vae),
                              _vae_fingerprint(audio_vae))[:16]
        self._dirs = {}
        self.stats = {}
        self.errors = []

    # ── layout ──
    def dir(self, kind):
        import os
        path = self._dirs.get(kind)
        if path is None:
            path = os.path.join(_temporal_root(self.storage), CACHE_DIRNAME, kind)
            self._dirs[kind] = path
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as exc:
            self.errors.append(f"{kind}: {exc}")
            return None
        return path

    def _path(self, kind, name):
        d = self.dir(kind)
        return os.path.join(d, name) if d else None

    def _count(self, what):
        self.stats[what] = self.stats.get(what, 0) + 1

    def summary(self):
        s = self.stats
        return ("refs %d hit / %d stored, conditioning %d hit / %d stored"
                % (s.get("ref_hit", 0), s.get("ref_put", 0),
                   s.get("cond_hit", 0), s.get("cond_put", 0)))

    # ── reference latents ──
    def ref_key(self, kind, source_id, params):
        return _cache_key("ref", CACHE_FORMAT, self.env, kind, source_id, params)

    def ref(self, kind, source_id, params, build):
        """The latent for `source_id`: from the cache when it is there, else
        the result of `build()` (which is then stored). `source_id` of None
        means 'not content addressable' and always builds."""
        if source_id is None:
            return build()
        key = self.ref_key(kind, source_id, params)
        hit = self._load_ref(kind, key)
        if hit is not None:
            return hit
        val = build()
        self._save_ref(kind, key, val)
        return val

    def _load_ref(self, kind, key):
        import os
        import safetensors.torch
        p = self._path("refs", f"{kind}_{key}.safetensors")
        if not p or not os.path.isfile(p):
            return None
        try:
            t = safetensors.torch.load_file(p, device="cpu")["latent"]
        except Exception as exc:
            self.errors.append(f"refs read {kind}: {exc}")
            return None
        self._count("ref_hit")
        return t

    def _save_ref(self, kind, key, latent):
        import os
        import safetensors.torch
        p = self._path("refs", f"{kind}_{key}.safetensors")
        if not p:
            return
        try:
            tmp = p + ".tmp"
            safetensors.torch.save_file(
                {"latent": latent.detach().to("cpu").contiguous()}, tmp)
            os.replace(tmp, p)
            self._count("ref_put")
        except Exception as exc:
            self.errors.append(f"refs write {kind}: {exc}")

    # ── encoded conditioning ──
    def cond_key(self, *parts):
        return _cache_key("cond", CACHE_FORMAT, self.env, *parts)

    def load_cond(self, key):
        """The stored conditioning for `key`, or None. Tensors come back on
        the CPU; the model's own extra_conds moves them to the execution
        device, so a CPU conditioning is a valid (and VRAM-cheaper) input."""
        import json
        import os
        import safetensors
        p = self._path("cond", f"{key}.safetensors")
        if not p or not os.path.isfile(p):
            return None
        try:
            with safetensors.safe_open(p, framework="pt", device="cpu") as f:
                blob = (f.metadata() or {}).get(_COND_META_KEY)
                if not blob:
                    return None
                # get_tensor hands back a VIEW of the mapped file, which keeps
                # the file mapped - and on Windows an open mapping makes the
                # file undeletable. Copy so the mapping dies with this block,
                # and the entry stays reclaimable while the tensors are alive.
                tensors = {k: f.get_tensor(k).clone() for k in f.keys()}
            cond = _unspill(json.loads(blob), tensors)
        except Exception as exc:
            self.errors.append(f"cond read: {exc}")
            return None
        self._count("cond_hit")
        return cond

    def save_cond(self, key, cond):
        import json
        import os
        import safetensors.torch
        p = self._path("cond", f"{key}.safetensors")
        if not p:
            return
        flat = {}
        try:
            struct = _spill(cond, flat)
        except TypeError:
            return                      # not round-trippable -> not cached
        if not flat:
            return
        try:
            tmp = p + ".tmp"
            safetensors.torch.save_file(
                flat, tmp,
                metadata={_COND_META_KEY: json.dumps(struct), "mmh3": "cond"})
            os.replace(tmp, p)
            self._count("cond_put")
        except Exception as exc:
            self.errors.append(f"cond write: {exc}")


def cache_status(storage):
    """(entries, bytes) per cache kind - used by the cache API route."""
    import os
    out = {}
    root = os.path.join(_temporal_root(storage), CACHE_DIRNAME)
    for kind in ("refs", "cond"):
        n = total = 0
        d = os.path.join(root, kind)
        if os.path.isdir(d):
            for f in os.listdir(d):
                p = os.path.join(d, f)
                if f.endswith(".safetensors") and os.path.isfile(p):
                    n += 1
                    total += os.path.getsize(p)
        out[kind] = {"entries": n, "bytes": total}
    out["total_bytes"] = sum(v["bytes"] for v in out.values()
                             if isinstance(v, dict))
    return out


def cache_clear(storage, kind=None):
    """Delete cached entries ('refs' / 'cond' / None for both).

    Returns {'removed': n, 'failed': n}. A failure is reported rather than
    hidden: on Windows a file that is still mapped cannot be removed, so a
    conditioning that some other part of this process is still holding can
    refuse to go. Freeing it is then a matter of a later call, and the UI can
    say so instead of claiming the directory is empty."""
    import os
    removed = failed = 0
    root = os.path.join(_temporal_root(storage), CACHE_DIRNAME)
    for k in (("refs", "cond") if kind not in ("refs", "cond") else (kind,)):
        d = os.path.join(root, k)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if not f.endswith(".safetensors"):
                continue
            p = os.path.join(d, f)
            try:
                os.remove(p)
                removed += 1
            except OSError:
                failed += 1
    return {"removed": removed, "failed": failed}


def _save_preview(sdir, index, merged_v, seg_v, prev_frame_img=None,
                  max_frames=48, fps=12):
    """Save per-segment preview WebPs (merged + segment) and the merged tail
    frame PNG (used as the next segment's auto reference candidate) into the
    session's 'previews' subdirectory. Returns a dict of SESSION-RELATIVE
    names ('previews/merged_preview_3.webp'), the form the ledger stores and
    the HTTP route serves. Failures degrade to a skipped preview."""
    out = {}
    try:
        from .mmh3_preview import decode_h3_video_frames, frames_to_animated_webp
        pdir = _session_subdir(sdir, PREVIEWS_SUBDIR)
        for kind, video in (("merged", merged_v), ("segment", seg_v)):
            try:
                frames = decode_h3_video_frames(
                    video.unsqueeze(0) if video.ndim == 4 else video,
                    max_frames=max_frames)["frames"]
                if not frames:
                    continue
                webp = frames_to_animated_webp(frames, fps=fps)
                name = f"{kind}_preview_{index}.webp"
                with open(_pjoin(pdir, name), "wb") as fh:
                    fh.write(webp)
                out[f"{kind}_preview"] = f"{PREVIEWS_SUBDIR}/{name}"
                if kind == "merged" and frames:
                    pname = f"merged_lastframe_{index}.png"
                    frames[-1].save(_pjoin(pdir, pname))
                    out["merged_lastframe"] = f"{PREVIEWS_SUBDIR}/{pname}"
            except Exception as exc:
                print(f"[MMH3-TemporalExtend] preview ({kind} {index}) failed: {exc}")
    except Exception as exc:
        print(f"[MMH3-TemporalExtend] preview module error: {exc}")
    return out


def _pjoin(*a):
    import os
    return os.path.join(*a)


def _load_session_ledger(sdir):
    """Load 'session.json' - the AUTHORITATIVE per-segment attempt ledger.

    The ledger decouples segment INDICES from stored FILES: every execution
    of a segment allocates a fresh monotonically increasing 'seq' and writes
    new attempt files (segment_<seq>.h3latent, merged_<seq>.h3latent and the
    matching previews), so re-running a segment never overwrites a previous
    attempt. 'segments' maps each segment index to the currently chosen
    attempt (updated on every run), 'runs' keeps the full history.

    Legacy sessions (index-named files, no session.json) are synthesized
    into a v2 ledger on first load, so resuming an old session keeps
    working and the next run seamlessly switches to seq-named files."""
    import json
    import os
    import re
    path = _pjoin(sdir, "session.json")
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            if isinstance(d, dict) and int(d.get("version", 0)) == 2:
                return d
        except Exception as exc:
            print(f"[MMH3-TemporalExtend] WARNING: unreadable session.json "
                  f"({exc}) - starting a fresh ledger")
    # ── legacy synthesis: index-named files, flat or in subdirectories ──
    idxs = set()
    for sub in (LATENTS_SUBDIR, PREVIEWS_SUBDIR):
        d = _pjoin(sdir, sub)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            m = re.fullmatch(r"merged_(\d+)\.h3latent", f)
            if m:
                idxs.add(int(m.group(1)))
    if os.path.isdir(sdir):
        for f in os.listdir(sdir):
            m = re.fullmatch(r"merged_(\d+)\.h3latent", f)
            if m:
                idxs.add(int(m.group(1)))
    if not idxs:
        return {"version": 2, "next_seq": 0, "last_segment": -1,
                "segments": [], "runs": []}
    nxt = max(idxs) + 1
    segments = []
    for k in sorted(idxs):
        files = {}
        for kind, name in (
                ("merged_latent", f"merged_{k}.h3latent"),
                ("segment_latent", f"segment_{k}.h3latent"),
                ("segment_preview", f"segment_preview_{k}.webp"),
                ("merged_preview", f"merged_preview_{k}.webp"),
                ("merged_lastframe", f"merged_lastframe_{k}.png")):
            p = _find_session_file(sdir, name)
            if p:
                files[kind] = os.path.relpath(p, sdir).replace("\\", "/")
        segments.append({"index": k, "chosen_seq": k, "files": files})
    return {"version": 2, "next_seq": nxt, "last_segment": max(idxs),
            "segments": segments, "runs": [],
            "legacy_synthesized": True}


def _save_session_ledger(sdir, ledger):
    """Persist the session ledger. Failure is non-fatal: sampling results
    are already on disk, only the mapping would be stale."""
    import json
    try:
        with open(_pjoin(sdir, "session.json"), "w", encoding="utf-8") as fh:
            json.dump(ledger, fh, ensure_ascii=False, indent=1)
    except Exception as exc:
        print(f"[MMH3-TemporalExtend] WARNING: could not write session.json: {exc}")


def _encode_text(clip, prompt):
    tokens = clip.tokenize(prompt)
    return clip.encode_from_tokens_scheduled(tokens)


def _build_ref2va_full(clip, vae, audio_vae, prompt, w, h, frame_count,
                       ref_images, ref_image_size="match",
                       ref_video=None, ref_video_audio=None, ref_audio=None,
                       latent_refs=None, extra_video_refs=None,
                       latent_audio_refs=None, cache=None, src=None):
    """Full MiniMaxH3ReferenceToVideo-style reference conditioning.

    References enter the presentation in the native fixed order: images, then
    videos (each paired soundtrack's <Audio j> label right before its
    <Video k>), then standalone audio. latent_refs (bit-perfect latent blocks,
    e.g. the previous segment's frame) come first as <Picture> blocks, matching
    _create_ref2va_conditioning's ordering convention. Returns the positive
    conditioning, or None when no reference of any kind is present.

    `cache` / `src` serve the reference-latent cache. `src` carries the
    content identity of each asset ('images' aligned with ref_images, plus
    'video' / 'video_audio' / 'audio'), and every VAE pass below is routed
    through `cache.ref`, which hands back a stored latent instead of encoding
    again. The presentation - and therefore the tokenizer input - is always
    rebuilt; only the VAE work is skipped."""
    from comfy_extras.nodes_minimax_h3 import (
        _resize, CANVAS_MULTIPLE, adapt_canvas, _encode_ref_audio,
    )
    import node_helpers

    ref_items = []   # tokenizer presentation (<Picture i> / <Video k> / <Audio j>)
    ref_blocks = []  # DiT payload, same order
    src = src or {}

    def ref_latent(kind, source_id, params, build):
        return _cached_ref_latent(cache, kind, source_id, params, build)

    def enc_audio(audio, source_id, kind):
        """(latent, latent_frames). _encode_ref_audio's second return value is
        the latent's own last dimension, so caching the latent is enough."""
        if cache is None or source_id is None:
            return _encode_ref_audio(audio_vae, audio)
        z = cache.ref(kind, source_id, (),
                      lambda: _encode_ref_audio(audio_vae, audio)[0])
        return z, z.shape[-1]

    # 0) pre-built latent blocks (bit-perfect, no VAE round-trip)
    for blk in latent_refs or ():
        ref_items.append({"type": "image", "data": torch.full((1, 64, 64, 3), 0.5)})
        ref_blocks.append(dict(blk))

    # 1) reference images
    img_ids = src.get("images") or []
    for i, img in enumerate(ref_images or ()):
        ih, iw = img.shape[1], img.shape[2]
        if ref_image_size == "max":
            scale = min(1.0, 2048 / min(iw, ih))
        else:
            scale = min(1.0, math.sqrt((w * h) / (iw * ih)))
        tw = max(CANVAS_MULTIPLE, round(iw * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        th = max(CANVAS_MULTIPLE, round(ih * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        resized = _resize(img[:1], tw, th, "disabled")
        ref_items.append({"type": "image", "data": resized})
        if vae is not None:
            sid = img_ids[i] if i < len(img_ids) else None
            ref_blocks.append({"kind": "image", "latent_h": th // 16,
                               "latent_w": tw // 16,
                               "latent": ref_latent("img", sid,
                                                    (w, h, ref_image_size),
                                                    lambda: vae.encode(resized))})

    # 2) reference video (+ optional index-paired soundtrack)
    if ref_video:
        vid = _load_input_video_frames(ref_video)          # [N, H, W, 3] @24fps
        vh, vw = vid.shape[1], vid.shape[2]
        cw, ch = adapt_canvas(vw, vh)
        if vw * vh < cw * ch:
            cw = max(CANVAS_MULTIPLE, round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            ch = max(CANVAS_MULTIPLE, round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        frames = _resize(vid, cw, ch, "disabled")
        if frames.shape[0] > frame_count:
            frames = frames[:frame_count]
        n = frames.shape[0]
        if n < 5:
            raise ValueError(
                f"ref video '{ref_video}' needs at least 5 frames (~0.2s at 24 fps)")
        while n % 17 != 5:
            n -= 1
        frames = frames[:n]
        soundtrack = None
        if ref_video_audio:
            if audio_vae is None:
                print("[MMH3-TemporalExtend] WARNING: ref_video_audio given "
                      "but audio_vae is not connected - soundtrack ignored")
            else:
                soundtrack = _load_input_audio(ref_video_audio)
        if soundtrack is not None:
            ref_items.append({"type": "audio"})
        # Qwen sees the video at 2 fps with timestamps
        sample_idx = list(range(0, frames.shape[0], 12))   # FPS // 2 = 12
        ref_items.append({"type": "video", "data": frames[sample_idx],
                          "timestamps": [i / 2.0 for i in range(len(sample_idx))]})
        if vae is None:
            print("[MMH3-TemporalExtend] WARNING: ref_video given but vae is "
                  "not connected - video ref only conditions the text encoder")
        else:
            z = ref_latent("vid", src.get("video"), (w, h, frame_count),
                           lambda: vae.encode(frames))
            audio_latent, ref_audio_t = (None, 0)
            if soundtrack is not None:
                audio_latent, ref_audio_t = enc_audio(
                    soundtrack, src.get("video_audio"), "vaud")
            ref_blocks.append({"kind": "video_audio" if ref_audio_t else "video",
                               "latent_t": z.shape[2], "latent_h": ch // 16,
                               "latent_w": cw // 16, "ref_audio_t": ref_audio_t,
                               "latent": z, "audio_latent": audio_latent})

    # 2b) externally supplied video references (bit-perfect latent blocks,
    # e.g. the previous segment's tail for the 'prev_tail' seam reference)
    for ex_items, ex_blocks in extra_video_refs or ():
        ref_items.extend(ex_items)
        ref_blocks.extend(ex_blocks)

    # 3) standalone reference audio - first any bit-perfect LATENT audio
    # refs ('previous audio' / 'initial audio': the previous / first
    # segment's soundtrack sliced straight off the accumulated latent, no
    # VAE decode/encode round-trip), then file-loaded audio. Both render
    # as native <Audio j> items + 'audio' DiT blocks.
    for alat in latent_audio_refs or ():
        ref_items.append({"type": "audio"})
        ref_blocks.append({"kind": "audio", "ref_audio_t": alat.shape[-1],
                           "audio_latent": alat.contiguous()})
    if ref_audio:
        ref_items.append({"type": "audio"})
        if audio_vae is None:
            print("[MMH3-TemporalExtend] WARNING: ref_audio given but "
                  "audio_vae is not connected - audio ref only conditions "
                  "the text encoder")
        else:
            audio_latent, ref_audio_t = enc_audio(
                _load_input_audio(ref_audio), src.get("audio"), "aud")
            ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t,
                               "audio_latent": audio_latent})

    if not ref_blocks:
        if ref_items:
            print("[MMH3-TemporalExtend] WARNING: references had no vae/"
                  "audio_vae encode path - text-only conditioning")
            tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
            return clip.encode_from_tokens_scheduled(tokens)
        return None

    tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
    cond = clip.encode_from_tokens_scheduled(tokens)
    return node_helpers.conditioning_set_values(cond, {"minimax_refs": ref_blocks})


def _sock_use(seg, sock_pool):
    """[(slot_name, [1,H,W,C])] - the sockets this segment actually presents,
    in slot order, minus the ones it ruled out (ref_socket_off).

    ONE SOCKET IS ONE PICTURE, matching MiniMaxH3ReferenceToVideo: the first
    image of each socket, sliced at its native resolution. The single source
    of truth for `_sock_refs`, `_sock_slot` and the cache keys - so the
    presentation and the identity hashed for the cache can never drift
    apart."""
    if not sock_pool:
        return []
    off = set(seg.get("ref_socket_off") or [])
    return [(name, t[:1]) for name, t in sock_pool if name not in off]


def _sock_refs(seg, sock_pool):
    """The socket images as a plain list for `_build_ref2va_full`, which then
    applies the segment's ref_image_size - socket images are scaled exactly
    like picked files, never stretched to a shared canvas by this node."""
    return [t for _, t in _sock_use(seg, sock_pool)]


def _sock_slot(seg, sock_pool, slot_name):
    """One named socket's first image (e.g. SOCK_SLOT_FIRST doubling as
    segment 0's FL2VA first frame), or None when that socket is unwired or
    explicitly ruled out by this segment. [1,H,W,C] like
    ``_load_input_image``, so callers need no special case."""
    for name, t in _sock_use(seg, sock_pool):
        if name == slot_name:
            return t
    return None


def _ref_image_plan(seg, sock_pool):
    """(loaders, ids) for a Ref2VA segment: the sockets it presents (already
    sliced), then its picked files - the exact order and content that
    `_build_ref2va_full` resizes, each with the identity the reference cache
    keys on ('sock:<slot>:<pixel hash>' / 'img:<path|mtime|size>').

    The images are returned as THUNKS, not tensors: `loaders[i]()` yields
    image i. A caller that turns out to have a conditioning-cache hit never
    calls one, so a hit skips not just the encodes but the file reads, the
    decodes, and the resizes that would have fed them - for a segment with a
    handful of references that is the difference between 'instant' and
    'everything again except the text encoder'."""
    loaders, ids = [], []
    for name, t in _sock_use(seg, sock_pool):
        ids.append(_tensor_source_id(t))
        loaders.append(lambda t=t: t)
    for name in seg.get("ref_images") or ():
        ids.append(_file_source_id(name))
        loaders.append(lambda n=name: _load_input_image(n))
    return loaders, ids


def _ref_src(seg, ref_ids):
    """The reference-cache identity map for a segment: 'images' aligned with
    the ref_images list, plus the single-asset slots."""
    return {
        "images": ref_ids,
        "video": _file_source_id(seg["ref_video"])
                 if seg.get("ref_video") else None,
        "video_audio": _file_source_id(seg["ref_video_audio"])
                       if seg.get("ref_video_audio") else None,
        "audio": _file_source_id(seg["ref_audio"])
                 if seg.get("ref_audio") else None,
    }


def _ref_cond_key(cache, tag, seg, w, h, frames, ref_ids, src, vae, audio_vae):
    """The conditioning key of a reference-conditioned segment.

    Covers everything that changes what the text encoder is fed: the prompt,
    the ORDERED identities of the references, the geometry that decides how
    they are resized, and whether each VAE is wired (a missing VAE silently
    drops the DiT-side block and leaves the text-encoder half alone, which is
    a different result). None means 'not cacheable'."""
    if cache is None or not ref_ids or not all(ref_ids):
        return None
    return cache.cond_key(tag, w, h, frames, seg.get("ref_image_size", "match"),
                          int(vae is not None), int(audio_vae is not None),
                          seg["prompt"], ref_ids, src["video"],
                          src["video_audio"], src["audio"])


def _text_cond(clip, prompt, cache=None):
    """Text-only conditioning, cached.

    Negatives and the reference-free (T2VA) path encode the same prompt over
    and over: once per segment, once per run. The key is the prompt plus the
    cache's model fingerprint, nothing else."""
    if cache is None:
        return _encode_text(clip, prompt)
    key = cache.cond_key("text", prompt)
    hit = cache.load_cond(key)
    if hit is not None:
        return hit
    cond = _encode_text(clip, prompt)
    cache.save_cond(key, cond)
    return cond


def _build_first_conditioning(clip, vae, audio_vae, seg, w, h, frames,
                              sock_pool=None, cache=None):
    """Conditioning for segment 0 (fresh generation, no seam anchor).

    FL2VA: first-frame keyframe (+ optional last-frame keyframe); the first/
    last slots fall back to the Tile Editor's 'First_or_Ref_Image_0' /
    'Last_or_Ref_Image_1' sockets when no file is picked for them.
    Ref2VA: native-order reference blocks - latent refs, images, a reference
    video (+ its soundtrack) and standalone reference audio (<Picture i> /
    <Video k> / <Audio j>). No refs: plain prompt encoding.

    `cache` enables the content-addressed conditioning cache. Segment 0 has no
    seam reference, so every one of its references is a fixed asset: its
    conditioning is fully content addressable and a re-run with an unchanged
    prompt/asset/size combination skips the text encoder entirely."""
    first_img = seg.get("fl2va_first_image") or ""
    last_img = seg.get("fl2va_last_image") or ""

    def pick(file_name, slot):
        """(image, cache identity) - a picked file wins over the wired
        socket, exactly like the presentation order below."""
        if file_name:
            return _load_input_image(file_name), _file_source_id(file_name)
        t = _sock_slot(seg, sock_pool, slot)
        if t is None:
            return None, None
        return t, f"sock:{slot}:{_tensor_source_id(t)}"

    if seg["mode"] == "FL2VA":
        from comfy_extras.nodes_minimax_h3 import _resize
        first, first_id = pick(first_img, SOCK_SLOT_FIRST)
        last, last_id = pick(last_img, SOCK_SLOT_LAST)
        # (role, cache id, resized image, keyframe frame index)
        plan = []
        if first is not None:
            # centre-aligned cover crop, same as the last frame: `_resize`'s
            # 4th argument is the crop mode, where "center" first crops to the
            # target aspect and then scales ("cover"), while "disabled" scales
            # straight to (w, h) and therefore DISTORTS whenever the input's
            # aspect ratio differs from the canvas.
            plan.append(("first", first_id, _resize(first, w, h, "center"), 0))
        if last is not None:
            plan.append(("last", last_id,
                         _resize(last, w, h, "center"), frames - 1))
        if plan and vae is None:
            # Without a VAE this frame would reach only the text encoder, so
            # the video would ignore it while the run still reported success -
            # the user reads that as "the plugin is broken". Fail loudly.
            raise ValueError(
                "FL2VA needs a VAE: the first/last frame is delivered as a "
                "keyframe latent, and with no VAE connected it would reach "
                "only the text encoder - the output would not follow the "
                "frame. Connect a VAE, or clear the first/last image.")

        key = None
        if cache is not None and all(kid for _, kid, _, _ in plan):
            key = cache.cond_key("f2v", w, h, frames, seg["prompt"],
                                 [f"kf:{role}:{kid}"
                                  for role, kid, _, _ in plan])
            hit = cache.load_cond(key)
            if hit is not None:
                return hit, _text_cond(clip, seg["negative"], cache)

        tokens = clip.tokenize(seg["prompt"],
                               images=[img for _, _, img, _ in plan])
        pos = clip.encode_from_tokens_scheduled(tokens)
        if plan:
            import node_helpers
            keyframes = [
                {"resolved_frame_index": fi,
                 "latent": _cached_ref_latent(cache, "kf", kid, (w, h, role),
                                              lambda im=img: vae.encode(im))}
                for role, kid, img, fi in plan]
            pos = node_helpers.conditioning_set_values(
                pos, {"minimax_keyframes": keyframes})
        if key is not None:
            cache.save_cond(key, pos)
        return pos, _text_cond(clip, seg["negative"], cache)

    sock_imgs = _sock_refs(seg, sock_pool)
    if seg["mode"] == "Ref2VA" and (seg["ref_images"] or seg.get("ref_video")
                                    or seg.get("ref_audio") or sock_imgs):
        loaders, ref_ids = _ref_image_plan(seg, sock_pool)
        src = _ref_src(seg, ref_ids)
        key = _ref_cond_key(cache, "r2v", seg, w, h, frames, ref_ids, src,
                            vae, audio_vae)
        if key is not None:
            hit = cache.load_cond(key)
            if hit is not None:
                return hit, _text_cond(clip, seg["negative"], cache)
        refs = [load() for load in loaders]     # only now is disk touched
        pos = _build_ref2va_full(
            clip, vae, audio_vae, seg["prompt"], w, h, frames, refs,
            seg.get("ref_image_size", "match"),
            ref_video=seg.get("ref_video") or None,
            ref_video_audio=seg.get("ref_video_audio") or None,
            ref_audio=seg.get("ref_audio") or None,
            cache=cache, src=src)
        if key is not None and pos is not None:
            cache.save_cond(key, pos)
        return pos, _text_cond(clip, seg["negative"], cache)

    # no usable reference: pure text conditioning
    return (_text_cond(clip, seg["prompt"], cache),
            _text_cond(clip, seg["negative"], cache))


def _build_seam_video_ref(vae, prev_video, k_split, n_frames=0, prev_audio=None):
    """Previous merged video's tail as a bit-perfect latent video reference.

    This implements the 'prev_tail' seam reference: seam continuity is
    carried by the native Ref2VA reference mechanism instead of a soft
    noise-mask (H3 generates broken content for intermediate mask values).
    The tail latents are fed bit-perfect to the DiT (no VAE round-trip); one
    video decode of the same latent provides the text-encoder presentation
    frames. `prev_audio` (the accumulated AV latent's audio stream, format
    [B, 32, 2, T_a]) rides along as the reference video's soundtrack: its
    tail is sliced on the same pixel-frame span (FRAME_RESCALE per pixel
    frame, time on the last axis) and the block becomes kind 'video_audio',
    which the DiT packs right before the block's video rows. Returns
    (ref_items, ref_blocks), or None when the tail is too short to be a
    valid video reference (needs >= 5 frames)."""
    tv = prev_video.shape[2]
    if n_frames and int(n_frames) > 0:
        # pixel-frame request on the 17m+5 grid -> 5m+2 latent tokens
        m = max(0, int(round((int(n_frames) - 5) / 17.0)))
        tokens = min(tv, max(2, 5 * m + 2))
    else:
        tokens = max(2, tv - int(k_split))
    tail = prev_video[:, :, tv - tokens:].contiguous()
    px = vae.decode(tail)                      # [1, N, H, W, 3] in [0, 1]
    px = px[0] if px.shape[0] == 1 else px
    n = px.shape[0]
    while n % 17 != 5 and n > 5:               # presentation needs 17m+5 frames
        n -= 1
    if n < 5:
        return None
    px = px[:n].contiguous()
    sample_idx = list(range(0, n, 12))         # Qwen sees the video at 2 fps
    ref_items = [{"type": "video", "data": px[sample_idx],
                  "timestamps": [i / 2.0 for i in range(len(sample_idx))]}]
    a_tail = None
    if prev_audio is not None:
        f_start = frames_for_tokens(tv - tokens)
        a_start = round(f_start * FRAME_RESCALE)
        if prev_audio.shape[-1] - a_start >= 2:
            a_tail = prev_audio[..., a_start:].contiguous()
    if a_tail is not None:
        # the soundtrack's <Audio j> label goes right before its <Video k>
        ref_items.insert(0, {"type": "audio"})
    ref_blocks = [{"kind": "video_audio" if a_tail is not None else "video",
                   "latent_t": tail.shape[2],
                   "latent_h": prev_video.shape[3],
                   "latent_w": prev_video.shape[4],
                   "ref_audio_t": (a_tail.shape[-1]
                                   if a_tail is not None else 0),
                   "latent": tail, "audio_latent": a_tail}]
    return ref_items, ref_blocks


def _build_continue_conditioning(clip, vae, audio_vae, seg, w, h, frames,
                                 prev_video, k_seam, anchor_strength,
                                 seam_ref=None, latent_audio_refs=None,
                                 sock_pool=None, cache=None):
    """Conditioning for a continuation segment (segment >= 1).

    The frame-0 keyframe slot is owned by the seam anchor (frozen tail's
    first frame). 'FL2VA' here means a LAST-frame-only keyframe (the user's
    target ending image); 'Ref2VA' means native-order reference blocks -
    an optional frame of the previous merged video (bit-perfect latent
    reference), manual images, a reference video (+ soundtrack) and/or
    standalone reference audio; 'none' means prompt only.

    `seam_ref` is an optional (ref_items, ref_blocks) pair carrying the
    previous merged video's tail as a bit-perfect latent VIDEO reference
    ('prev_tail' seam reference). It is merged into the presentation in the
    native order (after the segment's own video references) and is also
    honored when the segment itself has no references of its own.

    `cache` enables the conditioning cache, but ONLY for a segment whose
    references are all fixed assets. A reference derived from what the
    previous segments just produced (the seam tail, a frame of the merged
    video, a slice of the accumulated soundtrack) is ROLLING - identical
    inputs do not reproduce it, so such a segment always encodes fresh."""
    pos = None
    sock_last = _sock_slot(seg, sock_pool, SOCK_SLOT_LAST)

    if seg["mode"] == "FL2VA" and (seg.get("fl2va_last_image")
                                   or sock_last is not None):
        from comfy_extras.nodes_minimax_h3 import _resize
        import node_helpers
        if vae is None:
            # same reason as _build_first_conditioning: a last frame with no
            # VAE would silently degrade to a text-only prompt
            raise ValueError(
                "FL2VA needs a VAE: the last frame is delivered as a "
                "keyframe latent, and with no VAE connected it would reach "
                "only the text encoder - the output would not follow the "
                "frame. Connect a VAE, or switch this segment to Ref2VA.")
        # a picked file wins; otherwise the 'Last_or_Ref_Image_1' socket
        last_id = (_file_source_id(seg["fl2va_last_image"])
                   if seg.get("fl2va_last_image")
                   else f"sock:{SOCK_SLOT_LAST}:{_tensor_source_id(sock_last)}")
        # the tokenizer is either images= OR minimax_ref_items=; with a seam
        # reference the last frame joins the presentation as a <Picture> item,
        # which is exactly what makes the key depend on the seam too
        key = None
        if cache is not None and seam_ref is None:
            key = cache.cond_key("f2vc", w, h, frames, seg["prompt"],
                                 f"kf:last:{last_id}")
            hit = cache.load_cond(key)
            if hit is not None:
                return hit, _text_cond(clip, seg["negative"], cache)

        last = _load_input_image(seg["fl2va_last_image"]) \
            if seg.get("fl2va_last_image") else sock_last
        img = _resize(last, w, h, "center")
        kf = {"resolved_frame_index": frames - 1,
              "latent": _cached_ref_latent(cache, "kf", last_id,
                                           (w, h, "last"),
                                           lambda: vae.encode(img))}
        if seam_ref is not None:
            items, blocks = seam_ref
            tokens = clip.tokenize(seg["prompt"], minimax_ref_items=[
                {"type": "image", "data": img}] + items)
            pos = clip.encode_from_tokens_scheduled(tokens)
            pos = node_helpers.conditioning_set_values(
                pos, {"minimax_keyframes": [kf], "minimax_refs": blocks})
        else:
            tokens = clip.tokenize(seg["prompt"], images=[img])
            pos = clip.encode_from_tokens_scheduled(tokens)
            pos = node_helpers.conditioning_set_values(
                pos, {"minimax_keyframes": [kf]})
        if key is not None:
            cache.save_cond(key, pos)
    elif seg["mode"] == "Ref2VA" and (seg["ref_images"] or
                                      seg.get("ref_video") or
                                      seg.get("ref_audio") or
                                      seg["ref_source"] == "prev_frame" or
                                      latent_audio_refs or
                                      _sock_refs(seg, sock_pool)):
        loaders, ref_ids = _ref_image_plan(seg, sock_pool)
        src = _ref_src(seg, ref_ids)
        # 'prev_frame' latent refs, the seam tail and latent audio refs all
        # ride on the accumulated latent -> ROLLING, never cached
        rolling = (seg["ref_source"] == "prev_frame" or bool(latent_audio_refs)
                   or seam_ref is not None)
        key = None
        if not rolling:
            key = _ref_cond_key(cache, "r2vc", seg, w, h, frames, ref_ids, src,
                                vae, audio_vae)
            if key is not None:
                hit = cache.load_cond(key)
                if hit is not None:
                    return hit, _text_cond(clip, seg["negative"], cache)

        refs = [load() for load in loaders]     # only now is disk touched
        latent_refs = []
        if seg["ref_source"] == "prev_frame":
            tok = _frame_to_token(int(seg.get("prev_frame_index", -1))
                                  % prev_video.shape[2]
                                  if int(seg.get("prev_frame_index", -1)) >= 0
                                  else prev_video.shape[2] - 1,
                                  prev_video.shape[2])
            latent_refs.append({
                "kind": "image",
                "latent_h": prev_video.shape[3],
                "latent_w": prev_video.shape[4],
                "latent": prev_video[:, :, tok:tok + 1].contiguous(),
            })
        pos = _build_ref2va_full(
            clip, vae, audio_vae, seg["prompt"], w, h, frames, refs,
            seg.get("ref_image_size", "match"),
            ref_video=seg.get("ref_video") or None,
            ref_video_audio=seg.get("ref_video_audio") or None,
            ref_audio=seg.get("ref_audio") or None,
            latent_refs=latent_refs,
            extra_video_refs=[seam_ref] if seam_ref is not None else None,
            latent_audio_refs=latent_audio_refs,
            cache=cache, src=src)
        if key is not None and pos is not None:
            cache.save_cond(key, pos)
    if pos is None and seam_ref is not None:
        import node_helpers
        items, blocks = seam_ref
        tokens = clip.tokenize(seg["prompt"], minimax_ref_items=items)
        pos = clip.encode_from_tokens_scheduled(tokens)
        pos = node_helpers.conditioning_set_values(pos, {"minimax_refs": blocks})
    if pos is None:
        pos = _text_cond(clip, seg["prompt"], cache)
    return pos, _text_cond(clip, seg["negative"], cache)


# ---------------------------------------------------------------------------
# node
# ---------------------------------------------------------------------------

class MMH3SampleParams(io.ComfyNode):
    """Sampling parameter bundle for MMH3 Temporal Extend Video.

    Carries a two-stage split-sigma setup, mirroring two chained native
    SamplerCustom nodes fed by SplitSigmas: a HIGH stage (noise injection,
    sigma_max -> split sigma) and an optional LOW stage (NO noise added,
    split sigma -> 0), each with its own model/sampler/sigmas/cfg.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3SampleParams",
            display_name="MMH3 Sample Params",
            category="model/latent/minimax",
            description=(
                "Sampling parameters for 'MMH3 Temporal Extend Video'. "
                "Two-stage split-sigma setup: the HIGH stage runs sigma_max -> "
                "split sigma with noise injection; the optional LOW stage "
                "continues split sigma -> 0 with NO noise added, possibly with a "
                "different model/sampler/cfg. Connect all three model_low / "
                "sigma_low / sampler_low inputs to enable the LOW stage, or "
                "leave them all unconnected to run the HIGH stage only."
            ),
            inputs=[
                io.Model.Input("model_high",
                               tooltip="Diffusion model for the HIGH stage (noise injection, sigma_max -> split sigma)."),
                io.Sigmas.Input("sigma_high",
                                tooltip="Sigma schedule for the HIGH stage - e.g. the 'high' output of SplitSigmas; its LAST sigma is the split sigma."),
                io.Sampler.Input("sampler_high",
                                 tooltip="Sampler for the HIGH stage."),
                io.Float.Input("cfg_high", default=1.0, min=0.0, max=100.0, step=0.1, round=0.01,
                               tooltip="CFG of the HIGH stage (applies when negative conditioning is connected)."),
                io.Model.Input("model_low", optional=True,
                               tooltip="Diffusion model for the optional LOW stage (no noise added, split sigma -> 0). ALL THREE low inputs must be connected together to enable the LOW stage."),
                io.Sigmas.Input("sigma_low", optional=True,
                                tooltip="Sigma schedule for the optional LOW stage - e.g. the 'low' output of SplitSigmas; its FIRST sigma must equal sigma_high's last (split) sigma."),
                io.Sampler.Input("sampler_low", optional=True,
                                 tooltip="Sampler for the optional LOW stage."),
                io.Float.Input("cfg_low", default=1.0, min=0.0, max=100.0, step=0.1, round=0.01, optional=True,
                               tooltip="CFG of the LOW stage (applies when negative conditioning is connected). Defaults to cfg_high when unconnected."),
            ],
            outputs=[
                SAMPLE_PARAMS.Output("sample_params",
                                     tooltip="Connect to the 'sample_params' input of 'MMH3 Temporal Extend Video'."),
            ],
        )

    @classmethod
    def validate_inputs(cls, **kwargs) -> bool | str:
        low = [kwargs.get(n) for n in ("model_low", "sigma_low", "sampler_low")]
        present = [v is not None for v in low]
        if any(present) and not all(present):
            return ("model_low / sigma_low / sampler_low must be connected "
                    "together, or all left unconnected to skip the LOW stage.")
        return True

    @classmethod
    def execute(cls, model_high, sigma_high, sampler_high, cfg_high,
                model_low=None, sigma_low=None, sampler_low=None,
                cfg_low=None) -> io.NodeOutput:
        low_active = (model_low is not None and sigma_low is not None
                      and sampler_low is not None)
        params = {
            "model_high": model_high,
            "sampler_high": sampler_high,
            "sigmas_high": sigma_high,
            "cfg_high": cfg_high,
            "low_active": low_active,
            "model_low": model_low if low_active else None,
            "sampler_low": sampler_low if low_active else None,
            "sigmas_low": sigma_low if low_active else None,
            "cfg_low": (cfg_low if cfg_low is not None else cfg_high) if low_active else None,
        }
        return io.NodeOutput(params)


# ── one-click continuation presets (MMH3 Temporal Overlap Simple) ──────────
# The Simple node exposes a single `preset` combo instead of the thirteen
# knobs above. Each preset is expanded against this base into a COMPLETE
# parameter dict, so the preset alone decides the continuation behavior - a
# leftover value in the Tile Editor's global extend_params can never leak in
# and change it. Keys not mentioned by any preset keep the stock defaults.
OVERLAP_PRESET_BASE = {
    "tail_mode": "freeze_fade",
    "fade_value": 0.5,
    "fade_mode": "gradient",
    "fade_impl": "mask",
    "init_content_weight": 0.3,
    "max_mask_strength": 1.0,
    "keyframes_mode": "reanchor",
    "anchor_seam": True,
    "anchor_strength": 0.999,
    "seam_reference": "none",
    "seam_ref_frames": 0,
    "overlap_mode": "earlier",
    "overlap_blend": "linear",
}

# Each preset lists ONLY its differences from the base, so what separates two
# presets stays readable. `fade_frames` is never preset-specific: it is the one
# setting that must follow the segment's carried-tail length (the editor's
# per-segment 'overlap frames'), so it stays a user knob on the node.
OVERLAP_PRESETS = {
    # Smooth, priority on continuity; the more rounds you chain, the more
    # quality degrades.
    "high continuity": {},
    # Transition stays smooth with almost no deterioration, but content
    # consistency tends to jump.
    "low deterioration": {"max_mask_strength": 0.2,
                          "anchor_seam": False},
    # Roughly keeps continuity with a tiny seam jump, no visible deterioration.
    "soft restart": {"tail_mode": "free_resample",
                     "anchor_strength": 0.6,
                     "overlap_mode": "later",
                     "overlap_blend": "overwrite"},
    # Use H3's native reference-video feature; needs prompt cooperation and
    # samples much slower.
    "H3 video reference": {"tail_mode": "free_resample",
                           "keyframes_mode": "drop",
                           "anchor_seam": False,
                           "seam_reference": "prev_tail",
                           "seam_ref_frames": 0,
                           "overlap_mode": "later",
                           "overlap_blend": "overwrite"},
}


class MMH3TemporalOverlapParams(io.ComfyNode):
    """Global tail/fade/anchor/overlap parameters for MMH3 Temporal Extend Video.

    These apply to EVERY continuation segment (segments >= 1) of the chain.
    The per-segment exception is the carried-tail LENGTH, which each segment
    sets individually ("overlap frames" in the Temporal Tile Editor).
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3TemporalOverlapParams",
            display_name="MMH3 Temporal Overlap Params",
            category="model/latent/minimax",
            description=(
                "Global continuation parameters for 'MMH3 Temporal Extend "
                "Video': how the previous segment's tail is carried into the "
                "next one (freeze/fade behavior), the fade band, seam "
                "anchoring, the optional 'prev_tail' seam video reference and "
                "the overlap blend. Applies to every continuation segment; "
                "the carried-tail LENGTH is set per segment in the Temporal "
                "Tile Editor ('overlap frames')."
            ),
            search_aliases=["h3 overlap params", "fade", "tail", "seam anchor",
                            "overlap blend", "temporal overlap", "seam reference"],
            inputs=[
                io.Combo.Input("tail_mode", options=["freeze_fade", "free_resample"],
                               default="freeze_fade",
                               tooltip="Carried-tail behavior: 'freeze + fade' pins the frozen zone and fades into the new content; 'free resample' lets the whole tail re-sample (experimental)."),
                io.Int.Input("fade_frames", default=0, min=0, max=510, step=17,
                             tooltip="Length in PIXEL frames of the fade band between the carried tail and the new content. 0 = no fade: the whole tail stays frozen and the new content hard-cuts in. Other values snap up to multiples of 17."),
                io.Float.Input("fade_value", default=0.5, min=0.0, max=1.0, step=0.05,
                               tooltip="Fade strength at the transition (0 = hard cut toward the new content, 1 = keep the tail)."),
                io.Combo.Input("fade_mode", options=["flat", "gradient", "smoothstep"],
                               default="flat",
                               tooltip="Shape of the fade across the band."),
                io.Combo.Input("fade_impl", options=["mask", "qsample_init"],
                               default="mask",
                               tooltip="EXPERIMENTAL, may be removed. Fade implementation: 'mask' = the default soft noise-mask band (intermediate per-row timesteps; known to produce mosaic-like artifacts on H3). 'qsample_init' = keeps the band at full-strength timesteps and steers the transition by q_sample-blending the carried content into the INITIAL NOISE, decaying as sampling proceeds. Advantage over 'mask': the band never sees intermediate timesteps, so subsequent segments are far more likely to continue successfully, at zero extra cost. SAMPLER NOTE: the stochastic SDE type that re-injects fresh noise every step (sa_solver, er_sde, dpmpp_2m_sde, dpmpp_3m_sde, ...) is the BETTER FIT and is recommended; deterministic ODE samplers (euler, res_multistep, uni_pc, dpmpp_2m, ...) are more likely to develop brightness/saturation drift in the band."),
                io.Float.Input("max_mask_strength", default=1.0, min=0.0, max=1.0, step=0.05,
                               tooltip="Strength of the freeze + fade mask as a whole. 1 = maximum: the frozen zone is fully held (mask 0) and the fade band runs its full gradient. 0 = no mask at all: the frozen zone is not held and the fade band is gone, so the whole carried tail re-samples (equivalent to 'free resample'). Intermediate values hold the frozen zone proportionally and scale the fade gradient with it (mask value in the frozen zone = 1 - s, fade gradient runs from 1 - s up to 1)."),
                io.Float.Input("init_content_weight", default=0.3, min=-1.0, max=1.0, step=0.05,
                               tooltip="EXPERIMENTAL, only effective with fade_impl='qsample_init'. Caps how much carried-content STRUCTURE rides in the initial noise (1.0 = full structure as before; 0 = pure noise). The structured init at t=sigma_max is out-of-distribution and the model over-develops it - the source of the severe brightness/saturation drift. The higher this value, the more likely the fade band produces frames with color anomalies. NOTE ON SAMPLERS: the stochastic SDE type (er_sde / sa_solver / dpmpp_2m_sde / dpmpp_3m_sde) is the BETTER FIT and is recommended; deterministic ODE samplers (euler, res_multistep, uni_pc, dpmpp_2m, ...) tend to show drift in the band regardless of this value. NEGATIVE values (experiment): the band carries SIGN-FLIPPED, anti-correlated structure. This is NOT 'more different from the neighbour' (0 already is maximal independence) - the start stays pinned to the reference, just inverted; expect mirrored/inverted development or nothing at all."),
                io.Combo.Input("keyframes_mode", options=["reanchor", "drop", "keep"],
                               default="reanchor",
                               tooltip="How previous-segment keyframes are handled after the split: 'reanchor' moves them to the new split point, 'drop' removes them, 'keep' leaves them untouched."),
                io.Boolean.Input("anchor_seam", default=True,
                                 tooltip="Anchor the conditioning at the split frame so the seam stays in place."),
                io.Float.Input("anchor_strength", default=0.999, min=0.0, max=1.0, step=0.01,
                               tooltip="Strength of the seam anchor (1.0 = hard pin)."),
                io.Combo.Input("seam_reference", options=["none", "prev_tail"],
                               default="none",
                               tooltip="Seam continuity WITHOUT fade: 'prev_tail' feeds the previous merged video's tail to every continuation segment as a bit-perfect latent video reference - WITH its soundtrack (the tail's audio latent rides along, kind 'video_audio') - through the native Ref2VA mechanism. No soft noise-mask involved, so no mosaic artifacts. Recommended with fade_frames=0 and a generous per-segment 'overlap frames' setting."),
                io.Int.Input("seam_ref_frames", default=0, min=0, max=1020, step=17,
                             tooltip="How many pixel frames of the previous merged video the seam reference covers, snapped to the 17m+5 grid (5, 22, 39, ...). 0 = the entire carried tail (the segment's overlap frames). Ignored when seam_reference is 'none'."),
                io.Combo.Input("overlap_mode", options=["later", "earlier"], default="later",
                               tooltip="Who wins the overlap band: 'later' = the new segment, 'earlier' = the accumulated video."),
                io.Combo.Input("overlap_blend", options=["linear", "smoothstep", "midpoint", "overwrite"],
                               default="linear",
                               tooltip="Blend curve across the overlap band ('overwrite' = no blending). When a fade band exists, the blend ramp is confined to the frozen zone (where both sides carry identical content - it only smooths micro-differences e.g. from a refinement pass) and the re-developed fade band is assigned wholly to the new segment, since crossfading two genuinely different renditions would ghost."),
            ],
            outputs=[
                OVERLAP_PARAMS.Output("overlap_params",
                                      tooltip="Connect to the 'overlap_params' input of 'MMH3 Temporal Extend Video'."),
            ],
        )

    @classmethod
    def execute(cls, tail_mode="freeze_fade", fade_frames=0, fade_value=0.5,
                fade_mode="flat", fade_impl="mask",
                keyframes_mode="reanchor", anchor_seam=True,
                anchor_strength=0.999, overlap_mode="later",
                overlap_blend="linear", seam_reference="none",
                seam_ref_frames=0, init_content_weight=1.0,
                max_mask_strength=1.0) -> io.NodeOutput:
        return io.NodeOutput({
            "tail_mode": tail_mode,
            "fade_frames": int(fade_frames),
            "fade_value": float(fade_value),
            "fade_mode": fade_mode,
            "fade_impl": fade_impl,
            "init_content_weight": float(init_content_weight),
            "max_mask_strength": float(max_mask_strength),
            "keyframes_mode": keyframes_mode,
            "anchor_seam": bool(anchor_seam),
            "anchor_strength": float(anchor_strength),
            "overlap_mode": overlap_mode,
            "overlap_blend": overlap_blend,
            "seam_reference": seam_reference,
            "seam_ref_frames": int(seam_ref_frames),
        })


class MMH3TemporalOverlapSimple(io.ComfyNode):
    """One-click alternative to MMH3 Temporal Overlap Params.

    Same output socket, same effect on every continuation segment, but reduced
    to a `preset` combo plus the fade band length - meant as the default
    choice. 'MMH3 Temporal Overlap Params' remains for hand tuning.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3TemporalOverlapSimple",
            display_name="MMH3 Temporal Overlap Simple",
            category="model/latent/minimax",
            description=(
                "Simplified 'MMH3 Temporal Overlap Params': four ready-made "
                "continuation presets and nothing else to configure. Pick a "
                "preset, optionally set the fade band length, and connect the "
                "output to the 'overlap_params' input of 'MMH3 Temporal Extend "
                "Video' (same socket as the full node). Applies to every "
                "continuation segment; the carried-tail LENGTH is still set "
                "per segment in the Temporal Tile Editor ('overlap frames')."
            ),
            search_aliases=["h3 overlap simple", "overlap preset",
                            "temporal overlap simple", "one click overlap",
                            "fade preset", "h3 continue preset"],
            inputs=[
                io.Combo.Input("preset", options=list(OVERLAP_PRESETS),
                               default="high continuity",
                               tooltip="One-click continuation preset. 'high continuity' = smooth, freeze+fade gradient mask with full mask strength and a hard seam anchor (quality softens over many rounds). 'low deterioration' = near-no deterioration with a smooth transition, but content consistency tends to jump (weak 0.1 mask, no seam anchor). 'soft restart' = barely any deterioration, the carried tail re-samples and the new segment overwrites the band (rough continuity with a tiny seam jump). 'H3 video reference' = uses H3's native reference-video feature via the 'prev tail' seam reference spanning the whole tail (0 = all carried frames); needs prompting to cooperate and samples much slower."),
                io.Int.Input("fade_frames", default=0, min=0, max=510, step=17,
                             tooltip="Length in PIXEL frames of the fade band between the carried tail and the new content; other values snap up to multiples of 17. 0 = no fade: the whole carried tail stays frozen and the new content hard-cuts in. Must fit inside the segment's carried tail (its 'overlap frames' in the Tile Editor). Ignored by the 'soft restart' and 'H3 video reference' presets (they re-sample the whole tail); 'H3 video reference' is designed for 0, because its seam reference already carries the continuity."),
            ],
            outputs=[
                OVERLAP_PARAMS.Output("overlap_params",
                                      tooltip="Connect to the 'overlap_params' input of 'MMH3 Temporal Extend Video'."),
            ],
        )

    @classmethod
    def execute(cls, preset="high continuity", fade_frames=0) -> io.NodeOutput:
        overrides = OVERLAP_PRESETS.get(preset)
        if overrides is None:
            print(f"[MMH3-TemporalOverlapSimple] WARNING: unknown preset "
                  f"{preset!r} - falling back to 'high continuity'")
            overrides = OVERLAP_PRESETS["high continuity"]
        params = dict(OVERLAP_PRESET_BASE)
        params.update(overrides)
        params["fade_frames"] = max(0, int(fade_frames))
        if params["tail_mode"] == "free_resample" and params["fade_frames"] > 0:
            print("[MMH3-TemporalOverlapSimple] WARNING: this preset re-samples "
                  "the whole tail and ignores the fade "
                  f"band - fade_frames={params['fade_frames']} has no effect.")
        return io.NodeOutput(params)


class MMH3TemporalExtendVideo(io.ComfyNode):
    """Multi-segment H3 video generation: each segment continues the previous
    one's tail.

    Segment 0 generates a fresh video (FL2VA / Ref2VA / prompt-only) from an
    empty AV latent - or continues an externally supplied ``latent_seg_0``.
    Segments >= 1 split the accumulated latent's tail (snapped to the
    17-frame keyframe grid) as their frozen+fade head, guided by the per-
    segment conditioning from the Tile Editor (Ref2VA references, an optional
    last-frame keyframe, or prompt only) plus the frame-0 seam anchor.

    Sampling runs via the MMH3 Sample Params bundle: a HIGH stage (noise
    injection) and an optional LOW stage. The LOW stage starts from the HIGH
    stage's final x0 prediction (SamplerCustom's denoised_output - its frozen
    zone is the pristine carried tail, unlike the trajectory x which still
    carries split-sigma noise), re-noised to the split sigma with seed+1.

    After every segment the merged and segment latents are saved as
    '.h3latent' files with matching preview WebPs (tiny taeh3 VAE) into the
    session directory (ComfyUI/temp or output/latents), so a later run can
    RESUME from any stored segment without re-sampling the earlier ones.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3TemporalExtendVideo",
            display_name="MMH3 Temporal Extend Video",
            category="model/latent/minimax",
            description=(
                "Multi-segment video generation for MiniMax H3: each segment "
                "continues the previous segment's tail (frozen + fade zones "
                "on the 17-frame keyframe grid) and adds its own prompt, "
                "reference images, seed and duration from the 'MMH3 Temporal "
                "Tile Editor'. Segment 0 can start a fresh video (FL2VA / "
                "Ref2VA) instead. Sampling runs in two stages via 'MMH3 "
                "Sample Params' (HIGH with noise, optional LOW continuing "
                "from the HIGH stage's x0 prediction, like two chained "
                "SamplerCustom nodes on a split-sigma schedule). Every "
                "finished segment stores its merged/segment latents and "
                "preview WebPs into the session directory so later runs can "
                "resume from any segment without redoing earlier ones."
            ),
            search_aliases=["h3 temporal extend", "extend video",
                            "video extension", "h3 continue",
                            "multi segment video", "temporal split"],
            inputs=[
                io.Clip.Input("clip",
                              tooltip="MiniMax H3 CLIP - builds every segment's conditioning (prompt, keyframes, references) internally."),
                io.Vae.Input("vae",
                             tooltip="MiniMax H3 Video VAE - encodes reference images / keyframes for the per-segment conditioning."),
                io.Dict.Input("temporal_tile_config",
                              tooltip="Per-segment configuration from the 'MMH3 Temporal Tile Editor' node."),
                SAMPLE_PARAMS.Input("sample_params",
                                    tooltip="Sampling parameters from the 'MMH3 Sample Params' sub-node: a HIGH stage (noise injection) plus an optional LOW stage (no noise)."),
                io.Combo.Input("second_pass_audio", options=["second", "first"],
                               default="second",
                               tooltip="Which pass's AUDIO track is stitched when the second pass is enabled: 'second' = the re-denoised audio, 'first' = the untouched first-pass audio (the video always comes from the second pass)."),
                SAMPLE_PARAMS.Input("2nd_sample_params", optional=True,
                                    tooltip="OPTIONAL sampling parameters for the second (refinement) pass on continuation segments: the whole segment latent (carried tail + new content, video AND audio) is re-denoised with NO video noise mask and NO keyframes - references (images, seam reference) stay active. In 'bgm' audio reference mode the audio keeps its all-zero mask, so the music stays frozen while the video re-samples. Use a low-denoise sigma schedule (e.g. SplitSigmas 'low' output) to pull the accumulated content back toward the model manifold. Unconnected = disabled."),
                OVERLAP_PARAMS.Input("overlap_params", optional=True,
                                     tooltip="OPTIONAL global tail/fade/anchor/overlap parameters applied to every continuation segment - from the hand-tuning 'MMH3 Temporal Overlap Params' sub-node or the one-click 'MMH3 Temporal Overlap Simple' (same socket, use only one). When unconnected, the defaults (or the Tile Editor's legacy global extend_params) are used."),
                io.Vae.Input("audio_vae", optional=True,
                             tooltip="MiniMax H3 Audio VAE - needed only when a segment references an audio FILE (audio reference mode 'load audio', or a ref video soundtrack). The latent-level audio references ('previous audio' / 'initial audio') need NO audio VAE. Without it file-based reference audio only conditions the text encoder."),
                io.Latent.Input("latent_seg_0", optional=True,
                                tooltip="OPTIONAL fully-sampled H3 AV latent to continue from. When connected together with resume_from_segment=0, segment 0 EXTENDS this latent instead of starting fresh; with resume > 0 the session's stored merged latent is used and this input is ignored."),
                REF_IMAGES.Input("ref_image_slots", optional=True,
                                 tooltip="OPTIONAL reference images wired into the 'MMH3 Temporal Tile Editor' (its 'ref_image_slots' output): one record per connected socket at its NATIVE resolution, in slot order. Every segment uses them as 'load images' references except for the sockets it ruled out in the editor, each sized by that segment's 'reference image size' ('match' / 'max', aspect preserved) - the exact same path as the files picked in the dock panel. 'First_or_Ref_Image_0' / 'Last_or_Ref_Image_1' also serve as segment 0's FL2VA first/last frame when no file is picked for them. One socket is one '<Picture i>': a socket fed by a batch contributes its FIRST image only (the MiniMaxH3 Reference to Video convention). Unconnected = only the editor's picked files are used."),
            ],
            outputs=[
                io.Latent.Output("merged_latent",
                                 tooltip="The accumulated video: all segments stitched on one timeline (the final merged state; each run also stores it as merged_<seq>.h3latent, and with every segment locked it is reloaded from the session)."),
                io.Latent.Output("segment_latent",
                                 tooltip="The LAST segment as a standalone latent (its carried tail + the frames it generated) - NOT the merged timeline. With every segment locked it is reloaded from the session's stored segment latent, so a fully locked chain still reports the last segment here instead of duplicating merged_latent."),
                io.Dict.Output("segment_info",
                               tooltip="Per-segment execution summary: session directory, files written, split points, sampling stages."),
            ],
        )

    @classmethod
    def execute(cls, temporal_tile_config, sample_params, clip, vae,
                audio_vae=None, latent_seg_0=None, overlap_params=None,
                second_sample_params=None, second_pass_audio="second",
                ref_image_slots=None, **kwargs) -> io.NodeOutput:
        # The socket is named '2nd_sample_params' (not a valid Python
        # identifier), so ComfyUI hands it over via **kwargs - map it here.
        if second_sample_params is None:
            second_sample_params = kwargs.get("2nd_sample_params")
        import os

        cfg = temporal_tile_config or {}
        segs = cfg.get("segments") or []
        if not segs:
            raise ValueError("temporal_tile_config carries no segments - connect an "
                             "MMH3 Temporal Tile Editor and add at least one segment")
        ext = cfg.get("extend_params", {})
        # chain-wide continuation parameters: Tile Editor globals, overridden
        # by the MMH3 Temporal Overlap Params / Overlap Simple sub-node when
        # connected (both feed the same socket: full control vs one-click
        # preset)
        glob = dict(ext)
        if isinstance(overlap_params, dict) and overlap_params:
            op = overlap_params
            glob["tail_mode"] = op.get("tail_mode", glob.get("tail_mode", "freeze_fade"))
            ff = int(op.get("fade_frames", glob.get("fade_frames", 0)))
            glob["fade_frames"] = 0 if ff <= 0 else snap_17(ff)
            glob["fade_value"] = float(op.get("fade_value", glob.get("fade_value", 0.5)))
            glob["fade_mode"] = op.get("fade_mode", glob.get("fade_mode", "flat"))
            glob["fade_impl"] = op.get("fade_impl", glob.get("fade_impl", "mask"))
            glob["init_content_weight"] = float(op.get("init_content_weight",
                                                       glob.get("init_content_weight", 1.0)))
            glob["max_mask_strength"] = float(op.get("max_mask_strength",
                                                     glob.get("max_mask_strength", 1.0)))
            glob["keyframes_mode"] = op.get("keyframes_mode",
                                            glob.get("keyframes_mode", "reanchor"))
            glob["anchor_seam"] = bool(op.get("anchor_seam",
                                              glob.get("anchor_seam", True)))
            glob["anchor_strength"] = float(op.get("anchor_strength",
                                                   glob.get("anchor_strength", 0.999)))
            glob["overlap_mode"] = op.get("overlap_mode",
                                          glob.get("overlap_mode", "later"))
            glob["overlap_blend"] = op.get("overlap_blend",
                                           glob.get("overlap_blend", "linear"))
            glob["seam_reference"] = op.get("seam_reference",
                                            glob.get("seam_reference", "none"))
            glob["seam_ref_frames"] = int(op.get("seam_ref_frames",
                                                 glob.get("seam_ref_frames", 0)))
        base_prompt = cfg.get("base_prompt", "")
        base_negative = cfg.get("base_negative", "")

        sdir = _session_dir(cfg.get("storage", "temp"),
                            cfg.get("session_name", "session1"))
        # content-addressed reference / conditioning cache, shared by every
        # session of this storage root (see _ContentCache)
        cache = _ContentCache(cfg.get("storage", "temp"), clip=clip, vae=vae,
                              audio_vae=audio_vae)
        resume = int(cfg.get("resume_from_segment", 0) or 0)
        # resume == len(segs) means EVERY segment is locked: the chain is
        # complete and execution just outputs the stored merged latent
        # without sampling (see the last_seg_v branch below). Only resume
        # points BEYOND the chain get clamped.
        resume = max(0, min(resume, len(segs)))
        # authoritative per-segment attempt ledger (see _load_session_ledger)
        ledger = _load_session_ledger(sdir)

        # ── starting state: latent input / resumed file / fresh generation ──
        merged_v = merged_a = None
        start_src = "fresh"
        # frame position where the PREVIOUS segment's content starts on the
        # merged timeline (its 'split_frame', stored per attempt in the
        # ledger) - the anchor for the 'previous audio' latent reference.
        # Reconstructed geometrically when the entry predates the ledger
        # field (see _latent_audio_refs).
        prev_split_frame = None
        if resume > 0:
            # resolve the previous segment's merged latent through the
            # session ledger (seg index -> chosen attempt file), falling
            # back to the legacy index-named file when unmapped.
            prev_name = None
            for entry in ledger.get("segments", []):
                if entry.get("index") == resume - 1:
                    prev_name = (entry.get("files") or {}).get("merged_latent")
                    prev_split_frame = entry.get("split_frame")
                    break
            if not prev_name:
                prev_name = f"merged_{resume - 1}.h3latent"
            # the ledger name may be subdir-qualified ('latents/...') or a
            # bare legacy name whose file sits in the session dir
            prev_path = _find_session_file(sdir, prev_name, fallback_sub=LATENTS_SUBDIR)
            if prev_path is None or not os.path.isfile(prev_path):
                raise ValueError(
                    f"resume_from_segment={resume} but no stored merged latent "
                    f"found: {prev_path} - run the chain once first")
            prev = _load_h3latent(prev_path)["samples"]
            merged_v, merged_a = prev.tensors[0], prev.tensors[1]
            start_src = f"resumed {prev_name}"
        elif latent_seg_0 is not None:
            samples = latent_seg_0["samples"]
            if not is_h3_av_latent(samples):
                raise ValueError("latent_seg_0 must be a MiniMax H3 AV latent "
                                 "(nested video [B,24,T,H,W] + audio [B,32,2,T])")
            merged_v, merged_a = samples.tensors[0], samples.tensors[1]
            if merged_v.shape[0] != 1:
                raise ValueError("latent_seg_0 expects a single-video latent (batch 1)")
            start_src = "latent_seg_0 input"

        first = segs[0]
        W = int(first.get("width", 768))
        H = int(first.get("height", 768))
        if merged_v is not None:
            # a carried latent defines the chain resolution - the first
            # segment's configured size must match or the stitch breaks
            if (merged_v.shape[4] != W // 16 or merged_v.shape[3] != H // 16):
                print(f"[MMH3-TemporalExtend] WARNING: first segment resolution "
                      f"{W}x{H} differs from carried latent "
                      f"{merged_v.shape[4]*16}x{merged_v.shape[3]*16}; using the "
                      f"carried latent's resolution")
                W = merged_v.shape[4] * 16
                H = merged_v.shape[3] * 16

        # ── reference sockets wired into the Tile Editor ──
        # One record per socket at its native resolution - the editor no
        # longer resizes or merges them, so a segment's ref_socket_off still
        # rules whole sockets out by name (no index mapping needed) and each
        # socket's own aspect ratio survives into the reference pipeline.
        sock_pool = []
        wired_slots = ref_image_slots.get("slots") \
            if isinstance(ref_image_slots, dict) else None
        for slot in wired_slots or ():
            if not isinstance(slot, dict):
                continue
            name, t = slot.get("name"), slot.get("images")
            if not name or t is None or not hasattr(t, "shape") \
                    or not int(t.shape[0]):
                continue
            sock_pool.append((str(name), t))
        if sock_pool:
            print(f"[MMH3-TemporalExtend] {len(sock_pool)} wired reference "
                  f"image socket(s): "
                  + ", ".join(f"{n} ({int(t.shape[0])} img)"
                              for n, t in sock_pool))
            batched = [(n, int(t.shape[0])) for n, t in sock_pool
                       if int(t.shape[0]) > 1]
            if batched:
                print("[MMH3-TemporalExtend] NOTE: "
                      + ", ".join(f"'{n}' carries {c} images" for n, c in batched)
                      + " - only the FIRST image of each socket is used "
                        "(MiniMaxH3 Reference to Video convention)")

        print(f"[MMH3-TemporalExtend] session '{cfg.get('session_name')}' at {sdir} | "
              f"segments {len(segs)} | start from segment {resume} ({start_src}) | "
              f"{W}x{H}")

        seg_info = {
            "session_dir": sdir,
            "storage": cfg.get("storage", "temp"),
            "resume_from_segment": resume,
            "resolution": [W, H],
            "start_source": start_src,
            "segments": [],
        }

        def run_two_stage(piece, pos, neg, seed, tag, params=None):
            sp = params if params is not None else sample_params
            noise = Noise_RandomNoise(int(seed))
            # EXPERIMENTAL 'qsample_init': bias the initial noise with the
            # carried content (fade profile), see _InitBiasNoise. Popped from
            # the piece dict so it never reaches the LOW stage.
            fade_profile = piece.pop("_fade_init", None)
            if fade_profile is not None:
                # (profile, content sign) since negative init_content_weight
                # was allowed; the tuple form is the only producer.
                prof, csign = (fade_profile if isinstance(fade_profile, tuple)
                               else (fade_profile, 1.0))
                noise = _InitBiasNoise(noise, prof,
                                       piece["samples"].tensors[0], csign)
                print(f"[MMH3-TemporalExtend] {tag} fade_impl=qsample_init: "
                      "fade band biased into the initial noise")
            x0 = {} if sp["low_active"] else None
            seg = sample_piece(piece, pos, sp["model_high"], noise,
                               sp["sampler_high"],
                               sp["sigmas_high"], neg,
                               sp["cfg_high"], x0_output=x0)
            if sp["low_active"]:
                sh = sp["sigmas_high"]
                sl = sp["sigmas_low"]
                if not torch.allclose(sh[-1], sl[0]):
                    raise ValueError(
                        "sigma_low's first sigma does not match sigma_high's "
                        f"last sigma ({float(sl[0])} vs {float(sh[-1])}) - use "
                        "SplitSigmas so both schedules share the split sigma")
                x0l = x0.get("x0_latent")
                if x0l is None:
                    raise ValueError("HIGH stage produced no x0 prediction "
                                     "(empty sigma schedule?)")
                cont = dict(piece)
                cont["samples"] = x0l
                noise_low = Noise_RandomNoise(
                    (int(seed) + 1) & 0xFFFFFFFFFFFFFFFF)
                print(f"[MMH3-TemporalExtend] {tag} LOW stage: HIGH x0 "
                      f"re-noised to sigma {float(sl[0]):.4f} (seed {seed + 1})")
                seg = sample_piece(cont, pos, sp["model_low"],
                                   noise_low, sp["sampler_low"],
                                   sl, neg, sp["cfg_low"])
            return seg

        def save_segment(i, seg_v, seg_a, merged_v, merged_a, info):
            try:
                # every execution of a segment gets a fresh seq -> new
                # attempt files; previous attempts are never overwritten.
                seq = int(ledger.get("next_seq", 0))
                ledger["next_seq"] = seq + 1
                ldir = _session_subdir(sdir, LATENTS_SUBDIR)
                _save_h3latent(_pjoin(ldir, f"segment_{seq}.h3latent"),
                               comfy.nested_tensor.NestedTensor((seg_v, seg_a)))
                _save_h3latent(_pjoin(ldir, f"merged_{seq}.h3latent"),
                               comfy.nested_tensor.NestedTensor((merged_v, merged_a)))
                # previews land in their own subdirectory; the ledger stores
                # the SESSION-RELATIVE name for both ('latents/...',
                # 'previews/...'), which is what the HTTP route serves.
                files = _save_preview(sdir, seq, merged_v, seg_v)
                files["merged_latent"] = f"{LATENTS_SUBDIR}/merged_{seq}.h3latent"
                files["segment_latent"] = f"{LATENTS_SUBDIR}/segment_{seq}.h3latent"
                info["files"] = files
                info["attempt"] = seq
                # the newest attempt of segment i becomes its chosen mapping
                kept = [s for s in ledger.get("segments", [])
                        if s.get("index") != i]
                kept.append({"index": i, "chosen_seq": seq, "files": files,
                             "split_frame": info.get("split_frame")})
                kept.sort(key=lambda s: s.get("index", -1))
                ledger["segments"] = kept
                ledger["last_segment"] = max(int(ledger.get("last_segment", -1)), i)
                ledger.setdefault("runs", []).append({
                    "seq": seq, "seg_index": i,
                    "seed": info.get("seed"), "mode": info.get("mode"),
                    "files": files,
                })
                _save_session_ledger(sdir, ledger)
                _broadcast_segment_ready(seq, i, files, len(segs))
            except Exception as exc:
                print(f"[MMH3-TemporalExtend] saving segment {i} failed: {exc}")
                info["save_error"] = str(exc)
            # (the legacy index-keyed metadata.json write was replaced by the
            # authoritative session.json ledger above)

        def _last_segment_latent():
            """The FINAL segment's OWN standalone latent (its carried tail plus
            the frames it generated) - the tensor the 'segment_latent' output
            carries when that segment is sampled during this run.

            Needed when EVERY segment is locked (resume == len(segs)): the
            sampling loop never runs, so there is no in-memory segment latent
            and the last one has to come back off the session disk - the
            ledger-mapped attempt of the last segment, legacy index name as a
            fallback, the same way the resume path resolves the merged ledger.
            Falls back to the merged latent (what this output used to return
            unconditionally) only when the file is missing or unreadable."""
            idx = len(segs) - 1
            name = None
            for entry in ledger.get("segments", []):
                if entry.get("index") == idx:
                    name = (entry.get("files") or {}).get("segment_latent")
                    break
            if not name:
                name = f"segment_{idx}.h3latent"
            path = _find_session_file(sdir, name, fallback_sub=LATENTS_SUBDIR)
            if path is not None and os.path.isfile(path):
                try:
                    st = _load_h3latent(path)["samples"]
                    # clone(): the loader may hand back mmap-backed views,
                    # which keep the file locked on Windows
                    return (st.tensors[0].clone(), st.tensors[1].clone())
                except Exception as exc:
                    print(f"[MMH3-TemporalExtend] WARNING: could not load the "
                          f"stored segment-{idx} latent for the "
                          f"'segment_latent' output ({exc}); falling back to "
                          "the merged latent")
            else:
                print(f"[MMH3-TemporalExtend] WARNING: stored segment-{idx} "
                      f"latent not found ({path}) - the 'segment_latent' "
                      "output falls back to the merged latent")
            return merged_v, merged_a

        # seg0's audio for the 'initial audio' reference: captured in memory
        # once segment 0 is sampled this run, resolved from the session dir
        # otherwise (see _seg0_audio_ref). ref_cache memoizes the resolution.
        seg0_audio = None
        ref_cache = {}

        def _seg0_audio_ref():
            """Seg0's ORIGINAL soundtrack for the 'initial audio' reference.

            Source priority: the in-memory rendition once segment 0 has been
            sampled this run, else the stored segment-0 latent from the
            session dir (ledger-mapped attempt file, legacy index name as
            fallback). The stored segment latent is bit-perfect seg0 audio -
            unlike the merged latent's head, which later segments' carried
            tails may have micro-crossfaded over. Falls back to that merged
            head only when the stored segment latent is unavailable.
            Resolved once and cached."""
            if "seg0_a" in ref_cache:
                return ref_cache["seg0_a"]
            a = seg0_audio
            if a is None:
                name = None
                for entry in ledger.get("segments", []):
                    if entry.get("index") == 0:
                        name = (entry.get("files") or {}).get("segment_latent")
                        break
                if not name:
                    name = "segment_0.h3latent"
                path = _find_session_file(sdir, name)
                if path is not None and os.path.isfile(path):
                    try:
                        a = _load_h3latent(path)["samples"].tensors[1]
                    except Exception as exc:
                        print(f"[MMH3-TemporalExtend] WARNING: could not load "
                              f"the stored segment-0 latent for the 'initial "
                              f"audio' reference ({exc}); falling back to the "
                              "merged latent's audio head")
                else:
                    print(f"[MMH3-TemporalExtend] WARNING: stored segment-0 "
                          f"latent not found ({path}) - 'initial audio' "
                          "falls back to the merged latent's audio head")
            if a is None and merged_a is not None:
                a = merged_a[..., :audio_range(0, int(segs[0]["new_frames"]))[1]]
            ref_cache["seg0_a"] = a.contiguous() if a is not None else None
            return ref_cache["seg0_a"]

        def _latent_audio_refs(seg, f_split=None):
            """Standalone bit-perfect audio references (segments >= 1; seg0
            never carries one).

            'initial' = seg0's ORIGINAL soundtrack from its stored segment
            latent (in-memory this run, session dir on resume) - NOT the
            merged latent's head, which later carried tails crossfade over.
            'prev' = everything from the previous segment's split point
            onward - its rendition on the merged timeline, anchored at
            round(split_frame * 5/3). The split frame comes from the session
            ledger after a restart, from the in-loop previous iteration
            otherwise, and falls back to THIS segment's carried-tail
            geometry (slightly earlier anchor) when neither exists.
            Returns a list of [1,32,2,T] audio latent slices."""
            arm = seg.get("ref_audio_mode") or "none"
            if arm not in ("prev", "initial") or merged_a is None:
                return []
            ta = merged_a.shape[-1]
            if arm == "initial":
                r = _seg0_audio_ref()
                return [r] if r is not None else []
            base = prev_split_frame if prev_split_frame is not None \
                else f_split
            if base is None:
                tail = int(seg.get("overlap_frames")
                           or glob.get("tail_frames", 39))
                base = snap_split_frame(
                    frames_for_tokens(merged_v.shape[2]), tail)[1]
            a0 = max(0, min(round(int(base) * FRAME_RESCALE), ta))
            if ta - a0 < 2:   # a useless sliver - treat as no reference
                return []
            return [merged_a[..., a0:ta].contiguous()]

        # ── Phase A: pre-build every segment's CLIP/VAE conditioning that
        # does not depend on OTHER segments' SAMPLING OUTPUTS, so a
        # multi-segment run walks the text encoder / VAE once up front
        # instead of ping-ponging between the encoder and the DiT mid-run
        # (model thrash on limited VRAM). Not pre-buildable: segments
        # referencing the previous merged video's frames ('prev_frame'
        # reference) and - beyond the first sampled segment - the
        # 'prev_tail' seam reference (the tail rides on the accumulated
        # latent). Anchoring / keyframe stripping stay in the loop: pure
        # latent ops, no encoder models involved. The chain's geometry is
        # replayed here in token space (the exact integer arithmetic the
        # sampling loop will perform), so `total` per segment is known
        # without the accumulated tensors. ──
        def _seg_ep(seg):
            ep = dict(glob)
            if seg.get("overlap_frames") is not None:
                ep["tail_frames"] = snap_17n5(int(seg["overlap_frames"]))
            elif isinstance(seg.get("extend_params"), dict) and seg["extend_params"]:
                ep.update(seg["extend_params"])
            return ep

        pre_cond = {}
        # replay the chain's frame math in TOKEN space - the exact integer
        # arithmetic _prepare_continuation + temporal_stitch perform in-loop
        # (frames_for_tokens(tokens_for_frames(f)) overshoots for frame
        # counts off the 17n+5 grid, so tracking FRAMES would drift; tokens
        # are exact for any latent, including external ones)
        tv = (merged_v.shape[2] if merged_v is not None else None)
        for i in range(resume, len(segs)):
            seg = segs[i]
            ep = _seg_ep(seg)
            s_seam_ref = ep.get("seam_reference", "none")
            new_frames = int(seg["new_frames"])
            if i == 0 and merged_v is None:
                prompt = (base_prompt + "\n" + seg["prompt"]).strip()
                pre_cond[i] = _build_first_conditioning(
                    clip, vae, audio_vae,
                    {**seg, "prompt": prompt,
                     "negative": (base_negative + "\n" + seg["negative"]).strip()},
                    W, H, new_frames, sock_pool=sock_pool, cache=cache)
                tv = tokens_for_frames(new_frames)
                continue
            fc = frames_for_tokens(tv)
            k_split, f_split = snap_split_frame(fc, int(ep.get("tail_frames", 39)))
            total = (fc - f_split) + new_frames
            tv_next = max(tv, k_split + tokens_for_frames(total))
            if i > resume and (seg.get("ref_source") == "prev_frame"
                               or s_seam_ref == "prev_tail"
                               or seg.get("ref_audio_mode")
                               in ("prev", "initial")):
                # latent audio references for segments beyond the first
                # sampled one ride on merged_a AFTER that segment was
                # sampled - built in-loop like the other accumulated-latent
                # references
                tv = tv_next        # built in-loop: needs the accumulated latent
                continue
            seam_ref = None
            if s_seam_ref == "prev_tail" and i == resume:
                # first sampled segment: its merged latent comes from disk /
                # the latent input, so the seam tail is available right now
                seam_ref = _build_seam_video_ref(
                    vae, merged_v, k_split, int(ep.get("seam_ref_frames", 0)),
                    prev_audio=merged_a)
            pre_cond[i] = _build_continue_conditioning(
                clip, vae, audio_vae,
                {**seg, "prompt": (base_prompt + "\n" + seg["prompt"]).strip(),
                 "negative": (base_negative + "\n" + seg["negative"]).strip()},
                W, H, total, (merged_v if i == resume else None), k_split,
                float(ep.get("anchor_strength", 0.999)), seam_ref=seam_ref,
                latent_audio_refs=_latent_audio_refs(seg),
                sock_pool=sock_pool, cache=cache)
            tv = tv_next
        if pre_cond:
            print(f"[MMH3-TemporalExtend] pre-encoded conditioning for "
                  f"{len(pre_cond)}/{len(segs) - resume} sampled segment(s) "
                  "(single encoder pass before sampling)")

        # ── background music ('bgm' audio reference mode) ──
        # Every 'bgm' segment gets the slice covering its ABSOLUTE span on the
        # chain's timeline - the preceding segments' frame counts decide the
        # offset - and that slice is pinned by an all-zero audio mask (see
        # _prepare_continuation). Abutting slices therefore reassemble the music
        # itself: the stitched chain's soundtrack is the BGM's head, not a
        # rendition of it. Each slice is encoded ON DEMAND, the moment its
        # segment comes up: only that span is ever resident in the VAE, so the
        # peak scales with one segment instead of with the whole track (a
        # song-length encode is gigabytes). Slices are content-addressed like
        # every other reference latent, so re-runs and resumes re-use them.
        bgm_wave = None
        bgm_audio = None
        bgm_src = None
        if any(s.get("ref_audio_mode") == "bgm" for s in segs[resume:]):
            bgm_audio = cfg.get("bgm_audio")
            if bgm_audio is None:
                raise ValueError(
                    "audio reference mode 'bgm' reads the song from the "
                    "'audio_BGM' input of the MMH3 Temporal Tile Editor, which "
                    "is not connected")
            if audio_vae is None:
                raise ValueError(
                    "the 'bgm' audio reference mode needs a VAE: the music "
                    "becomes the segment's soundtrack by being encoded with the "
                    "audio VAE and frozen by the audio noise-mask. Connect the "
                    "'audio_vae' input, or switch that segment's audio "
                    "reference mode back to 'none'.")
            # the track is only RESAMPLED up front (cheap, linear in length, and
            # it keeps the slices on exact latent-frame boundaries) - the encode
            # itself happens one segment at a time in _bgm_slice below
            bgm_wave = _bgm_vae_waveform(audio_vae, bgm_audio)
            bgm_src = _tensor_source_id(bgm_audio["waveform"])
            if tv is not None:
                # a BGM shorter than the chain is NOT an error: the music
                # covers the head of the chain and the model keeps going on
                # its own from where the music ran out (its conditioning has
                # heard the music up to that frame, so the free tail sounds
                # like a continuation, not a hard cut). One info line tells
                # the user exactly where the hand-off happens - everything
                # else falls out of the partial-freeze noise-mask below.
                have = _bgm_total_frames(audio_vae, bgm_wave)
                need = round(frames_for_tokens(tv) * FRAME_RESCALE)
                if have < need:
                    fps = float(cfg.get("fps") or 24.0)
                    print(f"[MMH3-TemporalExtend] BGM: only "
                          f"{have / (fps * FRAME_RESCALE):.1f} s of music for "
                          f"a {need / (fps * FRAME_RESCALE):.1f} s chain - "
                          f"the last "
                          f"{(need - have) / (fps * FRAME_RESCALE):.1f} s "
                          f"will be generated freely (model continues the "
                          f"music).")
            print(f"[MMH3-TemporalExtend] BGM: "
                  f"{_bgm_total_frames(audio_vae, bgm_wave)} audio frames "
                  f"({len([s for s in segs[resume:] if s.get('ref_audio_mode') == 'bgm'])}"
                  " segment(s) take their soundtrack from it, one slice at a "
                  "time)")

        def _bgm_slice(a0, n):
            """This segment's [a0, a0 + n) slice of the music, encoded on the
            spot (and cached by content). None when there is nothing left to
            encode (a0 already past the music's end, or n == 0); the slice
            itself is silently truncated when the music ends inside [a0, a0+n)
            and the caller turns that into a free tail via the noise-mask."""
            if bgm_wave is None:
                return None
            if n <= 0 or a0 >= _bgm_total_frames(audio_vae, bgm_wave):
                return None
            return cache.ref(
                "bgm", bgm_src,
                (int(bgm_audio.get("sample_rate") or 0), int(a0), int(n)),
                lambda: _encode_bgm_slice(audio_vae, bgm_wave, a0, n))

        last_seg_v = last_seg_a = None
        for i, seg in enumerate(segs):
            if i < resume:
                continue
            seed = int(seg.get("seed", 0))
            prompt = (base_prompt + "\n" + seg["prompt"]).strip()
            info = {"index": i, "mode": seg.get("mode"),
                    "ref_source": seg.get("ref_source"),
                    "seed": seed}

            # per-segment continuation parameters: the chain-wide glob
            # (Overlap Params sub-node / editor globals), with the segment's
            # own overlap_frames (carried-tail length) taking priority.
            # Legacy per-seg extend_params dicts still override too.
            ep = dict(glob)
            if seg.get("overlap_frames") is not None:
                ep["tail_frames"] = snap_17n5(int(seg["overlap_frames"]))
            elif isinstance(seg.get("extend_params"), dict) and seg["extend_params"]:
                ep.update(seg["extend_params"])
            s_tail = int(ep.get("tail_frames", 39))
            s_tail_mode = ep.get("tail_mode", "freeze_fade")
            s_fade_frames = int(ep.get("fade_frames", 0))
            s_fade_value = float(ep.get("fade_value", 0.5))
            s_fade_mode = ep.get("fade_mode", "flat")
            s_fade_impl = ep.get("fade_impl", "mask")
            s_init_cw = float(ep.get("init_content_weight", 1.0))
            s_max_msk = float(ep.get("max_mask_strength", 1.0))
            s_kf_mode = ep.get("keyframes_mode", "reanchor")
            s_anchor_seam = bool(ep.get("anchor_seam", True))
            s_anchor_strength = float(ep.get("anchor_strength", 0.999))
            s_overlap_mode = ep.get("overlap_mode", "later")
            s_overlap_blend = ep.get("overlap_blend", "linear")
            s_seam_ref = ep.get("seam_reference", "none")
            s_seam_ref_frames = int(ep.get("seam_ref_frames", 0))

            # this segment's whole soundtrack comes out of the BGM socket
            # ('bgm' audio reference mode); bgm_wave is None when no sampled
            # segment asks for it. The music slice - and therefore the encode -
            # is only produced here, for this segment (see _bgm_slice).
            seg_bgm = (bgm_wave is not None
                       and seg.get("ref_audio_mode") == "bgm")

            pre = pre_cond.pop(i, None)   # Phase A conditioning, if hoistable

            if i == 0 and merged_v is None:
                # ── fresh generation ──
                frames = int(seg["new_frames"])           # 17n + 5
                if pre is not None:
                    pos, neg = pre          # Phase A already encoded this
                else:
                    pos, neg = _build_first_conditioning(
                        clip, vae, audio_vae, {**seg, "prompt": prompt,
                                    "negative": (base_negative + "\n" + seg["negative"]).strip()},
                        W, H, frames, sock_pool=sock_pool, cache=cache)
                piece, frame_count = _empty_piece(W, H, frames)
                if seg_bgm:
                    # a fresh segment starts at the chain's origin, so its
                    # slice is the head of the music; the piece's own audio
                    # length is what the mask has to cover
                    bgm_slice = _bgm_slice(
                        0, piece["samples"].tensors[1].shape[-1])
                    if bgm_slice is not None:
                        _freeze_bgm_fresh(piece, bgm_slice)
                print(f"[MMH3-TemporalExtend] segment 0: fresh {frames} frames "
                      f"({seg.get('mode')}, seed {seed}"
                      f"{', BGM soundtrack' if seg_bgm else ''})")
            else:
                # ── continuation: split the accumulated tail ──
                new_frames = int(seg["new_frames"])       # multiple of 17
                if new_frames % 17 != 0 or new_frames <= 0:
                    raise ValueError(
                        f"segment {i}: new_frames must be a positive multiple "
                        f"of 17; got {new_frames}")
                bgm_slice = None
                if seg_bgm:
                    # the slice spans exactly what the new piece's audio covers:
                    # from this segment's split point (absolute frame f_split)
                    # through the piece's end (absolute frame total) - the same
                    # integers _prepare_continuation recomputes internally
                    _fc = frames_for_tokens(merged_v.shape[2])
                    _k, _f = snap_split_frame(_fc, s_tail)
                    bgm_slice = _bgm_slice(
                        round(_f * FRAME_RESCALE),
                        round(((_fc - _f) + new_frames) * FRAME_RESCALE))
                piece, (k_split, f_split, tail_real, total, frozen_v,
                        frozen_a) = _prepare_continuation(
                    merged_v, merged_a, s_tail, new_frames,
                    s_tail_mode, s_fade_frames, s_fade_value, s_fade_mode,
                    fade_impl=s_fade_impl, init_content_weight=s_init_cw,
                    max_mask_strength=s_max_msk,
                    bgm_slice=bgm_slice)
                frame_count = total
                la_refs = _latent_audio_refs(seg, f_split)
                # anchor for the NEXT segment's 'previous audio' reference:
                # this segment's rendition starts at its split point on the
                # merged timeline (fresh segment 0 sets 0 in its branch)
                prev_split_frame = f_split
                if pre is not None:
                    pos, neg = pre          # Phase A already encoded this
                else:
                    ptxt = (base_prompt + "\n" + seg["prompt"]).strip()
                    ntxt = (base_negative + "\n" + seg["negative"]).strip()
                    seam_ref = None
                    if s_seam_ref == "prev_tail":
                        if vae is None:
                            print("[MMH3-TemporalExtend] WARNING: "
                                  "seam_reference='prev_tail' needs a VAE - ignored")
                        else:
                            seam_ref = _build_seam_video_ref(
                                vae, merged_v, k_split, s_seam_ref_frames,
                                prev_audio=merged_a)
                    pos, neg = _build_continue_conditioning(
                        clip, vae, audio_vae, {**seg, "prompt": ptxt, "negative": ntxt},
                        W, H, frame_count, merged_v, k_split, s_anchor_strength,
                        seam_ref=seam_ref, latent_audio_refs=la_refs,
                        sock_pool=sock_pool, cache=cache)
                if s_kf_mode == "drop":
                    pos, neg = strip_keyframes(pos), strip_keyframes(neg)
                if s_anchor_seam:
                    pos = anchor_conditioning(pos, merged_v, k_split,
                                              s_anchor_strength)
                info.update({"split_frame": f_split, "tail_frames": tail_real,
                             "total_frames": total})
                print(f"[MMH3-TemporalExtend] segment {i}: split at frame "
                      f"{f_split} (token {k_split}) | tail {tail_real} f | "
                      f"+{new_frames} f -> {total} f (seed {seed})")

            seg_samples = run_two_stage(piece, pos, neg, seed, f"segment {i}:")
            seg_v, seg_a = seg_samples.tensors[0], seg_samples.tensors[1]
            # discard the stashed qsample_init band coordinates (kept only for
            # the removed brightness-match feature; popped to keep the piece
            # dict from leaking into the LOW stage)
            piece.pop("_fade_band", None)

            # Second (refinement) pass, continuation segments only: re-denoise
            # the WHOLE segment latent (carried tail + new content, video AND
            # audio together) with an independent low-denoise schedule - no
            # video noise mask, no keyframes; references (images, seam
            # reference) stay active. Mirrors the spatial tile's unmasked
            # second pass: the refreshed content meets the untouched history
            # in the stitch's linear cross-fade over the overlap band. In
            # 'bgm' mode the audio keeps its all-zero mask so the frozen music
            # is not re-developed (see the seg_bgm branch below).
            if second_sample_params is not None and i > 0:
                piece2 = {"samples": comfy.nested_tensor.NestedTensor(
                    (seg_v, seg_a))}
                if seg_bgm:
                    # 'bgm' mode: pin the second pass's audio to the same
                    # frozen BGM slice the first pass used - the audio mask is
                    # all zero, so the music is never re-developed. The video
                    # mask stays all ones (the whole video re-samples, exactly
                    # as in the unmasked second pass).
                    piece2["noise_mask"] = comfy.nested_tensor.NestedTensor(
                        (torch.ones((1, 1, seg_v.shape[2], seg_v.shape[3],
                                     seg_v.shape[4]), dtype=torch.float32),
                         torch.zeros((1, 1, 1, seg_a.shape[-1]),
                                     dtype=torch.float32)))
                seg2 = run_two_stage(piece2, strip_keyframes(pos),
                                     strip_keyframes(neg), seed,
                                     f"segment {i} 2nd pass:",
                                     params=second_sample_params)
                seg_v2, seg_a2 = seg2.tensors[0], seg2.tensors[1]
                seg_v = seg_v2
                if second_pass_audio == "second" and not seg_bgm:
                    seg_a = seg_a2
                print(f"[MMH3-TemporalExtend] segment {i}: 2nd pass applied "
                      f"(stitched audio: "
                      f"{'BGM - frozen by mask' if seg_bgm else second_pass_audio + ' sample'})")

            if i == 0 and merged_v is None:
                merged_v, merged_a = seg_v, seg_a
                # fresh segment 0: its rendition owns the whole timeline
                prev_split_frame = 0
            else:
                a_split = round(f_split * FRAME_RESCALE)
                merged_v, merged_a = temporal_stitch(
                    merged_v, merged_a, seg_v, seg_a,
                    k_split, a_split, s_overlap_mode, s_overlap_blend,
                    blend_len_v=frozen_v, blend_len_a=frozen_a)

            info["merged_frames"] = merged_v.shape[2]
            seg_info["segments"].append(info)
            save_segment(i, seg_v, seg_a, merged_v, merged_a, info)
            last_seg_v, last_seg_a = seg_v, seg_a
            if i == 0:
                seg0_audio = seg_a   # 'initial audio' reference source

        if last_seg_v is None:
            # resume == len(segs): every segment is locked - nothing left to
            # sample. The merged output is the stored merged latent (the
            # completed chain); the segment output stays the LAST segment's
            # own standalone latent, reloaded from the session so it matches
            # what sampling that segment would have produced this run (see
            # _last_segment_latent) instead of duplicating the merged latent.
            if merged_v is None:
                raise ValueError(
                    f"resume_from_segment={resume} but no stored merged "
                    "latent found - run the chain once first")
            print(f"[MMH3-TemporalExtend] all segments locked "
                  f"(resume={resume}/{len(segs)}) - outputting the stored "
                  "merged latent without sampling")
            last_seg_v, last_seg_a = _last_segment_latent()

        out = {"samples": comfy.nested_tensor.NestedTensor((merged_v, merged_a))}
        seg_out = {"samples": comfy.nested_tensor.NestedTensor((last_seg_v, last_seg_a))}
        sampling = {
            "low_stage": bool(sample_params["low_active"]),
            "high_steps": int(sample_params["sigmas_high"].shape[-1]) - 1,
            "cfg_high": float(sample_params["cfg_high"]),
            "cfg_low": (float(sample_params["cfg_low"])
                        if sample_params["low_active"] else None),
        }
        seg_info["sampling"] = sampling
        seg_info["cache"] = {"dir": os.path.join(_temporal_root(
            cfg.get("storage", "temp")), CACHE_DIRNAME), **cache.stats}
        print(f"[MMH3-TemporalExtend] reference cache: {cache.summary()}")
        for err in cache.errors[:3]:
            print(f"[MMH3-TemporalExtend] WARNING: reference cache {err}")
        return io.NodeOutput(out, seg_out, seg_info)


def _empty_piece(w, h, frames):
    """Empty H3 AV latent for a fresh segment. Returns (piece, frame_count)."""
    from comfy_extras.nodes_minimax_h3 import _empty_av_latent
    piece, frame_count = _empty_av_latent(w, h, frames)
    return piece, frame_count


def _freeze_bgm_fresh(piece, bgm_slice):
    """Pin a FRESH segment's (segment 0's) soundtrack to the BGM's head.

    Same contract as _prepare_continuation's `bgm_slice`, but a fresh segment
    starts at the chain's origin, so its slice is the head of the music and
    starts at index 0. The video keeps an all-ones mask (sampled normally);
    the audio mask is 0 for the BGM-covered frames (frozen to the music) and
    1 for the rest (free - the model continues on its own). When the music
    ends inside the segment, the slice is silently truncated: only the head
    is pinned, the tail of the segment's audio is left at zeros and the
    model fills it freely."""
    v, a = piece["samples"].tensors[0], piece["samples"].tensors[1]
    n = max(0, min(a.shape[-1], bgm_slice.shape[-1]))
    if n > 0:
        a = a.clone()          # never write into the caller's latent
        a[:, :, :, :n] = bgm_slice[:, :, :, :n]
    prof_a = torch.ones(a.shape[-1], dtype=torch.float32)
    if n > 0:
        prof_a[:n] = 0.0
    piece["samples"] = comfy.nested_tensor.NestedTensor((v, a))
    piece["noise_mask"] = comfy.nested_tensor.NestedTensor(
        (torch.ones((1, 1, v.shape[2], v.shape[3], v.shape[4]),
                    dtype=torch.float32),
         prof_a.view(1, 1, 1, a.shape[-1])))


class _InitBiasNoise:
    """EXPERIMENTAL ('qsample_init' fade implementation, may be removed).

    Initial-noise generator that q_sample-blends the carried content into the
    video stream's initial noise using the fade profile. The content is
    STANDARDIZED per (channel, token) before blending:
    x_init = m * noise + sqrt(1 - m^2) * (x0 - mu) / sd
    so the initial state keeps an exactly N(0, 1) marginal - zero mean and
    unit variance - while carrying the content's structure in its spatial
    correlations. A raw blend m*noise + (1-m)*x0 violates both: the nonzero
    DC of the latent (correlated with luma) survives the trajectory on
    content-weighted rows (flow denoisers preserve low frequencies), which
    showed up as randomly bright frames; the variance deficit
    (m^2 + (1-m)^2*var(x0) < 1) compounds it. m = 1 on fresh tokens, 0 on
    the frozen side of the band. Every sampled row stays in the trained
    stream regime (the mask is strictly binary in this mode). Deliberately
    dumb: wraps the real noise generator and restores plain noise on any
    surprise."""

    def __init__(self, inner, profile, x0_video, content_sign=1.0):
        self.inner = inner
        self.seed = inner.seed
        self.profile = profile          # [tv_new] fade profile (1 = pure noise)
        self.x0_video = x0_video        # [1, C, tv_new, H, W] carried content
        # negative init_content_weight: flip the content term's sign only
        # (anti-correlated experiment); 0 keeps the plain continuation.
        self.content_sign = -1.0 if content_sign < 0 else 1.0

    def generate_noise(self, latent):
        n = self.inner.generate_noise(latent)
        try:
            if not (n.is_nested and self.x0_video is not None):
                return n
            v, a = n.unbind()
            v = v.clone()
            tv = min(v.shape[2], self.x0_video.shape[2],
                     self.profile.shape[0])
            m = self.profile[:tv].to(device=v.device, dtype=v.dtype)
            mb = m.view(1, 1, tv, 1, 1)
            x0 = self.x0_video[:, :, :tv].to(device=v.device, dtype=v.dtype)
            mu = x0.mean(dim=(3, 4), keepdim=True)
            sd = x0.std(dim=(3, 4), keepdim=True).clamp_min(1e-6)
            x0n = (x0 - mu) / sd
            mb2 = (1.0 - mb * mb).clamp_min(0.0).sqrt()
            blend = mb * v[:, :, :tv] + (self.content_sign * mb2) * x0n
            v[:, :, :tv] = torch.where(mb > 0, blend, x0)
            return comfy.nested_tensor.NestedTensor((v, a))
        except Exception as exc:
            print(f"[MMH3-TemporalExtend] WARNING: qsample_init noise bias "
                  f"failed ({exc}); falling back to plain noise")
            return n


def _prepare_continuation(video, audio, tail_frames, new_frames,
                          tail_mode, fade_frames, fade_value, fade_mode,
                          fade_impl="mask", init_content_weight=1.0,
                          max_mask_strength=1.0, bgm_slice=None):
    """Split the accumulated latent's tail as the new segment's head (frozen +
    fade noise mask) and zero-pad the extension. Returns
    (piece, (k_split, f_split, tail_real, total, frozen_v, frozen_a)).

    `bgm_slice` (the 'bgm' audio reference mode) is THIS segment's slice of the
    background music: [1, 32, 2, ta_new] (shorter only when the music runs out).
    The caller encodes it from the audio latent index the split point maps to -
    see _encode_bgm_slice - so a whole track is never encoded at once. When
    given, this segment's whole soundtrack is the music wherever the music
    still exists; the audio noise-mask freezes exactly that span (0) and
    leaves the rest free (1), so a chain that outlasts its music carries on
    with the model's own continuation instead of failing outright."""
    tv = video.shape[2]
    ta = audio.shape[-1]
    H, W = video.shape[3], video.shape[4]
    device, dtype = video.device, video.dtype
    frame_count = frames_for_tokens(tv)

    k_split, f_split = snap_split_frame(frame_count, tail_frames)
    tail_real = frame_count - f_split
    total = tail_real + new_frames
    tv_tail = tv - k_split
    a_src0, a_src1 = audio_range(f_split, frame_count)
    a_src1 = min(a_src1, ta)
    ta_head = a_src1 - a_src0
    tv_new = tokens_for_frames(total)
    ta_new = round(total * FRAME_RESCALE)
    if tv_new <= tv_tail:
        raise ValueError(f"new_frames={new_frames} produces no new video tokens")

    video_new = torch.zeros((1, video.shape[1], tv_new, H, W),
                            device=device, dtype=dtype)
    video_new[:, :, :tv_tail] = video[:, :, k_split:tv]
    audio_new = torch.zeros((1, audio.shape[1], audio.shape[2], ta_new),
                            device=device, dtype=dtype)
    audio_new[:, :, :, :ta_head] = audio[:, :, :, a_src0:a_src1]
    piece = {"samples": comfy.nested_tensor.NestedTensor((video_new, audio_new))}

    # freeze amounts are also read by the caller for the stitch's cross-fade
    # ramp; they stay 0 in every non-'freeze_fade' mode, where nothing is
    # frozen (the 'free_resample' tail mode re-samples the carried tail too)
    frozen_v = frozen_a = 0
    if tail_mode == "freeze_fade":
        if fade_frames > tail_real:
            raise ValueError(
                f"fade_frames={fade_frames} exceeds the realized tail length "
                f"({tail_real} frames); lower fade_frames or raise tail_frames")
        fv = tokens_for_frames(fade_frames) if fade_frames > 0 else 0
        frozen_v = max(tv_tail - fv, 0)
        # max_mask_strength scales the freeze + fade mask AS A WHOLE: 1 keeps
        # the original profile, 0 collapses it to all-ones (no frozen zone, no
        # fade band - the carried tail re-samples, equivalent to 'free
        # resample'), intermediate s values hold the frozen zone at mask
        # 1 - s and run the fade gradient from 1 - s up to 1. Applied to the
        # soft profile BEFORE the qsample_init block, so the binary mask there
        # releases the frozen zone for s < 1 while its init-noise profile
        # still carries the scaled (weaker) structure.
        s_msk = min(max(float(max_mask_strength), 0.0), 1.0)
        prof_v = torch.ones(tv_new, dtype=torch.float32)
        if frozen_v > 0:
            prof_v[:frozen_v] = 0.0
        if fv > 0 and frozen_v < tv_tail:
            band = (torch.full((min(fv, tv_tail - frozen_v),), float(fade_value))
                    if fade_mode == "flat"
                    else torch.linspace(0.0, 1.0, min(fv, tv_tail - frozen_v)))
            prof_v[frozen_v:tv_tail] = torch.minimum(prof_v[frozen_v:tv_tail], band)
        if s_msk < 1.0:
            prof_v = (1.0 - s_msk) + s_msk * prof_v
        if fade_impl == "qsample_init":
            # EXPERIMENTAL: strictly binary mask (frozen zone only, so every
            # sampled row stays in the trained stream regime); the soft
            # profile rides on the piece dict and is blended into the INITIAL
            # NOISE by the caller (_InitBiasNoise) instead. (The intermediate-
            # mask breakage that motivated this variant turned out to be an
            # upstream ComfyUI bug - denoise-mask velocity conversion, PR
            # #15988, fixed 2026-09 - not untrained per-row timesteps; this
            # remains as an alternative that carries structure through the
            # initial noise rather than the mask.)
            # init_content_weight caps the STRUCTURE carried by the init:
            # the profile m is the noise weight, so the content weight is
            # c = sqrt(1 - m^2); capping c at c * w gives
            # m_eff = sqrt(1 - w^2 * (1 - m^2)). A NEGATIVE w only flips the
            # sign of the content term (anti-correlated experiment) - the
            # noise weight depends on w^2, so |w| is what shapes the profile
            # and the sign travels to _InitBiasNoise separately. The
            # structured init at t = sigma_max is out-of-distribution and the
            # model over-develops it (severe brightness/saturation drift);
            # with the prev_tail seam reference carrying continuity, the init
            # can afford to be much noisier (w = 1 keeps the previous
            # full-strength behavior).
            w = min(max(float(init_content_weight), -1.0), 1.0)
            aw = abs(w)
            if aw < 1.0 and frozen_v < tv_tail:
                # IMPORTANT: transform the BAND ONLY - the frozen zone's
                # profile entry must stay 0 (pinned). An earlier version
                # transformed the whole profile, which mapped the frozen
                # zone's 0 to sqrt(1 - w^2) > 0 and silently released the
                # frozen zone after binarization.
                band_p = prof_v[frozen_v:tv_tail]
                prof_v[frozen_v:tv_tail] = torch.sqrt(
                    1.0 - (aw * aw) * (1.0 - band_p * band_p))
                print(f"[MMH3-TemporalExtend] qsample_init: init content "
                      f"weight set to {w:.2f} "
                      f"(min noise weight {float(prof_v[frozen_v]):.3f})")
            piece["_fade_init"] = (prof_v, -1.0 if w < 0 else 1.0)
            piece["_fade_band"] = (int(frozen_v), int(tv_tail))
            prof_v = (prof_v > 0.0).to(torch.float32)
        mv = prof_v.view(1, 1, tv_new, 1, 1).expand(1, 1, tv_new, H, W).contiguous()

        fa = audio_range(0, fade_frames)[1] if fade_frames > 0 else 0
        frozen_a = max(ta_head - fa, 0)
        prof_a = torch.ones(ta_new, dtype=torch.float32)
        if frozen_a > 0:
            prof_a[:frozen_a] = 0.0
        if fa > 0 and frozen_a < ta_head:
            band_a = (torch.full((min(fa, ta_head - frozen_a),), float(fade_value))
                      if fade_mode == "flat"
                      else torch.linspace(0.0, 1.0, min(fa, ta_head - frozen_a)))
            prof_a[frozen_a:ta_head] = torch.minimum(prof_a[frozen_a:ta_head], band_a)
        if s_msk < 1.0:
            prof_a = (1.0 - s_msk) + s_msk * prof_a
        ma = prof_a.view(1, 1, 1, ta_new)
        piece["noise_mask"] = comfy.nested_tensor.NestedTensor((mv, ma))

    if bgm_slice is not None:
        # 'bgm' audio mode: the segment's soundtrack IS the music where the
        # music still exists, and the model's own continuation where it has
        # already run out. The slice the caller encoded starts at this
        # segment's split point - exactly the index the stitch uses as
        # `a_split` - so abutting 'bgm' segments cut abutting music slices
        # and the stitched chain's audio is the music's head. The carried
        # tail's own copy is overwritten within the BGM span only: past the
        # music's end the tail keeps what the previous segment produced
        # (BGM ending mid-chain is the natural "continue" hand-off - the
        # model has been hearing the music up to that frame, so the free
        # tail sounds like a continuation rather than a hard cut).
        # fade_frames is still a VIDEO knob and is ignored on audio (an audio
        # fade band would re-render the music as a model rendition and then
        # cross-fade the two). audio_new is a fresh tensor built above, so
        # filling it in place cannot touch the accumulated latent.
        n = max(0, min(ta_new, bgm_slice.shape[-1]))
        if n > 0:
            audio_new[:, :, :, :n] = bgm_slice[:, :, :, :n]
        prof_a = torch.ones(ta_new, dtype=torch.float32)
        if n > 0:
            prof_a[:n] = 0.0
        # video mask: whatever the tail mode built (freeze_fade) or all-ones
        # (the tail is re-sampled); audio mask: 0 over the BGM span
        # (pinned to the music) and 1 over the free tail (the model's own).
        mv_bgm = (piece["noise_mask"].tensors[0] if "noise_mask" in piece
                  else torch.ones((1, 1, tv_new, H, W), dtype=torch.float32))
        piece["noise_mask"] = comfy.nested_tensor.NestedTensor(
            (mv_bgm, prof_a.view(1, 1, 1, ta_new)))

    return piece, (k_split, f_split, tail_real, total, int(frozen_v),
                   int(frozen_a))
