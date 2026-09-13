"""MMH3 offline latent preview — reusable decode API + experimental node.

This module's PRIMARY deliverable is the reusable function:
    decode_h3_video_frames(latent_source, approx_vae="", max_frames=0, max_resolution=0)
It turns an already-sampled H3 latent into preview video frames using the tiny H3
VAE (taeh3 in models/vae_approx) — far cheaper than the full H3 decoder. Other
nodes / future JS-embedded previews can call this directly; it accepts either a
file path (the "latent file location" interface) or an in-memory tensor.

`MMH3LatentPreview` is an EXPERIMENTAL wrapper node that prints the decoded
preview inline on the node (no IMAGE tensor output): it decodes a saved latent
file, encodes an animated WebP, and pushes it to the front-end via a WS message
which the companion JS (`web/mmh3_latent_preview.js`) renders on the node.
"""

import base64
import io as _py_io
import logging
import os

import torch

from comfy_api.latest import io

from .tiny_vae import load_tiny_vae_decoder

# Lambda caches the H3 decoder so repeated executions reuse it without a reload.
_DECODER_CACHE = {}


def _load_decoder(approx_vae, device=None):
    if not approx_vae:
        approx_vae = _default_approx_vae_name()
    key = (approx_vae, device)
    dec = _DECODER_CACHE.get(key)
    if dec is None:
        dec = load_tiny_vae_decoder(approx_vae, device=device)
        _DECODER_CACHE[key] = dec
    return dec


def _default_approx_vae_name():
    """Pick the first taeh3* file in models/vae_approx, else any entry, else None."""
    import folder_paths
    names = folder_paths.get_filename_list("vae_approx") or []
    for n in names:
        if n.lower().startswith("taeh3"):
            return n
    return names[0] if names else ""


def _coerce_latent_tensor(latent_source):
    """Given a tensor, a LatentDict ({"samples": ...}), a NestedTensor, or a
    saved-latent path string, return a [1, C, T, H, W] video-latent tensor.

    Handles both ComfyUI's standard format ({"samples": tensor}) and MiniMax
    H3's format (samples may be a NestedTensor with video+audio members, and
    saved .h3latent files store {tensor_count, latent_0, latent_1, ...})."""
    if isinstance(latent_source, dict):  # LATENT port: {"samples": tensor/NestedTensor}
        t = None
        for key in ("samples", "latent"):
            if key in latent_source:
                v = _unwrap_nested(latent_source[key])
                if isinstance(v, torch.Tensor):
                    t = v
                    break
        latent_source = t
    if isinstance(latent_source, str):
        t = _file_to_video_tensor(_load_latent_file(latent_source))
        latent_source = t
    latent_source = _unwrap_nested(latent_source)
    t = latent_source
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"latent_source must be a tensor, LATENT dict or path, got {type(t)!r}")
    # Normalize to [B, C, T, H, W]; bare [C, T, H, W] gets a batch dim.
    if t.ndim == 4:
        t = t.unsqueeze(0)
    elif t.ndim != 5:
        raise ValueError(f"expected 4D/5D H3 latent, got {t.ndim}D with shape {tuple(t.shape)}")
    return t.contiguous()


def _unwrap_nested(v):
    """If v is an H3 NestedTensor (video + audio members), keep the video one
    (first member). Plain tensors/dicts pass through unchanged."""
    if getattr(v, "is_nested", False):
        members = list(v.unbind())
        v = members[0] if members else None
    return v


def _file_to_video_tensor(sd):
    """Extract the H3 video latent from a saved file's contents.

    ComfyUI LoadLatent format: {"samples"/"latent": tensor (or NestedTensor)}.
    MiniMax H3 .h3latent format: {"tensor_count": tensor,
                                  "latent_0": video, "latent_1": audio, ...}.
    """
    if not isinstance(sd, dict):
        return _unwrap_nested(sd)
    if "tensor_count" in sd:  # MiniMax H3 AV latent
        count = int(sd["tensor_count"].item())
        if count < 1:
            raise ValueError(f"Invalid H3 latent tensor_count: {count}")
        return sd[f"latent_0"].float()
    t = None
    for key in ("samples", "latent", "z", "x"):
        if key in sd:
            v = _unwrap_nested(sd[key])
            if isinstance(v, torch.Tensor):
                t = v
                break
    return t


def _load_latent_file(path):
    """Resolve a latent file path (absolute or folder_paths-relative) and load
    its contents (a dict for safetensors, raw object for torch pickle)."""
    import folder_paths
    p = path
    if not os.path.isfile(p):
        p = folder_paths.get_annotated_filepath(path)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"latent file not found: {path}")
    ext = os.path.splitext(p)[1].lower()
    if ext == ".safetensors":
        import comfy.utils
        return comfy.utils.load_torch_file(p, safe_load=True)
    if ext == ".h3latent":  # MiniMax H3 AV latent (safetensors payload)
        import safetensors.torch
        return safetensors.torch.load_file(p, device="cpu")
    try:
        return torch.load(p, map_location="cpu")
    except TypeError:  # older torch without map_location in some paths
        return torch.load(p)


def _downsample(pil_img, max_resolution):
    if max_resolution and max(pil_img.size) > max_resolution:
        w, h = pil_img.size
        scale = max_resolution / float(max(w, h))
        pil_img = pil_img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    return pil_img


def decode_h3_video_frames(latent_source, approx_vae="", max_frames=0, max_resolution=0,
                           device=None, cpu_threads=0):
    """Decode a finished H3 latent to preview frames with the tiny H3 VAE.

    Args:
        latent_source: torch.Tensor [B,C,T,H,W] ([C,T,H,W] accepted) OR a file path
            (string) to a saved latent (.safetensors / .latent / .pt, LoadLatent-style).
        approx_vae: filename from models/vae_approx (defaults to a taeh3* file).
        max_frames: 0 = every decoded frame; else uniformly sample this many.
        max_resolution: longest-side pixel cap (0 = keep native).
        device: "cpu" / "cuda" to force the decode device, or None for ComfyUI default.
        cpu_threads: when decoding on CPU, threads to use (0 = all cores).

    Returns:
        dict { "frames": list[PIL.Image], "width": int, "height": int, "count": int }.
        Frames are 0-255 RGB PIL images. Raises on load/decode failure.
    """
    from PIL import Image

    dec = _load_decoder(approx_vae, device=device)
    if dec is None:
        raise RuntimeError(
            f"approximate H3 VAE '{approx_vae}' could not be loaded from vae_approx. "
            "Make sure models/vae_approx contains taeh3.safetensors."
        )

    import time
    t0 = time.perf_counter()

    latent = _coerce_latent_tensor(latent_source)
    if latent.shape[2] == 0:
        raise ValueError("latent has zero temporal length")

    # CPU benchmarking: let PyTorch actually use all cores. ComfyUI leaves thread
    # counts low, so a CPU decode otherwise runs near-single-threaded (~10% usage).
    if torch.device(dec.device).type == "cpu":
        _bump_cpu_threads(cpu_threads)

    # Sub-sample uniformly if requested; passing frame_indices handles the
    # temporal-MemBlock decode path correctly (prefix + subsample).
    if max_frames and max_frames > 0 and max_frames < latent.shape[2]:
        idx = torch.linspace(0, latent.shape[2] - 1, max_frames).round().long().tolist()
        frames_t = dec.decode_video(latent, frame_indices=idx)
        n_out = len(idx)
    else:
        frames_t = dec.decode_video(latent)
        n_out = frames_t.shape[0]

    # frames_t: [n, H, W, 3] float 0-1. Convert to 0-255 RGB PIL.
    frames = []
    for i in range(n_out):
        arr = frames_t[i]
        if arr.is_cuda:
            arr = arr.to("cpu", non_blocking=True)
        arr = arr.float().mul(255.0).clamp(0, 255).byte().numpy()
        img = Image.fromarray(arr, mode="RGB")
        if max_resolution and max_resolution > 0:
            img = _downsample(img, max_resolution)
        frames.append(img)

    elapsed = time.perf_counter() - t0
    return {
        "frames": frames,
        "width": frames[0].width if frames else 0,
        "height": frames[0].height if frames else 0,
        "count": len(frames),
        "elapsed": elapsed,
        "device": str(torch.device(dec.device)),
    }


def frames_to_animated_webp(frames, fps=12, quality=82):
    """Encode a list of PIL frames into a looping animated WebP and return bytes."""
    if not frames:
        return b""
    duration = max(1, int(1000.0 / max(1, fps)))
    buf = _py_io.BytesIO()
    frames[0].save(
        buf,
        format="WEBP",
        append_images=frames[1:],
        save_all=True,
        duration=duration,
        loop=0,
        quality=quality,
    )
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Experimental node: inline (on-node) preview, no IMAGE tensor output.
# ---------------------------------------------------------------------------

def _bump_cpu_threads(n=0):
    """Let PyTorch use all (or n) cores for the CPU decode. ComfyUI leaves the
    thread pool near-default, so a plain CPU run otherwise pegs <20%."""
    try:
        if n and n > 0:
            torch.set_num_threads(n)
            try:
                torch.set_num_interop_threads(n)
            except Exception:
                pass
        else:
            import multiprocessing
            count = multiprocessing.cpu_count()
            torch.set_num_threads(count)
            try:
                torch.set_num_interop_threads(count)
            except Exception:
                pass  # interop must be set before any parallel work started
    except Exception:
        pass


def _vae_approx_options():
    import folder_paths
    names = folder_paths.get_filename_list("vae_approx")
    return names or [""]


def _pick_default(name):
    if name.startswith("taeh3"):
        return name
    for n in _vae_approx_options():
        if n.startswith("taeh3"):
            return n
    return name or ""


class MMH3LatentPreview(io.ComfyNode):
    """EXPERIMENTAL: decode a finished H3 latent with the tiny H3 VAE and show the
    video inline on the node. No IMAGE tensor is emitted; the preview is pushed to
    the front-end JS (web/mmh3_latent_preview.js) over WebSocket."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        opts = _vae_approx_options()
        default = _pick_default(opts[0]) if opts else ""
        return io.Schema(
            node_id="MMH3LatentPreview",
            display_name="MMH3 Latent Preview (approx)",
            category="model/conditioning/minimax",
            description=(
                "EXPERIMENTAL. Decode an already-sampled H3 latent with the fast "
                "tiny H3 VAE (taeh3 in models/vae_approx) and show the video inline "
                "on this node. No IMAGE tensor is output. Points a saved latent "
                "file ('latent_file') + preview params at the reusable "
                "decode_h3_video_frames() / frames_to_animated_webp() API so other "
                "nodes / JS can call the same pipeline."
            ),
            search_aliases=["h3 preview", "h3 approx preview", "taeh3", "h3 fast preview"],
            is_experimental=True,
            inputs=[
                io.Latent.Input("latent", optional=True,
                                tooltip="In-memory H3 latent (e.g. from a Latent "
                                        "loader / VAE Encode / the Extend Video "
                                        "node). Takes priority over latent_file."),
                io.String.Input("latent_file", default="",
                                tooltip="Path or filename of a saved H3 latent "
                                        "(.safetensors / .latent). Resolved via "
                                        "folder_paths the same way as Load Latent. "
                                        "Used only when 'latent' is not connected."),
                io.Combo.Input("approx_vae", options=opts, default=default,
                               tooltip="Tiny H3 VAE decoder from models/vae_approx "
                                       "(e.g. taeh3.safetensors)."),
                io.Combo.Input("device", options=["auto", "cuda", "cpu"], default="auto",
                               tooltip="Decode device. 'auto' uses ComfyUI's default "
                                       "VAE device (CUDA if available); 'cuda'/'cpu' "
                                       "force that device for the preview (for "
                                       "benchmarking CPU speed)."),
                io.Int.Input("cpu_threads", default=0, min=0,
                             tooltip="CPU threads when device=cpu. 0 = all cores "
                                     "(8C16T -> 16). Ignored on cuda. Tune to "
                                     "compare CPU decode speed."),
                io.Int.Input("max_frames", default=0, min=0,
                             tooltip="0 = every decoded frame; else uniformly sample "
                                     "this many frames for the preview."),
                io.Int.Input("max_resolution", default=0, min=0,
                             tooltip="Longest-side pixel cap (0 = keep native). "
                                     "Shrinks the preview WebP."),
                io.Int.Input("fps", default=12, min=1,
                             tooltip="Preview playback frames-per-second for the "
                                     "animated WebP."),
            ],
            outputs=[
                io.String.Output("info",
                                 tooltip="Text summary of the decoded preview: "
                                         "frame count, dimensions, fps."),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, latent=None, latent_file="", approx_vae="", device="auto",
                max_frames=0, max_resolution=0, fps=12, cpu_threads=0) -> io.NodeOutput:
        node_id = getattr(cls.hidden, "unique_id", None)
        if latent is not None:
            source = latent  # LATENT dict; _coerce_latent_tensor extracts "samples"
        else:
            source = latent_file
        dev = None if device == "auto" else device
        try:
            result = decode_h3_video_frames(
                source, approx_vae=approx_vae, device=dev,
                max_frames=max_frames, max_resolution=max_resolution,
                cpu_threads=cpu_threads,
            )
            webp = frames_to_animated_webp(result["frames"], fps=fps or 12)
            payload = {
                "node_id": node_id,
                "webp": base64.b64encode(webp).decode("ascii"),
                "mime": "image/webp",
                "fps": fps or 12,
                "width": result["width"],
                "height": result["height"],
                "count": result["count"],
                "error": None,
            }
            _send(node_id, payload)
            info = (f'{result["count"]} frames @ {result["width"]}x{result["height"]} '
                    f'({fps or 12} fps) on {result["device"]}, '
                    f'{result["elapsed"]:.2f}s')
        except Exception as exc:
            import traceback
            logging.warning("[MMH3LatentPreview] preview failed (see full traceback "
                            "below); latent type=%s", type(source).__name__)
            logging.warning("".join(traceback.format_exc()))
            _send(node_id, {"node_id": node_id, "count": 0,
                            "error": f"{type(exc).__name__}: {exc}"})
            info = f"preview failed [{type(exc).__name__}]: {exc}"
        return io.NodeOutput(info)


def _send(node_id, payload):
    try:
        from server import PromptServer
        PromptServer.instance.send_sync(
            "mmh3_latent_preview", payload, PromptServer.instance.client_id)
    except Exception as exc:
        logging.warning("[MMH3LatentPreview] send_sync failed: %s", exc)