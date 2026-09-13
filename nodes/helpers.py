"""Shared helpers for the MMH3 Extend Video node.

Self-contained subset of the MMH3 Ultimate Upscale nodes.py that the Extend
Video tile sampler needs: H3 VAE constants, inpainting/mask/blend helpers,
conditioning crops and the sampling glue. Kept separate so the Extend Video
package has no dependency on the upscale/LTX code.
"""

import torch

import comfy.ldm.common_dit
import comfy.model_management
import comfy.nested_tensor
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
from comfy_api.latest import io

H3_INPAINT_PARAM = io.Custom("H3_INPAINT_PARAM")

# Spatial compression factor of the Minimax H3 3D VAE (16x).
VAE_DOWNSAMPLE = 16


def is_h3_av_latent(samples):
    return (samples is not None and samples.is_nested and len(samples.tensors) == 2
            and samples.tensors[0].ndim == 5 and samples.tensors[0].shape[1] == 24
            and samples.tensors[1].ndim == 4 and samples.tensors[1].shape[1] == 32)


def blend_weights(t, overlap_blend, overlap_mode):
    """Weight given to the NEW tile's content across an overlap band.

    t runs 0..1 from the done-seam toward the tile interior. overlap_mode 'later'
    hands the band to the new tile; 'earlier' to the accumulated content.
    overlap_blend selects the transition shape."""
    if overlap_blend == "overwrite":
        return torch.ones_like(t) if overlap_mode == "later" else torch.zeros_like(t)
    if overlap_blend == "midpoint":
        step = (t >= 0.5).to(t.dtype)
    elif overlap_blend == "smoothstep":
        step = t * t * (3.0 - 2.0 * t)
    else:
        step = t
    if overlap_mode == "earlier":
        return step
    return 1.0 - step


def crop_keyframes_to_tile(cond, src_h, src_w, r0, c0, tr, tc):
    """Spatially crop every keyframe's video latent to a tile of the source frame.

    Keyframes whose latent already matches the source spatial size are cropped to
    the tile's latent region. If a keyframe is at a DIFFERENT spatial scale (e.g.
    the source was latent-upscaled before tiling, or a different VAE/resolution
    produced the conditioning), it is resized to the source spatial size first so
    the cropped keyframe exactly matches the tile's row count - otherwise the
    model's cond/video row broadcast fails with a shape mismatch."""
    out = []
    for tensor, d in cond:
        nd = dict(d)
        kfs = nd.get("minimax_keyframes")
        if kfs:
            cropped = []
            for kf in kfs:
                nkf = dict(kf)
                lt = kf.get("latent")
                if lt is not None:
                    kh, kw = lt.shape[3], lt.shape[4]
                    if kh == src_h and kw == src_w:
                        crop = lt[:, :, :, r0:r0 + tr, c0:c0 + tc].contiguous()
                        nkf["latent"] = comfy.ldm.common_dit.pad_to_patch_size(crop, (1, 2, 2))
                    else:
                        B, C, T, H, W = lt.shape
                        lt_r = torch.nn.functional.interpolate(
                            lt.to(torch.float32).reshape(B * T, C, H, W),
                            size=(src_h, src_w), mode="bilinear", align_corners=False,
                        ).reshape(B, C, T, src_h, src_w)
                        nkf["latent"] = comfy.ldm.common_dit.pad_to_patch_size(
                            lt_r[:, :, :, r0:r0 + tr, c0:c0 + tc].contiguous(), (1, 2, 2))
                    cropped.append(nkf)
                else:
                    cropped.append(nkf)
            nd["minimax_keyframes"] = cropped
        out.append([tensor, nd])
    return out


def normalize_minimax_refs(cond):
    """Make minimax_refs blocks SELF-CONSISTENT for the H3 packed layout.

    The packed layout reserves frozen rows from each block's metadata and delivers
    cond rows from blocks that carry a latent, sized by its real shape. If the
    metadata disagrees with the latent (or a visual block lacks a latent), the two
    paths diverge and sampling crashes with a shape mismatch. Drop visual blocks
    without a latent and rewrite the metadata from the real latent shape."""
    out = []
    for tensor, d in cond:
        nd = dict(d)
        refs = nd.get("minimax_refs")
        if refs:
            fixed = []
            for blk in refs:
                nblk = dict(blk)
                lt = nblk.get("latent")
                if lt is None:
                    if nblk.get("kind") in ("image", "video", "video_audio"):
                        continue
                    fixed.append(nblk)
                    continue
                nblk["latent_h"] = int(lt.shape[3])
                nblk["latent_w"] = int(lt.shape[4])
                if nblk.get("kind") in ("video", "video_audio"):
                    nblk["latent_t"] = int(lt.shape[2])
                fixed.append(nblk)
            nd["minimax_refs"] = fixed
        out.append([tensor, nd])
    return out


def build_guider(model, cond, negative, cfg):
    guider = comfy.samplers.CFGGuider(model)
    if negative is not None:
        guider.set_conds(cond, negative)
        guider.set_cfg(cfg)
    else:
        guider.inner_set_conds({"positive": cond})
    return guider


def sample_piece(piece, cond, model, noise, sampler, sigmas, negative, cfg, x0_output=None,
                 noise_override=None):
    """Sample one tile. Mirrors SamplerCustomAdvanced, including the x0 preview
    callback. Returns nested samples (video+audio).

    `noise_override` (optional, EXPERIMENTAL - used by the temporal extend
    node's 'qsample_init' fade implementation) replaces the generated initial
    noise tensor; `noise` is still consulted for the seed.

    When `x0_output` (a dict) is given, it is filled with "x0_latent": the
    model's final-step denoised prediction in latent space (SamplerCustom's
    denoised_output). Unlike the trajectory samples - which at a nonzero
    sigma still carry that sigma's noise - the x0 prediction's noise-mask
    frozen zones hold the PRISTINE latent_image content (the mask's per-step
    output blend pins them there), making it the correct hand-off state for
    a masked continuation stage."""
    latent = dict(piece)
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(
        model, latent_image,
        latent.get("downscale_ratio_spacial", None),
        latent.get("downscale_ratio_temporal", None),
    )
    latent["samples"] = latent_image
    noise_mask = latent.get("noise_mask")

    guider = build_guider(model, cond, negative, cfg)
    # Use the caller's dict when one was passed (so the caller can read the
    # final x0 back); only create a local dict when nobody asked for it.
    if x0_output is None:
        x0_output = {}
    callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = guider.sample(
        noise.generate_noise(latent) if noise_override is None else noise_override,
        latent_image, sampler, sigmas,
        denoise_mask=noise_mask, callback=callback,
        disable_pbar=disable_pbar, seed=noise.seed,
    )
    samples = samples.to(comfy.model_management.intermediate_device())
    if x0_output is not None and "x0" in x0_output:
        # mirror SamplerCustom's denoised_output (nodes_custom_sampler.py)
        x0 = x0_output["x0"]
        if samples.is_nested and not x0.is_nested:
            latent_shapes = [x.shape for x in samples.unbind()]
            x0 = comfy.nested_tensor.NestedTensor(comfy.utils.unpack_latents(x0, latent_shapes))
        x0_output["x0_latent"] = model.model.process_latent_out(x0.cpu())
    return samples


def resolve_control_range(d, sigmas, sampling):
    """Resolve the active denoising (start, end) percent range from a start/end
    set. In 'percent' mode the pair is used as-is. In 'step' mode the steps map to
    the actual `sigmas` schedule as a half-open [start_step, end_step) range."""
    if d["start_end_set"] == "percent":
        return (d["start_percent"], d["end_percent"])
    n = max(int(sigmas.shape[-1]) - 1, 1)
    start_step = max(int(d["start_step"]), 0)
    end_step = max(int(d["end_step"]), start_step + 1)
    c0 = min(start_step, n)
    c1 = min(end_step, n)

    if c0 > 0:
        top = (float(sigmas[c0]) + float(sigmas[c0 - 1])) / 2.0
    else:
        top = (float(sigmas[0]) + 1.0) / 2.0
    last = max(c1 - 1, 0)
    if c1 < n:
        bottom = (float(sigmas[last]) + float(sigmas[c1])) / 2.0
    else:
        bottom = float(sigmas[last]) / 2.0

    alpha = float(getattr(sampling, "shift", 1.0))
    if alpha == 1.0:
        start_p, end_p = 1.0 - top, 1.0 - bottom
    else:
        start_p = 1.0 - top / (alpha - (alpha - 1.0) * top)
        end_p = 1.0 - bottom / (alpha - (alpha - 1.0) * bottom)
    return (min(start_p, end_p), max(start_p, end_p))


def apply_control(cond, c_net):
    """Set the conditioning's control field to a ready controlnet copy."""
    out = []
    for tensor, d in cond:
        nd = dict(d)
        nd["control"] = c_net
        nd["control_apply_to_uncond"] = True
        out.append([tensor, nd])
    return out
