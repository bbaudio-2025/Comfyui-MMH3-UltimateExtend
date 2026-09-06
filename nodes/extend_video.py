"""MMH3 Spatial Extend Video node.

Generates N MiniMax H3 AV tiles from a tile_config (produced by the companion
"MMH3 Spatial Tile Editor" node) and merges them into one wider/taller latent. Each tile
gets its own conditioning created internally from prompts and reference images.
"""

import torch

import comfy.controlnet
import comfy.nested_tensor

from comfy_api.latest import io

from .helpers import (
    VAE_DOWNSAMPLE,
    H3_INPAINT_PARAM,
    apply_control,
    blend_weights,
    crop_keyframes_to_tile,
    is_h3_av_latent,
    normalize_minimax_refs,
    resolve_control_range,
    sample_piece,
)


def _edge_fade_mask(th, tw, ol_h, ol_w, fade_h, fade_w,
                    done_top, done_bottom, done_left, done_right,
                    frozen_w=None, frozen_h=None, flat_val=None):
    """Per-tile video noise mask [th, tw]: 1 = sample freely, 0 = frozen.

    Each overlap strip shared with an already-sampled neighbor splits into a
    FROZEN segment on the seam side (mask 0, keeps the neighbour's content) and
    a FADE segment (mask 0 -> 1) toward the tile interior; 0 fade = whole strip
    frozen. Width and height axes use independent overlap and fade lengths, and
    every edge (top/bottom/left/right) is handled. frozen_w/frozen_h give each
    axis an explicit fully-frozen width instead of the implicit ol - fade split.

    flat_val: when set, the fade band is a spatially UNIFORM constant instead
    of a linspace ramp. The invariant that matters in H3 is per-token
    LABEL-INPUT CONSISTENCY: the mask value becomes each 2x2-pooled token's
    timestep label AND mixes that token's input (m*noise + (1-m)*anchor ==
    the anchor noised to level m*sigma), so any mode keeping label and input
    equal per token reads as "partially preserved content" the model extends
    from the anchor. A uniform band is consistent even at the patch level (a
    constant pools to itself); a ramp is consistent per pixel but its pooled
    label takes each patch's max (a mild sub-patch mismatch) - empirically
    both work, flat slightly ahead. What does NOT work is pairing a binary
    label with a ramped input (fade_mode 'hybrid'): clean anchor content
    labelled as full noise is read as noise structure (strange/disconnected
    content). The ORIGINAL mosaic came from per-step fade_steps rebuilds
    desyncing the once-baked labels, not from a static ramp. Uniform m also
    degenerates cleanly: flat_val=0 is a fully frozen band, flat_val=1
    disables the mask."""
    mask = torch.ones(th, tw, dtype=torch.float32)

    def band(f):
        if flat_val is not None:
            return torch.full((f,), float(flat_val))
        return torch.linspace(0.0, 1.0, f)

    def split(ol, fade, frozen):
        if frozen is None:
            f = min(fade, ol)
            return ol - f, f
        fz = min(frozen, ol)
        return fz, min(int(fade), ol - fz)

    if done_left and ol_w > 0:
        fz, f = split(ol_w, fade_w, frozen_w)
        if f == 0:
            mask[:, :fz] = 0.0
        else:
            mask[:, :fz] = 0.0
            mask[:, fz:fz + f] = torch.minimum(mask[:, fz:fz + f], band(f)[None, :])
    if done_right and ol_w > 0:
        fz, f = split(ol_w, fade_w, frozen_w)
        if f == 0:
            mask[:, tw - fz:] = 0.0
        else:
            b = band(f)
            mask[:, tw - fz:] = 0.0
            mask[:, tw - fz - f:tw - fz] = torch.minimum(
                mask[:, tw - fz - f:tw - fz], (1.0 - b)[None, :])
    if done_top and ol_h > 0:
        fz, f = split(ol_h, fade_h, frozen_h)
        if f == 0:
            mask[:fz, :] = 0.0
        else:
            mask[:fz, :] = 0.0
            mask[fz:fz + f, :] = torch.minimum(mask[fz:fz + f, :], band(f)[:, None])
    if done_bottom and ol_h > 0:
        fz, f = split(ol_h, fade_h, frozen_h)
        if f == 0:
            mask[th - fz:, :] = 0.0
        else:
            b = band(f)
            mask[th - fz:, :] = 0.0
            mask[th - fz - f:th - fz, :] = torch.minimum(
                mask[th - fz - f:th - fz, :], (1.0 - b)[:, None])
    return mask


def alternating_placements(n, axis):
    """(row, col) grid coords in sample order for the "alternating" scheme.

    The anchor sits at (0, 0); odd connect indices step to +1 on the active axis
    (right for horizontal, down for vertical), even indices to -1 (left / up),
    so tiles extend outward alternating +k, -k."""
    placements = []
    for idx in range(n):
        if idx == 0:
            placements.append((0, 0))
        else:
            k = (idx + 1) // 2
            step = k if idx % 2 == 1 else -k
            placements.append((step, 0) if axis == "vertical" else (0, step))
    return placements


def _axis_offsets(lengths, ol_list, n):
    """Absolute primary-axis origin (latent tokens) per tile, sample order.

    length[i] is each tile's extent along the active axis; ol_list[i] is how
    far tile i overlaps the already-placed same-side neighbour it faces. The
    anchor is placed at origin 0, so the seam stays anchored while per-tile
    extents and overlaps may differ."""
    offs = [0] * n
    lo_edge = 0          # leftmost (topmost) boundary reached
    hi_edge = lengths[0]  # rightmost (bottommost) boundary reached
    for i in range(1, n):
        if i % 2 == 1:   # grows toward +inf
            offs[i] = hi_edge - ol_list[i]
            hi_edge = offs[i] + lengths[i]
        else:            # grows toward -inf
            offs[i] = lo_edge + ol_list[i] - lengths[i]
            lo_edge = offs[i]
    return offs


def _register_hybrid_inpaint(model):
    """fade_mode 'hybrid': patch KSamplerX0Inpaint.__call__ while a tile samples.

    Stock masked-input mix: x_in = m*x + (1-m)*injected, injected =
    0.999*anchor + 0.001*noise. For any mid m the anchor's signal is scaled
    to (1-m) - a half-brightness content token the model never saw in
    training (the stock mix was built for the binary video-extend mask where
    m is only ever 0 or 1). That scaled anchor reads as noise structure and
    is the likely source of the fade band's residual grain.

    This patch rebuilds every masked video pixel's per-step input as

        anchor + pool(m) * sigma * eps        (pool = 2x2 amax of the mask)

    a FULL-STRENGTH anchor noised to exactly the level its pooled timestep
    label (1 - pool(m)*sigma) claims - the same construction as H3's
    reference/keyframe tokens. Free pixels (mask 1) and the audio stream
    keep the stock behaviour; the output blend is unchanged. Falls back to
    the stock __call__ when the latent is not an H3 AV latent or anything
    unexpected happens - a fallback prints one console warning per tile so a
    failed patch is never mistaken for the mode's real result, and the first
    successful step prints the fade-band stats. Returns a cleanup callable
    restoring the class."""
    import comfy.samplers as _cs
    import comfy.utils as _cu
    orig_call = _cs.KSamplerX0Inpaint.__call__
    engaged = [False]
    warned = [False]

    def hybrid_call(self, x, sigma, denoise_mask, model_options={}, seed=None, **kwargs):
        try:
            base = self.inner_model.inner_model
            shapes = getattr(base, "latent_shapes", None)
            if denoise_mask is None or shapes is None or len(shapes) < 2:
                raise TypeError
            if "denoise_mask_function" in model_options:
                denoise_mask = model_options["denoise_mask_function"](
                    sigma, denoise_mask,
                    extra_options={"model": self.inner_model, "sigmas": self.sigmas})
            vid_x, aud_x = _cu.unpack_latents(x, shapes)
            vid_anchor = _cu.unpack_latents(self.latent_image, shapes)[0]
            vid_eps = _cu.unpack_latents(self.noise, shapes)[0]
            pooled = base._token_grid_masks(denoise_mask, shapes)
            sigma_v = float(sigma.flatten()[0])
            anchor_noised = vid_anchor + pooled[0] * sigma_v * vid_eps
            vid_new = torch.where(pooled[0] < 1.0 - 1e-3, anchor_noised, vid_x)
            injected = base.scale_latent_inpaint(
                sigma=sigma, noise=self.noise, latent_image=self.latent_image)
            video_len = vid_x[0].numel()
            aud_inj = injected[..., video_len:].reshape(aud_x.shape)
            aud_mask = pooled[1] if len(pooled) > 1 else torch.ones_like(aud_x)
            aud_new = aud_mask * aud_x + (1.0 - aud_mask) * aud_inj
            x_new = _cu.pack_latents([vid_new, aud_new])[0].to(x.dtype)
        except Exception as e:
            if not warned[0]:
                warned[0] = True
                print(f"[MMH3-Extend] hybrid inpaint patch failed "
                      f"({type(e).__name__}: {e}); this tile falls back to the "
                      f"stock input mix")
            return orig_call(self, x, sigma, denoise_mask, model_options=model_options,
                             seed=seed, **kwargs)
        if not engaged[0]:
            engaged[0] = True
            patches = pooled[0][0, 0, :, ::2, ::2]
            mid = int(((patches > 1e-3) & (patches < 1.0 - 1e-3)).sum().item())
            print(f"[MMH3-Extend] hybrid inpaint engaged: {mid}/{patches.numel()} "
                  f"video patches in the fade band (min pooled mask "
                  f"{float(patches.min()):.3f})")
        out = self.inner_model(x_new, sigma, model_options=model_options, seed=seed)
        latent_mask = 1.0 - denoise_mask
        return out * denoise_mask + self.latent_image * latent_mask

    _cs.KSamplerX0Inpaint.__call__ = hybrid_call

    def _cleanup():
        _cs.KSamplerX0Inpaint.__call__ = orig_call
    return _cleanup


def _shrink_freeze_mask(m, shrink_tok, floor_tok=2):
    """Second-pass mask for the 'shrink_*_no_fade' second_pass_mode options.

    m is the tile's baked first-pass noise mask [H, W] (flat mode: exactly 0 on
    each seam's frozen band, fade_val on the fade band, 1 elsewhere). Every
    frozen run anchored at a tile edge is shrunk by `shrink_tok` latent tokens
    toward the seam (the seam-adjacent content stays frozen, the inner part is
    released) and the fade band is dropped - the result is a hard 0/1 mask, so
    no token carries a mid noise level in the second pass. Runs are measured
    per column from the top/bottom edges and per row from the left/right edges
    (seams are always edge-anchored); each keeps
    max(len - shrink_tok, min(len, floor_tok)) tokens, so narrow bands (e.g.
    the edge overlaps of 4_quadrants) never drop below floor_tok (32 px)
    unless they were narrower to begin with."""
    fz = (m <= 1e-4).to(torch.float32)
    if not bool(fz.any()):
        return torch.ones_like(m)
    k = int(shrink_tok)

    def shrink_runs(runs):
        return torch.maximum(runs - k, runs.clamp(max=floor_tok))

    top = shrink_runs(fz.cumprod(0).sum(0))            # per column, from the top edge
    bot = shrink_runs(fz.flip(0).cumprod(0).sum(0))    # per column, from the bottom edge
    left = shrink_runs(fz.cumprod(1).sum(1))           # per row, from the left edge
    right = shrink_runs(fz.flip(1).cumprod(1).sum(1))  # per row, from the right edge
    H, W = m.shape
    rr = torch.arange(H, device=m.device)[:, None]
    cc = torch.arange(W, device=m.device)[None, :]
    keep = ((rr < top[None, :]) | (rr >= H - bot[None, :])
            | (cc < left[:, None]) | (cc >= W - right[:, None]))
    return keep.to(m.dtype)


def _second_pass(model, noise, sampler_2nd, sigmas_2nd, negative, cfg, cond,
                 region, audio, m_static, masked_area_noise, fade_mode="flat",
                 second_pass_mode="unmasked"):
    """Re-denoise an already-blended tile.

    second_pass_mode 'unmasked': no noise mask at all - the ENTIRE region
    (overlap bands included) is sampled freely by sigmas_2nd; the blended tile
    latent is the starting latent_image, so a full-strength schedule
    regenerates it while a low-denoise schedule refines it globally.
    'tile_mask': the tile's baked overlap/fade mask is reused - the frozen
    band keeps the stitched content and only the feathered/free band is
    re-noised and resampled.
    'shrink_32_no_fade' / 'shrink_64_no_fade': second-pass-only masks - the
    baked mask's frozen bands are shrunk by 32/64 px toward the seams and the
    fade band is dropped (hard 0/1 mask, see _shrink_freeze_mask), so the
    released strip and the former fade band are re-denoised from the blended
    latent while a reduced frozen anchor stays glued to each seam; narrow
    bands keep at least 32 px of freeze.
    Audio is frozen in every masked mode; only 'unmasked' frees it.
    fade_mode 'hybrid' additionally patches the masked-input mix (see
    _register_hybrid_inpaint) in the masked modes. Returns the refined video
    latent."""
    if second_pass_mode == "unmasked":
        piece = {"samples": comfy.nested_tensor.NestedTensor((region, audio))}
        return sample_piece(piece, cond, model, noise, sampler_2nd, sigmas_2nd,
                            negative, cfg).tensors[0]
    if second_pass_mode.startswith("shrink_"):
        shrink_px = int(second_pass_mode.split("_")[1])
        m_static = _shrink_freeze_mask(m_static, shrink_px // VAE_DOWNSAMPLE)
    mv = (m_static + masked_area_noise * (1.0 - m_static))[None, None, None]
    ma = torch.zeros((1, 32, 2, audio.shape[-1]), device=audio.device, dtype=audio.dtype)
    piece = {
        "samples": comfy.nested_tensor.NestedTensor((region, audio)),
        "noise_mask": comfy.nested_tensor.NestedTensor((mv, ma)),
    }
    cleanup = (_register_hybrid_inpaint(model)
               if fade_mode == "hybrid" and bool((m_static < 1.0 - 1e-6).any())
               else None)
    try:
        out = sample_piece(piece, cond, model, noise, sampler_2nd, sigmas_2nd, negative, cfg)
    finally:
        if cleanup is not None:
            cleanup()
    return out.tensors[0]


def _merge(model, noise, sampler, sigmas, negative_list, cfg,
           audio, tconds, latents, plan,
           ol_ws, ol_hs, fws, fhs, overlap_mode, overlap_blend, inpaint=None,
           skip_first=False, masked_area_noise=0.0,
           fade_mode="flat", fade_val=0.5,
           sampler_2nd=None, sigmas_2nd=None, second_pass_mode="unmasked"):
    """Sample every tile and stitch them into one AV latent per a tile_plan.

    `tconds` / `negative_list` / `latents` are lists aligned to plan placements
    (latents used for their per-tile size and the shared audio). `ol_ws` /
    `ol_hs` / `fws` / `fhs` are per-tile overlap/fade extents (latent tokens),
    indexed by tile; the anchor tile never uses them. When `skip_first` is set,
    the anchor tile (latents[0]) is not sampled: its already-generated video and
    audio are dropped straight into the canvas and the extension tiles sample
    around it.

    fade_mode shapes the fade band's noise mask. The invariant that matters in
    H3 is per-token LABEL-INPUT CONSISTENCY: the conds mask is pooled to the
    2x2 token grid and becomes each token's timestep label, while the sampler
    mixes the token's input with the same mask value (m*noise + (1-m)*anchor
    == the anchor noised to level m*sigma). A mode is stable when each token's
    label matches its input:
      'flat'     - one constant fade_val over the whole band. Consistent even
                   at the patch level (a constant pools to itself). Empirically
                   the most stable mode.
      'gradient' - legacy linear 0->1 ramp (fade_val ignored). Consistent per
                   pixel, but patch-pooled labels take each 2x2 patch's max -
                   a mild sub-patch mismatch. Empirically close to flat. (The
                   original mosaic came from per-step fade_steps rebuilds that
                   desynced the once-baked labels, not from a static ramp.)
      'hybrid'   - same conds mask as 'gradient' (labels stay consistent), but
                   the masked-input mix is patched during sampling: instead of
                   the stock m*x + (1-m)*injected mix (whose anchor is scaled
                   to (1-m) brightness - out of distribution for mid m), each
                   masked pixel's input is rebuilt as anchor + pool(m)*sigma*eps
                   - a FULL-STRENGTH anchor noised to exactly the level its
                   label claims, the same construction as H3's reference
                   tokens. Targets the fade band's residual grain. Experimental.
    fade_val: mask value of the fade band in 'flat' mode; lower = stronger
    freeze (more neighbour content preserved), higher = more noise injected /
    freer generation. 0 = fully frozen band, 1 = mask off.
    `masked_area_noise` raises every mask value toward 1 (all three roles at
    once). Returns (acc_video, out_audio, tiles_info, t0) where t0 is tile 0's
    raw sample_piece output (None when the anchor came from skip_first)."""
    _, cp, T, _, _ = latents[0].tensors[0].shape
    sampling = getattr(model, "model_sampling", None)
    axis = plan["axis"]
    n = len(tconds)
    # Second pass runs on blocks from the 2nd on; disabled when sampler_2nd or a
    # usable sigmas_2nd (>= 1 step) is missing.
    second_on = (sampler_2nd is not None and sigmas_2nd is not None
                 and sigmas_2nd.shape[-1] > 1)

    heights = [l.tensors[0].shape[3] for l in latents]
    widths = [l.tensors[0].shape[4] for l in latents]
    if axis == "horizontal":
        primary = widths
        ol_pri = ol_ws
    else:
        primary = heights
        ol_pri = ol_hs
    offsets = _axis_offsets(primary, ol_pri, n)
    if axis == "horizontal":
        lo = min(offsets)
        hi = max(offsets[i] + widths[i] for i in range(n))
        acc_h = heights[0]
        acc_w = hi - lo
    else:
        lo = min(offsets)
        hi = max(offsets[i] + heights[i] for i in range(n))
        acc_w = widths[0]
        acc_h = hi - lo
    acc = torch.zeros((1, cp, T, acc_h, acc_w), device=latents[0].tensors[0].device,
                      dtype=latents[0].tensors[0].dtype)

    tiles_info = []
    out_a = None
    t0 = None
    for i in range(n):
        th_i, tw_i = heights[i], widths[i]
        oW, oH, fW, fH = ol_ws[i], ol_hs[i], fws[i], fhs[i]
        ro = offsets[i] - lo if axis == "vertical" else 0
        co = offsets[i] - lo if axis == "horizontal" else 0
        done_top = done_bottom = done_left = done_right = False
        if axis == "horizontal":
            if i % 2 == 1:
                done_left = True
            elif i > 0:
                done_right = True
        else:
            if i % 2 == 1:
                done_top = True
            elif i > 0:
                done_bottom = True

        if skip_first and i == 0:
            # Reuse the already-generated anchor video; no sampling, no seams.
            acc[:, :, :, ro:ro + th_i, co:co + tw_i] = latents[0].tensors[0]
            out_a = latents[0].tensors[1]
            tiles_info.append({
                "row": ro + lo if axis == "vertical" else 0,
                "col": co + lo if axis == "horizontal" else 0,
                "tile_h": th_i, "tile_w": tw_i, "scheme": plan["scheme"], "axis": axis,
                "done_top": False, "done_bottom": False, "done_left": False, "done_right": False,
                "overlap_w": oW, "overlap_h": oH, "fade_w": fW, "fade_h": fH,
                "overlap_mode": overlap_mode, "overlap_blend": overlap_blend,
                "skipped": True,
            })
            continue

        tile = torch.zeros((1, cp, T, th_i, tw_i), device=acc.device, dtype=acc.dtype)
        if done_left and oW > 0:
            tile[:, :, :, :, :oW] = acc[:, :, :, ro:ro + th_i, co:co + oW]
        if done_right and oW > 0:
            tile[:, :, :, :, tw_i - oW:] = acc[:, :, :, ro:ro + th_i, co + tw_i - oW:co + tw_i]
        if done_top and oH > 0:
            tile[:, :, :, :oH, :] = acc[:, :, :, ro:ro + oH, co:co + tw_i]
        if done_bottom and oH > 0:
            tile[:, :, :, th_i - oH:, :] = acc[:, :, :, ro + th_i - oH:ro + th_i, co:co + tw_i]

        m = _edge_fade_mask(th_i, tw_i, oH, oW, fH, fW,
                            done_top, done_bottom, done_left, done_right,
                            flat_val=(fade_val if fade_mode == "flat" else None))
        m_v = m + masked_area_noise * (1.0 - m)
        mv = m_v[None, None, None]
        tatk = audio.shape[-1]
        ma = (torch.ones if i == 0 else torch.zeros)(
            (1, 32, 2, tatk), device=audio.device, dtype=audio.dtype)
        piece = {
            "samples": comfy.nested_tensor.NestedTensor((tile, audio)),
            "noise_mask": comfy.nested_tensor.NestedTensor((mv, ma)),
        }

        cond = tconds[i]
        negative = negative_list[i]
        c_net = None
        if inpaint is not None and (done_left or done_right or done_top or done_bottom):
            vis = (1.0 - (m > 0.5).to(torch.float32))[None, None, None]
            vis = vis.expand(1, 1, T, th_i, tw_i).to(device=acc.device)
            masked_t = torch.zeros((1, 24, T, th_i, tw_i), device=vis.device, dtype=vis.dtype)
            masked_t[:, :, :, :, :] = acc[:, :, :, ro:ro + th_i, co:co + tw_i] * vis
            base = torch.zeros(1, 24, T, th_i, tw_i, device=vis.device, dtype=vis.dtype)
            inp_net = inpaint["control_net"].copy()
            inp_net.cond_hint = torch.cat([base, vis, masked_t], dim=1)
            inp_net.strength = inpaint["strength"]
            inp_net.timestep_percent_range = resolve_control_range(inpaint, sigmas, sampling)
            c_net = inp_net
        if c_net is not None:
            cond = apply_control(cond, c_net)

        cleanup = (_register_hybrid_inpaint(model)
                   if fade_mode == "hybrid" and bool((m < 1.0 - 1e-6).any())
                   else None)
        try:
            out = sample_piece(piece, cond, model, noise, sampler, sigmas, negative, cfg)
        finally:
            if cleanup is not None:
                cleanup()
        tile_v = out.tensors[0]
        if i == 0:
            out_a = out.tensors[1]
            t0 = out

        region = acc[:, :, :, ro:ro + th_i, co:co + tw_i].clone()
        if done_left and oW > 0:
            tt = torch.linspace(0.0, 1.0, oW, device=region.device, dtype=region.dtype)
            wts = blend_weights(tt, overlap_blend, overlap_mode)
            region[:, :, :, :, :oW] = (region[:, :, :, :, :oW] * (1.0 - wts[None, None, None, None, :])
                                       + tile_v[:, :, :, :, :oW] * wts[None, None, None, None, :])
        if done_right and oW > 0:
            tt = torch.linspace(0.0, 1.0, oW, device=region.device, dtype=region.dtype)
            wts = blend_weights(1.0 - tt, overlap_blend, overlap_mode)
            region[:, :, :, :, tw_i - oW:] = (region[:, :, :, :, tw_i - oW:] * (1.0 - wts[None, None, None, None, :])
                                              + tile_v[:, :, :, :, tw_i - oW:] * wts[None, None, None, None, :])
        if done_top and oH > 0:
            tt = torch.linspace(0.0, 1.0, oH, device=region.device, dtype=region.dtype)
            wts = blend_weights(tt, overlap_blend, overlap_mode)
            region[:, :, :, :oH, :] = (region[:, :, :, :oH, :] * (1.0 - wts[None, None, None, :, None])
                                       + tile_v[:, :, :, :oH, :] * wts[None, None, None, :, None])
        if done_bottom and oH > 0:
            tt = torch.linspace(0.0, 1.0, oH, device=region.device, dtype=region.dtype)
            wts = blend_weights(1.0 - tt, overlap_blend, overlap_mode)
            region[:, :, :, th_i - oH:, :] = (region[:, :, :, th_i - oH:, :] * (1.0 - wts[None, None, None, :, None])
                                              + tile_v[:, :, :, th_i - oH:, :] * wts[None, None, None, :, None])
        band = torch.zeros((1, 1, 1, th_i, tw_i), device=region.device, dtype=torch.bool)
        if done_left and oW > 0:
            band[:, :, :, :, :oW] = True
        if done_right and oW > 0:
            band[:, :, :, :, tw_i - oW:] = True
        if done_top and oH > 0:
            band[:, :, :, :oH, :] = True
        if done_bottom and oH > 0:
            band[:, :, :, th_i - oH:, :] = True
        region = torch.where(band, region, tile_v)
        # EXPERIMENTAL second pass: after the overlap blend, re-denoise this
        # block (from the 2nd on) with sampler_2nd/sigmas_2nd.
        if second_on and i > 0:
            region = _second_pass(model, noise, sampler_2nd, sigmas_2nd, negative,
                                  cfg, cond, region, out.tensors[1], m,
                                  masked_area_noise, fade_mode, second_pass_mode)
        acc[:, :, :, ro:ro + th_i, co:co + tw_i] = region

        tiles_info.append({
            "row": ro + lo if axis == "vertical" else 0,
            "col": co + lo if axis == "horizontal" else 0,
            "tile_h": th_i, "tile_w": tw_i, "scheme": plan["scheme"], "axis": axis,
            "done_top": done_top, "done_bottom": done_bottom,
            "done_left": done_left, "done_right": done_right,
            "overlap_w": oW, "overlap_h": oH, "fade_w": fW, "fade_h": fH,
            "overlap_mode": overlap_mode, "overlap_blend": overlap_blend,
            "skipped": False,
        })
    return acc, out_a, tiles_info, t0


def _quadrant_placements(sizes, ol_ws, ol_hs):
    """5-tile (center + four corners) placement and rectangle-closure validation.

    sizes[i] = (height, width) in latent tokens for tiles 0..4. ol_ws[i] /
    ol_hs[i] are tile i's horizontal / vertical overlap depth (tokens) toward
    its inner neighbours; the central tile 0 is the anchor and carries none.
    Tiles 1..4 sit top-left, top-right, bottom-left, bottom-right and together
    with the center must close into one complete rectangle:
      * the outer edges align - the two tiles on each side end at the same
        row/column;
      * each edge-band is covered with no gap - the side-by-side corner tiles
        reach each other (overlapping, like the center overlaps them).
    Tiles 0 & 1 (center and top-left) fix the top/left datum; tiles 2..4 are
    then checked against it. Returns per-tile absolute canvas origin (row, col).
    """
    (h0, w0), (h1, w1), (h2, w2), (h3, w3), (h4, w4) = sizes
    ow1, ow2, ow3, ow4 = ol_ws[1:]
    oh1, oh2, oh3, oh4 = ol_hs[1:]

    def _check(cond, msg):
        if not cond:
            raise ValueError(f"4_quadrants_expand: {msg}")

    # outer edges align (the two tiles on each side end on the same line)
    _check(oh1 - h1 == oh2 - h2,
           f"top edge does not close: top-right extends to row {oh2 - h2} but "
           f"top-left to row {oh1 - h1}; need overlap_height_2 - h_2 == "
           f"overlap_height_1 - h_1")
    _check(h3 - oh3 == h4 - oh4,
           f"bottom edge does not close: bottom-left starts at row {h3 - oh3} but "
           f"bottom-right at row {h4 - oh4}; need h_3 - overlap_height_3 == "
           f"h_4 - overlap_height_4")
    _check(ow1 - w1 == ow3 - w3,
           f"left edge does not close: top-left starts at col {ow1 - w1} but "
           f"bottom-left at col {ow3 - w3}; need overlap_width_1 - w_1 == "
           f"overlap_width_3 - w_3")
    _check(w2 - ow2 == w4 - ow4,
           f"right edge does not close: top-right ends at col {w2 - ow2 + w0} "
           f"but bottom-right at col {w4 - ow4 + w0}; need w_2 - overlap_width_2 "
           f"== w_4 - overlap_width_4")

    # edge-bands covered (the corner tiles reach each other around the center)
    _check(ow1 + ow2 >= w0,
           f"top band has a gap: top-left and top-right combined overlap "
           f"{ow1} + {ow2} = {ow1 + ow2} but must reach the center width {w0}")
    _check(ow3 + ow4 >= w0,
           f"bottom band has a gap: bottom-left and bottom-right combined overlap "
           f"{ow3 + ow4} must reach the center width {w0}")
    _check(oh1 + oh3 >= h0,
           f"left band has a gap: top-left and bottom-left combined overlap "
           f"{oh1 + oh3} must reach the center height {h0}")
    _check(oh2 + oh4 >= h0,
           f"right band has a gap: top-right and bottom-right combined overlap "
           f"{oh2 + oh4} must reach the center height {h0}")

    return [
        (0, 0),
        (oh1 - h1, ow1 - w1),
        (oh2 - h2, w0 - ow2),
        (h0 - oh3, ow3 - w3),
        (h0 - oh4, w0 - ow4),
    ]


def _edge_mask(depth, fade, fade_val=0.5, frozen=None, smooth=False):
    """Noise-mask profile across an overlap of `depth` tokens: 0 (frozen) over
    the seam side, then a fade band toward the tile interior. 0 fade freezes
    the whole band.
    fade_val: constant value used for the WHOLE fade band (fade_mode 'flat').
    A uniform band is label-input consistent even at the 2x2 patch level (a
    constant pools to itself); a ramp (smooth=True) is consistent per pixel
    but its pooled label takes each patch's max - a mild sub-patch mismatch.
    Empirically both work; the original mosaic came from per-step fade_steps
    rebuilds, not from a static ramp.
    smooth=True restores the linspace ramp - for fade_mode 'gradient'/'hybrid'
    masks and for post-sampling stitch weights (pure latent blending, no H3
    mechanism involved).
    frozen: explicit fully-frozen width (tokens) at the seam side. When None the
    frozen width is depth - fade (the whole band is frozen+fade); when set the
    band is frozen, then fade, then free (mask 1) up to depth."""
    depth = int(depth)
    f = min(int(fade), depth)
    if frozen is None:
        fz = depth - f
    else:
        fz = min(int(frozen), depth)
        f = min(int(fade), depth - fz)
    mask = torch.ones(depth, dtype=torch.float32)
    if f > 0:
        mask[fz:fz + f] = torch.linspace(0.0, 1.0, f) if smooth else fade_val
    mask[:fz] = 0.0
    return mask


def _tile_mask_from_rects(rects, hi, wi, fW, fH, fade_val=0.5,
                          frozenW=None, frozenH=None, smooth=False):
    """Build a tile noise mask from overlap rects (r0, r1, c0, c1, dx, dy).

    dx/dy are the signed offsets from this tile's center to the neighbor's
    center; each axis gets its own _edge_mask band, diagonal bands combine via
    screen (h + v - h*v). frozenW/frozenH give each axis an explicit fully-frozen
    width (tokens) instead of the implicit depth - fade split. smooth=True
    builds the linspace variant (fade_mode 'gradient'/'hybrid' and stitch
    weights)."""
    m = torch.ones(hi, wi, dtype=torch.float32)
    for (r0, r1, c0, c1, dx, dy) in rects:
        rh, cw = r1 - r0, c1 - c0
        if rh <= 0 or cw <= 0:
            continue
        profile = None
        if dx != 0:
            h_mask = _edge_mask(cw, fW, fade_val, frozen=frozenW, smooth=smooth)
            if dx > 0:
                h_mask = h_mask.flip(0)
            profile = h_mask[None, :].expand(rh, cw).clone()
        if dy != 0:
            v_mask = _edge_mask(rh, fH, fade_val, frozen=frozenH, smooth=smooth)
            if dy > 0:
                v_mask = v_mask.flip(0)
            v_profile = v_mask[:, None].expand(rh, cw)
            profile = v_profile.clone() if profile is None else profile + v_profile - profile * v_profile
        if profile is None:
            continue
        m[r0:r1, c0:c1] = torch.minimum(m[r0:r1, c0:c1], profile.to(m.dtype))
    return m


def _merge_2d(model, noise, sampler, sigmas, negative_list, cfg,
              audio, tconds, latents, plan,
              ol_ws, ol_hs, fws, fhs, overlap_mode, overlap_blend, inpaint=None,
              skip_first=False, masked_area_noise=0.0,
              bug_patch=None,
              fade_mode="flat", fade_val=0.5,
              sampler_2nd=None, sigmas_2nd=None, second_pass_mode="unmasked"):
    """Sample the 5 tiles of '4_quadrants_expand' and stitch them into one AV
    latent. Tile 0 is the center anchor; tiles 1..4 are sampled around it and
    blended over the rectangles where they overlap already-placed content (the
    center and their neighbouring corner tile). `bug_patch` (output of the
    'MMH3 Last Quadrant Patch' node, pixel units) replaces the default
    rects-based mask of any tile it enables with an explicit center/edge
    frozen+fade profile. Returns (acc_video, out_audio, tiles_info, t0) where
    t0 is tile 0's raw sample_piece output (None when the anchor came from
    skip_first)."""
    sizes = [(l.tensors[0].shape[3], l.tensors[0].shape[4]) for l in latents]
    origins = _quadrant_placements(sizes, ol_ws, ol_hs)
    n = len(tconds)
    _, cp, T, _, _ = latents[0].tensors[0].shape
    second_on = (sampler_2nd is not None and sigmas_2nd is not None
                 and sigmas_2nd.shape[-1] > 1)
    sampling = getattr(model, "model_sampling", None)
    top = origins[1][0]
    left = origins[1][1]
    acc_h = (origins[4][0] + sizes[4][0]) - top
    acc_w = (origins[4][1] + sizes[4][1]) - left
    acc = torch.zeros((1, cp, T, acc_h, acc_w), device=latents[0].tensors[0].device,
                      dtype=latents[0].tensors[0].dtype)

    prior = []  # already-placed canvas rects: ((row0, col0, row1, col1), tile_idx)
    tiles_info = []
    out_a = None
    t0 = None
    patch_tiles = (bug_patch.get("tiles", {})
                   if bug_patch is not None
                   and bug_patch.get("target") == "4_quadrants_expand" else {})
    for i in range(n):
        ro, co = origins[i]
        hi, wi = sizes[i]
        oW, oH = ol_ws[i], ol_hs[i]
        fW, fH = fws[i], fhs[i]
        # canvas-local origin of this tile (acc starts at the rectangle's top-left)
        at, ac = ro - top, co - left

        # rectangles where this tile overlaps already-placed content, tile-local,
        # tagged with the source tile (0 = the center anchor). Each seam abuts
        # one or two of the tile's edges, which sets the blend gradient
        # direction. The center tile (i==0) has no prior content.
        seams = []
        for ((pr0, pc0, pr1, pc1), src) in prior:
            ir0 = max(ro, pr0); ir1 = min(ro + hi, pr1)
            ic0 = max(co, pc0); ic1 = min(co + wi, pc1)
            if ir0 < ir1 and ic0 < ic1:
                seams.append((ir0 - ro, ir1 - ro, ic0 - co, ic1 - co,
                              pr0, pc0, pr1, pc1, src))

        done = {
            "top": any(r0 == 0 for r0, r1, c0, c1, *_ in seams),
            "bottom": any(r1 == hi for r0, r1, c0, c1, *_ in seams),
            "left": any(c0 == 0 for r0, r1, c0, c1, *_ in seams),
            "right": any(c1 == wi for r0, r1, c0, c1, *_ in seams),
        }

        # mask: 1 = sample freely, 0 = frozen (keep already-placed content); a
        # seam's own fade band ramps 0 at the seam -> 1 toward the tile interior.
        m = torch.ones(hi, wi, dtype=torch.float32)
        tile = torch.zeros((1, cp, T, hi, wi), device=acc.device, dtype=acc.dtype)
        # 2-D weight given to the NEW tile's content, 1 outside any seam.
        wt = torch.ones(hi, wi, dtype=acc.dtype)

        # NEW (ported from MMH3 Mask Preview / _compute_masks_all_pairs): the
        # fade direction per overlap comes from the neighbor's center position
        # relative to this tile's center, not from which tile edge the overlap
        # touches. Each axis gets its own _edge_mask band; diagonal overlaps
        # combine the two bands with the screen op (h + v - h*v) so the corner
        # stays smooth. The stitch weight follows the noise mask directly.
        rects = []
        for (r0, r1, c0, c1, pr0, pc0, pr1, pc1, _src) in seams:
            dx = (pc0 + pc1) / 2.0 - (co + wi / 2.0)
            dy = (pr0 + pr1) / 2.0 - (ro + hi / 2.0)
            rects.append((r0, r1, c0, c1, dx, dy))
            tile[:, :, :, r0:r1, c0:c1] = acc[:, :, :, at + r0:at + r1, ac + c0:ac + c1]
        m = _tile_mask_from_rects(rects, hi, wi, fW, fH, fade_val=fade_val,
                                  smooth=(fade_mode != "flat"))
        # Stitch weight keeps the smooth linspace ramp: it is a pure
        # post-sampling latent blend with no H3 mechanism involved, so the
        # flat noise-mask constant does not apply here.
        wt = _tile_mask_from_rects(rects, hi, wi, fW, fH, smooth=True).to(acc.dtype)

        # OLD (seam-edge based): gradient direction chosen by which tile edge
        # the overlap rect touches; wt shaped by overlap_mode/overlap_blend.
        # for (r0, r1, c0, c1) in seams:
        #     rh, cw = r1 - r0, c1 - c0
        #     prof = torch.ones(rh, cw, dtype=acc.dtype)
        #     mm = torch.ones(rh, cw, dtype=torch.float32)
        #     if c0 == 0:  # abuts the left edge
        #         depth = cw
        #         w = blend_weights(torch.linspace(0.0, 1.0, depth), overlap_blend, overlap_mode)
        #         prof *= w[None, :]
        #         mm *= _edge_mask(depth, fW)[None, :]
        #     if c1 == wi:  # right edge
        #         depth = cw
        #         w = blend_weights(torch.linspace(1.0, 0.0, depth), overlap_blend, overlap_mode)
        #         prof *= w[None, :]
        #         mm *= _edge_mask(depth, fW)[None, :]
        #     if r0 == 0:  # top edge
        #         depth = rh
        #         w = blend_weights(torch.linspace(0.0, 1.0, depth), overlap_blend, overlap_mode)
        #         prof *= w[:, None]
        #         mm *= _edge_mask(depth, fH)[:, None]
        #     if r1 == hi:  # bottom edge
        #         depth = rh
        #         w = blend_weights(torch.linspace(1.0, 0.0, depth), overlap_blend, overlap_mode)
        #         prof *= w[:, None]
        #         mm *= _edge_mask(depth, fH)[:, None]
        #     m[r0:r1, c0:c1] = torch.minimum(m[r0:r1, c0:c1], mm)
        #     wt[r0:r1, c0:c1] *= prof
        #     tile[:, :, :, r0:r1, c0:c1] = acc[:, :, :, at + r0:at + r1, ac + c0:ac + c1]

        # bug_patch (4_quadrants): explicit center/edge frozen+fade profile
        # replacing the rects-based mask for any tile enabled in patch_tiles.
        # Seams against the CENTER tile (src 0) use the center_* widths; seams
        # against the neighbouring corner tiles use the edge_* widths. The
        # per-seam fade direction comes from dx/dy exactly as in the default
        # mask, so TL/TR/BL/BR tiles all work with the same code path.
        if patch_tiles.get(i, False) and i > 0:
            cfz_w = int(bug_patch.get("center_mask_w_freeze", 0)) // VAE_DOWNSAMPLE
            cfz_h = int(bug_patch.get("center_mask_h_freeze", 0)) // VAE_DOWNSAMPLE
            cfd_w = int(bug_patch.get("center_mask_w_fade", 32)) // VAE_DOWNSAMPLE
            cfd_h = int(bug_patch.get("center_mask_h_fade", 32)) // VAE_DOWNSAMPLE
            efz = int(bug_patch.get("edge_mask_freeze", 0)) // VAE_DOWNSAMPLE
            efd = int(bug_patch.get("edge_fade", 32)) // VAE_DOWNSAMPLE
            smooth = (fade_mode != "flat")
            center_rects = []
            edge_rects = []
            for (r0, r1, c0, c1, pr0, pc0, pr1, pc1, src) in seams:
                dx = (pc0 + pc1) / 2.0 - (co + wi / 2.0)
                dy = (pr0 + pr1) / 2.0 - (ro + hi / 2.0)
                (center_rects if src == 0 else edge_rects).append(
                    (r0, r1, c0, c1, dx, dy))
            print(f"[MMH3SpatialExtendVideo] quadrant patch: tile {i} - "
                  f"{len(center_rects)} center seam(s), {len(edge_rects)} edge "
                  f"seam(s); center frozen w/h={cfz_w}/{cfz_h}tok "
                  f"fade w/h={cfd_w}/{cfd_h}tok; edge frozen={efz}tok "
                  f"fade={efd}tok")
            m = torch.ones(hi, wi, dtype=torch.float32)
            wt = torch.ones(hi, wi, dtype=acc.dtype)
            if center_rects:
                m = torch.minimum(
                    m, _tile_mask_from_rects(center_rects, hi, wi, cfd_w, cfd_h,
                                             fade_val=fade_val, frozenW=cfz_w,
                                             frozenH=cfz_h, smooth=smooth))
                wt = torch.minimum(
                    wt, _tile_mask_from_rects(center_rects, hi, wi, cfd_w, cfd_h,
                                              frozenW=cfz_w, frozenH=cfz_h,
                                              smooth=True).to(acc.dtype))
            if edge_rects:
                m = torch.minimum(
                    m, _tile_mask_from_rects(edge_rects, hi, wi, efd, efd,
                                             fade_val=fade_val, frozenW=efz,
                                             frozenH=efz, smooth=smooth))
                wt = torch.minimum(
                    wt, _tile_mask_from_rects(edge_rects, hi, wi, efd, efd,
                                              frozenW=efz, frozenH=efz,
                                              smooth=True).to(acc.dtype))

        m_v = m + masked_area_noise * (1.0 - m)
        mv = m_v[None, None, None]
        tatk = audio.shape[-1]
        ma = (torch.ones if i == 0 else torch.zeros)(
            (1, 32, 2, tatk), device=audio.device, dtype=audio.dtype)
        piece = {
            "samples": comfy.nested_tensor.NestedTensor((tile, audio)),
            "noise_mask": comfy.nested_tensor.NestedTensor((mv, ma)),
        }

        cond = tconds[i]
        negative = negative_list[i]
        c_net = None
        if inpaint is not None and any(done.values()):
            vis = (1.0 - (m > 0.5).to(torch.float32))[None, None, None]
            vis = vis.expand(1, 1, T, hi, wi).to(device=acc.device)
            masked_t = torch.zeros((1, 24, T, hi, wi), device=vis.device, dtype=vis.dtype)
            masked_t[:, :, :, :, :] = acc[:, :, :, at:at + hi, ac:ac + wi] * vis
            base = torch.zeros(1, 24, T, hi, wi, device=vis.device, dtype=vis.dtype)
            inp_net = inpaint["control_net"].copy()
            inp_net.cond_hint = torch.cat([base, vis, masked_t], dim=1)
            inp_net.strength = inpaint["strength"]
            inp_net.timestep_percent_range = resolve_control_range(inpaint, sigmas, sampling)
            c_net = inp_net
        if c_net is not None:
            cond = apply_control(cond, c_net)

        if skip_first and i == 0:
            acc[:, :, :, at:at + hi, ac:ac + wi] = latents[0].tensors[0]
            out_a = latents[0].tensors[1]
        else:
            cleanup = (_register_hybrid_inpaint(model)
                       if fade_mode == "hybrid" and bool((m < 1.0 - 1e-6).any())
                       else None)
            try:
                out = sample_piece(piece, cond, model, noise, sampler, sigmas, negative, cfg)
            finally:
                if cleanup is not None:
                    cleanup()
            tile_v = out.tensors[0]
            if i == 0:
                out_a = out.tensors[1]
                t0 = out
            # blend the sampled tile with the already-placed content it overlaps
            wt4 = wt[None, None, None]
            region = tile * (1.0 - wt4) + tile_v * wt4
            # EXPERIMENTAL second pass: after the overlap blend, re-denoise this
            # block (from the 2nd on) with sampler_2nd/sigmas_2nd.
            if second_on and i > 0:
                region = _second_pass(model, noise, sampler_2nd, sigmas_2nd, negative,
                                      cfg, cond, region, out.tensors[1], m,
                                      masked_area_noise, fade_mode, second_pass_mode)
            acc[:, :, :, at:at + hi, ac:ac + wi] = region

        tiles_info.append({
            "row": ro - top, "col": co - left,
            "tile_h": hi, "tile_w": wi, "scheme": plan["scheme"], "axis": plan.get("axis", "horizontal"),
            "done_top": done["top"], "done_bottom": done["bottom"],
            "done_left": done["left"], "done_right": done["right"],
            "overlap_w": oW, "overlap_h": oH, "fade_w": fW, "fade_h": fH,
            "overlap_mode": overlap_mode, "overlap_blend": overlap_blend,
            "fade_mode": fade_mode, "fade_val": fade_val,
            "skipped": bool(skip_first and i == 0),
        })
        prior.append(((ro, co, ro + hi, co + wi), i))

    return acc, out_a, tiles_info, t0


def _compute_masks_all_pairs(tc, fade_val=0.5, fade_mode="flat"):
    """Compute masks for all non-anchor tiles using all-pair overlap detection.

    For each tile, check overlap with every other tile. The fade direction is
    determined by the relative position of the neighbor (not by seam shape).
    Returns (masks, canvas_h, canvas_w, rect_info); rect_info is per-tile
    (rects, hi, wi) so a caller can rebuild masks at a different fade width.
    """
    from .tile_editor import _tile_placements, VAE_DOWNSAMPLE

    tiles = tc["tiles"]
    scheme = tc["scheme"]
    axis = tc.get("axis", "horizontal")
    placements = _tile_placements(tiles, scheme, axis)

    n = len(placements)
    # placements: list of (x, y, w, h) in pixel space
    # convert to latent space
    sizes = []
    origins = []
    fade_ws = []
    fade_hs = []
    for i, (x, y, w, h) in enumerate(placements):
        sizes.append((h // VAE_DOWNSAMPLE, w // VAE_DOWNSAMPLE))
        origins.append((y // VAE_DOWNSAMPLE, x // VAE_DOWNSAMPLE))
        if i == 0:
            fade_ws.append(0)
            fade_hs.append(0)
        else:
            fade_ws.append(tiles[i].get("fade_w", 32) // VAE_DOWNSAMPLE)
            fade_hs.append(tiles[i].get("fade_h", 32) // VAE_DOWNSAMPLE)

    masks = []
    rect_info = []
    for i in range(1, n):
        hi, wi = sizes[i]
        ro, co = origins[i]
        rects = []

        for j in range(n):
            if j == i:
                continue
            hj, wj = sizes[j]
            rj0, cj0 = origins[j]

            ir0 = max(ro, rj0)
            ir1 = min(ro + hi, rj0 + hj)
            ic0 = max(co, cj0)
            ic1 = min(co + wi, cj0 + wj)
            if ir0 >= ir1 or ic0 >= ic1:
                continue

            # overlap region in tile-i local coords
            rr0, rr1 = ir0 - ro, ir1 - ro
            rc0, rc1 = ic0 - co, ic1 - co

            # Determine neighbor's relative position using center coords
            ci_cx, ci_cy = co + wi / 2.0, ro + hi / 2.0
            cj_cx, cj_cy = cj0 + wj / 2.0, rj0 + hj / 2.0
            dx = cj_cx - ci_cx  # positive = neighbor is right
            dy = cj_cy - ci_cy  # positive = neighbor is below

            rects.append((rr0, rr1, rc0, rc1, dx, dy))

        m = _tile_mask_from_rects(rects, hi, wi, fade_ws[i], fade_hs[i],
                                  fade_val=fade_val, smooth=(fade_mode != "flat"))
        masks.append(m)
        rect_info.append((rects, hi, wi))

    if masks:
        # compute canvas size
        canvas_h = max(o[0] + s[0] for o, s in zip(origins, sizes))
        canvas_w = max(o[1] + s[1] for o, s in zip(origins, sizes))
        return torch.stack(masks, dim=0), canvas_h, canvas_w, rect_info
    return torch.zeros(0, 1, 1, dtype=torch.float32), 1, 1, []


class MMH3MaskPreview(io.ComfyNode):
    """Preview the masks that would be used for sampling.
    Experimental node for verifying mask computation logic."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3MaskPreview",
            display_name="MMH3 Mask Preview",
            category="MMH3",
            description="Preview sampling masks computed from tile_config. Experimental node for verifying mask logic.",
            inputs=[
                io.Dict.Input("tile_config",
                    tooltip="Output of 'MMH3 Spatial Tile Editor'."),
                io.Combo.Input("fade_mode", options=["flat", "gradient", "hybrid"], default="flat",
                    tooltip="How the fade band's noise mask is shaped (must match 'MMH3 Spatial Extend Video'). 'flat': constant flat_fade_value band (most stable). 'gradient': legacy 0->1 ramp (close to flat). 'hybrid': gradient labels + patched full-strength noise-anchored input mix (experimental)."),
                io.Float.Input("flat_fade_value", default=0.5, min=0.0, max=1.0, step=0.01,
                    tooltip="Mask value of the whole fade band (fade_mode 'flat'). Lower = stronger freeze (more neighbour content preserved); higher = more noise injected / freer generation. 0 = fully frozen band, 0.5 = half-preserved, 1 = mask off."),
            ],
            outputs=[
                io.Mask.Output("debug_mask", tooltip="[M, H, W] one feathered mask per tile (tiles 1..N)."),
                io.Image.Output("preview", tooltip="Colored overlay showing each tile's mask."),
                io.String.Output("debug_text", tooltip="Diagnostic information about mask computation."),
            ],
        )

    @classmethod
    def execute(cls, tile_config, fade_mode="flat",
                flat_fade_value=0.5) -> io.NodeOutput:
        tc = tile_config
        if not tc.get("tiles"):
            empty_mask = torch.zeros(1, 64, 64, dtype=torch.float32)
            empty_preview = torch.zeros(1, 64, 64, 3, dtype=torch.float32)
            return io.NodeOutput(empty_mask, empty_preview, "no tiles")

        masks, ch, cw, _ = _compute_masks_all_pairs(tc, fade_val=flat_fade_value,
                                                    fade_mode=fade_mode)
        n = masks.shape[0] if masks.numel() > 0 else 0

        from .tile_editor import _tile_placements, VAE_DOWNSAMPLE
        tiles_list = tc["tiles"]
        placements = _tile_placements(tiles_list, tc["scheme"], tc.get("axis", "horizontal"))
        debug_lines = [f"n={n} canvas={cw}x{ch}"]
        for idx in range(n):
            ti = idx + 1
            pi = idx + 1
            t = tiles_list[pi]
            fw = t.get("fade_w", 32) // VAE_DOWNSAMPLE
            fh = t.get("fade_h", 32) // VAE_DOWNSAMPLE
            debug_lines.append(f"  tile{ti}(idx={pi}): overlap_w={t.get('overlap_w','?')} overlap_h={t.get('overlap_h','?')} fw_i={fw} fh_i={fh} placement={placements[pi]}")
        for idx in range(n):
            ti = idx + 1
            m = masks[idx]
            debug_lines.append(f"tile{ti}: min={m.min():.3f} max={m.max():.3f} shape={list(m.shape)}")
            debug_lines.append(f"  row0_full={m[0,:].tolist()}")
            debug_lines.append(f"  col0_full={m[:,0].tolist()}")
            # print a middle row
            mid_r = m.shape[0] // 2
            debug_lines.append(f"  row{mid_r}_full={m[mid_r,:].tolist()}")

        # Build preview image: black canvas with colored tile regions
        if n > 0:
            from .tile_editor import _tile_placements, VAE_DOWNSAMPLE
            placements = _tile_placements(tc["tiles"], tc["scheme"], tc.get("axis", "horizontal"))
            preview = torch.zeros(ch, cw, 3, dtype=torch.float32)
            colors = [
                [1, 0, 0], [0, 1, 0], [0, 0, 1],
                [1, 1, 0], [0, 1, 1],
            ]
            for idx in range(n):
                tile_idx = idx + 1
                x, y, w, h = placements[tile_idx]
                y0 = y // VAE_DOWNSAMPLE
                x0 = x // VAE_DOWNSAMPLE
                ht = h // VAE_DOWNSAMPLE
                wt = w // VAE_DOWNSAMPLE
                c = colors[idx % len(colors)]
                mask_2d = masks[idx]
                for ch_idx in range(3):
                    preview[y0:y0+ht, x0:x0+wt, ch_idx] = (
                        mask_2d * c[ch_idx] +
                        preview[y0:y0+ht, x0:x0+wt, ch_idx] * (1.0 - mask_2d)
                    )
            preview = preview.unsqueeze(0)  # [1, H, W, 3]
        else:
            preview = torch.zeros(1, ch, cw, 3, dtype=torch.float32)

        return io.NodeOutput(masks, preview, "\n".join(debug_lines))


class MMH3LastQuadrantPatch(io.ComfyNode):
    """bug_patch for '4_quadrants_expand': explicit mask control for any of the
    four extension tiles, toggled per tile on the node itself.

    Every enabled tile gets its default rects-based noise mask replaced by two
    explicit profiles: seams against the CENTER tile (tile 0) use the center_*
    widths, seams against the neighbouring corner tiles (when the enabled tile
    is sampled after them) use the edge_* widths. The per-seam fade direction
    is derived automatically, so TL/TR/BL/BR tiles all work the same way.
    Connect the output to 'MMH3 Spatial Extend Video' bug_patch. No effect on other
    layouts."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3LastQuadrantPatch",
            display_name="MMH3 Last Quadrant Patch",
            category="model/latent/minimax",
            description="bug_patch for 'MMH3 Spatial Extend Video' (4_quadrants_expand only): per-tile toggles replace any of the four extension tiles' noise masks with explicit center-seam and edge-seam frozen/fade widths in pixels. Connect to the bug_patch input.",
            search_aliases=["h3 quadrant patch", "h3 bug patch", "h3 last tile"],
            inputs=[
                io.Boolean.Input("apply_tile1", default=False,
                                 tooltip="Apply the patch to tile 1 (TL quadrant). Its only seam is against the center tile, so only the center_* widths matter."),
                io.Boolean.Input("apply_tile2", default=False,
                                 tooltip="Apply the patch to tile 2 (TR quadrant). Center seam uses center_*; its left seam against tile 1 (if any) uses edge_*."),
                io.Boolean.Input("apply_tile3", default=False,
                                 tooltip="Apply the patch to tile 3 (BL quadrant). Center seam uses center_*; its top seam against tile 1 (if any) uses edge_*."),
                io.Boolean.Input("apply_tile4", default=False,
                                 tooltip="Apply the patch to tile 4 (BR quadrant). Center seam uses center_*; its top/left seams against tiles 2 and 3 (if any) use edge_*."),
                io.Int.Input("center_mask_w_freeze", default=0, min=0, max=4096, step=32,
                             tooltip="Fully-frozen width in PIXELS along the WIDTH axis of each enabled tile's seam against the CENTER tile (frozen = mask 0, keeps the already-stitched content exactly). The center overlap rect is not necessarily square, so w and h are set separately."),
                io.Int.Input("center_mask_h_freeze", default=0, min=0, max=4096, step=32,
                             tooltip="Fully-frozen width in PIXELS along the HEIGHT axis of each enabled tile's seam against the CENTER tile."),
                io.Int.Input("center_mask_w_fade", default=32, min=0, max=4096, step=32,
                             tooltip="Fade width in PIXELS along the WIDTH axis after the frozen band at the center seams; shaped by fade_mode/flat_fade_value on the main node. frozen + fade must fit within the tile's overlap_w for the band to stay inside the seam."),
                io.Int.Input("center_mask_h_fade", default=32, min=0, max=4096, step=32,
                             tooltip="Fade width in PIXELS along the HEIGHT axis after the frozen band at the center seams; frozen + fade must fit within the tile's overlap_h."),
                io.Int.Input("edge_mask_freeze", default=0, min=0, max=4096, step=32,
                             tooltip="Fully-frozen width in PIXELS at each enabled tile's seams against the NEIGHBOURING corner tiles."),
                io.Int.Input("edge_fade", default=32, min=0, max=4096, step=32,
                             tooltip="Fade width in PIXELS after the frozen band at the neighbouring-corner seams; shaped by fade_mode/flat_fade_value on the main node."),
            ],
            outputs=[
                io.Dict.Output("bug_patch", tooltip="{'target': '4_quadrants_expand', 'tiles': {1: on, 2: on, 3: on, 4: on}, 'center_mask_w_freeze': px, 'center_mask_h_freeze': px, 'center_mask_w_fade': px, 'center_mask_h_fade': px, 'edge_mask_freeze': px, 'edge_fade': px}. Connect to 'MMH3 Spatial Extend Video' bug_patch. Ignored (with a console note) on non-4-quadrant layouts."),
            ],
        )

    @classmethod
    def execute(cls, apply_tile1=False, apply_tile2=False, apply_tile3=False,
                apply_tile4=False, center_mask_w_freeze=0, center_mask_h_freeze=0,
                center_mask_w_fade=32, center_mask_h_fade=32,
                edge_mask_freeze=0, edge_fade=32) -> io.NodeOutput:
        tiles = {1: bool(apply_tile1), 2: bool(apply_tile2),
                 3: bool(apply_tile3), 4: bool(apply_tile4)}
        if any(tiles.values()):
            print(f"[MMH3LastQuadrantPatch] patch on tiles "
                  f"{[k for k, v in tiles.items() if v]} "
                  f"(center frozen w/h {center_mask_w_freeze}/"
                  f"{center_mask_h_freeze}px, fade w/h {center_mask_w_fade}/"
                  f"{center_mask_h_fade}px, edge frozen/fade "
                  f"{edge_mask_freeze}/{edge_fade}px)")
        else:
            print("[MMH3LastQuadrantPatch] no tile enabled - the patch is a "
                  "no-op (all four apply toggles are OFF)")
        return io.NodeOutput({
            "target": "4_quadrants_expand",
            "tiles": tiles,
            "center_mask_w_freeze": int(center_mask_w_freeze),
            "center_mask_h_freeze": int(center_mask_h_freeze),
            "center_mask_w_fade": int(center_mask_w_fade),
            "center_mask_h_fade": int(center_mask_h_fade),
            "edge_mask_freeze": int(edge_mask_freeze),
            "edge_fade": int(edge_fade),
        })


class MMH3SpatialExtendVideo(io.ComfyNode):
    """Generate N side-by-side/stacked H3 AV tiles from a tile_config
    and merge them into one latent."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3SpatialExtendVideo",
            display_name="MMH3 Spatial Extend Video",
            category="model/latent/minimax",
            description=(
                "Generate MiniMax H3 AV tiles from a 'MMH3 Spatial Tile Editor' config "
                "and merge them into one latent. Connect tile_config + clip + vae; "
                "conditioning is created internally from prompts and reference images."
            ),
            search_aliases=["h3 extend", "h3 expand", "h3 outpaint", "h3 tile", "h3 merge"],
            inputs=[
                io.Model.Input("model", tooltip="The diffusion model used to sample every tile."),
                io.Dict.Input("tile_config",
                    tooltip="Output of 'MMH3 Spatial Tile Editor'. Provides per-tile prompts, reference images, dimensions, overlap/fade, and layout scheme."),
                io.Clip.Input("clip",
                    tooltip="MiniMax H3 CLIP model for encoding prompts and reference images."),
                io.Vae.Input("vae",
                    tooltip="MiniMax H3 Video VAE for encoding reference images."),
                io.Noise.Input("noise", tooltip="Noise source; one noise tensor is generated per tile."),
                io.Sampler.Input("sampler", tooltip="Sampler used for every tile."),
                io.Sigmas.Input("sigmas", tooltip="Sigma schedule used for every tile."),
                io.Sampler.Input("sampler_2nd", optional=True,
                    tooltip="EXPERIMENTAL. Sampler for a second refinement pass run on each block (from the 2nd block on) after its first pass and overlap blend. Leave unconnected to disable the second pass."),
                io.Sigmas.Input("sigmas_2nd", optional=True,
                    tooltip="EXPERIMENTAL. Sigma schedule for the second pass; supply its own scheduler/steps/denoise (e.g. a BasicScheduler). The second pass is skipped when this is unconnected, empty, or has fewer than one step (denoise 0 / steps 0)."),
                io.Float.Input("cfg", default=1.0, min=0.0, max=100.0, step=0.1, round=0.01, tooltip="CFG scale used for tiles whose negative is connected."),
                # H3_INPAINT_PARAM.Input("inpaint_param", optional=True, tooltip="Output of 'MMH3 Spatial Inpaint Params'. When connected, each tile's overlap strips are pinned to its already-sampled neighbour, removing visible seams. Leave unconnected to disable."),
                io.Latent.Input("latent_tile_0", optional=True,
                    tooltip="Existing anchor video latent. When connected, the first tile is NOT sampled - its video/audio are taken straight from this latent as an already-generated anchor and the remaining tiles are extended around it. When unconnected, tile 0 (and all tiles) are sampled normally."),
                io.Int.Input("length", default=124, min=5, max=3600, step=17,
                             tooltip="Frame count at 24 fps (124 = ~5s, trained range ~124-362). Must be 17*n+5. Overridden by latent_tile_0 frame count when it is connected."),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match",
                               tooltip="Reference image sizing. 'match' scales each ref to the generation's pixel area; 'max' uses 2048px short edge for best identity fidelity. Only used in Ref2VA mode; ignored in FL2VA."),
                io.Float.Input("masked_area_noise", default=0.0, min=0.0, max=1.0, step=0.01, round=0.01,
                               tooltip="TEST PARAM. Raises every mask value toward 1 (label, input mix and output blend together): 0 (default) keeps the frozen/fade bands as configured; 1.0 disables the mask entirely and every tile is sampled freely."),
                io.Dict.Input("bug_patch", optional=True,
                              tooltip="Layout-specific special handling. Currently: output of 'MMH3 Last Quadrant Patch' (4_quadrants_expand only) replaces the noise mask of any tile enabled on that node (four per-tile toggles) with explicit center-seam/edge-seam frozen+fade widths. Ignored (console note) on other layouts. Leave unconnected for default behavior."),
                io.Combo.Input("fade_mode", options=["flat", "gradient", "hybrid"], default="flat",
                               tooltip="How the fade band's noise mask is shaped. The H3 invariant is per-token label-input consistency: the mask value becomes each 2x2-pooled token's timestep label AND mixes that token's input (m*noise+(1-m)*anchor = anchor noised to level m*sigma). 'flat': constant flat_fade_value band - consistent at every level, empirically the most stable. 'gradient': legacy 0->1 ramp - consistent per pixel, mild mismatch at the 2x2 patch pooling (label takes the patch max); empirically close to flat. 'hybrid': same labels as 'gradient', but during sampling the masked-input mix is patched - each masked pixel's input becomes full-strength anchor + pool(m)*sigma*noise (the stock mix scales the anchor to (1-m) brightness, an out-of-distribution half-brightness signal that likely reads as the fade band's residual grain; H3's reference tokens use the full-strength construction). Experimental - A/B against 'gradient' to isolate the input-mix effect."),
                io.Float.Input("flat_fade_value", default=0.5, min=0.0, max=1.0, step=0.01,
                               tooltip="Mask value of the whole fade band (fade_mode 'flat'). LOWER = stronger freeze, more neighbour content preserved; HIGHER = more noise injected, freer generation. 0 = fully frozen band, 0.5 = half-preserved (recommended), 1 = mask off. Ignored in 'gradient' mode."),
                io.Combo.Input("ref_mode", options=["use_ref_image", "use_overlap"], default="use_ref_image",
                               tooltip="What extension tiles (tiles 1..N) condition on. The two modes are mutually exclusive. 'use_ref_image': each tile uses its own reference source from the Tile Editor under that tile's own cond_mode (FL2VA: first/last keyframes; Ref2VA: reference blocks; modes can mix per tile) - the standard behaviour. 'use_overlap': tile 0 is finalized FIRST (freely sampled, or taken from latent_tile_0 when that anchor is connected), then each extension tile's reference images are IGNORED and replaced by the part of tile 0 its frozen band covers, injected as a leading Ref2VA reference block (<Picture 1>): in serpentine layouts a single full-length edge strip; in 4_quadrants the single corner rect the tile shares with the center (full-length edge strips there would leak the neighbouring quadrants' overlap zones into the reference). The strip's frame-0 latent is sliced directly from tile 0's latent (no VAE round-trip; the Qwen3-VL vision stream sees a neutral gray placeholder). Anchors only the seam, so the rest generates freely. Strips always come from tile 0 (the only tile that exists when extension conditionings are baked); in 4+ tile serpentine layouts the bands of tiles 3+ abut tiles 1/2 instead, so their strips carry tile 0's corresponding edge as scene context rather than the exact seam. A tile with zero overlap gets no strip reference. Total sampling work is unchanged."),
                io.Combo.Input("overlap_mode", options=["earlier", "later"], default="earlier", tooltip="Who wins each shared overlap band when stitching."),
                io.Combo.Input("overlap_blend", options=["linear", "smoothstep", "overwrite", "midpoint"], default="linear", tooltip="How the overlap band transitions when stitching."),
                io.Combo.Input("second_pass_mode", options=["unmasked", "tile_mask", "shrink_32_no_fade", "shrink_64_no_fade"], default="unmasked",
                               tooltip="Second-pass masking (needs sampler_2nd/sigmas_2nd connected). 'unmasked': no noise mask - the ENTIRE tile region (overlap bands included) is sampled freely by sigmas_2nd; the blended tile latent is the starting point, so a full-strength schedule regenerates the tile while a low-denoise schedule refines it globally. 'tile_mask': the tile's baked overlap/fade mask is reused - the frozen band keeps the stitched content and only the feathered/free band is re-denoised. 'shrink_32_no_fade' / 'shrink_64_no_fade': SECOND-PASS-ONLY mask variants - each seam's frozen band is shrunk by 32/64 px toward the seam and the fade band is dropped (hard frozen/free cut, no mid-value tokens), so the released strip plus the former fade band are re-denoised from the blended latent while a reduced frozen anchor stays glued to the seam. Narrow bands (e.g. the edge overlaps of 4_quadrants) keep at least 32 px of freeze (or their full width when narrower). Audio stays frozen as in 'tile_mask'."),
            ],
            outputs=[
                io.Latent.Output("latent", tooltip="The tiles merged into one MiniMax H3 AV latent. The audio channel is tile 0's generated audio."),
                io.Latent.Output("latent_tile_0", tooltip="Tile 0's own result as a standalone MiniMax H3 AV latent. When the latent_tile_0 INPUT is connected it is passed through unchanged (bypass - the exact input object); with ref_mode 'use_overlap' it is the anchor sampled early in Phase 5.5; otherwise it is tile 0's freshly sampled video+audio before merging."),
            ],
        )

    @classmethod
    def execute(cls, model, tile_config, clip, vae,
                noise, sampler, sigmas, cfg=1.0,
                sampler_2nd=None, sigmas_2nd=None,
                latent_tile_0=None,
                length=124, ref_image_size="match", masked_area_noise=0.0,
                bug_patch=None,
                fade_mode="flat", flat_fade_value=0.5, ref_mode="use_ref_image",
                overlap_mode="earlier", overlap_blend="linear",
                second_pass_mode="unmasked") -> io.NodeOutput:
        return cls._execute_from_config(
            model, tile_config, clip, vae,
            noise, sampler, sigmas, cfg,
            sampler_2nd=sampler_2nd,
            sigmas_2nd=sigmas_2nd,
            second_pass_mode=second_pass_mode,
            latent_tile_0=latent_tile_0,
            length=length,
            ref_image_size=ref_image_size,
            masked_area_noise=masked_area_noise,
            bug_patch=bug_patch,
            fade_mode=fade_mode,
            flat_fade_value=flat_fade_value,
            ref_mode=ref_mode,
            overlap_mode=overlap_mode, overlap_blend=overlap_blend,
        )

    # ── tile_config path: create per-tile conditionings from config dict ──

    @classmethod
    def _execute_from_config(cls, model, tile_config, clip, vae,
                             noise, sampler, sigmas, cfg, **kwargs):
        """Create per-tile conditionings from tile_config and run tiling.

        Conditioning is pre-generated for all tiles before sampling begins,
        avoiding repeated CLIP/VAE encoding during the sampling loop. Tiles
        that share the same prompt, reference images, and dimensions reuse
        the same conditioning (cache deduplication).
        """
        import folder_paths
        import comfy.nested_tensor
        import comfy.utils

        from .helpers import (
            VAE_DOWNSAMPLE, is_h3_av_latent,
            normalize_minimax_refs, crop_keyframes_to_tile,
        )

        try:
            from comfy_extras.nodes_minimax_h3 import (
                _resize, align_frame_count, video_latent_t,
                CANVAS_MULTIPLE, FPS,
            )
        except ImportError:
            raise ImportError(
                "MMH3SpatialExtendVideo tile_config path requires the MiniMax H3 nodes "
                "(comfy_extras/nodes_minimax_h3.py). Please ensure ComfyUI is "
                "installed correctly."
            )

        if clip is None:
            raise ValueError("tile_config requires the 'clip' input (MiniMax H3 CLIP)")
        if vae is None:
            raise ValueError("tile_config requires the 'vae' input (MiniMax H3 Video VAE)")

        scheme = tile_config["scheme"]
        axis = tile_config.get("axis", "horizontal")
        base_prompt = tile_config.get("base_prompt", "")
        base_negative = tile_config.get("base_negative", "")
        length = kwargs.get("length", 124)
        ref_image_size = kwargs.get("ref_image_size", "match")
        tiles_cfg = tile_config["tiles"]
        overlap_mode = kwargs.get("overlap_mode", "earlier")
        overlap_blend = kwargs.get("overlap_blend", "linear")
        n = len(tiles_cfg)
        ref_mode = kwargs.get("ref_mode", "use_ref_image")
        defer_refs = (ref_mode == "use_overlap" and n > 1)

        # skip_1st_tile is now inferred: the first tile is treated as an
        # already-generated anchor exactly when latent_tile_0 is connected.
        latent_tile_0 = kwargs.get("latent_tile_0", None)
        skip_first = latent_tile_0 is not None
        if skip_first:
            samples = latent_tile_0["samples"] if isinstance(latent_tile_0, dict) else latent_tile_0
            if hasattr(samples, "tensors") and len(samples.tensors) >= 1:
                frame_count = int(samples.tensors[0].shape[2])
            elif isinstance(samples, dict) and "samples" in samples:
                frame_count = int(samples["samples"].tensors[0].shape[2])
            else:
                frame_count = align_frame_count(max(5, length))
        else:
            frame_count = align_frame_count(max(5, length))

        # ── Phase 1: Pre-load all reference images ──
        # Compose crops arrive as front-end-computed geometry boxes
        # (tile["compose_crops"], plain numbers; legacy "fl2va_crops" and
        # in-memory `compose_frames` tensors still honoured) and are cropped
        # to tensors HERE at execution time. Each tile's ref_source decides:
        # "crop" -> use its compose split region; "own" -> its own image list.
        missing_refs = []
        all_refs = []
        cf_total_w = tile_config.get("total_width") or 0
        cf_total_h = tile_config.get("total_height") or 0
        for i, tile in enumerate(tiles_cfg):
            crops = tile.get("compose_crops", tile.get("fl2va_crops")) or []
            loaded = []
            ref_source = tile.get("ref_source", "crop" if crops else "own")
            if ref_source == "crop" and crops:
                tw = tile.get("width", 512)
                th = tile.get("height", 768)
                placement = tile.get("placement") or (0, 0, 0, 0)
                for cr in crops:
                    try:
                        loaded.append(_load_compose_crop(
                            cr, tw, th, placement, cf_total_w, cf_total_h))
                    except Exception:
                        missing_refs.append(cr.get("name") or "<compose crop>")
            else:
                compose_frames = tile.get("compose_frames") or []
                loaded = list(compose_frames)
                if not loaded:
                    for ref_name in (tile.get("ref_images", []) or []):
                        if not ref_name:
                            continue
                        try:
                            img_path = folder_paths.get_annotated_filepath(ref_name)
                            loaded.append(_load_input_image(img_path))
                        except Exception:
                            missing_refs.append(ref_name)
            all_refs.append(loaded)

        if missing_refs:
            unique = list(dict.fromkeys(missing_refs))
            raise FileNotFoundError(
                f"[MMH3SpatialExtendVideo] Reference images not found in input folder: "
                f"{', '.join(unique)}"
            )

        # ── Phase 2: Create all conditionings (with cache dedup) ──
        # use_overlap defers tiles 1..n-1: their <Picture 1> is the overlap
        # strip sliced from tile 0's latent, which only exists after tile 0 is
        # sampled (Phase 5.5 below).
        cond_cache = {}
        positives = {}
        negatives = {}

        for i, tile in enumerate(tiles_cfg):
            if defer_refs and i > 0:
                continue
            prompt = tile.get("prompt", "")
            full_prompt = f"{base_prompt} {prompt}".strip() if base_prompt else prompt
            tile_neg = tile.get("negative", "")
            full_neg = f"{base_negative} {tile_neg}".strip() if base_negative else tile_neg.strip()
            w = tile.get("width", 768)
            h = tile.get("height", 544)
            # Per-tile conditioning mode: FL2VA tiles treat refs[0]/refs[1] as
            # first/last frames; Ref2VA tiles treat every ref as a reference
            # block. Tiles can mix freely within one plan.
            tile_mode = tile.get("cond_mode", "FL2VA")
            refs = all_refs[i]
            # Compose refs are per-tile in-memory image tensors (from either
            # front-end compose_crops boxes or legacy compose_frames), so they
            # are unhashable; key by tile index + ref count instead.
            cf = tile.get("compose_frames") or []
            crops = tile.get("compose_crops", tile.get("fl2va_crops")) or []
            if crops or cf:
                ref_names = tuple(f"compose:{i}:{k}" for k in range(len(refs)))
            else:
                ref_names = tuple(tile.get("ref_images", []))
            cache_key = (full_prompt, full_neg, ref_names, w, h, tile_mode, frame_count)

            if cache_key in cond_cache:
                pos_cond, neg_cond = cond_cache[cache_key]
            else:
                pos_cond = _create_conditioning(
                    clip, vae, full_prompt, w, h, frame_count, refs,
                    tile_mode, ref_image_size)
                neg_cond = None
                if full_neg:
                    neg_cond = _create_conditioning(
                        clip, vae, full_neg, w, h, frame_count, [],
                        tile_mode, ref_image_size)
                cond_cache[cache_key] = (pos_cond, neg_cond)

            positives[i] = cond_cache[cache_key][0]
            negatives[i] = cond_cache[cache_key][1]

        # ── Phase 3: Create latents ──
        latents = {}
        for i, tile in enumerate(tiles_cfg):
            w = tile.get("width", 768)
            h = tile.get("height", 544)
            latent_t = video_latent_t(frame_count)
            audio_t = round(frame_count / FPS * 40)
            device = comfy.model_management.intermediate_device()
            video = torch.zeros(1, 24, latent_t, h // 16, w // 16, device=device)
            audio = torch.zeros(1, 32, 2, audio_t, device=device)
            latents[i] = {"samples": comfy.nested_tensor.NestedTensor((video, audio))}

        # ── Phase 4: Anchor tile from latent_tile_0 when connected ──
        if skip_first:
            latents[0] = latent_tile_0

        # ── Phase 5: Collect overlap/fade per tile ──
        ol_ws, ol_hs, fws, fhs = [], [], [], []
        for i, tile in enumerate(tiles_cfg):
            ow = tile.get("overlap_w", 128)
            oh = tile.get("overlap_h", 128)
            fw_ = tile.get("fade_w", 32)
            fh_ = tile.get("fade_h", 32)
            for val, name in [(ow, "overlap_w"), (oh, "overlap_h"),
                              (fw_, "fade_w"), (fh_, "fade_h")]:
                if val % 32 != 0:
                    raise ValueError(f"tile {i} {name}={val} must be a multiple of 32")
            if fw_ > ow:
                raise ValueError(f"tile {i}: fade_w={fw_} must not exceed overlap_w={ow}")
            if fh_ > oh:
                raise ValueError(f"tile {i}: fade_h={fh_} must not exceed overlap_h={oh}")
            ol_ws.append(ow // VAE_DOWNSAMPLE)
            ol_hs.append(oh // VAE_DOWNSAMPLE)
            fws.append(fw_ // VAE_DOWNSAMPLE)
            fhs.append(fh_ // VAE_DOWNSAMPLE)

        # ── Phase 5.5: use_overlap - overlap strips as reference blocks ──
        # Finalize tile 0 (sample it now unless the user supplied it via
        # latent_tile_0), then build the deferred extension-tile
        # conditionings. Each extension tile's own reference images are
        # IGNORED (use_overlap replaces them); instead it receives ONLY the
        # overlap strip it shares with tile 0 (the strip that becomes its
        # frozen band) as its leading Ref2VA reference block - NOT tile 0's
        # full frame, which the model would otherwise treat as the scene to
        # reproduce (observed: extensions copied tile 0's subject). The
        # strip's frame-0 latent is sliced straight from tile 0's sampled
        # latent (bit-perfect, no VAE round-trip); the Qwen3-VL vision stream
        # sees a neutral gray placeholder for these blocks. _merge then
        # reuses the sampled anchor via skip_first, so total sampling work is
        # unchanged.
        if defer_refs:
            if not skip_first:
                print("[MMH3SpatialExtendVideo] use_overlap: sampling tile 0 first so "
                      "its content can anchor the extensions")
                anchor_out = _sample_anchor_tile(
                    model, noise, sampler, sigmas, negatives[0], cfg,
                    positives[0], latents[0]["samples"],
                    masked_area_noise=kwargs.get("masked_area_noise", 0.0))
                latents[0] = {"samples": anchor_out}
                skip_first = True
            anchor_video = latents[0]["samples"].tensors[0]
            # per-tile reference regions: {i: [(desc, latent_slice), ...]}
            tile_refs = {}
            for i in range(1, n):
                regions = _tile0_ref_regions(scheme, axis, i, anchor_video,
                                             ol_ws, ol_hs)
                tile_refs[i] = regions
                if not regions:
                    print(f"[MMH3SpatialExtendVideo] use_overlap: tile {i} has no "
                          f"overlap with tile 0 - no strip reference for it")
            ignored = [i for i in range(1, n) if tiles_cfg[i].get("ref_images")]
            if ignored:
                print(f"[MMH3SpatialExtendVideo] use_overlap: ignoring ref_images of "
                      f"tile(s) {ignored} (replaced by the overlap strips)")
            strip_desc = ", ".join(
                f"tile{i}=" + "+".join(desc for desc, _z in tile_refs[i])
                for i in sorted(tile_refs) if tile_refs[i])
            print(f"[MMH3SpatialExtendVideo] use_overlap: overlap regions "
                  f"(latent-direct) -> <Picture 1+>: {strip_desc}")
            for i in range(1, n):
                tile = tiles_cfg[i]
                prompt = tile.get("prompt", "")
                full_prompt = f"{base_prompt} {prompt}".strip() if base_prompt else prompt
                tile_neg = tile.get("negative", "")
                full_neg = f"{base_negative} {tile_neg}".strip() if base_negative else tile_neg.strip()
                w = tile.get("width", 768)
                h = tile.get("height", 544)
                sides_sig = tuple(desc for desc, _z in tile_refs[i])
                cache_key = (full_prompt, full_neg, w, h, "Ref2VA+overlap",
                             sides_sig, frame_count)
                if cache_key in cond_cache:
                    pos_cond, neg_cond = cond_cache[cache_key]
                else:
                    latent_refs = [
                        {"kind": "image",
                         "latent_h": int(z.shape[3]),
                         "latent_w": int(z.shape[4]),
                         "latent": z}
                        for (_desc, z) in tile_refs[i]]
                    pos_cond = _create_ref2va_conditioning(
                        clip, vae, full_prompt, w, h, frame_count,
                        [], ref_image_size, latent_refs=latent_refs)
                    neg_cond = None
                    if full_neg:
                        neg_cond = _create_conditioning(
                            clip, vae, full_neg, w, h, frame_count, [],
                            "Ref2VA", ref_image_size)
                    cond_cache[cache_key] = (pos_cond, neg_cond)
                positives[i] = cond_cache[cache_key][0]
                negatives[i] = cond_cache[cache_key][1]
        elif ref_mode == "use_overlap":
            print("[MMH3SpatialExtendVideo] use_overlap: single tile, no extensions "
                  "to reference - mode ignored")

        if cond_cache:
            saved = n - len(cond_cache)
            msg = f"[MMH3SpatialExtendVideo] Conditioning: {n} tiles, {len(cond_cache)} unique"
            if saved > 0:
                msg += f" (cached {saved})"
            print(msg)

        # ── Phase 6: Validate and merge ──
        tile_plan = {"scheme": scheme, "axis": axis}
        idxs = sorted(range(n))

        if scheme == "4_quadrants_expand":
            if n != 5:
                raise ValueError("4_quadrants_expand needs exactly 5 tiles, "
                                 f"got {n}")
        else:
            if axis == "horizontal":
                heights = [latents[i]["samples"].tensors[0].shape[3] for i in idxs]
                if len(set(heights)) > 1:
                    raise ValueError("horizontal scheme requires every tile "
                                     f"to share the same height (got {sorted(set(heights))})")
            else:
                widths = [latents[i]["samples"].tensors[0].shape[4] for i in idxs]
                if len(set(widths)) > 1:
                    raise ValueError("vertical scheme requires every tile "
                                     f"to share the same width (got {sorted(set(widths))})")

        audio_src = latents[0]["samples"].tensors[1]
        tconds = [positives[i] for i in idxs]
        negative_list = [negatives[i] for i in idxs]
        video_list = [latents[i]["samples"] for i in idxs]
        T0 = video_list[0].tensors[0].shape[2]

        bug_patch = kwargs.get("bug_patch")
        if bug_patch is not None and scheme != "4_quadrants_expand":
            print(f"[MMH3SpatialExtendVideo] bug_patch target "
                  f"'{bug_patch.get('target', '?')}' does not apply to scheme "
                  f"'{scheme}' - ignored")
            bug_patch = None

        if scheme == "4_quadrants_expand":
            out_v, out_a, tiles_info, t0 = _merge_2d(
                model, noise, sampler, sigmas, negative_list, cfg,
                audio_src, tconds, video_list, tile_plan,
                ol_ws, ol_hs, fws, fhs,
                overlap_mode, overlap_blend,
                skip_first=skip_first,
                masked_area_noise=kwargs.get("masked_area_noise", 0.0),
                bug_patch=bug_patch,
                fade_mode=kwargs.get("fade_mode", "flat"),
                fade_val=kwargs.get("flat_fade_value", 0.5),
                sampler_2nd=kwargs.get("sampler_2nd"),
                sigmas_2nd=kwargs.get("sigmas_2nd"),
                second_pass_mode=kwargs.get("second_pass_mode", "unmasked"),
            )
        else:
            out_v, out_a, tiles_info, t0 = _merge(
                model, noise, sampler, sigmas, negative_list, cfg,
                audio_src, tconds, video_list, tile_plan,
                ol_ws, ol_hs, fws, fhs,
                overlap_mode, overlap_blend,
                skip_first=skip_first,
                masked_area_noise=kwargs.get("masked_area_noise", 0.0),
                fade_mode=kwargs.get("fade_mode", "flat"),
                fade_val=kwargs.get("flat_fade_value", 0.5),
                sampler_2nd=kwargs.get("sampler_2nd"),
                sigmas_2nd=kwargs.get("sigmas_2nd"),
                second_pass_mode=kwargs.get("second_pass_mode", "unmasked"),
            )

        out = {"samples": comfy.nested_tensor.NestedTensor((out_v, out_a))}

        # Tile 0's standalone result. When the latent_tile_0 INPUT supplied the
        # anchor, Phase 4 assigned it straight into latents[0], so this returns
        # the exact input object (true bypass). With use_overlap the anchor was
        # sampled early (Phase 5.5) and already lives in latents[0]. Otherwise
        # tile 0 was sampled inside _merge/_merge_2d and t0 carries its raw
        # video+audio output.
        if skip_first or defer_refs:
            tile0_out = latents[0]
        elif t0 is not None:
            tile0_out = {"samples": t0}
        else:
            tile0_out = out
        return io.NodeOutput(out, tile0_out)


# ── Helpers for the tile_config path ──

def _load_input_image(filename):
    """Load an image from the ComfyUI input folder into a [1, H, W, 3] tensor."""
    from PIL import Image as PILImage
    from PIL import ImageOps as PILImageOps
    import numpy as np
    img = PILImage.open(filename).convert("RGB")
    # Honour EXIF orientation (matches the browser preview thumbnails).
    img = PILImageOps.exif_transpose(img)
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)  # [1, H, W, 3]


def _load_compose_crop(crop, tw, th, placement=(0, 0, 0, 0), total_w=0, total_h=0):
    """Crop a whole-frame compose source into a [1, H, W, 3] tensor.

    crop is {kind: "first"|"last", name: <input file>, box: [L, T, R, B]} in
    source-image pixels, normally pre-computed by the Tile Editor JS to match
    its canvas preview exactly. Because the front-end only knows the image
    size after an async decode, a box may be absent (or s/ox/oy only); in that
    case the box is derived here after opening the image: s scales the whole
    canvas to fit, and ox/oy CENTRE it (never anchored at 0,0 -- that was the
    old "crop from top-left" bug). placement is the tile's rect on the layout
    canvas; total_w/total_h the whole canvas size. The cropped region feeds
    whichever conditioning mode the tile selected (FL2VA frame or Ref2VA
    block).
    """
    from PIL import Image as PILImage
    from PIL import ImageOps as PILImageOps
    import numpy as np
    import folder_paths
    name = crop.get("name")
    if not name:
        raise ValueError(f"fl2va crop missing name: {crop!r}")
    path = folder_paths.get_annotated_filepath(name)
    img = PILImage.open(path).convert("RGB")
    # Honour EXIF orientation so pixel coords match what the browser
    # preview shows (browsers apply orientation on decode, PIL does not).
    img = PILImageOps.exif_transpose(img)
    iw, ih = img.size

    box = crop.get("box")
    if not (box and len(box) == 4):
        # Front-end did not ship a resolved box -> derive it now from the
        # actual image size + placement (+ optional s/ox/oy, else fit+centre).
        px, py, pw, ph = placement
        s = crop.get("s")
        if not s:
            if total_w > 0 and total_h > 0:
                s = min(iw / total_w, ih / total_h)
            else:
                s = 1.0
        ox = crop.get("ox")
        oy = crop.get("oy")
        if ox is None:
            ox = (iw - total_w * s) / 2 if total_w > 0 else 0.0
        if oy is None:
            oy = (ih - total_h * s) / 2 if total_h > 0 else 0.0
        box = [ox + px * s, oy + py * s,
               ox + (px + pw) * s, oy + (py + ph) * s]

    L, T, R, B = [int(v) for v in box]
    # Clamp the window POSITION (not its size): narrowing the box here
    # made the resize stretch the content, so the generated first frame
    # drifted from the editor preview ("offset not applied" bug).
    bw, bh = R - L, B - T
    L = max(0, min(L, iw - bw)); R = L + bw
    T = max(0, min(T, ih - bh)); B = T + bh
    if R - L <= 0 or B - T <= 0:
        raise ValueError(f"empty fl2va crop box {box} for '{name}'")
    c = img.crop((L, T, R, B))
    c = c.resize((tw, th), PILImage.LANCZOS)
    arr = np.array(c).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)  # [1, H, W, 3]


def _tile0_ref_regions(scheme, axis, i, anchor_video, ol_ws, ol_hs):
    """Reference latent regions of tile 0 for extension tile i, as a list of
    (description, frame-0 latent slice). Empty list = no reference.

    Serpentine layouts: odd tiles grow toward +1 (right/down), even tiles
    toward -1; each gets ONE full-length edge strip (depth = that tile's own
    overlap on the facing axis).

    4_quadrants: tiles 1..4 are TL/TR/BL/BR around the center tile 0; each
    gets ONLY the CORNER rect it shares with tile 0 (depths = its own
    overlap_h x overlap_w) - exactly the content its frozen band covers.
    Full-length edge strips would leak the parts of tile 0 that belong to the
    NEIGHBOURING quadrants' overlap zones into the reference."""
    H, W = int(anchor_video.shape[3]), int(anchor_video.shape[4])
    if scheme == "4_quadrants_expand":
        corner = {1: "tl", 2: "tr", 3: "bl", 4: "br"}.get(i)
        if corner is None:
            return []
        dh = max(0, min(int(ol_hs[i]), H))
        dw = max(0, min(int(ol_ws[i]), W))
        if dh == 0 or dw == 0:
            return []
        if corner == "tl":
            z = anchor_video[:, :, :, :dh, :dw]
        elif corner == "tr":
            z = anchor_video[:, :, :, :dh, W - dw:]
        elif corner == "bl":
            z = anchor_video[:, :, :, H - dh:, :dw]
        else:
            z = anchor_video[:, :, :, H - dh:, W - dw:]
        return [(f"{corner} corner {dh}x{dw}tok ({dh * 16}x{dw * 16}px)",
                 z[:, :, :1].contiguous())]
    if axis == "vertical":
        side = "bottom" if i % 2 == 1 else "top"
    else:
        side = "right" if i % 2 == 1 else "left"
    depth = ol_ws[i] if side in ("left", "right") else ol_hs[i]
    limit = W if side in ("left", "right") else H
    depth = max(0, min(int(depth), int(limit)))
    if depth == 0:
        return []
    if side == "left":
        z = anchor_video[:, :, :, :, :depth]
    elif side == "right":
        z = anchor_video[:, :, :, :, -depth:]
    elif side == "top":
        z = anchor_video[:, :, :, :depth, :]
    else:
        z = anchor_video[:, :, :, -depth:, :]
    return [(f"{side} strip {depth}tok ({depth * 16}px)",
             z[:, :, :1].contiguous())]


def _sample_anchor_tile(model, noise, sampler, sigmas, negative, cfg, cond,
                        samples, masked_area_noise=0.0):
    """Freely sample tile 0 alone, mirroring _merge's i==0 piece exactly.

    Used by ref_mode 'use_overlap': the anchor is sampled before the extension
    tiles exist so its overlap strips can be sliced and injected as their
    <Picture 1>. _merge later reuses the result via skip_first and never
    resamples it, so the total sampling work is unchanged."""
    video, audio = samples.tensors
    th, tw = video.shape[3], video.shape[4]
    m = _edge_fade_mask(th, tw, 0, 0, 0, 0, False, False, False, False)  # all ones
    m_v = m + masked_area_noise * (1.0 - m)
    mv = m_v[None, None, None]
    ma = torch.ones((1, 32, 2, audio.shape[-1]), device=audio.device, dtype=audio.dtype)
    piece = {
        "samples": samples,
        "noise_mask": comfy.nested_tensor.NestedTensor((mv, ma)),
    }
    return sample_piece(piece, cond, model, noise, sampler, sigmas, negative, cfg)


def _create_conditioning(clip, vae, prompt, w, h, frame_count, ref_images,
                         cond_mode, ref_image_size="match"):
    """Create conditioning for a tile in the specified mode.

    cond_mode "FL2VA": ref_images[0] -> first_frame, ref_images[1] -> last_frame.
    cond_mode "Ref2VA": each ref_image becomes a reference block.
    ref_image_size: "match" or "max" for Ref2VA resize strategy.
    """
    import math
    try:
        from comfy_extras.nodes_minimax_h3 import (
            _resize, _empty_av_latent, CANVAS_MULTIPLE, FPS,
        )
    except ImportError:
        raise ImportError("tile_config conditioning requires comfy_extras/nodes_minimax_h3.py")

    if cond_mode == "FL2VA":
        return _create_fl2va_conditioning(
            clip, vae, prompt, w, h, frame_count, ref_images)
    else:
        return _create_ref2va_conditioning(
            clip, vae, prompt, w, h, frame_count, ref_images, ref_image_size)


def _create_fl2va_conditioning(clip, vae, prompt, w, h, frame_count, ref_images):
    """Create conditioning in FL2VA (ImageToVideo) mode.

    ref_images[0] -> first_frame, ref_images[1] -> last_frame (if present).
    """
    try:
        from comfy_extras.nodes_minimax_h3 import (
            _resize, _empty_av_latent,
        )
    except ImportError:
        raise ImportError("FL2VA mode requires comfy_extras/nodes_minimax_h3.py")

    latent_dict, _ = _empty_av_latent(w, h, frame_count)

    images = []
    keyframes = []
    if len(ref_images) >= 1:
        img = _resize(ref_images[0][:1], w, h, "disabled")
        images.append(img)
        keyframes.append({"resolved_frame_index": 0, "image": img})
    if len(ref_images) >= 2:
        img = _resize(ref_images[1][:1], w, h, "center")
        images.append(img)
        keyframes.append({"resolved_frame_index": frame_count - 1, "image": img})

    tokens = clip.tokenize(prompt, images=images)
    cond = clip.encode_from_tokens_scheduled(tokens)

    if keyframes:
        for kf in keyframes:
            kf["latent"] = vae.encode(kf.pop("image"))
        import node_helpers
        cond = node_helpers.conditioning_set_values(
            cond, {"minimax_keyframes": keyframes})

    return cond


def _create_ref2va_conditioning(clip, vae, prompt, w, h, frame_count, ref_images,
                                ref_image_size="match", latent_refs=None):
    """Create conditioning in Ref2VA (ReferenceToVideo) mode.

    Each image in ref_images becomes a reference block. Prompt should use
    <Picture i> tags to reference them. ref_image_size: "match" scales each ref
    to the generation's pixel area; "max" uses 2048px short edge for best
    identity fidelity.

    latent_refs: pre-built reference blocks ({"kind": "image", "latent_h",
    "latent_w", "latent"}) passed through to the model WITHOUT any VAE
    round-trip. The DiT side of a reference block consumes the raw latent
    directly, so this is bit-perfect; only the Qwen3-VL vision stream needs
    pixels, and it gets a neutral gray placeholder for these blocks (the
    tokenizer still emits the "<Picture N>: " intro so the text structure
    matches training). Blocks come first, so latent_refs[0] is <Picture 1>."""
    import math
    try:
        from comfy_extras.nodes_minimax_h3 import (
            _resize, _empty_av_latent, CANVAS_MULTIPLE,
        )
    except ImportError:
        raise ImportError("Ref2VA mode requires comfy_extras/nodes_minimax_h3.py")

    latent_dict, _ = _empty_av_latent(w, h, frame_count)

    ref_items = []
    ref_blocks = []

    for blk in latent_refs or ():
        ref_items.append({"type": "image", "data": torch.full((1, 64, 64, 3), 0.5)})
        ref_blocks.append(dict(blk))

    for img in ref_images:
        ih, iw = img.shape[1], img.shape[2]
        if ref_image_size == "max":
            scale = min(1.0, 2048 / min(iw, ih))
        else:
            scale = min(1.0, math.sqrt((w * h) / (iw * ih)))
        tw = max(CANVAS_MULTIPLE, round(iw * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        th = max(CANVAS_MULTIPLE, round(ih * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        resized = _resize(img[:1], tw, th, "disabled")
        z = vae.encode(resized)
        ref_items.append({"type": "image", "data": resized})
        ref_blocks.append({
            "kind": "image",
            "latent_h": th // 16, "latent_w": tw // 16,
            "latent": z,
        })

    tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
    cond = clip.encode_from_tokens_scheduled(tokens)
    if ref_blocks:
        import node_helpers
        cond = node_helpers.conditioning_set_values(
            cond, {"minimax_refs": ref_blocks})

    return cond
