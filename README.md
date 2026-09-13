# Comfyui-MMH3-UltimateExtend
Extend MiniMax H3 video in both spatial and temporal dimension.  
This node is under development...

> **Work in progress.** Node sockets, parameters and on-disk formats may still change between commits.

[English](README.md) · [中文](README.zh-CN.md)

## MMH3 Temporal Extender

Generates one long video as a chain of **segments**, where each segment continues the previous
one's tail instead of starting over. Built around the `MMH3 Temporal Extend Video` node and the
`MMH3 Temporal Tile Editor` panel.

### 1. Multi-round continuation, with conditions per round - easy re-rolls and rollbacks

- The chain is edited as a list of segments. Each segment carries its **own** conditions: prompt
  and negative prompt, seed, length, carried-tail (overlap) length, reference images and their
  sizing, reference video / audio mode, first or reference keyframes, and its own extend parameters.
- Every segment is sampled and saved on its own. Change one seed - or prompt, or reference - and
  **only that segment re-samples**: that is the "re-roll".
- Each execution writes a **new attempt** for its segment; earlier attempts are never overwritten.
  Rolling back is unlocking that segment and running it again, while the rest of the chain is
  reused exactly as it is.
- Segments can be locked. A locked segment is read back from disk and is never re-sampled.

### 2. Sessions - shut the machine down, resume the chain later

- The whole state of a run lives in one session folder (`mmh3_temporal/<session_name>/`): a
  `session.json` ledger plus `latents/` and `previews/`.
- The ledger maps every segment to its chosen attempt, its split point and its run history. After a
  ComfyUI restart - or after reloading the workflow - point `resume_from_segment` at the first
  unfinished segment and carry on.
- Editor state is persisted into the session as well and restored when the panel reopens.
- Storage can be the ComfyUI temp directory, or `output/latents` for a longer-lived session.

### 3. Caching - no wasted VAE or conditioning passes

- Everything expensive is content-addressed: the VAE latents of reference images and reference
  audio, and the encoded conditioning of a prompt plus a set of references.
- The key holds the **model fingerprint** (class names, tokenizer path, parameter count), the
  **source identity** (path + mtime + size, or a pixel / waveform hash) and every encoding
  parameter, so entries are reused across segments, runs and sessions.
- Change an input and the entry simply misses; nothing has to be cleared by hand. Changes in the
  plugin's own behaviour are covered by a cache format version.
- The background-music input is cached the same way and encoded **one segment at a time**, so only
  the span a segment actually uses ever reaches the audio VAE.

### 4. Seam controls - from maximum continuity to minimum deterioration

- A full set of seam knobs: who wins the overlap band, the blend curve, the seam anchor and its
  strength, the `prev tail` seam reference (H3's native reference-video path), the keyframe mode,
  the tail mode (`freeze + fade`, or re-sample the whole tail), fade length and shape, mask
  strength, and more.
- `MMH3 Temporal Overlap Simple` collapses them into one-click presets: **high continuity**,
  **low deterioration**, **soft restart** and **H3 video reference**.
- An optional second refinement pass can re-denoise a whole segment with its own low-denoise
  schedule (no noise mask, no keyframes, references still active).

### Nodes

| Node | Role |
|---|---|
| MMH3 Temporal Tile Editor | Builds the segment chain - conditions, references, per-segment extend parameters - and outputs the config |
| MMH3 Temporal Extend Video | Runs the sampling; outputs `merged_latent`, `segment_latent` (the last segment) and `segment_info` |
| MMH3 Sample Params | Two-stage sampling bundle: a HIGH stage (noise injection) plus an optional LOW stage |
| MMH3 Temporal Overlap Params | Hand-tuned seam parameters |
| MMH3 Temporal Overlap Simple | The four one-click seam presets |
| MMH3 Latent Preview (approx) | Fast approximate preview of an H3 latent |

## Spatial Extender

`MMH3 Spatial Extend Video` is still under development.

Without this node:  
![1](assets/normal.webp)

With this node:  
![1](assets/with_this_nodes.webp)
