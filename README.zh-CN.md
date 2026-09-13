# Comfyui-MMH3-UltimateExtend

ComfyUI 上的 **MiniMax H3** 视频生成扩展节点。

> **开发中。** 节点的插槽、参数与落盘格式仍可能随提交变化。

[English](README.md) · [中文](README.zh-CN.md)

## MMH3 Temporal Extender（时序扩展）

把一条长视频拆成**分段（segment）**来生成：每一段接续上一段留下的尾部，而不是从头重来。
核心是 `MMH3 Temporal Extend Video` 节点与 `MMH3 Temporal Tile Editor` 编辑面板。

### 1. 多轮接续，每轮条件独立可调 —— 方便"抽卡"与回滚

- 整条链在编辑器里就是一张分段列表，每段都有**自己的**条件：提示词与负面提示词、种子、段长、
  携带尾部（overlap）长度、参考图及其尺寸策略、参考视频 / 参考音频模式、首帧或参考关键帧，
  以及该段自己的 extend 参数。
- 每段单独采样、单独落盘。改一个种子（或提示词、参考）**只重跑那一段** —— 这就是"抽卡"。
- 每次执行都会为该段写出**一次新的 attempt**，旧结果永不覆盖。回滚就是解锁那一段重跑，
  链上其余段落原样复用（历史 attempt 仍在账本里）。
- 段落可以锁定；已锁定的段落直接从磁盘读取，不会再被采样。

### 2. Session 保存任务 —— 关机后继续上次的接续

- 一次运行的全部状态都在一个 session 目录里（`mmh3_temporal/<session_name>/`）：
  `session.json` 账本 + `latents/` + `previews/`。
- 账本记录每段选中的 attempt、切分点与运行历史，所以**重启 ComfyUI（甚至重新打开工作流）**后，
  把 `resume_from_segment` 指向第一个未完成的段落就能接着做，中间可以关机。
- 编辑器的分段状态也一并持久化进 session，面板重新打开时自动还原。
- 存储位置可选 ComfyUI 临时目录，或 `output/latents`（更持久）。

### 3. 完善的缓存 —— 不重复跑没必要的 VAE 与 conditioning

- 一切昂贵的部分都做内容寻址缓存：参考图 / 参考音频的 VAE latent，以及"提示词 + 一组参考"
  编码出的 conditioning。
- 缓存键包含**模型指纹**（类名、tokenizer 路径、参数量）、**源身份**（路径 + mtime + size，
  或像素 / 波形哈希）以及全部编码参数，因此可以跨段、跨运行、跨 session 复用。
- 输入变了自然 miss，不需要手动清缓存；插件自身行为变化由缓存格式版本号兜底。
- 背景音乐输入同样走缓存，并且**按段惰性编码** —— 只有当前段落真正用到的那一段才会进入
  音频 VAE，长歌不会一次性吃满显存。

### 4. 丰富的接缝设置 —— 从追求连续到避免劣化

- 完整的接缝旋钮：重叠带归谁、混合曲线、seam anchor 及其强度、`prev tail` 接缝参考
  （走 H3 原生的参考视频通路）、关键帧模式、尾部模式（`freeze + fade` 或整个尾部重采样）、
  淡入淡出长度与形状、掩码强度等。
- `MMH3 Temporal Overlap Simple` 把这些收成一键预设：**high continuity**（高连续）、
  **low deterioration**（低劣化）、**soft restart**（软重启）、**H3 video reference**。
- 可选第二遍 refinement pass，用独立的低降噪调度对整段重新降噪（不带噪声掩码、不带关键帧，
  参考仍然生效）。

### 节点

| 节点 | 作用 |
|---|---|
| MMH3 Temporal Tile Editor | 编辑分段链：条件、参考、逐段 extend 参数，输出 config |
| MMH3 Temporal Extend Video | 执行采样；输出 `merged_latent`、`segment_latent`（最后一段）与 `segment_info` |
| MMH3 Sample Params | 两段式采样参数：HIGH（加噪）＋ 可选 LOW |
| MMH3 Temporal Overlap Params | 手工调节的接缝参数 |
| MMH3 Temporal Overlap Simple | 四个一键接缝预设 |
| MMH3 Latent Preview (approx) | H3 latent 的快速近似预览 |

## Spatial Extender（空间扩展）

`MMH3 Spatial Extend Video` 节点仍在开发中。
