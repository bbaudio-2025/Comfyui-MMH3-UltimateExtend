// Dock-panel visual editor for the "MMH3 Temporal Tile Editor" node.
//
// Companion of nodes/temporal_tile_editor.py (pure-information node): all
// editor state is serialized into the hidden `tile_data` JSON string widget
// and parsed/normalized server-side by the python node.
//
// The UI mirrors the MMH3 Spatial Tile Editor dock: a floating window pinned
// to the screen (drag by its head, resize from the edges, zoom, minimize,
// close), with the same dark palette and blue accent.
//
// Features:
//  - per-segment cards: conditioning mode (segment 0: FL2VA / Ref2VA; later:
//    Ref2VA / FL2VA-last-frame-only), prompt, negative, seed, duration
//    (seconds <-> frames, snapped to the model's 17n+5 / 17n grids),
//    references (manual input images / previous-segment frame / none),
//    MiniMaxH3ReferenceToVideo-style extra references (ref video /
//    video audio / ref audio)
//  - segment 0 owns the chain resolution; it is LOCKED once the session
//    directory already stores latents (they must stitch seamlessly)
//  - preview panel: shows stored per-segment previews from the session
//    directory (temp or output/latents) and lets you set the resume point -
//    "re-run from segment n" reuses the stored merged latent of segment n-1
import { app } from "../../scripts/app.js";

const TARGET = "MMH3TemporalTileEditor";
const DATA_WIDGET = "tile_data";
const TAG = "[MMH3-TemporalTileEditor]";
const FPS = 24;

// ── i18n: dock-panel UI ──
// Same approach as the spatial editor: nodeDefs.json (locales/zh) translates
// the node-definition text; this block handles the dock-panel strings. Keys
// are the exact English literals.
const ZH_UI = {
    // Dock head / toolbar
    "MMH3 Temporal Tile Editor": "MMH3 时序分段编辑器",
    "Minimize / restore": "最小化 / 还原",
    "Zoom out": "缩小",
    "Zoom in": "放大",
    "Click to reset zoom": "点击重置缩放",
    "Close editor": "关闭编辑器",
    "Session:": "会话:",
    "refresh previews": "刷新预览",
    "restore from session": "从会话还原",
    "confirm restore": "确认还原",
    "no saved editor state in this session": "该会话没有已保存的编辑器状态",
    "editor state restored": "已从会话还原编辑器状态",
    "restore replaces current state": "还原将用会话中保存的分段设置替换当前全部分段，未保存的修改会丢失。确定继续？",
    "add": "添加",
    "random seed per segment": "每个分段随机种子",
    "Segments": "分段列表",
    "Previews / Resume": "预览 / 续跑",
    "no stored run yet for this session — previews appear after executing MMH3 Temporal Extend Video":
        "本会话还没有已存储的运行记录 — 执行 MMH3 Temporal Extend Video 后此处显示预览",
    "loading previews…": "正在加载预览…",
    // Segment cards
    "seg 0 (first)": "分段 0（首段）",
    "seg": "分段",
    "del": "删除",
    "delete segment": "删除分段",
    "later segment(s)": "个后续分段",
    "stored results become invalid": "其自身及其后所有分段已生成的采样结果将全部失效",
    "locked segments' stored latents are missing — unlocked automatically":
        "锁定分段对应的 latent 文件已不存在——已自动解锁",
    "cancel": "取消",
    "confirm delete": "确认删除",
    "lock": "锁定此段及其之前的所有段（运行时从磁盘复用，从此段之后接续采样）",
    "unlock": "解除锁定（恢复从分段 0 重新生成）",
    "not sampled yet": "该分段尚未完成采样，无法锁定",
    "resume here": "下次运行从此分段开始采样",
    "resume from segment": "从分段",
    "duration": "时长",
    "sec": "秒",
    "frames": "帧",
    "edit in seconds": "按秒编辑",
    "edit in frames": "按帧编辑",
    "frames (17n+5)": "帧（17n+5）",
    "new frames (×17)": "新增帧（×17）",
    "seconds → frames (17n+5)": "秒 → 帧（首段格点 17n+5）",
    "seconds → frames (×17)": "秒 → 帧（接续段格点 ×17）",
    "seed": "种子",
    "prompt": "正向提示词",
    "segment prompt": "分段提示词",
    "negative": "负向提示词",
    "negative (optional)": "负向提示词（可选）",
    "resolution": "分辨率",
    "multiple of 32 · locked once latents exist": "32 的倍数 · 生成过片段后锁定",
    "first frame": "首帧",
    "last frame": "尾帧",
    "last frame (target ending)": "尾帧（目标结尾）",
    "image reference mode": "图像参考模式",
    "no reference": "无参考",
    "load images": "加载图像",
    "previous segment frame": "上一分段帧",
    "ref images": "参考图像",
    "add image…": "添加图像…",
    "(none)": "（无）",
    "(no image)": "（输入文件夹中无图像）",
    "remove image": "移除该图像",
    "(no longer in input folder)": "（已不在输入文件夹中）",
    "(wired image)": "（已连入的图像）",
    "ref size": "参考尺寸",
    "match": "匹配",
    "max": "最大",
    "ref video": "参考视频",
    "video audio": "视频音轨",
    "audio reference mode": "音频参考模式",
    "load audio": "加载音频",
    "previous audio": "上一段音频",
    "initial audio": "开场音频",
    "auto crop bgm": "自动裁切背景音乐",
    "bgm range": "音乐区间",
    "re-covers the previous": "重覆盖前段尾部",
    "the carried tail covers the whole chain - the music restarts at the beginning of the song":
        "携带尾部已覆盖整条链——音乐将从歌曲开头重新开始",
    "ref audio": "参考音频",
    "ref video mode": "参考视频模式",
    "load video": "加载视频",
    "auto crop input ref": "自动裁切输入参考视频",
    "auto crop range": "裁切区间",
    "slots into the wired ref_video_input": "对应已连入的 ref_video_input",
    "a video must be connected to the node's ref_video_input input":
        "需将视频连接到该节点的 ref_video_input 输入",
    // fun control (MiniMax H3 Fun ControlNet)
    "fun control (this segment)": "Fun ControlNet（仅本段生效）",
    "control mode": "控制模式",
    "no control": "不启用控制",
    "auto crop input control": "自动裁切输入控制视频",
    "whole control video": "整个控制视频",
    "control strength": "控制强度",
    "0 disables the control for this segment": "0 = 本段不施加控制",
    "control start": "起始比例",
    "control end": "结束比例",
    "control range": "控制帧区间",
    "slots into the wired fun control video": "对应已连入的 fun control video",
    "a video must be connected to the node's fun_control_video input":
        "需将视频连接到该节点的 fun_control_video 输入",
    "control applies from the control video's first frame": "从控制视频第 0 帧开始取",
    "0–1 sigma percentage of the model's own schedule (native node semantics)":
        "0–1 的比例，相对模型自身的 sigma 表（与原生节点一致）",
    "the carried tail covers the whole chain - the control restarts at the beginning of the video":
        "携带尾部已覆盖整条链——控制视频将从开头重新开始",
    "frame #": "帧序号",
    "-1 = last frame of previous merged video (latent-exact, no decode)":
        "-1 = 上一合并视频的最后一帧（latent 精确，不解码）",
    "+ add segment": "+ 添加分段",
    "segment(s)": "个分段",
    "total": "总计",
    "total ≈": "总时长 ≈",
    "cumulative frames up to this segment": "累积帧数（到本段为止）",
    // Modes
    "FL2VA": "FL2VA（首/尾帧）",
    "Ref2VA": "Ref2VA（参考图像）",
    // Previews
    "merged through segment": "合并至该分段",
    "merged": "拼接结果",
    "segment alone": "仅该分段",
    "Close preview": "关闭预览",
    "resume point: segment": "续跑点：分段",
    "unlock all": "解锁全部",
    "will reuse segments": "将复用分段",
    "all segments locked — unlock one or add a new segment": "所有分段已锁定，没有待采样分段——请解锁一段或添加新分段",
    "continuation parameters (this segment)": "接续参数（仅本段生效）",
    "overlap (this segment)": "重叠参数（仅本段生效）",
    "overlap frames": "重叠帧数",
    "carried tail (17n+5)": "接续的尾部（17n+5）",
    "carried tail in pixel frames, snapped up to the 5, 22, 39, ... grid": "接续尾部长度（像素帧），向上对齐到 5, 22, 39, ... 网格",
    "hard cut (0)": "硬切（0，接缝不淡出）",
    "no fade and no blend across the seam: the segment starts fresh at the boundary and is stitched right after the previous one. The token grid gives every piece a minimum seam of 5 frames, and on a hard cut those 5 are frozen verbatim - they re-play frames the video already showed, so the cut lands exactly at the boundary and nothing at it is visible": "接缝处不做淡出、不做混合：该段从边界处重新开始，直接接在前一段之后。令牌网格给每段的接缝长度设了 5 帧下限；硬切时这 5 帧被逐字节冻结（重放视频里已经出现过的画面），所以剪切点正好落在边界上，接缝处看不出任何变化",
    "from disk": "及之前（从磁盘读取）",
    // Shared reference cache (mmh3_temporal/cache)
    "reference cache": "参考缓存",
    "clear cache": "清除缓存",
    "reference latents + encoded conditioning kept under mmh3_temporal/cache — shared by every session of this storage location": "缓存在 mmh3_temporal/cache 下（参考 latent 与已编码的 conditioning），同一存储位置的所有会话共享",
    "drop every cached reference latent and conditioning (the next run re-encodes them)": "清除全部缓存的参考 latent 与 conditioning（下次运行会重新编码）",
    // Clear-unused-files
    "clear unused latents": "清除无用 latent",
    "delete stored latent & preview files not referenced by the current chain": "删除当前链未引用的已存 latent 与 preview 文件",
    "deletes attempt files neither shown as segment/merged previews nor mapped by session.json": "删除未显示在当前分段/拼接预览中、也未被 session.json 引用的已存尝试文件",
    "stored file(s) not referenced by the current chain will be permanently deleted": "个未被当前链引用的已存文件将被永久删除",
    "this cannot be undone, continue?": "此操作不可撤销，是否继续？",
    "confirm clear": "确认清除",
    // Direct reference in frame (the reference video's pixels are spliced into
    // this segment's SAMPLING canvas and frozen, per segment; the strip is cut
    // back off after sampling, so nothing of it reaches the output)
    "direct reference in frame": "direct reference in frame（帧内直引）",
    "splice side": "拼接方向",
    "material left": "左侧",
    "material right": "右侧",
    "material top": "上方",
    "material bottom": "下方",
    "where the strip of the ref_video_input video sits on this segment's canvas - 'off' = this segment does not use the mode":
        "ref_video_input 的画面条贴在本段采样画布的哪一侧 —— 选“不启用素材”即本段不使用该模式",
    "expose at the seam": "接缝暴露宽度",
    "free band at the seam (px, multiple of 32 = one token column)": "接缝处留给模型重绘的带宽（像素，32 的倍数 = 一个 token 列）",
    "the rest of the strip stays frozen; this band lets the model synthesize the transition":
        "画面条的其余部分保持冻结；这条带宽让模型自己合成过渡",
    "cut at the split": "在 split 处裁掉画面条",
    "drop the strip when the HIGH stage hands over to the LOW stage: the reference builds the structure, then the detail stage finishes the picture from the prompt alone - nothing of the strip reaches the output either way. The refinement pass never sees the strip anyway (it always cuts it first), so this switch is about the FIRST pass only. Needs a LOW stage (a split to cut at) and no fun control on this segment; the node prints which way it went":
        "HIGH 阶段交给 LOW 阶段时把画面条裁掉：参考负责建立结构，细节阶段只由提示词完成 —— 两种做法的输出都不含画面条。第二遍 refinement 本来就不会看到画面条（它在开始前一定会先裁掉），所以这个开关只作用于第一遍。需要启用 LOW 阶段（才有 split 可裁），且本段不能同时用 fun control；节点会打印实际选择",
    "no material": "不启用素材",
    "direct reference in frame conflicts with 'auto crop input ref' on this segment: the ref_video_input socket carries ONE video":
        "本段同时使用帧内直引与“auto crop input ref”：ref_video_input 只有一个视频源，二者不能共存",
    "material window": "参考窗口",
    "drag this thumbnail onto another one to renumber the row: the leftmost image is <Picture 1> (the wired sockets are numbered before the picked files, and they cannot move)":
        "把缩略图拖到另一张上即可重排顺序：最左边那张就是 <Picture 1>（已接线的 socket 图片排在前面，且不可拖动）",
    "wired socket - fixed position (sockets are numbered before the picked files)":
        "已接线的 socket · 位置固定（socket 排在拾取文件之前）",
    // ── mute: the chain's stop point ('close the chain here') ──
    "mute: stop the chain before this segment — this segment and every later one are closed (not sampled in this run)":
        "关闭：链在本段之前停止 —— 本段及其后所有段都处于关闭状态（本次不采样）",
    "unmute: reopen the whole chain — every closed segment becomes active again":
        "解除关闭：整条链恢复全开 —— 之前关闭的所有段重新参与采样",
    "unmute this segment and every earlier one — the chain runs through it and stops right after it (later segments stay closed; on the last segment the whole chain reopens)":
        "解除本段及之前各段的关闭 —— 链跑到本段为止，停止点落在本段之后（之后的段落保持关闭；本段是最后一段时整条链恢复全开）",
    "the chain stops before this segment": "链在本段之前停止",
    "closed · not sampled in this run": "已关闭 · 本次不采样",
    "closed": "已关闭",
    "closed segment(s)": "个关闭分段",
    "unmute": "解除关闭",
    "running": "运行",
    "mute: the chain stops before segment": "关闭：链在此分段之前停止",
    "segment 0 is closed — the chain has nothing to sample. Unmute it (or add a segment) in the Tile Editor.":
        "分段 0 处于关闭状态 —— 整条链没有任何分段可以采样。请解除关闭（或新增分段）后再排队。",
};

function isZhLocale() {
    const s = [
        localStorage["Comfy.Locale"],
        localStorage["Comfy.Settings.Comfy.Locale"],
        localStorage["Comfy.Settings.Language"],
        localStorage["AGL.Locale"],
        localStorage["Comfy.Settings.AGL.Locale"],
    ].find(v => typeof v === "string" && !!v);
    if (s) return /zh/i.test(s);
    return /^(zh|zh-cn|zh-tw|zh-hans|zh-hant)/i.test(navigator.language || "");
}
const ZH_UI_ENABLED = isZhLocale();
function uistr(en) {
    if (ZH_UI_ENABLED) return ZH_UI[en] || en;
    return en;
}

// A segment's frame count can only sit on a 17-frame lattice, because the
// model generates whole video-latent tokens and 5 tokens = exactly 17 pixel
// frames (FRAME_PER_TOKEN = (1,4,4,4,4) summed over 5):
//     segment 0    : 17k + 5      (the first token covers a single frame)
//     segment >= 1 : 17k
// Both below return the ONE lattice point NEAREST the target (ties go down),
// so a target is at most 8 frames (0.33 s) away from its lattice point - which
// is exactly what makes the seconds <-> frames mapping definite and exactly
// invertible. See secToFrames() / durFramesToSec() after fmtSec().
function snap17n5(n) { n = Math.round(n) || 5; return Math.max(5, Math.round((n - 5) / 17) * 17 + 5); }
function snap17(n) { n = Math.round(n) || 17; return Math.max(17, Math.round(n / 17) * 17); }

// A CARRIED TAIL (and a seam-reference length) is not a duration: it can only
// be realized on the 17n+5 lattice, and the python nodes snap it UP onto it
// (snap_17n5 in nodes/temporal_tile_editor.py, applied again in
// nodes/temporal_extend.py) - never to the nearest point. Mirror that rule
// here, or the number the field shows is not the number the node uses:
// 17n+5 is a fixed point, while a 17n value is an ALIAS of the 17n+5 above
// (17 -> 22, 34 -> 39, 51 -> 56...). The (((c - n) % m) + m) % m form means
// the same in both languages (JS's % keeps the sign of its left operand,
// python's never goes negative).
function snap17n5Up(n) { n = Math.max(5, Math.round(n) || 5); return n + (((5 - n) % 17) + 17) % 17; }

// What a per-segment "overlap frames" editor state means. The number box holds
// a LENGTH, and a carried tail only exists on the 17n+5 lattice (floor 5): 0 is
// the HARD CUT - not a length at all - so it is not representable in that box
// and has its own tick next to it (see the overlap row). Both paths land here,
// so a 0 typed into the box means the hard cut as well, instead of being
// snapped up into a real 5-frame tail by snap17n5Up() (which floors at 5, and
// is why nothing may call it with a 0 it means to keep).
function overlapFramesValue(raw, hardCut) {
    if (hardCut) return 0;
    const v = parseInt(raw, 10);
    return (Number.isFinite(v) && v > 0) ? snap17n5Up(v) : 0;
}

// Where a dragged row item lands: `from`/`to` are indices into the same list
// and `after` = the pointer was past the target's midpoint (so the item lands
// to its right). Pulling the item OUT first shifts every later target down by
// one - the classic off-by-one that makes a drag to the right land one slot
// short. -1 = an index is unknown, so nothing moves.
// Standalone so a regression test can run the REAL rule (the drag needs a DOM,
// this arithmetic does not): mmh3-plugin-tests/mmh3_ui_dur_scroll_test.js.
function reorderTarget(from, to, after) {
    if (from < 0 || to < 0) return -1;
    let t = to + (after ? 1 : 0);
    if (from < t) t -= 1;
    return t;
}

// What the hard cut means, worded once: it is the tick's tooltip AND the hint
// the row shows while it is on, so the two can never drift apart.
const OVERLAP_HARD_CUT_HINT = "no fade and no blend across the seam: the segment starts fresh at the boundary and is stitched right after the previous one. The token grid gives every piece a minimum seam of 5 frames, and on a hard cut those 5 are frozen verbatim - they re-play frames the video already showed, so the cut lands exactly at the boundary and nothing at it is visible";

// ── direct reference in frame: ONE record per segment ──
// The mode splices a strip of the 'ref_video_input' video into the canvas and
// pins it with the video noise-mask, so the model generates the rest of the
// frame while literally SEEING the reference pixels (that is what the name
// says - the reference is in the frame, frame by frame). Where the strip sits
// ('side'), how wide the free band at its seam is ('expose') and whether it is
// dropped at the split ('cut_at_split') are all PER SEGMENT: 'side' === 'off'
// is that segment's switch, and it works whether or not the segment before it
// used the
// mode (the piece is spliced over the carried tail too, so the strip starts
// from this segment's overlap).
//
// The stage the record is normalized at (python: _sanitize_material in
// nodes/temporal_tile_editor.py, read back defensively by _material_record in
// nodes/temporal_extend.py) is the only source of truth; this is the JS
// mirror, pinned to the python ones by a table-driven test so the three
// copies cannot drift. 'expose' is quantized to 32 px because ONE token
// column is 32 px (16x latent downsample + 2x2 DiT patch), so 0 is the only
// other meaningful value (a hard edge at the seam).
// The CANVAS is per segment too: each segment splices its own strip, samples
// it, and the node cuts it back off afterwards. Because every card lays its
// own strip out, the sides may differ between segments - see _material_chain
// in nodes/temporal_tile_editor.py, the derived chain view that reports them.
// The strip's SIZE is not a field either: it follows the source's aspect ratio
// at the generation area's height (a left/right strip) or width (a top/bottom
// one), rounded up to 32, so the real SAMPLING canvas is the generation area
// PLUS the strip - exactly the numbers material_geometry() computes on the
// Extend node, which is the only side that has seen the source (the panel
// cannot know that canvas size and does not pretend to).
// The VIDEO is the mode's asset, and it rides the node's existing
// ref_video_input socket (the same wire an 'auto crop input ref' reference
// video uses - one wire, one role). The panel therefore reads the WIRE, not a
// field, to decide whether the mode can be used at all: see
// materialSideOptions below.
const MATERIAL_SIDES = ["off", "left", "right", "top", "bottom"];
// Which of those sides a segment's card may offer, given whether that socket
// is wired. An unwired socket is not "one more setting", it is the mode being
// unusable (there is no strip to cut), and the Extend node refuses the whole
// chain over it - so the list collapses to the off value rather than offering
// a direction the run would reject. Pure (a boolean in, a list out) so the
// regression test can run the REAL rule:
// mmh3-plugin-tests/mmh3_ui_dur_scroll_test.js.
function materialSideOptions(wired) {
    return MATERIAL_SIDES.filter((s) => wired || s === "off");
}
const DEFAULT_MATERIAL = {
    side: "off", expose: 32,
    // drop the strip at the HIGH -> LOW handoff instead of keeping it for the
    // whole schedule: the reference builds the structure, the detail stage
    // finishes the picture from the prompt alone. Needs a split (the LOW
    // stage); the node prints when it is left off.
    cut_at_split: false,
};
function normMaterial(m) {
    const d = { ...DEFAULT_MATERIAL };
    if (!m || typeof m !== "object") return d;
    if (MATERIAL_SIDES.includes(m.side)) d.side = m.side;
    const ex = parseInt(m.expose, 10);
    d.expose = Number.isFinite(ex) ? Math.max(0, Math.round(ex / 32) * 32) : 32;
    // mirrors the python normalizers exactly: a real boolean, or one of the
    // same textual truthy spellings (a number is NOT accepted on either side)
    const cut = m.cut_at_split;
    d.cut_at_split = cut === true
        || (typeof cut === "string"
            && ["1", "true", "yes", "on"].includes(cut.trim().toLowerCase()));
    return d;
}
// one segment's record, always in its normalized form
function segMaterial(seg) {
    return normMaterial(seg && seg.material);
}

// The BGM slice of a continuation segment starts one CARRIED TAIL before the
// boundary between segments (that tail is re-covered, which is what keeps the
// music continuous across the seam). Mirrors the Extend node's
// snap_split_frame(): the realized split is the 17-frame grid point nearest to
// (boundary - tail), never at or after the boundary - so a tail covering the
// whole accumulated chain clamps to 0 and that segment restarts the song.
function gridSplitNearest(boundary, tail) {
    const target = boundary - Math.max(5, tail);
    let best = 0, bestD = Math.abs(target);
    for (let k = 5; ; k += 5) {
        const f = 17 * (k / 5);          // frames_for_tokens(5j) = 17j
        if (f >= boundary) break;
        const d = Math.abs(f - target);
        if (d < bestD) { best = f; bestD = d; }
    }
    return best;
}
function fmtSec(frames) { return (Math.round(frames) / FPS).toFixed(2); }

// The absolute frame window [start, end) of ONE segment's fun-control slice,
// mirroring the Extend node's control_slice_window() + its chain replay
// (cum = the sum of the PRECEDING segments' new_frames, i.e. this segment's
// boundary on the chain timeline; nf = new_frames; whole = the 'whole control
// video' mode). A piece re-covers the previous segment's last `tail` frames,
// so an auto-crop slice starts one carried tail BEFORE the boundary; a tail
// that swallows the whole accumulated chain clamps the split to 0 (and the
// realized tail becomes the entire accumulated length - that is why the
// window length is built from tail_real, not from `tail`). A HARD CUT (tail 0)
// is NOT a tail of 0: gridSplitNearest floors its own tail at 5, because a
// piece can only be spliced in where its token-group anchors land on the
// chain's (multiples of 5 - see the python seam_split), so it realizes that
// same 5-frame seam. Who OWNS those 5 frames is a fade question, not a window
// one, and does not change this span.
// Standalone so a regression test can run the REAL rule against the python
// node's, value by value - do not inline it back into controlRows.
function controlWindowFrames(first, cum, tail, nf, whole) {
    const split = (first || cum <= 0) ? 0 : gridSplitNearest(cum, tail || 0);
    const tailReal = (first || cum <= 0) ? 0 : (cum - split);
    const total = tailReal + nf;
    const start = whole ? 0 : split;
    return [start, start + total];
}

// ── duration: the DEFINITE seconds <-> frames mapping ──
// Frames are what the model actually generates, so frames are the source of
// truth; the seconds shown are always derived from them. Mapping an integer
// second to its frame count is fully determined (no rounding ambiguity), it
// just differs per segment because the lattice differs (see snap17n5/snap17):
//
//   segment 0    (17k+5):  1s -> 22f (0.92s)   2s -> 56f (2.33s)
//                          3s -> 73f (3.04s)   4s -> 90f (3.75s)
//                          5s -> 124f (5.17s)  6s -> 141f (5.88s)
//   segment >= 1 (17k):    1s -> 17f (0.71s)   2s -> 51f (2.13s)
//                          3s -> 68f (2.83s)   4s -> 102f (4.25s)
//                          5s -> 119f (4.96s)  6s -> 136f (5.67s)
//
// secToFrames() is the definition of the mapping (the first DUR_TABLE_SECS
// rows of it are also written into the duration input's tooltip), and
// durFramesToSec() is its exact inverse: a lattice point is at most 8 frames
// (0.33 s) away from the typed target, i.e. well inside the 12-frame half
// second, so one round() recovers the typed second on EITHER lattice - no
// per-grid rounding rule is needed (that is what used to make a typed 3 s
// come back as 4 s).
const secToFrames = (sec, first) => (first ? snap17n5 : snap17)(Math.round(sec * FPS));
const durFramesToSec = (f) => Math.max(1, Math.round(f / FPS));
// What the seconds box displays. Normally the integer second (typing seconds
// always lands on one of those, so the box shows back exactly what was typed).
// A frame count typed directly in frames mode can miss every integer second by
// half a step: one lattice point in 24 sits on a .5 s boundary (204f = 8.5 s,
// 396f = 16.5 s), and there the exact decimal is shown so the box can never
// disagree with the frame count printed next to it.
const durSecFromFrames = (f, first) => {
    const s = durFramesToSec(f);
    return secToFrames(s, first) === f ? s : Number((f / FPS).toFixed(2));
};
const DUR_TABLE_SECS = 10;
function durTableText(first) {
    const rows = [];
    for (let s = 1; s <= DUR_TABLE_SECS; s++) rows.push(`${s}s → ${secToFrames(s, first)}f`);
    return rows.join(",  ");
}

// ── shared input media list (same API as the spatial editor, categorized) ──
// Unlike the spatial editor's cache-null pattern we keep `_media` always
// populated and RE-FETCH on demand (picker open / workflow "executed").
// The previous "invalidate once, never fetch again" logic left every image
// dropdown permanently empty after the first workflow run — that was the
// "add image 不可点击" bug.
let _media = { images: [], videos: [], audios: [] };
async function refreshInputMedia() {
    try {
        const resp = await fetch("/mmh3te/input_files?categorized=1");
        const data = await resp.json();
        if (Array.isArray(data)) {
            _media = { images: data, videos: [], audios: [] };   // legacy endpoint
        } else if (data && typeof data === "object") {
            _media = {
                images: Array.isArray(data.images) ? data.images : [],
                videos: Array.isArray(data.videos) ? data.videos : [],
                audios: Array.isArray(data.audios) ? data.audios : [],
            };
        }
    } catch { /* keep the previous list on transient failures */ }
    return _media;
}

// input-folder thumbnail URL (same trick as the spatial editor)
function imageUrl(filename) {
    if (!filename) return "";
    return `/view?filename=${encodeURIComponent(filename)}&type=input&preview=webp;80`;
}

// ── image preview overlay (spatial-editor style) ──
// Shows a large preview of an input image in the right panel of the dock
// that owns `anchorEl`, covering the generated previews. Closes via ✕ or by
// clicking anywhere outside the overlay (picker dropdowns/triggers excluded
// so selecting another image just swaps the preview instead of closing it).
function showImagePreview(filename, anchorEl) {
    if (!filename || !anchorEl) return;
    const dock = anchorEl.closest?.(".mmh3tte-dock");
    const panel = dock?.querySelector?.(".mmh3tte-pvpanel");
    if (!panel) return;
    panel.querySelector(":scope > .mmh3tte-imgprev")?.remove();
    const dismiss = (e) => {
        document.removeEventListener("pointerdown", dismiss, true);
        if (!overlay.isConnected) return;
        // clicks inside the overlay (the ✕ button) or on any picker
        // dropdown/trigger are not "elsewhere"
        if (overlay.contains(e.target)) { document.addEventListener("pointerdown", dismiss, true); return; }
        if (e.target.closest?.(".mmh3tte-ref-dropdown, .mmh3tte-ref-trigger")) { document.addEventListener("pointerdown", dismiss, true); return; }
        overlay.remove();
    };
    const bar = el("div", { class: "bar" },
        el("span", { class: "name", title: filename }, filename),
        el("button", {
            class: "mmh3tte-btn",
            title: uistr("Close preview"),
            onclick: () => { document.removeEventListener("pointerdown", dismiss, true); overlay.remove(); },
        }, "\u2715"));
    const img = el("img", { src: `/view?filename=${encodeURIComponent(filename)}&type=input` });
    const overlay = el("div", { class: "mmh3tte-imgprev" }, bar, img);
    panel.appendChild(overlay);
    document.addEventListener("pointerdown", dismiss, true);
}

// ── delete-confirmation modal ──
// Document-level dialog rendered ABOVE the dock. window.confirm() is
// unreliable inside ComfyUI's embedded browser contexts, so the delete
// flow uses this styled dialog with explicit 取消 / 确认 buttons.
function confirmDialog(message, okLabel, onOk) {
    document.querySelector(".mmh3tte-confirm")?.remove();
    const close = () => overlay.remove();
    const overlay = el("div", { class: "mmh3tte-confirm" },
        el("div", { class: "box" },
            el("div", { class: "msg" }, message),
            el("div", { class: "btns" },
                el("button", { class: "mmh3tte-btn", onclick: close },
                    uistr("cancel")),
                el("button", {
                    class: "mmh3tte-btn warn",
                    onclick: () => { close(); onOk(); },
                }, okLabel))));
    overlay.addEventListener("pointerdown", (e) => {
        if (e.target === overlay) close();
    });
    document.body.appendChild(overlay);
}

// ── thumbnail reference picker (spatial-editor style) ──
// Trigger shows 32px thumbnails; clicking opens a fixed-position dropdown
// (appended to document.body: the dock has overflow:hidden AND a CSS
// transform, which makes it the containing block for fixed descendants —
// a dropdown inside the dock would be clipped away).
let _pickerEls = [];
let _pickerDocBound = false;
function _bindPickerDismiss() {
    if (_pickerDocBound || typeof document === "undefined") return;
    _pickerDocBound = true;
    document.addEventListener("pointerdown", (e) => {
        for (const p of _pickerEls) {
            if (p.open && !p.dd.contains(e.target) && !p.trigger.contains(e.target)) p.closeFn();
        }
    }, true);
}
function makeImagePicker({ multiple = false, selected = [], onChanged,
                           sockets, ordinals = false } = {}) {
    _bindPickerDismiss();
    const state = { open: false, selected: [...selected] };
    const trigger = el("div", { class: "mmh3tte-ref-trigger" });
    const dd = el("div", { class: "mmh3tte-ref-dropdown" });
    document.body.appendChild(dd);
    // wired reference-image sockets (Tile Editor Autogrow slots). Passed as
    // a function so every re-render sees the live exclusion state.
    const socketsOf = typeof sockets === "function" ? sockets : () => (sockets || []);
    // `ordinals`: stamp each chip with the ordinal the model will give it in
    // the prompt. The presentation order IS sockets-then-picked-files (see
    // temporal_extend.py: `refs = _sock_refs(seg, sock_pool) + files`), and
    // the tokenizer numbers images 1..n in that order
    // ('<Picture 1>: ', '<Picture 2>: ', ...), so the row reads left-to-right
    // as exactly the <Picture i> the prompt has to name. Only enabled where
    // that numbering is guaranteed (a Ref2VA 'load images' row); the FL2VA
    // first/last rows use their own two-slot numbering.
    // The caption spells the ordinal out ('Picture 1' - the prompt's own
    // syntax) instead of abbreviating it, and the picked files can be DRAGGED
    // to renumber the row (see makeSortable); the sockets keep their order,
    // because the Extend node packs them first whatever the row shows.
    function ordBadge(n) {
        return ordinals ? el("span", { class: "mmh3tte-ord" }, `Picture ${n}`) : null;
    }

    // one "dummy" chip per wired socket: the socket's own label (e.g.
    // 'First_or_Ref_Image_0') replaces the thumbnail, the ✕ rules that image
    // out FOR THIS SEGMENT (the wire itself is untouched)
    function socketChip(s, onRemove, ord) {
        const label = s.label || s.name;
        // the caption is the column's first child (the CSS orders it under the
        // chip); the chip's own content is one row: placeholder + socket label
        // ...and a socket chip cannot be dragged: the Extend node packs every
        // wired socket BEFORE the picked files, so their order is not ours to
        // change (see makeSortable). The tooltip says so instead of offering a
        // gesture that would silently do nothing.
        const chip = el("span", { class: "mmh3tte-thumb",
                                  title: (ord ? `${label}  \u2192  <Picture ${ord}>  \u00b7  ` : "")
                                      + uistr("wired socket - fixed position (sockets are numbered before the picked files)") },
            ordBadge(ord),
            el("span", { class: "mmh3tte-sock" },
                el("span", { class: "mmh3tte-sock-ph" }, "\u25A3"),
                el("span", { class: "mmh3tte-sock-name" }, label)));
        if (onRemove) {
            const x = el("span", { class: "mmh3tte-thumb-x",
                                   title: uistr("remove image") }, "\u2715");
            x.addEventListener("click", (e) => { e.stopPropagation(); onRemove(); });
            chip.appendChild(x);
        }
        return chip;
    }

    // ── dragging a picked file onto another one ──
    // The row order IS the numbering the prompt sees ('<Picture 1>: ' ...), so
    // dragging is how one says which image is which without dropping and
    // re-adding them all. Only the PICKED files are reorderable: the Extend
    // node always packs the wired sockets first (`_sock_refs(seg) + files`),
    // so a file cannot overtake a socket and a drop is clamped to the files'
    // own region - the leftmost picked image is the first <Picture> after the
    // sockets.
    //
    // The gesture is driven by POINTER events, not by the HTML5 drag-and-drop
    // API: these chips live in a DOM overlay on top of ComfyUI's canvas, where
    // a native drag has to win against the browser's own image drag, the
    // canvas' node drag and the webview's text selection - any of the three
    // can swallow it before `dragstart` ever fires, which is exactly the
    // "the drag does not seem to exist" a user reports. Pointer capture has
    // none of those competitors: every event of the gesture lands on the chip
    // that started it, and `elementFromPoint` still reports what is really
    // underneath - all a drop needs. Nothing here is a native drag source
    // either (the chip AND its <img> are draggable="false").
    let dragFile = null;      // the file being dragged, null when idle
    let draggedAt = 0;        // when the last drag ended (see the click handler)
    let dropTarget = null;    // {file, after} the pointer is currently over
    function clearDropMarks() {
        for (const c of trigger.querySelectorAll(".drop-before, .drop-after"))
            c.classList.remove("drop-before", "drop-after");
    }
    function makeSortable(chip, file) {
        chip.classList.add("mmh3tte-grab");    // the grab cursor: "this one moves"
        chip.draggable = false;                // no native drag - see above
        chip.dataset.file = file;              // what this chip stands for
        chip.title = uistr("drag this thumbnail onto another one to renumber the row: the leftmost image is <Picture 1> (the wired sockets are numbered before the picked files, and they cannot move)");
        chip.addEventListener("pointerdown", (e) => {
            if (e.button !== 0) return;                        // left button only
            if (e.target.closest?.(".mmh3tte-thumb-x")) return; // the x removes
            const startX = e.clientX, startY = e.clientY;
            let moved = false;
            clearDropMarks();
            dropTarget = null;
            // capture: the whole gesture keeps reporting to THIS chip, even
            // while the pointer is over another one
            try { chip.setPointerCapture(e.pointerId); } catch {}
            const onMove = (me) => {
                if (!moved) {
                    // a few pixels of slop, so a plain click on the thumbnail
                    // still opens the picker instead of renumbering the row
                    if (Math.abs(me.clientX - startX) +
                        Math.abs(me.clientY - startY) < 6) return;
                    moved = true;
                    dragFile = file;
                    chip.classList.add("dragging");
                }
                const under = document.elementFromPoint(me.clientX, me.clientY)
                    ?.closest?.(".mmh3tte-thumb");
                // only a PICKED file is a drop target: a socket chip has no
                // file and cannot be overtaken (see the header of this block)
                const underFile = under ? (under.dataset.file || null) : null;
                clearDropMarks();
                dropTarget = null;
                if (underFile == null || underFile === file) return;
                const r = under.getBoundingClientRect();
                const after = (me.clientX - r.left) > r.width / 2;
                under.classList.add(after ? "drop-after" : "drop-before");
                dropTarget = { file: underFile, after };
            };
            const onUp = () => {
                chip.removeEventListener("pointermove", onMove);
                chip.removeEventListener("pointerup", onUp);
                chip.removeEventListener("pointercancel", onUp);
                chip.classList.remove("dragging");
                clearDropMarks();
                const target = dropTarget;
                dropTarget = null;
                dragFile = null;
                if (!moved) return;        // a plain click: not a drag at all
                moved = false;
                // the click that follows the release must not ALSO open the
                // dropdown (a native drag used to suppress it for us)
                draggedAt = Date.now();
                if (!target) return;
                const from = state.selected.indexOf(file);
                const to = reorderTarget(from, state.selected.indexOf(target.file),
                                         target.after);
                if (to < 0 || to === from) return;
                const [movedFile] = state.selected.splice(from, 1);
                state.selected.splice(to, 0, movedFile);
                refreshViews();
                onChanged?.(state.selected.slice());
            };
            chip.addEventListener("pointermove", onMove);
            chip.addEventListener("pointerup", onUp);
            chip.addEventListener("pointercancel", onUp);
        });
    }

    function renderTrigger() {
        trigger.innerHTML = "";
        const socks = socketsOf().filter((s) => s.included);
        const shown = multiple ? state.selected.slice(0, 8) : state.selected.slice(0, 1);
        if (shown.length || socks.length) {
            const thumbs = el("span", { class: "mmh3tte-thumbrow" });
            // WIRED SOCKETS FIRST, then the picked files. This is the order
            // the Extend node hands them to the model, so what the row shows
            // left to right is exactly <Picture 1>, <Picture 2>, ... in the
            // prompt. (Rendering the files first - as this used to - put the
            // sockets' index behind however many files were picked, i.e. the
            // row read in the opposite order to the one the model uses.)
            socks.forEach((s, i) => {
                thumbs.appendChild(socketChip(s, () => { s.toggle?.(); refreshViews(); },
                    i + 1));
            });
            shown.forEach((f, j) => {
                // draggable="false": the CHIP is the drag source, so the ghost
                // carries the ordinal caption along with the image
                const th = el("img", { src: imageUrl(f), title: f, draggable: "false" });
                th.addEventListener("mouseenter", () => showImagePreview(f, trigger));
                // per-thumbnail remove: a small red ✕ in the top-right corner
                // drops just THIS file from the selection (removing the whole
                // selection at once was too coarse)
                const x = el("span", { class: "mmh3tte-thumb-x",
                                       title: uistr("remove image") }, "\u2715");
                x.addEventListener("click", (e) => {
                    e.stopPropagation();
                    const i = state.selected.indexOf(f);
                    if (i >= 0) state.selected.splice(i, 1);
                    refreshViews();
                    onChanged?.(state.selected.slice());
                });
                const chip = el("span", { class: "mmh3tte-thumb" },
                    ordBadge(socks.length + j + 1), th, x);
                // a single-image slot has nothing to reorder in, and the
                // sockets are fixed - see makeSortable
                if (multiple) makeSortable(chip, f);
                thumbs.appendChild(chip);
            });
            trigger.appendChild(thumbs);
            if (multiple) {
                if (state.selected.length)
                    trigger.appendChild(el("span", { class: "name" },
                        `+${state.selected.length}`));
            } else if (state.selected.length) {
                trigger.appendChild(el("span", { class: "name" }, state.selected[0]));
            }
        } else {
            trigger.appendChild(el("span", { class: "name mmh3tte-muted" }, uistr("(none)")));
        }
        trigger.appendChild(el("span", { class: "arrow" }, "\u25BE"));
    }

    // rebuild BOTH views after any selection mutation. The dropdown stays
    // open while the trigger's ✕ is clicked (the global dismiss handler
    // deliberately ignores clicks inside the trigger), so refreshing only
    // the thumbnails would leave stale ✓ marks in the visible option list.
    function refreshViews() {
        renderTrigger();
        if (state.open) renderOptions();
    }

    function renderOptions() {
        dd.innerHTML = "";
        const listed = new Set(_media.images);
        // wired reference-image sockets are listed FIRST, matching both the
        // chip row and the model's <Picture i> numbering; same ✓ toggle as
        // the files below, so an image ruled out with the chip's ✕ can be
        // brought back. (No ordinal badge here: this is a menu, not the
        // ordered selection - a file's index only exists once it is picked.)
        for (const s of socketsOf()) {
            const opt = el("div", { class: "mmh3tte-ref-opt" + (s.included ? " selected" : ""),
                                    title: s.label || s.name },
                el("span", { class: "check" }, s.included ? "\u2713" : ""),
                el("span", { class: "mmh3tte-sock-ph" }, "\u25A3"),
                el("span", { class: "name" }, `${s.label} ${uistr("(wired image)")}`));
            opt.addEventListener("click", () => { s.toggle?.(); refreshViews(); });
            dd.appendChild(opt);
        }
        for (const f of _media.images) {
            const isSel = state.selected.includes(f);
            const opt = el("div", { class: "mmh3tte-ref-opt" + (isSel ? " selected" : "") },
                el("span", { class: "check" }, isSel ? "\u2713" : ""),
                el("img", { src: imageUrl(f) }),
                el("span", { class: "name" }, f));
            // hovering an option shows a large preview in the right panel
            opt.addEventListener("mouseenter", () => showImagePreview(f, trigger));
            opt.addEventListener("click", () => {
                if (multiple) {
                    const i = state.selected.indexOf(f);
                    if (i >= 0) state.selected.splice(i, 1); else state.selected.push(f);
                } else {
                    state.selected = (state.selected[0] === f) ? [] : [f];
                    closePicker();
                }
                refreshViews();
                onChanged?.(state.selected.slice());
            });
            dd.appendChild(opt);
        }
        // picked entries that are no longer listed in the input folder
        // (moved/renamed) would otherwise be impossible to deselect —
        // surface them here so they can be removed individually
        for (const f of state.selected) {
            if (listed.has(f)) continue;
            const opt = el("div", { class: "mmh3tte-ref-opt selected" },
                el("span", { class: "check" }, "\u2713"),
                el("img", { src: imageUrl(f) }),
                el("span", { class: "name" }, `${f} ${uistr("(no longer in input folder)")}`));
            opt.addEventListener("click", () => {
                const i = state.selected.indexOf(f);
                if (i >= 0) state.selected.splice(i, 1);
                refreshViews();
                onChanged?.(state.selected.slice());
            });
            dd.appendChild(opt);
        }
        if (!_media.images.length && !state.selected.length && !socketsOf().length) {
            dd.appendChild(el("div", { class: "mmh3tte-muted", style: "padding:6px" }, uistr("(no image)")));
        }
    }

    function openPicker() {
        // drop pickers whose node left the DOM (renderAll rebuilds cards)
        _pickerEls = _pickerEls.filter(p => {
            if (!p.trigger.isConnected) { p.dd.remove(); return false; }
            return true;
        });
        for (const p of _pickerEls) { if (p !== handle) p.closeFn(); }
        dd.classList.add("open");
        state.open = true;
        handle.open = true;   // the document-level dismiss handler keys off this
        // position below the trigger; flip up when it would overflow the viewport
        const r = trigger.getBoundingClientRect();
        const dw = dd.offsetWidth || 260, dh = dd.offsetHeight || 0;
        dd.style.left = Math.max(8, Math.min(r.left, window.innerWidth - dw - 8)) + "px";
        const below = r.bottom + 2 + dh <= window.innerHeight - 8;
        dd.style.top = Math.round(below ? r.bottom + 2 : Math.max(8, r.top - dh - 2)) + "px";
    }
    function closePicker() { dd.classList.remove("open"); state.open = false; handle.open = false; }

    trigger.addEventListener("click", async (e) => {
        e.stopPropagation();
        // the tail of a drag must not read as a click (native DnD suppresses
        // it in most browsers, but not in all of them)
        if (Date.now() - draggedAt < 300) return;
        if (state.open) { closePicker(); return; }
        await refreshInputMedia();   // always open with a fresh input-folder listing
        renderOptions();
        openPicker();
    });

    renderTrigger();
    const handle = { trigger, dd, open: false, closeFn: closePicker };
    _pickerEls.push(handle);
    return { root: trigger, close: closePicker };
}

function el(tag, attrs = {}, ...children) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
        if (k === "class") e.className = v;
        else if (k === "style") e.style.cssText = v;
        else if (k.startsWith("on") && typeof v === "function") e.addEventListener(k.slice(2), v);
        else if (v !== null && v !== undefined) e.setAttribute(k, v);
    }
    for (const c of children.flat()) {
        if (c == null) continue;
        e.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    }
    return e;
}

// ── CSS injection (same palette as the spatial editor dock) ──
function injectStyle() {
    if (document.getElementById("mmh3-tte-style")) return;
    const st = el("style", { id: "mmh3-tte-style" });
    st.textContent = `
        .mmh3tte-dock { position:fixed; z-index:8500; display:flex; flex-direction:column;
            background:var(--kj-dock-bg,#1a1a1a); border:1px solid var(--kj-dock-border,#555);
            border-radius:8px; box-shadow:0 8px 30px rgba(0,0,0,0.55);
            min-width:560px; min-height:380px; overflow:hidden; pointer-events:auto; }
        .mmh3tte-dock.minimized { min-height:0 !important; height:auto !important; }
        .mmh3tte-dock.minimized .mmh3tte-body, .mmh3tte-dock.minimized .mmh3tte-bar,
        .mmh3tte-dock.minimized .mmh3tte-rsz { display:none; }
        .mmh3tte-head { display:flex; align-items:center; gap:6px; padding:4px 8px;
            background:var(--kj-dock-head,#262626); cursor:move; font:12px sans-serif;
            color:#ccc; user-select:none; border-bottom:1px solid rgba(0,0,0,0.25); flex:0 0 auto; }
        .mmh3tte-head .title { flex:1; font:bold 12px sans-serif; }
        .mmh3tte-zoom-group { display:flex; align-items:center; gap:2px; }
        .mmh3tte-zoom-lbl { font:11px monospace; color:#aaa; min-width:36px; text-align:center; cursor:pointer; }
        .mmh3tte-zoom-lbl:hover { color:#fff; }
        .mmh3tte-btn { background:#333; border:1px solid #555; border-radius:4px;
            color:#bbb; font:11px sans-serif; cursor:pointer; padding:2px 8px;
            line-height:16px; white-space:nowrap; flex-shrink:0; }
        .mmh3tte-btn:hover { border-color:#46b4e6; color:#fff; }
        .mmh3tte-btn.primary { border-color:#46b4e6; color:#46b4e6; }
        .mmh3tte-btn.primary:hover { background:#2a3a42; }
        .mmh3tte-btn.warn { color:#e06050; }
        .mmh3tte-btn.warn:hover { border-color:#e06050; }
        .mmh3tte-btn:disabled { opacity:.4; cursor:default; }
        .mmh3tte-bar { display:flex; align-items:center; gap:6px; padding:4px 8px;
            background:#222; border-bottom:1px solid #333; font:11px sans-serif; color:#aaa;
            flex-wrap:wrap; flex:0 0 auto; }
        .mmh3tte-body { display:flex; flex:1 1 auto; min-height:0; padding:8px; gap:8px; }
        .mmh3tte-segcol { flex:1 1 auto; min-width:280px; overflow-y:auto; overflow-x:hidden;
            background:#1d1d1d; border-radius:4px; padding:8px; }
        .mmh3tte-pvpanel { position:relative; width:300px; flex:0 0 auto; display:flex; flex-direction:column;
            gap:6px; overflow-y:auto; background:#262626; border-radius:4px;
            font:11px sans-serif; color:#bbb; padding:8px; }
        /* large image-preview overlay (covers the generated previews until closed) */
        .mmh3tte-imgprev { position:absolute; inset:0; z-index:30; display:flex; flex-direction:column;
            gap:4px; background:#1a1a1a; border-radius:4px; padding:6px; }
        .mmh3tte-imgprev .bar { display:flex; align-items:center; gap:6px; flex:0 0 auto; }
        .mmh3tte-imgprev .bar .name { flex:1; overflow:hidden; text-overflow:ellipsis;
            white-space:nowrap; font:11px sans-serif; color:#ddd; }
        .mmh3tte-imgprev img { flex:1 1 auto; min-height:0; width:100%; object-fit:contain;
            border-radius:4px; background:#111; }
        /* delete-confirmation modal (document-level, above the dock) */
        .mmh3tte-confirm { position:fixed; inset:0; z-index:10000; display:flex;
            align-items:center; justify-content:center; background:rgba(0,0,0,.55); }
        .mmh3tte-confirm .box { background:#222; border:1px solid #555; border-radius:6px;
            padding:14px 16px; max-width:380px; display:flex; flex-direction:column; gap:12px;
            font:12px sans-serif; color:#ddd; box-shadow:0 6px 24px rgba(0,0,0,.5); }
        .mmh3tte-confirm .msg { line-height:1.55; white-space:pre-line; }
        .mmh3tte-confirm .btns { display:flex; justify-content:flex-end; gap:8px; }
        .mmh3tte-lbl { color:#888; font:11px sans-serif; flex:0 0 auto; min-width:60px; }
        .mmh3tte-row { display:flex; align-items:center; gap:6px; margin:3px 0; flex-wrap:wrap; }
        .mmh3tte-row.nowrap { flex-wrap:nowrap; }
        .mmh3tte-inp { background:#1d1d1d; border:1px solid #444; border-radius:4px;
            color:#ddd; font:11px sans-serif; padding:2px 6px; width:64px; text-align:right; }
        .mmh3tte-inp:focus { border-color:#46b4e6; outline:none; }
        .mmh3tte-inp:disabled { opacity:.5; }
        .mmh3tte-sel { background:#1d1d1d; border:1px solid #444; border-radius:4px;
            color:#ddd; font:11px sans-serif; padding:2px 4px; min-width:0; }
        .mmh3tte-sel:focus { border-color:#46b4e6; outline:none; }
        .mmh3tte-area { width:100%; box-sizing:border-box; background:#1d1d1d; border:1px solid #444;
            border-radius:4px; color:#ddd; font:12px monospace; padding:4px 6px; resize:vertical;
            min-height:36px; }
        .mmh3tte-area:focus { border-color:#46b4e6; outline:none; }
        .mmh3tte-sep { border:none; border-top:1px solid #333; margin:6px 0; }
        .mmh3tte-hdr { font:bold 11px sans-serif; color:#aaa; padding:2px 0; user-select:none; }
        .mmh3tte-hint { color:#666; font:10px sans-serif; }
        .mmh3tte-chk { display:flex; align-items:center; gap:4px; color:#888;
            font:11px sans-serif; flex:0 0 auto; cursor:pointer; user-select:none; }
        .mmh3tte-chk input { margin:0; cursor:pointer; }
        .mmh3tte-chk input:disabled { cursor:default; }
        .mmh3tte-chk:has(input:disabled) { opacity:.5; cursor:default; }
        .mmh3tte-muted { color:#777; font:11px sans-serif; }
        .mmh3tte-badge { color:#e6a046; font:11px sans-serif; }
        .mmh3tte-badge.err { color:#f08080; font-weight:bold; }
        .mmh3tte-sub { margin-top:8px; padding-top:7px; border-top:1px dashed #4a4a4a;
            color:#8ab4e8; font:bold 11px sans-serif; }
        /* segment card */
        .mmh3tte-seg { border:1px solid #444; border-radius:6px; margin:0 0 8px; background:#222; }
        .mmh3tte-seg > .seg-head { display:flex; gap:6px; align-items:center; padding:4px 6px;
            background:#262626; cursor:pointer; border-radius:6px 6px 0 0; user-select:none; }
        .mmh3tte-seg.open > .seg-head { border-bottom:1px solid #444; cursor:default; }
        .mmh3tte-seg > .seg-body { display:none; padding:6px 8px; }
        .mmh3tte-seg.open > .seg-body { display:block; }
        .mmh3tte-seg .seg-name { font:bold 11px sans-serif; color:#ddd; }
        .mmh3tte-seg .seg-dur { font:10px monospace; color:#888; }
        .mmh3tte-seg .seg-head .mmh3tte-sel { flex:0 1 auto; }
        .mmh3tte-seg .seg-head .spacer { flex:1; }
        /* locked segments (resume model): yellow highlight */
        .mmh3tte-seg.locked-seg { border-color:#e6a046; }
        .mmh3tte-seg.locked-seg > .seg-head { background:#33301f; }
        .mmh3tte-seg.locked-seg .seg-name { color:#e6a046; }
        /* locked segments are read-only: parameters can't change until the
           segment is unlocked. pointer-events:none also blocks the div-based
           image-picker triggers; opacity hints the frozen state. The lock
           button itself lives in the head and stays clickable. */
        .mmh3tte-seg.locked-seg .seg-body { pointer-events:none; opacity:.55; }
        .mmh3tte-seg.locked-seg .seg-head select { pointer-events:none; opacity:.55; }
        .mmh3tte-btn.lockon { color:#e6a046; border-color:#e6a046; }
        /* mute ('close the chain here'): the stop point and every segment
           behind it are CLOSED - the run samples nothing there. It is the
           LOCK's mirror image: the same tinting (border, head, name, button)
           in GRAY where the lock is amber - but with the opposite kind of
           freezing. A locked segment is read-only and reused from disk; a
           closed one keeps its controls live, because its parameters have to
           survive until it is reopened. The dashed border (the lock's is
           solid) plus the gray 'M' button mark the stop. */
        .mmh3tte-seg.closed-seg { border-style:dashed; border-color:#9aa0a6; }
        .mmh3tte-seg.closed-seg > .seg-head { background:#2b2e31; }
        .mmh3tte-seg.closed-seg .seg-name { color:#9aa0a6; }
        .mmh3tte-seg.closed-seg > .seg-body { opacity:.5; }
        .mmh3tte-btn.muteon { color:#9aa0a6; border-color:#9aa0a6; }
        .mmh3tte-badge.closed { color:#9aa0a6; }
        .mmh3tte-pv.locked-pv { border-color:#e6a046; box-shadow:0 0 0 1px #e6a046; }
        /* thumbnail reference picker (spatial-editor style) */
        .mmh3tte-ref-trigger { flex:1 1 auto; min-width:0; display:flex; align-items:center; gap:6px;
            background:#1d1d1d; border:1px solid #444; border-radius:4px; padding:4px 6px;
            cursor:pointer; min-height:40px; }
        .mmh3tte-ref-trigger:hover { border-color:#46b4e6; }
        .mmh3tte-ref-trigger .name { font:11px sans-serif; color:#ddd; flex:1; overflow:hidden;
            text-overflow:ellipsis; white-space:nowrap; }
        .mmh3tte-ref-trigger .name.mmh3tte-muted { color:#777; }
        .mmh3tte-ref-trigger .arrow { color:#888; font:10px sans-serif; flex:0 0 auto; }
        /* per-thumbnail remove: small red ✕ pinned to the top-right corner */
        .mmh3tte-thumb { position:relative; display:inline-flex; flex-direction:column;
            align-items:center; gap:1px; }
        .mmh3tte-thumb-x { position:absolute; top:-5px; right:-5px; width:14px; height:14px;
            border-radius:50%; background:#c0392b; color:#fff; font:10px/14px sans-serif;
            text-align:center; cursor:pointer; box-shadow:0 0 2px #000; user-select:none; }
        .mmh3tte-thumb-x:hover { background:#e74c3c; }
        /* ordinal caption: which <Picture N> this image becomes in the prompt.
           Wired sockets are numbered first, then the picked files, because
           that is the order the Extend node packs them in. The name is spelled
           OUT ('Picture 1', not 'P1') because it is the prompt's own syntax, so
           the caption sits UNDER the thumbnail (order:2 puts it after the
           image in the column) and stays click-through - clear of both the ✕
           (top-right) and the thumbnail's hover. */
        .mmh3tte-ord { order:2; font:9px/10px sans-serif; color:#8ab4e8;
            max-width:72px; overflow:hidden; text-overflow:ellipsis;
            white-space:nowrap; pointer-events:none; user-select:none; }
        /* dragging a picked thumbnail onto another one renumbers the row (the
           row order IS the <Picture i> order); the mark shows where it lands.
           Only the picked chips carry .mmh3tte-grab (the gesture is pointer
           driven - see makeSortable), so the grab cursor doubles as the
           affordance: a socket chip never shows it. touch-action/user-select
           keep a pen or touch drag from panning or selecting the caption
           instead of moving the chip. */
        .mmh3tte-thumb.mmh3tte-grab { cursor:grab; touch-action:none;
            user-select:none; }
        .mmh3tte-thumb.mmh3tte-grab:active { cursor:grabbing; }
        .mmh3tte-thumb.dragging { opacity:.4; }
        .mmh3tte-thumb.drop-before::before, .mmh3tte-thumb.drop-after::after {
            content:""; position:absolute; top:0; height:32px; width:2px;
            background:#46b4e6; border-radius:1px; }
        .mmh3tte-thumb.drop-before::before { left:-3px; }
        .mmh3tte-thumb.drop-after::after { right:-3px; }
        .mmh3tte-thumbrow { display:flex; gap:4px; flex-wrap:wrap; align-items:center; min-width:0; flex:0 1 auto; }
        .mmh3tte-thumbrow img { width:32px; height:32px; object-fit:cover; border-radius:3px; }
        /* wired reference-image sockets: a "dummy" chip standing in for an
           image that arrives through one of the node's reference sockets
           instead of the input folder. Shows the socket label; the ✕ rules
           that image out for THIS segment only (the wire itself is
           untouched). The chip is a ROW of its own inside the .mmh3tte-thumb
           column, so the ordinal caption sits under it exactly like it does
           under a thumbnail. */
        .mmh3tte-sock { display:inline-flex; align-items:center; gap:3px; height:32px;
            padding:0 5px;
            box-sizing:border-box; border:1px dashed #4e6b7a; border-radius:3px;
            background:#1b262b; color:#8ab4e8; font:10px/1 sans-serif;
            max-width:132px; user-select:none; }
        .mmh3tte-sock .mmh3tte-sock-ph { font-size:13px; color:#46b4e6; flex:0 0 auto; }
        .mmh3tte-sock .mmh3tte-sock-name { overflow:hidden; text-overflow:ellipsis;
            white-space:nowrap; }
        .mmh3tte-ref-opt .mmh3tte-sock-ph { font-size:13px; color:#46b4e6;
            width:32px; text-align:center; flex:0 0 auto; }
        .mmh3tte-ref-dropdown { position:fixed; z-index:10000;
            background:#1d1d1d; border:1px solid #444; border-radius:4px;
            display:none; max-height:300px; overflow-y:auto; min-width:240px; }
        .mmh3tte-ref-dropdown.open { display:block; }
        .mmh3tte-ref-opt { display:flex; align-items:center; gap:6px; padding:4px 6px;
            cursor:pointer; border-bottom:1px solid #333; }
        .mmh3tte-ref-opt:hover, .mmh3tte-ref-opt.selected { background:#2a3a42; }
        .mmh3tte-ref-opt img { width:32px; height:32px; object-fit:cover; border-radius:3px; }
        .mmh3tte-ref-opt .name { font:11px sans-serif; color:#ddd; flex:1; overflow:hidden;
            text-overflow:ellipsis; white-space:nowrap; }
        .mmh3tte-ref-opt .check { color:#46b4e6; font:11px sans-serif; width:14px; flex:0 0 auto; }
        /* previews */
        .mmh3tte-pvgrid { display:grid; grid-template-columns:repeat(auto-fill,minmax(120px,1fr)); gap:8px; }
        .mmh3tte-pv { border:1px solid #444; border-radius:6px; padding:4px; background:#1d1d1d; }
        .mmh3tte-pv img { width:100%; border-radius:4px; display:block; background:#000; cursor:pointer; }
        .mmh3tte-pv .cap { display:flex; justify-content:space-between; align-items:center;
            margin-top:3px; font:11px sans-serif; color:#aaa; gap:4px; }
        /* resize handles */
        .mmh3tte-rsz { position:absolute; z-index:20; touch-action:none; }
        .mmh3tte-rsz.n { top:0; left:11px; right:11px; height:6px; cursor:ns-resize; }
        .mmh3tte-rsz.s { bottom:0; left:11px; right:11px; height:6px; cursor:ns-resize; }
        .mmh3tte-rsz.e { right:0; top:11px; bottom:11px; width:6px; cursor:ew-resize; }
        .mmh3tte-rsz.w { left:0; top:11px; bottom:11px; width:6px; cursor:ew-resize; }
        .mmh3tte-rsz.se { bottom:0; right:0; width:12px; height:12px; cursor:nwse-resize; }
        .mmh3tte-rsz.sw { bottom:0; left:0; width:12px; height:12px; cursor:nesw-resize; }
        .mmh3tte-rsz.ne { top:0; right:0; width:12px; height:12px; cursor:nesw-resize; }
        .mmh3tte-rsz.nw { top:0; left:0; width:12px; height:12px; cursor:nwse-resize; }
        body.mmh3tte-dragging, body.mmh3tte-dragging * { cursor:move !important; }
    `;
    document.head.appendChild(st);
}

// ── mute: the chain's stop point ──
// ONE stop point per editor (not a flag per segment). Every segment from the
// stop on is CLOSED: this run samples nothing there and the output is the
// merged latent through the last active segment. The button is only offered on
// a segment with no stored latent yet - the same condition that disables its
// lock button - so the stop point can never sit before the resume point.
// Every click parks it right AFTER the clicked segment (see the renderSeg
// click handler): on a closed segment that reopens it together with every
// earlier one, while the later ones stay closed - the boundary only ever
// walks forward and a single click never wipes the chain.
//   -1 / out of range = no stop point, the whole chain runs (self-healing: a
//                       stop left behind by a shorter chain means nothing)
//   i (0..count-1)    = the chain runs segments 0..i-1
// The panel and the python node (temporal_tile_editor._mute_stop) both read the
// value through this normalization, so the panel shows what the run will do.
function muteStopIndex(muteFrom, count) {
    if (!Number.isInteger(muteFrom) || muteFrom < 0 || muteFrom >= count) return -1;
    return muteFrom;
}
// is segment `i` behind the stop point (closed for this run)?
function segIsClosed(stop, i) {
    return stop >= 0 && i >= stop;
}

function defaultSeg(i) {
    return {
        mode: i === 0 ? "FL2VA" : "Ref2VA",
        width: 768, height: 768,
        new_frames: i === 0 ? 124 : 102,
        seed: 0,
        prompt: "", negative: "",
        ref_source: "none",
        ref_images: [],
        // wired Ref_Image_N sockets this segment rules out (the ✕ on a dummy
        // chip in the reference row). Per-segment, non-destructive: the wire
        // stays connected, other segments keep using it.
        ref_socket_off: [],
        ref_image_size: "match",
        fl2va_first_image: "",
        fl2va_last_image: "",
        prev_frame_index: -1,
        ref_video: "",
        ref_video_audio: "",
        // reference video source: 'load' (picked ref_video file) | 'auto_crop'
        // (the 'auto crop input ref' mode - a slice of the wired
        // ref_video_input socket covering this segment's absolute span, the
        // sum of the PRECEDING segments' new_frames as its start and this
        // segment's own new_frames as its length, so abutting slices
        // reassemble the input video continuously).
        ref_video_mode: "load",
        ref_audio: "",
        // standalone audio reference: 'none' | 'load' (picked ref_audio
        // file) | 'prev' (previous segment's soundtrack, latent-level) |
        // 'initial' (first segment's soundtrack, latent-level)
        ref_audio_mode: "none",
        // fun control (MiniMax H3 Fun ControlNet): the ASSETS are chain-wide
        // (the node's optional 'fun_control_video' + mask/source video), the
        // APPLICATION is per segment -
        //   'off'       no control on this segment
        //   'auto_crop' the slice of the wired video covering this segment's
        //               absolute span on the chain's timeline (extended one
        //               carried tail backwards, because the piece re-covers
        //               that tail), so abutting slices reassemble the video
        //   'whole'     the control video from its first frame, clamped to
        //               this piece (the native Apply node's behavior)
        // strength scales the patch's residual; start/end are sigma
        // percentages of the model's own schedule (native semantics).
        control_mode: "off",
        control_strength: 1,
        control_start: 0,
        control_end: 1,
        // direct reference in frame: this segment's OWN record. 'side' ===
        // 'off' means it does not use the mode, and a new segment starts there
        // - the mode is opt-in per segment (a clone of the previous segment
        // inherits its record, see makeSegFrom).
        material: { ...DEFAULT_MATERIAL },
        // the ONLY per-segment continuation parameter: how many tail frames
        // this segment carries over from the previous one (17n+5 grid).
        // All other tail/fade/anchor/overlap settings are global - see the
        // "MMH3 Temporal Overlap Params" sub-node.
        ...(i === 0 ? {} : { overlap_frames: 39 }),
    };
}

// ── auto-refresh: every live editor dock's preview panel ──
// Registered by onNodeCreated, removed on node delete. Fired (debounced) when
// a workflow run finishes so saved latents/previews appear without a manual
// "refresh previews" click.
const _liveEditors = new Set();
let _autoRefreshTimer = 0;
function schedulePreviewRefresh() {
    clearTimeout(_autoRefreshTimer);
    _autoRefreshTimer = setTimeout(() => {
        _autoRefreshTimer = 0;
        for (const ed of _liveEditors) {
            try { ed.renderPreviews(); } catch {}
        }
    }, 400);
}

app.registerExtension({
    name: "MMH3UltimateExtend.TemporalTileEditor",
    // Queue-time guard: a stop point (mute) on segment 0 closes the whole
    // chain, and a resume point BEYOND the segment count has nothing to
    // sample. (resume == the effective segment count is fine: the executor
    // outputs the stored merged latent as-is.)
    // Block the queue HERE with a toast instead of failing deep inside
    // execution after all models have already been loaded.
    setup() {
        const origQueue = app.queuePrompt?.bind(app);
        if (!origQueue) return;
        app.queuePrompt = async function (...args) {
            const bad = [];
            // ...and the chains that are closed at their very first segment:
            // nothing at all would be sampled (the python node rejects that too,
            // this only fails before the models are loaded)
            const allClosed = [];
            for (const n of app.graph?._nodes || []) {
                if (n.type !== TARGET) continue;
                const w = n.widgets?.find(x => x.name === DATA_WIDGET);
                if (!w?.value) continue;
                try {
                    const d = JSON.parse(w.value);
                    const segs = Array.isArray(d.segments) ? d.segments : [];
                    const resume = Number.isInteger(d.resume_from_segment) ? d.resume_from_segment : 0;
                    // resume == segs.length (every segment locked) is
                    // ALLOWED and queues fine: the executor then passes the
                    // stored merged latent straight through to the outputs
                    // without sampling. Only a resume point beyond the chain
                    // is a real "nothing to run" state.
                    if (segs.length && resume > segs.length) {
                        bad.push(`"${n.title || TARGET}" (#${n.id})`);
                    }
                    if (muteStopIndex(Number.isInteger(d.mute_from_segment)
                                      ? d.mute_from_segment : -1, segs.length) === 0) {
                        allClosed.push(`"${n.title || TARGET}" (#${n.id})`);
                    }
                } catch {}
            }
            if (allClosed.length) {
                const msg0 = uistr("segment 0 is closed — the chain has nothing to sample. Unmute it (or add a segment) in the Tile Editor.");
                console.error(TAG, msg0, "->", allClosed.join(", "));
                const toast0 = app.extensionManager?.toast;
                if (toast0?.add) {
                    toast0.add({ severity: "error", summary: "MMH3 Temporal Tile Editor",
                                 detail: msg0, life: 6000 });
                } else {
                    alert(`MMH3 Temporal Tile Editor: ${msg0}`);
                }
                return;   // do not queue
            }
            if (bad.length) {
                const msg = uistr("all segments locked — unlock one or add a new segment");
                console.error(TAG, msg, "->", bad.join(", "));
                const toast = app.extensionManager?.toast;
                if (toast?.add) {
                    toast.add({ severity: "error", summary: "MMH3 Temporal Tile Editor",
                                detail: msg, life: 6000 });
                } else {
                    alert(`MMH3 Temporal Tile Editor: ${msg}`);
                }
                return;   // do not queue
            }
            return origQueue(...args);
        };
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== TARGET) return;
        injectStyle();

        await refreshInputMedia();
        try {
            // keep the media list fresh and auto-refresh preview panels after
            // every workflow execution ("executed" fires per output node,
            // "execution_success" once when the whole prompt finished)
            app.api?.addEventListener?.("executed", () => {
                refreshInputMedia();
                schedulePreviewRefresh();
            });
            app.api?.addEventListener?.("execution_success", () => schedulePreviewRefresh());
            // Backend pushes this after EVERY segment finishes sampling, so
            // the editor can refresh the just-written preview without waiting
            // for the whole multi-segment chain (a single synchronous run) to
            // end. schedulePreviewRefresh debounces + fan-outs to live editors.
            app.api?.addEventListener?.("mmh3te:segment_ready", () => schedulePreviewRefresh());
        } catch {}

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const node = this;
            if (origOnNodeCreated) origOnNodeCreated.call(this);

            const findW = (n) => node.widgets?.find(w => w.name === n);
            const dataWidget = findW(DATA_WIDGET);
            if (!dataWidget) { console.error(TAG, "missing", DATA_WIDGET); return; }
            dataWidget.hidden = true;
            dataWidget.computeSize = () => [0, -4];

            // ── State ──
            node._segments = [defaultSeg(0)];
            node._extendParams = {
                tail_frames: 39, tail_mode: "freeze_fade", fade_frames: 17,
                fade_value: 0.5, fade_mode: "flat", keyframes_mode: "reanchor",
                anchor_seam: true, anchor_strength: 0.999,
                overlap_mode: "later", overlap_blend: "linear",
            };
            node._resumeFrom = 0;
            node._openSeg = 0;
            node._lastSeg = -1;   // highest segment index stored on disk (from metadata)
            // ...and whether that number was actually READ from the session.
            // Until it has been, a segment that looks unsampled may well have
            // a latent on disk, so the delete button asks the server before it
            // skips its confirmation (see loadSessionMeta).
            node._metaKnown = false;
            // mute: the chain's stop point (see muteStopIndex). -1 = no stop.
            node._muteFrom = -1;

            // ── data sync (unchanged serialization format) ──
            function sync() {
                dataWidget.value = JSON.stringify({
                    segments: node._segments,
                    extend_params: node._extendParams,
                    resume_from_segment: node._resumeFrom,
                    // the NORMALIZED stop point, not the raw state field: a
                    // stop left behind by a deleted tail is written as "no
                    // stop", so the JSON always says what the run will do
                    mute_from_segment: muteStopIndex(node._muteFrom, node._segments.length),
                });
                scheduleEditorSave();
            }
            // ── editor-state persistence into the session ledger ──
            // The tile_data widget only lives inside the workflow JSON; if the
            // workflow is lost / the node is reset, every per-segment setting
            // would be gone even though the session dir still holds all the
            // latents. So every change also auto-saves a full snapshot under
            // session.json's 'editor' key (debounced). 'temp' storage is wiped
            // on restart, so it is skipped - persistence only matters for
            // non-temp locations.
            let _editorSaveTimer = null;
            let _editorSaveFn = null;
            function scheduleEditorSave() {
                const location = findW("storage_location")?.value || "temp";
                if (location === "temp") {
                    clearTimeout(_editorSaveTimer);
                    _editorSaveTimer = null;
                    _editorSaveFn = null;
                    return;
                }
                const session = findW("session_name")?.value || "session1";
                const state = {
                    segments: node._segments,
                    extend_params: node._extendParams,
                    resume_from_segment: node._resumeFrom,
                    mute_from_segment: muteStopIndex(node._muteFrom, node._segments.length),
                };
                const fire = () => {
                    _editorSaveTimer = null;
                    _editorSaveFn = null;
                    fetch("/mmh3te/temporal_session_state", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ location, session, state }),
                    }).catch(() => {});
                };
                clearTimeout(_editorSaveTimer);
                _editorSaveTimer = setTimeout(fire, 800);
                _editorSaveFn = fire;
            }
            // flush a pending save to the session it was scheduled for (used
            // right before the session/location widgets change, so edits never
            // leak into the new session)
            function flushEditorSave() {
                if (_editorSaveTimer != null) {
                    clearTimeout(_editorSaveTimer);
                    _editorSaveTimer = null;
                }
                if (_editorSaveFn) {
                    const f = _editorSaveFn;
                    _editorSaveFn = null;
                    f();
                }
            }
            // normalize a raw segment dict loaded from anywhere (tile_data
            // widget, session 'editor' snapshot) into the live shape
            function normSeg(s, i) {
                const seg = { ...defaultSeg(i), ...s };
                // migrate legacy audio references: a picked ref audio file
                // without an explicit mode means 'load audio' (the pre-modes
                // behavior)
                if (!s.ref_audio_mode && s.ref_audio)
                    seg.ref_audio_mode = "load";
                // migrate legacy per-seg extend_params: only tail_frames
                // survives, as overlap_frames
                if (seg.overlap_frames == null
                        && seg.extend_params?.tail_frames != null) {
                    seg.overlap_frames = seg.extend_params.tail_frames;
                }
                delete seg.extend_params;
                // the carried tail lives on the 17n+5 grid: a tail can only be
                // realized there and the python nodes snap it up onto it. An
                // older build offered a 0/17/34 step ladder, so a stored 17n
                // value means "the 17n+5 above it" - normalize on load, so the
                // field can never show a length the node will not use.
                if (seg.overlap_frames != null)
                    seg.overlap_frames = seg.overlap_frames <= 0
                        ? 0 : snap17n5Up(seg.overlap_frames);
                // fun control: the mode decides whether the Extend node
                // controls this segment at all, so an unknown / legacy value
                // must collapse to 'off' - never to a mode that silently
                // starts controlling. The numbers are clamped exactly like
                // python's _sanitize_control, so the field shows the value the
                // node will actually use.
                if (seg.control_mode !== "off" && seg.control_mode !== "auto_crop"
                        && seg.control_mode !== "whole")
                    seg.control_mode = "off";
                const c01 = (v, d) => {
                    const f = parseFloat(v);
                    return Number.isFinite(f) ? Math.min(1, Math.max(0, f)) : d;
                };
                seg.control_start = c01(seg.control_start, 0);
                seg.control_end = c01(seg.control_end, 1);
                if (seg.control_end < seg.control_start)
                    seg.control_end = seg.control_start;
                const cStrength = parseFloat(seg.control_strength);
                seg.control_strength = Number.isFinite(cStrength)
                    ? Math.max(0, cStrength) : 1;
                // direct reference in frame: the whole record is per segment
                // (python's _sanitize_material normalizes the same fields, so
                // the panel shows the values the node will actually use)
                seg.material = normMaterial(seg.material);
                return seg;
            }
            // Pre-rename builds kept the record CHAIN-WIDE ('material' next to
            // 'segments') and only the on/off switch per segment
            // ('material_freeze'). Lift that global onto the segments that had
            // it ticked, once, at load time - the mode is per segment now, and
            // a stale global must not resurrect itself through sync().
            function liftLegacyMaterial(d) {
                const g = normMaterial(d && d.material);
                if (!d || !Array.isArray(d.segments) || g.side === "off")
                    return;
                node._segments.forEach((s, i) => {
                    if (d.segments[i] && d.segments[i].material_freeze)
                        s.material = { ...g };
                });
            }
            function initialFromWidget() {
                try {
                    const d = JSON.parse(dataWidget.value || "{}");
                    if (Array.isArray(d.segments) && d.segments.length) {
                        node._segments = d.segments.map(normSeg);
                        if (!node._segments[0].width) node._segments[0].width = 768;
                        liftLegacyMaterial(d);
                    }
                    if (d.extend_params && typeof d.extend_params === "object")
                        node._extendParams = { ...node._extendParams, ...d.extend_params };
                    if (Number.isInteger(d.resume_from_segment) && d.resume_from_segment > 0)
                        node._resumeFrom = d.resume_from_segment;
                    if (Number.isInteger(d.mute_from_segment))
                        node._muteFrom = muteStopIndex(d.mute_from_segment,
                                                      node._segments.length);
                } catch {}
            }

            // ── wired reference-image sockets (Autogrow) ──
            // ComfyUI's Autogrow frontend grows one IMAGE socket per connect,
            // named '<group>.<label>' - group 'ref_images', labels
            // 'First_or_Ref_Image_0', 'Last_or_Ref_Image_1', 'Ref_Image_2'...
            // (the label is what reaches Python as a dict key, so the names
            // stay identifier-safe: '_or_' rather than '/'). Only CONNECTED
            // sockets count (inp.link != null); the unconnected spare slots
            // the frontend keeps around are noise. Every segment uses every
            // wired socket until that segment rules one out - exclusions live
            // in the segment's own ref_socket_off list, so the wire itself is
            // never touched.
            const SOCK_GROUP = "ref_images";
            const SOCK_DOT = SOCK_GROUP + ".";
            // the two sockets with a double role: slot 0 also serves as
            // segment 0's FL2VA first frame, slot 1 as the last frame. These
            // must match REF_SOCKET_NAMES in nodes/temporal_tile_editor.py.
            const SOCK_FIRST = "First_or_Ref_Image_0";
            const SOCK_LAST = "Last_or_Ref_Image_1";
            function wiredSocketNames() {
                const out = [];
                for (const inp of node.inputs || []) {
                    if (inp && inp.link != null && typeof inp.name === "string"
                            && inp.name.startsWith(SOCK_DOT))
                        out.push(inp.name.slice(SOCK_DOT.length));
                }
                return out;
            }
            function sockOffSet(seg) {
                return new Set(Array.isArray(seg.ref_socket_off)
                    ? seg.ref_socket_off : []);
            }
            // picker view model: name / display label / ✓ state / toggle.
            // `only` narrows to a single slot (FL2VA first/last frame rows).
            function sockViews(seg, only) {
                const off = sockOffSet(seg);
                return wiredSocketNames()
                    .filter((n) => !only || n === only)
                    .map((n) => ({
                        name: n,
                        // the socket's own label is already self-describing
                        // ('First_or_Ref_Image_0'), so show it verbatim
                        label: n,
                        included: !off.has(n),
                        toggle: () => toggleSock(seg, n),
                    }));
            }
            function toggleSock(seg, name) {
                if (!Array.isArray(seg.ref_socket_off)) seg.ref_socket_off = [];
                const i = seg.ref_socket_off.indexOf(name);
                if (i >= 0) seg.ref_socket_off.splice(i, 1);
                else seg.ref_socket_off.push(name);
                // the picker re-renders itself right after (see makeImagePicker):
                // toggling a socket never changes the reference MODE, only the
                // set of images it carries, so no full card rebuild is needed
                sync();
            }
            // a newly wired socket defaults the affected segments to 'load
            // images' (the mode that actually renders the reference row).
            // Only fires when the WIRED SET changes, so a deliberate 'no
            // reference' pick survives until another socket shows up. The
            // first observation is recorded, not applied - a loaded workflow
            // keeps its saved modes.
            function applySocketDefaults() {
                const names = wiredSocketNames();
                const sig = names.join("|");
                if (node._sockSig === undefined) { node._sockSig = sig; return; }
                if (sig === node._sockSig) return;
                const prev = new Set(node._sockSig ? node._sockSig.split("|") : []);
                node._sockSig = sig;
                if (!names.some((n) => !prev.has(n))) return;
                let changed = false;
                for (const s of node._segments) {
                    if (!s.ref_source || s.ref_source === "none") {
                        s.ref_source = "manual";
                        changed = true;
                    }
                }
                if (changed) sync();
            }
            // re-evaluate shortly after a wire changes: the Autogrow frontend
            // adds its replacement socket inside a requestAnimationFrame, so
            // reading immediately would still see the pre-connect set
            const prevConnChange = node.onConnectionsChange;
            node.onConnectionsChange = function (type, index, connected, linkInfo) {
                const r = prevConnChange?.apply(this, arguments);
                if (type === (window.LiteGraph?.INPUT ?? 1))
                    setTimeout(() => {
                        try { applySocketDefaults(); renderAll(); } catch {}
                    }, 60);
                return r;
            };

            // ── small builders ──
            function row(...children) {
                return el("div", { class: "mmh3tte-row" }, ...children);
            }
            function lbl(text) { return el("span", { class: "mmh3tte-lbl" }, text); }
            function makeSelect(options, value, onchange) {
                const sel = el("select", { class: "mmh3tte-sel" });
                for (const o of options) {
                    const v = typeof o === "string" ? o : o.value;
                    const t = typeof o === "string" ? uistr(o) : o.label;
                    sel.appendChild(el("option", { value: v }, t));
                }
                sel.value = value ?? (typeof options[0] === "string" ? options[0] : options[0].value);
                if (onchange) sel.addEventListener("change", () => onchange(sel.value, sel));
                return sel;
            }
            function makeNum(value, attrs = {}, onchange) {
                const inp = el("input", { type: "number", class: "mmh3tte-inp", value: value ?? 0, ...attrs });
                if (onchange) inp.addEventListener("change", () => onchange(inp));
                return inp;
            }

            function chipsEditor(seg, srcSel) {
                const picker = makeImagePicker({
                    multiple: true,
                    selected: seg.ref_images,
                    // wired Ref_Image_N sockets ride along as "dummy" chips
                    sockets: () => sockViews(seg),
                    // only Ref2VA actually feeds these images to the model as
                    // <Picture i> blocks, so only there does the row order
                    // equal the prompt's picture numbering
                    ordinals: seg.mode === "Ref2VA",
                    onChanged: (sel) => {
                        seg.ref_images = sel;
                        // clearing the last image drops the reference mode back
                        // to 'no reference' (same normalization the python node
                        // applies). For later segments the visible mode select
                        // would otherwise keep claiming 'load images', so mirror
                        // the flip and rebuild the card (the whole reference
                        // block is mode-dependent). Segment 0 has no such
                        // select - it just shows an empty picker.
                        if (!sel.length && seg.ref_source === "manual") {
                            seg.ref_source = "none";
                            sync();
                            if (srcSel) {
                                srcSel.value = "none";
                                picker.close?.();
                                renderAll();
                            }
                            return;
                        }
                        sync();
                    },
                });
                return row(lbl(uistr("ref images")), picker.root);
            }

            // single-image slot with thumbnail preview (first/last frame).
            // `sockName` is the wired socket that doubles as this frame when
            // no file is picked (SOCK_FIRST = first frame, SOCK_LAST = last
            // frame); its chip shows up in the row and its ✕ excludes it per
            // segment.
            function imageSlot(seg, labelText, value, onpick, sockName) {
                const picker = makeImagePicker({
                    multiple: false,
                    selected: value ? [value] : [],
                    sockets: () => sockViews(seg, sockName),
                    onChanged: (sel) => { onpick(sel[0] || ""); sync(); },
                });
                return row(lbl(uistr(labelText)), picker.root);
            }

            function fileSlot(labelText, value, list, onpick) {
                const sel = makeSelect(["", ...(list || [])], value || "", v => { onpick(v); sync(); });
                sel.firstChild.textContent = uistr("(none)");
                return row(lbl(uistr(labelText)), sel);
            }

            // audio reference mode select + the file slot that only appears in
            // that mode ('prev' / 'initial' reference the previous / first
            // segment's soundtrack directly at the LATENT level - no VAE
            // round-trip, no audio_vae needed; 'bgm' takes this segment's
            // soundtrack out of the audio_BGM socket, cut at the segment's own
            // position on the chain and frozen by the audio mask, so the chain
            // ends up with the music itself as its soundtrack). Segment 0 has
            // no earlier audio, so it only offers none/load/bgm.
            function audioModeRows(seg, first, i) {
                let opts = [
                    { value: "none", label: uistr("no reference") },
                    { value: "load", label: uistr("load audio") },
                    { value: "bgm", label: uistr("auto crop bgm") },
                ];
                if (!first) {
                    opts.push({ value: "prev", label: uistr("previous audio") });
                    opts.push({ value: "initial", label: uistr("initial audio") });
                }
                let active = seg.ref_audio_mode || "none";
                if (seg.mode === "FL2VA") {
                    // A FL2VA segment's soundtrack is driven only by the frozen
                    // BGM slice; the Ref2VA multi-modal <Audio j> reference path
                    // (load / prev / initial) is NEVER built for FL2VA segments,
                    // so those dead options are hidden. A stale value is reset.
                    opts = opts.filter(o => o.value === "none" || o.value === "bgm");
                    if (!opts.some(o => o.value === active)) {
                        active = "none";
                        seg.ref_audio_mode = "none";
                    }
                }
                const rows = [row(lbl(uistr("audio reference mode")),
                    makeSelect(opts, active,
                        v => { seg.ref_audio_mode = v; sync(); renderAll(); }))];
                if (active === "load") {
                    rows.push(fileSlot("ref audio", seg.ref_audio, _media.audios,
                        v => { seg.ref_audio = v; }));
                }
                if (active === "bgm") {
                    // Where this segment lands on the song. The segments before
                    // it own everything earlier on the chain's timeline, and the
                    // slice it takes starts one CARRIED TAIL earlier - the tail
                    // is re-covered so the music stays continuous across the
                    // seam. Same frame arithmetic the Extend node replays (see
                    // gridSplitNearest). Only the END of a slice is ever pushed
                    // past the previous segment; the START moves backwards.
                    let cum = 0;
                    for (let k = 0; k < i; k++)
                        cum += (node._segments[k].new_frames || 0);
                    const tail = (!first && seg.overlap_frames != null)
                        ? (seg.overlap_frames <= 0
                            ? 0 : snap17n5Up(seg.overlap_frames))
                        : (first ? 0 : snap17n5Up(node._extendParams.tail_frames || 39));
                    // a HARD CUT (tail 0) is not a tail of 0 either: the node
                    // realizes it as the smallest legal seam (see
                    // controlWindowFrames / the python seam_split), so the
                    // music is framed from that split - starting it at the
                    // literal boundary would frame it a few frames late and
                    // short, i.e. desynced from the picture.
                    const start = (!first && cum > 0)
                        ? gridSplitNearest(cum, tail || 0) : 0;
                    const end = cum + (seg.new_frames || 0);
                    const re = cum - start;
                    rows.push(row(lbl(uistr("bgm range")),
                        el("span", { class: "mmh3tte-hint" },
                            `${fmtSec(start)}s – ${fmtSec(end)}s` +
                            (re > 0
                                ? ` · ${uistr("re-covers the previous")} ${fmtSec(re)}s`
                                : ""))));
                    if (!first && cum > 0 && start === 0)
                        rows.push(row(el("span", {
                            class: "mmh3tte-hint", style: "color:#e6a046" },
                            uistr("the carried tail covers the whole chain - the music restarts at the beginning of the song"))));
                }
                return rows;
            }

            // Reference-video source for a Ref2VA segment: a picked file
            // ('load video') or a slice of the wired ref_video_input socket
            // ('auto crop input ref', analogous to auto crop bgm - each segment
            // takes the span covering its absolute place on the chain's
            // timeline, so abutting slices reassemble the input video).
            function refVideoRows(seg, i) {
                const first = i === 0;
                const active = seg.ref_video_mode || "load";
                const rows = [row(lbl(uistr("ref video mode")),
                    makeSelect([
                        { value: "load", label: uistr("load video") },
                        { value: "auto_crop", label: uistr("auto crop input ref") },
                    ], active, v => { seg.ref_video_mode = v; sync(); renderAll(); }))];
                if (active === "load") {
                    rows.push(fileSlot("ref video", seg.ref_video, _media.videos,
                        v => { seg.ref_video = v; }));
                    rows.push(fileSlot("video audio", seg.ref_video_audio, _media.audios,
                        v => { seg.ref_video_audio = v; }));
                } else {
                    // auto crop input ref: the span comes from the chain's
                    // timeline (sum of preceding new_frames as start, this
                    // segment's own new_frames as length) - the same integers
                    // the Extend node replays. No file picker: the frames are
                    // wired in via the node's 'ref_video_input' socket.
                    let cum = 0;
                    for (let k = 0; k < i; k++)
                        cum += (node._segments[k].new_frames || 0);
                    const start = cum;
                    const end = cum + (seg.new_frames || 0);
                    rows.push(row(lbl(uistr("auto crop range")),
                        el("span", { class: "mmh3tte-hint" },
                            `${fmtSec(start)}s – ${fmtSec(end)}s · ${uistr("slots into the wired ref_video_input")}`)));
                    // ...but only while the segment still has to be generated:
                    // a LOCKED segment is not sampled again, so the Extend node
                    // never reads the socket for it (python scopes the same
                    // check to segments from resume on) - warning about a
                    // missing video there would be stale.
                    const refLocked = i < (node._resumeFrom || 0);
                    if (first && cum === 0 && !refLocked)
                        rows.push(row(el("span", { class: "mmh3tte-hint" },
                            uistr("a video must be connected to the node's ref_video_input input"))));
                }
                return rows;
            }

            // ── fun control (MiniMax H3 Fun ControlNet) ──
            // The Tile Editor only says WHICH segments are controlled, how
            // strongly and over which sigma range; the Extend node slices the
            // window out of the wired 'fun_control_video' and mounts one Fun
            // ControlNet patch per segment. The window mirrors the Extend
            // node's replay exactly - the same integer arithmetic the BGM row
            // uses (gridSplitNearest == snap_split_frame): the piece RE-covers
            // the previous segment's last `tail` frames, so the control slice
            // starts one carried tail BEFORE the boundary, and a tail covering
            // the whole chain clamps it to 0 (the control video restarts).
            // A hard cut (overlap 0) is not a tail of 0 anywhere: the node
            // realizes the smallest legal seam for it too, and the BGM row and
            // this one both go through gridSplitNearest for every tail value.
            function controlRows(seg, i) {
                const first = i === 0;
                const active = seg.control_mode || "off";
                const rows = [row(lbl(uistr("control mode")),
                    makeSelect([
                        { value: "off", label: uistr("no control") },
                        { value: "auto_crop", label: uistr("auto crop input control") },
                        { value: "whole", label: uistr("whole control video") },
                    ], active, v => { seg.control_mode = v; sync(); renderAll(); }))];
                if (active === "off") return rows;

                const sInp = makeNum(seg.control_strength ?? 1,
                    { step: "0.01", min: "0", style: "width:80px" }, () => {
                        const v = parseFloat(sInp.value);
                        seg.control_strength = Number.isFinite(v) ? Math.max(0, v) : 1;
                        // write the clamped value back: sync() does not rebuild
                        // the card, so a typed negative would otherwise keep
                        // showing while the node runs with 0
                        sInp.value = String(seg.control_strength);
                        sync();
                    });
                sInp.title = uistr("0 disables the control for this segment");
                rows.push(row(lbl(uistr("control strength")), sInp));

                // start / end: sigma percentages of the model's own schedule.
                // Both boxes are rewritten from the normalized values (python
                // clamps too), so the shown range can never disagree with what
                // the patch activates on.
                const pct = (v, d) => {
                    const f = parseFloat(v);
                    return Number.isFinite(f) ? Math.min(1, Math.max(0, f)) : d;
                };
                const p0 = makeNum(seg.control_start ?? 0,
                    { step: "0.05", min: "0", max: "1", style: "width:80px" },
                    () => {
                        seg.control_start = pct(p0.value, 0);
                        if (seg.control_end < seg.control_start)
                            seg.control_end = seg.control_start;
                        p0.value = String(seg.control_start);
                        p1.value = String(seg.control_end);
                        sync();
                    });
                const p1 = makeNum(seg.control_end ?? 1,
                    { step: "0.05", min: "0", max: "1", style: "width:80px" },
                    () => {
                        seg.control_end = pct(p1.value, 1);
                        if (seg.control_end < seg.control_start)
                            seg.control_end = seg.control_start;
                        p0.value = String(seg.control_start);
                        p1.value = String(seg.control_end);
                        sync();
                    });
                const sigmaHint = el("span", { class: "mmh3tte-hint" },
                    uistr("0–1 sigma percentage of the model's own schedule (native node semantics)"));
                rows.push(row(lbl(uistr("control start")), p0, lbl(uistr("control end")), p1, sigmaHint));

                let cum = 0;
                for (let k = 0; k < i; k++)
                    cum += (node._segments[k].new_frames || 0);
                const nf = seg.new_frames || 0;
                const tail = first ? 0
                    : (seg.overlap_frames != null
                        ? (seg.overlap_frames <= 0 ? 0 : snap17n5Up(seg.overlap_frames))
                        : snap17n5Up(node._extendParams.tail_frames || 39));
                const whole = active === "whole";
                const [start, end] = controlWindowFrames(first, cum, tail, nf,
                                                         whole);
                // only an auto-crop slice re-covers: 'whole' is the control
                // video's own head, so nothing of the previous segment is
                // re-covered by it
                const re = whole ? 0 : (cum - start);
                rows.push(row(lbl(uistr("control range")),
                    el("span", { class: "mmh3tte-hint" },
                        `${fmtSec(start)}s – ${fmtSec(end)}s · ${uistr("slots into the wired fun control video")}` +
                        (re > 0
                            ? ` · ${uistr("re-covers the previous")} ${fmtSec(re)}s`
                            : ""))));
                if (whole)
                    rows.push(row(el("span", { class: "mmh3tte-hint" },
                        uistr("control applies from the control video's first frame"))));
                else if (!first && cum > 0 && start === 0)
                    rows.push(row(el("span", {
                        class: "mmh3tte-hint", style: "color:#e6a046" },
                        uistr("the carried tail covers the whole chain - the control restarts at the beginning of the video"))));
                // The assets live on the node's own sockets, so a controlled
                // segment with nothing wired would only fail at queue time -
                // say it here instead. An unconnected socket is either absent
                // or has no link (the same test wiredSocketNames uses).
                // A locked segment is out of scope: control only applies to
                // segments from resume on (python builds ctrl_segs the same
                // way), so the warning is left off while it cannot apply.
                const ctrlLocked = i < (node._resumeFrom || 0);
                const ctrlInp = node.inputs?.find(x => x.name === "fun_control_video");
                if (!ctrlLocked && (!ctrlInp || ctrlInp.link == null))
                    rows.push(row(el("span", {
                        class: "mmh3tte-hint", style: "color:#e6a046" },
                        uistr("a video must be connected to the node's fun_control_video input"))));
                return rows;
            }

            // ── the ref_video_input wire decides whether a card may offer the
            // mode at all ──
            // No wire = no strip: a segment asking for one would be refused by
            // the Extend node at queue time (the video the strip is cut from
            // does not exist), so the panel keeps the two halves in step itself
            // instead of letting a run die on it later. The side is forced to
            // 'off' - what the card shows is what the run does - and the choice
            // is remembered in a WeakMap so putting the video back restores it:
            // swapping a reference video means disconnecting, and that is not a
            // decision to stop using the mode. The card's select reads the same
            // wire through materialSideOptions, so both halves agree by
            // construction rather than by coincidence.
            function materialSocketWired() {
                const inp = node.inputs?.find(x => x.name === "ref_video_input");
                return !!(inp && inp.link != null);
            }
            function applyMaterialSocketGate() {
                if (!node._matSideKeep) node._matSideKeep = new WeakMap();
                const wired = materialSocketWired();
                let changed = false;
                for (const s of node._segments) {
                    const m = segMaterial(s);
                    const kept = node._matSideKeep.get(s);
                    if (wired) {
                        // the socket took the side away; give it back
                        if (kept && m.side === "off") {
                            s.material = { ...m, side: kept };
                            node._matSideKeep.delete(s);
                            changed = true;
                        }
                    } else if (m.side !== "off") {
                        node._matSideKeep.set(s, m.side);
                        s.material = { ...m, side: "off" };
                        changed = true;
                    }
                }
                if (changed) sync();
            }

            // One segment's direct-reference controls. Every field is THIS
            // segment's own: 'side' is the switch ('off' = the segment
            // generates the whole canvas as usual), and turning it on needs
            // nothing from the segment before it - its piece is spliced over
            // the carried tail too, so the strip starts inside this segment's
            // overlap rather than at its first sampled frame. The window needs
            // no field either: it is the part of the chain's timeline the
            // segment's piece covers, i.e. exactly the same arithmetic the
            // Extend node replays (controlWindowFrames is the shared mirror of
            // control_slice_window).
            // The switch is offered only while the strip has an asset to be cut
            // from: with ref_video_input unwired the select holds nothing but
            // 'no material' (materialSideOptions), which is also what
            // applyMaterialSocketGate has already written into the record - so
            // the value below is always one of the options, and the card can
            // never display a direction the run would refuse.
            function materialRows(seg, i, first) {
                const m = segMaterial(seg);
                const rows = [el("div", { class: "mmh3tte-sub" },
                                 uistr("direct reference in frame"))];
                const sideOpts = materialSideOptions(materialSocketWired());
                const sideSel = makeSelect(sideOpts.map(s => ({
                    value: s,
                    label: s === "off" ? uistr("no material")
                                       : uistr("material " + s),
                })), sideOpts.includes(m.side) ? m.side : "off", v => {
                    seg.material = { ...segMaterial(seg), side: v };
                    sync();
                    renderAll();
                });
                sideSel.title = uistr("where the strip of the ref_video_input video sits on this segment's canvas - 'off' = this segment does not use the mode");
                rows.push(row(lbl(uistr("splice side")), sideSel));
                if (m.side === "off") {
                    // while it is off this is the only row: the rest would be
                    // dead weight on a segment that does not use the mode (the
                    // 'off' the select shows IS the whole message)
                    return rows;
                }

                const exInp = makeNum(m.expose, { step: "32", min: "0",
                                                  style: "width:80px" }, () => {
                    const v = parseInt(exInp.value, 10);
                    seg.material = { ...segMaterial(seg),
                        expose: Number.isFinite(v)
                            ? Math.max(0, Math.round(v / 32) * 32) : 32 };
                    // write the quantized value back: sync() alone does not
                    // rebuild the card, so a typed 48 would otherwise keep
                    // showing 48 while the node runs with 32
                    exInp.value = String(segMaterial(seg).expose);
                    sync();
                    renderAll();
                });
                exInp.title = uistr("free band at the seam (px, multiple of 32 = one token column)");
                rows.push(row(lbl(uistr("expose at the seam")), exInp,
                    el("span", { class: "mmh3tte-hint" },
                        uistr("the rest of the strip stays frozen; this band lets the model synthesize the transition"))));

                // 'cut at the split': the second switch of the mode, and the
                // only one that changes WHEN the strip is there. The cut itself
                // happens at the HIGH -> LOW handoff in the node, so it is an
                // on/off here - the split sigma IS the cut point (the existing
                // SplitSigmas controls set it). Left off by the node when the
                // segment has no split or shares itself with fun control, and
                // it says so in the log.
                const cutChk = el("input", { type: "checkbox" });
                cutChk.checked = m.cut_at_split === true;
                cutChk.title = uistr("drop the strip when the HIGH stage hands over to the LOW stage: the reference builds the structure, then the detail stage finishes the picture from the prompt alone - nothing of the strip reaches the output either way. The refinement pass never sees the strip anyway (it always cuts it first), so this switch is about the FIRST pass only. Needs a LOW stage (a split to cut at) and no fun control on this segment; the node prints which way it went");
                cutChk.addEventListener("change", () => {
                    seg.material = { ...segMaterial(seg),
                                     cut_at_split: cutChk.checked };
                    sync();
                    renderAll();
                });
                rows.push(row(el("label", { class: "mmh3tte-chk" },
                    cutChk, uistr("cut at the split"))));

                let cum = 0;
                for (let k = 0; k < i; k++)
                    cum += (node._segments[k].new_frames || 0);
                const tail = first ? 0
                    : (seg.overlap_frames != null
                        ? (seg.overlap_frames <= 0 ? 0 : snap17n5Up(seg.overlap_frames))
                        : snap17n5Up(node._extendParams.tail_frames || 39));
                // The span the strip DEPICTS is the piece [start, end) - row r
                // of the strip is local frame r. The node reads exactly this
                // span; the 17-frame group grid it also has to sit on is what
                // controlWindowFrames now expresses for every tail including
                // the hard cut, and on a chain off the usual 17k+5 clock the
                // node may prepend the few frames that anchor the encode
                // window (material_window's `lead`), dropping those rows again
                // before the splice - so the numbers below stay the ones the
                // user is asking about.
                const [w0, w1] = controlWindowFrames(first, cum, tail,
                                                     seg.new_frames || 0, false);
                rows.push(row(lbl(uistr("material window")),
                    el("span", { class: "mmh3tte-hint" },
                        `${fmtSec(w0)}s – ${fmtSec(w1)}s · ${uistr("slots into the wired ref_video_input")}` +
                        (!first && cum > w0
                            ? ` · ${uistr("re-covers the previous")} ${fmtSec(cum - w0)}s`
                            : ""))));
                // The carried tail is part of the piece, so the strip is
                // written over it as well - that is what makes 'the previous
                // segment did not use the mode' work; the window row above is
                // the whole of what the panel says about it.
                // A chain where only SOME segments use the mode is legal (each
                // segment decides for itself, nothing has to agree): the chain
                // summary above the cards names those segments, which is why
                // this card does not repeat the list.
                // one wire, one role: the strip and an 'auto crop input ref'
                // reference video cannot share ref_video_input (the Extend
                // node refuses the combination)
                if (seg.ref_video_mode === "auto_crop")
                    rows.push(row(el("span", {
                        class: "mmh3tte-hint", style: "color:#e6a046" },
                        uistr("direct reference in frame conflicts with 'auto crop input ref' on this segment: the ref_video_input socket carries ONE video"))));
                return rows;
            }

            // ── segment card ──
            function renderSeg(seg, i) {
                const first = i === 0;
                // lock model: segments 0..resume-1 are locked (reused from the
                // session dir on disk); execution starts AT the resume index,
                // continuing from the last locked segment's merged latent.
                const locked = i < node._resumeFrom;
                const sampled = i <= node._lastSeg;
                // mute model: ONE stop point for the whole chain. This segment
                // is CLOSED when it sits at or behind it - the run ends before
                // it, so it is neither sampled nor reused (nothing of it is
                // read from disk: it is not the locked kind of frozen)
                const stop = muteStopIndex(node._muteFrom, node._segments.length);
                const closed = segIsClosed(stop, i);
                const card = el("div", { class: "mmh3tte-seg" + (i === node._openSeg ? " open" : "") +
                    (locked ? " locked-seg" : "") + (closed ? " closed-seg" : "") });

                const modeSel = makeSelect(
                    first ? ["FL2VA", "Ref2VA"] : ["Ref2VA", "FL2VA"],
                    seg.mode,
                    v => {
                        seg.mode = v;
                        if (!first && v === "FL2VA") seg.ref_source = "none";
                        // switching a later segment to Ref2VA implies the user
                        // wants to pick reference images — surface the picker
                        // instead of leaving "no reference" active
                        if (!first && v === "Ref2VA" && seg.ref_source === "none") seg.ref_source = "manual";
                        sync(); renderAll();
                    });

                // live duration badge in the card head, updated in place by the
                // duration row below (no card rebuild — spinner clicks on the
                // inputs must never land on a detached node)
                // accumulate the cumulative frames: frames from all segments up to
                // AND including this one. This is the end frame index on the
                // chain's timeline (start for the next is this value + 1).
                let cumFrames = 0;
                for (let k = 0; k <= i; k++)
                    cumFrames += (node._segments[k].new_frames || 0);
                const durChip = el("span", { class: "seg-dur" }, `${seg.new_frames}f · ${fmtSec(seg.new_frames)}s`);
                const cumChip = el("span", { class: "seg-dur", title: uistr("cumulative frames up to this segment") },
                    `${uistr("total")} ${cumFrames}f · ${fmtSec(cumFrames)}s`);
                // register both chips so a frame-count change on any segment can
                // refresh every cumulative total (they shift when any earlier
                // segment changes)
                (node._durChips || (node._durChips = {}))[i] =
                    { dur: durChip, cum: cumChip };

                // mute button (left of the lock button): only a segment with
                // NO stored latent on disk gets one - the same condition that
                // disables the lock button (nothing to keep, nothing to
                // invalidate). Every click parks the stop point right AFTER
                // its own segment: on an ACTIVE segment that closes it and
                // every later one; on a CLOSED one it reopens this segment and
                // every earlier one and leaves the later ones closed - a
                // single click never wipes the chain (the summary line's
                // unmute button is the one-click full reopen). There is only
                // ever ONE stop point, so no older stop can come back from
                // behind this one.
                // The glyph is the plain letter 'M': a speaker emoji reads as
                // 'sound' rather than as 'not sampled this run', and the state
                // is carried by the COLOR anyway - gray once it is on, the
                // same slot the lock button uses for its amber.
                const muteBtn = !sampled ? el("button", {
                    class: "mmh3tte-btn" + (closed ? " muteon" : ""),
                    title: closed
                        ? uistr("unmute this segment and every earlier one — the chain runs through it and stops right after it (later segments stay closed; on the last segment the whole chain reopens)")
                        : uistr("mute: stop the chain before this segment — this segment and every later one are closed (not sampled in this run)"),
                    onclick: () => {
                        // the stop point always ends up right after THIS
                        // segment, so a closed click walks it forward
                        // instead of clearing it (on the last segment the
                        // value normalizes to 'no stop': run everything)
                        node._muteFrom = closed
                            ? muteStopIndex(i + 1, node._segments.length)
                            : i;
                        sync(); renderAll();
                    },
                }, "M") : null;

                // delete: segment i and ALL later ones lose their stored
                // results (the chain shifted), so it asks first - UNLESS the
                // segment has no stored latent on disk (the same condition as
                // the mute / lock buttons), in which case nothing is lost and
                // it goes straight away. That condition comes from the session
                // metadata, which a click right after loading the workflow may
                // not have read yet: read it NOW instead of guessing, and keep
                // the dialog when the session cannot be read at all.
                const delBtn = (i > 0 && !locked) ? el("button", {
                    class: "mmh3tte-btn warn",
                    title: uistr("del"),
                    onclick: async () => {
                        let hasStored = i <= node._lastSeg;
                        if (!node._metaKnown)
                            hasStored = await loadSessionMeta()
                                ? (i <= node._lastSeg) : true;
                        if (!hasStored) { removeSeg(i); return; }
                        const later = node._segments.length - i - 1;
                        const msg = uistr("delete segment") + ` ${i}` +
                            (later > 0 ? ` + ${later} ` + uistr("later segment(s)") : "") +
                            "?\n" + uistr("stored results become invalid");
                        confirmDialog(msg, uistr("confirm delete"), () => removeSeg(i));
                    },
                }, "✕") : null;

                // lock button (left of delete): locks this segment and every
                // one before it; only enabled for segments whose outputs are
                // already stored in the session dir (sampled). Clicking the
                // lock of a locked segment unlocks IT and every one AFTER it;
                // segments before it stay locked.
                const lockBtn = el("button", {
                    class: "mmh3tte-btn" + (locked ? " lockon" : ""),
                    title: !sampled ? uistr("not sampled yet")
                        : (locked ? uistr("unlock") : uistr("lock")),
                    ...(sampled ? {} : { disabled: "" }),
                    onclick: () => {
                        if (!sampled) return;
                        node._resumeFrom = locked ? i : i + 1;
                        sync(); renderAll(); renderPreviews();
                    },
                }, locked ? "🔒" : "🔓");

                const head = el("div", { class: "seg-head", onclick: (ev) => {
                    if (ev.target.closest("select,button")) return;
                    node._openSeg = (node._openSeg === i) ? -1 : i;
                    renderAll();
                } },
                    el("span", { class: "seg-name" }, first ? uistr("seg 0 (first)") : `${uistr("seg")} ${i}`),
                    modeSel,
                    durChip,
                    cumChip,
                    // the stop point carries the same kind of marker the
                    // resume point does; the segments behind it say what they
                    // are: closed for this run
                    (i === stop)
                        ? el("span", { class: "mmh3tte-badge closed", title: uistr("the chain stops before this segment") }, "⏹")
                        : (closed ? el("span", { class: "mmh3tte-badge closed", title: uistr("closed · not sampled in this run") }, uistr("closed")) : null),
                    (!locked && node._resumeFrom > 0 && i === node._resumeFrom)
                        ? el("span", { class: "mmh3tte-badge", title: uistr("resume here") }, "⟳") : null,
                    el("span", { class: "spacer" }),
                    muteBtn,
                    lockBtn,
                    delBtn);

                const body = el("div", { class: "seg-body" });

                // duration: ONE input + sec/frames unit toggle. No two-way
                // binding between two inputs (that coupling caused the
                // "value bounces back" bugs). Rules:
                //  - frames mode: the input IS the authority; spinner steps 17
                //    (valid counts sit 17 apart on either lattice) and the value
                //    is snapped to this segment's lattice — safe to rewrite,
                //    the result is always on-grid
                //  - seconds mode: while the user stays on the card the typed
                //    number is kept VERBATIM in the input, and the frame count
                //    it maps to is shown right next to it. A re-render (switching
                //    to another segment and back, or toggling the unit) rebuilds
                //    the box from seg.new_frames, so it must recover the TYPED
                //    second — durSecFromFrames() does, on either lattice, because
                //    the mapping is definite (table above fmtSec()).
                const unit = () => (node._durUnit === "frames") ? "frames" : "time";
                const snapF = (f) => first ? snap17n5(f) : snap17(f);
                const minF = first ? 5 : 17;
                const durInp = el("input", { type: "number", class: "mmh3tte-inp" });
                const durHint = el("span", { class: "mmh3tte-hint" });
                // the DEFINITE mapping, spelled out in the tooltip
                durInp.title = uistr(first ? "seconds → frames (17n+5)"
                    : "seconds → frames (×17)") + ":  " + durTableText(first);
                // refresh step/min/hint/badge. NEVER writes durInp.value —
                // after the initial render no code path touches the input, so
                // nothing can ever bounce the user's typed/clicked value back.
                function refreshDur() {
                    if (unit() === "frames") {
                        durInp.step = "17"; durInp.min = String(minF);
                    } else {
                        durInp.step = "1"; durInp.min = "1";
                    }
                    // frames are what the model actually generates, so they are
                    // always spelled out — in seconds mode as the conversion
                    durHint.textContent = `${seg.new_frames} ${uistr("frames")} = ${fmtSec(seg.new_frames)}s`;
                    // a frame-count change on ANY segment shifts every later
                    // segment's cumulative total, so refresh all chips here
                    let cf = 0;
                    for (let k = 0; k < node._segments.length; k++) {
                        cf += (node._segments[k].new_frames || 0);
                        const chip = node._durChips && node._durChips[k];
                        if (!chip) continue;
                        chip.dur.textContent =
                            `${node._segments[k].new_frames}f · ${fmtSec(node._segments[k].new_frames)}s`;
                        chip.cum.textContent = `${uistr("total")} ${cf}f · ${fmtSec(cf)}s`;
                    }
                }
                function applyDur(raw) {
                    if (unit() === "time") {
                        // seconds -> frames through the definite mapping
                        const s = parseFloat(raw);
                        if (Number.isFinite(s) && s > 0) seg.new_frames = secToFrames(s, first);
                    } else {
                        // frames mode: snapped to this segment's lattice
                        const f = parseInt(raw);
                        seg.new_frames = snapF(Number.isFinite(f) ? f : minF);
                    }
                    sync();
                    refreshDur();
                }
                durInp.addEventListener("change", () => applyDur(durInp.value));
                const btnTime = el("button", {
                    class: "mmh3tte-btn" + (unit() === "time" ? " primary" : ""),
                    title: uistr("edit in seconds"),
                    onclick: () => { if (node._durUnit !== "time") { node._durUnit = "time"; renderAll(); } },
                }, uistr("sec"));
                const btnFrames = el("button", {
                    class: "mmh3tte-btn" + (unit() === "frames" ? " primary" : ""),
                    title: uistr("edit in frames"),
                    onclick: () => { if (node._durUnit !== "frames") { node._durUnit = "frames"; renderAll(); } },
                }, uistr("frames"));
                body.appendChild(row(lbl(uistr("duration")), durInp, btnTime, btnFrames,
                    el("span", { class: "mmh3tte-hint" }, first ? uistr("frames (17n+5)") : uistr("new frames (×17)")),
                    durHint));
                durInp.value = (unit() === "frames") ? seg.new_frames
                    : durSecFromFrames(seg.new_frames, first);
                refreshDur();

                // seed
                const seedInp = makeNum(seg.seed ?? 0, { step: "1", style: "width:110px" }, () => {
                    seg.seed = Number(BigInt(Math.round(Number(seedInp.value) || 0)) & 0xFFFFFFFFFFFFFFFFn);
                    sync();
                });
                body.appendChild(row(lbl(uistr("seed")), seedInp,
                    el("button", { class: "mmh3tte-btn", title: "random seed", onclick: () => {
                        seg.seed = Math.floor(Math.random() * 0xffffffff);
                        sync(); renderAll();
                    } }, "🎲")));

                // prompts
                const pTa = el("textarea", { class: "mmh3tte-area", placeholder: uistr("segment prompt") });
                pTa.value = seg.prompt || "";
                pTa.addEventListener("input", () => { seg.prompt = pTa.value; sync(); });
                body.appendChild(row(lbl(uistr("prompt")), pTa));
                const nTa = el("textarea", { class: "mmh3tte-area", placeholder: uistr("negative (optional)") });
                nTa.value = seg.negative || "";
                nTa.addEventListener("input", () => { seg.negative = nTa.value; sync(); });
                body.appendChild(row(lbl(uistr("negative")), nTa));

                // conditioning specifics
                if (first) {
                    // resolution (locked when session has stored latents)
                    const wInp = makeNum(seg.width || 768, { step: "32", min: "256" }, () => {
                        seg.width = Math.max(256, Math.round((parseInt(wInp.value) || 768) / 32) * 32);
                        sync();
                    });
                    const hInp = makeNum(seg.height || 768, { step: "32", min: "256" }, () => {
                        seg.height = Math.max(256, Math.round((parseInt(hInp.value) || 768) / 32) * 32);
                        sync();
                    });
                    const resRow = row(lbl(uistr("resolution")), wInp,
                        el("span", { class: "mmh3tte-hint" }, "×"), hInp,
                        el("span", { class: "mmh3tte-hint" }, uistr("multiple of 32 · locked while reusing stored segments")));
                    body.appendChild(resRow);
                    node._resRow = { wInp, hInp, row: resRow };

                    if (seg.mode === "FL2VA") {
                        body.appendChild(imageSlot(seg, "first frame", seg.fl2va_first_image,
                            v => { seg.fl2va_first_image = v; }, SOCK_FIRST));
                        body.appendChild(imageSlot(seg, "last frame", seg.fl2va_last_image,
                            v => { seg.fl2va_last_image = v; }, SOCK_LAST));
                        // BGM / loaded audio are orthogonal to the frame mode,
                        // so a FL2VA first segment still gets the audio
                        // reference controls (none / load / auto crop bgm).
                        for (const r of audioModeRows(seg, first, i)) body.appendChild(r);
                    } else {
                        body.appendChild(chipsEditor(seg));
                        body.appendChild(row(lbl(uistr("ref size")),
                            makeSelect(["match", "max"], seg.ref_image_size || "match",
                                v => { seg.ref_image_size = v; sync(); })));
                        for (const r of refVideoRows(seg, i)) body.appendChild(r);
                        for (const r of audioModeRows(seg, first, i)) body.appendChild(r);
                    }
                } else {
                    const srcSel = makeSelect([
                        { value: "none", label: uistr("no reference") },
                        { value: "manual", label: uistr("load images") },
                        { value: "prev_frame", label: uistr("previous segment frame") },
                    ], seg.ref_source, v => { seg.ref_source = v; sync(); renderAll(); });
                    body.appendChild(row(lbl(uistr("image reference mode")), srcSel));

                    if (seg.mode === "FL2VA") {
                        body.appendChild(imageSlot(seg, "last frame (target ending)",
                            seg.fl2va_last_image,
                            v => { seg.fl2va_last_image = v; }, SOCK_LAST));
                        // Continuation FL2VA segments get the audio reference
                        // controls too (filtered to none/bgm by audioModeRows).
                        for (const r of audioModeRows(seg, first, i)) body.appendChild(r);
                    } else if (seg.ref_source === "manual") {
                        body.appendChild(chipsEditor(seg, srcSel));
                        body.appendChild(row(lbl(uistr("ref size")),
                            makeSelect(["match", "max"], seg.ref_image_size || "match",
                                v => { seg.ref_image_size = v; sync(); })));
                    } else if (seg.ref_source === "prev_frame") {
                        const fi = makeNum(seg.prev_frame_index ?? -1, { step: "1", min: "-1" }, () => {
                            seg.prev_frame_index = parseInt(fi.value) || -1; sync();
                        });
                        body.appendChild(row(lbl(uistr("frame #")), fi,
                            el("span", { class: "mmh3tte-hint" }, uistr("-1 = last frame of previous merged video (latent-exact, no decode)"))));
                    }
                    if (seg.mode === "Ref2VA") {
                        // MiniMaxH3ReferenceToVideo-style extra references,
                        // independent of the ref_source above
                        for (const r of refVideoRows(seg, i)) body.appendChild(r);
                        for (const r of audioModeRows(seg, first, i)) body.appendChild(r);
                    }
                }

                // ── direct reference in frame (this segment's own record) ──
                for (const r of materialRows(seg, i, first)) body.appendChild(r);

                // ── per-segment overlap (segments >= 1) ──
                // The ONLY per-segment continuation parameter: how many tail
                // frames this segment carries over. All other tail/fade/
                // anchor/overlap settings are global via the "MMH3 Temporal
                // Overlap Params" sub-node; segment 0 generates fresh.
                if (!first) {
                    if (seg.overlap_frames == null) seg.overlap_frames = 39;
                    body.appendChild(el("div", { class: "mmh3tte-sub" },
                        uistr("overlap (this segment)")));
                    const ofFallback = snap17n5Up(node._extendParams?.tail_frames || 39);
                    // a carried tail is a LENGTH on the 17n+5 lattice, while the
                    // hard cut (0) is the ABSENCE of one - and the two cannot
                    // share a single <input type=number>, because such a box
                    // steps from its `min` (the step base):
                    //   min="0"/step="17" reaches 0 by stepping down to -17
                    //   (clamped), but its ladder is 0, 17, 34... - values this
                    //   node never uses (17 -> 22, 34 -> 39) - and stepping UP
                    //   re-aligns an off-ladder value first, so 22 would jump
                    //   to 56. min="5"/step="17" gives the real ladder but can
                    //   never reach 0.
                    // So the box holds the length on the ladder and the hard cut
                    // gets the tick beside it: the box then only ever shows a
                    // value it can display, whatever the browser makes of an
                    // out-of-range number, and the flag survives.
                    const ofHint = el("span", { class: "mmh3tte-hint" });
                    let ofLen = seg.overlap_frames > 0
                        ? snap17n5Up(seg.overlap_frames) : ofFallback;
                    const ofInp = makeNum(ofLen, { step: "17", min: "5" });
                    ofInp.title = uistr("carried tail in pixel frames, snapped up to the 5, 22, 39, ... grid");
                    const ofChk = el("input", { type: "checkbox" });
                    const ofCut = el("label", { class: "mmh3tte-chk" },
                        ofChk, uistr("hard cut (0)"));
                    ofCut.title = uistr(OVERLAP_HARD_CUT_HINT);
                    function refreshOverlap() {
                        ofInp.disabled = ofChk.checked;
                        ofHint.textContent = ofChk.checked
                            ? uistr(OVERLAP_HARD_CUT_HINT)
                            : uistr("carried tail (17n+5)");
                    }
                    // ONE handler for both controls: the tick decides whether a
                    // tail exists at all, the box supplies its length, and
                    // overlapFramesValue() is the single place that turns the
                    // pair into the number the node runs with.
                    function applyOverlap() {
                        const v = overlapFramesValue(ofInp.value, ofChk.checked);
                        if (!ofChk.checked && v === 0) {
                            // a 0 typed into the box (or an emptied box) is the
                            // hard cut too: it is not a length the box can
                            // hold, so show it as the tick it is and put the
                            // length box back on the ladder
                            ofChk.checked = true;
                            ofInp.value = String(ofLen);
                        } else if (v > 0) {
                            ofLen = v;
                            // write the snapped length back, because sync() does
                            // not re-render this card: without it the box would
                            // keep showing the typed/spun number while the node
                            // ran with the snapped one
                            ofInp.value = String(v);
                        }
                        seg.overlap_frames = v;   // 0 = hard cut
                        refreshOverlap();
                        sync();
                    }
                    ofInp.addEventListener("change", applyOverlap);
                    ofChk.addEventListener("change", applyOverlap);
                    ofChk.checked = !(seg.overlap_frames > 0);
                    refreshOverlap();
                    body.appendChild(row(lbl(uistr("overlap frames")), ofInp, ofCut, ofHint));
                }

                // ── fun control (this segment) ──
                // Its own divider because it is ORTHOGONAL to everything above:
                // a Fun ControlNet is a MODEL-side patch, not conditioning, so
                // it applies whatever the reference / frame mode is, and it is
                // the only per-segment part of the feature (the assets - the
                // control video and the optional inpaint pair - are wired into
                // this node's own sockets).
                body.appendChild(el("div", { class: "mmh3tte-sub" },
                    uistr("fun control (this segment)")));
                for (const r of controlRows(seg, i)) body.appendChild(r);

                card.append(head, body);
                // locked segments are read-only: disable every control in the
                // card body (and the mode select in the head) so parameters
                // can't change until the user unlocks the segment. The lock
                // button itself (in the head) stays clickable for unlocking;
                // div-based picker triggers are blocked by the CSS above.
                if (locked) {
                    body.querySelectorAll("input, select, textarea, button")
                        .forEach(c => { c.disabled = true; });
                    modeSel.disabled = true;
                }
                return card;
            }

            // ── previews / resume panel ──
            let pvBox = null;
            // Invalidate stored results from a segment index onward (delete
            // flow): the server drops the session.json mapping entries >=
            // index and clamps last_segment (attempt FILES are kept on
            // disk), then the preview panel re-renders to the shrunken chain
            // and the lock buttons re-derive their "sampled" state.
            async function pruneStored(fromIndex) {
                const session = findW("session_name")?.value || "session1";
                const location = findW("storage_location")?.value || "temp";
                try {
                    await fetch(`/mmh3te/temporal_prune?location=${encodeURIComponent(location)}&session=${encodeURIComponent(session)}&from=${fromIndex}`);
                } catch { /* non-fatal: panel just keeps the previous view */ }
                await renderPreviews();
            }
            // Delete segment i (shared by the instant path and the dialog):
            // the chain shifts left, so every stored result from i on is
            // invalid - pruneStored() drops the session.json mapping for those
            // indices (attempt FILES are kept on disk) and the panel re-renders
            // to the shrunken chain. Two chain-level indices ride on the
            // segment order and shift with it: the resume point and the mute
            // stop point. Deleting the stop point itself is NOT a cancel: the
            // segment that shifts into its slot was closed as well, so the
            // closed tail stays closed and the boundary stays where it is (it
            // drops out only when the deleted segment was the last one - then
            // there is nothing left to close).
            async function removeSeg(i) {
                node._segments.splice(i, 1);
                if (node._resumeFrom > node._segments.length - 1)
                    node._resumeFrom = 0;
                if (node._muteFrom > i) node._muteFrom--;
                else if (node._muteFrom === i)
                    node._muteFrom = muteStopIndex(i, node._segments.length);
                sync(); renderAll();
                await pruneStored(i);
            }
            // Restart-with-wiped-session flow: when the stored results the
            // locked chain depends on no longer exist (temp dir wiped on
            // restart, session files deleted manually, ...), a persisted
            // resume_from_segment > 0 would leave those segments locked
            // FOREVER — the lock buttons disable themselves once the chain
            // no longer counts as "sampled" (_lastSeg < resume), so the user
            // can't even click unlock, and execution would fail with "no
            // stored merged latent". Detect the missing files and unlock
            // automatically instead.
            function autoUnlockStaleChain() {
                if (node._resumeFrom <= 0) return;
                node._resumeFrom = 0;
                sync();
                renderAll();
                const msg = uistr("locked segments' stored latents are missing — unlocked automatically");
                console.warn(TAG, msg);
                const toast = app.extensionManager?.toast;
                if (toast?.add) {
                    toast.add({ severity: "warn",
                                summary: uistr("MMH3 Temporal Tile Editor"),
                                detail: msg, life: 6000 });
                }
            }
            // resolution follows segment 0's lock state: editable only while
            // the WHOLE chain regenerates (resume point 0 - nothing on disk
            // is reused, so a new resolution is safe). As soon as any stored
            // segment is reused (resume point > 0) the latents must keep
            // their resolution, so the inputs lock. Called from renderAll
            // (after the cards rebuild, which resets the inputs to enabled)
            // so every code path - lock buttons, auto-unlock, config load -
            // stays in sync.
            function updateResLock() {
                if (!node._resRow) return;
                const locked = node._resumeFrom > 0;
                node._resRow.wInp.disabled = locked;
                node._resRow.hInp.disabled = locked;
                node._resRow.row.classList.toggle("locked", locked);
                node._resRow.row.style.opacity = locked ? "0.55" : "";
            }
            // Both columns are rebuilt with innerHTML = "" (the cards in
            // renderAll, the previews here), which resets their scrollTop to 0.
            // Locking / unlocking a segment re-renders BOTH, so without this the
            // panel jumped back to the top on every lock click. Wrap the
            // rebuild: capture the position before it starts and re-apply it
            // afterwards - twice, because an <img> only gets its height once it
            // decodes and a container that is briefly too short clamps the
            // value it is given.
            function preserveScroll(container) {
                const top = container.scrollTop;
                return () => {
                    container.scrollTop = top;
                    requestAnimationFrame(() => { container.scrollTop = top; });
                };
            }
            // Ask the server ONCE for this session's metadata and publish the
            // highest segment index stored on disk (_lastSeg). The lock
            // buttons and the delete fast path both key off it, and a click
            // can land before the preview panel has fetched anything, so they
            // call this instead of trusting the initial -1. Returns false when
            // the session could not be read - the caller then treats the state
            // as UNKNOWN and keeps asking for confirmation.
            async function loadSessionMeta() {
                const session = findW("session_name")?.value || "session1";
                const location = findW("storage_location")?.value || "temp";
                let data;
                try {
                    const resp = await fetch(`/mmh3te/temporal_session?location=${encodeURIComponent(location)}&session=${encodeURIComponent(session)}`);
                    data = await resp.json();
                } catch { return false; }
                if (!data || typeof data !== "object") return false;
                const meta = data.session || data.metadata || {};
                node._lastSeg = Number.isInteger(meta.last_segment) ? meta.last_segment : -1;
                node._metaKnown = true;
                return true;
            }
            async function renderPreviews() {
                if (!pvBox) return;
                const restoreScroll = preserveScroll(pvPanel);
                const session = findW("session_name")?.value || "session1";
                const location = findW("storage_location")?.value || "temp";
                if (node._sessLbl) node._sessLbl.textContent = session;
                pvBox.innerHTML = "";
                pvBox.appendChild(el("div", { class: "mmh3tte-muted" }, uistr("loading previews…")));
                let data, read = true;
                try {
                    const resp = await fetch(`/mmh3te/temporal_session?location=${encodeURIComponent(location)}&session=${encodeURIComponent(session)}`);
                    data = await resp.json();
                } catch { data = { exists: false, files: [], metadata: null }; read = false; }
                // only a READ session may publish _lastSeg as authoritative:
                // the delete fast path treats it as the truth once _metaKnown
                // is set, and a failed fetch must not read as "nothing stored"
                if (read && data && typeof data === "object") node._metaKnown = true;
                pvBox.innerHTML = "";
                if (!data.exists) {
                    // session dir gone entirely (e.g. temp wiped on restart)
                    autoUnlockStaleChain();
                    pvBox.appendChild(el("div", { class: "mmh3tte-muted" },
                        uistr("no stored run yet for this session — previews appear after executing MMH3 Temporal Extend Video")));
                    renderCacheRow(location);
                    restoreScroll();
                    return;
                }
                const sess = data.session || null;
                const meta = sess || data.metadata || {};
                const last = Number.isInteger(meta.last_segment) ? meta.last_segment : -1;
                // per-segment chosen-attempt file mapping (session.json):
                // each re-run of a segment stores a NEW attempt under a fresh
                // seq-named file, and the mapping points at the chosen one.
                // Old sessions without session.json fall back to the legacy
                // index-named files.
                const segMap = (sess && Array.isArray(sess.segments)) ? sess.segments : null;
                const fileFor = (i, kind, legacy) => {
                    if (segMap) {
                        const entry = segMap.find((s) => s.index === i);
                        const f = entry && entry.files && entry.files[kind];
                        if (f) return f;
                    }
                    return legacy;
                };
                // publish "how many segments are stored on disk" — the lock
                // buttons in the segment cards key off this; re-render the
                // cards once when it changes (e.g. right after a run finishes)
                if (node._lastSeg !== last) {
                    node._lastSeg = last;
                    renderAll();
                }
                updateResLock();
                const grid = el("div", { class: "mmh3tte-pvgrid" });
                const openFile = (name) => window.open(
                    `/mmh3te/temporal_file?location=${location}&session=${encodeURIComponent(session)}&name=${name}`);
                const has = (f) => data.files.includes(f);
                // locked chain depends on the last locked segment's merged
                // latent (ledger mapping, legacy-name fallback — same
                // resolution the executor does). If that file no longer
                // exists on disk, unlock automatically.
                if (node._resumeFrom > 0) {
                    const li = node._resumeFrom - 1;
                    if (!has(fileFor(li, "merged_latent", `merged_${li}.h3latent`)))
                        autoUnlockStaleChain();
                }
                // one cell per segment: ONLY this segment's own result; the
                // stitched result goes into a final cell after the loop. The
                // rerun point is driven by the lock buttons on the segment
                // cards, so no per-preview rerun controls here — just a badge
                // marking which segment runs next.
                for (let i = 0; i <= last; i++) {
                    const segment = fileFor(i, "segment_preview", `segment_preview_${i}.webp`);
                    const cell = el("div", { class: "mmh3tte-pv" + (i < node._resumeFrom ? " locked-pv" : "") });
                    if (has(segment)) {
                        cell.appendChild(el("img", {
                            src: `/mmh3te/temporal_file?location=${location}&session=${encodeURIComponent(session)}&name=${segment}`,
                            title: uistr("segment alone") + ` ${i}`,
                            onclick: () => openFile(segment),
                        }));
                    }
                    cell.appendChild(el("div", { class: "cap" },
                        el("span", {}, `${uistr("seg")} ${i}`),
                        i === node._resumeFrom ? el("span", { class: "mmh3tte-badge", title: uistr("resume here") }, "⟳") : null));
                    grid.appendChild(cell);
                }
                // stitched (merged) result of all segments so far, last cell
                if (last >= 0) {
                    const merged = fileFor(last, "merged_preview", `merged_preview_${last}.webp`);
                    const cell = el("div", { class: "mmh3tte-pv" });
                    if (has(merged)) {
                        cell.appendChild(el("img", {
                            src: `/mmh3te/temporal_file?location=${location}&session=${encodeURIComponent(session)}&name=${merged}`,
                            title: uistr("merged through segment") + ` ${last}`,
                            onclick: () => openFile(merged),
                        }));
                    }
                    cell.appendChild(el("div", { class: "cap" },
                        el("span", {}, uistr("merged"))));
                    grid.appendChild(cell);
                }
                pvBox.appendChild(grid);
                pvBox.appendChild(row(
                    el("button", {
                        class: "mmh3tte-btn",
                        title: uistr("delete stored latent & preview files not referenced by the current chain"),
                        onclick: () => cleanupStaleFiles(location, session),
                    }, uistr("clear unused latents")),
                    el("span", { class: "mmh3tte-muted" },
                        uistr("deletes attempt files neither shown as segment/merged previews nor mapped by session.json"))));
                if (node._resumeFrom > 0) {
                    pvBox.appendChild(row(
                        el("span", { class: "mmh3tte-badge" }, `${uistr("resume point: segment")} ${node._resumeFrom}`),
                        el("button", {
                            class: "mmh3tte-btn",
                            title: uistr("unlock all"),
                            onclick: () => { node._resumeFrom = 0; sync(); renderAll(); renderPreviews(); },
                        }, uistr("unlock all"))));
                }
                renderCacheRow(location);
                restoreScroll();
            }

            // ── "clear unused latents" button handler ──
            // Two phases: first ask the server which stale attempt files (old
            // seq-named latents/previews NOT mapped by session.json) exist, then
            // require an explicit confirm before actually deleting them.
            async function cleanupStaleFiles(location, session) {
                const qp = `location=${encodeURIComponent(location)}&session=${encodeURIComponent(session)}`;
                let dry;
                try {
                    const r = await fetch(`/mmh3te/temporal_cleanup?${qp}`);
                    dry = await r.json();
                } catch { return; }
                const n = dry && Number.isInteger(dry.removed) ? dry.removed : 0;
                if (n <= 0) return;
                confirmDialog(
                    `${n} ${uistr("stored file(s) not referenced by the current chain will be permanently deleted")} ${uistr("this cannot be undone, continue?")}`,
                    uistr("confirm clear"),
                    async () => {
                        try {
                            await fetch(`/mmh3te/temporal_cleanup?${qp}&commit=1`);
                        } catch { /* refresh regardless */ }
                        renderPreviews();
                    });
            }

            // One line about the SHARED reference cache. It is deliberately
            // outside the per-session grid: the cache lives next to the
            // session directories and is read/written by every session of the
            // storage location, so it is worth reporting even when THIS
            // session has nothing stored yet. Its whole point is that a re-run
            // whose references did not change skips the VAE and text encoders.
            async function renderCacheRow(location) {
                let st;
                try {
                    const resp = await fetch(`/mmh3te/cache_status?location=${encodeURIComponent(location)}`);
                    st = await resp.json();
                } catch { return; }
                if (!pvBox || !st) return;
                const refs = (st.refs && st.refs.entries) || 0;
                const conds = (st.cond && st.cond.entries) || 0;
                if (!refs && !conds) return;    // nothing cached: stay quiet
                const mb = ((st.total_bytes || 0) / (1024 * 1024)).toFixed(1);
                pvBox.appendChild(row(
                    el("span", {
                        class: "mmh3tte-muted",
                        title: uistr("reference latents + encoded conditioning kept under mmh3_temporal/cache — shared by every session of this storage location"),
                    }, `${uistr("reference cache")}: ${refs} + ${conds} (${mb} MB)`),
                    el("button", {
                        class: "mmh3tte-btn",
                        title: uistr("drop every cached reference latent and conditioning (the next run re-encodes them)"),
                        onclick: async (ev) => {
                            ev.stopPropagation();
                            await fetch(`/mmh3te/cache_clear?location=${encodeURIComponent(location)}`);
                            renderPreviews();
                        },
                    }, uistr("clear cache"))));
            }

            // ── segments column ──
            let segList = null;
            function renderAll() {
                if (!segList) return;
                // the cards are rebuilt from scratch below; capture the column's
                // scroll position so lock / unlock / add / delete cannot snap it
                // back to the top (helper documented above renderPreviews)
                const restoreScroll = preserveScroll(segCol);
                node._durChips = {};   // segment -> {dur, cum} head chips (rebuilt now)
                // a wire that just appeared flips segments to 'load images'
                // before the cards are rebuilt, so the reference row shows up
                applySocketDefaults();
                // ...and a wire that is NOT there takes the splice sides away
                // (the video is the strip's asset), so the cards below render
                // the sides the run will really use
                applyMaterialSocketGate();
                segList.innerHTML = "";
                node._segments.forEach((s, i) => segList.appendChild(renderSeg(s, i)));
                updateResLock();
                // helper: clone the previous segment's parameters so a new
                // segment starts as a small edit, not from scratch
                const makeSegFrom = (prev) => {
                    const seg = {
                        ...structuredClone(prev),
                        // first-segment frames are 17n+5 (e.g. 124) — a
                        // continuation segment must sit on the 17n grid
                        new_frames: snap17(prev.new_frames),
                        fl2va_first_image: "",
                        // socket 'x' exclusions are per segment: the new
                        // segment starts with EVERY wired reference image
                        // (copying the previous segment's exclusions would
                        // silently hide an image the user never dropped)
                        ref_socket_off: [],
                    };
                    // continuation FL2VA = last-frame-only; cloned from a
                    // first segment without a last frame it is unusable:
                    // fall back to Ref2VA, reusing the previous segment's
                    // first-frame image when there is one
                    if (seg.mode === "FL2VA" && !seg.fl2va_last_image) {
                        seg.mode = "Ref2VA";
                        if (prev.fl2va_first_image)
                            seg.ref_images = [prev.fl2va_first_image];
                    }
                    // image reference mode mirrors what the new segment
                    // actually carries: picked images OR wired sockets
                    // load right away, with none it starts as 'no
                    // reference' ('previous segment frame' stays an
                    // explicit choice - it is not image-driven)
                    seg.ref_source = ((seg.ref_images || []).length
                        || wiredSocketNames().length)
                        ? "manual" : "none";
                    // segment 0's FL2VA first frame has no meaning for
                    // a continuation segment
                    seg.fl2va_first_image = "";
                    return seg;
                };
                // add-segment controls: how many + optional random seed
                const addCountInp = el("input", { type: "number", min: "1", max: "99",
                    value: String(node._addCount || 1), class: "mmh3tte-inp",
                    style: "width:56px" });
                const addRndChk = el("input", { type: "checkbox" });
                addRndChk.checked = !!node._addRandomSeed;
                const addBtn = el("button", {
                    class: "mmh3tte-btn primary",
                    onclick: () => {
                        const count = Math.min(99, Math.max(1,
                            parseInt(addCountInp.value, 10) || 1));
                        let added = 0;
                        for (let k = 0; k < count; k++) {
                            if (node._segments.length >= 99) break;
                            const seg = makeSegFrom(node._segments[node._segments.length - 1]);
                            // each added segment gets a fresh random seed when
                            // the toggle is on (otherwise it inherits the
                            // previous segment's seed via the clone)
                            if (addRndChk.checked)
                                seg.seed = Math.floor(Math.random() * 0xffffffff);
                            node._segments.push(seg);
                            added++;
                        }
                        if (added) {
                            node._addCount = count;
                            node._addRandomSeed = addRndChk.checked;
                            node._openSeg = node._segments.length - 1;
                            sync(); renderAll();
                        }
                    },
                }, uistr("+ add segment"));
                segList.appendChild(row(
                    el("span", { class: "mmh3tte-muted" }, uistr("add") + " ×"),
                    addCountInp,
                    el("label", { class: "mmh3tte-lbl",
                                  style: "display:inline-flex;align-items:center;gap:4px;cursor:pointer" },
                        addRndChk, " " + uistr("random seed per segment")),
                    addBtn));
                // the totals describe what the RUN produces, so a closed tail
                // is reported separately instead of being counted in: with a
                // stop point the line reads "8 segments · running 5 · total ≈
                // ..." (the panel never shows a length the run will not make)
                const runUpto = muteStopIndex(node._muteFrom, node._segments.length);
                const runSegs = runUpto >= 0 ? node._segments.slice(0, runUpto)
                                             : node._segments;
                const runFrames = runSegs.reduce((a, s) => a + s.new_frames, 0);
                segList.appendChild(row(
                    el("span", { class: "mmh3tte-muted" },
                        `${node._segments.length} ${uistr("segment(s)")}` +
                        (runUpto >= 0 ? ` · ${uistr("running")} ${runSegs.length}` : "") +
                        ` · ${uistr("total ≈")} ${runFrames}f · ${fmtSec(runFrames)}s`)));
                if (node._resumeFrom > 0) {
                    segList.appendChild(row(el("span", { class: "mmh3tte-badge" },
                        `🔒 ${uistr("will reuse segments")} 0..${node._resumeFrom - 1} ${uistr("from disk")} — ${uistr("resume from segment")} ${node._resumeFrom}`)));
                }
                // ...and the other chain-level point, the one the mute buttons
                // set. This button is the only one-click FULL reopen: the
                // per-segment mute buttons walk the boundary forward instead
                // and never clear it from behind.
                if (runUpto >= 0) {
                    segList.appendChild(row(
                        el("span", { class: "mmh3tte-badge closed" },
                            `⏹ ${uistr("mute: the chain stops before segment")} ${runUpto} — ` +
                            `${node._segments.length - runUpto} ${uistr("closed segment(s)")}`),
                        el("button", {
                            class: "mmh3tte-btn",
                            title: uistr("unmute: reopen the whole chain — every closed segment becomes active again"),
                            onclick: () => { node._muteFrom = -1; sync(); renderAll(); },
                        }, uistr("unmute"))));
                }
                restoreScroll();
            }

            function refreshAll() {
                renderAll();
                renderPreviews();
            }

            // ── restore editor state from the session ledger ──
            // The counterpart of scheduleEditorSave(): rebuild the whole chain
            // (segments, extend_params, resume point) from session.json's
            // 'editor' snapshot, so a node that was reset / a workflow that
            // was lost can pick the previous state back up from disk alone.
            async function restoreFromSession() {
                const session = findW("session_name")?.value || "session1";
                const location = findW("storage_location")?.value || "temp";
                const toast = app.extensionManager?.toast;
                const notify = (severity, detail) => {
                    if (toast?.add) {
                        toast.add({ severity, summary: "MMH3 Temporal Tile Editor",
                                    detail, life: 4000 });
                    } else {
                        console.warn(TAG, detail);
                    }
                };
                let data;
                try {
                    const resp = await fetch(`/mmh3te/temporal_session?location=${encodeURIComponent(location)}&session=${encodeURIComponent(session)}`);
                    data = await resp.json();
                } catch { data = {}; }
                const editor = (data.session && data.session.editor) || null;
                if (!editor || !Array.isArray(editor.segments)
                        || !editor.segments.length) {
                    notify("info",
                        uistr("no saved editor state in this session"));
                    return;
                }
                confirmDialog(uistr("restore replaces current state"),
                    uistr("confirm restore"), () => {
                        node._segments = editor.segments.map(normSeg);
                        if (!node._segments[0].width) node._segments[0].width = 768;
                        liftLegacyMaterial(editor);
                        if (editor.extend_params
                                && typeof editor.extend_params === "object") {
                            node._extendParams = { ...node._extendParams,
                                                   ...editor.extend_params };
                        }
                        node._resumeFrom =
                            (Number.isInteger(editor.resume_from_segment)
                             && editor.resume_from_segment > 0)
                                ? editor.resume_from_segment : 0;
                        node._muteFrom =
                            Number.isInteger(editor.mute_from_segment)
                                ? muteStopIndex(editor.mute_from_segment,
                                                node._segments.length) : -1;
                        node._openSeg = 0;
                        sync();
                        renderAll();
                        renderPreviews();
                        notify("success", uistr("editor state restored"));
                    });
            }

            // ══ Dock chrome (mirrors the spatial editor) ══
            const dock = el("div", { class: "mmh3tte-dock" });
            node._mmh3tteDock = dock;

            const head = el("div", { class: "mmh3tte-head" });
            const titleLabel = el("span", { class: "title" }, uistr("MMH3 Temporal Tile Editor"));
            head.appendChild(titleLabel);

            const minBtn = el("button", { class: "mmh3tte-btn", title: uistr("Minimize / restore"), style: "color:#e06050" }, "\u2014");
            head.appendChild(minBtn);

            // zoom controls
            const zoomGroup = el("div", { class: "mmh3tte-zoom-group" });
            const ZOOM_STEPS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0];
            function getZoom() { return node.properties.mmh3tteZoom || 1.0; }
            function setZoom(z) {
                z = Math.max(0.5, Math.min(2.0, z));
                node.properties.mmh3tteZoom = z;
                dock.style.transformOrigin = "0 0";
                dock.style.transform = `scale(${z})`;
                zoomLbl.textContent = Math.round(z * 100) + "%";
            }
            const zoomOut = el("button", { class: "mmh3tte-btn", title: uistr("Zoom out") }, "\u2212");
            const zoomLbl = el("span", { class: "mmh3tte-zoom-lbl", title: uistr("Click to reset zoom") });
            const zoomIn = el("button", { class: "mmh3tte-btn", title: uistr("Zoom in") }, "+");
            zoomGroup.append(zoomOut, zoomLbl, zoomIn);
            head.appendChild(zoomGroup);

            const closeBtn = el("button", { class: "mmh3tte-btn", title: uistr("Close editor") }, "\u2715");
            head.appendChild(closeBtn);
            dock.appendChild(head);

            // toolbar
            const bar = el("div", { class: "mmh3tte-bar" });
            const sessLbl = el("span", { class: "mmh3tte-lbl", style: "min-width:0" });
            node._sessLbl = sessLbl;
            const refreshBtn = el("button", { class: "mmh3tte-btn", onclick: () => renderPreviews() }, "⟳ " + uistr("refresh previews"));
            const restoreBtn = el("button", { class: "mmh3tte-btn", onclick: () => restoreFromSession() }, "↺ " + uistr("restore from session"));
            bar.append(
                el("span", { class: "mmh3tte-lbl", style: "min-width:0" }, uistr("Session:")),
                sessLbl,
                el("span", { style: "flex:1" }),
                restoreBtn,
                refreshBtn);
            dock.appendChild(bar);

            // body: segments column + previews panel
            const body = el("div", { class: "mmh3tte-body" });
            const segCol = el("div", { class: "mmh3tte-segcol" });
            const segHdr = el("div", { class: "mmh3tte-hdr" }, uistr("Segments"));
            segCol.appendChild(segHdr);
            segList = el("div");
            segCol.appendChild(segList);
            const pvPanel = el("div", { class: "mmh3tte-pvpanel" });
            pvPanel.appendChild(el("div", { class: "mmh3tte-hdr" }, uistr("Previews / Resume")));
            pvBox = el("div");
            pvPanel.appendChild(pvBox);
            body.append(segCol, pvPanel);
            dock.appendChild(body);

            // resize handles
            for (const d of ["n", "s", "e", "w", "se", "sw", "ne", "nw"]) {
                dock.appendChild(el("div", { class: `mmh3tte-rsz ${d}`, "data-dir": d }));
            }

            // ── dock behaviors ──
            minBtn.addEventListener("click", () => {
                node._dockVisible = !node._dockVisible;
                dock.classList.toggle("minimized", !node._dockVisible);
                if (node._dockVisible) dock.style.display = "";
                minBtn.textContent = node._dockVisible ? "\u2212" : "+";
            });

            closeBtn.addEventListener("click", () => {
                dock.style.display = "none";
                node._dockVisible = false;
                setWidgetVal("show_editor", false);
            });

            function setWidgetVal(name, val) {
                const w = findW(name);
                if (w) { w.value = val; w.callback?.(val); }
            }

            // drag head to move dock
            head.addEventListener("pointerdown", (e) => {
                if (e.target === minBtn || e.target === closeBtn || e.button !== 0) return;
                if (e.target.closest("button,select")) return;
                e.preventDefault();
                const sx0 = e.clientX, sy0 = e.clientY;
                const startLeft = dock.offsetLeft, startTop = dock.offsetTop;
                document.body.classList.add("mmh3tte-dragging");
                const onMove = (me) => {
                    const newLeft = Math.max(0, Math.min(window.innerWidth - 60, startLeft + me.clientX - sx0));
                    const newTop = Math.max(0, Math.min(window.innerHeight - 30, startTop + me.clientY - sy0));
                    dock.style.left = newLeft + "px";
                    dock.style.top = newTop + "px";
                };
                const onEnd = () => {
                    document.removeEventListener("pointermove", onMove);
                    document.removeEventListener("pointerup", onEnd);
                    document.body.classList.remove("mmh3tte-dragging");
                    node.properties.mmh3tteRect = {
                        x: dock.offsetLeft, y: dock.offsetTop,
                        w: dock.offsetWidth, h: dock.offsetHeight,
                    };
                };
                document.addEventListener("pointermove", onMove);
                document.addEventListener("pointerup", onEnd);
            });

            // resize handles
            dock.querySelectorAll(".mmh3tte-rsz").forEach(h => {
                h.addEventListener("pointerdown", (e) => {
                    e.preventDefault();
                    const dir = h.dataset.dir;
                    const sx0 = e.clientX, sy0 = e.clientY;
                    const startRect = { x: dock.offsetLeft, y: dock.offsetTop,
                                        w: dock.offsetWidth, h: dock.offsetHeight };
                    const onMove = (me) => {
                        const dx = me.clientX - sx0, dy = me.clientY - sy0;
                        let { x, y, w, h: hh } = startRect;
                        if (dir.includes("e")) w = Math.max(560, w + dx);
                        if (dir.includes("w")) { w = Math.max(560, w - dx); x = x + (startRect.w - w); }
                        if (dir.includes("s")) hh = Math.max(380, hh + dy);
                        if (dir.includes("n")) { hh = Math.max(380, hh - dy); y = y + (startRect.h - hh); }
                        dock.style.left = x + "px"; dock.style.top = y + "px";
                        dock.style.width = w + "px"; dock.style.height = hh + "px";
                    };
                    const onEnd = () => {
                        document.removeEventListener("pointermove", onMove);
                        document.removeEventListener("pointerup", onEnd);
                        node.properties.mmh3tteRect = {
                            x: dock.offsetLeft, y: dock.offsetTop,
                            w: dock.offsetWidth, h: dock.offsetHeight,
                        };
                    };
                    document.addEventListener("pointermove", onMove);
                    document.addEventListener("pointerup", onEnd);
                });
            });

            zoomIn.addEventListener("click", () => {
                const next = ZOOM_STEPS.find(s => s > getZoom() + 0.001);
                if (next) setZoom(next);
            });
            zoomOut.addEventListener("click", () => {
                const prev = [...ZOOM_STEPS].reverse().find(s => s < getZoom() - 0.001);
                if (prev) setZoom(prev);
            });
            zoomLbl.addEventListener("click", () => setZoom(1.0));

            // ── show_editor toggle: open/close dock ──
            function applyShowEditor() {
                const w = findW("show_editor");
                if (!w) return;
                if (w.value === true || w.value === "true" || w.value === 1) {
                    node._dockVisible = true;
                    dock.style.display = "";
                    dock.classList.remove("minimized");
                    minBtn.textContent = "\u2212";
                } else {
                    node._dockVisible = false;
                    dock.style.display = "none";
                }
            }
            const showEdW = findW("show_editor");
            if (showEdW) {
                const origCb = showEdW.callback;
                showEdW.callback = function (...args) {
                    const r = origCb ? origCb.apply(this, args) : undefined;
                    applyShowEditor();
                    return r;
                };
            }

            // ── placement / persistence ──
            function applyFloatRect() {
                const r = node.properties.mmh3tteRect;
                if (r) {
                    dock.style.left = r.x + "px";
                    dock.style.top = r.y + "px";
                    dock.style.width = r.w + "px";
                    dock.style.height = r.h + "px";
                }
                setZoom(getZoom());
            }

            const origConfigure = node.onConfigure;
            node.onConfigure = function (...args) {
                const r = origConfigure?.apply(this, args);
                initialFromWidget();
                refreshAll();
                setTimeout(applyShowEditor, 0);
                applyFloatRect();
                return r;
            };

            const origOnRemoved = node.onRemoved;
            const editorHandle = { renderPreviews: () => renderPreviews() };
            _liveEditors.add(editorHandle);
            node.onRemoved = function (...args) {
                _liveEditors.delete(editorHandle);
                dock.remove();
                node._mmh3tteDock = null;
                return origOnRemoved?.apply(this, args);
            };

            // ── init ──
            dock.style.width = "760px";
            dock.style.height = "560px";
            document.body.appendChild(dock);

            initialFromWidget();
            refreshAll();

            // refresh previews when storage/session widgets change; flush any
            // pending editor-state save to the OLD session first so edits
            // never leak into the newly selected one
            for (const wn of ["session_name", "storage_location"]) {
                const w = findW(wn);
                if (w) {
                    const origCb = w.callback;
                    w.callback = (v) => { flushEditorSave(); origCb?.(v); renderPreviews(); };
                }
            }

            applyShowEditor();
            setZoom(getZoom());
            requestAnimationFrame(() => {
                if (!node.properties.mmh3tteRect) {
                    node.properties.mmh3tteRect = {
                        x: Math.round((window.innerWidth - 760) / 2),
                        y: Math.round((window.innerHeight - 560) / 2),
                        w: 760, h: 560,
                    };
                }
                applyFloatRect();
            });
        };
    },
});
