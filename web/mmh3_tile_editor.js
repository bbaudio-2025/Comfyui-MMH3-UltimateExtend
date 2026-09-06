// Dock-panel visual editor for the "MMH3 Spatial Tile Editor" node.
//
// The server node exposes a hidden JSON string widget (`tile_data`) that stores
// per-tile configuration. This extension replaces the raw string editing with an
// interactive dock panel: a canvas showing the tile layout, a property panel for
// per-tile prompt/reference image editing, and a toolbar for global settings.
import { app } from "../../scripts/app.js";

const TARGET = "MMH3SpatialTileEditor";
const DATA_WIDGET = "tile_data";
const TAG = "[MMH3SpatialTileEditor]";

// ── i18n: dock-panel UI ──
// nodeDefs.json (locales/zh) translates the node-definition text; this block
// handles the dock-panel strings the locale system does not reach. The active
// ComfyUI language is persisted in localStorage; we detect any zh locale and
// fall back to the browser language. Keys are the exact English literals.
const ZH_UI = {
    // Dock head / toolbar
    "MMH3 Spatial Tile Editor": "MMH3 空间分块编辑器",
    "Minimize / restore": "最小化 / 还原",
    "Zoom out": "缩小",
    "Zoom in": "放大",
    "Click to reset zoom": "点击重置缩放",
    "Reset zoom/pan": "重置缩放/平移",
    "Plan:": "布局方案:",
    "Horizontal Alternating": "水平交替",
    "Vertical Alternating": "垂直交替",
    "4 Quadrants Expand": "四象限扩展",
    "Tiles:": "分块数:",
    "4 Quadrants always uses 5 tiles": "四象限始终使用 5 个分块",
    "Compose": "合成",
    "Close editor": "关闭编辑器",
    "Close preview": "关闭预览",
    "Drag to resize": "拖动调整大小",
    "Zoom": "缩放",
    // Canvas labels / roles
    "Tile": "分块",
    "Center": "中心",
    "First frame": "首帧",
    "Last frame": "尾帧",
    // Empty / hint
    "Click a tile on the canvas to edit it": "点击画布上的分块以编辑它",
    // Conditioning mode
    "Mode:": "条件模式:",
    "FL2VA (first/last frames)": "FL2VA（首/尾帧）",
    "Ref2VA (reference images)": "Ref2VA（参考图像）",
    // Prompts
    "Positive Prompt (this tile)": "正向提示词（本分块）",
    "Appended to base prompt": "追加到基础正向提示词",
    "Negative Prompt (this tile)": "负向提示词（本分块）",
    "Appended to base negative": "追加到基础负向提示词",
    // Dimensions
    "Dimensions": "尺寸",
    "All tiles share the same height": "所有分块共享相同高度",
    "All tiles share the same width": "所有分块共享相同宽度",
    "Width:": "宽度:",
    "Height:": "高度:",
    // Overlap / fade
    "Overlap / Fade": "重叠 / 渐变",
    "Overlap W:": "水平重叠:",
    "Overlap H:": "垂直重叠:",
    "Fade W:": "水平渐变:",
    "Fade H:": "垂直渐变:",
    "Not applicable for this plan/tile": "该布局/分块不适用",
    // References
    "First/Last Images": "首帧/尾帧图像",
    "Reference Images": "参考图像",
    "Source:": "来源:",
    "Tile crop": "分块裁剪区",
    "Own images": "自有图像",
    "Using this tile's compose split as its first/last frames. Switch Source to 'Own images' to load separate pictures.": "使用本分块的合成裁剪区作为其首/尾帧。将来源切换为“自有图像”以加载独立图片。",
    "Using this tile's compose split as its reference block(s). Switch Source to 'Own images' to load separate pictures.": "使用本分块的合成裁剪区作为其参考块。将来源切换为“自有图像”以加载独立图片。",
    "(no image)": "（无图像）",
    "File not found in input folder": "输入文件夹中未找到该文件",
    "+ Add Image": "+ 添加图像",
    "Remove image": "移除图像",
    // Compose panel
    "Active: tiles whose Source is 'Tile crop' use their region of the images below.": "已启用：来源为“分块裁剪区”的分块使用下方图像中各自的区域。",
    "Inactive: load a First or Last image below to enable compose.": "未启用：载入下方的首帧或尾帧图像以启用合成。",
    "Preview:": "预览:",
    "(none)": "（无）",
    "First": "首帧",
    "Last": "尾帧",
    "First:": "首帧:",
    "Last:": "尾帧:",
    "Offset:": "偏移:",
    "plan px": "方案像素",
    // Long explanatory tooltips
    "Toggle the right panel between tile parameters and the compose setup. Compose is active whenever a First or Last image is loaded; with both empty it is off.": "在右侧面板的分块参数与合成设置之间切换。只要已加载首帧或尾帧图像，合成即处于启用状态；两者均为空时关闭。",
    "Conditioning mode for this tile: FL2VA = its reference image(s) become the first (and optional last) frame; Ref2VA = they become subject reference blocks. Tiles can mix freely within one plan.": "本分块的条件模式：FL2VA = 其参考图像成为首帧（以及可选的尾帧）；Ref2VA = 它们成为主体参考块。同一布局方案内各分块可自由混合。",
    "Tile crop: use this tile's region of the compose First/Last image (as FL2VA frames or Ref2VA blocks, per the Mode above). Own images: load separate pictures below.": "分块裁剪区：使用该分块在合成首/尾帧图像中的区域（依据上方模式作为 FL2VA 帧或 Ref2VA 块）。自有图像：在下方加载独立图片。",
    "Which compose source (First/Last) is drawn on the canvas and re-framed by wheel/drag while the Compose panel is open. 'None' hides both images.": "合成面板打开时，画布上绘制并可由滚轮/拖动重新取景的合成源（首帧/尾帧）。选择“无”则同时隐藏两幅图像。",
    "Position of the tile-plan region relative to the image's top-left corner, in tile-plan pixels (the image is scaled so the tile plan fits it -- same units as tile width/height). 0 puts the plan at the image's left/top edge; First and Last are independent. Dragging or wheel-zooming the preview updates these values. Values typed before the image finishes decoding are applied automatically once it loads.": "分块方案区域相对图像左上角的位置，单位为方案像素（图像被缩放以适配方案 -- 与分块宽/高单位一致）。0 将方案置于图像的左/上边缘；首帧与尾帧相互独立。拖动或滚轮缩放预览会更新这些值。在图像解码完成前输入的值会在其加载后自动应用。",
};
// Return the localized string; English (or any non-zh) locale keeps the literal.
function uistr(en) {
    if (ZH_UI_ENABLED) return ZH_UI[en] || en;
    return en;
}
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

const TILE_COLORS = [
    "#3a6ea5", "#3f9142", "#b3872f", "#8b5cf6", "#e06050",
    "#46b4e6", "#e6a046", "#50b070", "#c06090", "#7080c0",
    "#a0a040", "#60c0c0",
];

// Role labels for alternating layouts
const ROLE_ALTERNATING = { 0: uistr("Center") };
for (let i = 1; i < 12; i++) {
    const k = Math.floor((i + 1) / 2);
    const side = i % 2 === 1 ? "R" : "L";
    ROLE_ALTERNATING[i] = k > 1 ? `${side}${k}` : side;
}
const ROLE_QUADRANT = { 0: uistr("Center"), 1: "TL", 2: "TR", 3: "BL", 4: "BR" };

// ── Plan helpers ──
// Plan names: "horizontal_alternating", "vertical_alternating", "4_quadrants_expand"
// These map to the Python node's (scheme, axis) pair.
const PLANS = ["horizontal_alternating", "vertical_alternating", "4_quadrants_expand"];
const PLAN_LABELS = {
    horizontal_alternating: uistr("Horizontal Alternating"),
    vertical_alternating: uistr("Vertical Alternating"),
    "4_quadrants_expand": uistr("4 Quadrants Expand"),
};

function planToSchemeAxis(plan) {
    if (plan === "4_quadrants_expand") return { scheme: "4_quadrants_expand", axis: "horizontal" };
    if (plan === "vertical_alternating") return { scheme: "alternating", axis: "vertical" };
    return { scheme: "alternating", axis: "horizontal" };
}

function tileRole(idx, plan) {
    if (plan === "4_quadrants_expand") return ROLE_QUADRANT[idx] || String(idx);
    return ROLE_ALTERNATING[idx] || String(idx);
}

// ── Layout geometry (pixel space) ──
function alternatingOffsets(widths, olWs) {
    const n = widths.length;
    const offs = new Array(n).fill(0);
    let lo = 0, hi = widths[0];
    for (let i = 1; i < n; i++) {
        if (i % 2 === 1) { offs[i] = hi - olWs[i]; hi = offs[i] + widths[i]; }
        else { offs[i] = lo + olWs[i] - widths[i]; lo = offs[i]; }
    }
    return offs;
}

function computePlacements(tiles, plan) {
    const { scheme, axis } = planToSchemeAxis(plan);
    if (scheme === "4_quadrants_expand" && tiles.length === 5) {
        return computeQuadrantPlacements(tiles);
    }
    const placements = [];
    if (axis === "horizontal") {
        const ws = tiles.map(t => t.width);
        const ows = tiles.map(t => t.overlap_w);
        const offs = alternatingOffsets(ws, ows);
        const lo = Math.min(...offs);
        for (let i = 0; i < tiles.length; i++) {
            placements.push({ x: offs[i] - lo, y: 0, w: tiles[i].width, h: tiles[i].height });
        }
    } else {
        const hs = tiles.map(t => t.height);
        const ohs = tiles.map(t => t.overlap_h);
        const offs = alternatingOffsets(hs, ohs);
        const lo = Math.min(...offs);
        for (let i = 0; i < tiles.length; i++) {
            placements.push({ x: 0, y: offs[i] - lo, w: tiles[i].width, h: tiles[i].height });
        }
    }
    return placements;
}

function computeQuadrantPlacements(tiles) {
    const [t0, t1, t2, t3, t4] = tiles;
    const origins = [
        { r: 0, c: 0 },
        { r: t1.overlap_h - t1.height, c: t1.overlap_w - t1.width },
        { r: t2.overlap_h - t2.height, c: t0.width - t2.overlap_w },
        { r: t0.height - t3.overlap_h, c: t3.overlap_w - t3.width },
        { r: t0.height - t4.overlap_h, c: t0.width - t4.overlap_w },
    ];
    const top = origins[1].r, left = origins[1].c;
    return origins.map((o, i) => ({
        x: o.c - left, y: o.r - top, w: tiles[i].width, h: tiles[i].height,
    }));
}

function totalSize(tiles, plan) {
    const pl = computePlacements(tiles, plan);
    if (!pl.length) return { w: 512, h: 512 };
    const maxX = Math.max(...pl.map(p => p.x + p.w));
    const maxY = Math.max(...pl.map(p => p.y + p.h));
    return { w: maxX, h: maxY };
}

// ── Image fetching from input folder ──
let _inputImages = null;
async function fetchInputImages() {
    if (_inputImages) return _inputImages;
    try {
        const resp = await fetch("/mmh3te/input_files");
        const files = await resp.json();
        _inputImages = Array.isArray(files) ? files : [];
    } catch { _inputImages = []; }
    return _inputImages;
}
function invalidateInputImages() { _inputImages = null; }

function imageUrl(filename) {
    if (!filename) return "";
    return `/view?filename=${encodeURIComponent(filename)}&type=input&preview=webp;80`;
}

// ── Serialization helpers ──
function parseTileData(widget) {
    try {
        const v = JSON.parse(widget.value || "{}");
        return (v && typeof v === "object" && !Array.isArray(v)) ? v : {};
    } catch { return {}; }
}

// Default tile dimensions (must match Python DEFAULT_TILE_* constants)
const DEF_W = 512, DEF_H = 768, DEF_OW = 128, DEF_OH = 128, DEF_FW = 32, DEF_FH = 32;

function defaultTile(idx) {
    return {
        prompt: "", negative: "", ref_images: [],
        width: DEF_W, height: DEF_H,
        overlap_w: DEF_OW, overlap_h: DEF_OH,
        fade_w: DEF_FW, fade_h: DEF_FH,
    };
}

// ── CSS injection ──
function injectStyle() {
    if (document.getElementById("mmh3te-style")) return;
    const s = document.createElement("style");
    s.id = "mmh3te-style";
    s.textContent = `
        .mmh3te-dock { position:fixed; z-index:8500; display:flex; flex-direction:column;
            background:var(--kj-dock-bg,#1a1a1a); border:1px solid var(--kj-dock-border,#555);
            border-radius:8px; box-shadow:0 8px 30px rgba(0,0,0,0.55);
            min-width:700px; min-height:500px; overflow:hidden; pointer-events:auto; }
        .mmh3te-dock.minimized { min-height:0 !important; height:auto !important; }
        .mmh3te-dock.minimized .mmh3te-body { display:none; }
        .mmh3te-dock.minimized .mmh3te-rsz { display:none; }
        .mmh3te-dock.pinned .mmh3te-rsz { display:none; }
        .mmh3te-dock.pinned .mmh3te-head { cursor:default; }
        .mmh3te-head { display:flex; align-items:center; gap:6px; padding:4px 8px;
            background:var(--kj-dock-head,#262626); cursor:move; font:12px sans-serif;
            color:#ccc; user-select:none; border-bottom:1px solid rgba(0,0,0,0.25); flex:0 0 auto; }
        .mmh3te-head .mmh3te-btn { padding:1px 7px; }
        .mmh3te-zoom-group { display:flex; align-items:center; gap:2px; margin-left:auto; }
        .mmh3te-zoom-lbl { font:11px monospace; color:#aaa; min-width:36px; text-align:center; cursor:pointer; }
        .mmh3te-zoom-lbl:hover { color:#fff; }
        .mmh3te-btn.on { background:#46b4e6; color:#111; border:1px solid #46b4e6; }
        .mmh3te-body { display:flex; flex:1 1 auto; min-height:0; padding:8px; gap:8px; }
        .mmh3te-cv-wrap { flex:1 1 auto; display:flex; align-items:center; justify-content:center;
            overflow:hidden; background:#111; border-radius:4px; min-width:300px; }
        .mmh3te-canvas { cursor:pointer; display:block; background:#1a1a1a; border-radius:4px; touch-action:none; }
        .mmh3te-cv-overlay { position:absolute; top:0; left:0; right:0; bottom:0; z-index:10;
            display:none; align-items:center; justify-content:center;
            background:rgba(0,0,0,0.75); border-radius:4px; pointer-events:auto; }
        .mmh3te-cv-overlay.visible { display:flex; }
        .mmh3te-cv-overlay img { max-width:90%; max-height:90%; object-fit:contain; border-radius:4px;
            box-shadow:0 4px 20px rgba(0,0,0,0.6); }
        .mmh3te-cv-close { position:absolute; top:8px; right:8px; z-index:11;
            background:rgba(0,0,0,0.6); border:1px solid #555; border-radius:50%;
            color:#ccc; font:16px sans-serif; width:28px; height:28px; cursor:pointer;
            display:flex; align-items:center; justify-content:center; line-height:1; }
        .mmh3te-cv-close:hover { background:#e06050; border-color:#e06050; color:#fff; }
        .mmh3te-panel { width:280px; min-width:180px; max-width:500px; flex:0 0 auto; display:flex; flex-direction:column;
            gap:6px; overflow-y:auto; background:#262626; border-radius:4px;
            font:11px sans-serif; color:#bbb; padding:8px; }
        .mmh3te-divider { width:4px; flex:0 0 auto; cursor:ew-resize; background:transparent;
            border-left:1px solid #333; border-right:1px solid #333; transition:background .15s; }
        .mmh3te-divider:hover, .mmh3te-divider.dragging { background:#46b4e6; }
        .mmh3te-bar { display:flex; align-items:center; gap:6px; padding:4px 8px;
            background:#222; border-bottom:1px solid #333; font:11px sans-serif; color:#aaa;
            flex-wrap:wrap; flex:0 0 auto; }
        .mmh3te-compose-pane { display:flex; flex-direction:column; gap:4px; flex:1 1 auto;
            min-width:100%; padding-top:2px; }
        .mmh3te-compose-pane .mmh3te-ref-picker { min-width:0; }
        .mmh3te-compose-pane .mmh3te-lbl { min-width:36px; }
        .mmh3te-compose-pane .mmh3te-row { width:100%; }
        .mmh3te-btn { background:#333; border:1px solid #555; border-radius:4px;
            color:#bbb; font:11px sans-serif; cursor:pointer; padding:2px 8px;
            line-height:16px; white-space:nowrap; flex-shrink:0; }
        .mmh3te-btn:hover { border-color:#46b4e6; color:#fff; }
        .mmh3te-btn.active { border-color:#46b4e6; color:#46b4e6; background:#2a3a42; }
        .mmh3te-area { width:100%; box-sizing:border-box; background:#1d1d1d; border:1px solid #444;
            border-radius:4px; color:#ddd; font:12px monospace; padding:4px 6px; resize:none;
            min-height:40px; flex:1 1 auto; }
        .mmh3te-area:focus { border-color:#46b4e6; outline:none; }
        .mmh3te-lbl { color:#888; font:11px sans-serif; flex:0 0 auto; min-width:60px; }
        .mmh3te-row { display:flex; align-items:center; gap:6px; }
        .mmh3te-inp { background:#1d1d1d; border:1px solid #444; border-radius:4px;
            color:#ddd; font:11px sans-serif; padding:2px 6px; width:60px; text-align:right; }
        .mmh3te-inp:focus { border-color:#46b4e6; outline:none; }
        .mmh3te-sel { background:#1d1d1d; border:1px solid #444; border-radius:4px;
            color:#ddd; font:11px sans-serif; padding:2px 4px; flex:1 1 auto; min-width:0; }
        .mmh3te-sel:focus { border-color:#46b4e6; outline:none; }
        .mmh3te-thumb { width:48px; height:48px; object-fit:cover; border:1px solid #555;
            border-radius:4px; cursor:pointer; transition:transform .2s ease, box-shadow .15s ease;
            flex-shrink:0; }
        .mmh3te-thumb:hover { transform:scale(3); box-shadow:0 4px 20px rgba(0,0,0,0.7);
            z-index:100; position:relative; }
        .mmh3te-ref-row { display:flex; align-items:center; gap:6px; padding:3px 0; }
        .mmh3te-ref-row .mmh3te-sel { flex:1; }
        .mmh3te-xbtn { background:none; border:none; color:#999; cursor:pointer;
            font:14px sans-serif; padding:0 4px; flex:0 0 auto; }
        .mmh3te-xbtn:hover { color:#e06050; }
        .mmh3te-addbtn { background:none; border:1px dashed #555; border-radius:4px;
            color:#888; cursor:pointer; font:11px sans-serif; padding:3px 8px;
            width:100%; text-align:center; flex:0 0 auto; }
        .mmh3te-addbtn:hover { border-color:#46b4e6; color:#ccc; }
        .mmh3te-dragbar { flex:0 0 auto; height:9px; cursor:ns-resize; touch-action:none;
            user-select:none; display:flex; align-items:center; justify-content:center;
            background:transparent; }
        .mmh3te-dragbar::before { content:""; width:26px; height:3px; border-radius:2px;
            background:#555; }
        .mmh3te-dragbar:hover::before, .mmh3te-dragbar.dragging::before { background:#46b4e6; }
        body.mmh3te-vdrag, body.mmh3te-vdrag * { cursor:ns-resize !important; user-select:none; }
        .mmh3te-sep { border:none; border-top:1px solid #333; margin:4px 0; }
        .mmh3te-hdr { font:bold 11px sans-serif; color:#aaa; padding:2px 0; user-select:none; }
        .mmh3te-hdr.collapsible { cursor:pointer; display:flex; align-items:center; gap:4px; }
        .mmh3te-hdr.collapsible:hover { color:#fff; }
        .mmh3te-hdr.collapsible::before { content:"\u25B6"; font:8px sans-serif; transition:transform .15s; }
        .mmh3te-hdr.collapsible.open::before { transform:rotate(90deg); }
        .mmh3te-drawer { overflow:hidden; transition:max-height .2s ease; }
        .mmh3te-drawer.closed { max-height:0 !important; }
        .mmh3te-drawer.resizable { display:flex; flex-direction:column; }
        .mmh3te-drawer.resizable .mmh3te-area { flex:1 1 auto; min-height:0; resize:none; }
        .mmh3te-ref-picker { position:relative; }
        .mmh3te-ref-trigger { display:flex; align-items:center; gap:6px; background:#1d1d1d;
            border:1px solid #444; border-radius:4px; padding:4px 6px; cursor:pointer;
            min-height:40px; }
        .mmh3te-ref-trigger:hover { border-color:#46b4e6; }
        .mmh3te-ref-trigger.missing { border-color:#e06050; }
        .mmh3te-ref-trigger.missing .name { color:#e06050; }
        .mmh3te-ref-trigger img { width:32px; height:32px; object-fit:cover; border-radius:3px; }
        .mmh3te-ref-trigger .name { font:11px sans-serif; color:#ddd; flex:1; overflow:hidden;
            text-overflow:ellipsis; white-space:nowrap; }
        .mmh3te-ref-trigger .arrow { color:#888; font:10px sans-serif; }
        .mmh3te-ref-dropdown { position:fixed; z-index:10000;
            background:#1d1d1d; border:1px solid #444; border-radius:4px;
            display:none; max-height:300px; overflow-y:auto; min-width:240px; }
        .mmh3te-ref-dropdown.open { display:block; }
        .mmh3te-ref-opt { display:flex; align-items:center; gap:6px; padding:4px 6px;
            cursor:pointer; border-bottom:1px solid #333; }
        .mmh3te-ref-opt:hover { background:#2a3a42; }
        .mmh3te-ref-opt.selected { background:#2a3a42; }
        .mmh3te-ref-opt img { width:32px; height:32px; object-fit:cover; border-radius:3px; }
        .mmh3te-ref-opt .name { font:11px sans-serif; color:#ddd; flex:1; overflow:hidden;
            text-overflow:ellipsis; white-space:nowrap; }
        .mmh3te-ref-refresh { background:none; border:1px solid #555; border-radius:3px;
            color:#888; cursor:pointer; font:10px sans-serif; padding:2px 6px; flex-shrink:0; }
        .mmh3te-ref-refresh:hover { border-color:#46b4e6; color:#fff; }
        .mmh3te-rsz { position:absolute; z-index:20; touch-action:none; }
        .mmh3te-rsz.n { top:0; left:11px; right:11px; height:6px; cursor:ns-resize; }
        .mmh3te-rsz.s { bottom:0; left:11px; right:11px; height:6px; cursor:ns-resize; }
        .mmh3te-rsz.e { right:0; top:11px; bottom:11px; width:6px; cursor:ew-resize; }
        .mmh3te-rsz.w { left:0; top:11px; bottom:11px; width:6px; cursor:ew-resize; }
        .mmh3te-rsz.se { bottom:0; right:0; width:12px; height:12px; cursor:nwse-resize; }
        .mmh3te-rsz.sw { bottom:0; left:0; width:12px; height:12px; cursor:nesw-resize; }
        .mmh3te-rsz.ne { top:0; right:0; width:12px; height:12px; cursor:nesw-resize; }
        .mmh3te-rsz.nw { top:0; left:0; width:12px; height:12px; cursor:nwse-resize; }
        body.mmh3te-dragging, body.mmh3te-dragging * { cursor:move !important; }
    `;
    document.head.appendChild(s);
}

// ── Dock tracking (follows node position) ──
const pinnedDocks = new Set();
let _dockRAF = 0, _dockIdle = 0;
const DOCK_IDLE_STOP = 6;

function applyDockTransform(node) {
    const fl = node._mmh3teDock;
    if (!fl) return;
    const c = app.canvas;
    if (!c || !c.graph) { fl.style.display = "none"; return; }
    fl.style.display = "";
    const pinned = node.properties.mmh3tePinned !== false;
    if (pinned) {
        const vue = !!window.LiteGraph?.vueNodesMode;
        if (vue) {
            const hostEl = document.querySelector(`[data-node-id="${node.id}"]`);
            if (hostEl && fl.parentElement !== hostEl) {
                hostEl.appendChild(fl);
                fl.style.position = "absolute";
                fl.style.transform = "";
            }
            const gr = node.properties.mmh3teDockPos || { x: 0, y: (node.size ? node.size[1] : 0) + 2 };
            const title = window.LiteGraph?.NODE_TITLE_HEIGHT ?? 30;
            fl.style.left = gr.x + "px";
            fl.style.top = (title + gr.y) + "px";
        } else {
            if (fl.parentElement !== document.body) {
                document.body.appendChild(fl);
                fl.style.position = "fixed";
                fl.style.transformOrigin = "top left";
            }
            const rect = c.canvas.getBoundingClientRect();
            const gr = node.properties.mmh3teDockPos || { x: 0, y: (node.size ? node.size[1] : 0) + 2 };
            const scale = c.ds.scale || 1;
            const baseLeft = rect.left + (node.pos[0] + gr.x + c.ds.offset[0]) * scale;
            const baseTop = rect.top + (node.pos[1] + gr.y + c.ds.offset[1]) * scale;
            const zoom = node.properties.mmh3teZoom || 1.0;
            fl.style.transformOrigin = "0 0";
            fl.style.transform = `translate(${baseLeft}px,${baseTop}px) scale(${zoom})`;
        }
    }
}

function tickDocks() {
    for (const n of pinnedDocks) applyDockTransform(n);
    if (pinnedDocks.size && ++_dockIdle < DOCK_IDLE_STOP) {
        _dockRAF = requestAnimationFrame(tickDocks);
    } else { _dockRAF = 0; }
}
function wakeDocks() {
    _dockIdle = 0;
    if (!_dockRAF && pinnedDocks.size) _dockRAF = requestAnimationFrame(tickDocks);
}

// ── Extension registration ──
app.registerExtension({
    name: "MMH3UltimateExtend.TileEditor",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== TARGET) return;
        injectStyle();

        // Fetch input images on load
        await fetchInputImages();

        // Listen for new uploads to refresh the image list
        try {
            app.api?.addEventListener?.("executed", () => invalidateInputImages());
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
            node._tiles = {};
            node._selectedIdx = -1;
            // Right-panel view: false = tile parameters, true = compose setup
            // (toggled by the toolbar Compose button; NOT compose on/off).
            node._composeView = false;
            node._dockVisible = true;

            // Canvas zoom/pan state
            node._cvZoom = 1;
            node._cvPanX = 0;
            node._cvPanY = 0;

            // Reference-image dropdowns live on document.body rather than inside
            // the dock. The dock has `overflow:hidden` AND a CSS `transform`
            // (set by setZoom), which turns it into the containing block for any
            // `position:fixed` descendant and clips it away — so a dropdown that
            // is a child of the panel would never be visible. Tracking them at
            // this scope keeps them removable across re-renders.
            let _refDropdownEls = [];
            let _composeDropdownEls = [];
            let _refArea = null;
            const _closeRefDropdown = (e) => {
                for (const d of _refDropdownEls) {
                    if (!d.classList.contains("open")) continue;
                    if (d.contains(e.target) || (_refArea && _refArea.contains(e.target))) continue;
                    d.classList.remove("open");
                }
                for (const d of _composeDropdownEls) {
                    if (!d.classList.contains("open")) continue;
                    if (d.contains(e.target)) continue;
                    d.classList.remove("open");
                }
            };
            document.addEventListener("pointerdown", _closeRefDropdown);

            function getWidgetVal(name, def) {
                const w = findW(name);
                return w != null ? w.value : def;
            }
            function setWidgetVal(name, val) {
                const w = findW(name);
                if (w) { w.value = val; w.callback?.(val); }
            }

            // Plan and tile count are JS-managed state persisted in tile_data
            // (top-level "plan"/"tile_count" keys); the python node exposes no
            // widgets for them anymore.
            function tileCount() {
                if (currentPlan() === "4_quadrants_expand") return 5;
                return Math.max(1, Math.min(12, node._tileCount || 2));
            }

            function currentPlan() {
                return node._plan || "horizontal_alternating";
            }

            function basePrompt() { return getWidgetVal("base_prompt", ""); }

            function ensureTiles() {
                const n = tileCount();
                for (let i = 0; i < n; i++) {
                    if (!node._tiles[i]) {
                        node._tiles[i] = defaultTile(i);
                    }
                    const t = node._tiles[i];
                    // Normalize per-tile mode/source (also migrates tiles saved
                    // before per-tile conditioning modes existed).
                    if (t.cond_mode !== "FL2VA" && t.cond_mode !== "Ref2VA") t.cond_mode = "FL2VA";
                    if (t.ref_source !== "crop" && t.ref_source !== "own") {
                        // Tiles that already carry manual refs keep using them;
                        // empty tiles default to their compose split crop.
                        t.ref_source = (t.ref_images && t.ref_images.length) ? "own" : "crop";
                    }
                }
                // Prune stale keys >= n so node._tiles never keeps leftovers from
                // a previous higher tile_count. Otherwise serialization
                // (composeCropsFor) can compute placements over a larger tile
                // set than the canvas preview, drifting crop boxes right.
                for (const k of Object.keys(node._tiles)) {
                    if (Number(k) >= n) delete node._tiles[k];
                }
            }

            // Array view of the tile map (totalSize/computePlacements need
            // a real array; node._tiles is an index-keyed object, and calling
            // totalSize(node._tiles, ...) threw a TypeError that silently
            // killed the compose viewport init / wheel / drag / offset edit).
            function tilesArr() {
                const a = [];
                for (let i = 0; i < tileCount(); i++) a.push(node._tiles[i]);
                return a;
            }

            // ── Dimension sync helpers ──
            // When changing a tile's shared dimension, update all other tiles
            function syncSharedDimension(changedIdx, prop, val) {
                const p = currentPlan();
                const n = tileCount();
                if (p === "horizontal_alternating" && prop === "height") {
                    // All tiles must share the same height
                    for (let i = 0; i < n; i++) {
                        if (i !== changedIdx) node._tiles[i].height = val;
                    }
                } else if (p === "vertical_alternating" && prop === "width") {
                    // All tiles must share the same width
                    for (let i = 0; i < n; i++) {
                        if (i !== changedIdx) node._tiles[i].width = val;
                    }
                }
                // 4_quadrants_expand: no automatic sync (each tile has independent size)
            }

            // ── DOM: Dock panel ──
            const dock = document.createElement("div");
            dock.className = "mmh3te-dock";
            node._mmh3teDock = dock;

            // Head
            const head = document.createElement("div");
            head.className = "mmh3te-head";
            const titleLabel = document.createElement("span");
            titleLabel.style.cssText = "flex:1; font:bold 12px sans-serif;";
            titleLabel.textContent = uistr("MMH3 Spatial Tile Editor");
            head.appendChild(titleLabel);

            const minBtn = document.createElement("button");
            minBtn.className = "mmh3te-btn";
            minBtn.textContent = "\u2014";
            minBtn.title = uistr("Minimize / restore");
            minBtn.style.color = "#e06050";
            head.appendChild(minBtn);

            // Zoom controls
            const zoomGroup = document.createElement("div");
            zoomGroup.className = "mmh3te-zoom-group";
            const ZOOM_STEPS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0];
            function getZoom() { return node.properties.mmh3teZoom || 1.0; }
            function setZoom(z) {
                z = Math.max(0.5, Math.min(2.0, z));
                node.properties.mmh3teZoom = z;
                dock.style.transformOrigin = "0 0";
                dock.style.transform = `scale(${z})`;
                zoomLbl.textContent = Math.round(z * 100) + "%";
            }
            const zoomOut = document.createElement("button");
            zoomOut.className = "mmh3te-btn";
            zoomOut.textContent = "\u2212";
            zoomOut.title = uistr("Zoom out");
            zoomGroup.appendChild(zoomOut);
            const zoomLbl = document.createElement("span");
            zoomLbl.className = "mmh3te-zoom-lbl";
            zoomLbl.title = uistr("Click to reset zoom");
            zoomGroup.appendChild(zoomLbl);
            const zoomIn = document.createElement("button");
            zoomIn.className = "mmh3te-btn";
            zoomIn.textContent = "+";
            zoomIn.title = uistr("Zoom in");
            zoomGroup.appendChild(zoomIn);
            head.appendChild(zoomGroup);

            zoomIn.addEventListener("click", () => {
                const cur = getZoom();
                const next = ZOOM_STEPS.find(s => s > cur + 0.001);
                if (next) setZoom(next);
            });
            zoomOut.addEventListener("click", () => {
                const cur = getZoom();
                const prev = [...ZOOM_STEPS].reverse().find(s => s < cur - 0.001);
                if (prev) setZoom(prev);
            });
            zoomLbl.addEventListener("click", () => setZoom(1.0));

            dock.appendChild(head);

            // Body
            const body = document.createElement("div");
            body.className = "mmh3te-body";
            dock.appendChild(body);

            // Resize handles
            for (const d of ["n", "s", "e", "w", "se", "sw", "ne", "nw"]) {
                const h = document.createElement("div");
                h.className = `mmh3te-rsz ${d}`;
                h.dataset.dir = d;
                dock.appendChild(h);
            }

            // ── Toolbar ──
            const bar = document.createElement("div");
            bar.className = "mmh3te-bar";
            dock.insertBefore(bar, body);

            function addToolbarCombo(label, values, currentVal, onChange, titleMap) {
                const lbl = document.createElement("span");
                lbl.className = "mmh3te-lbl";
                lbl.textContent = label;
                bar.appendChild(lbl);
                const sel = document.createElement("select");
                sel.className = "mmh3te-sel";
                sel.style.width = "auto";
                for (const v of values) {
                    const opt = document.createElement("option");
                    opt.value = v;
                    opt.textContent = titleMap?.[v] || v;
                    if (v === currentVal) opt.selected = true;
                    sel.appendChild(opt);
                }
                sel.addEventListener("change", () => { onChange(sel.value); refresh(); });
                bar.appendChild(sel);
                return sel;
            }

            function rebuildToolbar() {
                bar.innerHTML = "";
                // Clear stale body-level compose dropdowns (they are removed from
                // the DOM along with the toolbar, but escape the dock as siblings).
                for (const d of _composeDropdownEls) d.remove();
                _composeDropdownEls.length = 0;
                const cp = currentPlan();
                addToolbarCombo(uistr("Plan:"), PLANS, cp, v => {
                    node._plan = v;
                    // Force tile_count to 5 for 4_quadrants_expand
                    if (v === "4_quadrants_expand") {
                        node._tileCount = 5;
                    }
                    serialize();
                }, PLAN_LABELS);

                // Tile count
                const tcLbl = document.createElement("span");
                tcLbl.className = "mmh3te-lbl";
                tcLbl.textContent = uistr("Tiles:");
                bar.appendChild(tcLbl);
                const isQuadrant = currentPlan() === "4_quadrants_expand";
                const tcInp = document.createElement("input");
                tcInp.type = "number"; tcInp.min = 1; tcInp.max = 12;
                tcInp.className = "mmh3te-inp";
                tcInp.style.width = "40px";
                tcInp.value = isQuadrant ? 5 : tileCount();
                if (isQuadrant) { tcInp.disabled = true; tcInp.title = uistr("4 Quadrants always uses 5 tiles"); }
                tcInp.addEventListener("change", () => {
                    node._tileCount = Math.max(1, Math.min(12, parseInt(tcInp.value) || 2));
                    serialize();
                    ensureTiles();
                    refresh();
                });
                bar.appendChild(tcInp);

                // Reset zoom button
                const resetBtn = document.createElement("button");
                resetBtn.className = "mmh3te-btn";
                resetBtn.textContent = "1:1";
                resetBtn.title = uistr("Reset zoom/pan");
                resetBtn.addEventListener("click", () => {
                    node._cvZoom = 1;
                    node._cvPanX = 0;
                    node._cvPanY = 0;
                    drawCanvas();
                });
                bar.appendChild(resetBtn);

                // ── Compose view toggle ──
                // The button only switches the RIGHT PANEL between tile
                // parameters and the compose setup. Whether compose is active
                // is derived from the loaded images: First or Last present =
                // on, both empty = off (tiles fall back to their own images).
                const compBtn = document.createElement("button");
                compBtn.className = "mmh3te-btn" + (node._composeView ? " on" : "");
                compBtn.textContent = uistr("Compose");
                compBtn.title = uistr("Toggle the right panel between tile parameters and the compose setup. Compose is active whenever a First or Last image is loaded; with both empty it is off.");
                compBtn.addEventListener("click", () => {
                    node._composeView = !node._composeView;
                    compBtn.classList.toggle("on", node._composeView);
                    renderPanel();
                });
                bar.appendChild(compBtn);
            }

            // ── Canvas ──
            const cvWrap = document.createElement("div");
            cvWrap.className = "mmh3te-cv-wrap";
            cvWrap.style.position = "relative";
            body.appendChild(cvWrap);

            const canvas = document.createElement("canvas");
            canvas.className = "mmh3te-canvas";
            cvWrap.appendChild(canvas);
            const ctx = canvas.getContext("2d");

            // Canvas overlay for image preview on hover
            const cvOverlay = document.createElement("div");
            cvOverlay.className = "mmh3te-cv-overlay";
            const cvOverlayImg = document.createElement("img");
            cvOverlay.appendChild(cvOverlayImg);
            const cvOverlayClose = document.createElement("button");
            cvOverlayClose.className = "mmh3te-cv-close";
            cvOverlayClose.textContent = "\u00d7";
            cvOverlayClose.title = uistr("Close preview");
            cvOverlayClose.addEventListener("click", (e) => {
                e.stopPropagation();
                cvOverlay.classList.remove("visible");
            });
            cvOverlay.appendChild(cvOverlayClose);
            cvWrap.appendChild(cvOverlay);

            // ── Divider between canvas and panel ──
            const divider = document.createElement("div");
            divider.className = "mmh3te-divider";
            body.appendChild(divider);

            let _isDraggingDivider = false;
            divider.addEventListener("pointerdown", (e) => {
                if (e.button !== 0) return;
                e.preventDefault();
                _isDraggingDivider = true;
                divider.classList.add("dragging");
                const startX = e.clientX;
                const startW = panel.offsetWidth;
                const onMove = (me) => {
                    if (!_isDraggingDivider) return;
                    const dx = startX - me.clientX;
                    const newW = Math.max(180, Math.min(500, startW + dx));
                    panel.style.width = newW + "px";
                    drawCanvas();
                };
                const onEnd = () => {
                    _isDraggingDivider = false;
                    divider.classList.remove("dragging");
                    document.removeEventListener("pointermove", onMove);
                    document.removeEventListener("pointerup", onEnd);
                };
                document.addEventListener("pointermove", onMove);
                document.addEventListener("pointerup", onEnd);
            });

            // ── Property panel ──
            const panel = document.createElement("div");
            panel.className = "mmh3te-panel";
            body.appendChild(panel);

            // ── Canvas drawing ──
            function drawCanvas() {
                const dpr = window.devicePixelRatio || 1;
                const dispW = cvWrap.clientWidth || 400;
                const dispH = cvWrap.clientHeight || 300;

                ensureTiles();
                const tiles = [];
                const n = tileCount();
                for (let i = 0; i < n; i++) tiles.push(node._tiles[i]);

                const cp = currentPlan();
                const total = totalSize(tiles, cp);
                const margin = 30;
                const autoScaleX = (dispW - margin * 2) / Math.max(total.w, 1);
                const autoScaleY = (dispH - margin * 2) / Math.max(total.h, 1);
                const autoScale = Math.min(autoScaleX, autoScaleY, 3.0);
                const scale = autoScale * node._cvZoom;

                const bw = Math.round(dispW * dpr);
                const bh = Math.round(dispH * dpr);
                if (canvas.width !== bw || canvas.height !== bh) {
                    canvas.width = bw; canvas.height = bh;
                }
                canvas.style.width = dispW + "px";
                canvas.style.height = dispH + "px";
                ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
                ctx.clearRect(0, 0, dispW, dispH);

                // Background
                ctx.fillStyle = "#111";
                ctx.fillRect(0, 0, dispW, dispH);

                const placements = computePlacements(tiles, cp);
                const baseOx = (dispW - total.w * autoScale) / 2;
                const baseOy = (dispH - total.h * autoScale) / 2;
                const ox = baseOx + node._cvPanX;
                const oy = baseOy + node._cvPanY;

                // FL2VA compose: draw the source frame under the tiles so the
                // semi-transparent tile rects sit on top of it.
                const compOn = composeOnFlag();
                if (compOn) {
                    const c = composeData();
                    // Preview strictly follows the target: only the source whose
                    // "Preview" mode is active is drawn. "none" hides BOTH images
                    // (no first-fallback), so the user sees the bare tile plan.
                    let kind = null;
                    if (c.target === "first" && c.first?.name) kind = "first";
                    else if (c.target === "last" && c.last?.name) kind = "last";
                    if (kind) {
                        const src = kind === "first" ? c.first : c.last;
                        const img = kind === "first" ? node._composeImgFirst : node._composeImgLast;
                        if (src && img && img.complete && img.naturalWidth) {
                            const ds = src.s ?? Math.min(img.naturalWidth / total.w, img.naturalHeight / total.h);
                            // Clamp the source window POSITION (size preserved)
                            // so it never leaves the image -- identical to the
                            // crop math in composeCropsFor(), keeping the
                            // preview WYSIWYG with the generated first frame.
                            let sx = src.ox ?? 0;
                            let sy = src.oy ?? 0;
                            const sw = total.w * ds, sh = total.h * ds;
                            sx = Math.max(0, Math.min(sx, img.naturalWidth - sw));
                            sy = Math.max(0, Math.min(sy, img.naturalHeight - sh));
                            ctx.drawImage(img, sx, sy, sw, sh,
                                          ox, oy, total.w * scale, total.h * scale);
                        }
                    }
                }

                for (let i = 0; i < tiles.length; i++) {
                    const p = placements[i];
                    const color = TILE_COLORS[i % TILE_COLORS.length];
                    const sx = ox + p.x * scale;
                    const sy = oy + p.y * scale;
                    const sw = p.w * scale;
                    const sh = p.h * scale;

                    // Fill (lighter when the compose image is behind)
                    ctx.fillStyle = color + (compOn ? "2e" : "40");
                    ctx.fillRect(sx, sy, sw, sh);

                    // Border
                    const isSelected = i === node._selectedIdx;
                    ctx.strokeStyle = isSelected ? "#ff8c00" : color;
                    ctx.lineWidth = isSelected ? 2.5 : 1.5;
                    ctx.strokeRect(sx + 0.5, sy + 0.5, sw - 1, sh - 1);

                    // Role label
                    const role = tileRole(i, cp);
                    const label = `#${i} ${role}`;
                    ctx.font = "bold 12px sans-serif";
                    ctx.fillStyle = "#fff";
                    ctx.textAlign = "left";
                    ctx.textBaseline = "top";
                    ctx.fillText(label, sx + 5, sy + 4);

                    // Dimensions
                    ctx.font = "11px sans-serif";
                    ctx.fillStyle = "#ccc";
                    ctx.textAlign = "center";
                    ctx.textBaseline = "middle";
                    const dimY = sy + sh / 2;
                    ctx.fillText(`${tiles[i].width}\u00d7${tiles[i].height}`, sx + sw / 2, dimY);

                    // Reference image indicator (above dimensions)
                    const refs = tiles[i].ref_images || [];
                    if (refs.length > 0) {
                        ctx.font = "10px sans-serif";
                        ctx.fillStyle = "#46b4e6";
                        ctx.textAlign = "center";
                        ctx.textBaseline = "bottom";
                        ctx.fillText(`\uD83D\uDCF7\u00d7${refs.length}`, sx + sw / 2, dimY - 8);
                    }

                    // Prompt hint (below dimensions)
                    const prompt = tiles[i].prompt || "";
                    if (prompt) {
                        ctx.font = "10px sans-serif";
                        ctx.fillStyle = "#aaa";
                        ctx.textAlign = "center";
                        ctx.textBaseline = "top";
                        const hint = prompt.length > 18 ? prompt.slice(0, 18) + "\u2026" : prompt;
                        ctx.fillText(hint, sx + sw / 2, dimY + 8);
                    }

                    // Overlap bands
                    if (i > 0) {
                        const { scheme, axis } = planToSchemeAxis(cp);
                        ctx.fillStyle = "rgba(255,255,255,0.08)";
                        if (scheme === "alternating") {
                            if (axis === "horizontal") {
                                const ow = tiles[i].overlap_w * scale;
                                if (i % 2 === 1) ctx.fillRect(sx, sy, ow, sh);
                                else ctx.fillRect(sx + sw - ow, sy, ow, sh);
                            } else {
                                const oh = tiles[i].overlap_h * scale;
                                if (i % 2 === 1) ctx.fillRect(sx, sy, sw, oh);
                                else ctx.fillRect(sx, sy + sh - oh, sw, oh);
                            }
                        }
                    }
                }

                // Total dimensions label
                ctx.font = "10px sans-serif";
                ctx.fillStyle = "#666";
                ctx.textAlign = "right";
                ctx.textBaseline = "bottom";
                ctx.fillText(`${total.w}\u00d7${total.h} px`, dispW - 8, dispH - 4);

                // Zoom indicator
                ctx.textAlign = "left";
                ctx.fillText(`${uistr("Zoom")}: ${Math.round(node._cvZoom * 100)}%`, 8, dispH - 4);
            }

            // ── Canvas coordinate helpers ──
            function screenToCanvas(e) {
                const rect = canvas.getBoundingClientRect();
                const dispW = rect.width;
                const dispH = rect.height;
                const mx = e.clientX - rect.left;
                const my = e.clientY - rect.top;

                ensureTiles();
                const tiles = [];
                for (let i = 0; i < tileCount(); i++) tiles.push(node._tiles[i]);
                const cp = currentPlan();
                const total = totalSize(tiles, cp);
                const margin = 30;
                const autoScaleX = (dispW - margin * 2) / Math.max(total.w, 1);
                const autoScaleY = (dispH - margin * 2) / Math.max(total.h, 1);
                const autoScale = Math.min(autoScaleX, autoScaleY, 3.0);
                const scale = autoScale * node._cvZoom;
                const baseOx = (dispW - total.w * autoScale) / 2;
                const baseOy = (dispH - total.h * autoScale) / 2;
                const ox = baseOx + node._cvPanX;
                const oy = baseOy + node._cvPanY;

                // Convert screen coords to graph coords (tile pixel space)
                const gx = (mx - ox) / scale;
                const gy = (my - oy) / scale;
                return { gx, gy, mx, my, scale, ox, oy, tiles, cp, total };
            }

            // ── Canvas: wheel zoom + click select + drag pan ──
            canvas.addEventListener("wheel", (e) => {
                e.preventDefault();
                const rect = canvas.getBoundingClientRect();
                const mx = e.clientX - rect.left;
                const my = e.clientY - rect.top;

                // FL2VA compose: wheel zooms the source viewport (not the canvas)
                // -- only while the Compose panel is open; in tile view the
                // wheel keeps its normal canvas zoom so the plan stays navigable.
                const ed = node._composeView ? composeEditSrc() : null;
                if (ed?.ref?.name) {
                    const img = ed.kind === "first" ? node._composeImgFirst : node._composeImgLast;
                    if (img?.complete && img.naturalWidth) {
                        const total = totalSize(tilesArr(), currentPlan());
                        const iw = img.naturalWidth, ih = img.naturalHeight;
                        const sOld = ed.ref.s ?? Math.min(iw / total.w, ih / total.h);
                        const px = (ed.ref.ox ?? 0) + total.w * sOld / 2;
                        const py = (ed.ref.oy ?? 0) + total.h * sOld / 2;
                        ed.ref.s = sOld * (e.deltaY < 0 ? 0.85 : 1.18);
                        clampViewport(ed.ref, iw, ih, total);
                        ed.ref.ox = px - total.w * ed.ref.s / 2;
                        ed.ref.oy = py - total.h * ed.ref.s / 2;
                        clampViewport(ed.ref, iw, ih, total);
                        serialize(); drawCanvas();
                        syncComposeOffsetInputs();
                    }
                    return;
                }

                const oldZoom = node._cvZoom;
                const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9;
                node._cvZoom = Math.max(0.1, Math.min(20, node._cvZoom * zoomFactor));

                // Zoom toward cursor: adjust pan so the point under cursor stays fixed
                const newAutoScale = oldZoom; // autoScale doesn't change, only zoom does
                // The cursor position in graph coords before zoom:
                //   gx = (mx - baseOx - panX) / (autoScale * oldZoom)
                // After zoom, we want the same gx:
                //   mx = gx * (autoScale * newZoom) + baseOx + newPanX
                //   newPanX = mx - gx * (autoScale * newZoom) - baseOx
                // But since autoScale doesn't change, we can simplify:
                const ratio = node._cvZoom / oldZoom;
                node._cvPanX = mx - ratio * (mx - node._cvPanX);
                node._cvPanY = my - ratio * (my - node._cvPanY);

                drawCanvas();
            }, { passive: false });

            let _isDraggingCanvas = false;
            let _dragStartX, _dragStartY, _dragPanStartX, _dragPanStartY;
            // Cycle-through-layers state
            let _lastClickGx = NaN, _lastClickGy = NaN, _clickCycleIdx = 0;

            function pickTileAt(e) {
                const { gx, gy, tiles, cp } = screenToCanvas(e);
                const placements = computePlacements(tiles, cp);
                const hits = [];
                for (let i = tiles.length - 1; i >= 0; i--) {
                    const p = placements[i];
                    if (gx >= p.x && gx <= p.x + p.w && gy >= p.y && gy <= p.y + p.h) hits.push(i);
                }
                let selected = -1;
                if (hits.length > 0) {
                    // Same position → cycle through stacked tiles, else topmost
                    const samePos = gx === _lastClickGx && gy === _lastClickGy;
                    if (samePos && hits.length > 1) {
                        _clickCycleIdx = (_clickCycleIdx + 1) % hits.length;
                        selected = hits[_clickCycleIdx];
                    } else {
                        _clickCycleIdx = 0;
                        selected = hits[0];
                    }
                    _lastClickGx = gx;
                    _lastClickGy = gy;
                }
                if (selected >= 0) {
                    node._selectedIdx = selected;
                    drawCanvas();
                    renderPanel();
                }
                return selected >= 0;
            }

            canvas.addEventListener("pointerdown", (e) => {
                // Middle button or shift+click = pan
                if (e.button === 1 || (e.button === 0 && e.shiftKey)) {
                    e.preventDefault();
                    _isDraggingCanvas = true;
                    _dragStartX = e.clientX;
                    _dragStartY = e.clientY;
                    _dragPanStartX = node._cvPanX;
                    _dragPanStartY = node._cvPanY;
                    canvas.style.cursor = "grab";
                    return;
                }
                if (e.button !== 0) return;

                // FL2VA compose: defer the viewport drag until an actual drag so
                // a plain click below still selects the tile underneath. Only
                // while the Compose panel is open; tile view keeps canvas pan.
                const cs = (node._composeView && composeOnFlag()) ? composeEditSrc() : null;
                if (cs) {
                    _composeDrag = cs;
                    _composeDragStart = { ox: cs.ref.ox ?? 0, oy: cs.ref.oy ?? 0 };
                    _dragStartX = e.clientX;
                    _dragStartY = e.clientY;
                    _dragPanStartX = node._cvPanX;
                    _dragPanStartY = node._cvPanY;
                    _composePending = { active: true };
                    return;
                }

                // Left click = select tile, or pan if no tile hit
                if (!pickTileAt(e)) {
                    e.preventDefault();
                    _isDraggingCanvas = true;
                    _dragStartX = e.clientX;
                    _dragStartY = e.clientY;
                    _dragPanStartX = node._cvPanX;
                    _dragPanStartY = node._cvPanY;
                    canvas.style.cursor = "grab";
                }
            });

            canvas.addEventListener("pointermove", (e) => {
                if (_composePending?.active) {
                    // Engage the compose viewport drag only once the pointer actually
                    // moves past a small dead zone (so a plain click stays a click).
                    if (Math.hypot(e.clientX - _dragStartX, e.clientY - _dragStartY) < 4) return;
                    _composePending.active = false;
                    if (!_composeDrag?.ref) return;
                    _isDraggingCanvas = true;
                    canvas.style.cursor = "grab";
                }
                if (!_isDraggingCanvas) return;
                // FL2VA compose: drag pans the source viewport across the image
                if (_composeDrag?.ref) {
                    const ed = _composeDrag;
                    const dispW = cvWrap.clientWidth || 400, dispH = cvWrap.clientHeight || 300;
                    const total = totalSize(tilesArr(), currentPlan());
                    const autoScale = Math.min((dispW - 60) / Math.max(total.w, 1), (dispH - 60) / Math.max(total.h, 1), 3.0);
                    const scaleV = autoScale * node._cvZoom;
                    const s = ed.ref.s ?? 1;
                    const dx = (e.clientX - _dragStartX), dy = (e.clientY - _dragStartY);
                    ed.ref.ox = _composeDragStart.ox - (dx / scaleV) * s;
                    ed.ref.oy = _composeDragStart.oy - (dy / scaleV) * s;
                    const img = ed.kind === "first" ? node._composeImgFirst : node._composeImgLast;
                    if (img?.naturalWidth) {
                        ed.ref.ox = Math.max(0, Math.min(ed.ref.ox, img.naturalWidth - total.w * s));
                        ed.ref.oy = Math.max(0, Math.min(ed.ref.oy, img.naturalHeight - total.h * s));
                    }
                    drawCanvas();
                    return;
                }
                node._cvPanX = _dragPanStartX + (e.clientX - _dragStartX);
                node._cvPanY = _dragPanStartY + (e.clientY - _dragStartY);
                drawCanvas();
            });

canvas.addEventListener("pointerup", (e) => {
                if (_composePending?.active) {
                    // A click (no movement): cancel the pending viewport drag and
                    // select the tile under the cursor instead.
                    _composePending = null;
                    _composeDrag = null;
                    _composeDragStart = null;
                    if (e.button === 0) pickTileAt(e);
                    return;
                }
                if (_isDraggingCanvas) {
                    _isDraggingCanvas = false;
                    canvas.style.cursor = "pointer";
                }
                if (_composeDrag) {
                    serialize();
                    syncComposeOffsetInputs();
                    _composeDrag = null;
                    _composeDragStart = null;
                }
            });

            canvas.addEventListener("pointerleave", () => {
                _composePending = null;
                if (_isDraggingCanvas) {
                    _isDraggingCanvas = false;
                    canvas.style.cursor = "pointer";
                }
                if (_composeDrag) {
                    serialize();
                    syncComposeOffsetInputs();
                    _composeDrag = null;
                    _composeDragStart = null;
                }
            });

            // ── Whole-image compose support ──
            // Serialized as tile_data.compose = { target,
            //   first|last: { name, s (source px / plan px), ox, oy (source
            //   px of the plan-region origin) } }. Compose is mode-agnostic: it
            // only supplies whole images that the tile plan splits; each tile
            // independently decides (per its cond_mode / ref_source) whether
            // its split crop feeds FL2VA frames or Ref2VA blocks, or is
            // ignored in favour of its own images.
            function composeData() {
                return node.compose || (node.compose = { first:null, last:null, target:"none" });
            }
            // Compose is ACTIVE whenever a First or Last image is loaded; there
            // is no separate enable switch any more (the toolbar Compose button
            // only switches the right panel view).
            function composeOnFlag() {
                const c = composeData();
                return !!(c.first?.name || c.last?.name);
            }
            function composeEditSrc() {
                const c = composeData();
                if (!composeOnFlag()) return null;
                if (c.target === "first" && c.first?.name) return { ref: c.first, kind: "first" };
                if (c.target === "last" && c.last?.name) return { ref: c.last, kind: "last" };
                return null;
            }
            async function loadComposeImg(kind) {
                const c = composeData();
                const key = kind === "first" ? "_composeImgFirst" : "_composeImgLast";
                node[key] = null;
                const src = kind === "first" ? c.first : c.last;
                if (!src?.name) return;
                try {
                    const img = new Image();
                    img.src = imageUrl(src.name);
                    await img.decode();
                    node[key] = img;
                    src.iw = img.naturalWidth;
                    src.ih = img.naturalHeight;
                    if (src.s == null) {
                        const iw = img.naturalWidth, ih = img.naturalHeight;
                        const total = totalSize(tilesArr(), currentPlan());
                        const s = Math.min(iw / Math.max(total.w, 1), ih / Math.max(total.h, 1));
                        src.s = s;
                        src.ox = (iw - total.w * s) / 2;
                        src.oy = (ih - total.h * s) / 2;
                    }
                    // Apply an offset typed before the image finished
                    // decoding (the input stashes it as pendingOffX/Y instead
                    // of silently dropping it).
                    if (src.pendingOffX != null || src.pendingOffY != null) {
                        if (src.pendingOffX != null) src.ox = src.pendingOffX * src.s;
                        if (src.pendingOffY != null) src.oy = src.pendingOffY * src.s;
                        delete src.pendingOffX;
                        delete src.pendingOffY;
                        clampViewport(src, src.iw, src.ih,
                                      totalSize(tilesArr(), currentPlan()));
                    }
                    syncComposeOffsetInputs();
                    // Persist the computed viewport AND re-derive per-tile crop
                    // boxes (compose_crops) so tile_config always carries them
                    // once the source image is actually decoded and
                    // viewport-ready.
                    serialize();
                    // Re-render the compose panel so the Offset inputs pick up
                    // the decoded size (they stay disabled until s is known).
                    if (node._composeView) renderPanel();
                } catch { /* ignore decode failures */ }
                // Redraw once the compose source is actually available so the
                // split preview refreshes automatically after picking an image.
                if (drawCanvas) drawCanvas();
            }
            function clampViewport(ref, iw, ih, total) {
                const smax = Math.min(iw / Math.max(total.w, 1), ih / Math.max(total.h, 1));
                const smin = smax / 32;
                ref.s = Math.max(smin, Math.min(ref.s ?? smax, smax));
                ref.ox = Math.max(0, Math.min(ref.ox ?? 0, iw - total.w * ref.s));
                ref.oy = Math.max(0, Math.min(ref.oy ?? 0, ih - total.h * ref.s));
            }
            // Offset inputs of the compose panel, kept in sync with the
            // viewport (drag/zoom change ox/oy -> the displayed plan-space
            // position must follow). Rebuilt by buildComposePanel.
            const _composeOffInputs = { first: null, last: null };
            function syncComposeOffsetInputs() {
                const c = composeData();
                for (const kind of ["first", "last"]) {
                    const m = _composeOffInputs[kind];
                    if (!m) continue;
                    const r = c[kind];
                    for (const ax of ["X", "Y"]) {
                        const inp = m[ax];
                        if (!inp) continue;
                        inp.value = (r && r.s)
                            ? Math.round(r[ax === "X" ? "ox" : "oy"] / r.s) : 0;
                    }
                }
            }
            let _composeDrag = null, _composeDragStart = null, _composePending = null;
            // Outer-scope image-list refresh (the nested per-tile refreshInputImages
            // is not visible to the compose toolbar, so this wraps the module-level
            // fetch and exposes the shared cache to onNodeCreated handlers).
            async function refreshAllImages() {
                invalidateInputImages();
                window._mmh3teInputImages = await fetchInputImages();
            }

            // ── Property panel rendering ──
            function renderPanel() {
                panel.innerHTML = "";
                // Remove any body-level ref/compose dropdowns left over from a
                // previous render (they are not children of the panel, so
                // innerHTML="" does not clear them).
                for (const d of _refDropdownEls) d.remove();
                _refDropdownEls.length = 0;
                for (const d of _composeDropdownEls) d.remove();
                _composeDropdownEls.length = 0;
                _refArea = null;
                ensureTiles();

                // Compose view: right panel shows the compose setup instead of
                // the per-tile parameters (toolbar Compose button toggles).
                if (node._composeView) { buildComposePanel(); return; }

                const idx = node._selectedIdx;
                if (idx < 0 || idx >= tileCount()) {
                    const hint = document.createElement("div");
                    hint.style.cssText = "color:#666; text-align:center; padding:20px 0;";
                    hint.textContent = uistr("Click a tile on the canvas to edit it");
                    panel.appendChild(hint);
                    return;
                }

                const tile = node._tiles[idx];
                const cp = currentPlan();
                const role = tileRole(idx, cp);

                // Initialize drawer states
                if (!node._drawerStates) node._drawerStates = {};
                const ds = node._drawerStates;

                function makeDrawer(title, defaultOpen, buildFn, bodyClass, stateKey) {
                    // stateKey pins the open/closed state across title changes
                    // (e.g. the ref drawer is renamed per conditioning mode).
                    const key = `drawer_${stateKey || title}`;
                    if (ds[key] === undefined) ds[key] = defaultOpen;
                    const hdr = document.createElement("div");
                    hdr.className = "mmh3te-hdr collapsible" + (ds[key] ? " open" : "");
                    hdr.textContent = title;
                    const body = document.createElement("div");
                    body.className = "mmh3te-drawer" + (ds[key] ? "" : " closed") + (bodyClass ? " " + bodyClass : "");
                    hdr.addEventListener("click", () => {
                        ds[key] = !ds[key];
                        hdr.classList.toggle("open", ds[key]);
                        body.classList.toggle("closed", !ds[key]);
                    });
                    panel.appendChild(hdr);
                    panel.appendChild(body);
                    buildFn(body);
                }

                // Resizable prompt drawer. The whole drawer is a fixed-height flex
                // column: the textarea fills the space and the drag handle is a
                // flex item pinned at the bottom INSIDE the drawer, so it is never
                // pushed out or clipped regardless of the set height. Collapse
                // still works via the existing `.closed { max-height:0 }` rule
                // (max-height caps the inline height). Height is remembered per
                // prompt type across tiles/re-renders.
                function addPromptDrawer(title, defaultOpen, defaultH, placeholder, getVal, setVal, needsCanvas) {
                    makeDrawer(title, defaultOpen, (b) => {
                        if (!node._promptDrawerHeights) node._promptDrawerHeights = {};
                        const H = Math.max(60, node._promptDrawerHeights[title] ?? defaultH);
                        node._promptDrawerHeights[title] = H;
                        b.style.height = H + "px";

                        const area = document.createElement("textarea");
                        area.className = "mmh3te-area";
                        area.value = getVal() || "";
                        area.placeholder = placeholder;

                        area.addEventListener("input", () => {
                            setVal(area.value);
                            if (needsCanvas) { serialize(); drawCanvas(); }
                        });

                        const bar = document.createElement("div");
                        bar.className = "mmh3te-dragbar";
                        bar.title = uistr("Drag to resize");

                        const MIN = 60;
                        const MAX = Math.max(MIN + 80, Math.round(window.innerHeight * 0.5));
                        bar.addEventListener("pointerdown", (e) => {
                            e.preventDefault();
                            e.stopPropagation();
                            const startY = e.clientY;
                            const startH = b.offsetHeight;
                            bar.classList.add("dragging");
                            document.body.classList.add("mmh3te-vdrag");
                            const move = (ev) => {
                                const nh = Math.max(MIN, Math.min(MAX, startH + (ev.clientY - startY)));
                                b.style.height = nh + "px";
                                node._promptDrawerHeights[title] = nh;
                            };
                            const up = () => {
                                bar.classList.remove("dragging");
                                document.body.classList.remove("mmh3te-vdrag");
                                window.removeEventListener("pointermove", move);
                                window.removeEventListener("pointerup", up);
                            };
                            window.addEventListener("pointermove", move);
                            window.addEventListener("pointerup", up);
                        });

                        b.appendChild(area);
                        b.appendChild(bar);
                    }, "resizable");
                }

                // Header
                const hdr = document.createElement("div");
                hdr.className = "mmh3te-hdr";
                hdr.textContent = `${uistr("Tile")} #${idx} (${role})`;
                hdr.style.color = TILE_COLORS[idx % TILE_COLORS.length];
                panel.appendChild(hdr);

                // ── Conditioning mode (per tile) ──
                const modeRow = document.createElement("div");
                modeRow.className = "mmh3te-row";
                modeRow.style.margin = "4px 0";
                modeRow.appendChild(Object.assign(document.createElement("span"), {
                    className: "mmh3te-lbl", textContent: uistr("Mode:") }));
                const modeSel = document.createElement("select");
                modeSel.className = "mmh3te-sel";
                modeSel.title = uistr("Conditioning mode for this tile: FL2VA = its reference image(s) become the first (and optional last) frame; Ref2VA = they become subject reference blocks. Tiles can mix freely within one plan.");
                [[uistr("FL2VA (first/last frames)"), "FL2VA (first/last frames)"], [uistr("Ref2VA (reference images)"), "Ref2VA (reference images)"]].forEach(([lab, v]) => {
                    const o = document.createElement("option");
                    o.value = v; o.textContent = lab;
                    if (v === (tile.cond_mode || "FL2VA")) o.selected = true;
                    modeSel.appendChild(o);
                });
                modeSel.addEventListener("change", () => {
                    tile.cond_mode = modeSel.value;
                    serialize();
                    renderPanel(); // rebuild: drawer title / ref limits follow the mode
                });
                modeRow.appendChild(modeSel);
                panel.appendChild(modeRow);

                // ── Positive Prompt (collapsible, resizable) ──
                addPromptDrawer(
                    uistr("Positive Prompt (this tile)"), true, 110,
                    uistr("Appended to base prompt"),
                    () => tile.prompt, (v) => { tile.prompt = v; }, true
                );

                // ── Negative Prompt (collapsible, resizable) ──
                addPromptDrawer(
                    uistr("Negative Prompt (this tile)"), false, 90,
                    uistr("Appended to base negative"),
                    () => tile.negative, (v) => { tile.negative = v; }, false
                );

                panel.appendChild(Object.assign(document.createElement("hr"), { className: "mmh3te-sep" }));

                // ── Dimensions (collapsible) ──
                makeDrawer(uistr("Dimensions"), true, (body) => {
                    if (cp === "horizontal_alternating") {
                        const hint = document.createElement("div");
                        hint.style.cssText = "font:10px sans-serif; color:#666; padding:2px 0;";
                        hint.textContent = uistr("All tiles share the same height");
                        body.appendChild(hint);
                    } else if (cp === "vertical_alternating") {
                        const hint = document.createElement("div");
                        hint.style.cssText = "font:10px sans-serif; color:#666; padding:2px 0;";
                        hint.textContent = uistr("All tiles share the same width");
                        body.appendChild(hint);
                    }

                    function addDimRow(label, prop, step) {
                        const row = document.createElement("div");
                        row.className = "mmh3te-row";
                        const lbl = document.createElement("span");
                        lbl.className = "mmh3te-lbl";
                        lbl.textContent = label;
                        row.appendChild(lbl);
                        const inp = document.createElement("input");
                        inp.type = "number"; inp.className = "mmh3te-inp";
                        inp.min = 32; inp.max = 8192; inp.step = step || 32;
                        inp.value = tile[prop] || DEF_W;
                        inp.addEventListener("change", () => {
                            const newVal = Math.max(32, parseInt(inp.value) || 32);
                            tile[prop] = newVal;
                            syncSharedDimension(idx, prop, newVal);
                            serialize(); drawCanvas();
                            renderPanel();
                        });
                        row.appendChild(inp);
                        body.appendChild(row);
                    }
                    addDimRow(uistr("Width:"), "width", 32);
                    addDimRow(uistr("Height:"), "height", 32);
                });

                panel.appendChild(Object.assign(document.createElement("hr"), { className: "mmh3te-sep" }));

                // ── Overlap / Fade (collapsible) ──
                makeDrawer(uistr("Overlap / Fade"), false, (body) => {
                    function addOverlapRow(label, prop, globalVal, disabled) {
                        const row = document.createElement("div");
                        row.className = "mmh3te-row";
                        if (disabled) row.style.opacity = "0.4";
                        const lbl = document.createElement("span");
                        lbl.className = "mmh3te-lbl";
                        lbl.textContent = label;
                        row.appendChild(lbl);
                        const inp = document.createElement("input");
                        inp.type = "number"; inp.className = "mmh3te-inp";
                        inp.min = 0; inp.max = 4096; inp.step = 32;
                        inp.value = tile[prop] != null ? tile[prop] : "";
                        inp.placeholder = String(globalVal);
                        if (disabled) { inp.disabled = true; inp.title = uistr("Not applicable for this plan/tile"); }
                        inp.addEventListener("change", () => {
                            const v = parseInt(inp.value);
                            if (isNaN(v) || v < 0) {
                                delete tile[prop];
                                eff.textContent = `= ${globalVal}`;
                            } else {
                                tile[prop] = v;
                                eff.textContent = `= ${v}`;
                            }
                            serialize(); drawCanvas();
                        });
                        row.appendChild(inp);

                        const eff = document.createElement("span");
                        eff.style.cssText = "font:10px monospace; color:#666;";
                        const curVal = tile[prop];
                        eff.textContent = curVal != null ? `= ${curVal}` : `= ${globalVal}`;
                        row.appendChild(eff);

                        body.appendChild(row);
                    }

                    const isCenter = idx === 0;
                    const isHoriz = cp === "horizontal_alternating";
                    const isVert = cp === "vertical_alternating";
                    const owDisabled = isCenter || isVert;
                    const ohDisabled = isCenter || isHoriz;
                    const fwDisabled = isCenter || isVert;
                    const fhDisabled = isCenter || isHoriz;

                    addOverlapRow(uistr("Overlap W:"), "overlap_w", DEF_OW, owDisabled);
                    addOverlapRow(uistr("Overlap H:"), "overlap_h", DEF_OH, ohDisabled);
                    addOverlapRow(uistr("Fade W:"), "fade_w", DEF_FW, fwDisabled);
                    addOverlapRow(uistr("Fade H:"), "fade_h", DEF_FH, fhDisabled);
                });

                panel.appendChild(Object.assign(document.createElement("hr"), { className: "mmh3te-sep" }));

                // ── First/Last Images vs Reference Images (per tile mode) ──
                const refContainer = document.createElement("div");
                const tileMode = tile.cond_mode || "FL2VA";
                const refDrawerTitle = tileMode === "FL2VA"
                    ? uistr("First/Last Images") : uistr("Reference Images");
                // "Tile crop" source only exists while compose is on and at
                // least one compose source image is set.
                const composeAvailable = composeOnFlag();
                makeDrawer(refDrawerTitle, true, (body) => {
                    if (composeAvailable) {
                        const srcRow = document.createElement("div");
                        srcRow.className = "mmh3te-row";
                        srcRow.appendChild(Object.assign(document.createElement("span"), {
                            className: "mmh3te-lbl", textContent: uistr("Source:") }));
                        const srcSel = document.createElement("select");
                        srcSel.className = "mmh3te-sel";
                        srcSel.title = uistr("Tile crop: use this tile's region of the compose First/Last image (as FL2VA frames or Ref2VA blocks, per the Mode above). Own images: load separate pictures below.");
                        [[uistr("Tile crop"), "crop"], [uistr("Own images"), "own"]].forEach(([lab, v]) => {
                            const o = document.createElement("option");
                            o.value = v; o.textContent = lab;
                            if (v === (tile.ref_source || "crop")) o.selected = true;
                            srcSel.appendChild(o);
                        });
                        srcSel.addEventListener("change", () => {
                            tile.ref_source = srcSel.value;
                            serialize();
                            renderPanel(); // rebuild: crop mode hides the image list
                        });
                        srcRow.appendChild(srcSel);
                        body.appendChild(srcRow);
                    }
                    body.appendChild(refContainer);
                }, null, "refs");

                if (!tile.ref_images) tile.ref_images = [];

                function rebuildRefList() {
                    refContainer.innerHTML = "";
                    // Reset stale body-level dropdowns and point the shared
                    // outside-click handler at the current trigger area.
                    for (const d of _refDropdownEls) d.remove();
                    _refDropdownEls.length = 0;
                    _refArea = refContainer;

                    // Per-tile mode decides the drawer semantics and ref limit.
                    const mode = tile.cond_mode || "FL2VA";
                    const maxRefs = mode === "FL2VA" ? 2 : 10;
                    const roleHints = mode === "FL2VA" ? [uistr("First frame"), uistr("Last frame")] : [];

                    // Tile-crop source: this tile consumes its compose split
                    // region, so no own-image rows are shown.
                    if (composeAvailable && (tile.ref_source || "crop") === "crop") {
                        const note = document.createElement("div");
                        note.style.cssText = "font:11px sans-serif; color:#888; padding:6px 2px; line-height:1.5;";
                        note.textContent = tileMode === "FL2VA"
                            ? uistr("Using this tile's compose split as its first/last frames. Switch Source to 'Own images' to load separate pictures.")
                            : uistr("Using this tile's compose split as its reference block(s). Switch Source to 'Own images' to load separate pictures.");
                        refContainer.appendChild(note);
                        return;
                    }

                    const images = tile.ref_images;
                    const allImages = window._mmh3teInputImages || [];

                    for (let ri = 0; ri < images.length; ri++) {
                        const row = document.createElement("div");
                        row.className = "mmh3te-ref-row";

                        // Custom dropdown picker
                        const picker = document.createElement("div");
                        picker.className = "mmh3te-ref-picker";

                        // Trigger button
                        const trigger = document.createElement("div");
                        trigger.className = "mmh3te-ref-trigger";
                        const curFile = images[ri];
                        const imagesLoaded = !!window._mmh3teInputImages;
                        const fileExists = curFile && imagesLoaded && window._mmh3teInputImages.includes(curFile);
                        if (curFile && imagesLoaded) {
                            const tImg = document.createElement("img");
                            tImg.src = imageUrl(curFile);
                            trigger.appendChild(tImg);
                        }
                        if (curFile && imagesLoaded && !fileExists) {
                            trigger.classList.add("missing");
                            trigger.title = uistr("File not found in input folder");
                        }
                        const nameSpan = document.createElement("span");
                        nameSpan.className = "name";
                        nameSpan.textContent = curFile || uistr("(no image)");
                        trigger.appendChild(nameSpan);
                        const arrow = document.createElement("span");
                        arrow.className = "arrow";
                        arrow.textContent = "\u25BC";
                        trigger.appendChild(arrow);
                        picker.appendChild(trigger);

                        // Dropdown panel — appended to document.body, NOT inside
                        // the dock. A `position:fixed` element that is a child of
                        // the dock is confined to the dock (its `transform` makes
                        // it the containing block) and clipped by the dock's
                        // `overflow:hidden`, so it would never show. Positioned at
                        // the trigger's viewport rect instead.
                        const dropdown = document.createElement("div");
                        dropdown.className = "mmh3te-ref-dropdown";
                        _refDropdownEls.push(dropdown);
                        document.body.appendChild(dropdown);

                        // None option
                        const noneOpt = document.createElement("div");
                        noneOpt.className = "mmh3te-ref-opt" + (!curFile ? " selected" : "");
                        const noneName = document.createElement("span");
                        noneName.className = "name";
                        noneName.textContent = uistr("(no image)");
                        noneOpt.appendChild(noneName);
                        noneOpt.addEventListener("click", () => {
                            images[ri] = "";
                            dropdown.classList.remove("open");
                            cvOverlay.classList.remove("visible");
                            serialize(); drawCanvas();
                            rebuildRefList();
                        });
                        noneOpt.addEventListener("pointerenter", () => {
                            cvOverlay.classList.remove("visible");
                        });
                        dropdown.appendChild(noneOpt);

                        // Image options
                        for (const file of allImages) {
                            const opt = document.createElement("div");
                            opt.className = "mmh3te-ref-opt" + (file === curFile ? " selected" : "");
                            const oImg = document.createElement("img");
                            oImg.src = imageUrl(file);
                            opt.appendChild(oImg);
                            const oName = document.createElement("span");
                            oName.className = "name";
                            oName.textContent = file;
                            opt.appendChild(oName);
                            // Hover → show preview on canvas overlay
                            opt.addEventListener("pointerenter", () => {
                                cvOverlayImg.src = imageUrl(file);
                                cvOverlay.classList.add("visible");
                            });
                            opt.addEventListener("click", () => {
                                images[ri] = file;
                                dropdown.classList.remove("open");
                                cvOverlay.classList.remove("visible");
                                serialize(); drawCanvas();
                                rebuildRefList();
                            });
                            dropdown.appendChild(opt);
                        }
                        // (dropdown is appended to document.body above; trigger
                        //  is the only child that stays inside the picker)

                        // Toggle dropdown
                        trigger.addEventListener("click", (e) => {
                            e.stopPropagation();
                            // Close all other dropdowns
                            for (const d of _refDropdownEls) {
                                if (d !== dropdown) d.classList.remove("open");
                            }
                            const opening = !dropdown.classList.contains("open");
                            dropdown.classList.toggle("open");
                            if (opening) {
                                const r = trigger.getBoundingClientRect();
                                dropdown.style.left = Math.max(4, r.left) + "px";
                                dropdown.style.top = (r.bottom + 4) + "px";
                                dropdown.style.minWidth = Math.max(240, r.width) + "px";
                            }
                        });

                        row.appendChild(picker);
                        const xBtn = document.createElement("button");
                        xBtn.className = "mmh3te-xbtn";
                        xBtn.textContent = "\u00d7";
                        xBtn.title = uistr("Remove image");
                        xBtn.addEventListener("click", () => {
                            images.splice(ri, 1);
                            serialize(); drawCanvas();
                            rebuildRefList();
                        });
                        row.appendChild(xBtn);

                        // Role hint for FL2VA
                        if (mode === "FL2VA" && roleHints[ri]) {
                            const hint = document.createElement("span");
                            hint.style.cssText = "font:10px sans-serif; color:#666;";
                            hint.textContent = roleHints[ri];
                            row.appendChild(hint);
                        }

                        refContainer.appendChild(row);
                    }

                    // Add button
                    if (images.length < maxRefs) {
                        const addBtn = document.createElement("button");
                        addBtn.className = "mmh3te-addbtn";
                        addBtn.textContent = uistr("+ Add Image");
                        addBtn.addEventListener("click", async () => {
                            await refreshInputImages();
                            images.push("");
                            serialize(); drawCanvas();
                            rebuildRefList();
                            if (node._composeView) renderPanel();
                        });
                        refContainer.appendChild(addBtn);
                    }

                    // Outside-click closing is handled by the single shared
                    // document "pointerdown" listener registered in onNodeCreated
                    // (_closeRefDropdown), which also accounts for the body-level
                    // dropdowns and the current trigger area.
                }

                async function refreshInputImages() {
                    invalidateInputImages();
                    window._mmh3teInputImages = await fetchInputImages();
                }

                rebuildRefList();
                refreshInputImages().then(() => { rebuildRefList(); if (node._composeView) renderPanel(); });
            }


            // ── Compose panel (right side) ──
            // Shown instead of the tile parameters while the toolbar Compose
            // toggle is active. Compose itself is enabled whenever a First or
            // Last image is loaded (composeOnFlag derives it); the Offset X/Y
            // inputs shift the tile region relative to the loaded image,
            // independently for First and Last.
            function buildComposePanel() {
                const c = composeData();

                const hdr = document.createElement("div");
                hdr.className = "mmh3te-hdr";
                hdr.textContent = uistr("Compose");
                panel.appendChild(hdr);

                const note = document.createElement("div");
                note.style.cssText = "font:10px sans-serif; color:#666; padding:2px 0;";
                note.textContent = composeOnFlag()
                    ? uistr("Active: tiles whose Source is 'Tile crop' use their region of the images below.")
                    : uistr("Inactive: load a First or Last image below to enable compose.");
                panel.appendChild(note);

                // ── Preview target chooser ──
                const editRow = document.createElement("div");
                editRow.className = "mmh3te-row";
                editRow.appendChild(Object.assign(document.createElement("span"), {
                    className: "mmh3te-lbl", textContent: uistr("Preview:") }));
                const tgtSel = document.createElement("select");
                tgtSel.className = "mmh3te-sel";
                tgtSel.title = uistr("Which compose source (First/Last) is drawn on the canvas and re-framed by wheel/drag while the Compose panel is open. 'None' hides both images.");
                [[uistr("(none)"), "none"], [uistr("First"), "first"], [uistr("Last"), "last"]].forEach(([lab, v]) => {
                    const o = document.createElement("option");
                    o.value = v; o.textContent = lab;
                    if (v === c.target) o.selected = true;
                    tgtSel.appendChild(o);
                });
                tgtSel.addEventListener("change", () => {
                    composeData().target = tgtSel.value;
                    serialize(); drawCanvas();
                    if (tgtSel.value !== "none") loadComposeImg(tgtSel.value);
                });
                editRow.appendChild(tgtSel);
                panel.appendChild(editRow);

                // ── First / Last pickers + per-kind offset ──
                const imgs = window._mmh3teInputImages || [];
                for (const kind of ["first", "last"]) {
                    const ref = c[kind] || null;
                    const curFile = ref?.name || "";

                    const row = document.createElement("div");
                    row.className = "mmh3te-row";
                    row.appendChild(Object.assign(document.createElement("span"), {
                        className: "mmh3te-lbl",
                        textContent: kind === "first" ? uistr("First:") : uistr("Last:") }));

                    const picker = document.createElement("div");
                    picker.className = "mmh3te-ref-picker";
                    picker.style.flex = "1 1 auto";

                    const trigger = document.createElement("div");
                    trigger.className = "mmh3te-ref-trigger";
                    if (curFile) {
                        const tImg = document.createElement("img");
                        tImg.src = imageUrl(curFile);
                        trigger.appendChild(tImg);
                    }
                    trigger.appendChild(Object.assign(document.createElement("span"), {
                        className: "name", textContent: curFile || uistr("(no image)") }));
                    trigger.appendChild(Object.assign(document.createElement("span"), {
                        className: "arrow", textContent: "\u25BC" }));
                    picker.appendChild(trigger);

                    const dropdown = document.createElement("div");
                    dropdown.className = "mmh3te-ref-dropdown";
                    _composeDropdownEls.push(dropdown);
                    document.body.appendChild(dropdown);

                    const noneOpt = document.createElement("div");
                    noneOpt.className = "mmh3te-ref-opt" + (!curFile ? " selected" : "");
                    noneOpt.appendChild(Object.assign(document.createElement("span"), {
                        className: "name", textContent: uistr("(no image)") }));
                    noneOpt.addEventListener("pointerenter", () => cvOverlay.classList.remove("visible"));
                    noneOpt.addEventListener("click", () => {
                        composeData()[kind] = null;
                        dropdown.classList.remove("open");
                        cvOverlay.classList.remove("visible");
                        serialize(); refresh();
                    });
                    dropdown.appendChild(noneOpt);

                    for (const file of imgs) {
                        const opt = document.createElement("div");
                        opt.className = "mmh3te-ref-opt" + (file === curFile ? " selected" : "");
                        const oImg = document.createElement("img");
                        oImg.src = imageUrl(file);
                        opt.appendChild(oImg);
                        opt.appendChild(Object.assign(document.createElement("span"), {
                            className: "name", textContent: file }));
                        opt.addEventListener("pointerenter", () => {
                            cvOverlayImg.src = imageUrl(file);
                            cvOverlay.classList.add("visible");
                        });
                        opt.addEventListener("click", () => {
                            const wasOn = composeOnFlag();
                            const live = composeData();
                            live[kind] = { name: file, s: null, ox: 0, oy: 0 };
                            // Loading the first compose image turns compose on:
                            // drop manual per-tile refs so the split takes over
                            // (per-tile refs set later still override the crop).
                            if (!wasOn) {
                                for (const k in node._tiles) {
                                    if (node._tiles[k]) node._tiles[k].ref_images = [];
                                }
                                node._selectedIdx = -1;
                            }
                            if (live.target === "none") live.target = kind;
                            dropdown.classList.remove("open");
                            cvOverlay.classList.remove("visible");
                            serialize(); refresh();
                            loadComposeImg(kind);
                        });
                        dropdown.appendChild(opt);
                    }

                    trigger.addEventListener("click", (e) => {
                        e.stopPropagation();
                        for (const d of _composeDropdownEls) if (d !== dropdown) d.classList.remove("open");
                        const opening = !dropdown.classList.contains("open");
                        dropdown.classList.toggle("open");
                        if (opening) {
                            const r = trigger.getBoundingClientRect();
                            dropdown.style.left = Math.max(4, r.left) + "px";
                            dropdown.style.top = (r.bottom + 4) + "px";
                            dropdown.style.minWidth = Math.max(240, r.width) + "px";
                        }
                    });
                    trigger.addEventListener("pointerenter", () => cvOverlay.classList.remove("visible"));

                    row.appendChild(picker);
                    panel.appendChild(row);

                    // ── Offset X/Y: position of the tile-plan region
                    // relative to the image's TOP-LEFT corner, in tile-plan
                    // pixels (the image is scaled by s so the plan fits it;
                    // same units as tile width/height). Mirrors the viewport
                    // origin exactly: offset = ox / s, so typing a value and
                    // dragging the preview are the same operation and the
                    // crop always matches what the canvas shows. First and
                    // Last are independent.
                    const offRow = document.createElement("div");
                    offRow.className = "mmh3te-row";
                    offRow.style.margin = "0 0 4px 0";
                    offRow.appendChild(Object.assign(document.createElement("span"), {
                        className: "mmh3te-lbl", textContent: uistr("Offset:") }));
                    _composeOffInputs[kind] = {};
                    for (const ax of ["X", "Y"]) {
                        const key = ax === "X" ? "ox" : "oy";
                        const inp = document.createElement("input");
                        inp.type = "number";
                        inp.className = "mmh3te-inp";
                        inp.step = 8;
                        inp.value = (ref && ref.s) ? Math.round(ref[key] / ref.s) : 0;
                        inp.disabled = !ref;
                        inp.title = uistr("Position of the tile-plan region relative to the image's top-left corner, in tile-plan pixels (the image is scaled so the tile plan fits it -- same units as tile width/height). 0 puts the plan at the image's left/top edge; First and Last are independent. Dragging or wheel-zooming the preview updates these values. Values typed before the image finishes decoding are applied automatically once it loads.");
                        inp.addEventListener("change", () => {
                            // Re-read the LIVE ref: restore() can replace
                            // node.compose after this panel was built, and a
                            // stale closure would silently write to an orphan.
                            const r = composeData()[kind];
                            if (!r) return;
                            const v = parseFloat(inp.value) || 0;
                            if (r.s) {
                                r[key] = v * r.s;
                                if (r.iw) {
                                    clampViewport(r, r.iw, r.ih,
                                                  totalSize(tilesArr(), currentPlan()));
                                }
                            } else {
                                // Image not decoded yet: stash the intent and
                                // apply it in loadComposeImg after decode --
                                // never silently drop a typed value.
                                if (key === "ox") r.pendingOffX = v;
                                else r.pendingOffY = v;
                            }
                            // The input must never lie about the real state:
                            // re-derive the display from the actual viewport.
                            inp.value = r.s ? Math.round(r[key] / r.s) : v;
                            serialize(); drawCanvas();
                        });
                        _composeOffInputs[kind][ax] = inp;
                        offRow.appendChild(inp);
                    }
                    const offHint = document.createElement("span");
                    offHint.style.cssText = "font:10px monospace; color:#666;";
                    offHint.textContent = uistr("plan px");
                    offRow.appendChild(offHint);
                    panel.appendChild(offRow);
                }

                // Keep the image list fresh; rebuild only when it changed so an
                // open picker dropdown is not destroyed mid-interaction.
                (async () => {
                    const prev = JSON.stringify(window._mmh3teInputImages || []);
                    invalidateInputImages();
                    window._mmh3teInputImages = await fetchInputImages();
                    const now = JSON.stringify(window._mmh3teInputImages || []);
                    if (now !== prev && node._composeView) buildComposePanel();
                })();
            }

            // ── Serialize tile data to hidden widget ──
            // Pure-information node: no image tensors cross the boundary. The
            // whole-image compose crop boxes are computed HERE (front-end is
            // the single source of geometry, matching exactly what the canvas
            // previews) and shipped as plain numbers per tile; the consuming
            // Extend Video node performs the actual file crop at execution
            // time.
            function composeCropsFor(tile) {
                const c = composeData();
                const crops = [];
                if (!composeOnFlag()) return crops;
                // Only tiles opted into their compose crop ship boxes; tiles
                // with ref_source="own" use their own image list instead.
                if ((tile.ref_source || "crop") !== "crop") return crops;
                // Use the same pruned tile set as the canvas preview
                // (tilesArr = node._tiles[0..tileCount()-1]). Iterating
                // Object.keys(node._tiles) here could include stale keys left
                // over from a previous higher tile_count, so a tile's placement
                // (and thus its crop box) would be computed for a larger
                // arrangement and drift right of the previewed region.
                const arr = tilesArr();
                const placements = computePlacements(arr, currentPlan());
                const idx = arr.indexOf(tile);
                const pl = placements[idx];
                if (!pl) return crops;
                for (const kind of ["first", "last"]) {
                    const ref = c[kind];
                    if (!ref?.name) continue;
                    // Preferred: resolve the exact source-pixel box now. When
                    // s is still null (image decode not finished when the run
                    // was queued) we still ship the descriptor without a box;
                    // the Extend Video node then derives a fit+centred crop
                    // after opening the file, so a tile never silently gets no
                    // reference.
                    if (ref.s == null) {
                        crops.push({ kind, name: ref.name });
                        continue;
                    }
                    // Clamp the window POSITION (size preserved) so it
                    // never leaves the image: the Extend Video node crops
                    // exactly this box and the canvas preview clamps the same
                    // way, so the first frame matches the preview (no
                    // edge-clamp -> narrow -> stretch drift).
                    let L = ref.ox + pl.x * ref.s;
                    let T = ref.oy + pl.y * ref.s;
                    const cw = pl.w * ref.s, ch = pl.h * ref.s;
                    if (ref.iw) L = Math.max(0, Math.min(L, ref.iw - cw));
                    if (ref.ih) T = Math.max(0, Math.min(T, ref.ih - ch));
                    crops.push({
                        kind,
                        name: ref.name,
                        box: [
                            Math.max(0, Math.round(L)),
                            Math.max(0, Math.round(T)),
                            Math.max(0, Math.round(L + cw)),
                            Math.max(0, Math.round(T + ch)),
                        ],
                    });
                }
                return crops;
            }
            function serialize() {
                const data = {};
                // Only emit 0..tileCount()-1: stale keys > tile_count would be
                // ignored by the backend anyway, and re-emitting them could
                // expose a stale compose_crops box from an earlier arrangement.
                const _serTiles = tilesArr();
                for (let idx = 0; idx < _serTiles.length; idx++) {
                    const tile = _serTiles[idx];
                    const clone = { ...tile };
                    // Never leak stale crop boxes from older sessions/keys.
                    delete clone.fl2va_crops;
                    delete clone.compose_crops;
                    if (composeOnFlag() && (clone.ref_source || "crop") === "crop") {
                        clone.compose_crops = composeCropsFor(tile);
                    }
                    data[idx] = clone;
                }
                if (node.compose) data.compose = node.compose;
                data.plan = currentPlan();
                data.tile_count = tileCount();
                // Ground-truth trace: exactly the boxes the Extend Video node
                // will crop, so preview/output mismatches can be diagnosed
                // from the browser console alone.
                try {
                    const boxes = [];
                    for (const t of Object.values(data)) {
                        if (t && t.compose_crops && t.compose_crops.length) {
                            boxes.push(...t.compose_crops);
                        }
                    }
                    if (boxes.length) {
                        console.log("[MMH3SpatialTileEditor] compose boxes:",
                                    JSON.stringify(boxes));
                    }
                } catch (e) { /* logging must never break serialization */ }
                dataWidget.value = JSON.stringify(data);
            }

            // ── Full refresh ──
            function refresh() {
                ensureTiles();
                rebuildToolbar();
                drawCanvas();
                renderPanel();
            }

            // ── Close button ──
            const closeBtn = document.createElement("button");
            closeBtn.className = "mmh3te-btn";
            closeBtn.textContent = "\u2715";
            closeBtn.title = uistr("Close editor");
            head.appendChild(closeBtn);
            closeBtn.addEventListener("click", () => {
                dock.style.display = "none";
                node._dockVisible = false;
                setWidgetVal("show_editor", false);
            });

            // ── Minimize / Pin ──
            minBtn.addEventListener("click", () => {
                node._dockVisible = !node._dockVisible;
                dock.classList.toggle("minimized", !node._dockVisible);
                if (node._dockVisible) {
                    dock.style.display = "";
                }
                minBtn.textContent = node._dockVisible ? "\u2212" : "+";
            });

            // ── Drag head to move dock ──
            head.addEventListener("pointerdown", (e) => {
                if (e.target === minBtn || e.target === closeBtn || e.button !== 0) return;
                if (node.properties.mmh3tePinned !== false) return;
                e.preventDefault();
                const sx0 = e.clientX, sy0 = e.clientY;
                const startLeft = dock.offsetLeft, startTop = dock.offsetTop;
                document.body.classList.add("mmh3te-dragging");
                const onMove = (me) => {
                    const newLeft = Math.max(0, Math.min(window.innerWidth - 60, startLeft + me.clientX - sx0));
                    const newTop = Math.max(0, Math.min(window.innerHeight - 30, startTop + me.clientY - sy0));
                    dock.style.left = newLeft + "px";
                    dock.style.top = newTop + "px";
                };
                const onEnd = () => {
                    document.removeEventListener("pointermove", onMove);
                    document.removeEventListener("pointerup", onEnd);
                    document.body.classList.remove("mmh3te-dragging");
                    node.properties.mmh3teFloatRect = {
                        x: dock.offsetLeft, y: dock.offsetTop,
                        w: dock.offsetWidth, h: dock.offsetHeight,
                    };
                };
                document.addEventListener("pointermove", onMove);
                document.addEventListener("pointerup", onEnd);
            });

            // ── Resize handles ──
            dock.querySelectorAll(".mmh3te-rsz").forEach(h => {
                h.addEventListener("pointerdown", (e) => {
                    e.preventDefault();
                    const dir = h.dataset.dir;
                    const sx0 = e.clientX, sy0 = e.clientY;
                    const startRect = { x: dock.offsetLeft, y: dock.offsetTop,
                                        w: dock.offsetWidth, h: dock.offsetHeight };
                    const onMove = (me) => {
                        const dx = me.clientX - sx0, dy = me.clientY - sy0;
                        let { x, y, w, h: hh } = startRect;
                        if (dir.includes("e")) w = Math.max(400, w + dx);
                        if (dir.includes("w")) { w = Math.max(400, w - dx); x = x + (startRect.w - w); }
                        if (dir.includes("s")) hh = Math.max(300, hh + dy);
                        if (dir.includes("n")) { hh = Math.max(300, hh - dy); y = y + (startRect.h - hh); }
                        dock.style.left = x + "px"; dock.style.top = y + "px";
                        dock.style.width = w + "px"; dock.style.height = hh + "px";
                    };
                    const onEnd = () => {
                        document.removeEventListener("pointermove", onMove);
                        document.removeEventListener("pointerup", onEnd);
                        node.properties.mmh3teFloatRect = {
                            x: dock.offsetLeft, y: dock.offsetTop,
                            w: dock.offsetWidth, h: dock.offsetHeight,
                        };
                        drawCanvas();
                    };
                    document.addEventListener("pointermove", onMove);
                    document.addEventListener("pointerup", onEnd);
                });
            });

            // ── Restore from saved state ──
            function restore() {
                const data = parseTileData(dataWidget);
                // Plan / tile count are JS-managed; migrate older saves that
                // only carry per-tile entries (derive count from max index).
                node._plan = (typeof data.plan === "string" && PLANS.includes(data.plan))
                    ? data.plan : "horizontal_alternating";
                if (typeof data.tile_count === "number" && data.tile_count >= 1) {
                    node._tileCount = Math.max(1, Math.min(12, Math.round(data.tile_count)));
                } else {
                    const idxs = Object.keys(data)
                        .map(k => parseInt(k)).filter(k => Number.isInteger(k) && k >= 0);
                    node._tileCount = idxs.length ? Math.max(...idxs) + 1 : 2;
                }
                if (node._plan === "4_quadrants_expand") node._tileCount = 5;
                node._tiles = {};
                for (const [k, v] of Object.entries(data)) {
                    if (k === "compose" || k === "fl2va_compose") continue;
                    if (typeof v === "object" && v !== null && !Array.isArray(v)) {
                        node._tiles[parseInt(k)] = v;
                    }
                }
                // Compose state is serialized at top level of tile_data
                // (accept the historical "fl2va_compose" key from older saves).
                const cd = (data && typeof data === "object")
                    ? (data.compose || data.fl2va_compose) : null;
                node.compose = cd || { first:null, last:null, target:"none" };
                // Legacy saves carry an "enabled" flag; compose is now derived
                // from the loaded First/Last images, so drop the stale field.
                if (node.compose && "enabled" in node.compose) delete node.compose.enabled;
                // Legacy per-kind offX/offY (old "delta on top of the viewport"
                // semantics, in source pixels) fold into ox/oy: offsets are now
                // the absolute plan-space position ox/s relative to the image
                // top-left.
                for (const k of ["first", "last"]) {
                    const r = node.compose[k];
                    if (r) {
                        if (r.offX) r.ox = (r.ox || 0) + r.offX;
                        if (r.offY) r.oy = (r.oy || 0) + r.offY;
                        delete r.offX; delete r.offY;
                    }
                }
                ensureTiles();
                node._selectedIdx = -1;
                refresh();
                if (composeOnFlag()) {
                    loadComposeImg("first");
                    if (node.compose.last?.name) loadComposeImg("last");
                }
            }

            // ── Wire up widget callbacks ──
            for (const wName of ["base_prompt",
                                  "overlap_mode", "overlap_blend"]) {
                const w = findW(wName);
                if (w) {
                    const orig = w.callback;
                    w.callback = function (...args) {
                        const r = orig?.apply(this, args);
                        refresh();
                        return r;
                    };
                }
            }

            // Apply on configure (workflow load) and initial setup
            const origConfigure2 = node.onConfigure;
            node.onConfigure = function (...args) {
                const r = origConfigure2 ? origConfigure2.apply(this, args) : undefined;
                setTimeout(() => {
                    applyShowEditor();
                }, 0);
                return r;
            };
            setTimeout(() => {
                applyShowEditor();
            }, 0);

            // ── show_editor toggle: open/close dock ──
            function applyShowEditor() {
                const w = findW("show_editor");
                if (!w) return;
                if (w.value) {
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

            // ── Place dock and initialize ──
            dock.style.width = "740px";
            dock.style.height = "720px";
            document.body.appendChild(dock);

            // Initialize zoom
            if (node.properties.mmh3teZoom === undefined) node.properties.mmh3teZoom = 1.0;
            setZoom(getZoom());

            if (node.properties.mmh3tePinned === undefined) {
                node.properties.mmh3tePinned = false;
            }
            if (node.properties.mmh3tePinned !== false) {
                pinnedDocks.add(node);
            }

            const origConfigure = node.onConfigure;
            node.onConfigure = function (...args) {
                const r = origConfigure?.apply(this, args);
                restore();
                if (node.properties.mmh3tePinned !== false) {
                    pinnedDocks.add(node);
                }
                applyDockTransform(node);
                wakeDocks();
                return r;
            };

            // ── Cleanup on node deletion ──
            const origOnRemoved = node.onRemoved;
            node.onRemoved = function (...args) {
                pinnedDocks.delete(node);
                dock.remove();
                node._mmh3teDock = null;
                return origOnRemoved?.apply(this, args);
            };

            restore();

            // Position dock on first creation
            requestAnimationFrame(() => {
                if (node.properties.mmh3tePinned !== false) {
                    applyDockTransform(node);
                    wakeDocks();
                } else {
                    const dockW = dock.offsetWidth || 740;
                    const dockH = dock.offsetHeight || 720;
                    node.properties.mmh3teFloatRect = {
                        x: Math.round((window.innerWidth - dockW) / 2),
                        y: Math.round((window.innerHeight - dockH) / 2),
                        w: dockW, h: dockH,
                    };
                    const r = node.properties.mmh3teFloatRect;
                    dock.style.left = r.x + "px";
                    dock.style.top = r.y + "px";
                    dock.style.width = r.w + "px";
                    dock.style.height = r.h + "px";
                }
                dock.classList.toggle("pinned", node.properties.mmh3tePinned !== false);
            });
        };
    },
});
