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
    "ref audio": "参考音频",
    "frame #": "帧序号",
    "-1 = last frame of previous merged video (latent-exact, no decode)":
        "-1 = 上一合并视频的最后一帧（latent 精确，不解码）",
    "+ add segment": "+ 添加分段",
    "segment(s)": "个分段",
    "total ≈": "总时长 ≈",
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

function snap17n5(n) { n = Math.max(5, Math.round(n) || 5); return n + ((5 - (n % 17)) % 17); }
function snap17(n) { n = Math.max(17, Math.round(n) || 17); return n + ((17 - (n % 17)) % 17); }
function fmtSec(frames) { return (Math.round(frames) / FPS).toFixed(2); }

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
    function ordBadge(n) {
        return ordinals
            ? el("span", { class: "mmh3tte-ord", title: `<Picture ${n}>` }, "P" + n)
            : null;
    }

    // one "dummy" chip per wired socket: the socket's own label (e.g.
    // 'First_or_Ref_Image_0') replaces the thumbnail, the ✕ rules that image
    // out FOR THIS SEGMENT (the wire itself is untouched)
    function socketChip(s, onRemove, ord) {
        const label = s.label || s.name;
        const chip = el("span", { class: "mmh3tte-thumb mmh3tte-sock",
                                  title: ord ? `${label}  \u2192  <Picture ${ord}>` : label },
            ordBadge(ord),
            el("span", { class: "mmh3tte-sock-ph" }, "\u25A3"),
            el("span", { class: "mmh3tte-sock-name" }, label));
        if (onRemove) {
            const x = el("span", { class: "mmh3tte-thumb-x",
                                   title: uistr("remove image") }, "\u2715");
            x.addEventListener("click", (e) => { e.stopPropagation(); onRemove(); });
            chip.appendChild(x);
        }
        return chip;
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
                const th = el("img", { src: imageUrl(f), title: f });
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
                thumbs.appendChild(el("span", { class: "mmh3tte-thumb" },
                    ordBadge(socks.length + j + 1), th, x));
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
        .mmh3tte-thumb { position:relative; display:inline-flex; }
        .mmh3tte-thumb-x { position:absolute; top:-5px; right:-5px; width:14px; height:14px;
            border-radius:50%; background:#c0392b; color:#fff; font:10px/14px sans-serif;
            text-align:center; cursor:pointer; box-shadow:0 0 2px #000; user-select:none; }
        .mmh3tte-thumb-x:hover { background:#e74c3c; }
        /* ordinal badge: which <Picture N> this image becomes in the prompt.
           Wired sockets are numbered first, then the picked files, because
           that is the order the Extend node packs them in. Pinned to the
           top-left corner (the ✕ owns the top-right) and click-through so it
           never eats the thumbnail's hover/remove. */
        .mmh3tte-ord { position:absolute; top:-5px; left:-5px; height:14px; min-width:14px;
            padding:0 3px; box-sizing:border-box; border-radius:3px;
            background:#2b6ca3; color:#fff; font:9px/14px sans-serif;
            text-align:center; white-space:nowrap; pointer-events:none;
            box-shadow:0 0 2px #000; user-select:none; }
        .mmh3tte-thumbrow { display:flex; gap:4px; flex-wrap:wrap; align-items:center; min-width:0; flex:0 1 auto; }
        .mmh3tte-thumbrow img { width:32px; height:32px; object-fit:cover; border-radius:3px; }
        /* wired reference-image sockets: a "dummy" chip standing in for an
           image that arrives through one of the node's reference sockets
           instead of the input folder. Shows the socket label; the ✕ rules
           that image out for THIS segment only (the wire itself is
           untouched). */
        .mmh3tte-sock { align-items:center; gap:3px; height:32px; padding:0 5px;
            box-sizing:border-box; border:1px dashed #4e6b7a; border-radius:3px;
            background:#1b262b; color:#8ab4e8; font:10px/1 sans-serif;
            max-width:132px; }
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
        ref_audio: "",
        // standalone audio reference: 'none' | 'load' (picked ref_audio
        // file) | 'prev' (previous segment's soundtrack, latent-level) |
        // 'initial' (first segment's soundtrack, latent-level)
        ref_audio_mode: "none",
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
    // Queue-time guard: if the resume point of an editor goes BEYOND the
    // segment count there is nothing to sample. (resume == segment count is
    // fine: the executor outputs the stored merged latent as-is.)
    // Block the queue HERE with a toast instead of failing deep inside
    // execution after all models have already been loaded.
    setup() {
        const origQueue = app.queuePrompt?.bind(app);
        if (!origQueue) return;
        app.queuePrompt = async function (...args) {
            const bad = [];
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
                } catch {}
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

            // ── data sync (unchanged serialization format) ──
            function sync() {
                dataWidget.value = JSON.stringify({
                    segments: node._segments,
                    extend_params: node._extendParams,
                    resume_from_segment: node._resumeFrom,
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
                return seg;
            }
            function initialFromWidget() {
                try {
                    const d = JSON.parse(dataWidget.value || "{}");
                    if (Array.isArray(d.segments) && d.segments.length) {
                        node._segments = d.segments.map(normSeg);
                        if (!node._segments[0].width) node._segments[0].width = 768;
                    }
                    if (d.extend_params && typeof d.extend_params === "object")
                        node._extendParams = { ...node._extendParams, ...d.extend_params };
                    if (Number.isInteger(d.resume_from_segment) && d.resume_from_segment > 0)
                        node._resumeFrom = d.resume_from_segment;
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
                    // where this segment lands on the song: the segments before
                    // it own everything earlier on the chain's timeline, which is
                    // the same sum the Extend node replays in frame arithmetic
                    let cum = 0;
                    for (let k = 0; k < i; k++)
                        cum += (node._segments[k].new_frames || 0);
                    rows.push(row(lbl(uistr("bgm range")),
                        el("span", { class: "mmh3tte-hint" },
                            `${fmtSec(cum)}s – ${fmtSec(cum + (seg.new_frames || 0))}s`)));
                }
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
                const card = el("div", { class: "mmh3tte-seg" + (i === node._openSeg ? " open" : "") +
                    (locked ? " locked-seg" : "") });

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
                const durChip = el("span", { class: "seg-dur" }, `${fmtSec(seg.new_frames)}s · ${seg.new_frames}f`);

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
                    (!locked && node._resumeFrom > 0 && i === node._resumeFrom)
                        ? el("span", { class: "mmh3tte-badge", title: uistr("resume here") }, "⟳") : null,
                    el("span", { class: "spacer" }),
                    lockBtn,
                    (i > 0 && !locked) ? el("button", { class: "mmh3tte-btn warn", title: uistr("del"), onclick: () => {
                        // deleting segment i invalidates its own and ALL
                        // later segments' stored results (the chain shifted)
                        // - confirm first, then prune the session ledger
                        const later = node._segments.length - i - 1;
                        const msg = uistr("delete segment") + ` ${i}` +
                            (later > 0 ? ` + ${later} ` + uistr("later segment(s)") : "") +
                            "?\n" + uistr("stored results become invalid");
                        confirmDialog(msg, uistr("confirm delete"), async () => {
                            node._segments.splice(i, 1);
                            if (node._resumeFrom > node._segments.length - 1) node._resumeFrom = 0;
                            sync(); renderAll();
                            await pruneStored(i);
                        });
                    } }, "✕") : null);

                const body = el("div", { class: "seg-body" });

                // duration: ONE input + sec/frames unit toggle. No two-way
                // binding between two inputs (that coupling caused the
                // "value bounces back" bugs). Rules:
                //  - frames mode: the input IS the authority; spinner steps 17
                //    (valid counts sit 17 apart on both grids) and typed values
                //    snap upward — rewriting here is safe, result always on-grid
                //  - seconds mode: the typed integer is kept VERBATIM in the
                //    input; the snapped ACTUAL result is only shown in the hint
                //    + head badge. Rewriting 3s -> 3.54s back into an integer
                //    input made it display 4 again — never do that.
                const unit = () => (node._durUnit === "frames") ? "frames" : "time";
                const snapF = (f) => first ? snap17n5(f) : snap17(f);
                const minF = first ? 5 : 17;
                const durInp = el("input", { type: "number", class: "mmh3tte-inp" });
                const durHint = el("span", { class: "mmh3tte-hint" });
                // refresh step/min/hint/badge. NEVER writes durInp.value —
                // after the initial render no code path touches the input, so
                // nothing can ever bounce the user's typed/clicked value back.
                function refreshDur() {
                    if (unit() === "frames") {
                        durInp.step = "17"; durInp.min = String(minF);
                    } else {
                        durInp.step = "1"; durInp.min = "1";
                    }
                    durHint.textContent = `= ${fmtSec(seg.new_frames)}s`;
                    durChip.textContent = `${fmtSec(seg.new_frames)}s · ${seg.new_frames}f`;
                }
                function applyDur(raw) {
                    const f = parseInt(raw);
                    if (unit() === "time") {
                        // seconds -> frames, snapped upward to the grid
                        if (Number.isFinite(f) && f >= 1) seg.new_frames = snapF(Math.round(f * FPS));
                    } else {
                        // frames mode: snapped to the grid data-level only
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
                    : Math.max(1, Math.round(seg.new_frames / FPS));
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
                        body.appendChild(fileSlot("ref video", seg.ref_video, _media.videos,
                            v => { seg.ref_video = v; }));
                        body.appendChild(fileSlot("video audio", seg.ref_video_audio, _media.audios,
                            v => { seg.ref_video_audio = v; }));
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
                        body.appendChild(fileSlot("ref video", seg.ref_video, _media.videos,
                            v => { seg.ref_video = v; }));
                        body.appendChild(fileSlot("video audio", seg.ref_video_audio, _media.audios,
                            v => { seg.ref_video_audio = v; }));
                        for (const r of audioModeRows(seg, first, i)) body.appendChild(r);
                    }
                }

                // ── per-segment overlap (segments >= 1) ──
                // The ONLY per-segment continuation parameter: how many tail
                // frames this segment carries over. All other tail/fade/
                // anchor/overlap settings are global via the "MMH3 Temporal
                // Overlap Params" sub-node; segment 0 generates fresh.
                if (!first) {
                    if (seg.overlap_frames == null) seg.overlap_frames = 39;
                    body.appendChild(el("div", { class: "mmh3tte-sub" },
                        uistr("overlap (this segment)")));
                    const ofInp = makeNum(seg.overlap_frames, { step: "17", min: "5" }, () => {
                        seg.overlap_frames = snap17n5(parseInt(ofInp.value) || 39); sync();
                    });
                    body.appendChild(row(lbl(uistr("overlap frames")), ofInp,
                        el("span", { class: "mmh3tte-hint" }, uistr("carried tail (17n+5)"))));
                }

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
            async function renderPreviews() {
                if (!pvBox) return;
                const session = findW("session_name")?.value || "session1";
                const location = findW("storage_location")?.value || "temp";
                if (node._sessLbl) node._sessLbl.textContent = session;
                pvBox.innerHTML = "";
                pvBox.appendChild(el("div", { class: "mmh3tte-muted" }, uistr("loading previews…")));
                let data;
                try {
                    const resp = await fetch(`/mmh3te/temporal_session?location=${encodeURIComponent(location)}&session=${encodeURIComponent(session)}`);
                    data = await resp.json();
                } catch { data = { exists: false, files: [], metadata: null }; }
                pvBox.innerHTML = "";
                if (!data.exists) {
                    // session dir gone entirely (e.g. temp wiped on restart)
                    autoUnlockStaleChain();
                    pvBox.appendChild(el("div", { class: "mmh3tte-muted" },
                        uistr("no stored run yet for this session — previews appear after executing MMH3 Temporal Extend Video")));
                    renderCacheRow(location);
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
                // a wire that just appeared flips segments to 'load images'
                // before the cards are rebuilt, so the reference row shows up
                applySocketDefaults();
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
                segList.appendChild(row(
                    el("span", { class: "mmh3tte-muted" },
                        `${node._segments.length} ${uistr("segment(s)")} · ${uistr("total ≈")} ${fmtSec(node._segments.reduce((a, s) => a + s.new_frames, 0))}s`)));
                if (node._resumeFrom > 0) {
                    segList.appendChild(row(el("span", { class: "mmh3tte-badge" },
                        `🔒 ${uistr("will reuse segments")} 0..${node._resumeFrom - 1} ${uistr("from disk")} — ${uistr("resume from segment")} ${node._resumeFrom}`)));
                }
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
                        if (editor.extend_params
                                && typeof editor.extend_params === "object") {
                            node._extendParams = { ...node._extendParams,
                                                   ...editor.extend_params };
                        }
                        node._resumeFrom =
                            (Number.isInteger(editor.resume_from_segment)
                             && editor.resume_from_segment > 0)
                                ? editor.resume_from_segment : 0;
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
