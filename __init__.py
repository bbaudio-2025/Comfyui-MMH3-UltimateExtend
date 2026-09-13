"""Comfyui-MMH3-UltimateExtend - MMH3 Spatial Extend Video with Tile Editor.

Carries the tile-based H3 video extension nodes (Extend Video, Tile Editor)
out of the Ultimate Upscale plugin so the extending workflow can be developed
independently of the upscaler.
"""
import json
import os

from aiohttp import web

from comfy_api.latest import ComfyExtension
from typing_extensions import override

from .nodes import (
    MMH3SpatialExtendVideo,
    MMH3SpatialTileEditor,
    # MMH3MaskPreview,  # TEMPORARILY unregistered (debug helper, code kept)
    MMH3LastQuadrantPatch,
    MMH3LatentPreview,
    MMH3TemporalExtendVideo,
    MMH3TemporalTileEditor,
    MMH3SampleParams,
    MMH3TemporalOverlapParams,
    MMH3TemporalOverlapSimple,
)

NODE_CLASS_MAPPINGS = {
    "MMH3SpatialExtendVideo": MMH3SpatialExtendVideo,
    "MMH3SpatialTileEditor": MMH3SpatialTileEditor,
    # "MMH3MaskPreview": MMH3MaskPreview,
    "MMH3LastQuadrantPatch": MMH3LastQuadrantPatch,
    "MMH3LatentPreview": MMH3LatentPreview,
    "MMH3TemporalExtendVideo": MMH3TemporalExtendVideo,
    "MMH3TemporalTileEditor": MMH3TemporalTileEditor,
    "MMH3SampleParams": MMH3SampleParams,
    "MMH3TemporalOverlapParams": MMH3TemporalOverlapParams,
    "MMH3TemporalOverlapSimple": MMH3TemporalOverlapSimple,
}

# front-end JS: Tile Editor dock panel
WEB_DIRECTORY = "./web"

NODE_DISPLAY_NAME_MAPPINGS = {
    "MMH3SpatialExtendVideo": "MMH3 Spatial Extend Video",
    "MMH3SpatialTileEditor": "MMH3 Spatial Tile Editor",
    # "MMH3MaskPreview": "MMH3 Mask Preview",
    "MMH3LastQuadrantPatch": "MMH3 Last Quadrant Patch",
    "MMH3LatentPreview": "MMH3 Latent Preview (approx)",
    "MMH3TemporalExtendVideo": "MMH3 Temporal Extend Video",
    "MMH3TemporalTileEditor": "MMH3 Temporal Tile Editor",
    "MMH3SampleParams": "MMH3 Sample Params",
    "MMH3TemporalOverlapParams": "MMH3 Temporal Overlap Params",
    "MMH3TemporalOverlapSimple": "MMH3 Temporal Overlap Simple",
}


# ── API route: list input images ──
def _register_api_routes():
    try:
        from server import PromptServer
        from folder_paths import get_input_directory

        if not hasattr(PromptServer, "instance") or PromptServer.instance is None:
            return
        routes = PromptServer.instance.routes

        @routes.get("/mmh3te/input_files")
        async def list_input_files(request):
            input_dir = get_input_directory()
            IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
            VIDEO_EXTS = {".mp4", ".webm", ".mov", ".avi", ".mkv", ".m4v"}
            AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".wma"}
            images, videos, audios = [], [], []
            if os.path.isdir(input_dir):
                for f in os.listdir(input_dir):
                    if os.path.isfile(os.path.join(input_dir, f)):
                        _, ext = os.path.splitext(f)
                        ext = ext.lower()
                        if ext in IMAGE_EXTS:
                            images.append(f)
                        elif ext in VIDEO_EXTS:
                            videos.append(f)
                        elif ext in AUDIO_EXTS:
                            audios.append(f)
            for lst in (images, videos, audios):
                lst.sort()
            if request.query.get("categorized"):
                return web.json_response({"images": images, "videos": videos,
                                          "audios": audios})
            return web.json_response(images)

        # ── API route: temporal session files (previews + metadata) ──
        def _temporal_session_dir(location, session, create=False):
            """Resolve a session directory through the node module, so the
            route and the executor can never disagree about the layout
            (<mmh3_temporal>/<session>/{latents,previews} + the shared
            <mmh3_temporal>/cache). `create=False` because a GET must not
            conjure a session into existence."""
            from .nodes.temporal_extend import _session_dir
            return _session_dir(location, session, create=create)

        def _session_files(sdir):
            """Every file of a session as a SESSION-RELATIVE name with '/'
            separators ('latents/merged_3.h3latent',
            'previews/merged_preview_1.webp'). This is exactly the form
            session.json stores and the Tile Editor compares against, so a
            listing and a ledger entry are directly comparable."""
            out = []
            for root, dirs, names in os.walk(sdir):
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                for f in names:
                    if f.startswith("."):
                        continue
                    rel = os.path.relpath(os.path.join(root, f), sdir)
                    out.append(rel.replace("\\", "/"))
            return sorted(out)

        @routes.get("/mmh3te/temporal_session")
        async def temporal_session(request):
            """List a session directory: session.json ledger + metadata.json
            (legacy) + preview/latent files (recursively, so the latents/ and
            previews/ subdirectories are visible to the UI)."""
            q = request.query
            sdir = _temporal_session_dir(q.get("location", "temp"),
                                         q.get("session", ""))
            if not os.path.isdir(sdir):
                return web.json_response({"exists": False, "files": [],
                                          "session": None, "metadata": None})
            sess = meta = None
            for key, fname in (("session", "session.json"),
                               ("metadata", "metadata.json")):
                p = os.path.join(sdir, fname)
                if os.path.isfile(p):
                    try:
                        with open(p, "r", encoding="utf-8") as fh:
                            val = json.load(fh)
                        if isinstance(val, dict):
                            if key == "metadata" and val.get("version") == 2:
                                val = None  # stray v2 ledger, not legacy metadata
                    except Exception:
                        val = None
                    if val is not None:
                        if key == "session":
                            sess = val
                        else:
                            meta = val
            return web.json_response({"exists": True, "files": _session_files(sdir),
                                      "session": sess, "metadata": meta},
                                     headers={"Cache-Control": "no-store"})

        @routes.post("/mmh3te/temporal_session_state")
        async def temporal_session_state(request):
            """Persist the Tile Editor's editor state - the full segments
            snapshot + extend_params + resume point - into session.json under
            the 'editor' key. This is what lets a FRESH node (workflow lost,
            node reset, restart) rebuild the whole chain from disk alone.

            The editor key lives in the same ledger that the executor, prune
            and cleanup load-modify-save, so their writes preserve it; the
            editor never touches the segments/attempt mapping the executor
            owns."""
            from .nodes.temporal_extend import (_load_session_ledger,
                                                _save_session_ledger)
            try:
                body = await request.json()
            except Exception:
                return web.json_response({"ok": False, "reason": "bad json"},
                                         status=400)
            location = body.get("location", "temp")
            session = body.get("session", "")
            state = body.get("state")
            if not isinstance(state, dict):
                return web.json_response({"ok": False,
                                          "reason": "missing state"},
                                         status=400)
            sdir = _temporal_session_dir(location, session, create=True)
            ledger = _load_session_ledger(sdir)
            ledger["editor"] = state
            _save_session_ledger(sdir, ledger)
            return web.json_response({"ok": True, "saved": True},
                                     headers={"Cache-Control": "no-store"})

        @routes.get("/mmh3te/temporal_file")
        async def temporal_file(request):
            """Serve one file (preview webp/png) from a session directory."""
            from .nodes.temporal_extend import _find_session_file
            q = request.query
            sdir = _temporal_session_dir(q.get("location", "temp"),
                                         q.get("session", ""))
            name = q.get("name", "")
            safe = os.path.normpath(name)
            if (not safe or safe.startswith((".", "/", "\\")) or ".." in safe
                    or os.path.isabs(safe)):
                return web.Response(status=400, text="bad name")
            # a subdirectory-qualified name first; a bare name may still be a
            # flat pre-subdirectory file, which the resolver finds anywhere
            path = _find_session_file(sdir, safe.replace("\\", "/"))
            if not path or not os.path.isfile(path):
                return web.Response(status=404, text="not found")
            ext = os.path.splitext(path)[1].lower()
            mime = {".webp": "image/webp", ".png": "image/png"}.get(ext)
            if mime is None:
                return web.Response(status=400, text="not previewable")
            with open(path, "rb") as fh:
                body = fh.read()
            # no-store: attempt files are overwritten names-free per seq, but
            # a rerun of the SAME seg swaps which file the UI maps to - never
            # let the browser serve a cached preview of an older attempt
            return web.Response(body=body, content_type=mime,
                                headers={"Cache-Control": "no-store"})

        @routes.get("/mmh3te/cache_status")
        async def cache_status_route(request):
            """Size/entry count of the SHARED reference + conditioning cache
            (<mmh3_temporal>/cache), which every session of a storage root
            reads from and writes to."""
            from .nodes.temporal_extend import cache_status
            location = request.query.get("location", "temp")
            st = cache_status(location)
            st["exists"] = bool(st["total_bytes"]) or any(
                st[k]["entries"] for k in ("refs", "cond"))
            return web.json_response(st, headers={"Cache-Control": "no-store"})

        @routes.get("/mmh3te/cache_clear")
        async def cache_clear_route(request):
            """Drop cached reference latents and/or conditioning. `kind` is
            'refs' / 'cond' / omitted for both. Files that cannot be removed
            yet (still mapped by a conditioning this process is holding) are
            counted as 'failed' instead of being silently ignored, so the
            caller can say 'N freed, M still in use'."""
            from .nodes.temporal_extend import cache_clear, cache_status
            q = request.query
            location = q.get("location", "temp")
            kind = q.get("kind") or None
            if kind not in (None, "refs", "cond"):
                return web.json_response({"ok": False, "reason": "bad kind"})
            res = cache_clear(location, kind)
            return web.json_response({"ok": True, "kind": kind or "all",
                                      "status": cache_status(location), **res},
                                     headers={"Cache-Control": "no-store"})

        @routes.get("/mmh3te/temporal_prune")
        async def temporal_prune(request):
            """Invalidate stored results from a segment index onward. Called
            when the user deletes segment i in the Tile Editor: its own and
            ALL later segments' stored results become stale (the chain
            shifted). Updates session.json - drops the segments-mapping
            entries >= 'from' and clamps last_segment - while KEEPING every
            attempt file on disk (history policy)."""
            from .nodes.temporal_extend import (_load_session_ledger,
                                                _save_session_ledger)
            q = request.query
            sdir = _temporal_session_dir(q.get("location", "temp"),
                                         q.get("session", ""))
            try:
                frm = max(0, int(q.get("from", "0")))
            except (TypeError, ValueError):
                frm = 0
            if not os.path.isdir(sdir):
                return web.json_response({"ok": False,
                                          "reason": "no session directory"})
            ledger = _load_session_ledger(sdir)
            kept = [e for e in ledger.get("segments", [])
                    if isinstance(e, dict) and int(e.get("index", -1)) < frm]
            ledger["segments"] = kept
            ledger["last_segment"] = min(
                int(ledger.get("last_segment", -1)), frm - 1)
            _save_session_ledger(sdir, ledger)
            return web.json_response({"ok": True, "pruned_from": frm,
                                      "last_segment": ledger["last_segment"],
                                      "session": ledger})

        @routes.get("/mmh3te/temporal_cleanup")
        async def temporal_cleanup(request):
            """Delete stored latent/preview attempt FILES that the current
            chain no longer references (every old seq-named attempt whose
            entry is not the one mapped by session.json). Two phases: without
            `commit=1` it only reports how many files WOULD be removed (so the
            UI can confirm first); with it, they are deleted. session.json and
            metadata.json are never touched, and the shared cache/ is out of
            scope (it has its own clear endpoint)."""
            from .nodes.temporal_extend import (_load_session_ledger,
                                                LATENTS_SUBDIR,
                                                PREVIEWS_SUBDIR)
            q = request.query
            commit = q.get("commit", "0") == "1"
            sdir = _temporal_session_dir(q.get("location", "temp"),
                                         q.get("session", ""))
            if not os.path.isdir(sdir):
                return web.json_response({"ok": True, "commit": commit,
                                          "removed": 0, "failed": 0,
                                          "deleted": []})
            ledger = _load_session_ledger(sdir)
            keep = set()
            for seg in ledger.get("segments", []):
                files = (seg or {}).get("files") or {}
                for rel in files.values():
                    if isinstance(rel, str):
                        keep.add(rel.replace("\\", "/"))
            DATA_EXT = {".h3latent", ".latent", ".safetensors", ".webp", ".png"}
            rows = []
            for sub in (LATENTS_SUBDIR, PREVIEWS_SUBDIR):
                d = os.path.join(sdir, sub)
                if not os.path.isdir(d):
                    continue
                for f in sorted(os.listdir(d)):
                    if f.startswith("."):
                        continue
                    if os.path.splitext(f)[1].lower() in DATA_EXT:
                        rows.append((os.path.join(d, f), f"{sub}/{f}"))
            # flat pre-subdirectory-era files still sitting at the session root
            if os.path.isdir(sdir):
                for f in sorted(os.listdir(sdir)):
                    if f.startswith(".") or f in ("session.json", "metadata.json"):
                        continue
                    p = os.path.join(sdir, f)
                    if (os.path.isfile(p)
                            and os.path.splitext(f)[1].lower() in DATA_EXT):
                        rows.append((p, f))
            stale = [(abs, rel) for abs, rel in rows if rel not in keep]
            if not commit:
                return web.json_response({"ok": True, "commit": False,
                                          "removed": len(stale)}, 
                                         headers={"Cache-Control": "no-store"})
            removed, failed, deleted = 0, 0, []
            for abs, rel in stale:
                try:
                    os.remove(abs)
                    removed += 1
                    deleted.append(rel)
                except Exception:
                    failed += 1
            return web.json_response({"ok": True, "commit": True,
                                      "removed": removed, "failed": failed,
                                      "deleted": deleted},
                                     headers={"Cache-Control": "no-store"})

    except Exception:
        pass


_register_api_routes()


class MMH3SpatialExtendVideoExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type]:
        return [
            MMH3SpatialExtendVideo,
            MMH3SpatialTileEditor,
            # MMH3MaskPreview,
            MMH3LastQuadrantPatch,
            MMH3LatentPreview,
            MMH3TemporalExtendVideo,
            MMH3TemporalTileEditor,
            MMH3SampleParams,
        ]


async def comfy_entrypoint() -> MMH3SpatialExtendVideoExtension:
    return MMH3SpatialExtendVideoExtension()
