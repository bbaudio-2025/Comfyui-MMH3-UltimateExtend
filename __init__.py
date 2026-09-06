"""Comfyui-MMH3-UltimateExtend - MMH3 Spatial Extend Video with Tile Editor.

Carries the tile-based H3 video extension nodes (Extend Video, Tile Editor)
out of the Ultimate Upscale plugin so the extending workflow can be developed
independently of the upscaler.
"""
import os

from aiohttp import web

from comfy_api.latest import ComfyExtension
from typing_extensions import override

from .nodes import (
    MMH3SpatialExtendVideo,
    MMH3SpatialTileEditor,
    # MMH3MaskPreview,  # TEMPORARILY unregistered (debug helper, code kept)
    MMH3LastQuadrantPatch,
)

NODE_CLASS_MAPPINGS = {
    "MMH3SpatialExtendVideo": MMH3SpatialExtendVideo,
    "MMH3SpatialTileEditor": MMH3SpatialTileEditor,
    # "MMH3MaskPreview": MMH3MaskPreview,
    "MMH3LastQuadrantPatch": MMH3LastQuadrantPatch,
}

# front-end JS: Tile Editor dock panel
WEB_DIRECTORY = "./web"

NODE_DISPLAY_NAME_MAPPINGS = {
    "MMH3SpatialExtendVideo": "MMH3 Spatial Extend Video",
    "MMH3SpatialTileEditor": "MMH3 Spatial Tile Editor",
    # "MMH3MaskPreview": "MMH3 Mask Preview",
    "MMH3LastQuadrantPatch": "MMH3 Last Quadrant Patch",
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
            files = []
            if os.path.isdir(input_dir):
                for f in os.listdir(input_dir):
                    if os.path.isfile(os.path.join(input_dir, f)):
                        _, ext = os.path.splitext(f)
                        if ext.lower() in IMAGE_EXTS:
                            files.append(f)
            files.sort()
            return web.json_response(files)
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
        ]


async def comfy_entrypoint() -> MMH3SpatialExtendVideoExtension:
    return MMH3SpatialExtendVideoExtension()
