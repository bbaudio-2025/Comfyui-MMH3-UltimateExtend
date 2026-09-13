// MMH3 Latent Preview — inline (on-node) preview for the experimental
// "MMH3 Latent Preview (approx)" node.
//
// The python node decodes a finished H3 latent with the tiny H3 VAE and pushes an
// animated WebP over WebSocket ("mmh3_latent_preview"). This extension simply
// shows that video on the node via a DOM widget. Keep this minimal — it exists to
// prove out the decode pipeline and is a template for embedding the same preview
// into richer JS UI later.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TARGET = "MMH3LatentPreview";

function b64ToBlob(b64, mime) {
    const bin = atob(b64);
    const arr = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
    return new Blob([arr], { type: mime });
}

// Resolves "12:7:5"-style subgraph ids if present, else plain node id.
function findNode(qid) {
    if (qid == null) return null;
    const parts = String(qid).split(":");
    if (parts.length === 1) return app.graph.getNodeById(parseInt(parts[0], 10));
    let graph = app.graph;
    for (let i = 0; i < parts.length - 1; i++) {
        const parent = graph?.getNodeById?.(parseInt(parts[i], 10));
        if (!parent?.subgraph) return null;
        graph = parent.subgraph;
    }
    return graph?.getNodeById?.(parseInt(parts[parts.length - 1], 10)) || null;
}

app.registerExtension({
    name: "MMH3.LatentPreview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== TARGET) return;

        nodeType.prototype.onNodeCreated = function () {
            const node = this;

            const root = document.createElement("div");
            root.style.display = "flex";
            root.style.flexDirection = "column";
            root.style.width = "100%";
            root.style.height = "100%";
            root.style.overflow = "hidden";

            const img = document.createElement("img");
            img.style.width = "100%";
            img.style.height = "100%";
            img.style.objectFit = "contain";
            img.style.background = "#111";
            img.style.cursor = "pointer";
            img.alt = "";
            img.addEventListener("mousedown", (e) => e.stopPropagation());

            const status = document.createElement("div");
            status.style.position = "absolute";
            status.style.top = "2px";
            status.style.left = "2px";
            status.style.fontSize = "11px";
            status.style.color = "#ccc";
            status.style.textShadow = "0 0 2px #000";
            status.textContent = "offline preview: waiting for run…";

            root.appendChild(img);
            root.appendChild(status);
            node.addDOMWidget("mmh3_preview", "mmh3_preview", root, { serialize: false });
            node.setSize([Math.max(node.size?.[0] ?? 300, 300), Math.max(node.size?.[1] ?? 300, 300)]);

            node._mmh3PreviewHandler = (data) => {
                try {
                    if (data.error) {
                        status.textContent = "error: " + data.error;
                        img.removeAttribute("src");
                        return;
                    }
                    if (typeof data.webp === "string" && data.webp.length > 0) {
                        const url = URL.createObjectURL(b64ToBlob(data.webp, data.mime || "image/webp"));
                        img.src = url;
                        status.textContent = `${data.count ?? "?"}f · ${data.width ?? "?"}×${data.height ?? "?"} · ${data.fps ?? ""}fps`;
                    } else {
                        status.textContent = `${data.count ?? 0} frames (empty)`;
                        img.removeAttribute("src");
                    }
                } catch (err) {
                    console.warn("[MMH3LatentPreview] display failed:", err);
                    status.textContent = "display error";
                }
            };
            node.onRemoved = node.onRemoved || (() => {});
            const origRemoved = node.onRemoved;
            node.onRemoved = function (...args) { node._mmh3PreviewHandler = null; return origRemoved.apply(this, args); };
        };
    },
});

api.addEventListener("mmh3_latent_preview", (e) => {
    const data = e.detail;
    if (!data || data.node_id == null) return;
    const node = findNode(data.node_id);
    node?._mmh3PreviewHandler?.(data);
});