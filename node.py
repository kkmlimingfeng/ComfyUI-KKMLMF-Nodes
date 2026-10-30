import csv
import json
import os
from pathlib import Path
from aiohttp import web

import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence

import comfy.model_management
import folder_paths
import node_helpers
from comfy_api.latest import ComfyExtension, io
from server import PromptServer

from .prompt_parser import parse_prompt, serialize_prompt
from .translator import translate_tags, get_model_status


def load_saved_tags(raw):
    """Parse the frontend's tags_state JSON. Returns None when unusable."""
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(data, list):
        return None
    tags = []
    for item in data:
        if not isinstance(item, dict):
            return None
        try:
            weight = float(item.get("weight", 1.0))
        except (TypeError, ValueError):
            return None
        tags.append({
            "text": str(item.get("text", "")).strip(),
            "translation": str(item.get("translation", "") or ""),
            "weight": weight,
            "disabled": bool(item.get("disabled", False)),
        })
    return tags


def _editor_execute(prompt, translated_prompt, target_language, tags_state=""):
    tags = parse_prompt(prompt)
    # When tags_state is consistent with the prompt text, prefer it: it
    # carries weights, translations and disabled flags.
    saved = load_saved_tags(tags_state)
    if saved is not None:
        enabled = [t for t in saved if not t["disabled"]]
        if serialize_prompt(enabled) == serialize_prompt(tags):
            tags = saved
    return io.NodeOutput(
        serialize_prompt(tags),
        translated_prompt or "",
        json.dumps(tags, ensure_ascii=False),
    )


class PromptTagEditor(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PromptTagEditor",
            display_name="Prompt Tag Editor",
            category="Prompt Tools",
            description="Interactive synchronized prompt/tag editor with optional Hy-MT2 translation.",
            inputs=[
                io.String.Input(
                    "prompt",
                    default="",
                    multiline=True,
                    tooltip="Original prompt. Tags are separated by commas.",
                ),
                io.String.Input(
                    "translated_prompt",
                    default="",
                    multiline=True,
                    tooltip="Translated tags. Kept in sync by the frontend editor UI.",
                ),
                io.Combo.Input(
                    "target_language",
                    options=[
                        "Chinese", "Japanese", "Korean", "English",
                        "French", "German", "Spanish", "Russian",
                    ],
                    default="Chinese",
                    tooltip="Language used by the translation result Tags.",
                ),
                io.String.Input(
                    "tags_state",
                    default="",
                    tooltip="Internal editor state (tags JSON). Managed by the frontend UI; do not edit.",
                ),
            ],
            outputs=[
                io.String.Output("prompt", display_name="Prompt"),
                io.String.Output("translated_prompt", display_name="Translated Prompt"),
                io.String.Output("tags_json", display_name="Tags JSON"),
            ],
        )

    @classmethod
    def execute(cls, prompt, translated_prompt, target_language, tags_state=""):
        return _editor_execute(prompt, translated_prompt, target_language, tags_state)


class PromptTagEditorPlus(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PromptTagEditorPlus",
            display_name="Prompt Tag Editor Plus",
            category="Prompt Tools",
            description="Prompt Tag Editor with tag suggestions: type English or Chinese in the "
                        "prompt box to get the most-used matching tags, click one to insert it.",
            inputs=[
                io.String.Input(
                    "prompt",
                    default="",
                    multiline=True,
                    tooltip="Original prompt. Tags are separated by commas.",
                ),
                io.String.Input(
                    "translated_prompt",
                    default="",
                    multiline=True,
                    tooltip="Translated tags. Kept in sync by the frontend editor UI.",
                ),
                io.Combo.Input(
                    "target_language",
                    options=[
                        "Chinese", "Japanese", "Korean", "English",
                        "French", "German", "Spanish", "Russian",
                    ],
                    default="Chinese",
                    tooltip="Language used by the translation result Tags.",
                ),
                io.String.Input(
                    "dataset",
                    default="",
                    tooltip="Suggestion dataset (CSV file in the data/ folder). "
                            "Managed by the frontend select; empty = first dataset.",
                ),
                io.String.Input(
                    "tags_state",
                    default="",
                    tooltip="Internal editor state (tags JSON). Managed by the frontend UI; do not edit.",
                ),
            ],
            outputs=[
                io.String.Output("prompt", display_name="Prompt"),
                io.String.Output("translated_prompt", display_name="Translated Prompt"),
                io.String.Output("tags_json", display_name="Tags JSON"),
            ],
        )

    @classmethod
    def execute(cls, prompt, translated_prompt, target_language, dataset="", tags_state=""):
        return _editor_execute(prompt, translated_prompt, target_language, tags_state)


class PromptTagEditorExtension(ComfyExtension):
    async def get_node_list(self):
        from .reroute import MultiReroute
        return [PromptTagEditor, PromptTagEditorPlus, LoadImageFromPath, MultiReroute]


# ---- load image from path -------------------------------------------------

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}


def resolve_image_path(raw):
    """Resolve a user-supplied path to an existing image file.

    Absolute paths are used as-is. Relative paths are tried against the
    input directory, the output directory and the ComfyUI base folder.
    Raises ValueError with a user-facing message when unusable.
    """
    path = str(raw).strip().strip('"').strip("'").strip()
    if not path:
        raise ValueError("Image path is empty.")

    candidates = [path]
    if not os.path.isabs(path):
        for base in (folder_paths.get_input_directory(),
                     folder_paths.get_output_directory(),
                     folder_paths.base_path):
            candidates.append(os.path.join(base, path))

    for cand in candidates:
        cand = os.path.abspath(cand)
        if os.path.isfile(cand):
            if os.path.splitext(cand)[1].lower() not in _IMAGE_EXTENSIONS:
                raise ValueError(f"Not a supported image file: {cand}")
            return cand
    raise ValueError(f"Image file not found: {path}")


class LoadImageFromPath(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoadImageFromPath",
            display_name="Load Image From Path",
            category="Prompt Tools",
            description="Load an image (and optional alpha mask) from an arbitrary file path.",
            inputs=[
                io.String.Input(
                    "image_path",
                    default="",
                    tooltip="Image file path. Absolute path, or relative to the input / output / ComfyUI folders.",
                ),
            ],
            outputs=[
                io.Image.Output(),
                io.Mask.Output(),
            ],
        )

    @classmethod
    def validate_inputs(cls, image_path):
        try:
            resolve_image_path(image_path)
        except ValueError as exc:
            return str(exc)
        return True

    @classmethod
    def fingerprint_inputs(cls, image_path):
        try:
            path = resolve_image_path(image_path)
        except ValueError:
            return str(image_path)
        try:
            return f"{path}:{os.path.getmtime(path)}"
        except OSError:
            return path

    @classmethod
    def execute(cls, image_path):
        path = resolve_image_path(image_path)

        dtype = comfy.model_management.intermediate_dtype()
        device = comfy.model_management.intermediate_device()

        img = node_helpers.pillow(Image.open, path)

        # Same loading logic as the core LoadImage node: EXIF-aware,
        # animated files are flattened to same-size frames, alpha becomes
        # an inverted mask, frames without alpha get an empty mask.
        output_images = []
        output_masks = []
        w = h = None
        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)
            image = i.convert("RGB")
            if w is None:
                w, h = image.size
            if image.size[0] != w or image.size[1] != h:
                continue
            image = np.array(image).astype(np.float32) / 255.0
            output_images.append(torch.from_numpy(image)[None, ])
            if "A" in i.getbands():
                mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
                output_masks.append((1.0 - torch.from_numpy(mask)).unsqueeze(0))
            else:
                output_masks.append(torch.zeros((1, 64, 64), dtype=torch.float32))

        if not output_images:
            raise ValueError(f"Could not read any frames from image: {path}")

        output_image = torch.cat(output_images, dim=0).to(device=device, dtype=dtype)
        output_mask = torch.cat(output_masks, dim=0).to(device=device, dtype=dtype)
        return io.NodeOutput(output_image, output_mask)


# ---- tag suggestion datasets (PromptTagEditorPlus) -------------------------

_DATA_DIR = Path(__file__).parent / "data"
_dataset_cache = {}  # filename -> {"mtime": float, "rows": list}


def _list_datasets():
    """CSV files available in the data/ folder (sorted by name)."""
    try:
        return sorted(p.name for p in _DATA_DIR.glob("*.csv") if p.is_file())
    except OSError:
        return []


def _load_dataset(name):
    """Load a dataset CSV into memory (cached by filename + mtime).

    Rows: name, count, zh (manual translation), zh_mt (machine translation),
    aliases (| separated). Sorted by usage count descending.
    """
    safe = os.path.basename(str(name or "").strip())
    if not safe:
        raise ValueError("数据集名为空")
    path = _DATA_DIR / safe
    if not path.is_file():
        raise ValueError(f"数据集不存在: {safe}")

    try:
        mtime = path.stat().st_mtime
    except OSError as exc:
        raise ValueError(f"无法读取数据集: {exc}")
    cached = _dataset_cache.get(safe)
    if cached and cached["mtime"] == mtime:
        return cached["rows"]

    rows = []
    # utf-8-sig: the CSVs carry a BOM; plain utf-8 would hide the first
    # column ("name") behind it.
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            try:
                count = int(r.get("count") or 0)
            except (TypeError, ValueError):
                count = 0
            rows.append({
                "name": (r.get("name") or "").strip(),
                "count": count,
                "zh": (r.get("zh") or "").strip(),
                "zh_mt": (r.get("zh_mt") or "").strip(),
                "aliases": [a.strip() for a in (r.get("aliases") or "").split("|") if a.strip()],
                "cat": (r.get("cat") or "").strip(),
            })
    rows.sort(key=lambda r: -r["count"])
    _dataset_cache[safe] = {"mtime": mtime, "rows": rows}
    return rows


def _has_cjk(s):
    return any("\u4e00" <= ch <= "\u9fff" for ch in s)


def _search_tags(rows, query, limit):
    """Search a loaded dataset. English queries match name/aliases (spaces in
    the query match underscores in tags, and multi-word queries match in any
    order); Chinese queries match the manual (zh) and machine (zh_mt)
    translations. Results ranked by match quality then usage count."""
    ql = query.strip().lower()
    if not ql:
        return []
    tokens = [t for t in ql.replace("_", " ").split() if t]
    nq = "_".join(tokens)  # "long hair" -> "long_hair"
    cjk = _has_cjk(ql)

    hits = []
    for row in rows:
        ranks = []
        nl = row["name"].lower()
        als = [a.lower() for a in row["aliases"]]
        if nq:
            if nl == nq:
                ranks.append(0)
            elif nl.startswith(nq):
                ranks.append(1)
            elif nq in nl:
                ranks.append(4)
            if any(a == nq for a in als):
                ranks.append(2)
            elif any(a.startswith(nq) for a in als):
                ranks.append(3)
            elif any(nq in a for a in als):
                ranks.append(5)
            if len(tokens) > 1:
                hay = (nl + " " + " ".join(als)).replace("_", " ")
                if all(t in hay for t in tokens):
                    ranks.append(6)
        if cjk:
            zh, zh_mt = row["zh"], row["zh_mt"]
            if zh:
                if zh.startswith(query):
                    ranks.append(1.5)
                elif query in zh:
                    ranks.append(3.5)
            if zh_mt and query in zh_mt:
                ranks.append(5.5)
        if ranks:
            hits.append((min(ranks), -row["count"], row))

    hits.sort(key=lambda h: (h[0], h[1]))
    return [h[2] for h in hits[:limit]]


@PromptServer.instance.routes.get("/prompt_tag_editor/datasets")
async def prompt_tag_editor_datasets(request):
    datasets = _list_datasets()
    return web.json_response({"ok": True, "datasets": datasets})


@PromptServer.instance.routes.get("/prompt_tag_editor/suggest")
async def prompt_tag_editor_suggest(request):
    q = request.query.get("q", "")
    try:
        limit = max(1, min(int(request.query.get("limit", "50")), 100))
    except ValueError:
        limit = 50

    dataset = request.query.get("dataset", "").strip()
    if not dataset:
        datasets = _list_datasets()
        if not datasets:
            return web.json_response({"ok": False, "error": "data 目录下没有数据集 CSV"}, status=400)
        dataset = datasets[0]

    try:
        rows = _load_dataset(dataset)
    except ValueError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)
    except Exception as exc:
        return web.json_response({"ok": False, "error": f"数据集加载失败: {exc}"}, status=500)

    results = _search_tags(rows, q, limit) if q.strip() else []
    return web.json_response({"ok": True, "dataset": dataset, "results": [
        {"name": r["name"], "zh": r["zh"], "zh_mt": r["zh_mt"],
         "count": r["count"], "cat": r["cat"]} for r in results
    ]})


@PromptServer.instance.routes.post("/prompt_tag_editor/translate")
async def prompt_tag_editor_translate(request):
    try:
        data = await request.json()
        tags = data.get("tags", [])
        language = data.get("target_language", "Chinese")

        texts = [str(t.get("text", "")).strip() for t in tags]
        translations = await translate_tags(texts, language)

        return web.json_response({
            "ok": True,
            "translations": translations,
            "model_status": get_model_status(),
        })
    except Exception as exc:
        return web.json_response({
            "ok": False,
            "error": str(exc),
            "model_status": get_model_status(),
        }, status=500)


@PromptServer.instance.routes.get("/prompt_tag_editor/status")
async def prompt_tag_editor_status(request):
    return web.json_response(get_model_status())


@PromptServer.instance.routes.get("/prompt_tag_editor/browse")
async def prompt_tag_editor_browse(request):
    """List directories and image files for the frontend path picker.

    `path` = directory to list; empty → input directory;
    `__drives__` → drive list (Windows). Only names/paths are returned,
    no file contents.
    """
    raw = request.query.get("path", "").strip()
    try:
        if raw == "__drives__":
            drives = [f"{letter}:\\" for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                      if os.path.exists(f"{letter}:\\")]
            return web.json_response({
                "ok": True, "path": "此电脑（所有盘符）", "parent": "",
                "dirs": [{"name": d, "path": d} for d in drives], "files": [],
            })

        path = os.path.abspath(raw or folder_paths.get_input_directory())
        if not os.path.isdir(path):
            return web.json_response(
                {"ok": False, "error": f"目录不存在: {path}"}, status=400)

        dirs, files = [], []
        try:
            with os.scandir(path) as it:
                for entry in it:
                    try:
                        if entry.is_dir():
                            dirs.append({"name": entry.name, "path": entry.path})
                        elif entry.is_file() and os.path.splitext(entry.name)[1].lower() in _IMAGE_EXTENSIONS:
                            files.append({"name": entry.name, "path": entry.path})
                    except OSError:
                        continue  # unreadable entry (e.g. Windows junction)
        except (PermissionError, OSError) as exc:
            return web.json_response(
                {"ok": False, "error": f"无法读取目录 {path}: {exc}"}, status=400)

        parent = os.path.dirname(path)
        if parent == path:
            parent = "__drives__"
        dirs.sort(key=lambda d: d["name"].lower())
        files.sort(key=lambda f: f["name"].lower())
        return web.json_response({
            "ok": True, "path": path, "parent": parent, "dirs": dirs, "files": files,
        })
    except Exception as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=500)


_CONTENT_TYPES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp",
    ".tif": "image/tiff", ".tiff": "image/tiff",
}


@PromptServer.instance.routes.get("/prompt_tag_editor/view")
async def prompt_tag_editor_view(request):
    """Serve an image file's content for the path picker's preview thumbnail."""
    raw = request.query.get("path", "")
    try:
        path = resolve_image_path(raw)
    except ValueError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)
    try:
        if os.path.getsize(path) > 64 * 1024 * 1024:
            return web.json_response(
                {"ok": False, "error": "文件过大，无法预览（>64MB）"}, status=400)
        with open(path, "rb") as f:
            body = f.read()
    except OSError as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)
    ext = os.path.splitext(path)[1].lower()
    return web.Response(
        body=body,
        content_type=_CONTENT_TYPES.get(ext, "application/octet-stream"),
        headers={"Cache-Control": "no-cache"},
    )
