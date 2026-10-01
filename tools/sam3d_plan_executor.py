"""
工具：按分割计划执行 SAM 3D 建模
支持 prompt / box_prompts / mask_urls 三种方法，并分批执行。

执行流程：
1) 先读取 sam_segmentation_plan（决策层）
2) 物化参数（自动补齐 mask_urls / box_prompts）
3) 调用 SAM 3D 建模并下载 GLB
"""
from __future__ import annotations

import json
import os
import re
import struct
import zlib
from datetime import datetime
from typing import Any

import fal_client
import httpx

from config import (
    FAL_KEY,
    FAL_SAM2D_MODEL,
    FAL_SAM3D_MODEL,
    MODEL_OUTPUT_DIR,
    PROJECT_DIR,
)
from token_tracker import tracker

os.environ["FAL_KEY"] = FAL_KEY

GUIDANCE_ROOT = os.path.join(PROJECT_DIR, "debug_snapshots", "sam_guidance")
BOX_SCORE_THRESHOLD = 0.35
BOX_MIN_AREA_RATIO = 0.002
BOX_MAX_AREA_RATIO = 0.95


async def execute_sam_segmentation_plan(image_path: str, plan: dict[str, Any]) -> dict[str, Any]:
    """
    按 plan 执行 SAM 3D 建模，返回可直接用于后续导入的本地模型列表。
    """
    image_url = await fal_client.upload_file_async(image_path)
    guidance_dir = _create_guidance_dir()
    print("  ☁️ 图片已上传，开始按分割计划执行")
    print(f"  🗂️ 2D 引导调试目录: {guidance_dir}")

    objects_cfg = _normalize_plan_objects(plan)
    if not objects_cfg:
        return {"objects": [], "raw_result": {"runs": [], "failures": ["empty plan"]}}

    global_policy = plan.get("global_policy", {}) if isinstance(plan, dict) else {}
    default_thresholds = global_policy.get("default_thresholds")
    if not isinstance(default_thresholds, list):
        default_thresholds = [None, 0.2, 0.1]

    ordered = sorted(
        objects_cfg,
        key=lambda x: (
            _batch_rank(x.get("batch", "secondary"), plan),
            _safe_int(x.get("priority"), 999),
        ),
    )

    runs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    local_objects: list[dict[str, Any]] = []

    for idx, obj in enumerate(ordered):
        label = str(obj.get("label") or f"object_{idx}").strip()
        print(f"  🎯 执行对象: {label} | batch={obj.get('batch')} | method={obj.get('method')}")

        result = await _run_single_object(
            image_url=image_url,
            obj=obj,
            thresholds=default_thresholds,
            guidance_dir=guidance_dir,
            obj_index=idx,
        )
        result["label"] = label
        runs.append(result)

        if not result.get("ok"):
            failures.append(
                {
                    "label": label,
                    "reason": result.get("reason", "unknown"),
                    "materialized": result.get("materialized", []),
                }
            )
            print(f"  ⚠️ 对象失败: {label} | {result.get('reason', 'unknown')}")
            continue

        raw = result.get("raw", {})
        extracted = _extract_objects_from_result(raw, label)
        if not extracted:
            failures.append({"label": label, "reason": "no glb in response"})
            print(f"  ⚠️ 对象无可用 GLB: {label}")
            continue

        for j, item in enumerate(extracted):
            glb_url = item.get("glb_url")
            if not glb_url:
                continue
            safe_label = _safe_name(str(item.get("label", label)))
            local_name = f"sam3d_{safe_label}_{idx}_{j}.glb"
            local_path = os.path.join(MODEL_OUTPUT_DIR, local_name)
            dl = await _download_with_retry(glb_url, local_path, retries=1)
            if dl.get("ok"):
                local_objects.append(
                    {
                        "label": item.get("label", label),
                        "glb_path": local_path,
                        "transform": item.get("transform", {}),
                    }
                )
                attempts = dl.get("attempts", 1)
                if attempts > 1:
                    print(f"  ✅ 模型已下载(重试成功): {local_path} | attempts={attempts}")
                else:
                    print(f"  ✅ 模型已下载: {local_path}")
                continue

            errors = dl.get("errors") or []
            reason = str(errors[-1]) if errors else "unknown"
            failures.append(
                {
                    "label": label,
                    "reason": f"download failed after 2 attempts: {reason}",
                    "download_errors": errors,
                }
            )
            print(f"  ⚠️ 下载失败(重试后仍失败): {label} | {reason}")

    tracker.record_sam3d(len(local_objects), "SAM 3D 建模")
    return {
        "objects": local_objects,
        "raw_result": {
            "runs": runs,
            "failures": failures,
            "plan": plan,
            "guidance_dir": guidance_dir,
        },
    }


async def _materialize_object_guidance_for_method(
    image_url: str,
    obj: dict[str, Any],
    guidance_dir: str,
    obj_index: int,
    target_method: str,
    sam2d_cache: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    """
    懒物化：只为当前目标方法补参数，不做全量预热。
    - target_method == mask_urls: 只补 mask_url(s)
    - target_method == box_prompts: 只补 box_prompt(s)
    """
    item = dict(obj)
    label = str(item.get("label") or f"object_{obj_index}").strip()

    need_mask = target_method == "mask_urls" and not _normalize_mask_urls(
        item.get("mask_url") or item.get("mask_urls")
    )
    need_box = target_method == "box_prompts" and not _normalize_box_prompts(
        item.get("box_prompt") or item.get("box_prompts")
    )

    if not (need_mask or need_box):
        return (
            item,
            {"skipped": True, "target_method": target_method, "reason": "already has params"},
            sam2d_cache,
        )

    sam2d = sam2d_cache
    if sam2d is None:
        sam2d = await _run_sam2d_guidance(image_url=image_url, label=label)

    info: dict[str, Any] = {
        "skipped": False,
        "target_method": target_method,
        "need_mask": need_mask,
        "need_box": need_box,
        "sam2d_ok": sam2d.get("ok", False),
        "sam2d_reason": sam2d.get("reason", ""),
    }

    if not sam2d.get("ok"):
        _write_json(
            os.path.join(guidance_dir, f"{obj_index:02d}_{_safe_name(label)}_{target_method}.json"),
            info,
        )
        return item, info, sam2d

    best = sam2d.get("best") or {}
    best_mask_url = str(best.get("mask_url") or "").strip()
    best_box_xyxy = best.get("box_xyxy")
    width = _safe_int(best.get("width"), 0)
    height = _safe_int(best.get("height"), 0)
    score = float(best.get("score") or 0.0)

    local_mask_path = ""
    if best_mask_url and (need_mask or need_box):
        local_mask_path = os.path.join(
            guidance_dir,
            f"{obj_index:02d}_{_safe_name(label)}_mask.png",
        )
        try:
            await _download(best_mask_url, local_mask_path)
        except Exception:
            local_mask_path = ""

    if need_mask:
        if best_mask_url:
            item["mask_url"] = best_mask_url
            item["mask_urls"] = [best_mask_url]
            info["mask_source"] = "sam2d"
        else:
            info["mask_source"] = "missing"

    if need_box:
        # 优先：2D 直接返回的框
        if best_box_xyxy and _is_box_stable(best_box_xyxy, width, height, score):
            item["box_prompt"] = [
                {
                    "x_min": best_box_xyxy[0],
                    "y_min": best_box_xyxy[1],
                    "x_max": best_box_xyxy[2],
                    "y_max": best_box_xyxy[3],
                    "object_id": 1,
                }
            ]
            info["box_source"] = "sam2d_direct_box"
            info["box_score"] = score
        # 失败/不稳：从 mask 回推
        elif local_mask_path:
            backproj_box = _bbox_from_png_mask(local_mask_path)
            if backproj_box:
                item["box_prompt"] = [
                    {
                        "x_min": backproj_box[0],
                        "y_min": backproj_box[1],
                        "x_max": backproj_box[2],
                        "y_max": backproj_box[3],
                        "object_id": 1,
                    }
                ]
                info["box_source"] = "mask_backproject"
            else:
                info["box_source"] = "missing"
        else:
            info["box_source"] = "missing"

    info["mask_local_path"] = local_mask_path
    _write_json(
        os.path.join(guidance_dir, f"{obj_index:02d}_{_safe_name(label)}_{target_method}.json"),
        {
            **info,
            "materialized_method": item.get("method"),
            "materialized_box_prompt": item.get("box_prompt"),
            "materialized_mask_url": item.get("mask_url"),
            "sam2d_candidate_count": len(sam2d.get("candidates", [])),
        },
    )
    return item, info, sam2d


async def _run_sam2d_guidance(image_url: str, label: str) -> dict[str, Any]:
    try:
        raw = await fal_client.subscribe_async(
            FAL_SAM2D_MODEL,
            arguments={
                "image_url": image_url,
                "prompt": label,
                "apply_mask": False,
                "output_format": "png",
                "return_multiple_masks": True,
                "max_masks": 3,
                "include_scores": True,
                "include_boxes": True,
            },
        )
    except Exception as e:
        return {"ok": False, "reason": str(e), "candidates": []}

    candidates = _extract_sam2d_candidates(raw)
    best = _pick_best_candidate(candidates)
    return {"ok": True, "raw": raw, "candidates": candidates, "best": best}


def _extract_sam2d_candidates(raw: dict[str, Any]) -> list[dict[str, Any]]:
    masks = raw.get("masks") if isinstance(raw, dict) else None
    boxes = raw.get("boxes") if isinstance(raw, dict) else None
    scores = raw.get("scores") if isinstance(raw, dict) else None
    metadata = raw.get("metadata") if isinstance(raw, dict) else None

    if not isinstance(masks, list):
        masks = []
    if not isinstance(boxes, list):
        boxes = []
    if not isinstance(scores, list):
        scores = []
    if not isinstance(metadata, list):
        metadata = []

    n = max(len(masks), len(boxes), len(scores), len(metadata), 1)
    candidates: list[dict[str, Any]] = []
    for i in range(n):
        mask = masks[i] if i < len(masks) and isinstance(masks[i], dict) else {}
        meta = metadata[i] if i < len(metadata) and isinstance(metadata[i], dict) else {}

        mask_url = str(mask.get("url") or "").strip()
        width = _safe_int(mask.get("width"), 0)
        height = _safe_int(mask.get("height"), 0)

        score = 0.0
        if i < len(scores):
            try:
                score = float(scores[i])
            except Exception:
                pass
        elif "score" in meta:
            try:
                score = float(meta.get("score"))
            except Exception:
                pass

        box_norm = None
        if i < len(boxes) and isinstance(boxes[i], list):
            box_norm = boxes[i]
        elif isinstance(meta.get("box"), list):
            box_norm = meta.get("box")

        box_xyxy = None
        if box_norm and width > 0 and height > 0:
            box_xyxy = _norm_cxcywh_to_xyxy(box_norm, width, height)

        candidates.append(
            {
                "index": i,
                "mask_url": mask_url,
                "width": width,
                "height": height,
                "score": score,
                "box_norm": box_norm,
                "box_xyxy": box_xyxy,
            }
        )
    return candidates


def _pick_best_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        return {}

    def _rank_key(c: dict[str, Any]) -> tuple[float, float]:
        score = float(c.get("score") or 0.0)
        box = c.get("box_xyxy")
        width = _safe_int(c.get("width"), 0)
        height = _safe_int(c.get("height"), 0)
        area = 0.0
        if box and width > 0 and height > 0:
            w = max(0, box[2] - box[0])
            h = max(0, box[3] - box[1])
            area = (w * h) / max(1, width * height)
        return score, area

    return sorted(candidates, key=_rank_key, reverse=True)[0]


async def _run_single_object(
    image_url: str,
    obj: dict[str, Any],
    thresholds: list[Any],
    guidance_dir: str,
    obj_index: int,
) -> dict[str, Any]:
    label = str(obj.get("label") or "object")
    method = str(obj.get("method") or "prompt")
    retry_methods = obj.get("retry_methods")
    if not isinstance(retry_methods, list) or not retry_methods:
        retry_methods = [method]

    work_obj = dict(obj)
    materialized_events: list[dict[str, Any]] = []
    sam2d_cache: dict[str, Any] | None = None
    last_error = "unknown"

    for m in retry_methods:
        method_name = str(m)
        args = _build_method_args(image_url=image_url, obj=work_obj, method=method_name)

        # 懒物化：仅当当前方法缺参，且方法为 box/mask 时才触发
        if args is None and method_name in {"box_prompts", "mask_urls"}:
            work_obj, info, sam2d_cache = await _materialize_object_guidance_for_method(
                image_url=image_url,
                obj=work_obj,
                guidance_dir=guidance_dir,
                obj_index=obj_index,
                target_method=method_name,
                sam2d_cache=sam2d_cache,
            )
            materialized_events.append(info)
            args = _build_method_args(image_url=image_url, obj=work_obj, method=method_name)

        if args is None:
            materialized_events.append(
                {
                    "skipped": True,
                    "target_method": method_name,
                    "reason": "missing required params",
                }
            )
            continue

        for th in thresholds:
            req = dict(args)
            req["export_textured_glb"] = True
            if th is not None:
                req["detection_threshold"] = th
            try:
                raw = await fal_client.subscribe_async(FAL_SAM3D_MODEL, arguments=req)
                return {
                    "ok": True,
                    "raw": raw,
                    "method": method_name,
                    "threshold": th,
                    "materialized": materialized_events,
                }
            except Exception as e:
                last_error = str(e)
    return {"ok": False, "reason": f"{label}: {last_error}", "materialized": materialized_events}


def _build_method_args(image_url: str, obj: dict[str, Any], method: str) -> dict[str, Any] | None:
    base = {"image_url": image_url}
    label = str(obj.get("label") or "object").strip()

    if method == "prompt":
        prompt = str(obj.get("prompt") or label).strip()
        if not prompt:
            return None
        return {**base, "prompt": prompt}

    if method == "box_prompts":
        box_prompts = _normalize_box_prompts(obj.get("box_prompt") or obj.get("box_prompts"))
        if not box_prompts:
            return None
        return {**base, "box_prompts": box_prompts}

    if method == "mask_urls":
        mask_urls = _normalize_mask_urls(obj.get("mask_url") or obj.get("mask_urls"))
        if not mask_urls:
            return None
        return {**base, "mask_urls": mask_urls}

    return None


def _normalize_box_prompts(raw: Any) -> list[dict[str, int]]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        raw = [raw]
    if isinstance(raw, list) and raw and all(isinstance(x, (int, float)) for x in raw):
        raw = [raw]
    if not isinstance(raw, list):
        return []

    prompts: list[dict[str, int]] = []
    object_id = 1
    for item in raw:
        if isinstance(item, dict):
            try:
                x_min = int(item["x_min"])
                y_min = int(item["y_min"])
                x_max = int(item["x_max"])
                y_max = int(item["y_max"])
                oid = int(item.get("object_id", object_id))
                prompts.append(
                    {
                        "x_min": x_min,
                        "y_min": y_min,
                        "x_max": x_max,
                        "y_max": y_max,
                        "object_id": oid,
                    }
                )
            except Exception:
                continue
        elif isinstance(item, list) and len(item) >= 4:
            try:
                prompts.append(
                    {
                        "x_min": int(item[0]),
                        "y_min": int(item[1]),
                        "x_max": int(item[2]),
                        "y_max": int(item[3]),
                        "object_id": object_id,
                    }
                )
            except Exception:
                continue
        object_id += 1
    return prompts


def _normalize_mask_urls(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw] if raw.strip() else []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if isinstance(x, str) and x.strip()]
    return []


def _normalize_plan_objects(plan: Any) -> list[dict[str, Any]]:
    if not isinstance(plan, dict):
        return []
    objects = plan.get("objects")
    if not isinstance(objects, list):
        return []
    out: list[dict[str, Any]] = []
    for o in objects:
        if not isinstance(o, dict):
            continue
        label = str(o.get("label") or "").strip()
        if not label:
            continue
        out.append(dict(o))
    return _dedupe_plan_objects_for_execution(out)


def _dedupe_plan_objects_for_execution(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    for obj in objects:
        key = _semantic_object_key(obj)
        if not key:
            continue

        if key not in seen:
            seen[key] = obj
            order.append(key)
            continue

        old = seen[key]
        new = obj
        old_rank = _exec_object_rank(old)
        new_rank = _exec_object_rank(new)
        keep = old if old_rank <= new_rank else new
        drop = new if keep is old else old

        keep["retry_methods"] = _merge_retry_methods_for_execution(
            keep.get("retry_methods"),
            drop.get("retry_methods"),
            keep.get("method"),
        )
        if not keep.get("box_prompt") and drop.get("box_prompt"):
            keep["box_prompt"] = drop.get("box_prompt")
        if not keep.get("box_prompts") and drop.get("box_prompts"):
            keep["box_prompts"] = drop.get("box_prompts")
        if not keep.get("mask_url") and drop.get("mask_url"):
            keep["mask_url"] = drop.get("mask_url")
        if not keep.get("mask_urls") and drop.get("mask_urls"):
            keep["mask_urls"] = drop.get("mask_urls")
        if not keep.get("prompt") and drop.get("prompt"):
            keep["prompt"] = drop.get("prompt")

        seen[key] = keep

    return [seen[k] for k in order if k in seen]


def _semantic_object_key(obj: dict[str, Any]) -> str:
    label = str(obj.get("label") or "").strip().lower()
    if not label:
        return ""
    label = _normalize_label_for_key(label)
    if not label:
        return ""
    return label


def _normalize_label_for_key(label: str) -> str:
    norm = re.sub(r"[^a-z0-9\s]+", " ", label.lower())
    norm = re.sub(r"\s+", " ", norm).strip()
    tree_keys = ("tree", "trunk", "branch", "canopy", "leaf", "leaves", "foliage", "wood structure")
    if any(k in norm for k in tree_keys):
        return "tree"
    return norm


def _exec_object_rank(obj: dict[str, Any]) -> tuple[int, int, int]:
    batch = str(obj.get("batch") or "secondary")
    batch_rank = 1
    if batch == "anchor":
        batch_rank = 0
    elif batch == "optional":
        batch_rank = 2

    priority = _safe_int(obj.get("priority"), 999)
    method = str(obj.get("method") or "prompt")
    method_rank = {"mask_urls": 0, "box_prompts": 1, "prompt": 2}.get(method, 3)
    return (batch_rank, priority, method_rank)


def _merge_retry_methods_for_execution(
    base: Any,
    incoming: Any,
    default_method: Any,
) -> list[str]:
    allow = ("prompt", "box_prompts", "mask_urls")
    methods: list[str] = []

    def _push(raw: Any) -> None:
        if not isinstance(raw, list):
            return
        for x in raw:
            m = str(x).strip()
            if m in allow and m not in methods:
                methods.append(m)

    _push(base)
    _push(incoming)
    dm = str(default_method or "prompt").strip()
    if dm in allow and dm not in methods:
        methods.insert(0, dm)
    if not methods:
        methods = ["prompt", "box_prompts", "mask_urls"]
    for m in allow:
        if m not in methods:
            methods.append(m)
    return methods


def _batch_rank(batch: str, plan: dict[str, Any]) -> int:
    policy = plan.get("global_policy", {}) if isinstance(plan, dict) else {}
    order = policy.get("batch_order") if isinstance(policy, dict) else None
    if not isinstance(order, list):
        order = ["anchor", "secondary", "optional"]
    try:
        return order.index(batch)
    except ValueError:
        return len(order)


def _extract_objects_from_result(result: dict[str, Any], label: str) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    metadata = result.get("metadata") or []

    individual_glbs = result.get("individual_glbs") or []
    for idx, glb in enumerate(individual_glbs):
        url = glb.get("url") if isinstance(glb, dict) else None
        if not url:
            continue
        transform = metadata[idx] if idx < len(metadata) and isinstance(metadata[idx], dict) else {}
        suffix = f"_{idx}" if len(individual_glbs) > 1 else ""
        objects.append(
            {
                "label": f"{label}{suffix}",
                "glb_url": url,
                "transform": transform,
            }
        )

    if objects:
        return objects

    model_glb = result.get("model_glb") or {}
    model_glb_url = model_glb.get("url") if isinstance(model_glb, dict) else None
    if model_glb_url:
        first_meta = metadata[0] if metadata and isinstance(metadata[0], dict) else {}
        objects.append(
            {
                "label": label,
                "glb_url": model_glb_url,
                "transform": first_meta,
            }
        )
    return objects


def _is_box_stable(box: list[int], width: int, height: int, score: float) -> bool:
    if not box or width <= 0 or height <= 0:
        return False
    x_min, y_min, x_max, y_max = box
    if x_max <= x_min or y_max <= y_min:
        return False
    area_ratio = ((x_max - x_min) * (y_max - y_min)) / max(1, width * height)
    if area_ratio < BOX_MIN_AREA_RATIO or area_ratio > BOX_MAX_AREA_RATIO:
        return False
    return score >= BOX_SCORE_THRESHOLD


def _norm_cxcywh_to_xyxy(box: list[Any], width: int, height: int) -> list[int] | None:
    if len(box) < 4 or width <= 0 or height <= 0:
        return None
    try:
        cx = float(box[0])
        cy = float(box[1])
        bw = float(box[2])
        bh = float(box[3])
    except Exception:
        return None

    x_min = int(round((cx - bw / 2.0) * width))
    y_min = int(round((cy - bh / 2.0) * height))
    x_max = int(round((cx + bw / 2.0) * width))
    y_max = int(round((cy + bh / 2.0) * height))

    x_min = max(0, min(x_min, width - 1))
    y_min = max(0, min(y_min, height - 1))
    x_max = max(0, min(x_max, width - 1))
    y_max = max(0, min(y_max, height - 1))
    if x_max <= x_min or y_max <= y_min:
        return None
    return [x_min, y_min, x_max, y_max]


def _bbox_from_png_mask(path: str) -> list[int] | None:
    try:
        with open(path, "rb") as f:
            data = f.read()
        return _bbox_from_png_bytes(data)
    except Exception:
        return None


def _bbox_from_png_bytes(data: bytes) -> list[int] | None:
    signature = b"\x89PNG\r\n\x1a\n"
    if not data.startswith(signature):
        return None

    pos = len(signature)
    width = 0
    height = 0
    bit_depth = 0
    color_type = 0
    interlace = 0
    idat = bytearray()

    while pos + 8 <= len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        pos += 4
        chunk_type = data[pos : pos + 4]
        pos += 4
        chunk_data = data[pos : pos + length]
        pos += length
        pos += 4  # crc

        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _, _, interlace = struct.unpack(">IIBBBBB", chunk_data[:13])
        elif chunk_type == b"IDAT":
            idat.extend(chunk_data)
        elif chunk_type == b"IEND":
            break

    if width <= 0 or height <= 0 or bit_depth != 8 or interlace != 0:
        return None

    channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(color_type)
    if channels is None:
        return None

    try:
        raw = zlib.decompress(bytes(idat))
    except Exception:
        return None

    stride = width * channels
    expected = height * (1 + stride)
    if len(raw) < expected:
        return None

    min_x = width
    min_y = height
    max_x = -1
    max_y = -1

    prev = bytearray(stride)
    cursor = 0
    for y in range(height):
        filter_type = raw[cursor]
        cursor += 1
        scan = bytearray(raw[cursor : cursor + stride])
        cursor += stride
        recon = _png_unfilter(filter_type, scan, prev, channels)
        prev = recon

        for x in range(width):
            if _mask_pixel_on(recon, x, channels, color_type):
                if x < min_x:
                    min_x = x
                if y < min_y:
                    min_y = y
                if x > max_x:
                    max_x = x
                if y > max_y:
                    max_y = y

    if max_x < min_x or max_y < min_y:
        return None
    return [min_x, min_y, max_x, max_y]


def _png_unfilter(filter_type: int, scan: bytearray, prev: bytearray, bpp: int) -> bytearray:
    out = bytearray(scan)
    if filter_type == 0:
        return out

    if filter_type == 1:  # Sub
        for i in range(len(out)):
            left = out[i - bpp] if i >= bpp else 0
            out[i] = (out[i] + left) & 0xFF
        return out

    if filter_type == 2:  # Up
        for i in range(len(out)):
            out[i] = (out[i] + prev[i]) & 0xFF
        return out

    if filter_type == 3:  # Average
        for i in range(len(out)):
            left = out[i - bpp] if i >= bpp else 0
            up = prev[i]
            out[i] = (out[i] + ((left + up) // 2)) & 0xFF
        return out

    if filter_type == 4:  # Paeth
        for i in range(len(out)):
            a = out[i - bpp] if i >= bpp else 0
            b = prev[i]
            c = prev[i - bpp] if i >= bpp else 0
            p = a + b - c
            pa = abs(p - a)
            pb = abs(p - b)
            pc = abs(p - c)
            pr = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
            out[i] = (out[i] + pr) & 0xFF
        return out

    # 未知滤波器，回退原始数据
    return out


def _mask_pixel_on(row: bytearray, x: int, channels: int, color_type: int) -> bool:
    i = x * channels
    if color_type == 0:  # Gray
        return row[i] > 16
    if color_type == 2:  # RGB
        return max(row[i], row[i + 1], row[i + 2]) > 16
    if color_type == 4:  # Gray + Alpha
        return row[i + 1] > 16 and row[i] > 16
    if color_type == 6:  # RGBA
        return row[i + 3] > 16 and max(row[i], row[i + 1], row[i + 2]) > 16
    return False


def _create_guidance_dir() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(GUIDANCE_ROOT, ts)
    os.makedirs(path, exist_ok=True)
    return path


def _write_json(path: str, payload: dict[str, Any]) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


async def _download(url: str, path: str) -> None:
    async with httpx.AsyncClient(timeout=90) as http:
        resp = await http.get(url)
        resp.raise_for_status()
        with open(path, "wb") as f:
            f.write(resp.content)


async def _download_with_retry(url: str, path: str, retries: int = 1) -> dict[str, Any]:
    """
    下载重试封装：
    - 初次尝试 + retries 次重试
    - 返回结构化结果，不抛出异常给调用方
    """
    errors: list[str] = []
    total = max(0, retries) + 1
    for attempt in range(1, total + 1):
        try:
            await _download(url, path)
            return {"ok": True, "attempts": attempt, "errors": errors}
        except Exception as e:
            errors.append(str(e))
    return {"ok": False, "attempts": total, "errors": errors}


def _safe_name(name: str) -> str:
    safe = name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    return "".join(c for c in safe if c.isalnum() or c in {"_", "-"}) or "object"


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default
