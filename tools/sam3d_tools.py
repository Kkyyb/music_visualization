"""
工具：SAM 3D Objects 分割建模 (via fal.ai)
从场景图片中分割物体并生成 3D 模型。
"""
from __future__ import annotations

import os
import httpx
import fal_client

from config import FAL_KEY, FAL_SAM3D_MODEL, MODEL_OUTPUT_DIR
from token_tracker import tracker

os.environ["FAL_KEY"] = FAL_KEY


async def segment_and_model(
    image_path: str,
    object_prompts: list[str],
) -> dict:
    """
    使用 SAM 3D Objects 从图片中分割物体并生成 3D 模型。

    Args:
        image_path: 本地图片路径
        object_prompts: 要分割的物体文字列表, 如 ["guitar", "stage", "lamp"]

    Returns:
        {
            "objects": [
                {
                    "label": str,
                    "glb_path": str,       # 本地 GLB 文件路径
                    "transform": dict,     # 变换矩阵
                },
                ...
            ]
        }
    """
    print(f"  🧩 SAM 3D 分割建模: {object_prompts}")

    # 上传图片到 fal
    image_url = await fal_client.upload_file_async(image_path)

    # 规范化 + 去重，避免无效标签（如 record_player -> record player）
    normalized_prompts = _normalize_object_prompts(object_prompts)
    if not normalized_prompts:
        normalized_prompts = ["object"]
    print(f"  🧭 规范化后标签: {normalized_prompts}")

    # 逐物体调用，兼容当前 fal sam-3/3d-objects 的 prompt 单字段 schema
    result = await _call_sam3d(image_url, normalized_prompts)

    # 下载每个物体的 GLB 模型
    objects = []
    raw_objects = result.get("objects", [])

    for i, obj in enumerate(raw_objects):
        label = obj.get("label") or f"object_{i}"

        # 获取 GLB URL（统一内部对象格式）
        glb_url = obj.get("glb_url")

        if not glb_url:
            print(f"  ⚠️ 物体 '{label}' 未获取到 GLB URL, 跳过")
            continue

        # 下载 GLB
        safe_label = label.replace(" ", "_").replace("/", "_")
        glb_filename = f"sam3d_{safe_label}_{i}.glb"
        glb_path = os.path.join(MODEL_OUTPUT_DIR, glb_filename)

        try:
            async with httpx.AsyncClient(timeout=60) as http:
                resp = await http.get(glb_url)
                resp.raise_for_status()
                with open(glb_path, "wb") as f:
                    f.write(resp.content)
        except Exception as e:
            print(f"  ⚠️ 物体 '{label}' 下载失败: {e}, 跳过")
            continue

        transform = obj.get("transform", {})

        objects.append({
            "label": label,
            "glb_path": glb_path,
            "transform": transform,
        })
        print(f"  ✅ 物体 '{label}' 模型已下载: {glb_path}")

    tracker.record_sam3d(len(objects), "SAM 3D 建模")

    return {"objects": objects, "raw_result": result}


async def _call_sam3d(image_url: str, prompts: list[str]) -> dict:
    """
    调用 SAM 3D API（fal-ai/sam-3/3d-objects）。
    当前 schema 使用单个 `prompt` 字段，因此按物体逐个调用并聚合结果。
    """
    all_objects: list[dict] = []
    raw_results: list[dict] = []

    for prompt in prompts:
        variants = _prompt_variants(prompt)
        success = False

        for variant in variants:
            try:
                r = await _call_sam3d_single_prompt(image_url, variant)
                raw_results.append(r)

                objects = _extract_objects_from_result(r, label=prompt)
                if not objects:
                    print(f"  ⚠️ '{variant}' 未返回可用 GLB，尝试下一个候选")
                    continue

                all_objects.extend(objects)
                print(f"  ✅ '{prompt}' 分割成功（命中: '{variant}'）")
                success = True
                break
            except Exception as e:
                print(f"  ⚠️ '{variant}' 分割失败: {e}")

        if not success:
            print(f"  ⚠️ '{prompt}' 所有候选提示词均失败，跳过")

    return {"objects": all_objects, "raw_results": raw_results}


async def _call_sam3d_single_prompt(image_url: str, prompt: str) -> dict:
    """单个 prompt 调用，按阈值递进重试。"""
    thresholds: list[float | None] = [None, 0.2, 0.1]
    last_error: Exception | None = None

    for threshold in thresholds:
        args: dict = {
            "image_url": image_url,
            "prompt": prompt,
            "export_textured_glb": True,
        }
        if threshold is not None:
            args["detection_threshold"] = threshold

        try:
            if threshold is None:
                print(f"  🔍 尝试分割: '{prompt}' (default threshold)")
            else:
                print(f"  🔍 尝试分割: '{prompt}' (detection_threshold={threshold})")
            return await fal_client.subscribe_async(FAL_SAM3D_MODEL, arguments=args)
        except Exception as e:
            last_error = e

    if last_error is not None:
        raise last_error
    raise RuntimeError("SAM 3D 调用失败: 未知错误")


def _extract_objects_from_result(result: dict, label: str) -> list[dict]:
    """
    解析 fal-ai/sam-3/3d-objects 返回：
    - 优先 individual_glbs（多物体）
    - 回退 model_glb（单物体/合并模型）
    """
    objects: list[dict] = []
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


def _normalize_object_prompts(object_prompts: list[str]) -> list[str]:
    """清洗标签并去重，提升自动分割命中率。"""
    import re

    normalized: list[str] = []
    seen: set[str] = set()
    for raw in object_prompts:
        if not raw:
            continue
        p = raw.strip().lower()
        p = p.replace("_", " ").replace("-", " ")
        p = re.sub(r"[^a-z0-9\s]", " ", p)
        p = re.sub(r"\s+", " ", p).strip()
        if not p:
            continue
        if p not in seen:
            seen.add(p)
            normalized.append(p)
    return normalized


def _prompt_variants(prompt: str) -> list[str]:
    """为单个标签生成候选提示词。第一个始终是原词。"""
    variants: list[str] = [prompt]

    synonyms = {
        "record player": ["turntable", "phonograph"],
        "recordplayer": ["record player", "turntable"],
        "swing": ["hanging chair", "chair"],
        "umbrella": ["parasol"],
        "table": ["desk", "wooden table"],
        "diorama base": ["base", "platform"],
    }
    for candidate in synonyms.get(prompt, []):
        if candidate not in variants:
            variants.append(candidate)

    # recordplayer 这类连写词再兜底一次
    if " " not in prompt and prompt.endswith("player") and "record" in prompt:
        spaced = prompt.replace("record", "record ").strip()
        if spaced not in variants:
            variants.append(spaced)

    return variants
