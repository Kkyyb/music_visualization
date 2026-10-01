"""
Workflow 阶段节点
负责：歌词搜索 → 融合 Prompt → 图片生成 → 图片分析 → SAM 3D 分割建模
"""
from __future__ import annotations

import base64
import json
import re
from state import GraphState, BlenderObject
from config import (
    QWEN_KEY, QWEN_BASE_URL,
    MODEL_CREATIVE, MODEL_VISION,
)
from token_tracker import tracker
from langchain_openai import ChatOpenAI


# ═══════════════════════════════════════════════════════════
# 节点 1：搜索歌词
# ═══════════════════════════════════════════════════════════
async def search_lyrics_node(state: GraphState) -> dict:
    """联网搜索歌词 (若用户未提供)"""
    print("\n" + "=" * 50)
    print("🔍 节点: 歌词搜索")
    print("=" * 50)

    # 如果用户已提供歌词，跳过搜索
    if state.get("lyrics", "").strip():
        print("  ℹ️ 用户已提供歌词，跳过搜索")
        return {}

    from tools.search_tools import search_lyrics
    lyrics, _ = await search_lyrics(state["song_name"], state["artist"])

    return {"lyrics": lyrics}


# ═══════════════════════════════════════════════════════════
# 节点 2：融合 Prompt 生成
# ═══════════════════════════════════════════════════════════
async def generate_prompt_node(state: GraphState) -> dict:
    """将歌曲信息 + 环境 + 心情融合为图片生成 Prompt"""
    print("\n" + "=" * 50)
    print("✍️ 节点: 融合 Prompt 生成")
    print("=" * 50)

    llm = ChatOpenAI(
        model=MODEL_CREATIVE,
        api_key=QWEN_KEY,
        base_url=QWEN_BASE_URL,
    )

    system = """你是一位专业的 AI 艺术指导。
根据用户提供的歌曲信息、环境、心情和歌词，生成一段高质量的英文图片生成 Prompt。

要求：
1. Prompt 必须用英文撰写（英文质量更好）
2. 融合歌曲情感、环境氛围、听者心情为一个统一的视觉场景
3. 包含具体的视觉元素描述：光照、色调、材质、构图
4. 包含氛围关键词和艺术风格参考
5. 必须采用 isometric / axonometric diorama composition（45-degree bird’s-eye, top-down oblique view）：场景主体是一个独立底座（可为沙盘、浮岛、平台、地形基座等），所有物体都放置在底座上
6. Hard framing constraints (non-negotiable):
   - The entire base platform and the complete diorama silhouette must be fully visible inside the frame
   - The outer bounding box of the base must leave a clear 12-15% empty margin on all four sides
   - The base platform itself may not touch, crop, or cross any image border
   - No object, ornament, shadow, reflection, glow, or splash may touch or cross the image border
   - All geometry must stay strictly within the base footprint, with visible breathing room from the base edge
   - Keep the camera pulled back to a medium-long distance; if composition becomes tight, zoom out instead of enlarging the base
   - Base footprint should read as about 60-68% of the frame, never filling the canvas
7. 物体复杂度必须受控（Object Budget）：1 个主视觉大件 + 2-4 个次级大件 + 0-1 个小件点缀；禁止大量零碎小物件和密集装饰
8. 构图不必固定为“正方形房间 + 两面墙 + 地板”的 cutaway 模式；应根据提供的歌曲信息、环境、心情和歌词选择最优的构图方式
9. 场景中应包含可被分割建模的具体物体，不能出现人物；优先选择轮廓清晰、体块明确、便于单独建模的物体
10. 场景背景应该纯净，易与主体模型（底座+物体）清晰地区分开来
11. Camera and lens constraints: orthographic-like or telephoto compression, minimal perspective distortion, no depth of field, sharp focus across the whole scene
12. 长度控制在 120-200 词，表达精炼，避免过多微观细节描述
13. 只返回 Prompt 本身，不要额外说明"""

    user_content = f"""歌曲: 《{state['song_name']}》 - {state['artist']}
环境场景: {state['scene']}
心情: {state['mood']}
歌词摘要: {state['lyrics'][:500]}"""

    response = await llm.ainvoke([
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ])

    meta = response.usage_metadata or {}
    tracker.record_llm(
        node_name="融合Prompt生成",
        model=MODEL_CREATIVE,
        input_tokens=meta.get("input_tokens", 0),
        output_tokens=meta.get("output_tokens", 0),
    )

    image_prompt = response.content.strip()
    print(f"  📝 生成 Prompt:\n{image_prompt[:200]}...")

    return {"image_prompt": image_prompt}


# ═══════════════════════════════════════════════════════════
# 节点 3：图片生成
# ═══════════════════════════════════════════════════════════
async def generate_image_node(state: GraphState) -> dict:
    """使用 FLUX 2 Pro 生成场景图片"""
    print("\n" + "=" * 50)
    print("🎨 节点: 图片生成 (FLUX 2 Pro)")
    print("=" * 50)

    from tools.image_tools import generate_image

    base_prompt = (state.get("image_prompt") or "").strip()
    if not base_prompt:
        raise ValueError("image_prompt 为空，无法生成图片")

    # 第 1 次使用原始 Prompt；后续重试叠加更强约束，提升稳定性
    retry_suffixes = [
        "",
        "\n\nStrict regeneration constraints:\n"
        "- The base platform itself must remain fully inside frame with a visible 12-15% empty margin on every side.\n"
        "- Zoom the camera out if necessary; do not enlarge the base to fill the canvas.\n"
        "- No object, prop, base edge, shadow, reflection, or glow may touch image borders.\n"
        "- Keep all objects inset from the base edge with clear breathing room.\n"
        "- Simplify composition: keep only major masses, remove tiny clutter.\n"
        "- If tree elements exist, use a compact trimmed canopy entirely within base bounds.",
        "\n\nCritical hard constraints (must obey):\n"
        "- Treat the outer 12% of the frame on all four sides as a forbidden empty margin.\n"
        "- The base outer silhouette must stay entirely within the safe central area.\n"
        "- Zero border contact for base, objects, shadows, reflections, and all visible geometry.\n"
        "- Use a smaller overall composition with more background around it.\n"
        "- Reduce scene to large readable blocks only; no micro-details.\n"
        "- Prefer open-base composition; avoid cutaway room template.",
    ]

    last_path = ""
    last_reason = "unknown"
    for attempt, suffix in enumerate(retry_suffixes, start=1):
        trial_prompt = base_prompt if not suffix else f"{base_prompt}{suffix}"
        print(f"  🔁 生成尝试 {attempt}/{len(retry_suffixes)}")
        local_path = await generate_image(
            prompt=trial_prompt,
            filename="scene.png",
        )
        last_path = local_path

        passed, reason = await _quality_gate_image(local_path, state)
        if passed:
            print(f"  ✅ 质检通过（attempt {attempt}）")
            return {"image_local_path": local_path}

        last_reason = reason
        print(f"  ⚠️ 质检未通过（attempt {attempt}）: {reason}")

    print(f"  ⚠️ 所有重试均未通过质检，返回最后一次结果: {last_reason}")
    return {"image_local_path": last_path}


async def _quality_gate_image(image_local_path: str, state: GraphState) -> tuple[bool, str]:
    """
    视觉质检：检查是否存在触边/越界，以及是否出现明显碎物体堆积。
    输出 PASS 才视为通过。
    """
    llm = ChatOpenAI(
        model=MODEL_VISION,
        api_key=QWEN_KEY,
        base_url=QWEN_BASE_URL,
    )

    with open(image_local_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    system = """你是严格的图像构图质检器。你只负责判定是否通过，不负责创作。
检查项：
1) 是否有任何主体物体触碰/越过画面边缘
2) 是否存在明显越出底座的树枝/树叶/细长结构
3) 是否出现明显小物体过多、碎细节过密（影响后续分割建模）
4) 是否偏向“固定房间切角模板（两面墙+地板）”且违背开放式底座倾向

输出必须是单行 JSON，不要 Markdown，不要解释文本：
{"pass": true/false, "reason": "short reason in Chinese"}"""

    quality_gate_system = """你是严格的图像构图质检器。你只负责判定是否通过，不负责创作。
检查项：
1) base platform 是否完整地出现在画面内，四边是否都有明显留白，留白至少约 10-12%
2) base 外轮廓是否过大，是否看起来几乎铺满画面
3) 是否有任何主体物体、底座边缘触碰或越过画面边缘
4) 是否存在明显越出底座的树枝、树叶或细长结构
5) 是否出现明显小物件过多、细节过密，影响后续分割建模
6) 是否偏向“固定房间 cutaway 模板（两面墙+地板）”而违背开放式底座构图

判定规则：
- 只要 base platform 任一侧离画面边缘过近、没有明确留白，就必须判定 pass=false
- 只要底座或主体看起来“几乎铺满画面”，即使没有真实裁切，也必须判定 pass=false
- 对边距不足要从严，不要放过

输出必须是单行 JSON，不要 Markdown，不要解释文本：
{"pass": true/false, "reason": "short reason in Chinese"}"""

    user_text = (
        f"歌曲: 《{state.get('song_name', '')}》 - {state.get('artist', '')}; "
        f"场景: {state.get('scene', '')}; 心情: {state.get('mood', '')}. "
        "请按检查项严格判定。"
    )

    response = await llm.ainvoke([
        {"role": "system", "content": quality_gate_system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            ],
        },
    ])

    meta = response.usage_metadata or {}
    tracker.record_llm(
        node_name="图片质检",
        model=MODEL_VISION,
        input_tokens=meta.get("input_tokens", 0),
        output_tokens=meta.get("output_tokens", 0),
    )

    content = _resp_text(response.content)
    parsed = _parse_quality_json(content)
    if parsed is None:
        print(f"  ⚠️ 质检结果解析失败，按不通过处理: {content[:160]}")
        return False, "质检返回格式异常"
    return parsed["pass"], parsed["reason"]


def _resp_text(content: object) -> str:
    """兼容字符串或分块列表的消息内容。"""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()
    return str(content).strip()


def _parse_quality_json(text: str) -> dict | None:
    """从模型返回文本中提取质量判定 JSON。"""
    try:
        obj = json.loads(text)
        return _normalize_quality_payload(obj)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return _normalize_quality_payload(obj)
    except Exception:
        return None


def _parse_json_object(text: str) -> dict | None:
    """从文本中提取单个 JSON 对象。"""
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _normalize_quality_payload(obj: object) -> dict | None:
    if not isinstance(obj, dict):
        return None
    passed = obj.get("pass")
    reason = obj.get("reason", "")
    if not isinstance(passed, bool):
        return None
    if not isinstance(reason, str):
        reason = str(reason)
    return {"pass": passed, "reason": reason.strip() or ("通过" if passed else "未通过")}


# ═══════════════════════════════════════════════════════════
# 节点 4：图片分析 → 场景任务书
# ═══════════════════════════════════════════════════════════
async def analyze_image_node(state: GraphState) -> dict:
    """
    用 Qwen-VL 分析图片:
    仅生成场景目标任务书 (动态效果、氛围等)。
    """
    print("\n" + "=" * 50)
    print("👁️ 节点: 图片分析")
    print("=" * 50)

    llm = ChatOpenAI(
        model=MODEL_VISION,
        api_key=QWEN_KEY,
        base_url=QWEN_BASE_URL,
    )

    # 读取图片为 base64
    with open(state["image_local_path"], "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    system = """你是一位经验丰富的3D场景策划师。分析提供的图片，输出“场景目标任务书”。

任务书必须包含：
1) 整体氛围和色调目标
2) 需要添加的灯光效果（类型、颜色、强度）
3) 需要添加的动态效果（粒子系统、烟雾、光晕等）
4) 相机运动建议
5) 材质和纹理优化方向

注意：
- 这里只写任务书，不要输出 OBJECTS 清单
- 输出中文，结构清晰"""

    response = await llm.ainvoke([
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"歌曲: 《{state['song_name']}》- {state['artist']}，环境: {state['scene']}，心情: {state['mood']}"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                },
            ],
        },
    ])

    meta = response.usage_metadata or {}
    tracker.record_llm(
        node_name="图片分析",
        model=MODEL_VISION,
        input_tokens=meta.get("input_tokens", 0),
        output_tokens=meta.get("output_tokens", 0),
    )

    content = _resp_text(response.content)
    print(f"  📋 分析结果:\n{content[:300]}...")

    return {
        "scene_task_doc": content,
    }


# ═══════════════════════════════════════════════════════════
# 节点 5：生成 SAM 分割计划
# ═══════════════════════════════════════════════════════════
async def plan_sam_segmentation_node(state: GraphState) -> dict:
    """根据图片与任务书生成结构化分割计划 JSON。"""
    print("\n" + "=" * 50)
    print("🗺️ 节点: SAM 分割计划")
    print("=" * 50)

    llm = ChatOpenAI(
        model=MODEL_VISION,
        api_key=QWEN_KEY,
        base_url=QWEN_BASE_URL,
    )

    with open(state["image_local_path"], "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    system = """你是 3D 重建分割规划器。请根据图片和任务书输出一个 JSON 计划。
只输出 JSON，不要 Markdown，不要额外解释。

JSON Schema:
{
  "version": "1.0",
  "global_policy": {
    "max_objects": 10,
    "batch_order": ["anchor", "secondary", "optional"],
    "default_thresholds": [null, 0.2, 0.1]
  },
  "objects": [
    {
      "id": "base",
      "label": "base",
      "batch": "anchor",
      "priority": 1,
      "method": "prompt",
      "prompt": "diorama base",
      "box_prompt": null,
      "mask_url": null,
      "retry_methods": ["prompt", "box_prompts", "mask_urls"],
      "depends_on": []
    }
  ]
}

depends_on 填写规则：
- 表示该物体底部实际接触并依附的对象 id。
- 物体底部接触什么，就 depends_on 什么。
- 若物体直接落在基座/地面上，则 depends_on 填该基座对象 id；只有最底层承载物（通常是 base）填写空数组 []。
- 该字段只填写直接接触的父对象，不要填写间接支撑链路，不要同时依赖多个不必要对象。
- 例如：base 的 depends_on = []；chair 放在 base 上，则 chair 的 depends_on = ["base"]；radiogram 放在 chair 上，则 radiogram 的 depends_on = ["chair"]；street lamp 立在 base 上，则 street lamp 的 depends_on = ["base"]。

规划规则：
1) 总对象数 4-10，优先大件主体（base、主建筑、主树体、主家具），且在可见时必须纳入可见数量的小件物体。
2) 对遮挡或重叠严重的对象，method 优先用 box_prompts；极难分割对象可用 mask_urls。
3) 若出现乔木（arbor tree），默认作为单对象 tree 建模；只有在树干与叶片边界非常清晰时才拆分 wood/leaves。
4) 灌木、花草、地被植物默认不强制分层，通常直接作为单对象建模。
5) label 统一英文短词，1-3 个单词，便于后续建模和命名。
6) batch 顺序遵循 anchor -> secondary -> optional。
7) 必须正确填写 depends_on：按“物体底部接触什么，就 depends_on 什么”的原则填写；最底层 base 填 []，放在 base 上的对象填 ["base"]，放在椅子/桌面/台面上的对象填对应承载物 id。"""

    user_text = (
        f"歌曲: 《{state.get('song_name', '')}》 - {state.get('artist', '')}\n"
        f"场景: {state.get('scene', '')}\n"
        f"心情: {state.get('mood', '')}\n"
        f"任务书:\n{state.get('scene_task_doc', '')[:2000]}"
    )

    response = await llm.ainvoke([
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            ],
        },
    ])

    meta = response.usage_metadata or {}
    tracker.record_llm(
        node_name="SAM计划生成",
        model=MODEL_VISION,
        input_tokens=meta.get("input_tokens", 0),
        output_tokens=meta.get("output_tokens", 0),
    )

    content = _resp_text(response.content)
    plan = _parse_json_object(content) or {}

    # 小物体补漏：当主计划中小件不足时，额外做一次“仅补小件”识别
    existing_labels = []
    if isinstance(plan.get("objects"), list):
        existing_labels = [
            str(x.get("label") or "").strip().lower()
            for x in plan.get("objects", [])
            if isinstance(x, dict)
        ]

    if _count_small_object_labels(existing_labels) < 2:
        supplements = await _supplement_small_objects_plan(
            llm=llm,
            img_b64=img_b64,
            user_text=user_text,
            existing_labels=existing_labels,
        )
        if supplements:
            plan = _merge_plan_objects(plan, supplements)
            print(f"  ➕ 小件补漏: {len(supplements)}")

    plan = _normalize_segmentation_plan(plan)
    print(f"  📦 分割计划对象数: {len(plan.get('objects', []))}")

    return {"sam_segmentation_plan": plan}


# ═══════════════════════════════════════════════════════════
# 节点 6：SAM 3D 分割建模
# ═══════════════════════════════════════════════════════════
async def sam3d_modeling_node(state: GraphState) -> dict:
    """按分割计划使用 SAM 3D Objects 生成 3D 模型"""
    print("\n" + "=" * 50)
    print("🧩 节点: SAM 3D 分割建模")
    print("=" * 50)

    plan = state.get("sam_segmentation_plan") or {}
    if not plan:
        print("  ⚠️ 未发现分割计划，使用退化策略构造最小计划")
        fallback = _extract_objects(state.get("scene_task_doc", ""))
        if not fallback:
            fallback = ["base", "main object", "secondary object"]
        plan = _build_fallback_plan(fallback)

    from tools.sam3d_plan_executor import execute_sam_segmentation_plan
    result = await execute_sam_segmentation_plan(
        image_path=state["image_local_path"],
        plan=plan,
    )

    # 转换为 BlenderObject 列表
    blender_objects = []
    for obj in result.get("objects", []):
        blender_objects.append(
            BlenderObject(
                name="",           # 导入后再填
                source_file=obj["glb_path"],
                label=obj["label"],
            )
        )

    return {
        "sam3d_result": result,
        "blender_objects": blender_objects,
    }


def _build_fallback_plan(labels: list[str]) -> dict:
    objects = []
    for i, label in enumerate(labels[:8], start=1):
        batch = "anchor" if i <= 3 else "secondary"
        objects.append(
            {
                "id": f"obj_{i}",
                "label": label,
                "batch": batch,
                "priority": i,
                "method": "prompt",
                "prompt": label,
                "box_prompt": None,
                "mask_url": None,
                "retry_methods": ["prompt", "box_prompts", "mask_urls"],
                "depends_on": [],
            }
        )
    return {
        "version": "1.0",
        "global_policy": {
            "max_objects": 10,
            "batch_order": ["anchor", "secondary", "optional"],
            "default_thresholds": [None, 0.2, 0.1],
        },
        "objects": objects,
    }


def _normalize_segmentation_plan(plan: dict) -> dict:
    if not isinstance(plan, dict):
        return _build_fallback_plan(["base", "main object", "secondary object"])

    objects_raw = plan.get("objects")
    if not isinstance(objects_raw, list):
        objects_raw = []

    norm_objects = []
    for i, item in enumerate(objects_raw, start=1):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip().lower()
        if not label:
            continue
        method = str(item.get("method") or "prompt").strip()
        if method not in {"prompt", "box_prompts", "mask_urls"}:
            method = "prompt"
        batch = str(item.get("batch") or "secondary").strip()
        if batch not in {"anchor", "secondary", "optional"}:
            batch = "secondary"

        priority = _safe_int(item.get("priority"), i)
        retry_methods = _normalize_retry_methods(item.get("retry_methods"), default_method=method)
        norm_objects.append(
            {
                "id": str(item.get("id") or f"obj_{i}"),
                "label": label,
                "batch": batch,
                "priority": priority,
                "method": method,
                "prompt": item.get("prompt") or label,
                "box_prompt": item.get("box_prompt"),
                "mask_url": item.get("mask_url"),
                "retry_methods": retry_methods,
                "depends_on": item.get("depends_on") or [],
            }
        )

    if not norm_objects:
        return _build_fallback_plan(["base", "main object", "secondary object"])

    # 植被策略：
    # - 乔木默认整体 tree 建模（稳定优先）
    # - 灌木/花草：默认单层
    norm_objects = _apply_plant_layering_strategy(norm_objects)
    norm_objects = _dedupe_plan_objects(norm_objects)

    policy = plan.get("global_policy", {})
    if not isinstance(policy, dict):
        policy = {}
    max_objects = _safe_int(policy.get("max_objects"), 8)
    max_objects = max(8, min(max_objects, 12))
    norm_objects = norm_objects[:max_objects]

    thresholds = policy.get("default_thresholds")
    if not isinstance(thresholds, list) or not thresholds:
        thresholds = [None, 0.2, 0.1]

    batch_order = policy.get("batch_order")
    if not isinstance(batch_order, list) or not batch_order:
        batch_order = ["anchor", "secondary", "optional"]

    return {
        "version": str(plan.get("version", "1.0")),
        "global_policy": {
            "max_objects": max_objects,
            "batch_order": batch_order,
            "default_thresholds": thresholds,
        },
        "objects": norm_objects,
    }


def _safe_int(value: object, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except Exception:
        return default


def _apply_plant_layering_strategy(objects: list[dict]) -> list[dict]:
    out: list[dict] = []
    for obj in objects:
        label = str(obj.get("label", "")).strip().lower()
        if not label:
            continue

        # 灌木/花草保持单层，避免过度拆分
        if _is_small_plant_label(label):
            out.append(_with_method_policy(obj, method="prompt", retry_methods=["prompt", "box_prompts", "mask_urls"]))
            continue

        # 树木对象默认整体建模，避免 wood/leaves 双层都“抓整树”
        if _is_tree_leaves_label(label) or _is_tree_wood_label(label) or _is_arbor_tree_label(label):
            item = dict(obj)
            item["label"] = "tree"
            item["prompt"] = "single complete tree with trunk, branches, and leaves"
            out.append(_with_method_policy(item, method="prompt", retry_methods=["prompt", "box_prompts", "mask_urls"]))
            continue

        out.append(obj)

    return out


def _with_method_policy(obj: dict, method: str, retry_methods: list[str]) -> dict:
    item = dict(obj)
    item["method"] = method
    item["retry_methods"] = retry_methods
    return item


def _is_arbor_tree_label(label: str) -> bool:
    # 乔木判定：包含 tree 或常见树种词，不是灌木/花草，也不是已拆分 trunk/leaves/canopy
    tree_species = ("oak", "pine", "maple", "banyan", "willow", "cypress", "spruce")
    if "tree" not in label and not any(s in label for s in tree_species):
        return False
    if _is_small_plant_label(label):
        return False
    if _is_tree_wood_label(label) or _is_tree_leaves_label(label):
        return False
    return True


def _is_tree_wood_label(label: str) -> bool:
    keys = ("trunk", "branch", "wood structure", "tree wood")
    return any(k in label for k in keys)


def _is_tree_leaves_label(label: str) -> bool:
    keys = ("canopy", "leaf", "leaves", "foliage")
    return any(k in label for k in keys)


def _is_small_plant_label(label: str) -> bool:
    keys = ("shrub", "bush", "flower", "flowers", "grass", "weed", "potted plant", "plant pot")
    return any(k in label for k in keys)


def _normalize_retry_methods(raw: object, default_method: str) -> list[str]:
    allow = ("prompt", "box_prompts", "mask_urls")
    methods: list[str] = []
    if isinstance(raw, list):
        for x in raw:
            m = str(x).strip()
            if m in allow and m not in methods:
                methods.append(m)
    if default_method in allow and default_method not in methods:
        methods.insert(0, default_method)
    if not methods:
        methods = [default_method if default_method in allow else "prompt"]
    for m in allow:
        if m not in methods:
            methods.append(m)
    return methods


def _normalize_label_for_key(label: str) -> str:
    norm = re.sub(r"[^a-z0-9\s]+", " ", label.lower())
    norm = re.sub(r"\s+", " ", norm).strip()
    if _is_tree_leaves_label(norm) or _is_tree_wood_label(norm) or _is_arbor_tree_label(norm):
        return "tree"
    return norm


def _batch_rank_for_dedupe(batch: str) -> int:
    if batch == "anchor":
        return 0
    if batch == "secondary":
        return 1
    if batch == "optional":
        return 2
    return 3


def _object_rank_for_dedupe(obj: dict) -> tuple[int, int]:
    return (
        _batch_rank_for_dedupe(str(obj.get("batch") or "secondary")),
        _safe_int(obj.get("priority"), 999),
    )


def _merge_retry_methods(base: list[str], incoming: list[str]) -> list[str]:
    out: list[str] = []
    for m in base + incoming:
        if m in {"prompt", "box_prompts", "mask_urls"} and m not in out:
            out.append(m)
    if not out:
        out = ["prompt", "box_prompts", "mask_urls"]
    return out


def _dedupe_plan_objects(objects: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    order: list[str] = []

    for obj in objects:
        label = str(obj.get("label") or "").strip()
        if not label:
            continue
        key = _normalize_label_for_key(label)
        if not key:
            continue

        if key not in seen:
            item = dict(obj)
            item["retry_methods"] = _normalize_retry_methods(item.get("retry_methods"), str(item.get("method") or "prompt"))
            seen[key] = item
            order.append(key)
            continue

        old = seen[key]
        new = dict(obj)
        old_rank = _object_rank_for_dedupe(old)
        new_rank = _object_rank_for_dedupe(new)
        keep = old if old_rank <= new_rank else new
        drop = new if keep is old else old

        keep_retry = _normalize_retry_methods(keep.get("retry_methods"), str(keep.get("method") or "prompt"))
        drop_retry = _normalize_retry_methods(drop.get("retry_methods"), str(drop.get("method") or "prompt"))
        keep["retry_methods"] = _merge_retry_methods(keep_retry, drop_retry)

        if not keep.get("box_prompt") and drop.get("box_prompt"):
            keep["box_prompt"] = drop.get("box_prompt")
        if not keep.get("mask_url") and drop.get("mask_url"):
            keep["mask_url"] = drop.get("mask_url")
        if (not keep.get("prompt")) and drop.get("prompt"):
            keep["prompt"] = drop.get("prompt")

        seen[key] = keep

    return [seen[k] for k in order if k in seen]


def _is_small_object_label(label: str) -> bool:
    keys = (
        "book", "books", "vinyl", "record", "umbrella", "cup", "mug", "bottle",
        "lamp", "lantern", "speaker", "radio", "clock", "vase", "shoe", "shoes",
        "sandal", "bag", "hat", "camera", "frame", "plant pot", "pillow",
    )
    label_l = label.lower()
    return any(k in label_l for k in keys)


def _count_small_object_labels(labels: list[str]) -> int:
    count = 0
    for label in labels:
        if _is_small_object_label(label):
            count += 1
    return count


async def _supplement_small_objects_plan(
    llm: ChatOpenAI,
    img_b64: str,
    user_text: str,
    existing_labels: list[str],
) -> list[dict]:
    system = """你是“分割计划补漏器”。你的唯一目标是补充可见小物体。
只输出 JSON，不要 Markdown，不要解释。

JSON Schema:
{
  "objects": [
    {"label": "book", "prompt": "book on table"},
    {"label": "vinyl record", "prompt": "vinyl record disc"}
  ]
}

约束：
1) 只输出图中清晰可见、可单独建模的小物体，最多 4 个。
2) 不要输出 base/house/tree/terrain/furniture 这类大件。
3) label 必须是英文 1-3 词；若与已有对象重复则不要再输出。"""
    user_prompt = (
        f"{user_text}\n"
        f"已有对象: {', '.join(existing_labels) if existing_labels else '(none)'}\n"
        "请只补充遗漏的小件。"
    )

    response = await llm.ainvoke([
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            ],
        },
    ])

    meta = response.usage_metadata or {}
    tracker.record_llm(
        node_name="SAM小件补漏",
        model=MODEL_VISION,
        input_tokens=meta.get("input_tokens", 0),
        output_tokens=meta.get("output_tokens", 0),
    )

    parsed = _parse_json_object(_resp_text(response.content)) or {}
    objs = parsed.get("objects")
    if not isinstance(objs, list):
        return []

    existing_keys = {_normalize_label_for_key(x) for x in existing_labels if x}
    out: list[dict] = []
    seq = 1
    for item in objs:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip().lower()
        if not label or not _is_small_object_label(label):
            continue
        key = _normalize_label_for_key(label)
        if not key or key in existing_keys:
            continue
        existing_keys.add(key)

        prompt = str(item.get("prompt") or label).strip() or label
        out.append(
            {
                "id": f"small_prop_{seq}",
                "label": label,
                "batch": "optional",
                "priority": 90 + seq,
                "method": "prompt",
                "prompt": prompt,
                "box_prompt": None,
                "mask_url": None,
                "retry_methods": ["prompt", "box_prompts", "mask_urls"],
                "depends_on": [],
            }
        )
        seq += 1
        if len(out) >= 4:
            break
    return out


def _merge_plan_objects(plan: dict, supplements: list[dict]) -> dict:
    merged = dict(plan) if isinstance(plan, dict) else {}
    objects = merged.get("objects")
    if not isinstance(objects, list):
        objects = []
    merged["objects"] = objects + supplements

    policy = merged.get("global_policy")
    if not isinstance(policy, dict):
        policy = {}
    old_max = _safe_int(policy.get("max_objects"), 8)
    policy["max_objects"] = max(old_max, min(12, len(merged["objects"])))
    merged["global_policy"] = policy
    return merged


def _extract_objects(text: str, max_count: int = 5) -> list[str]:
    """从分析文本中提取 OBJECTS 行（兼容 Markdown 加粗），限制数量并清洗标签"""

    # 兼容:
    # OBJECTS: ...
    # **OBJECTS**: ...
    # - **OBJECTS**: ...
    # OBJECTS：... (全角冒号)
    objects_line_pattern = re.compile(
        r"^\s*(?:[-*]\s*)?(?:\*\*)?\s*OBJECTS\s*(?:\*\*)?\s*[:：]\s*(.+?)\s*$",
        re.IGNORECASE,
    )

    for line in text.splitlines():
        match = objects_line_pattern.match(line)
        if not match:
            continue

        raw = match.group(1).strip()
        # 兼容中英文逗号、分号
        parts = re.split(r"[,，;；]", raw)
        items = []
        for o in parts:
            # 清洗：去除括号内容、序号、Markdown 标记和特殊字符，只保留英文单词
            cleaned = re.sub(r"\(.*?\)", "", o).strip()
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", cleaned).strip()
            cleaned = cleaned.replace("*", "")
            cleaned = re.sub(r"[^a-zA-Z\s]", "", cleaned).strip()
            cleaned = re.sub(r"\s+", " ", cleaned)
            if cleaned and len(cleaned) < 30:
                items.append(cleaned)
        return items[:max_count]
    return []
