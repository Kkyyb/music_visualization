r"""
分阶段调试入口（不修改原有业务文件）

能力：
1. 按节点阶段执行（可指定 start/end）
2. 在关键阶段暂停（默认 generate_image / sam3d_modeling）
3. 每阶段自动保存快照 JSON，后续可从快照继续
4. 可选重跑当前阶段，便于调参和质量对比

示例：
    # 从头跑到生成图片，并在 generate_image / sam3d_modeling 阶段暂停
    .venv\\Scripts\python.exe debug_workflow_run.py --end-stage sam3d_modeling

    # 从已有快照继续跑
    .venv\Scripts\python.exe .\debug_workflow_run.py --snapshot-in .\debug_snapshots\20260415_155213_analyze_image.json --start-stage plan_sam_segmentation --end-stage plan_sam_segmentation

"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from nodes.workflow_nodes import (
    search_lyrics_node,
    generate_prompt_node,
    generate_image_node,
    analyze_image_node,
    plan_sam_segmentation_node,
    sam3d_modeling_node,
)
from nodes.agent_nodes import (
    generate_task_list_node,
)
from state import BlenderObject, GraphState, TokenUsage


STAGE_ORDER = [
    "search_lyrics",
    "generate_prompt",
    "generate_image",
    "analyze_image",
    "plan_sam_segmentation",
    "sam3d_modeling",
    "generate_task_list",
]

STAGE_FUNCS = {
    "search_lyrics": search_lyrics_node,
    "generate_prompt": generate_prompt_node,
    "generate_image": generate_image_node,
    "analyze_image": analyze_image_node,
    "plan_sam_segmentation": plan_sam_segmentation_node,
    "sam3d_modeling": sam3d_modeling_node,
    "generate_task_list": generate_task_list_node,
}

DEFAULT_PAUSE_STAGES = {"generate_image", "sam3d_modeling"}
SNAPSHOT_DIR = "debug_snapshots"


def _default_state() -> GraphState:
    return {
        "song_name": "",
        "artist": "",
        "scene": "",
        "mood": "",
        "lyrics": "",
        "image_prompt": "",
        "image_url": "",
        "image_local_path": "",
        "scene_task_doc": "",
        "sam_segmentation_plan": {},
        "sam3d_result": {},
        "blender_objects": [],
        "scene_info": "",
        "task_plan": {},
        "planning_context": {},
        "current_master_task_id": "",
        "current_subtask_id": "",
        "current_action": {},
        "current_command": "",
        "agent_execution_status": "",
        "subtask_memory": {},
        "blender_screenshot": "",
        "execution_log": [],
        "token_log": [],
        "messages": [],
    }


def _serialize(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    return obj


def _hydrate_models(state: dict[str, Any]) -> GraphState:
    hydrated = _default_state()
    hydrated.update(state)

    hydrated["blender_objects"] = [
        o if isinstance(o, BlenderObject) else BlenderObject(**o)
        for o in hydrated.get("blender_objects", [])
    ]
    hydrated["token_log"] = [
        t if isinstance(t, TokenUsage) else TokenUsage(**t)
        for t in hydrated.get("token_log", [])
    ]
    return hydrated


def _merge_state(state: GraphState, updates: dict[str, Any]) -> GraphState:
    """尽量模拟 Graph reducer 行为，保证离线分阶段执行时状态一致。"""
    merged = dict(state)
    for key, value in updates.items():
        if key in ("execution_log", "token_log", "messages"):
            merged[key] = list(merged.get(key, [])) + list(value)
            continue
        merged[key] = value
    return _hydrate_models(merged)


def _save_snapshot(state: GraphState, stage_name: str) -> str:
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SNAPSHOT_DIR, f"{ts}_{stage_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_serialize(state), f, ensure_ascii=False, indent=2)
    return path


def _load_snapshot(path: str) -> GraphState:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return _hydrate_models(data)


def _choose_range(start_stage: str, end_stage: str) -> list[str]:
    if start_stage not in STAGE_ORDER:
        raise ValueError(f"未知 start_stage: {start_stage}")
    if end_stage not in STAGE_ORDER:
        raise ValueError(f"未知 end_stage: {end_stage}")
    s = STAGE_ORDER.index(start_stage)
    e = STAGE_ORDER.index(end_stage)
    if s > e:
        raise ValueError("start_stage 必须早于或等于 end_stage")
    return STAGE_ORDER[s : e + 1]


def _ensure_user_inputs(state: GraphState):
    if not state.get("song_name"):
        state["song_name"] = input("🎵 歌曲名: ").strip()
    if not state.get("artist"):
        state["artist"] = input("🎤 歌手: ").strip()
    if not state.get("scene"):
        state["scene"] = input("🌃 环境场景: ").strip() or "默认室内环境"
    if not state.get("mood"):
        state["mood"] = input("💭 心情: ").strip() or "平静"

    if not state["song_name"] or not state["artist"]:
        raise ValueError("歌曲名和歌手不能为空")


def _print_stage_summary(stage: str, state: GraphState):
    print("\n" + "=" * 66)
    print(f"✅ 阶段完成: {stage}")
    if stage == "generate_image":
        print(f"   image_local_path: {state.get('image_local_path', '')}")
    elif stage == "analyze_image":
        preview = (state.get("scene_task_doc", "") or "").strip().replace("\n", " ")
        print(f"   scene_task_doc预览: {preview[:220]}...")
    elif stage == "plan_sam_segmentation":
        plan = state.get("sam_segmentation_plan", {}) or {}
        obj_count = len(plan.get("objects", [])) if isinstance(plan, dict) else 0
        print(f"   SAM计划对象数: {obj_count}")
    elif stage == "sam3d_modeling":
        objs = state.get("blender_objects", [])
        print(f"   生成模型数: {len(objs)}")
        for i, obj in enumerate(objs):
            print(f"   - [{i}] {obj.label} -> {obj.source_file}")
    elif stage == "generate_task_list":
        plan = state.get("task_plan", {}) or {}
        master_tasks = plan.get("master_tasks", []) if isinstance(plan, dict) else []
        print(f"   master_tasks数量: {len(master_tasks)}")
        for task in master_tasks[:4]:
            subtasks = task.get("subtasks", [])
            print(f"   - {task.get('id', '')}: {len(subtasks)} 个 subtasks")
    else:
        keys = [
            "lyrics",
            "image_prompt",
            "image_local_path",
            "scene_task_doc",
            "sam_segmentation_plan",
            "sam3d_result",
        ]
        changed = [k for k in keys if state.get(k)]
        print(f"   关键字段已就绪: {changed}")
    print("=" * 66)


def _prompt_after_pause() -> str:
    print("\n暂停中：")
    print("  [c] 继续到下一阶段")
    print("  [r] 重跑当前阶段")
    print("  [q] 保存并退出")
    choice = input("请选择 (c/r/q): ").strip().lower() or "c"
    return choice


async def _run_stages(
    state: GraphState,
    stages: list[str],
    pause_stages: set[str],
    pause_all: bool,
) -> GraphState:
    for stage in stages:
        func = STAGE_FUNCS[stage]
        while True:
            print("\n" + "#" * 72)
            print(f"🚀 执行阶段: {stage}")
            print("#" * 72)
            updates = await func(state)
            if not isinstance(updates, dict):
                updates = {}
            state = _merge_state(state, updates)
            _print_stage_summary(stage, state)

            snapshot = _save_snapshot(state, stage)
            print(f"💾 快照已保存: {snapshot}")

            need_pause = pause_all or (stage in pause_stages)
            if not need_pause:
                break

            choice = _prompt_after_pause()
            if choice == "r":
                print("🔄 将重跑当前阶段...")
                continue
            if choice == "q":
                print("🛑 调试已暂停，你可以下次从该快照继续。")
                return state
            break
    return state


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="音乐可视化项目分阶段调试入口")
    parser.add_argument("--snapshot-in", default="", help="从已有快照恢复（JSON）")
    parser.add_argument("--start-stage", default="search_lyrics", choices=STAGE_ORDER)
    parser.add_argument("--end-stage", default="sam3d_modeling", choices=STAGE_ORDER)
    parser.add_argument(
        "--pause-stages",
        default="generate_image,sam3d_modeling",
        help="逗号分隔，例如 generate_image,sam3d_modeling",
    )
    parser.add_argument("--pause-all", action="store_true", help="每个阶段都暂停")

    # 可选快捷注入（便于直接从中间阶段起跑）
    parser.add_argument("--song-name", default="")
    parser.add_argument("--artist", default="")
    parser.add_argument("--scene", default="")
    parser.add_argument("--mood", default="")
    parser.add_argument("--lyrics", default="")
    parser.add_argument("--image-local-path", default="")
    parser.add_argument("--scene-task-doc", default="")
    return parser.parse_args()


async def _amain():
    args = _parse_args()

    if args.snapshot_in:
        state = _load_snapshot(args.snapshot_in)
        print(f"📂 已加载快照: {args.snapshot_in}")
    else:
        state = _default_state()

    # CLI 参数可覆盖快照内容，方便快速试验
    if args.song_name:
        state["song_name"] = args.song_name
    if args.artist:
        state["artist"] = args.artist
    if args.scene:
        state["scene"] = args.scene
    if args.mood:
        state["mood"] = args.mood
    if args.lyrics:
        state["lyrics"] = args.lyrics
    if args.image_local_path:
        state["image_local_path"] = args.image_local_path
    if args.scene_task_doc:
        state["scene_task_doc"] = args.scene_task_doc

    _ensure_user_inputs(state)

    pause_stages = {s.strip() for s in args.pause_stages.split(",") if s.strip()}
    stages = _choose_range(args.start_stage, args.end_stage)

    print("\n" + "=" * 66)
    print("🎛️ 分阶段调试模式")
    print(f"阶段范围: {stages[0]} -> {stages[-1]}")
    print(f"暂停策略: {'所有阶段' if args.pause_all else sorted(pause_stages)}")
    print("=" * 66)

    final_state = await _run_stages(
        state=state,
        stages=stages,
        pause_stages=pause_stages,
        pause_all=args.pause_all,
    )

    final_snapshot = _save_snapshot(final_state, "final")
    print(f"\n🏁 本次调试结束，最终快照: {final_snapshot}")
    print("   你可以将该快照作为下一阶段的起点，避免重复调用前序 API。")


if __name__ == "__main__":
    asyncio.run(_amain())
