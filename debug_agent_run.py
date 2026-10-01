"""
Single-subtask debug runner.

Loads a saved snapshot, points the route cursor at one target subtask, runs
`route_subtask_node`, then runs one `subtask_agent_node`, and saves a new
snapshot.

PowerShell example:
    .venv\\Scripts\\python.exe debug_agent_run.py `
        --snapshot-in debug_snapshots\\20260407_185059_generate_task_list.json `
        --target-subtask anchor_1_base `
        --isolated
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Any

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from dotenv import load_dotenv

load_dotenv(os.path.join(PROJECT_DIR, ".env"))

from pydantic import BaseModel

from nodes.agent_nodes import route_subtask_node, subtask_agent_node
from state import BlenderObject, GraphState, TokenUsage

SNAPSHOT_DIR = os.path.join(PROJECT_DIR, "debug_snapshots")


def _serialize(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    return obj


def _hydrate(state: dict[str, Any]) -> GraphState:
    defaults: GraphState = {
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
    defaults.update(state)
    defaults["blender_objects"] = [
        obj if isinstance(obj, BlenderObject) else BlenderObject(**obj)
        for obj in defaults.get("blender_objects", [])
    ]
    defaults["token_log"] = [
        item if isinstance(item, TokenUsage) else TokenUsage(**item)
        for item in defaults.get("token_log", [])
    ]
    return defaults


def _merge(state: GraphState, updates: dict[str, Any]) -> GraphState:
    merged = dict(state)
    for key, value in updates.items():
        if key in ("execution_log", "token_log", "messages"):
            merged[key] = list(merged.get(key, [])) + list(value if value else [])
        else:
            merged[key] = value
    return _hydrate(merged)


def _save(state: GraphState, tag: str) -> str:
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SNAPSHOT_DIR, f"{ts}_{tag}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_serialize(state), handle, ensure_ascii=False, indent=2)
    print(f"\nSaved snapshot: {path}")
    return path


def _find_subtask(task_plan: dict[str, Any], subtask_id: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    for master_task in task_plan.get("master_tasks", []):
        for subtask in master_task.get("subtasks", []):
            if subtask.get("id") == subtask_id:
                return master_task, subtask
    return None, None


def _refresh_master_task_status(task_plan: dict[str, Any]) -> None:
    for master_task in task_plan.get("master_tasks", []):
        subtasks = master_task.get("subtasks", [])
        statuses = [str(subtask.get("status", "")).strip() for subtask in subtasks]
        if not statuses:
            master_task["status"] = "completed"
        elif all(status == "completed" for status in statuses):
            master_task["status"] = "completed"
        elif all(status in {"completed", "skipped"} for status in statuses):
            master_task["status"] = "completed"
        elif any(status == "in_progress" for status in statuses):
            master_task["status"] = "in_progress"
        elif any(status == "blocked" for status in statuses):
            master_task["status"] = "blocked"
        elif any(status == "pending" for status in statuses):
            master_task["status"] = "pending"
        else:
            master_task["status"] = statuses[0] if statuses else "pending"


def _mark_dependency_chain_completed(
    task_plan: dict[str, Any],
    subtask_id: str,
    visited: set[str] | None = None,
) -> None:
    visited = visited or set()
    if subtask_id in visited:
        return
    visited.add(subtask_id)

    _, subtask = _find_subtask(task_plan, subtask_id)
    if not subtask:
        return

    for dep_id in subtask.get("depends_on", []):
        dep_id = str(dep_id).strip()
        if not dep_id:
            continue
        _mark_dependency_chain_completed(task_plan, dep_id, visited)
        _, dep_subtask = _find_subtask(task_plan, dep_id)
        if dep_subtask:
            dep_subtask["status"] = "completed"
            dep_subtask["retries"] = 0


def _prepare_isolated_task_plan(task_plan: dict[str, Any], target_subtask_id: str) -> None:
    _mark_dependency_chain_completed(task_plan, target_subtask_id)
    _, target_subtask = _find_subtask(task_plan, target_subtask_id)
    if target_subtask:
        target_subtask["status"] = "pending"
        target_subtask["retries"] = 0
    _refresh_master_task_status(task_plan)


def _print_task_plan(task_plan: dict[str, Any]) -> None:
    for master_task in task_plan.get("master_tasks", []):
        print(f"  [{master_task.get('status', ''):12s}] {master_task.get('id', '')}")
        for subtask in master_task.get("subtasks", []):
            arrow = "->" if subtask.get("status") == "in_progress" else "  "
            label = str(subtask.get("target_label", "")).strip()
            label_part = f"  label={label}" if label else ""
            print(f"    {arrow}[{subtask.get('status', ''):12s}] {subtask.get('id', '')}{label_part}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Debug-run a single scene-assembly subtask.")
    parser.add_argument("--snapshot-in", required=True, help="Input snapshot path.")
    parser.add_argument("--target-subtask", required=True, help="Target subtask id.")
    parser.add_argument(
        "--isolated",
        action="store_true",
        help="Clear debug memory/logs and pre-complete dependency subtasks before running the target subtask.",
    )
    args = parser.parse_args()

    snapshot_path = args.snapshot_in
    if not os.path.isabs(snapshot_path):
        snapshot_path = os.path.join(PROJECT_DIR, snapshot_path)

    with open(snapshot_path, encoding="utf-8") as handle:
        raw = json.load(handle)

    # Always start debug runs without inherited per-subtask memory.
    raw["subtask_memory"] = {}
    if args.isolated:
        raw["execution_log"] = []
        raw["current_action"] = {}
        raw["current_command"] = ""
        raw["agent_execution_status"] = ""

    state = _hydrate(raw)
    print(f"\nLoaded snapshot: {snapshot_path}")

    task_plan = state.get("task_plan") or {}
    master_task, subtask = _find_subtask(task_plan, args.target_subtask)
    if not master_task or not subtask:
        available = [
            st.get("id", "")
            for mt in task_plan.get("master_tasks", [])
            for st in mt.get("subtasks", [])
        ]
        print(f"Subtask not found: {args.target_subtask}")
        print(f"Available subtasks: {available}")
        sys.exit(1)

    if args.isolated:
        _prepare_isolated_task_plan(task_plan, args.target_subtask)
        state = _merge(state, {"task_plan": task_plan})
        master_task, subtask = _find_subtask(task_plan, args.target_subtask)

    print(f"\nTarget: {args.target_subtask}  (master={master_task['id']}, status={subtask['status']})")
    print("\nCurrent task plan:")
    _print_task_plan(task_plan)

    state = _merge(
        state,
        {
            "current_master_task_id": master_task["id"],
            "current_subtask_id": subtask["id"],
            "current_action": {},
        },
    )

    print("\n" + "=" * 60)
    print("route_subtask_node")
    route_updates = await route_subtask_node(state)
    state = _merge(state, route_updates)

    action = state.get("current_action") or {}
    print(f"  action_type    : {action.get('action_type', '')}")
    print(f"  target_label   : {action.get('target_label', '')}")
    print(f"  requires_import: {action.get('requires_import', '')}")
    print(f"  source_file    : {action.get('source_file', '')}")

    if action.get("action_type") == "__DONE__":
        print("\nAll subtasks are already complete.")
        return

    print("\n" + "=" * 60)
    print(f"subtask_agent_node -> {args.target_subtask}")
    print("=" * 60)

    agent_updates = await subtask_agent_node(state)
    state = _merge(state, agent_updates)

    status = state.get("agent_execution_status", "")
    print(f"\nagent_execution_status: {status}")

    logs = state.get("execution_log") or []
    if logs:
        print("Recent execution log:")
        for entry in logs[-5:]:
            print(f"  {entry}")

    _save(state, f"agent_{args.target_subtask}")

    print("\nFinal task plan:")
    _print_task_plan(state.get("task_plan") or {})


if __name__ == "__main__":
    asyncio.run(main())
