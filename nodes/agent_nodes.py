"""Agent 阶段节点：负责在 Blender 中逐步搭建场景。"""
from __future__ import annotations

import base64
import copy
import json
import os
import re
from typing import Any

from langchain_openai import ChatOpenAI

from config import MODEL_VISION, QWEN_BASE_URL, QWEN_KEY
from state import BlenderObject, GraphState
from token_tracker import tracker
from tools.blender_tools import BlenderMCP, BlenderSceneAssembler


blender = BlenderMCP()
assembler = BlenderSceneAssembler(blender)

MAX_RETRIES = 3
MAX_AGENT_STEPS = 8


async def generate_task_list_node(state: GraphState) -> dict:
    """在任何模型导入之前，先生成分层任务计划。"""
    print("\n" + "=" * 50)
    print("Node: generate hierarchical task plan")
    print("=" * 50)

    await assembler.connect()
    scene_result = await assembler.get_scene_info()
    screenshot_result = await assembler.get_blender_screenshot()
    scene_info = scene_result.get("scene_info", "")
    blender_screenshot = screenshot_result.get("screenshot_path", "")

    llm = ChatOpenAI(
        model=MODEL_VISION,
        api_key=QWEN_KEY,
        base_url=QWEN_BASE_URL,
    )

    object_catalog = _build_object_catalog(state)
    system = """You are a Blender scene assembly planner.
Create a hierarchical JSON task plan for progressive scene assembly.

Rules:
1. Models are not imported yet. The first execution stage must progressively import and align objects.
2. Choose anchor objects based on target-image importance, segmentation priority, and transform summaries.
3. Status values must be: pending, in_progress, completed, failed, skipped, blocked.
4. Use subtask type import_and_align for object-specific import-and-alignment tasks.
5. Use scene_adjustment for global refinement, lighting, camera, and final polish tasks.
6. Only reference object labels that exist in the provided catalog.
7. Output JSON only.
"""

    user_text = (
        f"Song: {state.get('song_name', '')} - {state.get('artist', '')}\n"
        f"Scene: {state.get('scene', '')}\n"
        f"Mood: {state.get('mood', '')}\n\n"
        f"Scene task doc:\n{state.get('scene_task_doc', '')[:3000]}\n\n"
        f"Current Blender scene info:\n{scene_info[:2500]}\n\n"
        f"Object catalog:\n{json.dumps(object_catalog, ensure_ascii=False, indent=2)}"
    )

    response = await llm.ainvoke(
        [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": _build_multimodal_content(
                    text=user_text,
                    image_paths=[state.get("image_local_path", ""), blender_screenshot],
                ),
            },
        ]
    )
    meta = response.usage_metadata or {}
    tracker.record_llm(
        node_name="hierarchical_task_plan",
        model=MODEL_VISION,
        input_tokens=meta.get("input_tokens", 0),
        output_tokens=meta.get("output_tokens", 0),
    )

    raw_plan = _parse_json_object(_resp_text(response.content)) or {}
    task_plan = _normalize_task_plan(raw_plan, state)
    master_id, subtask_id = _find_next_pending_subtask(task_plan)

    return {
        "task_plan": task_plan,
        "planning_context": task_plan.get("planning_summary", {}),
        "current_master_task_id": master_id,
        "current_subtask_id": subtask_id,
        "current_action": {},
        "current_command": "",
        "agent_execution_status": "",
        "subtask_memory": {},
        "scene_info": scene_info,
        "blender_screenshot": blender_screenshot,
    }


async def route_subtask_node(state: GraphState) -> dict:
    """从任务计划中选择当前要执行的子任务，并组装 current_action。"""
    print("\n" + "-" * 40)
    print("Node: route current subtask")
    print("-" * 40)

    await assembler.connect()
    scene_result = await assembler.get_scene_info()
    scene_info = scene_result.get("scene_info", "") or state.get("scene_info", "")

    task_plan = copy.deepcopy(state.get("task_plan") or {})
    master_task, subtask = _get_current_subtask(task_plan, state)
    if not master_task or not subtask:
        master_id, subtask_id = _find_next_pending_subtask(task_plan)
        if not master_id or not subtask_id:
            return {
                "current_action": {"action_type": "__DONE__"},
                "current_command": "",
                "agent_execution_status": "done",
            }
        master_task, subtask = _get_current_subtask(
            task_plan,
            {
                **state,
                "current_master_task_id": master_id,
                "current_subtask_id": subtask_id,
            },
        )

    if not master_task or not subtask:
        return {
            "current_action": {"action_type": "__DONE__"},
            "current_command": "",
            "agent_execution_status": "done",
        }

    if subtask["status"] == "pending":
        subtask["status"] = "in_progress"
    if master_task["status"] == "pending":
        master_task["status"] = "in_progress"

    target_obj = _find_blender_object(state, subtask.get("target_label", ""))
    requires_import = False
    if subtask.get("type") == "import_and_align":
        requires_import = (
            not target_obj
            or not target_obj.name
            or not _object_exists_in_scene(scene_info, target_obj.name)
        )

    current_action = {
        "action_type": subtask.get("type", "scene_adjustment"),
        "master_task_id": master_task["id"],
        "master_title": master_task["title"],
        "subtask_id": subtask["id"],
        "instruction": subtask.get("detail", ""),
        "target_label": subtask.get("target_label", ""),
        "target_blender_name": target_obj.name if target_obj else "",
        "source_file": target_obj.source_file if target_obj else "",
        "requires_import": requires_import,
        "success_criteria": subtask.get("success_criteria", []),
        "attempt": int(subtask.get("retries", 0)) + 1,
    }

    return {
        "task_plan": task_plan,
        "current_master_task_id": master_task["id"],
        "current_subtask_id": subtask["id"],
        "current_action": current_action,
        "current_command": "",
        "agent_execution_status": "",
        "scene_info": scene_info,
    }


async def subtask_agent_node(state: GraphState) -> dict:
    """围绕当前子任务运行单节点 ReAct 循环。"""
    action = copy.deepcopy(state.get("current_action") or {})
    if not action or action.get("action_type") == "__DONE__":
        return {"agent_execution_status": "done"}

    print("\n" + "-" * 40)
    print("Node: subtask agent")
    print("-" * 40)

    await assembler.connect()

    task_plan = copy.deepcopy(state.get("task_plan") or {})
    master_task, subtask = _get_current_subtask(task_plan, state)
    if not master_task or not subtask:
        return {"agent_execution_status": "route_subtask"}

    updated_objects = list(state.get("blender_objects", []))
    memory = copy.deepcopy(state.get("subtask_memory") or {})
    logs: list[str] = []
    scene_result = await assembler.get_scene_info()
    screenshot_result = await assembler.get_blender_screenshot()
    scene_info = scene_result.get("scene_info", "") or state.get("scene_info", "")
    screenshot_path = screenshot_result.get("screenshot_path", "") or state.get("blender_screenshot", "")
    current_command = state.get("current_command", "")

    for step_index in range(1, MAX_AGENT_STEPS + 1):
        recent_memory = _build_recent_memory(memory, action.get("subtask_id", ""))
        auto_finish_reason = _get_auto_finish_reason(action, recent_memory)
        if auto_finish_reason:
            return _complete_subtask(
                task_plan,
                master_task,
                subtask,
                action,
                updated_objects,
                scene_info,
                screenshot_path,
                memory,
                logs,
                auto_finish_reason,
            )

        tool_plan = _build_guardrail_tool_plan(action, recent_memory)
        if not tool_plan:
            tool_plan = await _plan_next_agent_step(
                state={
                    **state,
                    "blender_objects": updated_objects,
                    "scene_info": scene_info,
                    "blender_screenshot": screenshot_path,
                    "subtask_memory": memory,
                    "current_command": current_command,
                },
                action=action,
                step_index=step_index,
            )

        thought = str(tool_plan.get("thought") or "").strip()
        tool_name = str(tool_plan.get("tool") or "").strip()
        args = tool_plan.get("args") or {}

        if not tool_name:
            observation = {"success": False, "error": "Agent did not choose a tool."}
            memory = _record_agent_step(memory, action, step_index, thought, "", {}, observation)
            logs.append(f"[{action.get('subtask_id', '')}][step_{step_index}] missing tool")
            continue

        if tool_name == "finish_subtask":
            return _complete_subtask(
                task_plan,
                master_task,
                subtask,
                action,
                updated_objects,
                scene_info,
                screenshot_path,
                memory,
                logs,
                str(args.get("reason") or thought or "Agent marked the subtask as complete."),
            )

        step_result = await _execute_agent_tool(
            tool_name,
            args,
            action,
            updated_objects,
            scene_info,
            screenshot_path,
            current_command,
        )

        observation = step_result["observation"]
        updated_objects = step_result["blender_objects"]
        scene_info = step_result["scene_info"]
        screenshot_path = step_result["blender_screenshot"]
        current_command = step_result["current_command"]
        action = step_result["current_action"]

        memory = _record_agent_step(
            memory,
            action,
            step_index,
            thought,
            tool_name,
            args,
            observation,
        )
        logs.append(
            f"[{action.get('subtask_id', '')}][step_{step_index}] "
            f"{tool_name}: {_summarize_observation(observation)}"
        )

        recent_memory = _build_recent_memory(memory, action.get("subtask_id", ""))
        auto_finish_reason = _get_auto_finish_reason(action, recent_memory)
        if auto_finish_reason:
            return _complete_subtask(
                task_plan,
                master_task,
                subtask,
                action,
                updated_objects,
                scene_info,
                screenshot_path,
                memory,
                logs,
                auto_finish_reason,
            )

    return _handle_agent_rollover(
        task_plan,
        master_task,
        subtask,
        action,
        updated_objects,
        scene_info,
        screenshot_path,
        current_command,
        memory,
        logs,
        "Agent reached the maximum internal steps without finishing the current subtask.",
    )


def should_continue_agent(state: GraphState) -> str:
    """在子任务路由完成后决定图流程如何继续。"""
    action = state.get("current_action") or {}
    if action.get("action_type") == "__DONE__":
        return "finish"
    if not state.get("current_master_task_id") or not state.get("current_subtask_id"):
        return "finish"
    return "execute"


def route_after_agent_execution(state: GraphState) -> str:
    """在 subtask agent 执行后决定下一跳。"""
    status = str(state.get("agent_execution_status") or "").strip()
    if status == "route_subtask":
        return "route_subtask"
    if status == "done":
        return "finish"
    return "retry_agent"


async def generate_command_node(state: GraphState) -> dict:
    """兼容旧接口：转调到 route_subtask_node。"""
    return await route_subtask_node(state)


async def execute_command_node(state: GraphState) -> dict:
    """兼容旧接口：转调到 subtask_agent_node。"""
    return await subtask_agent_node(state)


_AGENT_TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "import_model",
            "description": "Import a 3D model (GLB/OBJ/FBX) into the Blender scene.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the model file. Omit to use the current action's source_file.",
                    },
                    "label": {
                        "type": "string",
                        "description": "Semantic label for the object (e.g. 'tree', 'bench').",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_scene_info",
            "description": "Query the current Blender scene for object names, transforms, and composition notes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_prompt": {
                        "type": "string",
                        "description": "Optional focus question for the scene summary.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_blender_screenshot",
            "description": "Capture a viewport screenshot of the current Blender scene.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_object_summary",
            "description": (
                "Get spatial data (location, scale, dimensions, bounding box) for a specific Blender object. "
                "Use this to verify object placement when post_execution_object_summary is unavailable."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "object_name": {
                        "type": "string",
                        "description": "Blender object name. Omit to use the current target object.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_blender_code",
            "description": (
                "Execute a bpy Python script in Blender. "
                "Use for transforms, material edits, or any scene modification. "
                "Never delete objects, clear the scene, reset Blender, or call import ops inside this script."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Complete bpy Python script to execute.",
                    },
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish_subtask",
            "description": "Mark the current subtask as successfully completed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Brief explanation of why the subtask is complete.",
                    },
                },
                "required": ["reason"],
            },
        },
    },
]


def _format_tool_schemas_for_prompt() -> str:
    """将 _AGENT_TOOL_SCHEMAS 转换为系统提示用的工具说明文本。"""
    lines: list[str] = []
    for entry in _AGENT_TOOL_SCHEMAS:
        fn = entry["function"]
        name = fn["name"]
        desc = fn.get("description", "")
        props = fn.get("parameters", {}).get("properties", {})
        required = set(fn.get("parameters", {}).get("required", []))

        lines.append(f"• {name}")
        lines.append(f"  {desc}")
        if props:
            lines.append("  Args:")
            for param_name, param_info in props.items():
                req_tag = "required" if param_name in required else "optional"
                param_desc = param_info.get("description", "")
                lines.append(f"    - {param_name} ({req_tag}): {param_desc}")
        else:
            lines.append("  Args: none")
        lines.append("")
    return "\n".join(lines).rstrip()


async def _plan_next_agent_step(
    state: GraphState,
    action: dict[str, Any],
    step_index: int,
) -> dict[str, Any]:
    """为当前子任务规划下一步工具调用（JSON 文本格式，兼容 Qwen）。"""
    llm = ChatOpenAI(
        model=MODEL_VISION,
        api_key=QWEN_KEY,
        base_url=QWEN_BASE_URL,
    )

    recent_memory = _build_recent_memory(
        state.get("subtask_memory") or {},
        action.get("subtask_id", ""),
    )
    memory_warning = _summarize_recent_memory_warning(recent_memory)
    finish_guidance = _build_finish_guidance(action, recent_memory)
    object_summary = [
        {"label": obj.label, "name": obj.name, "source_file": obj.source_file}
        for obj in state.get("blender_objects", [])
    ]

    system = (
        "You are a multimodal ReAct agent assembling a Blender scene one subtask at a time.\n\n"
        "You MUST respond with a JSON object only — no prose, no markdown fences. Schema:\n"
        "{\n"
        '  "thought": "<short reasoning in Chinese>",\n'
        '  "tool": "<tool name>",\n'
        '  "args": { <tool arguments> }\n'
        "}\n\n"
        "Available tools:\n"
        + _format_tool_schemas_for_prompt()
        + "\n\n"
        "Rules:\n"
        "1. If the subtask is complete, use finish_subtask.\n"
        "2. Prefer observing before large edits.\n"
        "3. If the target object is not in Blender yet, call import_model first.\n"
        "4. After execute_blender_code succeeds, the observation includes post_execution_object_summary "
        "with the object's updated spatial data — use it to decide if the subtask is done.\n"
        "5. Never delete objects, clear the scene, reset Blender, or call import ops inside execute_blender_code.\n"
        "6. Do not invent absolute transforms. Base every edit on the target image, current screenshot, "
        "scene info, and object summaries.\n"
        "7. Keep every step focused on the current subtask only.\n"
        "8. Prefer manipulating the imported MESH object directly. Ignore EMPTY glTF parent nodes unless no mesh "
        "object is available.\n"
        "9. If post_execution_object_summary is missing, found=false, or unchanged across repeated edit attempts, "
        "do not repeat the same edit immediately. Call a diagnostic tool first: get_object_summary, "
        "get_scene_info, or get_blender_screenshot.\n"
        "10. For import_and_align tasks, use finish_subtask when the target is imported, a valid object summary "
        "exists, and recent checks indicate the object state is stable or already good enough relative to the "
        "success criteria.\n"
        "11. Do not spend multiple consecutive steps on screenshots alone. After diagnostics, either make a "
        "specific change, verify with get_object_summary, or finish_subtask.\n"
    )

    user_text = (
        f"Current master task: {action.get('master_title', '')}\n"
        f"Current subtask: {action.get('instruction', '')}\n"
        f"Subtask id: {action.get('subtask_id', '')}\n"
        f"Action type: {action.get('action_type', '')}\n"
        f"Target label: {action.get('target_label', '')}\n"
        f"Target Blender object name: {action.get('target_blender_name', '')}\n"
        f"Target source file: {action.get('source_file', '')}\n"
        f"Requires import now: {action.get('requires_import', False)}\n"
        f"Current node step: {step_index}\n"
        f"Success criteria: {json.dumps(action.get('success_criteria', []), ensure_ascii=False)}\n\n"
        f"Planning summary:\n{json.dumps(state.get('planning_context', {}), ensure_ascii=False, indent=2)}\n\n"
        f"Scene task doc:\n{state.get('scene_task_doc', '')[:2200]}\n\n"
        f"Current scene info:\n{str(state.get('scene_info', ''))[:3200]}\n\n"
        f"Known objects:\n{json.dumps(object_summary, ensure_ascii=False, indent=2)}\n\n"
        f"Finish guidance:\n{finish_guidance or 'none'}\n\n"
        f"Recent memory warning:\n{memory_warning or 'none'}\n\n"
        f"Recent subtask memory:\n{json.dumps(recent_memory, ensure_ascii=False, indent=2)}\n"
    )

    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": _build_multimodal_content(
                text=user_text,
                image_paths=[state.get("image_local_path", ""), state.get("blender_screenshot", "")],
            ),
        },
    ]

    response = await llm.ainvoke(messages)
    meta = response.usage_metadata or {}
    tracker.record_llm(
        node_name=f"subtask_agent[{action.get('subtask_id', '')}]",
        model=MODEL_VISION,
        input_tokens=meta.get("input_tokens", 0),
        output_tokens=meta.get("output_tokens", 0),
    )

    result = _normalize_agent_step(_resp_text(response.content))
    if result.get("tool"):
        return result

    # 模型未输出合法 tool 字段时，追加强提示重试一次
    retry_response = await llm.ainvoke(
        messages
        + [
            {"role": "assistant", "content": _resp_text(response.content)},
            {
                "role": "user",
                "content": (
                    'Your response must be a JSON object with exactly these keys: "thought", "tool", "args". '
                    f'The "tool" value must be one of: import_model, get_scene_info, get_blender_screenshot, '
                    f'get_object_summary, execute_blender_code, finish_subtask. Output JSON only.'
                ),
            },
        ]
    )
    retry_meta = retry_response.usage_metadata or {}
    tracker.record_llm(
        node_name=f"subtask_agent_retry[{action.get('subtask_id', '')}]",
        model=MODEL_VISION,
        input_tokens=retry_meta.get("input_tokens", 0),
        output_tokens=retry_meta.get("output_tokens", 0),
    )
    return _normalize_agent_step(_resp_text(retry_response.content))


async def _execute_agent_tool(
    tool_name: str,
    args: dict[str, Any],
    action: dict[str, Any],
    updated_objects: list[BlenderObject],
    scene_info: str,
    screenshot_path: str,
    current_command: str,
) -> dict[str, Any]:
    """执行 agent 选择的单个工具。"""
    observation: dict[str, Any]
    next_action = copy.deepcopy(action)

    if tool_name == "import_model":
        file_path = str(args.get("file_path") or action.get("source_file") or "").strip()
        label = str(args.get("label") or action.get("target_label") or "").strip()
        observation = await assembler.import_model(file_path=file_path, label=label)
        if observation.get("success"):
            root_name = str(observation.get("root_object_name") or "").strip()
            updated_objects = _update_blender_object_name(updated_objects, label, root_name)
            next_action["target_blender_name"] = root_name
            next_action["requires_import"] = False
            scene_result = await assembler.get_scene_info()
            scene_info = scene_result.get("scene_info", "") or scene_info
            screenshot_result = await assembler.get_blender_screenshot()
            screenshot_path = screenshot_result.get("screenshot_path", "") or screenshot_path
    elif tool_name == "get_scene_info":
        prompt = str(args.get("user_prompt") or "").strip()
        observation = await assembler.get_scene_info(user_prompt=prompt or _DEFAULT_SCENE_PROMPT)
        scene_info = observation.get("scene_info", "") or scene_info
    elif tool_name == "get_blender_screenshot":
        observation = await assembler.get_blender_screenshot()
        screenshot_path = observation.get("screenshot_path", "") or screenshot_path
    elif tool_name == "get_object_summary":
        object_name = str(args.get("object_name") or action.get("target_blender_name") or "").strip()
        observation = await assembler.get_object_summary(object_name)
    elif tool_name == "execute_blender_code":
        code = str(args.get("code") or "").strip()
        validation_error = _validate_generated_code(code)
        if validation_error:
            observation = {"success": False, "error": f"Invalid bpy code: {validation_error}"}
        else:
            observation = await assembler.execute_blender_code(code)
            current_command = code
            # 执行成功后自动附加目标对象的空间状态，让 LLM 在下一步直接判断是否 finish_subtask
            if observation.get("success"):
                scene_result = await assembler.get_scene_info()
                scene_info = scene_result.get("scene_info", "") or scene_info
                target_name = str(next_action.get("target_blender_name") or "").strip()
                if target_name:
                    post_summary = await assembler.get_object_summary(target_name)
                    observation["post_execution_object_summary"] = post_summary
    else:
        observation = {"success": False, "error": f"Unsupported tool: {tool_name}"}

    return {
        "observation": observation,
        "blender_objects": updated_objects,
        "scene_info": scene_info,
        "blender_screenshot": screenshot_path,
        "current_command": current_command,
        "current_action": next_action,
    }


def _complete_subtask(
    task_plan: dict[str, Any],
    master_task: dict[str, Any],
    subtask: dict[str, Any],
    action: dict[str, Any],
    updated_objects: list[BlenderObject],
    scene_info: str,
    screenshot_path: str,
    memory: dict[str, Any],
    logs: list[str],
    reason: str,
) -> dict:
    """把当前子任务标记为完成，并推进到下一个任务。"""
    subtask["status"] = "completed"
    _refresh_master_task_status(task_plan)
    memory = copy.deepcopy(memory or {})
    master_id, subtask_id = _find_next_pending_subtask(task_plan)
    logs.append(f"[{action.get('subtask_id', '')}] completed: {reason}")
    return {
        "task_plan": task_plan,
        "blender_objects": updated_objects,
        "scene_info": scene_info,
        "blender_screenshot": screenshot_path,
        "current_master_task_id": master_id,
        "current_subtask_id": subtask_id,
        "current_action": {},
        "current_command": "",
        "agent_execution_status": "route_subtask" if master_id and subtask_id else "done",
        "subtask_memory": memory,
        "execution_log": logs,
    }


def _handle_agent_rollover(
    task_plan: dict[str, Any],
    master_task: dict[str, Any] | None,
    subtask: dict[str, Any] | None,
    action: dict[str, Any],
    updated_objects: list[BlenderObject],
    scene_info: str,
    screenshot_path: str,
    current_command: str,
    memory: dict[str, Any],
    logs: list[str],
    reason: str,
) -> dict:
    """处理单次 agent 调用内未能完成子任务的情况。"""
    if not subtask:
        return {
            "blender_objects": updated_objects,
            "scene_info": scene_info,
            "blender_screenshot": screenshot_path,
            "current_action": action,
            "current_command": current_command,
            "agent_execution_status": "retry_agent",
            "subtask_memory": memory,
            "execution_log": logs,
        }

    subtask["retries"] = int(subtask.get("retries", 0)) + 1
    if master_task and master_task.get("status") == "pending":
        master_task["status"] = "in_progress"

    if subtask["retries"] >= MAX_RETRIES:
        overflow_status = "blocked" if _should_block_subtask(subtask) else "skipped"
        subtask["status"] = overflow_status
        _refresh_master_task_status(task_plan)
        memory = copy.deepcopy(memory or {})
        memory.pop(str(subtask.get("id") or ""), None)
        master_id, subtask_id = _find_next_pending_subtask(task_plan)
        logs.append(f"[{action.get('subtask_id', '')}] {overflow_status}: {reason}")
        return {
            "task_plan": task_plan,
            "blender_objects": updated_objects,
            "scene_info": scene_info,
            "blender_screenshot": screenshot_path,
            "current_master_task_id": master_id,
            "current_subtask_id": subtask_id,
            "current_action": {},
            "current_command": "",
            "agent_execution_status": "route_subtask" if overflow_status == "skipped" else "done",
            "subtask_memory": memory,
            "execution_log": logs,
        }

    subtask["status"] = "in_progress"
    _refresh_master_task_status(task_plan)
    logs.append(f"[{action.get('subtask_id', '')}] retry: {reason}")
    return {
        "task_plan": task_plan,
        "blender_objects": updated_objects,
        "scene_info": scene_info,
        "blender_screenshot": screenshot_path,
        "current_action": action,
        "current_command": current_command,
        "agent_execution_status": "retry_agent",
        "subtask_memory": memory,
        "execution_log": logs,
    }


_DEFAULT_SCENE_PROMPT = (
    "Summarize the current Blender scene. "
    "List objects, counts, transforms, and composition notes."
)


def _normalize_agent_step(text: str) -> dict[str, Any]:
    """从 LLM 文本输出解析工具调用，兼容多种 JSON 字段命名习惯。"""
    parsed = _parse_json_object(text)
    if not isinstance(parsed, dict):
        return {}

    # "tool" 或 "action" 均视为工具名
    tool = str(parsed.get("tool") or parsed.get("action") or "").strip()

    # args 可能在 "args" 字段，也可能平铺在顶层
    args = parsed.get("args") or {}
    if not isinstance(args, dict):
        args = {}
    if not args and tool:
        _skip = {"tool", "action", "thought", "reasoning", "subtask_id", "step"}
        args = {k: v for k, v in parsed.items() if k not in _skip}

    # 字段别名归一化
    if "object_label" in args and "label" not in args:
        args["label"] = args.pop("object_label")
    if "file" in args and "file_path" not in args:
        args["file_path"] = args.pop("file")
    if "prompt" in args and "user_prompt" not in args and tool == "get_scene_info":
        args["user_prompt"] = args.pop("prompt")

    # execute_blender_code：code 字段缺失时尝试从围栏代码块提取
    if tool == "execute_blender_code" and not str(args.get("code") or "").strip():
        code = _extract_code(text)
        if code:
            args["code"] = code

    thought = str(parsed.get("thought") or parsed.get("reasoning") or "").strip()
    return {"thought": thought, "tool": tool, "args": args}


def _extract_code(text: str) -> str:
    """从文本中提取第一个代码围栏块。"""
    if "```python" in text:
        return text.split("```python", 1)[1].split("```", 1)[0].strip()
    if "```" in text:
        return text.split("```", 1)[1].split("```", 1)[0].strip()
    return ""


def _summarize_observation(observation: dict[str, Any]) -> str:
    if not isinstance(observation, dict):
        return str(observation)[:240]
    compact = copy.deepcopy(observation)
    for key in ("raw_result", "stdout", "scene_info", "error"):
        if key in compact:
            compact[key] = str(compact[key])[:200]
    return json.dumps(compact, ensure_ascii=False)[:240]


def _build_object_catalog(state: GraphState) -> list[dict[str, Any]]:
    batch_lookup: dict[str, str] = {}
    priority_lookup: dict[str, int] = {}
    for item in (state.get("sam_segmentation_plan") or {}).get("objects", []):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip().lower()
        if label:
            batch_lookup[label] = str(item.get("batch") or "secondary")
            priority_lookup[label] = int(item.get("priority", 999) or 999)

    transform_lookup = _build_transform_lookup(state)
    return [
        {
            "label": obj.label,
            "source_file": obj.source_file,
            "segmentation_batch": batch_lookup.get(obj.label.strip().lower(), "secondary"),
            "segmentation_priority": priority_lookup.get(obj.label.strip().lower(), 999),
            "transform_summary": transform_lookup.get(obj.label.strip().lower(), {}),
        }
        for obj in state.get("blender_objects", [])
    ]


def _build_transform_lookup(state: GraphState) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for item in (state.get("sam3d_result") or {}).get("objects", []):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip().lower()
        if not label:
            continue
        transform = item.get("transform") or {}
        lookup[label] = {
            "scale": _flatten_numeric(transform.get("scale")),
            "rotation": _flatten_numeric(transform.get("rotation")),
            "translation": _flatten_numeric(transform.get("translation")),
        }
    return lookup


def _flatten_numeric(value: Any) -> list[float]:
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list):
        return []
    out: list[float] = []
    for item in value:
        try:
            out.append(round(float(item), 4))
        except Exception:
            continue
    return out


def _normalize_task_plan(raw_plan: dict[str, Any], state: GraphState) -> dict[str, Any]:
    if not isinstance(raw_plan, dict):
        raw_plan = {}
    master_tasks = raw_plan.get("master_tasks")
    if not isinstance(master_tasks, list) or not master_tasks:
        return _build_fallback_task_plan(state)

    available_labels = {obj.label.strip().lower() for obj in state.get("blender_objects", [])}
    normalized_tasks: list[dict[str, Any]] = []
    for index, raw_master in enumerate(master_tasks, start=1):
        if not isinstance(raw_master, dict):
            continue
        normalized_subtasks: list[dict[str, Any]] = []
        for sub_index, raw_sub in enumerate(raw_master.get("subtasks", []), start=1):
            if not isinstance(raw_sub, dict):
                continue
            target_label = str(raw_sub.get("target_label") or "").strip()
            sub_type = str(raw_sub.get("type") or "scene_adjustment").strip()
            if sub_type == "import_and_align" and target_label.lower() not in available_labels:
                continue
            normalized_subtasks.append(
                {
                    "id": str(raw_sub.get("id") or f"master_{index}_sub_{sub_index}"),
                    "type": sub_type,
                    "target_label": target_label,
                    "detail": str(raw_sub.get("detail") or "").strip(),
                    "status": _normalize_status(raw_sub.get("status")),
                    "retries": int(raw_sub.get("retries", 0) or 0),
                    "is_anchor": bool(raw_sub.get("is_anchor", False)),
                    "depends_on": [str(dep).strip() for dep in raw_sub.get("depends_on", []) if str(dep).strip()],
                    "success_criteria": [str(item).strip() for item in raw_sub.get("success_criteria", []) if str(item).strip()],
                }
            )
        normalized_tasks.append(
            {
                "id": str(raw_master.get("id") or f"master_{index}"),
                "title": str(raw_master.get("title") or f"Task Group {index}").strip(),
                "status": _normalize_status(raw_master.get("status")),
                "priority": int(raw_master.get("priority", index) or index),
                "success_criteria": [str(item).strip() for item in raw_master.get("success_criteria", []) if str(item).strip()],
                "subtasks": normalized_subtasks,
            }
        )

    if not any(task.get("subtasks") for task in normalized_tasks):
        return _build_fallback_task_plan(state)

    plan = {
        "plan_version": str(raw_plan.get("plan_version") or "1.0"),
        "planning_summary": raw_plan.get("planning_summary") or {},
        "master_tasks": sorted(normalized_tasks, key=lambda item: item.get("priority", 999)),
    }
    _refresh_master_task_status(plan)
    return plan


def _build_fallback_task_plan(state: GraphState) -> dict[str, Any]:
    seg_objects = [item for item in (state.get("sam_segmentation_plan") or {}).get("objects", []) if isinstance(item, dict)]
    label_to_obj = {obj.label.strip().lower(): obj for obj in state.get("blender_objects", [])}

    def collect_by_batch(batch: str) -> list[str]:
        labels: list[str] = []
        for item in sorted(seg_objects, key=lambda x: x.get("priority", 999)):
            label = str(item.get("label") or "").strip().lower()
            if item.get("batch") == batch and label in label_to_obj:
                labels.append(label_to_obj[label].label)
        return labels

    anchor_labels = collect_by_batch("anchor") or [obj.label for obj in state.get("blender_objects", [])[:3]]
    secondary_labels = collect_by_batch("secondary") + collect_by_batch("optional")
    secondary_labels = [label for label in secondary_labels if label not in anchor_labels]
    master_tasks = [
        {
            "id": "anchor_import_and_alignment",
            "title": "Import and align anchor objects",
            "status": "pending",
            "priority": 1,
            "success_criteria": ["Anchor objects establish stable scale and layout references."],
            "subtasks": [_build_import_subtask(label, True, i) for i, label in enumerate(anchor_labels, start=1)],
        },
        {
            "id": "secondary_import_and_alignment",
            "title": "Import and align secondary objects",
            "status": "pending",
            "priority": 2,
            "success_criteria": ["Secondary objects support the composition without breaking anchors."],
            "subtasks": [_build_import_subtask(label, False, i) for i, label in enumerate(secondary_labels, start=1)],
        },
        {
            "id": "global_composition_refinement",
            "title": "Refine global composition",
            "status": "pending",
            "priority": 3,
            "success_criteria": ["Overall composition matches the target image."],
            "subtasks": [_build_scene_subtask("global_refine_layout", "Refine the global scale, spacing, and composition after key objects are placed.")],
        },
    ]
    plan = {
        "plan_version": "1.0",
        "planning_summary": {
            "needs_scale_alignment": True,
            "anchor_objects": anchor_labels,
            "diagnosis": ["Planner JSON unavailable, using fallback task plan."],
        },
        "master_tasks": [task for task in master_tasks if task["subtasks"]],
    }
    _refresh_master_task_status(plan)
    return plan


def _build_import_subtask(label: str, is_anchor: bool, index: int) -> dict[str, Any]:
    return {
        "id": f"{'anchor' if is_anchor else 'secondary'}_{index}_{_slug(label)}",
        "type": "import_and_align",
        "target_label": label,
        "detail": f"Import {label} and align its scale, position, and orientation to match the target image.",
        "status": "pending",
        "retries": 0,
        "is_anchor": is_anchor,
        "depends_on": [],
        "success_criteria": [
            f"{label} is imported successfully.",
            f"{label} matches the target-image scale relationship.",
            f"{label} sits in a reasonable position relative to existing anchors.",
        ],
    }


def _build_scene_subtask(subtask_id: str, detail: str) -> dict[str, Any]:
    return {
        "id": subtask_id,
        "type": "scene_adjustment",
        "target_label": "",
        "detail": detail,
        "status": "pending",
        "retries": 0,
        "is_anchor": False,
        "depends_on": [],
        "success_criteria": [detail],
    }


def _normalize_status(status: Any) -> str:
    valid = {"pending", "in_progress", "completed", "failed", "skipped", "blocked"}
    status = str(status or "pending").strip().lower()
    return status if status in valid else "pending"


def _should_block_subtask(subtask: dict[str, Any]) -> bool:
    if bool(subtask.get("is_anchor", False)):
        return True
    if str(subtask.get("type") or "").strip() == "import_and_align":
        return str(subtask.get("target_label") or "").strip().lower() in {"base", "tree", "wooden house"}
    return False


def _refresh_master_task_status(task_plan: dict[str, Any]) -> None:
    for master_task in task_plan.get("master_tasks", []):
        subtasks = master_task.get("subtasks", [])
        statuses = [sub.get("status", "pending") for sub in subtasks]
        if not subtasks:
            master_task["status"] = "completed"
        elif all(status == "completed" for status in statuses):
            master_task["status"] = "completed"
        elif all(status in {"completed", "skipped"} for status in statuses):
            master_task["status"] = "completed"
        elif any(status == "in_progress" for status in statuses):
            master_task["status"] = "in_progress"
        elif any(status == "blocked" for status in statuses):
            master_task["status"] = "blocked"
        elif any(status == "failed" for status in statuses):
            master_task["status"] = "failed"
        else:
            master_task["status"] = "pending"


def _find_next_pending_subtask(task_plan: dict[str, Any]) -> tuple[str, str]:
    for master_task in task_plan.get("master_tasks", []):
        for subtask in master_task.get("subtasks", []):
            if subtask.get("status") in {"pending", "in_progress"}:
                return master_task.get("id", ""), subtask.get("id", "")
    return "", ""


def _get_current_subtask(
    task_plan: dict[str, Any],
    state: GraphState | dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    master_id = state.get("current_master_task_id", "")
    subtask_id = state.get("current_subtask_id", "")
    for master_task in task_plan.get("master_tasks", []):
        if master_task.get("id") != master_id:
            continue
        for subtask in master_task.get("subtasks", []):
            if subtask.get("id") == subtask_id:
                return master_task, subtask
    return None, None


def _find_blender_object(state: GraphState | dict[str, Any], label: str) -> BlenderObject | None:
    target_label = label.strip().lower()
    for obj in state.get("blender_objects", []):
        if obj.label.strip().lower() == target_label:
            return obj
    return None


def _object_exists_in_scene(scene_info: str, object_name: str) -> bool:
    if not scene_info or not object_name:
        return False
    try:
        parsed = json.loads(scene_info)
    except Exception:
        parsed = None
    if isinstance(parsed, dict):
        for item in parsed.get("objects", []):
            if isinstance(item, dict) and str(item.get("name", "")).strip() == object_name:
                return True
    return object_name in scene_info


def _update_blender_object_name(
    objects: list[BlenderObject],
    label: str,
    blender_name: str,
) -> list[BlenderObject]:
    updated: list[BlenderObject] = []
    target_label = label.strip().lower()
    for obj in objects:
        if obj.label.strip().lower() == target_label:
            updated.append(BlenderObject(name=blender_name or obj.name, source_file=obj.source_file, label=obj.label))
        else:
            updated.append(obj)
    return updated


def _validate_generated_code(code: str) -> str:
    if not code.strip():
        return "empty code"
    banned_patterns = {
        "bpy.ops.object.delete": "must not delete objects",
        "bpy.ops.object.select_all": "must not use select_all scene reset patterns",
        "bpy.ops.wm.read_factory_settings": "must not reset Blender",
        "bpy.ops.import_scene.gltf": "must not import in execute_blender_code",
        "bpy.ops.import_scene.fbx": "must not import in execute_blender_code",
        "bpy.ops.wm.obj_import": "must not import in execute_blender_code",
    }
    lowered = code.lower()
    for pattern, reason in banned_patterns.items():
        if pattern.lower() in lowered:
            return reason
    for pattern in [
        'location = transform_summary["translation"]',
        "location=transform_summary[\"translation\"]",
        "location = transform_summary['translation']",
        "location=transform_summary['translation']",
    ]:
        if pattern in lowered:
            return "must not copy reconstruction translation directly into Blender world coordinates"
    return ""


def _build_multimodal_content(text: str, image_paths: list[str]) -> list[dict[str, Any]] | str:
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for path in image_paths:
        if not path or not os.path.exists(path):
            continue
        with open(path, "rb") as handle:
            img_b64 = base64.b64encode(handle.read()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})
    return content if len(content) > 1 else text


def _resp_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts).strip()
    return str(content).strip()


def _parse_json_object(text: str) -> dict[str, Any] | None:
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


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    return cleaned.strip("_") or "item"


def _record_agent_step(
    memory: dict[str, Any],
    action: dict[str, Any],
    step_index: int,
    thought: str,
    tool_name: str,
    args: dict[str, Any],
    observation: dict[str, Any],
) -> dict[str, Any]:
    memory = copy.deepcopy(memory or {})
    subtask_id = str(action.get("subtask_id") or "unknown")
    record = memory.get(subtask_id) or {"subtask_id": subtask_id, "steps": []}
    steps = list(record.get("steps", []))
    compact_args = copy.deepcopy(args or {})
    if "code" in compact_args:
        compact_args["code"] = str(compact_args["code"])[:800]
    compact_observation = copy.deepcopy(observation or {})
    for key in ("raw_result", "stdout", "scene_info", "error"):
        if key in compact_observation:
            compact_observation[key] = str(compact_observation[key])[:800]
    steps.append(
        {
            "step": step_index,
            "thought": thought[:300],
            "tool": tool_name,
            "args": compact_args,
            "observation": compact_observation,
        }
    )
    record["steps"] = steps[-10:]
    memory[subtask_id] = record
    return memory


def _build_recent_memory(memory: dict[str, Any], subtask_id: str) -> list[dict[str, Any]]:
    record = (memory or {}).get(str(subtask_id), {})
    steps = record.get("steps", [])
    return steps[-6:] if isinstance(steps, list) else []


def _summarize_recent_memory_warning(recent_memory: list[dict[str, Any]]) -> str:
    if not isinstance(recent_memory, list) or len(recent_memory) < 3:
        return ""

    tail = recent_memory[-3:]
    if not all(isinstance(step, dict) and step.get("tool") == "execute_blender_code" for step in tail):
        return ""

    codes = {
        str((step.get("args") or {}).get("code") or "").strip()
        for step in tail
    }
    summaries = {
        json.dumps(
            (step.get("observation") or {}).get("post_execution_object_summary") or {},
            ensure_ascii=False,
            sort_keys=True,
        )
        for step in tail
    }

    if len(codes) == 1 and len(summaries) == 1:
        return (
            "The last 3 execute_blender_code steps repeated the same code and produced the same "
            "post_execution_object_summary. Do not repeat that edit again; either call a diagnostic tool "
            "or finish_subtask if the success criteria are already met."
        )
    if len(summaries) == 1:
        return (
            "The last 3 execute_blender_code steps produced the same post_execution_object_summary. "
            "Do not keep editing blindly; inspect with a diagnostic tool or finish_subtask."
        )
    return ""


def _build_finish_guidance(action: dict[str, Any], recent_memory: list[dict[str, Any]]) -> str:
    if str(action.get("action_type") or "").strip() != "import_and_align":
        return ""

    latest_summary = _get_latest_valid_summary(recent_memory)
    if not latest_summary:
        return (
            "This import_and_align subtask is not finishable yet. Import the object and obtain a valid object "
            "summary first."
        )

    auto_finish_reason = _get_auto_finish_reason(action, recent_memory)
    if auto_finish_reason:
        return (
            "This subtask is finishable now. Use finish_subtask unless there is concrete contradictory evidence "
            f"in the current screenshot. Reason: {auto_finish_reason}"
        )

    return (
        "This subtask can be finished once recent object summaries stop changing or diagnostics confirm the "
        "current state is already good enough."
    )


def _build_guardrail_tool_plan(action: dict[str, Any], recent_memory: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(recent_memory, list) or len(recent_memory) < 2:
        return {}

    last_step = recent_memory[-1] if isinstance(recent_memory[-1], dict) else {}
    last_tool = str(last_step.get("tool") or "").strip()
    target_name = str(action.get("target_blender_name") or "").strip()

    if _has_repeated_execute_without_progress(recent_memory):
        if target_name:
            return {
                "thought": "连续编辑后对象summary无变化，先强制读取对象摘要确认当前状态。",
                "tool": "get_object_summary",
                "args": {"object_name": target_name},
            }
        return {
            "thought": "连续编辑后缺少可靠对象信息，先强制刷新场景信息。",
            "tool": "get_scene_info",
            "args": {"user_prompt": _DEFAULT_SCENE_PROMPT},
        }

    if last_tool == "get_blender_screenshot" and _count_trailing_tool(recent_memory, "get_blender_screenshot") >= 2:
        if target_name:
            return {
                "thought": "已连续截图而无新结论，先读取对象摘要，不再继续盲目截图。",
                "tool": "get_object_summary",
                "args": {"object_name": target_name},
            }
        return {
            "thought": "已连续截图而无新结论，先刷新场景信息。",
            "tool": "get_scene_info",
            "args": {"user_prompt": _DEFAULT_SCENE_PROMPT},
        }

    return {}


def _get_auto_finish_reason(action: dict[str, Any], recent_memory: list[dict[str, Any]]) -> str:
    if str(action.get("action_type") or "").strip() != "import_and_align":
        return ""

    target_name = str(action.get("target_blender_name") or "").strip()
    if not target_name:
        return ""

    latest_summary = _get_latest_valid_summary(recent_memory)
    if not latest_summary:
        return ""
    if str(latest_summary.get("name") or "").strip() != target_name:
        return ""

    stable_reasons: list[str] = []
    if _count_trailing_same_summary(recent_memory) >= 2:
        stable_reasons.append("recent object summaries are stable")

    if (
        _count_trailing_tool(recent_memory, "get_blender_screenshot") >= 2
        and _has_recent_execute_step(recent_memory)
    ):
        stable_reasons.append("diagnostics continued after a valid object summary without new edits")

    if not stable_reasons:
        return ""

    return (
        "Auto-finished import_and_align because the target is imported, a valid object summary exists, and "
        + " / ".join(stable_reasons)
        + "."
    )


def _get_latest_valid_summary(recent_memory: list[dict[str, Any]]) -> dict[str, Any]:
    for step in reversed(recent_memory or []):
        summary = _extract_summary_from_step(step)
        if _is_valid_summary(summary):
            return summary
    return {}


def _extract_summary_from_step(step: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(step, dict):
        return {}
    observation = step.get("observation") or {}
    if not isinstance(observation, dict):
        return {}

    tool_name = str(step.get("tool") or "").strip()
    if tool_name == "get_object_summary":
        return observation if isinstance(observation, dict) else {}

    summary = observation.get("post_execution_object_summary") or {}
    return summary if isinstance(summary, dict) else {}


def _is_valid_summary(summary: dict[str, Any]) -> bool:
    return bool(
        isinstance(summary, dict)
        and summary.get("success") is True
        and summary.get("found") is True
        and str(summary.get("name") or "").strip()
    )


def _summary_signature(summary: dict[str, Any]) -> str:
    if not _is_valid_summary(summary):
        return ""
    compact = {
        "name": summary.get("name"),
        "type": summary.get("type"),
        "location": summary.get("location"),
        "scale": summary.get("scale"),
        "dimensions": summary.get("dimensions"),
        "bbox_min": summary.get("bbox_min"),
        "bbox_max": summary.get("bbox_max"),
    }
    return json.dumps(compact, ensure_ascii=False, sort_keys=True)


def _count_trailing_same_summary(recent_memory: list[dict[str, Any]]) -> int:
    count = 0
    latest_signature = ""
    for step in reversed(recent_memory or []):
        signature = _summary_signature(_extract_summary_from_step(step))
        if not signature:
            continue
        if not latest_signature:
            latest_signature = signature
            count = 1
            continue
        if signature != latest_signature:
            break
        count += 1
    return count


def _count_trailing_tool(recent_memory: list[dict[str, Any]], tool_name: str) -> int:
    count = 0
    for step in reversed(recent_memory or []):
        if not isinstance(step, dict) or str(step.get("tool") or "").strip() != tool_name:
            break
        count += 1
    return count


def _has_recent_execute_step(recent_memory: list[dict[str, Any]]) -> bool:
    return any(
        isinstance(step, dict) and str(step.get("tool") or "").strip() == "execute_blender_code"
        for step in recent_memory or []
    )


def _has_repeated_execute_without_progress(recent_memory: list[dict[str, Any]]) -> bool:
    if not isinstance(recent_memory, list) or len(recent_memory) < 2:
        return False

    tail = recent_memory[-2:]
    if not all(isinstance(step, dict) and step.get("tool") == "execute_blender_code" for step in tail):
        return False

    signatures = [_summary_signature(_extract_summary_from_step(step)) for step in tail]
    return bool(signatures[0] and signatures[0] == signatures[1])
