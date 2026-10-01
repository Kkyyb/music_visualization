"""
LangGraph 全局状态定义。
"""
from __future__ import annotations

from typing import Annotated, Any

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class TokenUsage(BaseModel):
    """单次 LLM 调用的 token 统计。"""

    node_name: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0


class TaskItem(BaseModel):
    """为兼容旧快照保留的扁平任务项。"""

    index: int = 0
    description: str = ""
    status: str = "pending"
    retries: int = 0


class PlanSubtask(BaseModel):
    """顶层任务中的可执行子任务。"""

    id: str = ""
    type: str = "scene_adjustment"
    target_label: str = ""
    detail: str = ""
    status: str = "pending"
    retries: int = 0
    is_anchor: bool = False
    depends_on: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)


class MasterTask(BaseModel):
    """Agent 阶段的顶层任务组。"""

    id: str = ""
    title: str = ""
    status: str = "pending"
    priority: int = 0
    success_criteria: list[str] = Field(default_factory=list)
    subtasks: list[PlanSubtask] = Field(default_factory=list)


class BlenderObject(BaseModel):
    """跨 SAM 输出与 Blender 导入过程的对象跟踪记录。"""

    name: str = ""
    source_file: str = ""
    label: str = ""


def _merge_token_log(left: list[TokenUsage], right: list[TokenUsage]) -> list[TokenUsage]:
    """追加 token 日志。"""

    return left + right


class GraphState(TypedDict):
    """全局图状态。"""

    song_name: str
    artist: str
    scene: str
    mood: str
    lyrics: str

    image_prompt: str
    image_url: str
    image_local_path: str
    scene_task_doc: str
    sam_segmentation_plan: dict[str, Any]
    sam3d_result: dict[str, Any]

    blender_objects: list[BlenderObject]
    scene_info: str

    task_plan: dict[str, Any]
    planning_context: dict[str, Any]
    current_master_task_id: str
    current_subtask_id: str
    current_action: dict[str, Any]
    current_command: str
    agent_execution_status: str
    subtask_memory: dict[str, Any]
    blender_screenshot: str
    execution_log: Annotated[list[str], lambda a, b: a + b]

    token_log: Annotated[list[TokenUsage], _merge_token_log]
    messages: Annotated[list, add_messages]
