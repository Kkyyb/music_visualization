"""
LangGraph 图定义：串联 Workflow 阶段与 Agent 阶段。
"""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from nodes.agent_nodes import (
    generate_task_list_node,
    route_after_agent_execution,
    route_subtask_node,
    should_continue_agent,
    subtask_agent_node,
)
from nodes.workflow_nodes import (
    analyze_image_node,
    generate_image_node,
    generate_prompt_node,
    plan_sam_segmentation_node,
    sam3d_modeling_node,
    search_lyrics_node,
)
from state import GraphState


def build_graph() -> StateGraph:
    """
    构建完整的 LangGraph 流程图。

    Workflow 阶段是线性的：
        搜索歌词 -> 生成图像提示 -> 生成目标图 -> 分析目标图
        -> 规划 SAM 分割 -> SAM 3D 建模

    Agent 阶段是循环的：
        生成总任务计划 -> 路由当前子任务 -> subtask agent 执行
        -> 继续当前子任务 / 下一个子任务 / 结束
    """
    graph = StateGraph(GraphState)

    # Workflow 节点
    graph.add_node("search_lyrics", search_lyrics_node)
    graph.add_node("generate_prompt", generate_prompt_node)
    graph.add_node("generate_image", generate_image_node)
    graph.add_node("analyze_image", analyze_image_node)
    graph.add_node("plan_sam_segmentation", plan_sam_segmentation_node)
    graph.add_node("sam3d_modeling", sam3d_modeling_node)

    # Agent 节点
    graph.add_node("generate_task_list", generate_task_list_node)
    graph.add_node("route_subtask", route_subtask_node)
    graph.add_node("subtask_agent", subtask_agent_node)

    # Workflow 主干
    graph.add_edge(START, "search_lyrics")
    graph.add_edge("search_lyrics", "generate_prompt")
    graph.add_edge("generate_prompt", "generate_image")
    graph.add_edge("generate_image", "analyze_image")
    graph.add_edge("analyze_image", "plan_sam_segmentation")
    graph.add_edge("plan_sam_segmentation", "sam3d_modeling")

    # 进入 Agent 阶段
    graph.add_edge("sam3d_modeling", "generate_task_list")
    graph.add_edge("generate_task_list", "route_subtask")

    # 路由当前子任务；没有剩余任务时结束
    graph.add_conditional_edges(
        "route_subtask",
        should_continue_agent,
        {
            "execute": "subtask_agent",
            "finish": END,
        },
    )

    # subtask agent 自己负责观察、执行和决定是否完成当前子任务
    graph.add_conditional_edges(
        "subtask_agent",
        route_after_agent_execution,
        {
            "retry_agent": "subtask_agent",
            "route_subtask": "route_subtask",
            "finish": END,
        },
    )

    return graph


def create_app():
    """创建带 checkpoint 的可运行图应用。"""
    graph = build_graph()
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
