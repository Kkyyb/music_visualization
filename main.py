"""
Project entry point.
"""
from __future__ import annotations

import asyncio

from graph import create_app
from token_tracker import tracker


async def main():
    print("=" * 54)
    print("   音乐可视化场景生成器 v2")
    print("=" * 54)

    song_name = input("\n歌曲名: ").strip()
    artist = input("歌手: ").strip()
    scene = input("环境场景: ").strip()
    mood = input("心情: ").strip()
    lyrics = input("歌词(可留空自动搜索): ").strip()

    if not song_name or not artist:
        print("歌曲名和歌手不能为空。")
        return

    initial_state = {
        "song_name": song_name,
        "artist": artist,
        "scene": scene or "默认室内环境",
        "mood": mood or "平静",
        "lyrics": lyrics,
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
        "blender_screenshot": "",
        "execution_log": [],
        "token_log": [],
        "messages": [],
    }

    app = create_app()
    config = {"configurable": {"thread_id": "main"}}

    print("\n流程启动...\n")
    async for event in app.astream(initial_state, config, stream_mode="updates"):
        for node_name, data in event.items():
            if node_name == "__end__":
                continue
            keys_updated = list(data.keys()) if isinstance(data, dict) else []
            if keys_updated:
                print(f"  节点 [{node_name}] 更新了: {keys_updated}")

    tracker.print_summary()

    final_state = app.get_state(config)
    task_plan = final_state.values.get("task_plan", {}) or {}
    master_tasks = task_plan.get("master_tasks", []) if isinstance(task_plan, dict) else []

    print("\n任务执行总结:")
    for task in master_tasks:
        status_icon = {
            "completed": "✅",
            "in_progress": "🔄",
            "pending": "🕒",
            "skipped": "⏭️",
            "failed": "❌",
        }.get(task.get("status", ""), "•")
        print(f"  {status_icon} {task.get('title', task.get('id', ''))} ({task.get('status', '')})")

    print("\n流程结束。")


if __name__ == "__main__":
    asyncio.run(main())
