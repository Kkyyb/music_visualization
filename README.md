# AI 音乐可视化场景生成系统

将「歌曲 + 歌词 + 场景 + 心情」自动转为可编辑的 Blender 3D 场景。

项目采用 **LangGraph 双阶段架构**：
- **Workflow 阶段**：歌词检索 → Prompt 生成 → 图像生成与质检 → SAM 分割规划与 3D 建模
- **Agent 阶段**：分层任务规划 → 子任务动作生成与执行 → 视觉评估回路

---

## 1. 项目目标

输入歌曲信息后，系统自动完成以下链路：
1. 生成具备可建模约束的场景图片（FLUX）
2. 基于图片输出 SAM 分割计划（JSON）
3. 按对象执行 SAM 3D，下载 GLB 模型
4. 在 Blender 中按层级任务逐步搭建场景并闭环评估

---

## 2. 技术栈

- 编排：`LangGraph`
- 文本/视觉模型：`Qwen-Max` / `Qwen-Plus` / `Qwen-VL-Plus`
- 文生图：`fal-ai/flux-2-pro`
- 分割建模：`fal-ai/sam-3/3d-objects` + `fal-ai/sam-3/image`
- DCC 执行：`Blender MCP` (`uvx blender-mcp`)
- 语言：Python 3.12+

关键配置见：[config.py](config.py)

---

## 3. 系统架构

### 3.1 Graph 主流程（当前版本）

图定义见：[graph.py](graph.py)

**Workflow 6 节点（线性）**
1. `search_lyrics`
2. `generate_prompt`
3. `generate_image`
4. `analyze_image`
5. `plan_sam_segmentation`
6. `sam3d_modeling`

**Agent 4 节点（循环）**
7. `generate_task_list`
8. `generate_command`
9. `execute_command`
10. `evaluate`

`generate_command -> execute_command -> evaluate -> generate_command` 循环，直到无剩余 pending/in_progress 子任务。

### 3.2 全局状态

状态结构见：[state.py](state.py)

核心字段：
- 输入：`song_name / artist / scene / mood / lyrics`
- 图像链路：`image_prompt / image_local_path / scene_task_doc`
- SAM：`sam_segmentation_plan / sam3d_result / blender_objects`
- Agent：`task_plan / planning_context / current_action / current_command / execution_log`
- 观测：`token_log / messages`

---

## 4. Workflow 关键设计

节点实现见：[nodes/workflow_nodes.py](nodes/workflow_nodes.py)

### 4.1 Prompt 约束生成（`generate_prompt`）

该节点不是“自由创作 Prompt”，而是注入强约束：
- isometric/axonometric diorama 视角
- 主体必须完整落入画幅，四周预留边距
- 禁止物体触边、越出底座 footprint
- 控制对象预算，抑制碎物体过多
- 非固定房间模板，允许开放场景
- 相机要求低透视、无景深

### 4.2 图像质检闭环（`generate_image` + `_quality_gate_image`）

同一张图最多多轮重生成：
1. 首轮使用原始 Prompt
2. 未通过质检则叠加更强约束 suffix 重试
3. 质检由视觉模型判定 JSON：
   - 触边/越界
   - 树枝越出底座
   - 小碎物体过密
   - 固定 cutaway 房间模板倾向

### 4.3 SAM 分割计划（`plan_sam_segmentation`）

输出结构化 JSON（对象级）：
- `label / batch / priority / method / retry_methods / depends_on`
- `global_policy`：`max_objects / batch_order / default_thresholds`

并执行三层增强：
1. 计划归一化（字段清洗、方法纠偏）
2. 乔木策略（默认整树建模，避免 trunk/leaves 双失败）
3. 小物体补漏（二次识别补 `book/vinyl/umbrella` 等）
4. 语义去重（tree/canopy/trunk 等归一 key）

---

## 5. SAM 建模执行策略

执行器见：[tools/sam3d_plan_executor.py](tools/sam3d_plan_executor.py)

### 5.1 三阶段执行

1. **读取并去重 plan**（执行前再次兜底去重）
2. **按对象逐个执行**（`batch + priority` 排序）
3. **成功后下载 GLB** 到 `model_generation/`

### 5.2 多方法与重试

每个对象按 `retry_methods` 顺序尝试：
- `prompt`
- `box_prompts`
- `mask_urls`

每种方法内部再按阈值重试（`default_thresholds`，如 `None/0.2/0.1`）。

### 5.3 懒物化补参（box/mask）

如果当前方法缺参，不会直接失败，而是按需调用 2D SAM：
- 模型：`fal-ai/sam-3/image`
- 先取 candidate masks/boxes/scores
- `mask_urls`：直接填充最佳 mask URL
- `box_prompts`：
  - 优先用 2D 直接 box（稳定性判定）
  - 不稳定则从 mask PNG 回推 bbox

调试产物写入：
- `debug_snapshots/sam_guidance/<timestamp>/...json`
- 对应 mask PNG（若下载成功）

### 5.4 失败处理

- 对象建模失败：写入 `failures`，继续后续对象
- 返回无 glb：写入 `failures`
- glb 下载失败：自动重试 1 次；二次仍失败写入 `failures`，不中断流程

> 说明：旧模块 [sam3d_tools.py](tools/sam3d_tools.py) 保留为早期实现，当前主流程使用 `sam3d_plan_executor.py`。

---

## 6. Agent 阶段执行闭环

节点实现见：[nodes/agent_nodes.py](nodes/agent_nodes.py)

### 6.1 层级任务计划（`generate_task_list`）

先采集 Blender 上下文（scene_info + screenshot），再生成层级 JSON：
- `master_tasks`
- `subtasks`（`import_and_align` / `scene_adjustment`）
- `status`、`priority`、`success_criteria`

若规划 JSON 不可用，自动落回 deterministic fallback 计划。

### 6.2 动作选择（`generate_command`）

只负责“选当前动作”，不直接产 bpy：
- 定位当前 subtask
- 维护状态机（pending -> in_progress）
- 构造 `current_action`（目标对象、是否需导入、尝试次数、成功标准）

### 6.3 执行（`execute_command`）

- 若 `requires_import=True`，先逐对象导入并更新 label->blender_name 映射
- 刷新 scene context（scene_info + screenshot）
- 收集目标对象与锚点对象 summary
- 调用 LLM 生成单步 bpy 脚本并执行

代码生成有硬规则校验：
- 禁止删场景/重置/再次导入
- 禁止占位值
- 禁止直接照搬 reconstruction translation 到 Blender 世界坐标

### 6.4 评估（`evaluate`）

- 结合执行日志 + 多模态评估判定当前 subtask 是否完成
- 失败按 `MAX_RETRIES=3` 重试，超限标记 `skipped`
- 全部任务状态由子任务状态聚合

Blender MCP 封装见：[tools/blender_tools.py](tools/blender_tools.py)

---

## 7. 成本与观测

成本追踪见：[token_tracker.py](token_tracker.py)

- LLM token 成本：按 `PRICE_TABLE` 计费
- 固定成本：
  - FLUX 2 Pro 单次
  - SAM 3D 按对象数计费

每个节点调用后实时记录，并在流程末输出汇总。

---

## 8. 目录说明

- `nodes/`：Workflow 与 Agent 节点
- `tools/`：图像、分割、Blender、搜索等工具封装
- `debug_snapshots/`：阶段快照与 SAM 引导调试数据
- `image_output/`：生成图片输出
- `model_generation/`：下载的 GLB 模型
- `debug_workflow_run.py`：分阶段调试入口
- `main.py`：交互式主入口

---

## 9. 环境准备

### 9.1 安装依赖

```bash
uv sync
```

或使用 `pip` 自行安装 `pyproject.toml` 中依赖。

### 9.2 配置环境变量

复制项目根目录的 `.env.example` 为 `.env`，再填入自己的 API 密钥。`.env` 已加入 Git 忽略规则：

```env
QWEN_KEY=your_qwen_api_key
FAL_KEY=your_fal_api_key
```

---

## 10. 运行方式

### 10.1 全流程运行

```bash
python main.py
```

### 10.2 分阶段调试（强烈推荐）

入口见：[debug_workflow_run.py](debug_workflow_run.py)

可用阶段名：
- `search_lyrics`
- `generate_prompt`
- `generate_image`
- `analyze_image`
- `plan_sam_segmentation`
- `sam3d_modeling`
- `generate_task_list`

> 注意：命令参数里使用阶段名，不是函数名（例如 `sam3d_modeling`，不是 `sam3d_modeling_node`）。

示例（PowerShell）：

```powershell
.venv\Scripts\python.exe .\debug_workflow_run.py --end-stage generate_image
```

```powershell
.venv\Scripts\python.exe .\debug_workflow_run.py --snapshot-in .\debug_snapshots\20260403_135954_plan_sam_segmentation.json --start-stage sam3d_modeling --end-stage sam3d_modeling
```

---

## 11. 快照与可复现调试

每个阶段执行后会自动落地快照：
- 文件命名：`debug_snapshots/<timestamp>_<stage>.json`
- 可从任意快照恢复，跳过前序 API 调用

SAM 参数物化额外调试目录：
- `debug_snapshots/sam_guidance/<timestamp>/`
- 包含候选参数、补参结果、mask 文件

---

## 12. 已知限制与下一步

- SAM 3D 仍可能受输入图像复杂度影响，尤其是遮挡严重对象。
- 当前对象建模是串行执行（并发未启用）。
- Agent 执行目前是单动作循环，后续可引入更强的任务层级记忆与自适应重规划。

---

## 13. 维护建议

- 新增节点时，务必同步更新：
  - [graph.py](graph.py)
  - [debug_workflow_run.py](debug_workflow_run.py)
  - [state.py](state.py)
- 调参优先使用快照链路，避免重复消耗文生图和建模费用。
