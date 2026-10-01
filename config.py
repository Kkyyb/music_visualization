"""
项目配置：环境变量、模型常量、价格表
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────
QWEN_KEY = os.getenv("QWEN_KEY", "")
FAL_KEY = os.getenv("FAL_KEY", "")

# ── Qwen 模型 ────────────────────────────────────────────
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 各节点使用的模型
MODEL_CREATIVE = "qwen3.6-plus"            # 创意写作 / 复杂推理
MODEL_BALANCED = "qwen-plus"           # 性价比文本任务
MODEL_VISION   = "qwen3.6-plus"        # 多模态视觉理解

# ── fal.ai ────────────────────────────────────────────────
FAL_IMAGE_MODEL  = "fal-ai/flux-2-pro"           # FLUX 2 Pro 文生图
FAL_SAM3D_MODEL  = "fal-ai/sam-3/3d-objects"      # SAM 3D Objects
FAL_SAM2D_MODEL  = "fal-ai/sam-3/image"           # SAM 2D 分割（产 mask/boxes）

# ── 价格表 (¥ / 百万 Token) ───────────────────────────────
PRICE_TABLE = {
    "qwen-max": {
        "input":  2.4,     # ¥/百万Token (0-32K 档)
        "output": 9.6,     # 含 CoT
    },
    "qwen-plus": {
        "input":  0.8,
        "output": 2.0,
    },
    "qwen-vl-plus": {
        "input":  1.0,
        "output": 9.6,
    },
}

# ── 固定费用 (¥) ──────────────────────────────────────────
PRICE_FLUX2_PRO  = 0.36    # $0.05 ≈ ¥0.36
PRICE_SAM3D_EACH = 0.144   # $0.02 ≈ ¥0.144

# ── 项目路径 ──────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_OUTPUT_DIR = os.path.join(PROJECT_DIR, "model_generation")
IMAGE_OUTPUT_DIR = os.path.join(PROJECT_DIR, "image_output")

# 确保输出目录存在
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
