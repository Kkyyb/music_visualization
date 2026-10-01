"""
工具：FLUX 2 Pro 文生图 (via fal.ai)
"""
from __future__ import annotations

import os
import httpx
import fal_client

from config import FAL_KEY, FAL_IMAGE_MODEL, IMAGE_OUTPUT_DIR
from token_tracker import tracker

# 设置 fal 认证
os.environ["FAL_KEY"] = FAL_KEY


async def generate_image(prompt: str, filename: str = "scene.png") -> str:
    """
    使用 FLUX 2 Pro 生成图片。

    Args:
        prompt: 图片生成 Prompt (英文效果更佳)
        filename: 保存的文件名

    Returns:
        本地图片路径
    """
    print(f"  🎨 正在生成图片 (FLUX 2 Pro)...")

    result = await fal_client.subscribe_async(
        FAL_IMAGE_MODEL,
        arguments={
            "prompt": prompt,
            "image_size": {"width": 1536, "height": 1536},       # 16:9 横屏
            "num_images": 1,
            "output_format": "png",
            "enable_safety_checker": False,
        },
    )

    # 下载图片到本地
    image_url = result["images"][0]["url"]
    local_path = os.path.join(IMAGE_OUTPUT_DIR, filename)

    async with httpx.AsyncClient() as http:
        resp = await http.get(image_url)
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(resp.content)

    tracker.record_image_gen("图片生成(FLUX 2 Pro)")
    print(f"  ✅ 图片已保存: {local_path}")

    return local_path
