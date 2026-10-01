"""
工具：联网搜索歌词
使用 DashScope 内置联网搜索插件，一次调用完成搜索+整理。
"""
from __future__ import annotations

from openai import AsyncOpenAI
from config import QWEN_KEY, QWEN_BASE_URL, MODEL_BALANCED
from token_tracker import tracker


async def search_lyrics(song_name: str, artist: str) -> tuple[str, dict]:
    """
    使用 Qwen 内置联网搜索歌词。

    Returns:
        (lyrics_text, raw_response)
    """
    client = AsyncOpenAI(
        api_key=QWEN_KEY,
        base_url=QWEN_BASE_URL,
    )

    response = await client.chat.completions.create(
        model=MODEL_BALANCED,
        messages=[
            {
                "role": "system",
                "content": "你是一位音乐爱好者，请根据用户提供的歌曲名和歌手，搜索并返回该歌曲的完整歌词。只返回歌词内容，不要额外说明。",
            },
            {
                "role": "user",
                "content": f"搜索歌曲《{song_name}》（{artist}）的完整歌词",
            },
        ],
        extra_body={"enable_search": True},
    )

    lyrics_text = response.choices[0].message.content
    usage = response.usage
    input_tokens = usage.prompt_tokens if usage else 0
    output_tokens = usage.completion_tokens if usage else 0

    tracker.record_llm(
        node_name="歌词搜索",
        model=MODEL_BALANCED,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    print(lyrics_text[:60])
    return lyrics_text, {}
