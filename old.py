import asyncio
from typing import Optional
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

from langchain.agents import create_agent
from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, AnyMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from tripo3d import TripoClient

load_dotenv()

class Music_info(BaseModel):
    music_style: str = Field(description = '歌词的风格，如：甜蜜浪漫、轻松活泼等')
    music_mood: str = Field(description = '音乐给人的情绪感受，如：悲伤、欢快、愉悦等')
    music_genre: str = Field(description = '音乐流派，如：流行歌曲')
    music_era: str = Field(description = '音乐出现的年代，如：2010年代')
    music_writer: str = Field(description = '音乐作者')

class Blender_info(BaseModel):
    scene_ground: str = Field(description = '关于地面操作的指令')
    scene_light: str = Field(description = '关于灯光设置的指令')
    scene_dynamic_elements: str = Field(description = '关于动态元素的指令')
    scene_decorations: str = Field(description = '关于装饰设置的指令')

llm = ChatOpenAI(
    model = "qwen-max",
    api_key= os.getenv('QWEN_KEY'),
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

async def create_search_agent(music_name : str):
    websearch_client = MultiServerMCPClient(
        {
            "zhipu-web-search-sse": {
                "url": "https://open.bigmodel.cn/api/mcp/web_search/sse?Authorization=5c8aa377ad4b459fb750149ea0a20c09.17hbeBdhYfGqMjpu",
                "transport": "sse"
            },
        }
    )

    tools = await websearch_client.get_tools()
    websearch_agent = create_agent(
        llm,
        tools=tools,
        response_format= ToolStrategy(Music_info),
        system_prompt= f'你是一名认真的音乐爱好者，现在要根据音乐名称：{music_name}，使用web_search工具搜索其风格、情绪、流派、年代、作者等信息，并提取Music_info字段。'
    )

    result = await websearch_agent.ainvoke({'messages': [{'role': 'user', 'content': music_name}]})
    return result['structured_response']

def scene_description(music_info: str):
    scene_description_agent = create_agent(
        llm,
        system_prompt = f'''
        - 你是一位想象力丰富的'舞台or场景策划师'
        - 根据提供的音乐信息：<music_info>{music_info}</music_info>，生成一段符合音乐风格、氛围、年代的场景描述。
        - 要求场景描述包含场地特征描述、场景中人物特征描述（也可以没有人物）、场景中心焦点区域细节描述、整体氛围感描述
        - 只需返回场景描述结果
        - 示例：一个充满神秘深邃感的 70年代摇滚露天音乐节场地。场地是大型室外空间，地面为粗糙沙地，视野广阔，四周有简单搭建的围栏。人群中，人们穿着皮革制成的服饰和牛仔布料的衣物。舞台是金属框架搭建上面乐器的金属部件熠熠生辉。整个场地弥漫着神秘未知又自由不羁的氛围，呈现出叛逆反抗、激情四溢的风格，细节丰富、逼真，画面光线闪烁，好似置身于 70 年代经典复古的摇滚现场。
        '''
    )

    result = scene_description_agent.invoke({'messages': [{'role': 'user', 'content': music_info}]})
    return result['messages'][-1].content

def tripo_agent(scene_generation: str):
    tripo_description_agent = create_agent(
        llm,
        system_prompt = '''
        - 你是一位经验丰富的的'建模专家'，使用简体中文回答
        - 根据输入的场景描述，想象一个可以体现这种氛围的悬浮沙盘模型（形状可以是矩形、圆形、不规则形等，要根据场景描述想象），生成在 TripoAI 中使用的 prompt，描述沙盘模型上的物体，要求prompt内容不必太细致，减少氛围或感受的描述，减少对远景的描述，只保留对物体的描述。字数在100字以内，言简意赅，准确描述。
        - 示例：一座漂浮的岛屿，岛屿上是一个充满活力的现代城市广场，有街道贯穿，四周环绕着玻璃幕墙高楼的轮廓，广场上有示意性的金属路灯杆和特色塑料垃圾桶的形态，低多边形风格。
        '''
    )

    result = tripo_description_agent.invoke({'messages': [{'role': 'user', 'content': f'场景描述:{scene_generation}'}]})
    return result['messages'][-1].content

async def model_genaration(prompt):
    async with TripoClient(api_key=os.getenv('TRIPO_KEY')) as client:
        from tripo3d import TaskStatus

        task_id = await client.text_to_model(
            prompt= prompt,
            negative_prompt="",
        )
        print(f"Task ID: {task_id}")

        task = await client.wait_for_task(task_id, verbose=True)
        if task.status == TaskStatus.SUCCESS:
            files = await client.download_task_models(task, "D:/Python/music_visualization/model_generation")
            for model_type, path in files.items():
                print(f"Downloaded {model_type}: {path}")

def blender_instruction(scene_description: str, tripo_instruction: str):
    blender_description_agent = create_agent(
        llm,
        # response_format = ToolStrategy(Blender_info),
        system_prompt = f'''
        - 你是一位经验丰富的的'建模专家'，使用简体中文回答
        - 现在已经有了模型，根据'场景描述'以及'tripo建模提示词'内容，希望你为现有模型添加一些动态效果(例如镜头的旋转、动态颗粒的添加等等，具体情况要根据场景描述选择动态效果)，并生成对应的可给 blender-mcp 使用的多步骤建模操作指令，减少氛围或感受的描述，只保留操作指令
        - 指令应言简意赅，专业且准确
        - 允许从POLY HAVEN下载素材需要的素材，但要确保素材与主体模型比例一致
        '''
    )

    result = blender_description_agent.invoke({'messages': [{'role': 'user', 'content': f'场景描述:{scene_description}，Tripo建模提示词:{tripo_instruction}'}]})
    return result['messages'][-1].content

async def blender_agent(instruction: str):
    model = ChatOpenAI(
        model = "qwen-max",
        api_key= os.getenv('QWEN_KEY'),
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    belnder_client = MultiServerMCPClient(
        {"blender": {
            "command": "uvx",
            "args": ["blender-mcp"],
            "transport": "stdio"
            }
        }
    )

    tools = await belnder_client.get_tools()
    blender_operator = create_deep_agent(
        model = model,
        tools = tools,
        system_prompt = f'''
            - 你是一名专业的blender建模师
            - 根据输入的blender指令，调用blender_mcp在blender中进行操作，不需要参考引用资料，不需要markdown语法输出
            - 确保所有模型比例符合逻辑
            - blender_mcp 支持从POLY HAVEN下载素材
             ''',
        checkpointer=InMemorySaver()
    )

    '''await blender_operator.ainvoke({'messages': [{'role': 'user', 'content': '在"D:/Python/music_visualization/model_generation"中读取主体模型文件以及材质，并放入blender场景中'}]})

    for step in instruction_list:
        result = await blender_operator.ainvoke({'messages': [{'role': 'user', 'content': step}]},{"recursion_limit": 1000})
        print (result['messages'][-1].content, '\n')'''

    config = {"configurable": {"thread_id": "main"}, "recursion_limit": 1000}

    async for chunk in blender_operator.astream(
        {'messages':[{'role':'user', 'content':instruction}]},
        config,
        stream_mode="updates"
    ):
        for step, data in chunk.items():
            print(f"step: {step}")
            print(f"content: {data['messages'][-1].content_blocks}")


    while True:
        print('输入quit退出交互模式')
        input_msg = input('请输入修改意见：')
        if input_msg.lower() in {'quit', 'exit', '退出'}:
            break
        if not input_msg:
            continue
        user_instruction = HumanMessage(content = input_msg)

        await blender_operator.ainvoke(user_instruction, config)


async def main():

    music_name = input('输入歌曲名：')
    print('=========歌曲信息搜索中=========')
    music_info = await create_search_agent(music_name)
    print(music_info,'\n')

    print('=========场景描述生成中=========')
    scene_generation = scene_description(f'{music_info}')
    print(scene_generation,'\n')

    print('=========tripo描述生成中=========')
    tripo_prompt = tripo_agent(f'{scene_generation}')
    print(tripo_prompt,'\n')
    # await model_genaration(tripo_prompt)

    print('=========blender操作指令生成中=========')
    insturctions = blender_instruction(f'{scene_generation}', f'{tripo_prompt}')
    # instruction_list = insturctions.scene_ground, insturctions.scene_light, insturctions.scene_dynamic_elements, insturctions.scene_decorations
    print(insturctions,'\n')

    '''print('=========blender_agent运行中=========')
    await blender_agent(instruction_list)'''



if __name__ == "__main__":
    asyncio.run(main())
