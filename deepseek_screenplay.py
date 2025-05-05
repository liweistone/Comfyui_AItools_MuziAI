import json
import requests
import comfy
import folder_paths
from aiohttp import web
from nodes import PromptServer

# 配置管理（API密钥安全存储）
class DeepSeekConfig:
    @classmethod
    def get_api_key(cls):
        return getattr(cls, '_api_key', None)

    @classmethod
    def set_api_key(cls, api_key):
        cls._api_key = api_key

# API请求模块
class DeepSeekAPI:
    @staticmethod
    def generate(prompt, config, max_tokens=2000):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DeepSeekConfig.get_api_key()}"
        }

        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "你是一个专业的短视频编剧和分镜师"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": max_tokens
        }

        try:
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"API Error: {str(e)}"

# 第一个节点：脚本和分镜生成
class DeepSeekScriptNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "theme": ("STRING", {"multiline": True, "default": "科技与未来"}),
                "duration": ("INT", {"default": 60, "min": 15, "max": 300}),
                "style": (["专业解说", "幽默搞笑", "情感故事", "产品测评"],),
                "max_tokens": ("INT", {"default": 2000, "min": 500, "max": 4000})
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("script", "storyboard")
    FUNCTION = "generate"
    CATEGORY = "🎨公众号懂AI的木子做号工具/DeepSee剧本生成"
    OUTPUT_NODE = True

    def generate(self, theme, duration, style, max_tokens):
        prompt = f"""请生成一个{duration}秒的{style}风格短视频：
主题：{theme}
要求：
1. 包含完整旁白脚本
2. 按秒数划分镜头
3. 每个镜头注明画面要素
4. 包含转场方式建议"""

        result = DeepSeekAPI.generate(prompt, max_tokens)
        
        # 分割脚本和分镜
        parts = result.split("【分镜描述】")
        script = parts[0].replace("【视频脚本】", "").strip()
        storyboard = parts[1].strip() if len(parts) > 1 else ""
        
        return (script, storyboard)

# 第二个节点：分镜提示词生成
class StoryboardPromptNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "script": ("STRING", {"forceInput": True}),
                "storyboard": ("STRING", {"forceInput": True}),
                "art_style": (["写实风格", "卡通渲染", "赛博朋克", "水墨风格"],),
                "detail_level": ("INT", {"default": 3, "min": 1, "max": 5})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompts",)
    FUNCTION = "generate_prompts"
    CATEGORY = "🎨公众号懂AI的木子做号工具/DeepSee剧本生成"
    OUTPUT_NODE = True

    def generate_prompts(self, script, storyboard, art_style, detail_level):
        prompt = f"""根据以下脚本和分镜生成AI绘画提示词：
脚本摘要：{script[:500]}
分镜内容：{storyboard}
要求：
1. 使用{art_style}
2. 详细程度等级：{detail_level}/5
3. 包含画面要素、构图、灯光、色彩等参数
4. 使用英文逗号分隔关键词"""

        result = DeepSeekAPI.generate(prompt, max_tokens=1000)
        return (result,)

# 配置节点
class DeepSeekConfigNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""})
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "config"
    CATEGORY = "🎨公众号懂AI的木子做号工具/DeepSee剧本生成"
    OUTPUT_NODE = True

    def config(self, api_key):
        DeepSeekConfig.set_api_key(api_key)
        return ()

# 注册节点
NODE_CLASS_MAPPINGS = {
    "DeepSeekScriptNode": DeepSeekScriptNode,
    "StoryboardPromptNode": StoryboardPromptNode,
    "DeepSeekConfigNode": DeepSeekConfigNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DeepSeekScriptNode": "📖DeepSeek 脚本生成",
    "StoryboardPromptNode": "📖分镜提示词生成",
    "DeepSeekConfigNode": "📖DeepSeek 配置"
}