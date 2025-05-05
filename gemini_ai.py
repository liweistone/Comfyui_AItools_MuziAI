import os
import comfy
import torch
import numpy as np
import logging
import time
import asyncio
from PIL import Image
import google.generativeai as genai
import requests
from io import BytesIO
import base64

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=os.path.join(os.path.dirname(__file__), 'gemini_debug.log')
)

#===========Gemini视频提示生成器=============
class GeminiImageAnalyzer:
    """
    🎬 Gemini视频提示生成器 🎬
    功能：分析视频帧图像并生成高质量的视频生成提示词
    版本：2.2
    开发者：懂AI的木子
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "输入视频帧图像，将用于生成视频提示词"
                }),
                "model_name": ([
                "gemini-1.5-flash", 
                "gemini-1.5-pro", 
                "gemini-1.5-flash-8b", 
                "gemini-2.0-flash-exp",
                "gemini-2.0-flash-lite", 
                "gemini-2.0-flash-exp-image-generation",
                "gemini-2.0-flash-thinking-exp-01-21", 
                "gemini-2.0-flash", 
                "gemini-2.0-pro", 
                "gemini-2.5-flash-preview-04-17"
                "gemini-2.5-pro-preview",
                "gemini-2.5-pro-preview-03-25", 
                "gemini-2.5-flash", 
                "gemini-2.5-pro", 
                ], {
                    "default": "gemini-1.5-pro",
                    "tooltip": "选择Gemini模型版本"
                }),
                "max_tokens": ("INT", {
                    "default": 350, 
                    "min": 100, 
                    "max": 6000,
                    "tooltip": "控制生成内容的长度"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7, 
                    "min": 0, 
                    "max": 1, 
                    "step": 0.05,
                    "tooltip": "控制生成内容的随机性(0=确定性高,1=创造性高)"
                }),
                "timeout": ("INT", {
                    "default": 30, 
                    "min": 30, 
                    "max": 300,
                    "tooltip": "API请求超时时间(秒)"
                }),
                "api_key": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "tooltip": "输入Gemini API密钥"
                }),
            },
            "optional": {
                "custom_prompt": ("STRING", {
                    "default": "请仔细分析这张视频帧图片，为我生成文生视频的提示词，你要做最大程度的详细分析，请你请根据画面内容，为视频添加合适的动效及视觉效果，你的分析要完美的表现视频质量及视频效果。请同时提供中文和英文版本，用以下格式输出：\n\n[中文开始]\n(中文内容)\n[中文结束]\n\n[英文开始]\n(英文内容)\n[英文结束]", 
                    "multiline": True,
                    "tooltip": "自定义分析指令"
                }),
                "proxy_url": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "tooltip": "代理服务器地址(如需)"
                }),
                "help_text": ("STRING", {
                    "default": """📌 Gemini视频提示生成器使用指南

功能说明：
分析视频帧图像并生成高质量的视频生成提示词，适用于文生视频工作流。

🔧 核心参数：
- image: 输入视频帧图像
- model_name: 选择Gemini模型版本
- max_tokens: 控制输出长度(100-6000)
- temperature: 控制创造性(0-1)
- timeout: API超时时间(30-300秒)
- api_key: Gemini API密钥

🎯 可选参数:
- custom_prompt: 自定义分析指令
- proxy_url: 代理服务器地址

💡 使用场景：
- 视频内容创作
- 动态效果设计
- 视频分镜脚本生成

📝 输出说明：
- prompt_cn: 生成的视频提示词(中文)
- prompt_en: 生成的视频提示词(英文)

更多AI工具请关注公众号: 懂AI的木子
                    """,
                    "multiline": True
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING","STRING",)
    RETURN_NAMES = ("prompt_cn","prompt_en",)
    FUNCTION = "analyze_image"
    CATEGORY = "🎨公众号懂AI的木子做号工具/gemini提示词增强"
    OUTPUT_NODE = True

    def __init__(self):
        self.initialized = False
        self.session = requests.Session()

    def initialize_model(self, api_key, proxy_url=""):
        if not self.initialized:
            genai.configure(
                api_key=api_key,
                transport="rest"
            )
            self.initialized = True
            logging.info("Gemini模型初始化完成")

    def preprocess_image(self, image):
        try:
            image_np = 255. * image[0].cpu().numpy()
            pil_image = Image.fromarray(np.clip(image_np, 0, 255).astype(np.uint8))
            
            max_size = 768
            if max(pil_image.size) > max_size:
                ratio = max_size / max(pil_image.size)
                new_size = (int(pil_image.width * ratio), int(pil_image.height * ratio))
                pil_image = pil_image.resize(new_size, Image.LANCZOS)
                logging.info(f"图像已优化至 {new_size}")
            
            return pil_image
        except Exception as e:
            logging.error(f"图像预处理失败: {str(e)}")
            raise

    def parse_response(self, text):
        """从响应文本中提取中英文内容"""
        try:
            cn_start = text.find("[中文开始]")
            cn_end = text.find("[中文结束]")
            en_start = text.find("[英文开始]")
            en_end = text.find("[英文结束]")
            
            if cn_start == -1 or cn_end == -1 or en_start == -1 or en_end == -1:
                return text, text  # 如果格式不正确，返回相同内容
            
            cn_content = text[cn_start+6:cn_end].strip()
            en_content = text[en_start+6:en_end].strip()
            
            return cn_content, en_content
        except Exception as e:
            logging.error(f"解析响应失败: {str(e)}")
            return text, text

    def analyze_image(self, image, model_name, max_tokens, temperature, timeout, api_key, custom_prompt="", proxy_url="", help_text="", **kwargs):
        try:
            # 初始化模型
            self.initialize_model(api_key, proxy_url)
            
            # 图像预处理
            pil_image = self.preprocess_image(image)
            
            # 准备提示词
            prompt_parts = [custom_prompt if custom_prompt.strip() else "请分析此视频帧并生成视频生成提示，同时提供中文和英文版本，用以下格式输出：\n\n[中文开始]\n(中文内容)\n[中文结束]\n\n[英文开始]\n(英文内容)\n[英文结束]", pil_image]
            
            # 调用API
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                prompt_parts,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature
                ),
                request_options={"timeout": timeout}
            )
            
            if not response.text:
                raise ValueError("API返回空响应")
                
            # 解析响应
            cn_content, en_content = self.parse_response(response.text)
            return (cn_content, en_content)
            
        except Exception as e:
            error_msg = f"处理失败: {str(e)}"
            logging.error(error_msg, exc_info=True)
            return (error_msg, error_msg)


#===================Gemini图片提示生成器========================
class GeminiPromptGenerator:
    """
    🖼️ Gemini图片提示生成器 🖼️
    功能：分析图像并生成高质量的文生图提示词
    版本：2.2
    开发者：懂AI的木子
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "输入图像，将用于生成提示词"
                }),
                "model_name": ([
                "gemini-1.5-flash", 
                "gemini-1.5-pro", 
                "gemini-1.5-flash-8b", 
                "gemini-2.0-flash-exp",
                "gemini-2.0-flash-lite", 
                "gemini-2.0-flash-exp-image-generation",
                "gemini-2.0-flash-thinking-exp-01-21", 
                "gemini-2.0-flash", 
                "gemini-2.0-pro", 
                "gemini-2.5-flash-preview-04-17"
                "gemini-2.5-pro-preview",
                "gemini-2.5-pro-preview-03-25", 
                "gemini-2.5-flash", 
                "gemini-2.5-pro", 
                ], {
                    "default": "gemini-1.5-flash", 
                    "tooltip": "选择Gemini模型版本"
                }),
                "max_tokens": ("INT", {
                    "default": 1000, 
                    "min": 100, 
                    "max": 4000,
                    "tooltip": "控制生成内容的长度"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7, 
                    "min": 0, 
                    "max": 1, 
                    "step": 0.05,
                    "tooltip": "控制生成内容的随机性(0=确定性高,1=创造性高)"
                }),
                "timeout": ("INT", {
                    "default": 30, 
                    "min": 30, 
                    "max": 300,
                    "tooltip": "API请求超时时间(秒)"
                }),
                "api_key": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "tooltip": "输入Gemini API密钥"
                }),
            },
            "optional": {
                "custom_prompt": ("STRING", {
                    "default": "请详细分析这张图片并为我生成高质量的文生图提示词。要求：1. 详细描述画面内容 2.包含艺术风格 3. 描述光照和色彩 4. 包含画面氛围 5. 请同时提供中文和英文版本，用以下格式输出：\n\n[中文开始]\n(中文内容)\n[中文结束]\n\n[英文开始]\n(英文内容)\n[英文结束]", 
                    "multiline": True,
                    "tooltip": "自定义分析指令"
                }),
                "proxy_url": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "tooltip": "代理服务器地址(如需)"
                }),
                "help_text": ("STRING", {
                    "default": """📌 Gemini图片提示生成器使用指南

功能说明：
分析图像并生成高质量的文生图提示词，适用于AI绘画工作流。

🔧 核心参数：
- image: 输入图像
- model_name: 选择Gemini模型版本
- max_tokens: 控制输出长度(100-4000)
- temperature: 控制创造性(0-1)
- timeout: API超时时间(30-300秒)
- api_key: Gemini API密钥

🎯 可选参数:
- custom_prompt: 自定义分析指令
- proxy_url: 代理服务器地址

💡 使用场景：
- AI绘画提示词生成
- 图像内容分析
- 艺术风格转换

📝 输出说明：
- prompt_en: 生成的图像提示词(英文)
- prompt_cn: 生成的图像提示词(中文)

更多AI工具请关注公众号: 懂AI的木子
                    """,
                    "multiline": True
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING","STRING",)
    RETURN_NAMES = ("prompt_en","prompt_cn",)
    FUNCTION = "generate_prompt"
    CATEGORY = "🎨公众号懂AI的木子做号工具/gemini提示词增强"
    OUTPUT_NODE = True

    def __init__(self):
        self.initialized = False
        self.session = requests.Session()

    def initialize_model(self, api_key, proxy_url=""):
        if not self.initialized:
            genai.configure(
                api_key=api_key,
                transport="rest"
            )
            self.initialized = True
            logging.info("Gemini模型初始化完成")

    def preprocess_image(self, image):
        try:
            image_np = 255. * image[0].cpu().numpy()
            pil_image = Image.fromarray(np.clip(image_np, 0, 255).astype(np.uint8))
            
            max_size = 1024
            if max(pil_image.size) > max_size:
                ratio = max_size / max(pil_image.size)
                new_size = (int(pil_image.width * ratio), int(pil_image.height * ratio))
                pil_image = pil_image.resize(new_size, Image.LANCZOS)
                logging.info(f"图像已优化至 {new_size}")
            
            return pil_image
        except Exception as e:
            logging.error(f"图像预处理失败: {str(e)}")
            raise

    def parse_response(self, text):
        """从响应文本中提取中英文内容"""
        try:
            cn_start = text.find("[中文开始]")
            cn_end = text.find("[中文结束]")
            en_start = text.find("[英文开始]")
            en_end = text.find("[英文结束]")
            
            if cn_start == -1 or cn_end == -1 or en_start == -1 or en_end == -1:
                return text, text  # 如果格式不正确，返回相同内容
            
            en_content = text[en_start+6:en_end].strip()
            cn_content = text[cn_start+6:cn_end].strip()
            
            return en_content, cn_content
        except Exception as e:
            logging.error(f"解析响应失败: {str(e)}")
            return text, text

    def generate_prompt(self, image, model_name, max_tokens, temperature, timeout, api_key, custom_prompt="", proxy_url="", help_text="", **kwargs):
        try:
            # 初始化模型
            self.initialize_model(api_key, proxy_url)
            
            # 图像预处理
            pil_image = self.preprocess_image(image)
            
            # 准备提示词
            prompt_parts = [custom_prompt if custom_prompt.strip() else "详细分析这张图片并生成高质量的文生图提示词，同时提供中文和英文版本，用以下格式输出：\n\n[中文开始]\n(中文内容)\n[中文结束]\n\n[英文开始]\n(英文内容)\n[英文结束]", pil_image]
            
            # 调用API
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                prompt_parts,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature
                ),
                request_options={"timeout": timeout}
            )
            
            if not response.text:
                raise ValueError("API返回空响应")
                
            # 解析响应
            en_content, cn_content = self.parse_response(response.text)
            return (en_content, cn_content)
            
        except Exception as e:
            error_msg = f"处理失败: {str(e)}"
            logging.error(error_msg, exc_info=True)
            return (error_msg, error_msg)


# ====================提示词优化节点 ===========
class GeminiPromptOptimizer:
    """
    ✨ Gemini提示词优化器 ✨
    功能：优化和增强AI绘画提示词
    版本：2.2
    开发者：懂AI的木子
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_prompt": ("STRING", {
                    "default": "", 
                    "multiline": True,
                    "tooltip": "输入需要优化的提示词"
                }),
                "model_name": ([
                "gemini-1.5-flash", 
                "gemini-1.5-pro", 
                "gemini-1.5-flash-8b", 
                "gemini-2.0-flash-exp",
                "gemini-2.0-flash-lite", 
                "gemini-2.0-flash-exp-image-generation",
                "gemini-2.0-flash-thinking-exp-01-21", 
                "gemini-2.0-flash", 
                "gemini-2.0-pro", 
                "gemini-2.5-flash-preview-04-17"
                "gemini-2.5-pro-preview",
                "gemini-2.5-pro-preview-03-25", 
                "gemini-2.5-flash", 
                "gemini-2.5-pro", 
                ], {
                    "default": "gemini-1.5-flash",
                    "tooltip": "选择Gemini模型版本"
                }),
                "max_tokens": ("INT", {
                    "default": 1000, 
                    "min": 100, 
                    "max": 4000,
                    "tooltip": "控制生成内容的长度"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7, 
                    "min": 0, 
                    "max": 1, 
                    "step": 0.05,
                    "tooltip": "控制生成内容的随机性(0=确定性高,1=创造性高)"
                }),
                "timeout": ("INT", {
                    "default": 30, 
                    "min": 30, 
                    "max": 300,
                    "tooltip": "API请求超时时间(秒)"
                }),
                "api_key": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "tooltip": "输入Gemini API密钥"
                }),
                "optimization_level": (["basic", "detailed", "professional"], {
                    "default": "detailed",
                    "tooltip": "选择优化级别"
                }),
            },
            "optional": {
                "custom_instruction": ("STRING", {
                    "default": "你是一个专业的FLUX提示词工程师，请优化以下提示词。要求：1.添加合理的艺术化细节 2.保持长度在400字符以上 3.避免负面描述 4.请同时提供英文和中文版本，用以下格式输出：\n\n[英文开始]\n(英文内容)\n[英文结束]\n\n[中文开始]\n(中文内容)\n[中文结束]", 
                    "multiline": True,
                    "tooltip": "自定义优化指令"
                }),
                "proxy_url": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "tooltip": "代理服务器地址(如需)"
                }),
                "help_text": ("STRING", {
                    "default": """📌 Gemini提示词优化器使用指南

功能说明：
优化和增强AI绘画提示词，提高生成图像的质量和准确性。

🔧 核心参数：
- input_prompt: 输入需要优化的提示词
- model_name: 选择Gemini模型版本
- max_tokens: 控制输出长度(100-4000)
- temperature: 控制创造性(0-1)
- timeout: API超时时间(30-300秒)
- api_key: Gemini API密钥
- optimization_level: 优化级别(basic/detailed/professional)

🎯 可选参数:
- custom_instruction: 自定义优化指令
- proxy_url: 代理服务器地址

💡 使用场景：
- 提升AI绘画提示词质量
- 添加艺术化细节
- 专业级商业用途优化

📝 输出说明：
- optimized_prompt_en: 优化后的提示词(英文)
- optimized_prompt_cn: 优化后的提示词(中文)

更多AI工具请关注公众号: 懂AI的木子
                    """,
                    "multiline": True
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING","STRING",)
    RETURN_NAMES = ("optimized_prompt_en","optimized_prompt_cn",)
    FUNCTION = "optimize_prompt"
    CATEGORY = "🎨公众号懂AI的木子做号工具/gemini提示词增强"
    OUTPUT_NODE = True

    def __init__(self):
        self.initialized = False
        self.session = requests.Session()

    def initialize_model(self, api_key, proxy_url=""):
        if not self.initialized:
            genai.configure(
                api_key=api_key,
                transport="rest"
            )
            self.initialized = True
            logging.info("Gemini模型初始化完成")

    def parse_response(self, text):
        """从响应文本中提取中英文内容"""
        try:
            cn_start = text.find("[中文开始]")
            cn_end = text.find("[中文结束]")
            en_start = text.find("[英文开始]")
            en_end = text.find("[英文结束]")
            
            if cn_start == -1 or cn_end == -1 or en_start == -1 or en_end == -1:
                return text, text  # 如果格式不正确，返回相同内容
            
            en_content = text[en_start+6:en_end].strip()
            cn_content = text[cn_start+6:cn_end].strip()
            
            return en_content, cn_content
        except Exception as e:
            logging.error(f"解析响应失败: {str(e)}")
            return text, text

    def optimize_prompt(self, input_prompt, model_name, max_tokens, temperature, timeout, api_key, optimization_level="detailed", custom_instruction="", proxy_url="", help_text="", **kwargs):
        try:
            # 初始化模型
            self.initialize_model(api_key, proxy_url)
            
            # 根据优化级别调整提示
            level_instructions = {
                "basic": "请对以下提示词进行基础优化，主要改善语法和清晰度。请同时提供英文和中文版本，用以下格式输出：\n\n[英文开始]\n(英文内容)\n[英文结束]\n\n[中文开始]\n(中文内容)\n[中文结束]",
                "detailed": "请详细优化以下提示词，增加更多细节描述，同时保持原意。请同时提供英文和中文版本，用以下格式输出：\n\n[英文开始]\n(英文内容)\n[英文结束]\n\n[中文开始]\n(中文内容)\n[中文结束]",
                "professional": "请专业级优化以下提示词，使其适合商业用途，包含丰富的专业术语和精确描述。请同时提供英文和中文版本，用以下格式输出：\n\n[英文开始]\n(英文内容)\n[英文结束]\n\n[中文开始]\n(中文内容)\n[中文结束]"
            }
            
            base_instruction = custom_instruction if custom_instruction.strip() else level_instructions.get(optimization_level, "")
            
            # 准备完整提示
            full_prompt = f"{base_instruction}\n\n原始提示词：{input_prompt}"
            
            # 调用API
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature
                ),
                request_options={"timeout": timeout}
            )
            
            if not response.text:
                raise ValueError("API返回空响应")
                
            # 解析响应
            en_content, cn_content = self.parse_response(response.text)
            return (en_content, cn_content)
            
        except Exception as e:
            error_msg = f"提示词优化失败: {str(e)}"
            logging.error(error_msg, exc_info=True)
            return (error_msg, error_msg)


#======================Gemini聊天器=======================
class GeminiChatPro:
    """
    💬 Gemini聊天器 💬
    功能：与Gemini模型进行对话交互
    版本：2.2
    开发者：懂AI的木子
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "输入你想询问的内容"
                }),
                "model_name": ([
                "gemini-1.5-flash", 
                "gemini-1.5-pro", 
                "gemini-1.5-flash-8b", 
                "gemini-2.0-flash-exp",
                "gemini-2.0-flash-lite", 
                "gemini-2.0-flash-exp-image-generation",
                "gemini-2.0-flash-thinking-exp-01-21", 
                "gemini-2.0-flash", 
                "gemini-2.0-pro", 
                "gemini-2.5-flash-preview-04-17"
                "gemini-2.5-pro-preview",
                "gemini-2.5-pro-preview-03-25", 
                "gemini-2.5-flash", 
                "gemini-2.5-pro", 
                ], {
                    "default": "gemini-1.5-pro",
                    "tooltip": "选择Gemini模型版本"
                }),
                "max_tokens": ("INT", {
                    "default": 1000, 
                    "min": 100, 
                    "max": 4000,
                    "tooltip": "控制生成内容的长度"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7, 
                    "min": 0, 
                    "max": 1, 
                    "step": 0.05,
                    "tooltip": "控制生成内容的随机性(0=确定性高,1=创造性高)"
                }),
                "timeout": ("INT", {
                    "default": 30, 
                    "min": 30, 
                    "max": 300,
                    "tooltip": "API请求超时时间(秒)"
                }),
                "api_key": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "tooltip": "输入Gemini API密钥"
                }),
            },
            "optional": {
                "custom_instruction": ("STRING", {
                    "default": "请用中文和英文两种语言回答我的问题，格式如下：\n\n[中文开始]\n(中文回答)\n[中文结束]\n\n[英文开始]\n(英文回答)\n[英文结束]", 
                    "multiline": True,
                    "tooltip": "自定义对话指令"
                }),
                "proxy_url": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "tooltip": "代理服务器地址(如需)"
                }),
                "help_text": ("STRING", {
                    "default": """📌 Gemini聊天器使用指南

功能说明：
与Gemini模型进行对话交互，可用于多种AI相关任务。

🔧 核心参数：
- input_prompt: 输入你想询问的内容
- model_name: 选择Gemini模型版本
- max_tokens: 控制输出长度(100-4000)
- temperature: 控制创造性(0-1)
- timeout: API超时时间(30-300秒)
- api_key: Gemini API密钥

🎯 可选参数:
- custom_instruction: 自定义对话指令
- proxy_url: 代理服务器地址

💡 使用场景：
- AI相关问题咨询
- 创意内容生成
- 技术问题解答
- 提示词优化建议

📝 输出说明：
- response_cn: Gemini模型的回复内容(中文)
- response_en: Gemini模型的回复内容(英文)

更多AI工具请关注公众号: 懂AI的木子
                    """,
                    "multiline": True
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING","STRING",)
    RETURN_NAMES = ("response_cn","response_en",)
    FUNCTION = "chat_with_gemini"
    CATEGORY = "🎨公众号懂AI的木子做号工具/gemini提示词增强"
    OUTPUT_NODE = True

    def __init__(self):
        self.initialized = False
        self.session = requests.Session()

    def initialize_model(self, api_key, proxy_url=""):
        if not self.initialized:
            genai.configure(
                api_key=api_key,
                transport="rest"
            )
            self.initialized = True
            logging.info("Gemini模型初始化完成")

    def parse_response(self, text):
        """从响应文本中提取中英文内容"""
        try:
            cn_start = text.find("[中文开始]")
            cn_end = text.find("[中文结束]")
            en_start = text.find("[英文开始]")
            en_end = text.find("[英文结束]")
            
            if cn_start == -1 or cn_end == -1 or en_start == -1 or en_end == -1:
                return text, text  # 如果格式不正确，返回相同内容
            
            cn_content = text[cn_start+6:cn_end].strip()
            en_content = text[en_start+6:en_end].strip()
            
            return cn_content, en_content
        except Exception as e:
            logging.error(f"解析响应失败: {str(e)}")
            return text, text

    def chat_with_gemini(self, input_prompt, model_name, max_tokens, temperature, timeout, api_key, custom_instruction="", proxy_url="", help_text="", **kwargs):
        try:
            # 初始化模型
            self.initialize_model(api_key, proxy_url)
            
            # 准备完整提示
            full_prompt = f"{custom_instruction}\n\n用户输入：{input_prompt}" if custom_instruction.strip() else f"请用中文和英文两种语言回答我的问题，格式如下：\n\n[中文开始]\n(中文回答)\n[中文结束]\n\n[英文开始]\n(英文回答)\n[英文结束]\n\n我的问题是：{input_prompt}"
            
            # 调用API
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature
                ),
                request_options={"timeout": timeout}
            )
            
            if not response.text:
                raise ValueError("API返回空响应")
                
            # 解析响应
            cn_content, en_content = self.parse_response(response.text)
            return (cn_content, en_content)
            
        except Exception as e:
            error_msg = f"对话失败: {str(e)}"
            logging.error(error_msg, exc_info=True)
            return (error_msg, error_msg)


NODE_CLASS_MAPPINGS = {
    "GeminiImageAnalyzer": GeminiImageAnalyzer,
    "GeminiPromptGenerator": GeminiPromptGenerator,
    "GeminiPromptOptimizer": GeminiPromptOptimizer,
    "GeminiChatPro": GeminiChatPro
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiImageAnalyzer": "🎬 Gemini视频提示生成器",
    "GeminiPromptGenerator": "🖼️ Gemini图片提示生成器",
    "GeminiPromptOptimizer": "✨ Gemini提示词优化器",
    "GeminiChatPro": "💬 Gemini聊天器"
}