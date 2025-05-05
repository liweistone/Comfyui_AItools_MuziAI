import os
import json
import logging
from pathlib import Path
import google.generativeai as genai
import torch
import numpy as np
from PIL import Image
import random
import time
from comfy.sd import CLIP
from comfy.cli_args import LatentPreviewMethod
import folder_paths
import comfy.utils
import comfy.sample
import comfy.samplers
import comfy.sd
import comfy.model_management
from comfy_extras.nodes_clip_sdxl import CLIPTextEncodeSDXL
from nodes import CLIPTextEncode
from comfy.model_patcher import ModelPatcher

# 配置日志 - 修正路径拼接
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=os.path.join(log_dir, 'gemini_video_creator.log')
)

class ConfigManager:
    @classmethod
    def get_templates(cls):
        """更健壮的模板加载方式"""
        try:
            template_dir = os.path.join(os.path.dirname(__file__), 'templates')
            templates = {}
            for file in Path(template_dir).glob('*.json'):
                with open(file, 'r', encoding='utf-8') as f:
                    templates[file.stem] = json.load(f)
            return templates
        except Exception as e:
            logging.error(f"加载模板失败: {str(e)}")
            return {
                "drama": {"instruction": "默认剧本模板"},
                "comedy": {"instruction": "默认喜剧模板"},
                "documentary": {"instruction": "默认纪录片模板"},
                "poetry": {"instruction": "默认古诗模板"},
                "classical_poetry": {"instruction": "高度还原古诗词模板"},
            }

TEMPLATES = ConfigManager.get_templates()

#==============Gemini短视频创作器=============
class GeminiVideoCreator:
    """
    🎬 Gemini短视频创作器 🎬
    功能：
    - 根据创意自动生成短视频剧本
    - 自动生成分镜描述和AI绘画提示词
    - 支持多种视频类型和风格
    微信：stone_liwei
    版本：2.1（新增使用说明和参数说明）
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "idea": ("STRING", {
                    "default": "", 
                    "multiline": True,
                    "tooltip": "输入你的视频创意或主题描述"
                }),
                "genre": (list(TEMPLATES.keys()), {
                    "default": "drama",
                    "tooltip": "选择视频类型/风格模板"
                }),
                "duration": (["30s", "1m", "3m", "5m"], {
                    "default": "1m",
                    "tooltip": "选择视频时长"
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
                    "min": 500, 
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
                "api_key": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "tooltip": "输入Gemini API密钥(可从Google AI Studio获取)"
                }),
            },
            "optional": {
                "custom_instruction": ("STRING", {
                    "default": "请根据以下创意生成一个短视频剧本，要求：1.包含完整故事结构 2.有明确的开头、发展和结尾 3.适合短视频平台传播 4.包含3-5个主要场景 5.不要使用Markdown格式",
                    "multiline": True,
                    "tooltip": "自定义生成指令，覆盖默认模板"
                }),
                "proxy_url": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "tooltip": "代理服务器地址(如需)"
                }),
                "style_reference": ("IMAGE", {
                    "tooltip": "上传参考图像以保持风格一致"
                }),
                "help_text": ("STRING", {
                    "default": """📜 Gemini短视频创作器使用说明

功能说明：根据创意自动生成短视频剧本、分镜描述和AI绘画提示词

🔧 参数说明：
idea: 输入你的视频创意或主题描述
genre: 选择视频类型/风格模板
duration: 选择视频时长
model_name: 选择Gemini模型版本
max_tokens: 控制生成内容的长度
temperature: 控制生成内容的随机性(0=确定性高,1=创造性高)
api_key: Gemini API密钥(可从Google AI Studio获取)

🎯 可选参数:
custom_instruction: 自定义生成指令
proxy_url: 代理服务器地址(如需)
style_reference: 上传参考图像以保持风格一致

💡 使用场景：
- 短视频内容创作
- 短剧剧本生成
- 分镜脚本自动生成
- AI绘画提示词批量生成

📌 输出说明：
1. 剧本: 完整的故事剧本
2. 分镜内容: 详细的镜头描述
3. 分镜提示词: 可用于AI绘画的提示词

更多AI工具请关注公众号: 懂AI的木子
                    """,
                    "multiline": True
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("剧本", "分镜内容", "分镜提示词")
    FUNCTION = "generate_video_plan"
    CATEGORY = "🎨公众号懂AI的木子做号工具/gemini视频创作短视频短剧版"

    def __init__(self):
        self.initialized = False
        self.session = None

    def initialize_model(self, api_key, proxy_url=None):
        """更健壮的模型初始化"""
        if not self.initialized:
            try:
                config = {
                    "api_key": api_key or os.getenv("GOOGLE_API_KEY"),
                    "transport": "rest"
                }
                if proxy_url:
                    config["client_options"] = {
                        "api_endpoint": proxy_url
                    }
                
                genai.configure(**config)
                self.initialized = True
                logging.info("Gemini模型初始化成功")
            except Exception as e:
                logging.error(f"模型初始化失败: {str(e)}")
                raise

    def generate_video_plan(self, idea, genre, duration, model_name, max_tokens, temperature, api_key, custom_instruction="", proxy_url="", style_reference=None, help_text="", unique_id=None, **kwargs):
        try:
            # 初始化模型
            self.initialize_model(api_key, proxy_url if proxy_url else None)
            
            # 准备提示词
            template = TEMPLATES.get(genre, TEMPLATES["drama"])
            prompt = f"""请严格按照以下格式返回内容（不要遗漏任何部分）：
            
            === 剧本 ===
            {custom_instruction if custom_instruction else template['instruction']}
            创意: {idea}
            时长: {duration}
            
            === 分镜 ===
            按顺序描述每个镜头，包含:
            1. 镜头编号
            2. 镜头类型（全景/中景/特写）
            3. 画面内容
            4. 时长估算
            
            === 提示词 ===
            为每个镜头生成AI绘画提示词，要求:
            1. 风格一致
            2. 包含镜头类型描述
            3. 使用中文
            """
            
            # 处理图像输入
            image_part = None
            if style_reference is not None:
                image_part = self.process_image(style_reference)
            
            # 调用API
            model = genai.GenerativeModel(model_name)
            if image_part:
                response = model.generate_content([prompt, image_part],
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=max_tokens,
                        temperature=temperature
                    )
                )
            else:
                response = model.generate_content(prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=max_tokens,
                        temperature=temperature
                    )
                )
            
            if not response.text:
                raise ValueError("API返回空响应")
                
            # 更健壮的响应解析
            return self.parse_response(response.text)
            
        except Exception as e:
            error_msg = f"生成失败: {str(e)}"
            logging.error(error_msg, exc_info=True)
            return (error_msg, error_msg, error_msg)

    def process_image(self, image_tensor):
        """更安全的图像处理"""
        try:
            image_np = 255. * image_tensor[0].cpu().numpy()
            pil_image = Image.fromarray(np.clip(image_np, 0, 255).astype(np.uint8))
            
            # 限制最大尺寸
            max_size = 1024
            if max(pil_image.size) > max_size:
                ratio = max_size / max(pil_image.size)
                new_size = (int(pil_image.width * ratio), int(pil_image.height * ratio))
                pil_image = pil_image.resize(new_size, Image.LANCZOS)
            
            return pil_image
        except Exception as e:
            logging.error(f"图像处理失败: {str(e)}")
            return None

    def parse_response(self, text):
        """改进的响应解析器"""
        try:
            result = {
                "script": "",
                "storyboard": "",
                "prompts": ""
            }
            
            # 更精确的段落分割
            sections = [s.strip() for s in text.split("===") if s.strip()]
            for i in range(0, len(sections)-1, 2):
                section_type = sections[i].lower()
                content = sections[i+1]
                
                if "剧本" in section_type:
                    result["script"] = content
                elif "分镜" in section_type or "storyboard" in section_type:
                    result["storyboard"] = content
                elif "提示" in section_type or "prompt" in section_type:
                    result["prompts"] = content
            
            # 回退机制：如果未按格式返回，尝试智能分割
            if not all(result.values()):
                parts = text.split("\n\n")
                if len(parts) >= 3:
                    result = {
                        "script": parts[0],
                        "storyboard": parts[1],
                        "prompts": "\n".join(parts[2:])
                    }
            
            return (result["script"], result["storyboard"], result["prompts"])
        except Exception as e:
            logging.error(f"解析响应失败: {str(e)}")
            return (text, text, text)

#===============分镜转提示词优化器==================
class GeminiStoryboardToPrompts:
    """
    📹 分镜转提示词优化器 📹
    功能：
    - 将分镜描述转换为高质量的AI绘画提示词
    - 支持多种艺术风格
    - 保持多镜头间风格一致性
    微信：stone_liwei
    版本：2.1（新增使用说明和参数说明）
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "storyboard": ("STRING", {
                    "default": "", 
                    "multiline": True,
                    "tooltip": "输入分镜描述文本"
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
                "style": (["realistic", "anime", "cinematic", "watercolor", "3d_render"], {
                    "default": "cinematic",
                    "tooltip": "选择艺术风格"
                }),
                "max_tokens": ("INT", {
                    "default": 1500, 
                    "min": 500, 
                    "max": 4000,
                    "tooltip": "控制生成内容的长度"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.6, 
                    "min": 0, 
                    "max": 1, 
                    "step": 0.05,
                    "tooltip": "控制生成内容的随机性(0=确定性高,1=创造性高)"
                }),
                "api_key": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "tooltip": "输入Gemini API密钥(可从Google AI Studio获取)"
                }),
                "timeout": ("INT", {
                    "default": 45, 
                    "min": 10, 
                    "max": 180,
                    "tooltip": "API请求超时时间(秒)"
                }),
            },
            "optional": {
                "custom_instruction": ("STRING", {
                    "default": "请根据以下分镜描述生成AI绘画提示词，要求：1.保持风格一致 2.包含场景细节 3.使用英文 4.每个提示词不少于100个字符",
                    "multiline": True,
                    "tooltip": "自定义生成指令"
                }),
                "proxy_url": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "tooltip": "代理服务器地址(如需)"
                }),
                "help_text": ("STRING", {
                    "default": """📜 分镜转提示词优化器使用说明

功能说明：将分镜描述转换为高质量的AI绘画提示词

🔧 参数说明：
storyboard: 输入分镜描述文本
model_name: 选择Gemini模型版本
style: 选择艺术风格
max_tokens: 控制生成内容的长度
temperature: 控制生成内容的随机性
api_key: Gemini API密钥
timeout: API请求超时时间(秒)

🎯 可选参数:
custom_instruction: 自定义生成指令
proxy_url: 代理服务器地址(如需)

💡 使用场景：
- 将分镜脚本转换为AI绘画提示词
- 批量生成风格一致的提示词
- 为动画制作准备素材

📌 输出说明：
1. prompts: 可用于AI绘画的详细提示词(英文)

更多AI工具请关注公众号: 懂AI的木子
                    """,
                    "multiline": True
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompts",)
    FUNCTION = "generate_prompts"
    CATEGORY = "🎨公众号懂AI的木子做号工具/gemini视频创作短视频短剧版"

    def __init__(self):
        self.initialized = False
        self.max_retries = 3  # 新增重试机制

    def initialize_model(self, api_key, proxy_url=None):
        """改进的模型初始化"""
        if not self.initialized:
            try:
                config = {
                    "api_key": api_key or os.getenv("GOOGLE_API_KEY"),
                    "transport": "rest"
                }
                if proxy_url:
                    config["client_options"] = {
                        "api_endpoint": proxy_url
                    }
                
                genai.configure(**config)
                self.initialized = True
                logging.info("Gemini模型初始化成功")
            except Exception as e:
                logging.error(f"模型初始化失败: {str(e)}")
                self.initialized = False
                raise

    def generate_prompts(self, storyboard, model_name, style, max_tokens, temperature, api_key, timeout, custom_instruction="", proxy_url="", help_text="", unique_id=None, **kwargs):
        try:
            # 初始化模型（带重试机制）
            for attempt in range(self.max_retries):
                try:
                    self.initialize_model(api_key, proxy_url if proxy_url else None)
                    break
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise
                    logging.warning(f"初始化失败，正在重试 ({attempt+1}/{self.max_retries})")
                    time.sleep(2 ** attempt)
            
            # 构建提示词模板
            prompt_template = f"""
            {custom_instruction if custom_instruction else f"请生成{style}风格的AI绘画提示词"}
            
            分镜内容:
            {storyboard}
            
            生成要求:
            1. 每个分镜对应一个提示词
            2. 保持视觉风格一致性
            3. 包含镜头角度描述
            4. 使用英文
            5. 格式示例: [Scene 1]: detailed prompt here
            """
            
            logging.info(f"开始生成提示词，超时设置: {timeout}s")
            
            # 调用API（带超时控制）
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                prompt_template,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature
                ),
                request_options={"timeout": timeout}
            )
            
            # 验证响应
            if not response.text:
                raise ValueError("API返回空响应")
            if len(response.text) < 50:
                raise ValueError("响应内容过短")
                
            logging.info("提示词生成成功")
            return (response.text.strip(),)
            
        except Exception as e:
            error_msg = f"提示词生成失败: {str(e)}"
            logging.error(error_msg, exc_info=True)
            return (f"ERROR: {error_msg}",)

#===========提示词批次选择器 (木子AI)=======================
class PromptSequencer:
    """
    ✨ 木子AI提示词序列器 ✨
    功能：
    - 从多行文本中选择指定行（1-based）
    - 支持顺序/随机两种读取模式
    微信：stone_liwei
    版本：2.1（新增使用说明和参数说明）
    """
    def __init__(self):
        self.prompts = []
        self.current_index = 0  # 用于顺序模式
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt_list": ("STRING", {
                    "multiline": True, 
                    "default": "第一行提示词\n第二行提示词\n第三行提示词",
                    "dynamicPrompts": False,
                    "tooltip": "输入多行提示词，每行一个"
                }),
                "mode": (["sequential", "random", "manual"], {
                    "default": "sequential",
                    "tooltip": "sequential:顺序读取\nrandom:随机选择\nmanual:手动指定行号"
                }),
                "seed": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 0xffffffffffffffff,
                    "tooltip": "随机模式下使用的种子值\n0表示每次随机"
                }),
            },
            "optional": {
                "line_number": ("INT", {
                    "default": 1, 
                    "min": 1, 
                    "max": 9999,
                    "step": 1,
                    "display": "number",
                    "tooltip": "手动模式下指定要选择的行号"
                }),
                "reset_counter": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "重置顺序模式的计数器"
                }),
                "help_text": ("STRING", {
                    "default": """📜 提示词批次选择器使用说明

功能说明：从多行提示词中选择指定行/让批量生图变得更简单，支持顺序/随机/手动三种模式

🔧 参数说明：
prompt_list: 输入多行提示词，每行一个
mode:
sequential - 顺序读取(自动循环)
random - 随机选择
manual - 手动指定行号
seed: 随机模式种子值(0=真随机)
line_number: 手动模式下的行号
reset_counter: 重置顺序模式的计数器

💡 使用场景：
批量生成多张图片时自动切换提示词
A/B测试不同提示词效果
制作动画时按顺序切换场景
更多AI工具请关注公众号: 懂AI的木子
                    """,
                    "multiline": True
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }
    
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("prompt", "行号")
    FUNCTION = "get_selected_prompt"
    CATEGORY = "🎨公众号懂AI的木子做号工具/gemini视频创作短视频短剧版"
    
    def get_selected_prompt(self, prompt_list, mode, seed=0, line_number=1, reset_counter=False, help_text="", unique_id=None):
        try:
            # 处理文本输入
            self.prompts = [line.strip() for line in prompt_list.split('\n') if line.strip()]
            
            if not self.prompts:
                return ("[错误] 没有有效的提示词内容", 0)
            
            # 重置计数器
            if reset_counter:
                self.current_index = 0
            
            selected_index = 0
            line_num = 0
            
            if mode == "manual":
                # 手动指定模式
                selected_index = max(0, min(line_number - 1, len(self.prompts) - 1))
                line_num = line_number
            elif mode == "random":
                # 随机模式
                random.seed(seed if seed != 0 else None)
                selected_index = random.randint(0, len(self.prompts) - 1)
                line_num = selected_index + 1
            else:
                # 顺序模式
                if self.current_index >= len(self.prompts):
                    self.current_index = 0  # 循环读取
                
                selected_index = self.current_index
                line_num = self.current_index + 1
                self.current_index += 1
            
            return (self.prompts[selected_index], line_num)
            
        except Exception as e:
            return (f"[错误] 发生异常: {str(e)}", 0)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "GeminiVideoCreator": GeminiVideoCreator,
    "GeminiStoryboardToPrompts": GeminiStoryboardToPrompts,
    "PromptSequencer": PromptSequencer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiVideoCreator": "🎬短剧短视频创作器",
    "GeminiStoryboardToPrompts": "📹分镜转提示词优化器",
    "PromptSequencer": "📜提示词批次选择器 (懂AI的木子)"
}