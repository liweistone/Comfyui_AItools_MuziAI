import os
import random
from pathlib import Path
import comfy.sd

class PromptMasterPATH:
    """✍木子AI做号工具 - 提示词大师
    功能：
    - 通过上传txt文件加载提示词
    - 支持顺序/随机两种读取模式
    - 提供使用教程输出
    版本：3.1（增加顺序读取模式）
    """
    
    def __init__(self):
        self.tutorial_text = self._generate_tutorial()
        self.current_line = 0  # 新增：记录当前读取行位置
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_file": ("STRING", {
                    "default": "",
                    "tooltip": "上传或输入txt文件路径"
                }),
                "seed": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 0xffffffffffffffff,
                    "tooltip": "输入seed（0表示顺序读取，非0表示随机）"
                }),
                "mode": (["sequential", "random"], {
                    "default": "sequential",
                    "tooltip": "选择读取模式：顺序或随机"
                }),
            },
            "optional": {
                "reset_counter": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "重置行计数器（用于顺序模式）"
                }),
                "help_text": ("STRING", {
                    "default": """# 从指定位置加载txt文件，按顺序或随机读取每行内容""",
                    "multiline": True
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("Prompt words", "使用教程", "当前行号")
    FUNCTION = "load_selected_prompt"
    CATEGORY = "🎨公众号懂AI的木子做号工具/懒人做号/提示词"
    OUTPUT_NODE = True

    DESCRIPTION = """
    📌 使用说明：
    1. 上传或输入txt文件路径
    2. 设置seed和读取模式：
       - seed=0 + sequential=顺序读取
       - seed≠0 + random=随机读取
    3. 输出包含：
       - 提示词内容
       - 使用教程
       - 当前行号（顺序模式有效）
    """

    def _generate_tutorial(self):
        """生成使用教程内容"""
        return """
        🌟 使用教程 🌟
        --------------------------
        1. 上传提示词文本文件
        2. 模式选择：
           - 顺序模式：按文件行号依次读取
           - 随机模式：随机选择一行
        3. 重置计数器：
           - 勾选reset_counter可重新从第一行开始
        
        📞 技术支持：微信 stone_liwei
        """

    def load_selected_prompt(self, prompt_file, seed, mode, reset_counter=False, **kwargs):
        # 检查文件是否存在
        if not prompt_file or not os.path.exists(prompt_file):
            raise ValueError("请上传有效的txt文件")
        
        if not prompt_file.lower().endswith('.txt'):
            raise ValueError("只支持txt格式文件")
        
        # 读取文件内容
        with open(prompt_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        if not lines:
            raise ValueError("文件中没有有效提示词")

        # 重置行计数器
        if reset_counter:
            self.current_line = 0

        # 选择模式
        if mode == "random":
            random.seed(seed if seed != 0 else None)
            selected_line = random.choice(lines)
            return (selected_line, self.tutorial_text, 0)
        else:
            # 顺序模式
            if self.current_line >= len(lines):
                self.current_line = 0  # 循环读取
            
            selected_line = lines[self.current_line]
            current_pos = self.current_line + 1  # 返回人类可读的行号（从1开始）
            self.current_line += 1
            
            return (selected_line, self.tutorial_text, current_pos)

NODE_CLASS_MAPPINGS = {
    "PromptMasterPATH": PromptMasterPATH,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptMasterPATH": "📗提示词批次（路径版/顺序/随机）",
}