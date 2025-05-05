import os
import random
from pathlib import Path
import comfy.sd

class RandomPromptLoader:
    """✍木子AI做号工具 - 提示词加载器
    功能：
    - 从指定目录加载文本文件
    - 支持顺序/随机两种读取模式
    - 新增使用教程输出接口
    版本：3.0（增加顺序读取模式）
    """
    
    def __init__(self):
        self.update_file_list()
        self.tutorial_text = self._generate_tutorial()  # 初始化教程内容
        self.current_line = 0  # 新增：记录当前读取行位置
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "风格选择": (cls.get_txt_files(), {
                    "default": "治愈系风景",
                    "tooltip": "选择包含提示词的文本文件"
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
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")  # 新增第三个输出接口
    RETURN_NAMES = ("Prompt words", "使用教程", "当前行号")   # 输出接口命名
    FUNCTION = "load_selected_prompt"
    CATEGORY = "🎨公众号懂AI的木子做号工具/懒人做号/提示词"
    OUTPUT_NODE = True
    DESCRIPTION = """
    📌 使用说明：
    1. 设置seed和读取模式：
       - seed=0 + sequential=顺序读取
       - seed≠0 + random=随机读取
    2. 输出包含三个接口：
       - 随机选择的提示词
       - 使用教程文本
       - 当前行号（顺序模式有效）
    🚩 数据文件路径：{data_path}
    """

    def _generate_tutorial(self):
        """生成使用教程内容"""
        return f"""
        🌟 使用教程 🌟
        --------------------------        
        📞 技术支持：微信 stone_liwei
        公众号：懂ai的木子
        """

    @classmethod
    def get_txt_files(cls):
        current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
        data_dir = current_dir / "data"
        return [f.name for f in data_dir.glob("*") if f.is_file()]

    @classmethod
    def get_data_path(cls):
        current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
        return str(current_dir / "data").replace("\\", "/")

    def update_file_list(self):
        self.txt_files = self.get_txt_files()

    def load_selected_prompt(self, 风格选择, seed, mode, reset_counter=False, **kwargs):
        # 动态更新帮助说明
        self.__class__.DESCRIPTION = self.__class__.DESCRIPTION.format(
            data_path=self.get_data_path()
        )
        
        target_file = Path(__file__).parent / "data" / 风格选择
        
        if not target_file.exists():
            raise ValueError(f"文件 {风格选择} 不存在于数据目录")
        
        # 读取文件内容
        with open(target_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        if not lines:
            raise ValueError(f"{风格选择} 中没有有效提示词")

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
    "RandomPromptLoader": RandomPromptLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RandomPromptLoader": "📗提示词大师批量无脑生图（需定制内容）",
}