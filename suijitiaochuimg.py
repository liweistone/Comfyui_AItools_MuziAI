import os
import random
from PIL import Image
import numpy as np
import torch
import folder_paths
import comfy.utils

class ImageLoaderFromFolder:
    """
    ✨ 木子AI图片加载器 ✨
    功能：
    - 从指定文件夹加载图片
    - 支持顺序/随机两种读取模式
    - 可选择子目录加载图片
    版本：3.1（移除image_path输出）
    微信：stone_liwei
    """
    def __init__(self):
        # 定义图片目录（相对于插件目录）
        self.base_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "picdata")
        # 支持的图片扩展名
        self.allowed_extensions = ['.del', '.jpg', '.jpeg', '.png', '.webp']
        self.current_index = 0  # 用于顺序模式
        self.image_cache = []   # 图片缓存
        self.available_subfolders = []  # 可用的子目录列表

    @classmethod
    def INPUT_TYPES(cls):
        # 在类加载时扫描可用子目录
        base_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "picdata")
        available_subfolders = []
        if os.path.exists(base_folder):
            available_subfolders = [d for d in os.listdir(base_folder) 
                                  if os.path.isdir(os.path.join(base_folder, d))]
            available_subfolders.sort()
        
        # 如果没有子目录，则添加一个默认选项
        if not available_subfolders:
            available_subfolders = ["."]  # 表示根目录
        
        return {
            "required": {
                "subfolder": (available_subfolders, {
                    "default": available_subfolders[0],
                    "tooltip": "选择要加载的子目录"
                }),
                "mode": (["sequential", "random"], {
                    "default": "sequential",
                    "tooltip": "选择读取模式：顺序或随机"
                }),
                "seed": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 0xffffffffffffffff,
                    "tooltip": "随机模式下的种子值（0表示完全随机）"
                }),
            },
            "optional": {
                "reset_counter": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "重置顺序模式的计数器"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("image", "当前序号")
    FUNCTION = "load_image"
    CATEGORY = "🎨公众号懂AI的木子做号工具/懒人做号/图片相关"
    OUTPUT_NODE = True

    def scan_images(self, subfolder):
        """扫描指定子目录中的图片文件"""
        image_folder = os.path.join(self.base_folder, subfolder)
        
        if not os.path.exists(image_folder):
            raise ValueError(f"图片目录不存在: {image_folder}")
        
        image_files = []
        for f in sorted(os.listdir(image_folder)):  # 排序保证顺序一致性
            ext = os.path.splitext(f)[1].lower()
            if ext in self.allowed_extensions:
                image_files.append(os.path.join(image_folder, f))
        
        if not image_files:
            raise ValueError(f"目录中没有图片文件: {image_folder}")
        
        return image_files

    def load_image(self, subfolder, mode, seed=0, reset_counter=False):
        # 扫描图片文件（如果首次运行或缓存为空）
        if not self.image_cache or reset_counter:
            self.image_cache = self.scan_images(subfolder)
        
        if not self.image_cache:
            raise ValueError("没有可用的图片文件")
        
        # 重置计数器
        if reset_counter:
            self.current_index = 0
        
        # 选择图片
        if mode == "random":
            random.seed(seed if seed != 0 else None)
            selected_index = random.randint(0, len(self.image_cache) - 1)
        else:
            # 顺序模式
            if self.current_index >= len(self.image_cache):
                self.current_index = 0  # 循环读取
            selected_index = self.current_index
            self.current_index += 1
        
        selected_image = self.image_cache[selected_index]
        
        # 加载图片并转换为ComfyUI兼容格式
        try:
            image = Image.open(selected_image)
            image = image.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image)[None, ]
            
            return (image_tensor, selected_index + 1)  # 只返回图像张量和序号
        except Exception as e:
            raise ValueError(f"图片加载错误: {str(e)}")

NODE_CLASS_MAPPINGS = {
    "ImageLoaderFromFolder": ImageLoaderFromFolder
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageLoaderFromFolder": "懒人做号IMG加载生成器（子目录/顺序/随机/定制）"
}
