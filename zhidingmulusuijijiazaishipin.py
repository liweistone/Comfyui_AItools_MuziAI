import os
import random
import glob
import cv2
import torch
import numpy as np
from PIL import Image
import folder_paths

class VideoLoader:
    """
    ✨ 木子AI视频加载器 ✨
    功能：
    - 从指定目录加载视频并提取首帧
    - 支持顺序/随机两种读取模式
    版本：2.0（新增顺序模式）
    微信：stone_liwei
    """
    def __init__(self):
        self.current_index = 0  # 用于顺序模式
        self.video_cache = []  # 缓存视频列表
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {
                    "default": "请输入完整路径",
                    "tooltip": "相对于ComfyUI输入目录的路径"
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
                "gpu_acceleration": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "是否启用GPU加速"
                }),
                "reset_counter": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "重置顺序模式的计数器"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "IMAGE", "INT")
    RETURN_NAMES = ("video_path", "first_frame", "当前序号")
    FUNCTION = "load_video"
    CATEGORY = "🎨公众号懂AI的木子做号工具/懒人做号/视频相关"
    OUTPUT_NODE = True

    def scan_video_files(self, directory):
        """扫描目录中的视频文件"""
        base_dir = folder_paths.get_input_directory()
        video_dir = os.path.join(base_dir, directory)
        
        video_extensions = ['*.mp4', '*.mov', '*.avi', '*.mkv', '*.webm']
        video_files = []
        for ext in video_extensions:
            video_files.extend(sorted(glob.glob(os.path.join(video_dir, ext))))
        
        if not video_files:
            raise ValueError(f"目录中没有视频文件: {video_dir}")
        
        return video_files

    def load_video(self, directory, mode, seed=0, gpu_acceleration=True, reset_counter=False):
        # 重置计数器
        if reset_counter:
            self.current_index = 0
        
        # 扫描视频文件（如果目录改变或首次运行）
        if not hasattr(self, 'last_directory') or self.last_directory != directory or not self.video_cache:
            self.video_cache = self.scan_video_files(directory)
            self.last_directory = directory
            self.current_index = 0
        
        if not self.video_cache:
            raise ValueError("没有可用的视频文件")
        
        # 选择视频
        if mode == "random":
            random.seed(seed if seed != 0 else None)
            selected_index = random.randint(0, len(self.video_cache) - 1)
        else:
            # 顺序模式
            if self.current_index >= len(self.video_cache):
                self.current_index = 0  # 循环读取
            selected_index = self.current_index
            self.current_index += 1
        
        selected_video = self.video_cache[selected_index]
        
        # 读取视频
        try:
            if gpu_acceleration and torch.cuda.is_available():
                cap = cv2.VideoCapture(selected_video, cv2.CAP_ANY)
                cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
            else:
                cap = cv2.VideoCapture(selected_video)
            
            # 读取第一帧
            success, frame = cap.read()
            cap.release()
            
            if not success:
                raise ValueError(f"无法读取视频首帧: {selected_video}")
            
            # 转换图像格式
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            image_tensor = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
            
            return (selected_video, image_tensor, selected_index + 1)  # 返回1-based序号
        except Exception as e:
            raise ValueError(f"视频加载错误: {str(e)}")

NODE_CLASS_MAPPINGS = {
    "VideoLoader": VideoLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoLoader": "📺批次视频加载器（从路径）"
}