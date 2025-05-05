import os
import random
import glob
import cv2
import torch
import numpy as np
from PIL import Image

class RandomVideoLoadertwo:
    """
    从相对路径vdata目录随机加载视频并提取首帧
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "gpu_acceleration": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("video_path", "first_frame")
    FUNCTION = "load_random_video"
    CATEGORY = "🎨公众号懂AI的木子做号工具/懒人做号/视频相关"
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

    def load_random_video():
        # 动态更新帮助说明
        self.__class__.DESCRIPTION = self.__class__.DESCRIPTION.format(
            data_path=self.get_data_path()
        )


    def load_random_video(self, seed, gpu_acceleration=True):
        # 设置固定目录名
        directory = "vdata"
        
        # 设置随机种子
        random.seed(seed)
        
        # 获取当前插件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 构建视频目录路径(相对于插件目录)
        video_dir = os.path.join(current_dir, directory)
        
        # 扫描目录下的视频文件
        video_extensions = ['*.del', '*.mp4', '*.mov', '*.avi', '*.mkv', '*.webm']
        video_files = []
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(video_dir, ext)))
        
        if not video_files:
            raise ValueError(f"No video files found in directory: {video_dir}")
        
        # 随机选择一个视频
        selected_video = random.choice(video_files)
        
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
                raise ValueError(f"Failed to read first frame from video: {selected_video}")
            
            # 转换图像格式
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            image_tensor = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
            
            return (selected_video, image_tensor)
        except Exception as e:
            raise ValueError(f"Error loading video: {str(e)}")

NODE_CLASS_MAPPINGS = {
    "RandomVideoLoadertwo": RandomVideoLoadertwo
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RandomVideoLoadertwo": "📺懒人批量制作视频（无人值守\批量|定制类节点）"
}