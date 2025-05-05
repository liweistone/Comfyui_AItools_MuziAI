import os
import cv2
import torch
import comfy.utils
import folder_paths
from pathlib import Path

class VideoFirstFrameNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "multiline": False}),
                "output_dir": ("STRING", {"default": "output/first_frames", "multiline": False}),
                "filename_prefix": ("STRING", {"default": "first_frame"}),
                "format": (["jpg", "png", "webp"], {"default": "jpg"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "saved_path")
    FUNCTION = "extract_first_frame"
    CATEGORY = "🎨公众号懂AI的木子做号工具/懒人做号/视频相关"
    OUTPUT_NODE = True


    def extract_first_frame(self, video_path, output_dir, filename_prefix, format):
        # 校验输入文件
        if not os.path.isfile(video_path):
            raise ValueError(f"视频文件不存在: {video_path}")
            
        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 读取视频第一帧
        cap = cv2.VideoCapture(video_path)
        success, frame = cap.read()
        cap.release()
        
        if not success:
            raise ValueError("无法读取视频首帧")
            
        # 转换颜色空间 BGR->RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 转换为Tensor
        tensor_frame = torch.from_numpy(rgb_frame).float() / 255.0
        tensor_frame = tensor_frame.unsqueeze(0)
        
        # 保存图片
        save_path = os.path.join(output_dir, f"{filename_prefix}.{format}")
        save_image = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)  # 转换回BGR保存
        
        if format == "jpg":
            cv2.imwrite(save_path, save_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        elif format == "webp":
            cv2.imwrite(save_path, save_image, [int(cv2.IMWRITE_WEBP_QUALITY), 95])
        else:  # PNG
            cv2.imwrite(save_path, save_image)
            
        return (tensor_frame, save_path)

NODE_CLASS_MAPPINGS = {
    "VideoFirstFrameNode": VideoFirstFrameNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoFirstFrameNode": "📺输出视频第一帧"
}
