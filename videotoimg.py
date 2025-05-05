import os
import cv2
import comfy
import torch
import numpy as np  # 确保已安装
import folder_paths
from tqdm import tqdm

class VideoToFramesNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "multiline": False}),
                "output_dir": ("STRING", {"default": "output/video_frames", "multiline": False}),
                "filename_prefix": ("STRING", {"default": "frame"}),
                "format": (["jpg", "png", "webp"], {"default": "jpg"}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("output_info", "images")
    FUNCTION = "convert_video"
    CATEGORY = "🎨公众号懂AI的木子做号工具/懒人做号/视频相关"
    OUTPUT_NODE = True

    def convert_video(self, video_path, output_dir, filename_prefix, format):
        # 校验输入文件
        if not os.path.isfile(video_path):
            raise ValueError(f"视频文件不存在: {video_path}")
            
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if not cap.isOpened():
            raise ValueError("无法打开视频文件")
        
        # 处理帧数据
        success, image = cap.read()
        count = 0
        images = []
        
        # 创建进度条
        pbar = tqdm(total=total_frames, desc="Processing video frames")
        
        while success:
            # 构造输出路径
            frame_number = str(count).zfill(6)
            output_path = os.path.join(
                output_dir, 
                f"{filename_prefix}_{frame_number}.{format}"
            )
            
            # 保存图像
            if format == "jpg":
                cv2.imwrite(output_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            elif format == "webp":
                cv2.imwrite(output_path, image, [int(cv2.IMWRITE_WEBP_QUALITY), 95])
            else:
                cv2.imwrite(output_path, image)
            
            # 转换为ComfyUI图像格式 (RGB 0-1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_norm = image_rgb.astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_norm)
            images.append(image_tensor)
            
            count += 1
            success, image = cap.read()
            pbar.update(1)
        
        cap.release()
        pbar.close()
        
        # 堆叠所有帧为张量 [N H W C]
        if len(images) > 0:
            images_tensor = torch.stack(images, dim=0)
        else:
            images_tensor = torch.zeros((0, 1, 1, 3))  # 空张量
            
        return (f"成功转换 {count} 帧到 {output_dir}", images_tensor)

NODE_CLASS_MAPPINGS = {
    "VideoToFramesNode": VideoToFramesNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoToFramesNode": "📺视频转图片序列"
}