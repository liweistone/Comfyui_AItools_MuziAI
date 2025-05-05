import os
import cv2
import torch
import numpy as np
import subprocess
from tqdm import tqdm
import folder_paths

class FramesToVideoNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_path": ("STRING", {"default": "[time]/output.mp4", "multiline": False}),
                "frame_rate": ("INT", {"default": 30, "min": 1, "max": 120}),
                "video_format": (["mp4", "avi", "mov"], {"default": "mp4"}),
                "filename_pattern": ("STRING", {"default": "frame_%06d.png"}),
            },
            "optional": {
                "frames_dir": ("STRING", {"default": "input/frames", "forceInput": True}),
                "audio_path": ("STRING", {"default": "", "forceInput": True}),
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "create_video"
    CATEGORY = "🎨公众号懂AI的木子做号工具/懒人做号/视频相关"
    OUTPUT_NODE = True

    def create_video(self, output_path, frame_rate, video_format, filename_pattern, frames_dir="", audio_path="", images=None):
        # 处理动态路径
        output_path = self._process_path(output_path)
        
        # 优先处理张量输入
        if images is not None and images.nelement() > 0:
            frames_dir = self._save_tensor_images(images)
            filename_pattern = "temp_frame_%06d.png"

        # 验证输入目录
        if not os.path.exists(frames_dir):
            raise ValueError(f"无效的图片目录: {frames_dir}")

        # 获取图片列表
        img_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not img_files:
            raise ValueError("未找到有效图片文件")

        # 创建视频编码器
        sample_img = cv2.imread(os.path.join(frames_dir, img_files[0]))
        height, width, _ = sample_img.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v' if video_format == 'mp4' else 'XVID')
        video_writer = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))

        # 生成视频
        for filename in tqdm(sorted(img_files), desc="生成视频帧"):
            frame = cv2.imread(os.path.join(frames_dir, filename))
            video_writer.write(frame)
        video_writer.release()

        # 添加音频
        if os.path.isfile(audio_path):
            output_path = self._add_audio(output_path, audio_path)

        return (output_path,)

    def _process_path(self, path):
        """处理包含时间戳的路径"""
        if "[time]" in path:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            path = path.replace("[time]", timestamp)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def _save_tensor_images(self, images):
        temp_dir = os.path.join(folder_paths.get_temp_directory(), "frame2video_temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        for i, img_tensor in enumerate(images):
            img_np = 255.0 * img_tensor.cpu().numpy().squeeze().clip(0, 1)
            cv2.imwrite(
                os.path.join(temp_dir, f"temp_frame_{i:06d}.png"),
                cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2BGR)
            )
        return temp_dir

    def _add_audio(self, video_path, audio_path):
        output_path = video_path.replace(".mp4", "_with_audio.mp4")
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-i', audio_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-shortest', output_path
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            os.replace(output_path, video_path)
            return video_path
        except Exception as e:
            print(f"音频合并失败，保持无声视频: {str(e)}")
            return video_path

NODE_CLASS_MAPPINGS = {"FramesToVideoNode": FramesToVideoNode}
NODE_DISPLAY_NAME_MAPPINGS = {"FramesToVideoNode": "📺图片序列转视频"}