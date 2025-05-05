import os
import cv2
import torch
import numpy as np
import subprocess
from datetime import datetime

class VideoProcessorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "input/video.mp4", "multiline": False}),
                "frame_output_dir": ("STRING", {"default": "[time]/frames"}),
                "audio_output_dir": ("STRING", {"default": "[time]/audio"}),
                "extract_interval": ("INT", {"default": 1, "min": 1, "max": 60}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE", "STRING", "IMAGE")
    RETURN_NAMES = ("video_path", "frame_seq", "audio_path", "first_frame")
    FUNCTION = "process_video"
    CATEGORY = "🎨公众号懂AI的木子做号工具/懒人做号/视频相关"
    OUTPUT_NODE = True

    def process_video(self, video_path, frame_output_dir, audio_output_dir, extract_interval):
        # 动态路径处理
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        frame_dir = frame_output_dir.replace("[time]", timestamp)
        audio_dir = audio_output_dir.replace("[time]", timestamp)
        
        # 提取首帧
        first_frame = self._extract_first_frame(video_path)
        
        # 提取帧序列
        frame_seq = self._extract_frames(video_path, frame_dir, extract_interval)
        
        # 提取音频
        audio_path = self._extract_audio(video_path, audio_dir)
        
        return (video_path, frame_seq, audio_path, first_frame)

    def _extract_first_frame(self, video_path):
        cap = cv2.VideoCapture(video_path)
        success, frame = cap.read()
        cap.release()
        if not success:
            raise ValueError("无法读取视频首帧")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(frame_rgb.astype(np.float32) / 255.0).unsqueeze(0)

    def _extract_frames(self, video_path, output_dir, interval):
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % interval == 0:
                cv2.imwrite(os.path.join(output_dir, f"frame_{frame_count:06d}.png"), frame)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(frame_rgb.astype(np.float32) / 255.0)
                frames.append(tensor)
            frame_count += 1
        
        cap.release()
        return torch.cat(frames, dim=0) if frames else torch.zeros((0, 1, 1, 3))

    def _extract_audio(self, video_path, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        audio_path = os.path.join(output_dir, "audio.mp3")
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-q:a', '0', '-map', 'a', audio_path
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return audio_path
        except Exception as e:
            print(f"音频提取失败: {str(e)}")
            return ""

NODE_CLASS_MAPPINGS = {"VideoProcessorNode": VideoProcessorNode}
NODE_DISPLAY_NAME_MAPPINGS = {"VideoProcessorNode": "⒎11加载视频♈微信stone_liwei"}