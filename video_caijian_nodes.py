import os
import subprocess
import cv2
from datetime import datetime
import folder_paths

class VideoTrimNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_path": ("STRING", {"default": "input/video.mp4", "multiline": False}),
                "start_time": ("STRING", {"default": "00:00:00"}),
                "end_time": ("STRING", {"default": "00:00:10"}),
                "output_path": ("STRING", {"default": "[time]/trimmed_video.mp4", "multiline": False})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "trim_video"
    CATEGORY = "🎨公众号懂AI的木子做号工具/懒人做号/视频相关"
    OUTPUT_NODE = True

    def trim_video(self, input_path, start_time, end_time, output_path):
        # 输入验证
        if not os.path.exists(input_path):
            raise ValueError(f"视频文件不存在: {input_path}")

        # 处理输出路径
        final_output_path = self.process_output_path(output_path)
        
        # 获取视频元数据
        total_duration = self.get_video_duration(input_path)
        
        # 时间格式转换
        start_sec = self.time_to_seconds(start_time)
        end_sec = self.time_to_seconds(end_time)
        
        # 有效性检查
        self.validate_times(start_sec, end_sec, total_duration)

        # 执行裁剪命令
        self.execute_ffmpeg_trim(input_path, start_sec, end_sec, final_output_path)

        return (final_output_path,)

    def get_video_duration(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError("无法打开视频文件")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps
        cap.release()
        return round(duration, 2)

    def time_to_seconds(self, time_str):
        parts = []
        try:
            parts = list(map(float, time_str.split(':')))
        except ValueError:
            raise ValueError("无效时间格式，请使用 HH:MM:SS / MM:SS / SS 格式")

        multipliers = [3600, 60, 1]  # 时:分:秒
        if len(parts) > 3:
            raise ValueError("时间格式最多包含小时、分钟、秒")
            
        # 对齐时间单位
        aligned = [0] * (3 - len(parts)) + parts
        return sum(multipliers[-len(aligned):][i] * aligned[i] for i in range(len(aligned)))

    def seconds_to_time(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def process_output_path(self, raw_path):
        """智能处理输出路径"""
        # 替换时间标记
        if "[time]" in raw_path:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            raw_path = raw_path.replace("[time]", timestamp)
        
        # 分离路径和文件名
        dir_path, file_name = os.path.split(raw_path)
        
        # 自动生成文件名
        if not file_name:
            file_name = "trimmed_video.mp4"
        elif '.' not in file_name:
            file_name += ".mp4"
        elif not file_name.lower().endswith(('.mp4', '.avi', '.mov')):
            file_name = os.path.splitext(file_name)[0] + ".mp4"
        
        # 重建完整路径
        final_path = os.path.join(dir_path, file_name)
        
        # 创建目录结构（自动处理中文路径）
        os.makedirs(dir_path, exist_ok=True)
        
        return final_path

    def validate_times(self, start, end, total):
        if start >= end:
            raise ValueError("开始时间必须早于结束时间")
        if end > total:
            raise ValueError(f"结束时间超过视频总时长 ({self.seconds_to_time(total)})")

    def execute_ffmpeg_trim(self, input_path, start, end, output_path):
        duration = end - start
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(start),
            '-i', input_path,
            '-t', str(duration),
            '-c:v', 'copy',
            '-c:a', 'copy',
            '-avoid_negative_ts', '1',
            output_path
        ]
        
        try:
            # 处理中文路径编码
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace',
                env=env
            )
            print(f"FFmpeg命令执行成功: {' '.join(cmd)}")
        except subprocess.CalledProcessError as e:
            error_detail = self.parse_ffmpeg_error(e.stderr)
            raise RuntimeError(
                f"视频裁剪失败！\n"
                f"原因: {error_detail['reason']}\n"
                f"解决方案: {error_detail['solution']}\n"
                f"完整错误: {e.stderr}"
            )

    def parse_ffmpeg_error(self, error_msg):
        """解析FFmpeg错误信息"""
        error_map = {
            "Invalid argument": {
                "reason": "输出路径包含特殊字符或格式错误",
                "solution": "请使用英文路径并确保包含.mp4扩展名"
            },
            "Permission denied": {
                "reason": "文件写入权限不足",
                "solution": "以管理员身份运行ComfyUI或更换输出目录"
            },
            "No such file or directory": {
                "reason": "路径不存在",
                "solution": "检查输出目录是否有效"
            },
            "default": {
                "reason": "未知错误",
                "solution": "检查时间参数有效性并查看完整错误日志"
            }
        }
        
        for key in error_map:
            if key in error_msg:
                return error_map[key]
        return error_map['default']

NODE_CLASS_MAPPINGS = {"VideoTrimNode": VideoTrimNode}
NODE_DISPLAY_NAME_MAPPINGS = {"VideoTrimNode": "✂️视频裁剪工具"}