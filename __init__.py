"""
木子AI超级工具包 v3.0
集成功能模块：
1. 人物特征调节（5种LoRA模型）
2. 视频处理工具（裁剪/帧提取/合成）
3. 抖音作品下载器
4. Gemini AI提示词增强
5. 批量图片处理工具
6. 随机素材生成器
7. Gemini AI图像编辑（新增）
8. 视频处理器（新增）
"""

# --------------------- 导入各模块映射 ---------------------
from .nodes import (
    NODE_CLASS_MAPPINGS as CHARACTER_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as CHARACTER_DISPLAY_MAPPINGS
)

from .video_caijian_nodes import (
    NODE_CLASS_MAPPINGS as VIDEO_TRIM_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as VIDEO_TRIM_DISPLAY_MAPPINGS
)

from .Douyin_Downloader import (
    NODE_CLASS_MAPPINGS as DOUYIN_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as DOUYIN_DISPLAY_MAPPINGS
)

from .gemini_ai import (
    NODE_CLASS_MAPPINGS as GEMINI_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as GEMINI_DISPLAY_MAPPINGS
)

from .data_to_onevideo import (
    NODE_CLASS_MAPPINGS as RANDOM_VIDEO_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as RANDOM_VIDEO_DISPLAY_MAPPINGS
)

from .piliangtupianmingming import (
    NODE_CLASS_MAPPINGS as IMAGE_RENAME_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as IMAGE_RENAME_DISPLAY_MAPPINGS
)

from .shuchuvideooneimg import (
    NODE_CLASS_MAPPINGS as VIDEO_FRAME_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as VIDEO_FRAME_DISPLAY_MAPPINGS
)

from .suijitiaochuimg import (
    NODE_CLASS_MAPPINGS as RANDOM_IMAGE_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as RANDOM_IMAGE_DISPLAY_MAPPINGS
)

from .videotoimg import (
    NODE_CLASS_MAPPINGS as VIDEO_TO_IMG_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as VIDEO_TO_IMG_DISPLAY_MAPPINGS
)

from .xulieimgtovideo import (
    NODE_CLASS_MAPPINGS as IMG_TO_VIDEO_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as IMG_TO_VIDEO_DISPLAY_MAPPINGS
)

# --------------------- 新增模块映射 ---------------------
from .zhidingmulusuijijiazaishipin import (
    NODE_CLASS_MAPPINGS as CUSTOM_VIDEO_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as CUSTOM_VIDEO_DISPLAY_MAPPINGS
)

from .prompt_master import (
    NODE_CLASS_MAPPINGS as prompt_master_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as prompt_master_DISPLAY_MAPPINGS
) 

# 新增VideoProcessor模块
from .VideoProcessor.nodes import (
    NODE_CLASS_MAPPINGS as VIDEO_PROCESSOR_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as VIDEO_PROCESSOR_DISPLAY_MAPPINGS
)

# 新增video_creator视频制作系列模块
from .video_creator.nodes import (
    NODE_CLASS_MAPPINGS as VIDEO_ZHIZUO_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as VIDEO_ZHIZUO_DISPLAY_MAPPINGS
)

# 新增aitxt_list从路径文件批量提示词列模块
from .aitxt_list.nodes import (
    NODE_CLASS_MAPPINGS as aitxt_list_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as aitxt_list_DISPLAY_MAPPINGS
)

# 新增MarkdownTableToExcel  md格式转表格模块
from .MarkdownTableToExcel.nodes import (
    NODE_CLASS_MAPPINGS as MarkdownTableToExcel_MAPPINGS,  
    NODE_DISPLAY_NAME_MAPPINGS as MarkdownTableToExcel_DISPLAY_MAPPINGS  
)

 

# --------------------- 合并所有映射 ---------------------
NODE_CLASS_MAPPINGS = {
    **MarkdownTableToExcel_MAPPINGS, #md格式转表格模块
    **aitxt_list_MAPPINGS,
    # 人物特征调节
    **CHARACTER_MAPPINGS,
    
    # 视频处理工具
    **VIDEO_TRIM_MAPPINGS,
    **VIDEO_FRAME_MAPPINGS,
    **VIDEO_TO_IMG_MAPPINGS,
    **IMG_TO_VIDEO_MAPPINGS,
    **VIDEO_PROCESSOR_MAPPINGS,  # 新增视频处理器    
    **VIDEO_ZHIZUO_MAPPINGS,  # 新增视频制作创作
    
    # 抖音下载器
    **DOUYIN_MAPPINGS,
    
    # Gemini AI
    **GEMINI_MAPPINGS,
 
    
    # 图片处理
    **IMAGE_RENAME_MAPPINGS,
    **RANDOM_IMAGE_MAPPINGS,
    
    # 随机素材
    **RANDOM_VIDEO_MAPPINGS,
    
    # 新增模块
    **CUSTOM_VIDEO_MAPPINGS,      # 指定目录视频加载
    **prompt_master_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **MarkdownTableToExcel_DISPLAY_MAPPINGS, # md格式转表格模块
    **aitxt_list_DISPLAY_MAPPINGS,
    # 人物特征调节
    **CHARACTER_DISPLAY_MAPPINGS,
    
    # 视频处理工具
    **VIDEO_TRIM_DISPLAY_MAPPINGS,
    **VIDEO_FRAME_DISPLAY_MAPPINGS,
    **VIDEO_TO_IMG_DISPLAY_MAPPINGS,
    **IMG_TO_VIDEO_DISPLAY_MAPPINGS,
    **VIDEO_PROCESSOR_DISPLAY_MAPPINGS,  # 新增视频处理器
    **VIDEO_ZHIZUO_DISPLAY_MAPPINGS,  # 新增视频制作创作
    
    # 抖音下载器
    **DOUYIN_DISPLAY_MAPPINGS,
    
    # Gemini AI
    **GEMINI_DISPLAY_MAPPINGS,
 
    
    # 图片处理
    **IMAGE_RENAME_DISPLAY_MAPPINGS,
    **RANDOM_IMAGE_DISPLAY_MAPPINGS,
    
    # 随机素材
    **RANDOM_VIDEO_DISPLAY_MAPPINGS,
    
    # 新增模块
    **CUSTOM_VIDEO_DISPLAY_MAPPINGS,  # "⒊指定目录加载视频♎微信stone_liwei"
    **prompt_master_DISPLAY_MAPPINGS,  # "⒎随机生成+教程输出♊微信stone_liwei"  
}

# --------------------- 模块配置 ---------------------
__all__ = [
    'NODE_CLASS_MAPPINGS',
    'NODE_DISPLAY_NAME_MAPPINGS'
]

WEB_DIRECTORY = "./web"

# --------------------- 环境检查 ---------------------
try:
    import torch
    import comfy
    from packaging import version
    
    def check_environment():
        # PyTorch版本检查
        torch_min = version.parse("1.12.0")
        torch_current = version.parse(torch.__version__)
        if torch_current < torch_min:
            print(f"⚠️ 需要PyTorch版本 >= {torch_min} (当前: {torch_current}")
        
        # ComfyUI版本检查
        if hasattr(comfy, '__version__'):
            comfy_min = version.parse("1.0.0")
            comfy_current = version.parse(comfy.__version__)
            if comfy_current < comfy_min:
                print(f"⚠️ 需要ComfyUI版本 >= {comfy_min} (当前: {comfy_current}")
        
        # CUDA可用性检查
        if not torch.cuda.is_available():
            print("⚠️ 未检测到CUDA加速，建议使用GPU运行")
    
    check_environment()
except Exception as e:
    print(f"环境检查失败: {str(e)}")