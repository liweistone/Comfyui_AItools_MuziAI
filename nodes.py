import os
import folder_paths
import torch
import comfy.utils
from comfy.sd import load_lora_for_models
from nodes import LoraLoader
from tqdm import tqdm
import requests
import time
import traceback
import numpy as np
from PIL import Image, ImageOps
import node_helpers
from colour.io.luts.iridas_cube import read_LUT_IridasCube
import inspect  # 新增关键导入

##############################################
#                Lora下载器                  #
##############################################
class LoraDownloader:
    """
    完整的Lora文件下载器，支持多源下载和自动重试
    """
    @classmethod
    def get_lora_dir(cls):
        """获取Lora存储目录"""
        plugin_dir = os.path.dirname(os.path.realpath(__file__))
        loras_dir = os.path.join(plugin_dir, "loras")
        os.makedirs(loras_dir, exist_ok=True)
        return loras_dir

    @classmethod
    def download_lora(cls, lora_url, lora_name, max_retries=3, timeout=60):
        """
        完整的Lora下载方法
        """
        loras_dir = cls.get_lora_dir()
        lora_path = os.path.join(loras_dir, lora_name)
        
        # 检查文件是否已存在且完整
        if os.path.exists(lora_path) and os.path.getsize(lora_path) > 1024*1024:
            try:
                # 验证文件是否可加载
                comfy.utils.load_torch_file(lora_path, safe_load=True)
                print(f"[Lora下载器] 使用现有有效文件: {lora_path}")
                return lora_path
            except:
                print(f"[Lora下载器] 文件损坏，将重新下载: {lora_path}")
                os.remove(lora_path)

        # 准备备用下载源
        mirror_url = lora_url.replace(
            "https://huggingface.co", 
            "https://hf-mirror.com"
        )
        urls_to_try = [mirror_url, lora_url]

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        }

        last_error = None
        for retry in range(max_retries):
            for url in urls_to_try:
                try:
                    print(f"\n[Lora下载器] 尝试下载(源: {url})...")
                    response = requests.get(
                        url,
                        headers=headers,
                        stream=True,
                        timeout=timeout
                    )
                    response.raise_for_status()

                    total_size = int(response.headers.get('content-length', 0))
                    if total_size < 1024*1024:
                        raise ValueError(f"文件大小异常: {total_size}字节")

                    # 下载文件
                    with open(lora_path, 'wb') as f:
                        with tqdm(
                            total=total_size,
                            unit='B',
                            unit_scale=True,
                            unit_divisor=1024,
                            desc=f"下载 {lora_name}",
                            miniters=1
                        ) as pbar:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    pbar.update(len(chunk))

                    # 验证文件完整性
                    if os.path.getsize(lora_path) == total_size:
                        try:
                            # 验证文件可加载
                            comfy.utils.load_torch_file(lora_path, safe_load=True)
                            print(f"[Lora下载器] 下载成功: {lora_path}")
                            return lora_path
                        except Exception as e:
                            os.remove(lora_path)
                            raise ValueError(f"下载文件无法加载: {str(e)}")
                    else:
                        os.remove(lora_path)
                        raise IOError("文件下载不完整")

                except Exception as e:
                    last_error = e
                    print(f"[Lora下载器] 下载失败: {str(e)}")
                    if os.path.exists(lora_path):
                        os.remove(lora_path)
                    
                    if retry < max_retries - 1:
                        wait_time = 2 ** retry
                        print(f"[Lora下载器] {wait_time}秒后重试...")
                        time.sleep(wait_time)
                    continue

        print(f"\n[Lora下载器] 所有尝试失败! 请手动下载:")
        print(f"URL: {lora_url}")
        print(f"保存到: {loras_dir}")
        return None

##############################################
#             基础Lora加载类                 #
##############################################
class BaseLoraLoader:
    """
    完整的Lora加载基类
    """
    def __init__(self):
        self.loaded_lora = None
        self.lora_name = None
        self.lora_url = None

    def get_lora_path(self):
        """获取Lora路径的完整实现"""
        if not self.lora_name or not self.lora_url:
            raise ValueError("Lora文件名或URL未设置")

        print(f"\n[Lora加载器] 检查Lora文件: {self.lora_name}")
        return LoraDownloader.download_lora(self.lora_url, self.lora_name)

    def apply_lora(self, model, clip, strength, info_text=None):
        """应用Lora的完整实现"""
        if strength == 0:
            print("[Lora加载器] 强度为0，跳过应用")
            return (model, clip)

        lora_path = self.get_lora_path()
        if not lora_path:
            print("[Lora加载器] 无法获取Lora文件，使用原始模型")
            return (model, clip)

        # 检查是否已加载相同的Lora
        if self.loaded_lora and self.loaded_lora[0] == lora_path:
            print("[Lora加载器] 使用已加载的Lora缓存")
            lora = self.loaded_lora[1]
        else:
            # 清理旧Lora
            if self.loaded_lora:
                old_lora = self.loaded_lora
                self.loaded_lora = None
                del old_lora

            # 加载新Lora
            try:
                print(f"[Lora加载器] 加载Lora文件: {lora_path}")
                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                self.loaded_lora = (lora_path, lora)
                print("[Lora加载器] Lora文件加载成功")
            except Exception as e:
                print(f"[Lora加载器] 加载失败: {str(e)}")
                traceback.print_exc()
                return (model, clip)

        # 应用Lora
        try:
            print(f"[Lora加载器] 应用Lora，强度: {strength}")
            model_lora, clip_lora = load_lora_for_models(
                model, 
                clip, 
                lora,
                strength,
                strength
            )
            print("[Lora加载器] Lora应用成功")
            return (model_lora, clip_lora)
        except Exception as e:
            print(f"[Lora加载器] 应用失败: {str(e)}")
            traceback.print_exc()
            return (model, clip)

##############################################
#             各功能调节器实现                #
##############################################

#==============💓胸部大小调节器==================
class BreastSizeAdjuster(BaseLoraLoader):
    def __init__(self):
        super().__init__()
        self.lora_name = "breast_size_lora.safetensors"
        self.lora_url = "https://huggingface.co/liguanwei/mymodels/resolve/main/breast_size_lora.safetensors"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "size_strength": ("FLOAT", {
                    "default": 0.3, 
                    "min": -1.0, 
                    "max": 2.0, 
                    "step": 0.1,
                    "display": "slider"
                }),
            },
            "optional": {
                "info_text": ("STRING", {
                    "multiline": True,
                    "default": "💓胸部大小调节器使用说明:\n"
                              "1. 连接模型和CLIP到本节点\n"
                              "2. 调整size_strength参数控制胸部大小\n"
                              "   - 正值增大胸部\n"
                              "   - 负值减小胸部\n"
                              "   - 0表示不调整\n"
                              "3. 建议值范围:0.1-0.5(自然调整)\n"
                              "4. 首次使用会自动下载LoRA模型(约100MB)\n\n"
                              "⚠️注意:过度调整可能导致不自然效果"
                }),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "apply_breast_size"
    CATEGORY = "🎨公众号懂AI的木子做号工具/人物增强调节"
    OUTPUT_NODE = True
    
    def apply_breast_size(self, model, clip, size_strength, info_text=None):
        return self.apply_lora(model, clip, size_strength, info_text)

#==============💓NSFW版胸部调节器===============
class BreastSizeAdjusternswf(BaseLoraLoader):
    def __init__(self):
        super().__init__()
        self.lora_name = "breast_size_nswf.safetensors"
        self.lora_url = "https://huggingface.co/liguanwei/mymodels/resolve/main/breast_size_nswf.safetensors"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "size_strength": ("FLOAT", {
                    "default": 0.3, 
                    "min": -1.0, 
                    "max": 2.0, 
                    "step": 0.1,
                    "display": "slider"
                }),
            },
            "optional": {
                "info_text": ("STRING", {
                    "multiline": True,
                    "default": "💓胸部大小调节器NSFW版使用说明:\n"
                              "1. 此版本适用于更大幅度的胸部调整\n"
                              "2. 连接模型和CLIP到本节点\n"
                              "3. 调整size_strength参数控制胸部大小\n"
                              "   - 正值增大胸部(效果更明显)\n"
                              "   - 负值减小胸部\n"
                              "   - 0表示不调整\n"
                              "4. 建议值范围:0.1-1.0\n"
                              "5. 首次使用会自动下载LoRA模型\n\n"
                              "⚠️注意:此版本调整幅度较大,请谨慎使用"
                }),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "apply_breast_size"
    CATEGORY = "🎨公众号懂AI的木子做号工具/人物增强调节"
    OUTPUT_NODE = True
    
    def apply_breast_size(self, model, clip, size_strength, info_text=None):
        return self.apply_lora(model, clip, size_strength, info_text)

#==============✋手部稳定调节器=================
class HandStabilityAdjuster(BaseLoraLoader):
    def __init__(self):
        super().__init__()
        self.lora_name = "hand_stability_lora.safetensors"
        self.lora_url = "https://huggingface.co/liguanwei/mymodels/resolve/main/hand_stability_lora.safetensors"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "stability_strength": ("FLOAT", {
                    "default": 0.3, 
                    "min": 0.0, 
                    "max": 2.0, 
                    "step": 0.05,
                    "display": "slider"
                }),
            },
            "optional": {
                "info_text": ("STRING", {
                    "multiline": True,
                    "default": "✋手部稳定调节器使用说明:\n"
                              "1. 连接模型和CLIP到本节点\n"
                              "2. 调整stability_strength参数控制手部稳定性\n"
                              "   - 0.1-0.3: 轻微改善\n"
                              "   - 0.3-0.7: 明显改善\n"
                              "   - 0.7-1.0: 强力修正\n"
                              "3. 首次使用会自动下载LoRA模型\n\n"
                              "💡提示:\n"
                              "- 适用于修复扭曲的手指和手部\n"
                              "- 过高值可能导致手部细节丢失"
                }),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "apply_hand_stability"
    CATEGORY = "🎨公众号懂AI的木子做号工具/人物增强调节"
    OUTPUT_NODE = True
    
    def apply_hand_stability(self, model, clip, stability_strength, info_text=None):
        return self.apply_lora(model, clip, stability_strength, info_text)

#==============🔥性感风格调节器=================
class SexyStyleAdjuster(BaseLoraLoader):
    def __init__(self):
        super().__init__()
        self.lora_name = "sexy_style_lora.safetensors"
        self.lora_url = "https://huggingface.co/liguanwei/mymodels/resolve/main/sexy_style_lora.safetensors"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "sexy_strength": ("FLOAT", {
                    "default": 0.5, 
                    "min": 0.0, 
                    "max": 2.0, 
                    "step": 0.05,
                    "display": "slider"
                }),
            },
            "optional": {
                "info_text": ("STRING", {
                    "multiline": True,
                    "default": "🔥性感风格调节器使用说明:\n"
                              "1. 连接模型和CLIP到本节点\n"
                              "2. 调整sexy_strength参数控制性感程度\n"
                              "   - 0.1-0.5: 轻微性感风格\n"
                              "   - 0.5-1.0: 明显性感风格\n"
                              "   - 1.0-2.0: 强烈性感风格\n"
                              "3. 首次使用会自动下载LoRA模型\n\n"
                              "💡提示:\n"
                              "- 适用于女性角色性感风格增强\n"
                              "- 过高值可能导致服装过于暴露"
                }),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "apply_sexy_style"
    CATEGORY = "🎨公众号懂AI的木子做号工具/人物增强调节"
    OUTPUT_NODE = True
    
    def apply_sexy_style(self, model, clip, sexy_strength, info_text=None):
        return self.apply_lora(model, clip, sexy_strength, info_text)

#==============😍网红网感调节器=================
class Influencer_regulator(BaseLoraLoader):
    def __init__(self):
        super().__init__()
        self.lora_name = "Beautifulgirl_size_lora.safetensors"
        self.lora_url = "https://huggingface.co/liguanwei/mymodels/resolve/main/Beautifulgirl_size_lora.safetensors"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "size_strength": ("FLOAT", {
                    "default": 0.3, 
                    "min": 0.0, 
                    "max": 2.0, 
                    "step": 0.1,
                    "display": "slider"
                }),
            },
            "optional": {
                "info_text": ("STRING", {
                    "multiline": True,
                    "default": "😍网感调节器使用说明:\n"
                              "1. 连接模型和CLIP到本节点\n"
                              "2. 调整size_strength参数控制网红风格强度\n"
                              "   - 0.1-0.5: 轻微网红风格\n"
                              "   - 0.5-1.0: 明显网红风格\n"
                              "   - 1.0-2.0: 强烈网红风格\n"
                              "3. 首次使用会自动下载LoRA模型\n\n"
                              "💡效果说明:\n"
                              "- 增强面部特征(大眼睛、小脸等)\n"
                              "- 优化皮肤质感\n"
                              "- 增加时尚感妆容"
                }),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "apply_breast_size"
    CATEGORY = "🎨公众号懂AI的木子做号工具/人物增强调节"
    OUTPUT_NODE = True
    
    def apply_breast_size(self, model, clip, size_strength, info_text=None):
        return self.apply_lora(model, clip, size_strength, info_text)

##############################################
#                LUT滤镜相关                 #
##############################################
class LUTDownloader:
    LUT_REPO = "https://huggingface.co/datasets/liguanwei/luts/resolve/main/"
    LUT_FILES = [
        "快速电影.cube",
        "时尚电影.cube", 
        "胶片颗粒质感电影.cube"
    ]
    
    @classmethod
    def get_lut_dir(cls):
        current_script_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
        plugin_dir = os.path.dirname(current_script_path)
        lut_dir = os.path.join(plugin_dir, "luts")
        os.makedirs(lut_dir, exist_ok=True)
        return lut_dir
    
    @classmethod 
    def download_luts(cls):
        try:
            lut_dir = cls.get_lut_dir()
            
            for lut_file in cls.LUT_FILES:
                dest_path = os.path.join(lut_dir, lut_file)
                if not os.path.exists(dest_path):
                    try:
                        url = cls.LUT_REPO + lut_file
                        print(f"[LUT下载器] 正在下载: {url}")
                        response = requests.get(url, stream=True)
                        response.raise_for_status()
                        
                        total_size = int(response.headers.get('content-length', 0))
                        with open(dest_path, 'wb') as f:
                            with tqdm(
                                desc=f"下载 {lut_file}",
                                total=total_size,
                                unit='iB',
                                unit_scale=True,
                                unit_divisor=1024,
                            ) as bar:
                                for data in response.iter_content(chunk_size=1024):
                                    size = f.write(data)
                                    bar.update(size)
                        print(f"[LUT下载器] 成功下载到: {dest_path}")
                    except Exception as e:
                        print(f"[LUT下载器] 下载失败 {lut_file}: {str(e)}")
        except Exception as e:
            print(f"[LUT下载器] 初始化错误: {str(e)}")
        finally:
            return True  # 确保始终返回，避免阻塞节点注册

class ESSImageApplyLUT:
    @classmethod
    def INPUT_TYPES(s):
        # 添加异常捕获防止下载失败影响节点注册
        try:
            LUTDownloader.download_luts()
        except Exception as e:
            print(f"[滤镜节点] LUT预下载失败: {str(e)}")
        
        lut_dir = LUTDownloader.get_lut_dir()
        lut_files = []
        try:
            lut_files = [f for f in os.listdir(lut_dir) 
                        if f.lower().endswith('.cube') and os.path.isfile(os.path.join(lut_dir, f))]
        except Exception as e:
            print(f"[滤镜节点] 获取LUT列表错误: {str(e)}")
        
        return {
            "required": {
                "image": ("IMAGE",),
                "lut_file": (sorted(lut_files),),
                "gamma_correction": ("BOOLEAN", { "default": True }),
                "clip_values": ("BOOLEAN", { "default": True }),
                "strength": ("FLOAT", {
                    "default": 20.0, 
                    "min": 0.0, 
                    "max": 80.0, 
                    "step": 0.01,
                    "display": "slider"
                }),
            },
            "optional": {
                "info_text": ("STRING", {
                    "multiline": True,
                    "default": "😍图像调色滤镜:\n"
                              "1. 连接到图片节点\n"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "🎨公众号懂AI的木子做号工具/人物增强调节"  # 与截图分类保持一致
    OUTPUT_NODE = True

    def execute(self, image, lut_file, gamma_correction, clip_values, strength, info_text=None):
        try:
            lut_dir = LUTDownloader.get_lut_dir()
            lut_file_path = os.path.join(lut_dir, lut_file)
            
            if not os.path.isfile(lut_file_path):
                raise FileNotFoundError(f"LUT文件不存在: {lut_file_path}")
            if not os.access(lut_file_path, os.R_OK):
                raise PermissionError(f"无法读取LUT文件: {lut_file_path}")
            
            lut = read_LUT_IridasCube(lut_file_path)
            lut.name = lut_file

            if clip_values:
                if lut.domain[0].max() == lut.domain[0].min() and lut.domain[1].max() == lut.domain[1].min():
                    lut.table = np.clip(lut.table, lut.domain[0, 0], lut.domain[1, 0])
                else:
                    if len(lut.table.shape) == 2:
                        for dim in range(3):
                            lut.table[:, dim] = np.clip(lut.table[:, dim], lut.domain[0, dim], lut.domain[1, dim])
                    else:
                        for dim in range(3):
                            lut.table[:, :, :, dim] = np.clip(lut.table[:, :, :, dim], lut.domain[0, dim], lut.domain[1, dim])

            out = []
            for img in image:
                lut_img = img.cpu().numpy().copy()

                is_non_default_domain = not np.array_equal(lut.domain, np.array([[0., 0., 0.], [1., 1., 1.]]))
                dom_scale = None
                if is_non_default_domain:
                    dom_scale = lut.domain[1] - lut.domain[0]
                    lut_img = lut_img * dom_scale + lut.domain[0]
                if gamma_correction:
                    lut_img = lut_img ** (1/2.2)
                lut_img = lut.apply(lut_img)
                if gamma_correction:
                    lut_img = lut_img ** (2.2)
                if is_non_default_domain:
                    lut_img = (lut_img - lut.domain[0]) / dom_scale

                lut_img = torch.from_numpy(lut_img).to(image.device)
                if strength < 1.0:
                    lut_img = strength * lut_img + (1 - strength) * img
                out.append(lut_img)

            return (torch.stack(out), )
            
        except Exception as e:
            print(f"[滤镜节点] 处理错误: {str(e)}")
            return (image, )

##############################################
#               字符选择器                 #
##############################################
class HiddenStringSwitch:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 5000,
                    "step": 1,
                    "display": "counter"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("string", "index")
    FUNCTION = "switch"
    CATEGORY = "🎨公众号懂AI的木子做号工具/工具杂项"

    def switch(self, index):
        max_num = 5000
        selected_index = max(1, min(index, max_num))
        return (str(selected_index), selected_index)

##############################################
#               显示我的二维码                 #
##############################################
class LoadImagecode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}

    CATEGORY = "🎨公众号懂AI的木子做号工具/工具杂项"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "load_image"
    OUTPUT_NODE = True

    def load_image(self, **kwargs):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(current_dir, "code", "code.png")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片未找到: {image_path}")

        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img).convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np)[None,]
        
        return (img_tensor,)

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return "static_image"

##############################################
#               节点注册部分                 #
##############################################
NODE_CLASS_MAPPINGS = {
    "BreastSizeAdjuster": BreastSizeAdjuster,
    "BreastSizeAdjusternswf": BreastSizeAdjusternswf,
    "HandStabilityAdjuster": HandStabilityAdjuster,
    "SexyStyleAdjuster": SexyStyleAdjuster,
    "Influencer_regulator": Influencer_regulator,
    "ESSImageApplyLUT": ESSImageApplyLUT,  # 确保注册
    "HiddenStringSwitch": HiddenStringSwitch,
    "LoadImagecode": LoadImagecode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BreastSizeAdjuster": "💓胸部大小调节器(木子AI)",
    "BreastSizeAdjusternswf": "💓胸部大小调节器NSWF版(木子AI)",
    "HandStabilityAdjuster": "✋手部稳定调节器(木子AI)", 
    "SexyStyleAdjuster": "🔥性感风格调节器(木子AI)",
    "Influencer_regulator": "😍网感调节器(木子AI)",
    "ESSImageApplyLUT": "🔧 滤镜风格调节器",  # 显示名称
    "HiddenStringSwitch": "字符串切换器",
    "LoadImagecode": "微信公众号二维码"
}
