import os
import folder_paths
import torch
import comfy.utils
from comfy.sd import load_lora_for_models
from nodes import LoraLoader




#==============💓胸部大小调节器(公众号懂AI的木子)==================
class BreastSizeAdjuster:
    def __init__(self):
        self.loaded_lora = None
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
    CATEGORY = "🎨公众号懂AI的木子做号工具/人物增加调节"
    OUTPUT_NODE = True
    
    def get_lora_path(self):
        plugin_dir = os.path.dirname(os.path.realpath(__file__))
        loras_dir = os.path.join(plugin_dir, "loras")
        os.makedirs(loras_dir, exist_ok=True)
        
        lora_path = os.path.join(loras_dir, self.lora_name)
        
        if not os.path.exists(lora_path):
            try:
                import requests
                print(f"正在下载胸部大小LoRA模型...")
                response = requests.get(self.lora_url, stream=True)
                response.raise_for_status()
                
                with open(lora_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("胸部LoRA模型下载完成!")
            except Exception as e:
                print(f"胸部模型下载失败: {e}")
                return None
        
        return lora_path
    
    def apply_breast_size(self, model, clip, size_strength, info_text=None):
        if size_strength == 0:
            return (model, clip)
            
        lora_path = self.get_lora_path()
        if not lora_path:
            print("无法加载胸部LoRA模型，使用原始模型")
            return (model, clip)
        
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        if lora is None:
            try:
                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                self.loaded_lora = (lora_path, lora)
            except Exception as e:
                print(f"加载胸部LoRA文件失败: {e}")
                return (model, clip)
        
        try:
            model_lora, clip_lora = load_lora_for_models(
                model, 
                clip, 
                lora,
                size_strength,
                size_strength
            )
            return (model_lora, clip_lora)
        except Exception as e:
            print(f"应用胸部LoRA时出错: {e}")
            return (model, clip)


#==============💓胸部大小调节器nswf版(木子AI)==================
class BreastSizeAdjusternswf:
    def __init__(self):
        self.loaded_lora = None
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
    CATEGORY = "🎨公众号懂AI的木子做号工具/人物增加调节"
    OUTPUT_NODE = True
    
    def get_lora_path(self):
        plugin_dir = os.path.dirname(os.path.realpath(__file__))
        loras_dir = os.path.join(plugin_dir, "loras")
        os.makedirs(loras_dir, exist_ok=True)
        
        lora_path = os.path.join(loras_dir, self.lora_name)
        
        if not os.path.exists(lora_path):
            try:
                import requests
                print(f"正在下载胸部大小LoRA模型...")
                response = requests.get(self.lora_url, stream=True)
                response.raise_for_status()
                
                with open(lora_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("胸部LoRA模型下载完成!")
            except Exception as e:
                print(f"胸部模型下载失败: {e}")
                return None
        
        return lora_path
    
    def apply_breast_size(self, model, clip, size_strength, info_text=None):
        if size_strength == 0:
            return (model, clip)
            
        lora_path = self.get_lora_path()
        if not lora_path:
            print("无法加载胸部LoRA模型，使用原始模型")
            return (model, clip)
        
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        if lora is None:
            try:
                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                self.loaded_lora = (lora_path, lora)
            except Exception as e:
                print(f"加载胸部LoRA文件失败: {e}")
                return (model, clip)
        
        try:
            model_lora, clip_lora = load_lora_for_models(
                model, 
                clip, 
                lora,
                size_strength,
                size_strength
            )
            return (model_lora, clip_lora)
        except Exception as e:
            print(f"应用胸部LoRA时出错: {e}")
            return (model, clip)



#===============✋手部稳定调节器(木子AI)=============================
class HandStabilityAdjuster:
    def __init__(self):
        self.loaded_lora = None
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
    CATEGORY = "🎨公众号懂AI的木子做号工具/人物增加调节"
    OUTPUT_NODE = True
    
    def get_lora_path(self):
        plugin_dir = os.path.dirname(os.path.realpath(__file__))
        loras_dir = os.path.join(plugin_dir, "loras")
        os.makedirs(loras_dir, exist_ok=True)
        
        lora_path = os.path.join(loras_dir, self.lora_name)
        
        if not os.path.exists(lora_path):
            try:
                import requests
                print(f"正在下载手部稳定LoRA模型...")
                response = requests.get(self.lora_url, stream=True)
                response.raise_for_status()
                
                with open(lora_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("手部LoRA模型下载完成!")
            except Exception as e:
                print(f"手部模型下载失败: {e}")
                return None
        
        return lora_path
    
    def apply_hand_stability(self, model, clip, stability_strength, info_text=None):
        if stability_strength == 0:
            return (model, clip)
            
        lora_path = self.get_lora_path()
        if not lora_path:
            print("无法加载手部LoRA模型，使用原始模型")
            return (model, clip)
        
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        if lora is None:
            try:
                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                self.loaded_lora = (lora_path, lora)
            except Exception as e:
                print(f"加载手部LoRA文件失败: {e}")
                return (model, clip)
        
        try:
            model_lora, clip_lora = load_lora_for_models(
                model, 
                clip, 
                lora,
                stability_strength,
                stability_strength
            )
            return (model_lora, clip_lora)
        except Exception as e:
            print(f"应用手部LoRA时出错: {e}")
            return (model, clip)

#=================性感风格调节器===============
class SexyStyleAdjuster:
    def __init__(self):
        self.loaded_lora = None
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
    CATEGORY = "🎨公众号懂AI的木子做号工具/人物增加调节"
    OUTPUT_NODE = True
    
    def get_lora_path(self):
        plugin_dir = os.path.dirname(os.path.realpath(__file__))
        loras_dir = os.path.join(plugin_dir, "loras")
        os.makedirs(loras_dir, exist_ok=True)
        
        lora_path = os.path.join(loras_dir, self.lora_name)
        
        if not os.path.exists(lora_path):
            try:
                import requests
                print(f"正在下载性感风格LoRA模型...")
                response = requests.get(self.lora_url, stream=True)
                response.raise_for_status()
                
                with open(lora_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("性感风格LoRA模型下载完成!")
            except Exception as e:
                print(f"性感风格模型下载失败: {e}")
                return None
        
        return lora_path
    
    def apply_sexy_style(self, model, clip, sexy_strength, info_text=None):
        if sexy_strength == 0:
            return (model, clip)
            
        lora_path = self.get_lora_path()
        if not lora_path:
            print("无法加载性感风格LoRA模型，使用原始模型")
            return (model, clip)
        
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        if lora is None:
            try:
                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                self.loaded_lora = (lora_path, lora)
            except Exception as e:
                print(f"加载性感风格LoRA文件失败: {e}")
                return (model, clip)
        
        try:
            model_lora, clip_lora = load_lora_for_models(
                model, 
                clip, 
                lora,
                sexy_strength,
                sexy_strength
            )
            return (model_lora, clip_lora)
        except Exception as e:
            print(f"应用性感风格LoRA时出错: {e}")
            return (model, clip)



#==============网红网感调节器==================
class Influencer_regulator:
    def __init__(self):
        self.loaded_lora = None
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
    CATEGORY = "🎨公众号懂AI的木子做号工具/人物增加调节"
    OUTPUT_NODE = True
    
    def get_lora_path(self):
        plugin_dir = os.path.dirname(os.path.realpath(__file__))
        loras_dir = os.path.join(plugin_dir, "loras")
        os.makedirs(loras_dir, exist_ok=True)
        
        lora_path = os.path.join(loras_dir, self.lora_name)
        
        if not os.path.exists(lora_path):
            try:
                import requests
                print(f"正在下载网感调节器模型...")
                response = requests.get(self.lora_url, stream=True)
                response.raise_for_status()
                
                with open(lora_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("网感调节器模型下载完成!")
            except Exception as e:
                print(f"网感调节器模型下载失败: {e}")
                return None
        
        return lora_path
    
    def apply_breast_size(self, model, clip, size_strength, info_text=None):
        if size_strength == 0:
            return (model, clip)
            
        lora_path = self.get_lora_path()
        if not lora_path:
            print("无法加载网感调节器模型，使用原始模型")
            return (model, clip)
        
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        if lora is None:
            try:
                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                self.loaded_lora = (lora_path, lora)
            except Exception as e:
                print(f"加载网感调节器文件失败: {e}")
                return (model, clip)
        
        try:
            model_lora, clip_lora = load_lora_for_models(
                model, 
                clip, 
                lora,
                size_strength,
                size_strength
            )
            return (model_lora, clip_lora)
        except Exception as e:
            print(f"应用网感调节器时出错: {e}")
            return (model, clip)



#-============图像应用lut 滤镜风格调节器 代码构建开始=========
import os
import requests
from pathlib import Path
import torch
import numpy as np
import folder_paths
from tqdm import tqdm
import inspect


class LUTDownloader:
    LUT_REPO = "https://huggingface.co/datasets/liguanwei/luts/resolve/main/"  # 注意结尾的斜杠
    LUT_FILES = [
        "快速电影.cube",
        "时尚电影.cube",
        "胶片颗粒质感电影.cube"
    ]  # 示例LUT文件列表
    
    @classmethod
    def get_lut_dir(cls):
        """获取与nodes.py同级的luts目录"""
        # 获取nodes.py的绝对路径
        current_script_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
        plugin_dir = os.path.dirname(current_script_path)
        lut_dir = os.path.join(plugin_dir, "luts")
        os.makedirs(lut_dir, exist_ok=True)
        return lut_dir
    
    @classmethod
    def download_luts(cls, verbose=False):
        try:
            lut_dir = cls.get_lut_dir()
            if verbose:
                print(f"[LUT下载器] LUT存储目录: {lut_dir}")
            
            for lut_file in cls.LUT_FILES:
                dest_path = os.path.join(lut_dir, lut_file)
                if not os.path.exists(dest_path):
                    try:
                        url = cls.LUT_REPO + lut_file
                        if verbose:
                            print(f"[LUT下载器] 正在下载: {url}")
                        response = requests.get(url, stream=True)
                        response.raise_for_status()
                        
                        total_size = int(response.headers.get('content-length', 0))
                        with open(dest_path, 'wb') as f:
                            if verbose:
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
                            else:
                                for data in response.iter_content(chunk_size=1024):
                                    f.write(data)
                    except Exception as e:
                        print(f"[LUT下载器] 下载失败 {lut_file}: {str(e)}")
                elif verbose:
                    print(f"[LUT下载器] 文件已存在: {dest_path}")
        except Exception as e:
            print(f"[LUT下载器] 初始化错误: {str(e)}")


class ESSImageApplyLUT:
    @classmethod
    def INPUT_TYPES(s):
        # 确保LUT文件已下载
        LUTDownloader.download_luts()
        
        # 从插件目录获取LUT文件
        lut_dir = LUTDownloader.get_lut_dir()
        lut_files = []
        try:
            lut_files = [f for f in os.listdir(lut_dir) 
                        if f.lower().endswith('.cube') and os.path.isfile(os.path.join(lut_dir, f))]
            print(f"[滤镜节点] 可用LUT文件: {lut_files}")
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
    CATEGORY = "🎨公众号懂AI的木子做号工具/人物增加调节"
    OUTPUT_NODE = True

    def execute(self, image, lut_file, gamma_correction, clip_values, strength, info_text=None):
        try:
            # 获取插件目录下的LUT文件
            lut_dir = LUTDownloader.get_lut_dir()
            lut_file_path = os.path.join(lut_dir, lut_file)
            
            print(f"[滤镜节点] 正在加载LUT: {lut_file_path}")
            
            # 验证文件是否存在且可读
            if not os.path.isfile(lut_file_path):
                raise FileNotFoundError(f"LUT文件不存在: {lut_file_path}")
            if not os.access(lut_file_path, os.R_OK):
                raise PermissionError(f"无法读取LUT文件: {lut_file_path}")
            
            from colour.io.luts.iridas_cube import read_LUT_IridasCube
            
            # 处理图像
            device = image.device
            lut = read_LUT_IridasCube(lut_file_path)
            lut.name = lut_file

            if clip_values:
                if lut.domain[0].max() == lut.domain[0].min() and lut.domain[1].max() == lut.domain[1].min():
                    lut.table = np.clip(lut.table, lut.domain[0, 0], lut.domain[1, 0])
                else:
                    if len(lut.table.shape) == 2:  # 3x1D
                        for dim in range(3):
                            lut.table[:, dim] = np.clip(lut.table[:, dim], lut.domain[0, dim], lut.domain[1, dim])
                    else:  # 3D
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

                lut_img = torch.from_numpy(lut_img).to(device)
                if strength < 1.0:
                    lut_img = strength * lut_img + (1 - strength) * img
                out.append(lut_img)

            return (torch.stack(out), )
            
        except Exception as e:
            print(f"[滤镜节点] 处理错误: {str(e)}")
            return (image, )

#--------------------------图像应用lut 代码构建结束------------------

#--------------------------字符切换------------------
import comfy
import torch

class HiddenStringSwitch:
    """
    隐藏字符串列表的切换器，只显示计数器
    """
    def __init__(self):
        pass  # 不再需要预先生成列表
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 5000,  # 最大值设为5000
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
        max_num = 5000  # 设置最大值
        selected_index = max(1, min(index, max_num))  # 确保索引在1~5000范围内
        return (str(selected_index), selected_index)  # 动态生成字符串

#--------------------------me  code------------------
import os
import torch
import numpy as np
from PIL import Image, ImageOps
import folder_paths
import node_helpers

class LoadImagecode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}

    CATEGORY = "🎨公众号懂AI的木子做号工具/工具杂项"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "load_image"
    OUTPUT_NODE = True  # 允许直接显示输出

    def load_image(self, **kwargs):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(current_dir, "code", "code.png")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片未找到: {image_path}")

        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img).convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np)[None,]
        
        # 返回图像数据（ComfyUI 会自动尝试渲染）
        return (img_tensor,)

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return "static_image"

 
 
 
 
Influencer_regulator
# 节点映射
NODE_CLASS_MAPPINGS = {
    "BreastSizeAdjuster": BreastSizeAdjuster,
    "BreastSizeAdjusternswf": BreastSizeAdjusternswf,
    "HandStabilityAdjuster": HandStabilityAdjuster,
    "SexyStyleAdjuster": SexyStyleAdjuster,
    "Influencer_regulator": Influencer_regulator,
    "ESS ImageApplyLUT": ESSImageApplyLUT, 
    "HiddenStringSwitch": HiddenStringSwitch,
    "LoadImagecode": LoadImagecode
}

# 节点显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "BreastSizeAdjuster": "💓胸部大小调节器(木子AI)",
    "BreastSizeAdjusternswf": "💓胸部大小调节器NSWF版(木子AI)",
    "HandStabilityAdjuster": "✋手部稳定调节器(木子AI)",
    "SexyStyleAdjuster": "🔥性感风格调节器(木子AI)",
    "Influencer_regulator": "😍网感调节器(木子AI)",
    "ESS ImageApplyLUT": "🔧 滤镜风格调节器", 
    "HiddenStringSwitch": "字符串切换器",
    "LoadImagecode": "微信公众号二维码"
}


 

 
