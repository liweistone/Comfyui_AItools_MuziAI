import os
import shutil
from PIL import Image
import comfy.utils

class BatchImageRenamer:
    """
    批量图片重命名与格式转换节点
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_folder": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "placeholder": "源文件夹绝对路径（如 /path/to/input）"
                }),
                "target_folder": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "placeholder": "目标文件夹绝对路径（如 /path/to/output）"
                }),
                "filename_template": ("STRING", {
                    "default": "image_{index:04d}",
                    "placeholder": "文件名模板（可用{index}占位符）"
                }),
                "start_index": ("INT", {"default": 1, "min": 0, "max": 999999}),
                "new_extension": (["jpg", "png", "webp"], {"default": "jpg"}),
                "overwrite": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "rename_images"
    CATEGORY = "🎨公众号懂AI的木子做号工具/工具杂项"
    OUTPUT_NODE = True

    def rename_images(self, source_folder, target_folder, filename_template, start_index, new_extension, overwrite):
        # 输入验证
        if not os.path.exists(source_folder):
            raise ValueError(f"❌ 源文件夹不存在: {source_folder}")
            
        os.makedirs(target_folder, exist_ok=True)

        # 获取所有图片文件
        supported_ext = ['.jpg', '.jpeg', '.png', '.webp']
        image_files = []
        for f in os.listdir(source_folder):
            file_path = os.path.join(source_folder, f)
            if os.path.isfile(file_path) and os.path.splitext(f)[1].lower() in supported_ext:
                image_files.append(file_path)

        if not image_files:
            raise ValueError("❌ 源文件夹中没有支持的图片文件（jpg/png/webp）")

        # 处理重命名
        success_count = 0
        for idx, src_path in enumerate(sorted(image_files), start=start_index):
            try:
                # 生成新文件名
                new_filename = filename_template.format(index=idx)
                new_path = os.path.join(target_folder, f"{new_filename}.{new_extension}")

                # 处理重复文件
                if os.path.exists(new_path) and not overwrite:
                    raise FileExistsError(f"文件已存在: {new_path}")

                # 转换并保存图片
                img = Image.open(src_path).convert("RGB")
                save_kwargs = {"optimize": True}
                if new_extension == "jpg":
                    save_kwargs.update({"quality": 95, "subsampling": 0})
                elif new_extension == "webp":
                    save_kwargs["quality"] = 90

                img.save(new_path, **save_kwargs)
                success_count += 1
            except Exception as e:
                comfy.utils.print_error(f"处理文件失败 {src_path}: {str(e)}")

        return {"ui": {"text": [f"成功处理 {success_count}/{len(image_files)} 文件"]}, "result": (f"完成！输出至：{target_folder}",)}

NODE_CLASS_MAPPINGS = {"BatchImageRenamer": BatchImageRenamer}
NODE_DISPLAY_NAME_MAPPINGS = {"BatchImageRenamer": "😋批量图片重命名和格式转换"}