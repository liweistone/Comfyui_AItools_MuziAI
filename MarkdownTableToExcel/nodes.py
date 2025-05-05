import os
import re
import pandas as pd
from datetime import datetime
from pathlib import Path

class MarkdownTableToExcel:
    """
    ComfyUI节点：将Markdown表格或类似表格文本转换为Excel文件
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "markdown_text": ("STRING", {
                    "multiline": True,
                    "default": "| 姓名 | 年龄 |\n|------|------|\n| 张三 | 28   |"
                }),
                "output_dir": ("STRING", {
                    "default": "output/excel_tables",
                    "folder": True
                }),
                "filename": ("STRING", {
                    "default": "converted_table"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "convert"
    CATEGORY = "🎨公众号懂AI的木子做号工具/工具杂项"
    
    def convert(self, markdown_text, output_dir, filename):
        # 确保输出目录存在
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_filename = f"{filename}_{timestamp}.xlsx" if filename else f"table_{timestamp}.xlsx"
        excel_path = output_path / excel_filename  # 修正了这里的变量名
        
        # 转换表格
        df = self._text_to_dataframe(markdown_text)
        
        # 保存为Excel
        if not df.empty:
            df.to_excel(excel_path, index=False)
            return (str(excel_path),)
        else:
            raise ValueError("未检测到有效的表格格式")

    def _text_to_dataframe(self, text):
        """将文本表格转换为DataFrame，支持多种格式"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # 检测是否为Markdown表格
        if any('|' in line for line in lines):
            return self._markdown_to_dataframe(lines)
        # 否则按简单表格处理
        else:
            return self._simple_table_to_dataframe(lines)
    
    def _markdown_to_dataframe(self, lines):
        """处理Markdown格式表格"""
        # 移除表头分隔线
        lines = [line for line in lines if not re.match(r'^\|?[-:]+(\|[-:]+)*\|?$', line)]
        
        data = []
        for line in lines:
            # 移除行首尾的管道符
            line = line.strip('|')
            cells = [cell.strip() for cell in line.split('|') if cell.strip()]
            data.append(cells)
        
        if len(data) < 1:
            return pd.DataFrame()
        
        # 第一行作为表头，其余作为数据
        header = data[0] if len(data) > 0 else []
        rows = data[1:] if len(data) > 1 else []
        
        # 确保每行数据与表头列数一致
        max_cols = len(header)
        processed_rows = []
        for row in rows:
            if len(row) > max_cols:
                row = row[:max_cols]
            elif len(row) < max_cols:
                row.extend([''] * (max_cols - len(row)))
            processed_rows.append(row)
        
        return pd.DataFrame(processed_rows, columns=header) if header else pd.DataFrame(processed_rows)
    
    def _simple_table_to_dataframe(self, lines):
        """处理简单的表格文本（类似CSV）"""
        data = []
        for line in lines:
            # 尝试按制表符分割，然后按空格分割
            cells = re.split(r'\t|\s{2,}', line.strip())
            cells = [cell.strip() for cell in cells if cell.strip()]
            data.append(cells)
        
        if not data:
            return pd.DataFrame()
        
        # 自动检测表头（如果第一行看起来像表头）
        if len(data) > 1 and len(data[0]) == len(data[1]):
            header = data[0]
            rows = data[1:]
        else:
            header = None
            rows = data
        
        return pd.DataFrame(rows, columns=header) if header else pd.DataFrame(rows)

NODE_CLASS_MAPPINGS = {
    "MarkdownTableToExcel": MarkdownTableToExcel
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MarkdownTableToExcel": "表格文本转Excel"
}