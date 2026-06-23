"""
TVG ComfyUI 插件入口
=========================================

把本插件的节点注册到 ComfyUI。安装方式：
    把本文件夹（grid_split_node）整体复制到
    ComfyUI/custom_nodes/ 目录下，重启 ComfyUI 即可。

包含节点（统一以 TVG_ 前缀命名）：
    - TVG_LoadImageWithFilename : 加载图片并输出文件名（不含扩展名）
    - TVG_GridSplitAuto         : 九宫格/网格图像智能切分，保证输出数量
                                   精确等于 expect_rows * expect_cols，
                                   按原始精确尺寸保存到磁盘（不做任何
                                   resize/pad，完整保留原始像素）
"""

from .nodes import (
    TVG_LoadImageWithFilename,
    TVG_GridSplitAuto,
)

NODE_CLASS_MAPPINGS = {
    "TVG_LoadImageWithFilename": TVG_LoadImageWithFilename,
    "TVG_GridSplitAuto": TVG_GridSplitAuto,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TVG_LoadImageWithFilename": "TVG 加载图片(带文件名输出)",
    "TVG_GridSplitAuto": "TVG 九宫格智能切分(精度优先)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
