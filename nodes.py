"""
TVG ComfyUI 节点定义
=========================================

公司前缀：TVG_

包含两个节点：

1. TVG_LoadImageWithFilename
   原生 LoadImage 节点的增强版。原生节点只输出 IMAGE/MASK，不附带
   文件名信息，导致下游节点无法知道"这张图原来叫什么"。本节点在
   保留原有功能的基础上，额外输出一个不含扩展名的文件名字符串
   （例如 "happy_4K.png" -> "happy_4K"），供下游节点用于生成
   有意义的输出文件名。

2. TVG_GridSplitAuto
   九宫格/网格图像智能切分节点。核心算法见 grid_split_core.py，
   三层级联：连通区域法 -> 自适应调参 -> 网格法保底，保证一定能
   切出 rows*cols 个格子，且优先采用精度最高的方法。

   切分结果直接按原始精确尺寸保存到磁盘（不做任何 resize/pad，
   完整保留原始像素；也不经过 ComfyUI 的 batch tensor，因为各
   格子尺寸可能不同，无法无损地堆叠成统一 batch），保存路径为：
       ComfyUI/output/<source_filename>/<时间戳>/<source_filename>_r{row}_c{col}.png
"""

import os
import datetime
import numpy as np
import torch
from PIL import Image

import folder_paths

from .grid_split_core import split_grid_image


# ----------------------------------------------------------------------
# 节点一：TVG_LoadImageWithFilename
# ----------------------------------------------------------------------

class TVG_LoadImageWithFilename:
    """
    加载图片，同时输出不含扩展名的文件名，供下游节点（如
    TVG_GridSplitAuto）用于生成有意义的输出路径/文件名。
    """

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [
            f for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ]
        return {
            "required": {
                "image": (sorted(files), {
                    "image_upload": True,
                    "tooltip": "要加载的图片文件（从 ComfyUI 的 input 目录中选择，或点击上传新文件）。",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "filename")
    OUTPUT_TOOLTIPS = (
        "加载的图片，格式为 ComfyUI 标准 IMAGE 类型。",
        "图片的 alpha 通道遮罩；若原图没有 alpha 通道，则输出全 1（完全不透明）的占位遮罩。",
        "不含扩展名的原始文件名（例如 happy_4K.png -> happy_4K），用于下游节点生成有意义的输出文件名。",
    )
    FUNCTION = "load_image"
    CATEGORY = "TVG/image"
    DESCRIPTION = "加载图片，并额外输出不含扩展名的原始文件名，弥补原生 Load Image 节点不传递文件名信息的缺陷。"

    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)

        img = Image.open(image_path)
        img = Image.merge('RGB', img.convert('RGB').split())  # 规范化为RGB（去除可能的调色板等特殊模式）

        image_np = np.array(img).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,]  # [1, H, W, 3]

        # mask：如果原图有alpha通道则用alpha，否则给一个全1的占位mask
        pil_img_full = Image.open(image_path)
        if 'A' in pil_img_full.getbands():
            mask_np = np.array(pil_img_full.getchannel('A')).astype(np.float32) / 255.0
            mask_tensor = torch.from_numpy(mask_np)[None,]
        else:
            mask_tensor = torch.ones((1, image_np.shape[0], image_np.shape[1]), dtype=torch.float32)

        # 文件名（不含扩展名），用于下游命名
        base_name = os.path.basename(image_path)
        filename_no_ext = os.path.splitext(base_name)[0]

        return (image_tensor, mask_tensor, filename_no_ext)

    @classmethod
    def IS_CHANGED(cls, image):
        image_path = folder_paths.get_annotated_filepath(image)
        import hashlib
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        if not folder_paths.exists_annotated_filepath(image):
            return f"无法找到文件: {image}"
        return True


# ----------------------------------------------------------------------
# 节点二：TVG_GridSplitAuto
# ----------------------------------------------------------------------

class TVG_GridSplitAuto:
    """
    九宫格/网格图像智能切分节点。

    保证输出恰好 expect_rows * expect_cols 张图，优先使用精度最高的
    连通区域检测法，仅在检测失败时才依次退化到自适应调参、网格等分法。

    每个格子按原始精确尺寸直接保存为 PNG 文件（不做统一 resize，
    不经过 ComfyUI batch tensor），保存路径：
        ComfyUI/output/<source_filename>/<时间戳>/
            <source_filename>_r{row}_c{col}.png
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "待切分的拼图（九宫格或不规则横竖混合布局），格子之间需要用接近白色的背景/分割线隔开。",
                }),
                "source_filename": ("STRING", {
                    "default": "image",
                    "tooltip": "原始文件名（不含扩展名），用于生成输出文件夹名和每个格子的文件名。建议接 TVG_LoadImageWithFilename 的 filename 输出自动获取，也可以手动填写。",
                }),
                "expect_rows": ("INT", {
                    "default": 3, "min": 1, "max": 50,
                    "tooltip": "预期的行数。必须填写准确——本节点会保证最终输出恰好 expect_rows * expect_cols 张图，但如果这个数字本身和图片实际布局不符，输出结果会没有意义（哪怕数量是对的）。",
                }),
                "expect_cols": ("INT", {
                    "default": 3, "min": 1, "max": 50,
                    "tooltip": "预期的列数。同 expect_rows，必须填写准确。",
                }),
            },
            "optional": {
                "bright_thresh": ("FLOAT", {
                    "default": 235.0, "min": 0.0, "max": 255.0, "step": 1.0,
                    "tooltip": "判定一个像素属于「白色背景/分割线」的亮度阈值（0-255，越高越严格）。\n"
                               "如果检测到的格子数量持续偏少（格子被误判连通成一块）：尝试调低这个值，让算法更容易识别出变窄或偏暗的分割线。\n"
                               "如果检测到的格子数量持续偏多（一个格子被误拆成几块，常见原因是格子内部有大片白色背景，如白盘子、白衬衫）：尝试调高这个值，让判定标准更严格。\n"
                               "默认 235 适合分割线接近纯白的常见情况。",
                }),
                "std_thresh": ("FLOAT", {
                    "default": 18.0, "min": 0.0, "max": 100.0, "step": 0.5,
                    "tooltip": "判定一行/一列像素属于「纯色背景」的颜色标准差阈值（越低越严格）。\n"
                               "分割线应该是颜色非常均匀的一条线，标准差很低；如果内容区域的颜色变化也被误判成背景，可以调低这个值收紧标准。\n"
                               "主要在网格法（保底方案）和逐格精修阶段起作用。默认 18 通常无需调整。",
                }),
                "min_area_ratio": ("FLOAT", {
                    "default": 0.01, "min": 0.0001, "max": 0.5, "step": 0.001,
                    "tooltip": "连通区域法中，判定为「有效格子」所需的最小面积，以占整图面积的比例表示（默认 0.01 即 1%）。\n"
                               "面积小于这个比例的连通区域会被当作噪声/误检碎片丢弃，不计入格子数量。\n"
                               "如果你的格子本身很小（比如切分出的格子数量很多、每个格子占比很小），需要调低这个值，否则真实的小格子会被误删。",
                }),
                "min_gap": ("INT", {
                    "default": 3, "min": 0, "max": 100,
                    "tooltip": "网格法（保底方案）中，合并相邻分割带的最大间隔像素数。\n"
                               "如果一条分割线因为噪声被误判成两段不连续的白色带，且两段之间的间隔不超过这个值，会被自动合并成一条完整的分割线。\n"
                               "默认 3 通常无需调整，仅在网格法生效时才会用到这个参数。",
                }),
                "extra_margin": ("INT", {
                    "default": 1, "min": 0, "max": 50,
                    "tooltip": "逐格精修完成后，再额外向内收缩的像素数。\n"
                               "用于消除抗锯齿造成的浅灰色过渡像素残留。如果切出来的图边缘仍能看到一丝白边，调大这个值；如果发现边缘内容被切掉了一点，调小这个值（甚至设为 0）。",
                }),
                "refine_max": ("INT", {
                    "default": 30, "min": 0, "max": 200,
                    "tooltip": "逐格精修阶段，单条边最多允许向内收缩的像素数上限（安全阀）。\n"
                               "精修逻辑会从格子边缘开始逐行/逐列检测白边并向内收缩，直到遇到真实内容为止。如果格子内部本身有大片纯白背景（比如天空、白墙），可能被误判成「白边」一直收缩下去；这个上限保证即使误判，最多损失这么多像素，不会切掉过多有效内容。\n"
                               "如果你的素材常有大片白色背景内容，建议调小这个值（比如 10）更安全。",
                }),
                "structure_tolerance_ratio": ("FLOAT", {
                    "default": 0.06, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "连通区域法中，校验检测到的格子是否能排列成 expect_rows x expect_cols 网格结构时的对齐容差，以占图像尺寸的比例表示（默认 0.06 即 6%）。\n"
                               "数值越大，对「格子之间不完全对齐」的容忍度越高（更宽松地接受结果）；数值越小，要求格子排列得越整齐才会采用连通区域法的结果，否则会触发自适应调参或退化到网格法。",
                }),
                "border_strip": ("INT", {
                    "default": 8, "min": 0, "max": 200,
                    "tooltip": "连通区域法中，图像最外圈强制视为背景的像素宽度。\n"
                               "用途：防止图片自带的深色/黑色装饰性外边框，把原本被白色分割线隔开的所有格子，从图像最外围「缝合」成一个连通区域（导致检测数量远小于预期，比如检测到 1 个而不是 9 个）。\n"
                               "如果你的图片外围没有深色边框，可以设为 0；如果外边框比较粗，适当调大这个值。",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("output_folder", "method_used", "saved_filenames", "cell_count")
    OUTPUT_TOOLTIPS = (
        "本次切分结果实际保存到的完整文件夹路径（ComfyUI/output/<文件名>/<时间戳>/）。可以直接接到 TVG_LoadImagesFromFolder 的 folder_path 输入，把切出来的格子重新读回 IMAGE。",
        "本次实际使用的切分方法：connected_components（连通区域法，精度最高）/ connected_components_adaptive（连通区域法+自适应调参）/ grid_fallback（网格等分法，保底方案）。",
        "本次保存的所有文件名，换行分隔的字符串。",
        "本次实际保存的图片数量，恒等于 expect_rows * expect_cols。",
    )
    FUNCTION = "split"
    CATEGORY = "TVG/grid_split"
    OUTPUT_NODE = True
    DESCRIPTION = ("九宫格/不规则网格图像智能切分。保证输出数量精确等于 expect_rows*expect_cols，"
                   "优先使用精度最高的连通区域检测法，仅在失败时依次退化到自适应调参、网格等分法。"
                   "每个格子按原始精确尺寸保存为独立 PNG 文件，不做统一缩放。")

    def split(self, image, source_filename, expect_rows, expect_cols,
              bright_thresh=235.0, std_thresh=18.0,
              min_area_ratio=0.01, min_gap=3,
              extra_margin=1, refine_max=30,
              structure_tolerance_ratio=0.06,
              border_strip=8):

        log_lines = []
        def log_fn(msg):
            print(msg)
            log_lines.append(msg)

        # ComfyUI IMAGE tensor: [batch, H, W, C], float 0-1
        # 只处理 batch 中的第一张（九宫格切分场景下通常 batch=1）
        img_tensor = image[0]
        arr = (img_tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]  # 丢弃alpha通道，切分逻辑只需要RGB

        cells, method_used = split_grid_image(
            arr, expect_rows, expect_cols,
            bright_thresh=bright_thresh,
            std_thresh=std_thresh,
            min_area_ratio=min_area_ratio,
            min_gap=min_gap,
            extra_margin=extra_margin,
            refine_max=refine_max,
            structure_tolerance_ratio=structure_tolerance_ratio,
            border_strip=border_strip,
            log_fn=log_fn,
        )

        # 安全处理文件名：去除可能导致路径问题的字符
        safe_name = "".join(
            c for c in source_filename if c.isalnum() or c in ("_", "-")
        ) or "image"

        output_root = folder_paths.get_output_directory()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(output_root, safe_name, timestamp)
        os.makedirs(save_dir, exist_ok=True)

        saved_names = []
        for (r, c, cell_arr) in cells:
            filename = f"{safe_name}_r{r}_c{c}.png"
            save_path = os.path.join(save_dir, filename)
            Image.fromarray(cell_arr).save(save_path)
            saved_names.append(filename)
            log_fn(f"  已保存: {save_path}")

        saved_filenames_str = "\n".join(saved_names)
        cell_count = len(cells)

        log_fn(f"完成。本次使用方法: {method_used}，共保存 {cell_count} 张图到 {save_dir}")

        return (save_dir, method_used, saved_filenames_str, cell_count)

