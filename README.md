# 2026年6月29
# TVG 九宫格智能切分 ComfyUI 插件

把多宫格拼图（九宫格、不规则横竖混合布局等）精确切分成独立图片，
保证输出数量恰好等于用户指定的 `expect_rows * expect_cols`，并优先
使用精度最高的检测方法。每个格子按原始精确尺寸保存，**不做任何
resize 或 pad**，完整保留原始像素。

## 安装

把整个 `grid_split_node` 文件夹复制到 `ComfyUI/custom_nodes/` 目录下，
重启 ComfyUI。

依赖：`scipy`（其余 numpy/Pillow 通常 ComfyUI 自带）。如果启动后报缺少
scipy，在 ComfyUI 的 Python 环境中执行：

```bash
pip install scipy
```

## 包含的节点（统一以 TVG_ 前缀命名）

### 1. TVG 加载图片(带文件名输出) —— TVG_LoadImageWithFilename

和原生 Load Image 节点功能一致（输出 IMAGE / MASK），额外多输出一个
`filename` 字符串（不含扩展名，例如 `happy_4K.png` -> `happy_4K`）。

原生 Load Image 节点不会把文件名传给下游节点，导致后续保存文件时无法
得知原始文件名。本节点解决了这个问题。

### 2. TVG 九宫格智能切分(精度优先) —— TVG_GridSplitAuto

核心切分节点。

**输入：**
- `image`：图片（接 TVG_LoadImageWithFilename 或任意上游节点的 IMAGE 输出）
- `source_filename`：原始文件名（接 TVG_LoadImageWithFilename 的 filename
  输出，或手动填写），用于生成输出路径和文件名
- `expect_rows` / `expect_cols`：预期的行数/列数，**必须填写准确**，
  输出数量恒等于这两个数的乘积
- 其余为可选的高级调参，每个参数在节点界面上鼠标悬停即可看到详细
  中文说明（也可参考下方"参数说明"表格），默认值已能处理大多数情况

**输出：**
- `output_folder`：本次实际保存到的完整路径
- `method_used`：本次使用的方法（`connected_components` /
  `connected_components_adaptive` / `grid_fallback`）
- `saved_filenames`：保存的所有文件名（换行分隔的字符串）
- `cell_count`：实际保存数量（恒等于 expect_rows * expect_cols）

**保存位置：**

```
ComfyUI/output/<source_filename>/<时间戳>/
    <source_filename>_r0_c0.png
    <source_filename>_r0_c1.png
    ...
```

每次运行自动用时间戳建立新的子文件夹，不会覆盖之前的结果。每个格子
都是从原图直接精确裁切保存，**不经过任何缩放、填充或压缩处理**。

## 切分算法说明（三层级联，自动选择，无需手动干预）

1. **连通区域法**（精度最高）：把图像按"是否为白色背景"二值化，
   检测每一块独立的内容区域作为一个格子。不依赖任何网格假设，
   横版竖版混合、大小不一的不规则布局也能正确处理。
2. **自适应调参搜索**：如果步骤1检测到的格子数量不等于
   `expect_rows*expect_cols`，说明默认参数在这张图上失效了（可能是
   格子被误连通、或被误拆分）。此时根据数量偏差的方向，自动调整
   `bright_thresh`（背景亮度判定阈值）和 `border_strip`（图像最外圈
   忽略宽度，用于防止深色外边框把所有格子缝合连通）重新搜索，直到
   命中正确数量。
3. **网格法**（保底）：如果上述方法都未能命中，退化为"假设图像能
   等分成 rows x cols 网格"的方法，保证一定能输出指定数量的格子，
   但精度可能略低于前两种方法。

无论走哪一层，最后都会对每个格子做"逐边精修"：从粗略边界开始，
逐行逐列向内扫描，直到没有白边残留为止，确保最终结果干净、不带
白边、不切掉内容。

## 参数说明（TVG_GridSplitAuto 全部可选参数）

每个参数在 ComfyUI 节点界面上鼠标悬停即可看到完整的中文说明（包括
"该参数偏高/偏低时应该往哪个方向调整"的具体建议），不需要查文档。
以下是简要对照表：

| 参数 | 默认值 | 含义 |
|---|---|---|
| bright_thresh | 235 | 判定一个像素属于"白色背景/分割线"的亮度阈值（0-255，越高越严格）。检测数量偏少时调低；偏多时调高。 |
| std_thresh | 18 | 判定一行/一列像素属于"纯色背景"的颜色标准差阈值，越低越严格。主要在网格法和逐格精修阶段起作用。 |
| min_area_ratio | 0.01 | 连通区域法中，判定为"有效格子"所需的最小面积，占整图面积的比例。格子很小、数量很多时需要调低。 |
| min_gap | 3 | 网格法（保底方案）中，合并相邻分割带的最大间隔像素数。 |
| extra_margin | 1 | 逐格精修完成后，再额外向内收缩的像素数。边缘有残留白边就调大；边缘内容被切掉就调小。 |
| refine_max | 30 | 逐格精修阶段单条边最多允许向内收缩的像素数上限（安全阀），防止格子内部大片白色背景被误判为白边一直收缩。 |
| structure_tolerance_ratio | 0.06 | 连通区域法中，校验格子是否能排列成预期网格结构时的对齐容差比例。 |
| border_strip | 8 | 连通区域法中，图像最外圈强制视为背景的像素宽度，防止深色外边框把所有格子缝合连通。 |

### 常见调参场景

- **检测数量持续不足**（格子被误连通）：尝试增大 `border_strip`，
  或降低 `bright_thresh`
- **检测数量持续过多**（格子内部被误拆）：尝试提高 `bright_thresh`，
  或增大 `min_area_ratio` 过滤掉误拆出的小碎片
- **格子边缘仍有残留白边**：增大 `extra_margin`
- **格子边缘被切掉了一点内容**：减小 `extra_margin`，或检查
  `std_thresh` 是否过于宽松误判了内容边缘为背景

## 示例工作流

```
[TVG 加载图片(带文件名输出)] --image-------> [TVG 九宫格智能切分(精度优先)]
                            --filename---->  (接到 source_filename)
                                              expect_rows: 3
                                              expect_cols: 3
```

运行后直接到 `ComfyUI/output/<文件名>/<时间戳>/` 目录查看切分结果，
每张图都是原始精确像素，未经任何缩放或修改。
