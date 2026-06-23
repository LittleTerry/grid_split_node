"""
核心切分算法模块（从命令行版 split_auto.py 移植）
=========================================

与命令行版的区别：
    命令行版直接读写文件路径。这里改为接受/返回 numpy array（uint8,
    HxWx3），方便上层节点直接喂入从 ComfyUI tensor 转换来的图像数据，
    不需要先写临时文件再读回来。

算法逻辑保持不变，三层级联：
    1. 连通区域法（默认参数）-- 精度最高
    2. 连通区域法 + 自适应调参搜索 -- 默认参数失败时，根据数量偏差
       方向自动调整 bright_thresh / border_strip 重新搜索
    3. 网格法（等分网格假设）-- 上面两层都失败时的保底方案，
       保证一定能切出 rows*cols 个格子
"""

import numpy as np
from scipy import ndimage


# ----------------------------------------------------------------------
# 公共工具：精修单个格子的四条边
# ----------------------------------------------------------------------

def is_line_white(line_pixels, bright_thresh, std_thresh):
    if line_pixels.size == 0:
        return False
    return line_pixels.mean() > bright_thresh and line_pixels.std() < std_thresh


def refine_single_cell(cell_arr, bright_thresh, std_thresh, refine_max, extra_margin):
    h, w = cell_arr.shape[0], cell_arr.shape[1]

    top = 0
    while top < min(refine_max, h - 1):
        if is_line_white(cell_arr[top, :, :], bright_thresh, std_thresh):
            top += 1
        else:
            break

    bottom = h
    steps = 0
    while steps < min(refine_max, h - 1):
        if is_line_white(cell_arr[bottom - 1, :, :], bright_thresh, std_thresh):
            bottom -= 1
            steps += 1
        else:
            break

    left = 0
    while left < min(refine_max, w - 1):
        if is_line_white(cell_arr[:, left, :], bright_thresh, std_thresh):
            left += 1
        else:
            break

    right = w
    steps = 0
    while steps < min(refine_max, w - 1):
        if is_line_white(cell_arr[:, right - 1, :], bright_thresh, std_thresh):
            right -= 1
            steps += 1
        else:
            break

    left += extra_margin
    top += extra_margin
    right -= extra_margin
    bottom -= extra_margin

    if right <= left or bottom <= top:
        return (0, 0, w, h)

    return (left, top, right, bottom)


# ----------------------------------------------------------------------
# 方法一：连通区域法
# ----------------------------------------------------------------------

def binarize_non_white(arr, bright_thresh, border_strip=0):
    gray = arr.mean(axis=2)
    mask = gray <= bright_thresh
    if border_strip > 0:
        mask[:border_strip, :] = False
        mask[-border_strip:, :] = False
        mask[:, :border_strip] = False
        mask[:, -border_strip:] = False
    return mask


def find_connected_regions(content_mask, min_area):
    structure = np.ones((3, 3), dtype=int)
    labeled, num_features = ndimage.label(content_mask, structure=structure)

    boxes = []
    for label_id in range(1, num_features + 1):
        coords = np.where(labeled == label_id)
        area = len(coords[0])
        if area < min_area:
            continue
        y0, y1 = coords[0].min(), coords[0].max() + 1
        x0, x1 = coords[1].min(), coords[1].max() + 1
        boxes.append((y0, y1, x0, x1))

    return boxes


def merge_overlapping_boxes(boxes, iou_thresh=0.3):
    def box_iou_or_contains(a, b):
        ay0, ay1, ax0, ax1 = a
        by0, by1, bx0, bx1 = b
        iy0, iy1 = max(ay0, by0), min(ay1, by1)
        ix0, ix1 = max(ax0, bx0), min(ax1, bx1)
        if iy1 <= iy0 or ix1 <= ix0:
            return False
        inter = (iy1 - iy0) * (ix1 - ix0)
        area_a = (ay1 - ay0) * (ax1 - ax0)
        area_b = (by1 - by0) * (bx1 - bx0)
        if inter / area_a > 0.9 or inter / area_b > 0.9:
            return True
        union = area_a + area_b - inter
        return inter / union > iou_thresh

    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                if box_iou_or_contains(merged[i], merged[j]):
                    ay0, ay1, ax0, ax1 = merged[i]
                    by0, by1, bx0, bx1 = merged[j]
                    new_box = (min(ay0, by0), max(ay1, by1), min(ax0, bx0), max(ax1, bx1))
                    merged[i] = new_box
                    del merged[j]
                    changed = True
                    break
            if changed:
                break
    return merged


def try_assign_grid_structure(boxes, rows, cols, h, w, tolerance_ratio=0.06):
    if len(boxes) != rows * cols:
        return None

    row_tol = tolerance_ratio * h
    col_tol = tolerance_ratio * w

    boxes_sorted = sorted(boxes, key=lambda b: b[0])
    row_groups = []
    for box in boxes_sorted:
        placed = False
        for group in row_groups:
            if abs(box[0] - group[0][0]) <= row_tol:
                group.append(box)
                placed = True
                break
        if not placed:
            row_groups.append([box])

    if len(row_groups) != rows:
        return None
    for group in row_groups:
        if len(group) != cols:
            return None

    row_groups.sort(key=lambda g: np.mean([b[0] for b in g]))

    for i in range(len(row_groups)):
        row_groups[i] = sorted(row_groups[i], key=lambda b: b[2])

    for c in range(cols):
        lefts = [row_groups[r][c][2] for r in range(rows)]
        if max(lefts) - min(lefts) > col_tol:
            return None

    grid = [[row_groups[r][c] for c in range(cols)] for r in range(rows)]
    return grid


def adaptive_connected_components_search(arr, rows, cols, h, w,
                                           bright_thresh, std_thresh,
                                           min_area_ratio, border_strip,
                                           structure_tolerance_ratio,
                                           bright_search_range=40,
                                           bright_step=5,
                                           border_search_max=40,
                                           border_step=4,
                                           log_fn=print):
    expected_count = rows * cols
    min_area = min_area_ratio * h * w

    def try_one(bt, bs):
        content_mask = binarize_non_white(arr, bt, border_strip=bs)
        boxes = find_connected_regions(content_mask, min_area)
        boxes = merge_overlapping_boxes(boxes)
        return boxes

    boxes = try_one(bright_thresh, border_strip)
    count = len(boxes)

    if count == expected_count:
        grid = try_assign_grid_structure(boxes, rows, cols, h, w, structure_tolerance_ratio)
        if grid is not None:
            return boxes, bright_thresh, border_strip

    attempts_log = [(bright_thresh, border_strip, count)]

    if count < expected_count:
        for bs in range(border_strip, border_search_max + 1, border_step):
            for bt in range(int(bright_thresh), int(bright_thresh) - bright_search_range - 1, -bright_step):
                if bt <= 0:
                    break
                boxes = try_one(bt, bs)
                count = len(boxes)
                attempts_log.append((bt, bs, count))
                if count == expected_count:
                    grid = try_assign_grid_structure(boxes, rows, cols, h, w, structure_tolerance_ratio)
                    if grid is not None:
                        log_fn(f"  [自适应搜索] 命中：bright_thresh={bt}, border_strip={bs}")
                        return boxes, bt, bs
    else:
        for bt in range(int(bright_thresh), int(bright_thresh) + bright_search_range + 1, bright_step):
            if bt >= 255:
                break
            boxes = try_one(bt, border_strip)
            count = len(boxes)
            attempts_log.append((bt, border_strip, count))
            if count == expected_count:
                grid = try_assign_grid_structure(boxes, rows, cols, h, w, structure_tolerance_ratio)
                if grid is not None:
                    log_fn(f"  [自适应搜索] 命中：bright_thresh={bt}, border_strip={border_strip}")
                    return boxes, bt, border_strip

    log_fn(f"  [自适应搜索] 搜索范围内未能命中。已尝试 {len(attempts_log)} 组参数，"
           f"检测数量分布: {sorted(set(c for _, _, c in attempts_log))}")
    return None, None, None


# ----------------------------------------------------------------------
# 方法二：网格法（保底）
# ----------------------------------------------------------------------

def detect_white_bands_1d(brightness, std, bright_thresh, std_thresh):
    is_white = (brightness > bright_thresh) & (std < std_thresh)
    bands = []
    start = None
    for i, flag in enumerate(is_white):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            bands.append((start, i - 1))
            start = None
    if start is not None:
        bands.append((start, len(is_white) - 1))
    return bands


def merge_close_bands(bands, min_gap):
    if not bands:
        return bands
    merged = [bands[0]]
    for b in bands[1:]:
        prev_start, prev_end = merged[-1]
        cur_start, cur_end = b
        if cur_start - prev_end <= min_gap:
            merged[-1] = (prev_start, cur_end)
        else:
            merged.append(b)
    return merged


def bands_to_lines(bands, axis_length):
    lines = [0]
    for (s, e) in bands:
        center = (s + e) // 2
        if 2 < center < axis_length - 2:
            lines.append(center)
    lines.append(axis_length)
    return sorted(set(lines))


def pick_n_plus_1_lines(lines, n_cells, axis_length):
    target_count = n_cells + 1
    if len(lines) == target_count:
        return lines

    ideal = [round(i * axis_length / n_cells) for i in range(target_count)]

    if len(lines) > target_count:
        first, last = lines[0], lines[-1]
        middle_lines = lines[1:-1]
        middle_ideal = ideal[1:-1]
        chosen = []
        used = set()
        for ideal_pos in middle_ideal:
            best = min(
                (l for idx, l in enumerate(middle_lines) if idx not in used),
                key=lambda l: abs(l - ideal_pos),
                default=None,
            )
            if best is not None:
                used.add(middle_lines.index(best))
                chosen.append(best)
        return sorted(set([first] + chosen + [last]))

    result = list(lines)
    for ideal_pos in ideal:
        if not any(abs(ideal_pos - l) < axis_length / n_cells / 2 for l in result):
            result.append(ideal_pos)
    return sorted(set(result))


def compute_cell_boundaries(lines, n_cells, margin):
    boundaries = []
    for i in range(n_cells):
        start = lines[i]
        end = lines[i + 1]
        s = start if i == 0 else start + margin
        e = end if i == n_cells - 1 else end - margin
        if e <= s:
            e = end
            s = start
        boundaries.append((s, e))
    return boundaries


def grid_method_split(arr, rows, cols, bright_thresh, std_thresh, min_gap, coarse_margin=0):
    h, w = arr.shape[0], arr.shape[1]

    row_brightness = arr.mean(axis=(1, 2))
    row_std = arr.std(axis=(1, 2))
    row_bands = merge_close_bands(detect_white_bands_1d(row_brightness, row_std, bright_thresh, std_thresh), min_gap)
    row_lines = pick_n_plus_1_lines(bands_to_lines(row_bands, h), rows, h)

    col_brightness = arr.mean(axis=(0, 2))
    col_std = arr.std(axis=(0, 2))
    col_bands = merge_close_bands(detect_white_bands_1d(col_brightness, col_std, bright_thresh, std_thresh), min_gap)
    col_lines = pick_n_plus_1_lines(bands_to_lines(col_bands, w), cols, w)

    row_bounds = compute_cell_boundaries(row_lines, rows, coarse_margin)
    col_bounds = compute_cell_boundaries(col_lines, cols, coarse_margin)

    grid = [[None] * cols for _ in range(rows)]
    for r in range(rows):
        y0, y1 = row_bounds[r]
        for c in range(cols):
            x0, x1 = col_bounds[c]
            grid[r][c] = (y0, y1, x0, x1)
    return grid


# ----------------------------------------------------------------------
# 主入口：三层级联决策
# ----------------------------------------------------------------------

def split_grid_image(arr, rows, cols,
                      bright_thresh=235, std_thresh=18,
                      min_area_ratio=0.01, min_gap=3,
                      extra_margin=1, refine_max=30,
                      structure_tolerance_ratio=0.06,
                      border_strip=8,
                      log_fn=print):
    """
    主入口函数。

    输入：
        arr: numpy array, shape (H, W, 3), dtype uint8
        rows, cols: 预期的行列数，保证最终输出恰好 rows*cols 个格子

    返回：
        cells: 长度为 rows*cols 的列表，每个元素是 (r, c, cropped_arr)
               cropped_arr 是该格子精修后的 numpy array (h, w, 3) uint8，
               各格子尺寸可能不同（保留原始精确裁切尺寸，不做统一resize）
        method_used: 字符串，说明本次实际使用的方法
    """
    h, w = arr.shape[0], arr.shape[1]
    expected_count = rows * cols
    grid = None
    method_used = None

    content_mask = binarize_non_white(arr, bright_thresh, border_strip=border_strip)
    min_area = min_area_ratio * h * w
    boxes = find_connected_regions(content_mask, min_area)
    boxes = merge_overlapping_boxes(boxes)

    log_fn(f"[方法一: 连通区域法] 检测到 {len(boxes)} 个候选格子（期望 {expected_count} 个）")

    if len(boxes) == expected_count:
        candidate_grid = try_assign_grid_structure(boxes, rows, cols, h, w, structure_tolerance_ratio)
        if candidate_grid is not None:
            grid = candidate_grid
            method_used = "connected_components"
            log_fn("[方法一: 连通区域法] 数量正确且结构校验通过 -> 采用此方法（精度最高）")
        else:
            log_fn(f"[方法一: 连通区域法] 数量正确，但几何结构校验失败 -> 尝试自适应调参")
    else:
        log_fn("[方法一: 连通区域法] 数量不匹配 -> 尝试自适应调参")

    if grid is None:
        log_fn(f"[自适应搜索] 已知 rows*cols={expected_count} 一定正确，"
               f"根据偏差方向动态调整 bright_thresh / border_strip 重新搜索...")
        adj_boxes, used_bt, used_bs = adaptive_connected_components_search(
            arr, rows, cols, h, w,
            bright_thresh, std_thresh,
            min_area_ratio, border_strip,
            structure_tolerance_ratio,
            log_fn=log_fn,
        )
        if adj_boxes is not None:
            grid = try_assign_grid_structure(adj_boxes, rows, cols, h, w, structure_tolerance_ratio)
            method_used = "connected_components_adaptive"
            log_fn(f"[方法一: 连通区域法-自适应] 成功 -> 采用此方法"
                   f"（最终参数 bright_thresh={used_bt}, border_strip={used_bs}）")
        else:
            log_fn("[自适应搜索] 搜索范围内仍未能命中 -> 转用网格法")

    if grid is None:
        grid = grid_method_split(arr, rows, cols, bright_thresh, std_thresh, min_gap, coarse_margin=0)
        method_used = "grid_fallback"
        log_fn(f"[方法二: 网格法] 已使用等分网格假设强制切出 {rows}x{cols} = {expected_count} 个格子")

    cells = []
    for r in range(rows):
        for c in range(cols):
            y0, y1, x0, x1 = grid[r][c]
            coarse_crop = arr[y0:y1, x0:x1, :]
            left, top, right, bottom = refine_single_cell(
                coarse_crop, bright_thresh, std_thresh, refine_max, extra_margin
            )
            final_crop = coarse_crop[top:bottom, left:right, :]
            cells.append((r, c, final_crop))
            log_fn(f"  格子 [{r},{c}] 最终尺寸 {final_crop.shape[1]}x{final_crop.shape[0]}")

    assert len(cells) == expected_count, "内部错误：输出数量与预期不符"
    return cells, method_used
