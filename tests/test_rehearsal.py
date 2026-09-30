"""E1 数字字符网格路径的端到端彩排（DEV-0001 验收）：

GT 多边形 → mask → 16×16 网格 → 数字文本（包在模型式输出里）→ 解析回网格 → 回填 mask → cIoU/gIoU/Boundary IoU。
直接运行 `python -m tests.test_rehearsal` 打印数值。
"""

import numpy as np

from src.constants import GRID_SIZE_E1
from src.executors.raster import mask_to_rle, polygons_to_mask, rle_to_mask
from src.grid.codec import grid_to_mask, grid_to_text, mask_to_grid, parse_grid_text
from src.metrics.seg import boundary_iou, ciou, giou, mask_inter_union

H, W = 90, 140  # H ≠ W
# 不对称的五边形（绝对像素 xyxy 约定，x 向右 y 向下）
GT_POLYGON = [20.0, 15.0, 110.0, 25.0, 125.0, 70.0, 60.0, 80.0, 30.0, 55.0]


def rehearse() -> dict:
    gt = polygons_to_mask([GT_POLYGON], H, W)
    assert np.array_equal(rle_to_mask(mask_to_rle(gt)), gt)
    grid = mask_to_grid(gt, GRID_SIZE_E1)
    model_output = "The object occupies these cells:\n```\n" + grid_to_text(grid) + "\n```"
    parsed = parse_grid_text(model_output, GRID_SIZE_E1)
    assert parsed.failure is None and np.array_equal(parsed.grid, grid)
    pred = grid_to_mask(parsed.grid, H, W)
    stats = [mask_inter_union(pred, gt)]
    return {
        "H": H,
        "W": W,
        "gt_pixels": int(gt.sum()),
        "grid_ones": int(grid.sum()),
        "pred_pixels": int(pred.sum()),
        "inter_union": stats[0],
        "cIoU": ciou(stats),
        "gIoU": giou(stats),
        "boundary_IoU": boundary_iou(pred, gt),
    }


def test_rehearsal_e1_grid_path():
    r = rehearse()
    assert r["gt_pixels"] > 0 and r["grid_ones"] > 0
    # 16×16 量化是唯一的误差来源：IoU 高但不为 1；Boundary IoU 对这种量化更敏感
    assert 0.85 < r["cIoU"] < 1.0
    assert r["cIoU"] == r["gIoU"]  # 单样本时二者相等
    assert 0.0 < r["boundary_IoU"] < r["cIoU"]
    # 网格化本身没转置：原 mask 的转置不应得到同样的网格
    assert not np.array_equal(mask_to_grid(polygons_to_mask([GT_POLYGON], H, W), GRID_SIZE_E1).T, mask_to_grid(polygons_to_mask([GT_POLYGON], H, W), GRID_SIZE_E1))


if __name__ == "__main__":
    for k, v in rehearse().items():
        print(f"{k}: {v}")
