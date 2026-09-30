"""钉住 constants 注释里的数字，并检查其他模块没有重述这些约定。"""

import re
from pathlib import Path

from src import constants as C

SRC = Path(C.__file__).parent


def test_values_match_task_spec():
    assert (C.GRID_SIZE_E3B, C.GRID_SIZE_E1) == (32, 16)
    assert C.GRID_FG_AREA_THRESHOLD == 0.5
    assert (C.DEPTH_NUM_BINS, C.D_MIN, C.D_MAX) == (64, 0.5, 10.0)
    assert C.NYU_EVAL_HW == (480, 640)
    assert C.NYU_EIGEN_CROP == (45, 471, 41, 601)
    assert (C.NYU_DEPTH_MIN, C.NYU_DEPTH_MAX) == (1e-3, 10.0)
    assert C.BOUNDARY_IOU_DILATION_RATIO == 0.02
    assert (C.ADE_NUM_CLASSES, C.ADE_REDUCE_ZERO_LABEL, C.ADE_IGNORE_INDEX) == (150, True, 255)
    assert (C.BOOTSTRAP_N_RESAMPLES, C.BOOTSTRAP_CI_LEVEL) == (1000, 0.95)
    assert C.MAX_NEW_GRID_TOKENS == 300
    assert C.BOX_ACC_IOU_THRESHOLD == 0.5
    assert C.SIGNIFICANCE_STD_MULTIPLIER == 2.0


def test_eigen_crop_inside_image():
    r0, r1, c0, c1 = C.NYU_EIGEN_CROP
    h, w = C.NYU_EVAL_HW
    assert 0 <= r0 < r1 <= h and 0 <= c0 < c1 <= w


def test_conventions_not_restated_elsewhere():
    # 这些字面量只该出现在 constants.py
    forbidden = [r"\b471\b", r"\b601\b", r"\b0\.02\b", str(C.BOOTSTRAP_SEED), r"\b150\b", r"\b1e-3\b"]
    offenders = []
    for p in SRC.rglob("*.py"):
        if p.name == "constants.py":
            continue
        text = p.read_text()
        for pat in forbidden:
            if re.search(pat, text):
                offenders.append((p.name, pat))
    assert offenders == []
