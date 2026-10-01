"""E1 数据的约定（DEV-0012）。只在这里定义，落地脚本、加载器、点对生成与测试一律 import。

跨模块的已有约定（Eigen crop、NYU 有效深度区间、叠加渲染的颜色与 alpha）仍在 src/constants.py，这里只引用。
标"暂定"的数值以 E1 预注册为准。
"""

from __future__ import annotations

from src.constants import OVERLAY_PALETTE

# ---------------------------------------------------------------------------
# 抽样（开发子集、COCO 500、ODinW ≤50、点对的 A/B 分配）
# ---------------------------------------------------------------------------

# 一切抽样都用"哈希排序"：按 sha256(f"{SUBSET_SALT}|{用途}|{id}") 的十六进制串升序排，取前 k 个，再按 id 排序输出。
# 与 Python/numpy 版本和平台无关，逐字节可复现；k 变大时前 k 个保持不变（嵌套）。
# 改盐等于换一批样本：只能在预注册前改，改了要重建 configs/e1_dev_subsets.json 与 configs/e1_nyu_pairs.json。
SUBSET_SALT = "visint-e1-data-v1"

# 只有测试划分的基准：开发子集 = min(DEV_SUBSET_MAX, floor(n * DEV_SUBSET_FRAC_NUM / DEV_SUBSET_FRAC_DEN))。
# 依据：D11-6"每个 50 条或 10%，取较小者"；10% 向下取整（BLINK val 124 → 12，CountBench 491 → 49）。
DEV_SUBSET_MAX = 50
DEV_SUBSET_FRAC_NUM = 1
DEV_SUBSET_FRAC_DEN = 10

# 有训练划分的数据集：RefCOCO 系列各取 val 的 200 句（按句抽，每个数据集各 200）；NYUv2 取官方 train 的 50 张图。
# 依据：DEV-0012 任务书 §2。NYUv2 的张数任务书没给，取 50 与 DEV_SUBSET_MAX 一致（暂定）。
REFCOCO_DEV_SENTENCES = 200
NYU_DEV_IMAGES = 50

# COCO val2017 开放词汇检测子集：先剔除 D20 的探针用图，再在其余图像（含无标注的 48 张）里抽 500 张。依据：D11-5、D20。
COCO_OVD_SUBSET_SIZE = 500
# ODinW13：每个数据集在 GLIP 测试配置所用的划分里抽 ≤50 张（不足 50 张全取）。依据：DEV-0012 任务书。
ODINW_SUBSET_MAX = 50

# E1 评测子集（configs/e1_eval_subsets.json；DEV-0012 审查第 1 轮，D28-3、D28-10）。一律从剔除开发子集与 D20 之后的
# 评测总体里按哈希排序抽，每类子集一个独立的盐（同一总体上的不同类子集互相独立，允许重叠）。
EVAL_POLY_PER_SPLIT = 1000  # 多边形分割：RefCOCO/+ 的 unc val/testA/testB、RefCOCOg 的 umd val/test，每个划分
EVAL_GRID16_N = 500  # 16×16 数字网格：RefCOCO unc val
EVAL_THINK_N = 200  # 思考子集：每个任务（不足就全取）
EVAL_THINK8192_N = 50  # 8192 预算敏感性：从每个任务的思考子集里再抽
EVAL_EXTRA_REFCOCO_N = 500  # 附加对照（Qwen 约 1K token、InternVL 原生动态 tile）：RefCOCO unc val
EVAL_EXTRA_COCO_N = 100  # 附加对照：COCO 500 子集里的图
EVAL_SUBSET_SALTS = {
    "poly": "visint-e1-eval-poly-v1",
    "grid16": "visint-e1-eval-grid16-v1",
    "think": "visint-e1-eval-think-v1",
    "think8192": "visint-e1-eval-think8192-v1",
    "extra": "visint-e1-eval-extra-v1",
}

# ---------------------------------------------------------------------------
# D20：P0 探针用过的图像，从一切评测子集中排除
# ---------------------------------------------------------------------------

# 规范图像键见 src/data/records.py：COCO 的 2014 与 2017 共用 image id，键写成 coco:<id>。
PROBE_IMAGE_KEYS = ("coco:39769", "coco:139")
# 两张图的感知哈希（src/data/dedupe.py 的 phash，64 位十六进制）。开发机从官方 coco_url 下载后计算
# （000000039769.jpg 173,131 B、000000000139.jpg 161,811 B）；tests/test_data_dedupe.py 用 fixture 钉住算法。
PROBE_IMAGE_PHASH = {
    "coco:39769": "f030474e4a5b59f9",
    "coco:139": "95d3f8748359f805",
}

# ---------------------------------------------------------------------------
# 感知哈希（D11-7：按 image_id 与感知哈希去重）
# ---------------------------------------------------------------------------

# pHash：灰度（PIL "L"）→ LANCZOS 缩到 (PHASH_SIZE·PHASH_HIGHFREQ)² → 二维 DCT-II → 取左上 PHASH_SIZE² 个系数 →
# 与其中位数比较得 64 位。与 imagehash.phash 的默认定义相同（hash_size=8, highfreq_factor=4）。
PHASH_SIZE = 8
PHASH_HIGHFREQ = 4
# 汉明距离 ≤ 此值判为近重复（暂定；imagehash 文档常用 ≤ 8–10 作近重复阈值）。
PHASH_MAX_HAMMING = 8

# ---------------------------------------------------------------------------
# NYUv2 相对深度点对（E1 的 T 文本任务；DEV-0012 任务书 §3，默认值以 E1 预注册为准）
# ---------------------------------------------------------------------------

NYU_PAIRS_PER_IMAGE = 5
# 两点像素距离 ≥ 短边 × NUM / DEN（480 × 1/10 = 48 px）。比较用整数：DEN²·d² ≥ NUM²·短边²。
NYU_PAIR_MIN_DIST_NUM = 1
NYU_PAIR_MIN_DIST_DEN = 10
# 深度相对差 ≥ 15%：max(dA, dB) ≥ (1 + 0.15) · min(dA, dB)。
NYU_PAIR_MIN_REL_DEPTH_DIFF = 0.15
# 每张图最多尝试多少次随机点对，凑不满 NYU_PAIRS_PER_IMAGE 就记缺口（不放宽约束）。
NYU_PAIR_MAX_ATTEMPTS = 20000
# 点所用的深度：rawDepths（未补洞，与 BTS 的 Eigen 测试 GT 同源，见 data_sources B §1.5）。"深度有效" =
# src/constants.py 的 (NYU_DEPTH_MIN, NYU_DEPTH_MAX) 开区间。
NYU_PAIR_DEPTH_KEY = "depth_raw"
# 深度边缘保护（遮挡边界），"相邻像素跳变"判据（DEV-0012 审查第 1 轮打开，第 2 轮改判据）：
# - 以点为中心的 (2w+1)² 窗口内深度全部有效；
# - 窗口内任意 4 邻接的两个像素，深度 max/min ≤ 1 + NYU_PAIR_NEIGHBOR_MAX_REL。
# 只排除遮挡边界（标记环会跨两个表面，答案含糊）。不看整个窗口的 max/min，平滑的斜面（掠射角的地面、纵深的墙）不剔。
# w 取标记环的外半径 NYU_MARKER_OUTER_RADIUS（9，tests 钉住两者相等）；0 表示关闭。
NYU_PAIR_LOCAL_WINDOW = 9
NYU_PAIR_NEIGHBOR_MAX_REL = 0.05

# 点标记：环形（外半径、内半径，像素，环宽 = 外 − 内），用 src/executors/overlay.render_overlay 叠加，
# 两个点与两个字母合成一个 mask，颜色 = OVERLAY_PALETTE[0]（同色，只靠字母区分 A、B，避免颜色线索）。
# render_overlay 的填充 alpha 与 2 px 实色轮廓让 4 px 宽的环与 3 px 宽的笔画都成为实色。数值暂定。
NYU_MARKER_COLOR_INDEX = 0
NYU_MARKER_COLOR = OVERLAY_PALETTE[NYU_MARKER_COLOR_INDEX]
NYU_MARKER_OUTER_RADIUS = 9
NYU_MARKER_INNER_RADIUS = 5
# 标签：5×7 点阵字形放大 NYU_LABEL_SCALE 倍（15×21 px），放在环的斜上/斜下方，与环相隔 NYU_LABEL_GAP px。
# 候选位置的顺序：右上、左上、右下、左下；取第一个完全在图内、且不压到另一个点的环与标签的位置。
NYU_LABEL_SCALE = 3
NYU_LABEL_GAP = 2
NYU_LABEL_GLYPHS = {
    "A": (
        ".###.",
        "#...#",
        "#...#",
        "#####",
        "#...#",
        "#...#",
        "#...#",
    ),
    "B": (
        "####.",
        "#...#",
        "#...#",
        "####.",
        "#...#",
        "#...#",
        "####.",
    ),
}
