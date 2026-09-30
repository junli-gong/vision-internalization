"""跨模块约定的唯一来源（D4）。

其他模块和测试一律从这里 import，禁止在别处重述这些数字。
每条约定的注释写明依据；标"暂定"的在对应预注册前定稿。
"""

# ---------------------------------------------------------------------------
# 坐标
# ---------------------------------------------------------------------------

# 内部统一用绝对像素、浮点、xyxy；原点在左上角，x 向右，y 向下。
# 依据：DEV-0001 任务书 §2；简报 §12"各家族坐标约定不同，内部统一换算"。
BOX_FORMAT = "xyxy_abs_float"

# 像素 (i, j)（第 i 行、第 j 列）占据连续区间 [j, j+1) × [i, i+1)，中心在 (j+0.5, i+0.5)。
# 依据：DEV-0001 任务书 §2。
PIXEL_CENTER_OFFSET = 0.5

# 框光栅化：像素中心落在框内即为 1。中心恰好落在框边上时取 (lo, hi]，
# 即 x0 < cx <= x1 且 y0 < cy <= y1。
# 依据：任务书只规定"中心落在框内"，未规定边界归属；(lo, hi] 与 pycocotools 对矩形多边形的
# 光栅化逐位一致（tests/test_raster.py 钉住），[lo, hi) 在半整数坐标上与之不一致。
# 坐标为整数时两种取法结果相同。待管理员确认。
BOX_RASTER_RULE = "pixel_center_in_(lo,hi]"

# 多边形光栅化一律走 pycocotools，与 COCO/RefCOCO 的 GT 同一套规则。
# 依据：DEV-0001 任务书 §2。
POLYGON_RASTERIZER = "pycocotools"

# 0–1000 归一化坐标的满量程：x_norm = x / W * NORM1000_SCALE（浮点，不取整）。
# 各家族的取整/量化方式属于适配器，不在这里。依据：DEV-0001 任务书 §4。
NORM1000_SCALE = 1000.0

# ---------------------------------------------------------------------------
# 网格（像素 token 通道 E3b、E1 数字字符网格）
# ---------------------------------------------------------------------------

# E3b 默认网格边长。依据：简报 §7.2"0/1，32×32（默认）"。
GRID_SIZE_E3B = 32
# E1 零样本数字字符网格边长。依据：DEV-0001 任务书 §2。
GRID_SIZE_E1 = 16

# mask → 网格：格内前景像素占比 ≥ 0.5 记 1（用整数比较 2*前景 >= 格面积，无浮点误差）。
# 依据：DEV-0001 任务书 §2。
GRID_FG_AREA_THRESHOLD = 0.5

# 网格 → mask：第 k 条格边界在 floor(k * H / N)（行）、floor(k * W / N)（列）。
# mask → 网格也用同一组格边界，保证往返一致。依据：DEV-0001 任务书 §2。
GRID_CELL_EDGE_RULE = "floor(k*size/N)"

# E3b 允许新增的 token 总数上限。依据：简报 §4 第 4 条、§7.2。
MAX_NEW_GRID_TOKENS = 300

# ---------------------------------------------------------------------------
# 深度分档（暂定，E3b 预注册前定）
# ---------------------------------------------------------------------------

# 64 档对数均匀，[D_MIN, D_MAX] 米；越界裁剪；反解码取档的几何中心。
# 依据：简报 §7.2"64 档对数深度"；D_MIN/D_MAX 为 DEV-0001 任务书给的暂定值。
DEPTH_NUM_BINS = 64
D_MIN = 0.5
D_MAX = 10.0

# ---------------------------------------------------------------------------
# NYUv2 深度评测
# ---------------------------------------------------------------------------

# Eigen crop：在 480×640 上取行 45:471、列 41:601。
# 依据：cleinc/bts pytorch/bts_eval.py 里 eval_mask[45:471, 41:601] = 1（AdaBins 相同）。
NYU_EVAL_HW = (480, 640)
NYU_EIGEN_CROP = (45, 471, 41, 601)  # (row_start, row_end, col_start, col_end)，半开

# 有效 GT 深度区间 (NYU_DEPTH_MIN, NYU_DEPTH_MAX]；预测先裁剪到 [NYU_DEPTH_MIN, NYU_DEPTH_MAX]。
# 依据：cleinc/bts pytorch/bts_eval.py 的 gt > min_depth_eval 且 gt < max_depth_eval（两端都开，AdaBins 相同）；
# D10(1) 裁定上端取开区间，对齐官方评测（取代任务书最初写的闭区间）。
NYU_DEPTH_MIN = 1e-3
NYU_DEPTH_MAX = 10.0
NYU_DEPTH_MAX_INCLUSIVE = False

# ---------------------------------------------------------------------------
# 分割指标
# ---------------------------------------------------------------------------

# Boundary IoU 的膨胀比例（相对图像对角线）。
# 依据：bowenc0221/boundary-iou-api boundary_iou/utils/boundary_utils.py mask_to_boundary 的默认值。
BOUNDARY_IOU_DILATION_RATIO = 0.02

# 框定位：IoU ≥ 0.5 记对。依据：简报 §7.1 Acc@IoU0.5。
BOX_ACC_IOU_THRESHOLD = 0.5

# ADE20K：150 类，reduce_zero_label=True，忽略值 255。依据：mmseg ADE20KDataset 配置。
ADE_NUM_CLASSES = 150
ADE_REDUCE_ZERO_LABEL = True
ADE_IGNORE_INDEX = 255

# 开放词汇检测：预测全部没有分数时，AP 取 K 次随机打破平局（图内与跨图一起随机置换）的平均，固定种子。
# 依据：DEV-0001 addendum-1 第 3 条、D10(5)；K 取 20 是暂定（独立审查实测编号置换下 AP50 的标准差约 0.005，
# 20 次平均的蒙特卡洛误差约 0.001）。
DETECTION_TIEBREAK_K = 20
DETECTION_TIEBREAK_SEED = 20260930
# 单一工作点 P/R/F1 的 IoU 阈值。依据：addendum-1 第 3 条"IoU=0.5 下的 P、R、F1"。
DETECTION_OPERATING_IOU = 0.5

# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------

# bootstrap：默认重采样 1000 次，95%，固定种子。依据：DEV-0001 任务书 §2；简报 §11.5。
BOOTSTRAP_N_RESAMPLES = 1000
BOOTSTRAP_CI_LEVEL = 0.95
BOOTSTRAP_SEED = 20260929

# 简报 §8.4 "显著"：|均值差| > SIGNIFICANCE_STD_MULTIPLIER × 合并标准差，且配对 bootstrap CI 不含 0。
SIGNIFICANCE_STD_MULTIPLIER = 2.0
