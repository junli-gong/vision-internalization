"""E5 合成数据生成器（简报 §7.4、§8.8）的约定：只在这里定义，src/synth 的其余模块与测试一律 import。

跨模块约定（默认网格边长、mask → 网格规则、像素中心与框的边界归属）仍然来自 src/constants.py、
src/grid/codec.py，这里不重述。多边形光栅化是例外：E5 合成数据用 src/synth/raster.py 的精确整数规则（DEV-0009、D24），
不走 src/executors/raster.py 的 pycocotools（后者仍用于对齐 COCO/RefCOCO 真值）。

本文件的数值都是 DEV-0007 的暂定默认值（DEV-0009 只加了确定性相关的几项），最终以 E5 预注册为准，由项目负责人确认。
改动任何一个会改变生成结果的数值，都必须同时把 SYNTH_GENERATOR_VERSION 加一，并重新钉住
tests/data/synth_digests_v2.json 里的样本摘要（样本 = 纯函数(任务, 划分, 序号, 生成器版本)；
核对规则见 tests/test_synth_repro.py，重新生成用 scripts/synth_pin_digests.py）。
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 版本与种子
# ---------------------------------------------------------------------------

# 生成器版本：进入每个样本的种子，也写进样本元数据。
# 1：DEV-0007（作废，没有产出过数据，D24）；2：DEV-0009，跨机器逐位复现的修复。
SYNTH_GENERATOR_VERSION = 2

# 种子树的根，固定不变；与版本一起进入 numpy.random.SeedSequence 的 entropy。
SYNTH_SEED_ROOT = 20260930

# 任务名。字母与简报 §7.4 表格一致；a（出平面旋转）、b（视角变化）留给第二部分（3D）。
TASKS = ("occlusion", "physics", "imagery")
TASK_ALIASES = {"c": "occlusion", "d": "physics", "e": "imagery"}
# 任务 → 固定整数，进入种子。不用 hash()（字符串哈希每个进程随机化）。1、2 留给 a、b。
TASK_CODES = {"occlusion": 3, "physics": 4, "imagery": 5}

SPLITS = ("train", "val", "test")
SPLIT_CODES = {"train": 0, "val": 1, "test": 2}
# 场景种子 = SPLIT_CODES[split] * SCENE_SEED_STRIDE + index，0 <= index < SCENE_SEED_STRIDE。
# 三个划分的场景种子落在互不相交的整数区间里，按构造不相交。
SCENE_SEED_STRIDE = 1 << 48

# 单个样本最多尝试的次数（拒绝采样）；超过即抛 SynthGenerationError，不静默降级。
MAX_ATTEMPTS = 400

# d（短时物理）的规范平台（D24）：(platform.machine(), platform.system())。d 的数据只在这里生成，
# 钉住的 d 摘要也只在这里比对。c、e 在任何平台上都必须逐位相同。
CANONICAL_PLATFORM = ("x86_64", "Linux")

# ---------------------------------------------------------------------------
# 物体身份与划分
# ---------------------------------------------------------------------------

# 身份 = (形状族, 形状变体序号, 外观序号)。每个 (形状族, 形状变体) 下的全部外观按
# sha256(IDENTITY_SPLIT_SALT + 身份键) 排序，前 80% 进 train、接着 10% 进 val、其余进 test（shapes.py 详述）。
IDENTITY_SPLIT_FRACTIONS = (("train", 0.8), ("val", 0.1), ("test", 0.1))
IDENTITY_SPLIT_SALT = "visint-e5-identity-split-v1"

# ---------------------------------------------------------------------------
# 画布与数值量化（跨机器逐位复现，见 shapes.py 模块说明）
# ---------------------------------------------------------------------------

CANVAS_HW = (512, 512)  # (H, W)，RGB
CANVAS_MARGIN_PX = 2.0  # 物体离画布边缘至少这么远（c、e 的全部物体，d 的全部帧）

COORD_QUANTUM = 1.0 / 256  # 画布上的顶点坐标、中心、半径量化到 1/256 px（raster.py 的整数单位由它导出）
UNIT_SHAPE_QUANTUM = 2.0**-24  # 单位形状（外接半径 1）的顶点量化步长
TRIG_QUANTUM = 2.0**-40  # 旋转与方向用的 cos、sin 量化步长（dmath.rot_cs）
ANGLE_QUANTUM_DEG = 1.0 / 256  # c、d 的连续角度量化步长（度）
CURVE_SEGMENTS = 128  # 整圆的折线段数；圆弧按角度比例
# 确定的三角函数（dmath.py）：decimal 的有效位数。结果误差 ~1e-38，远小于 TRIG_QUANTUM。
DEC_PREC = 40
# 会影响分档、拒绝或写进元数据的平均值（e 的平均位移），逐项量化到这个步长（像素）后精确求和（dmath.fixed_point_mean）。
# e 的单项位移 < 2^11 px、项数 < 2^17：项数 × 单项 / 步长 < 2^52，部分和都是精确整数。
MEAN_TERM_QUANTUM = 2.0**-24

# ---------------------------------------------------------------------------
# 形状族与变体（形状参数离散化；单位形状的定义见 shapes.py）
# ---------------------------------------------------------------------------

SHAPE_FAMILIES = ("circle", "ellipse", "rectangle", "triangle", "polygon", "star", "cross", "l_shape", "crescent")
ELLIPSE_AXIS_RATIOS = (0.45, 0.60, 0.75)  # 短轴 / 长轴
RECT_ASPECTS = (0.35, 0.50, 0.65, 0.80, 1.00)  # 高 / 宽；1.00 即正方形
# (类型, 参数)：等边；等腰（高 / 底）；直角（水平直角边 / 竖直直角边）
TRIANGLE_VARIANTS = (("equilateral", 1.0), ("isosceles", 1.50), ("isosceles", 0.60), ("right", 1.00), ("right", 1.60))
POLYGON_SIDES = (5, 6, 7, 8)
STAR_POINTS = (5, 6, 8)
STAR_INNER_RATIOS = (0.40, 0.55)  # 内顶点半径 / 外顶点半径
CROSS_ARM_WIDTHS = (0.30, 0.45)  # 臂宽 / 全长
L_ARM_WIDTHS = (0.30, 0.45)  # 臂宽 / 高
L_WIDTH_RATIOS = (0.65, 1.00)  # 宽 / 高
CRESCENT_CUTS = ((0.90, 0.30), (0.75, 0.50))  # (切割圆半径, 切割圆圆心右移量)，都相对外圆半径

# ---------------------------------------------------------------------------
# 外观（颜色或纹理）与背景
# ---------------------------------------------------------------------------

# 10 种命名颜色（RGB）。都避开背景的浅色和纹理的黑色。
COLORS = (
    ("red", (220, 30, 45)),
    ("orange", (245, 130, 25)),
    ("yellow", (235, 200, 20)),
    ("green", (40, 150, 55)),
    ("teal", (0, 140, 140)),
    ("blue", (40, 90, 210)),
    ("purple", (125, 50, 170)),
    ("pink", (235, 105, 175)),
    ("brown", (135, 80, 35)),
    ("gray", (125, 125, 125)),
)
PATTERNS = ("solid", "striped", "checkered")
PATTERN_SECOND_COLOR = ("black", (25, 25, 25))
# 条纹与棋盘格的格宽 = 物体外接半径 × PATTERN_CELL_FRAC（在物体自身坐标系里，随物体旋转）
PATTERN_CELL_FRAC = 0.2

BACKGROUND_COLORS = ((236, 236, 236), (246, 240, 224), (224, 234, 246), (228, 242, 228))
BACKGROUND_KINDS = ("solid", "vgradient")

# ---------------------------------------------------------------------------
# c 遮挡补全
# ---------------------------------------------------------------------------

# 难度 = 目标被遮住的面积比例 1 - |可见| / |非模态|（全分辨率像素计），档为半开区间 [lo, hi)。
OCCLUSION_BINS = ((0.1, 0.3), (0.3, 0.5), (0.5, 0.7))
OCC_TARGET_RADIUS = (0.16, 0.26)  # 外接半径 / min(H, W)
OCC_OCCLUDER_RADIUS = (0.10, 0.22)
OCC_DISTRACTOR_RADIUS = (0.06, 0.14)
OCC_NUM_OCCLUDERS = (1, 3)  # 闭区间
OCC_NUM_DISTRACTORS = (0, 2)  # 闭区间
OCC_OFFSET_RANGE = (0.35, 1.0)  # 遮挡物初始偏移 / (r_目标 + r_遮挡物)，再乘公共缩放 λ
OCC_BISECTION_STEPS = 14
OCC_BIN_MARGIN = 0.01  # 目标遮挡比例在 [lo + margin, hi - margin] 内均匀抽
# "非平凡"：可见部分凸包与真值的 IoU < OCC_NONTRIVIAL_HULL_IOU。
# 每个样本以 OCC_NONTRIVIAL_QUOTA 的概率被要求非平凡（拒绝凸包 IoU 达到阈值的场景；这类样本的目标遮挡比例
# 从 max(档下限, 1 - 阈值) 起抽），于是每一档里非平凡样本的比例 ≥ OCC_NONTRIVIAL_QUOTA（在大样本意义上）。
OCC_NONTRIVIAL_HULL_IOU = 0.85
OCC_NONTRIVIAL_QUOTA = 0.5
# 被要求非平凡的尝试，先用连续几何算凸包 IoU；≥ 阈值 + 这个余量的直接重来（只省时间，最终以像素判定为准）
OCC_GEOM_PRECHECK_SLACK = 0.02

# ---------------------------------------------------------------------------
# d 短时物理（pymunk，侧视）
# ---------------------------------------------------------------------------

# 一步 = 1/30 秒，内部细分为 PHYSICS_SUBSTEPS 个子步（pymunk 每次 step 1/300 秒）。
PHYSICS_STEP_SECONDS = 1.0 / 30.0
PHYSICS_SUBSTEPS = 10
PHYSICS_ITERATIONS = 20  # pymunk 约束求解迭代次数（默认 10）
# 难度 = 步数 k，四档单点。
PHYSICS_K_BINS = (5, 10, 20, 40)
PHYSICS_ACTIONS = ("drop", "push", "collide")
# 长度单位 = 画布宽 W（像素）。重力、速度都按 W 计：512 px 画布上重力 = 1024 px/s²。
PHYSICS_GRAVITY = 2.0  # W / s²，方向向下（图像 y 轴）
PHYSICS_GROUND_FRAC = 0.9  # 地面上表面在 0.9 H 处（取整到像素）
PHYSICS_GROUND_COLOR = (70, 70, 80)
PHYSICS_FAMILIES = ("rectangle", "circle")  # 盒子与圆
PHYSICS_RADIUS = (0.06, 0.12)  # 外接半径 / W
PHYSICS_SPEED_HUNDREDTHS = (25, 75)  # 初速 = 整数 / 100 W/s，闭区间
PHYSICS_DROP_GAP = (0.10, 0.45)  # 放下时物体最低点离支撑面的高度 / W
PHYSICS_STACK_PROB = 0.4  # 放下的物体正下方放一个静止物体的概率
PHYSICS_OBSTACLE_PROB = 0.5  # 推的方向上放一个静止障碍物的概率
PHYSICS_OBSTACLE_GAP = (0.02, 0.20)  # 被推物体与障碍物之间的间隙 / W
PHYSICS_COLLIDE_GAP = (0.05, 0.30)  # 相向运动的两个物体之间的初始间隙 / W
PHYSICS_NUM_BYSTANDERS = (0, 2)  # 动作之外的静止物体数，闭区间
PHYSICS_MIN_GAP = 0.02  # 地面上物体之间的最小水平间隙 / W
# 摩擦、弹性（pymunk 在接触处把两边的系数相乘）
PHYSICS_FRICTION_OBJECT = 0.6
PHYSICS_FRICTION_GROUND = 1.0
PHYSICS_ELASTICITY_OBJECT = 0.3
PHYSICS_ELASTICITY_GROUND = 1.0
PHYSICS_DENSITY = 1.0e-3  # 质量 = 面积(px²) × 密度；只影响相对质量
# 合理性检查的容差（生成时作为拒绝条件，测试里也用）
PHYSICS_REST_TOL_PX = 0.5  # 未被碰到的静止物体，k 步内位移上限
PHYSICS_REST_TOL_DEG = 0.5  # 同上，转角上限
PHYSICS_PENETRATION_TOL_PX = 2.0  # 任一接触点的穿透深度上限（pymunk 接触距离的负值，先量化到 COORD_QUANTUM）
PHYSICS_OVERLAP_TOL = 0.01  # 第 k 步两两 mask 的公共像素数 / 较小者像素数 的上限（第 0 步要求没有公共像素）
PHYSICS_SUPPORT_TOL_PX = 1.5  # "落在地面上"：最低点离地面上表面的距离上限

# ---------------------------------------------------------------------------
# e 文本到意象
# ---------------------------------------------------------------------------

# 坐标用画布的归一化坐标，保留两位小数（imagery.py 详述）。
IMAGERY_DECIMALS = 2
# 难度 = 目标物体各点的平均位移（按对称性取最小的等价旋转）/ min(H, W)，档为半开区间 [lo, hi)。
IMAGERY_BINS = ((0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.40))
IMAGERY_NUM_OBJECTS = (1, 3)  # 闭区间
IMAGERY_RADIUS_HUNDREDTHS = (6, 18)  # 外接半径（百分之一画布），闭区间
IMAGERY_MIN_RADIUS_HUNDREDTHS = 4  # 缩放后外接半径下限
IMAGERY_MAX_RADIUS_HUNDREDTHS = 30  # 缩放后外接半径上限
IMAGERY_SCALE_RANGE = (0.5, 2.0)
IMAGERY_TRANSFORMS = ("translate", "rotate", "scale", "rotate_translate", "scale_translate", "scale_rotate_translate")
IMAGERY_BIN_MARGIN = 0.002
IMAGERY_BISECTION_STEPS = 30
