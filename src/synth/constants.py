"""E5 合成数据生成器（简报 §7.4、§8.8）的约定：只在这里定义，src/synth 的其余模块与测试一律 import。

跨模块约定（默认网格边长、mask → 网格规则、像素中心与框的边界归属）仍然来自 src/constants.py、
src/grid/codec.py，这里不重述。多边形光栅化是例外：E5 合成数据用 src/synth/raster.py 的精确整数规则（DEV-0009、D24），
不走 src/executors/raster.py 的 pycocotools（后者仍用于对齐 COCO/RefCOCO 真值）。

本文件的数值都是 DEV-0007 的暂定默认值（DEV-0009 只加了确定性相关的几项，DEV-0010 加了 3D 的 a、b），
最终以 E5 预注册为准，由项目负责人确认。
改动任何一个会改变生成结果的数值，都必须同时把受影响任务的 TASK_GENERATOR_VERSIONS 加一（SYNTH_GENERATOR_VERSION
随之更新为各任务版本的最大值），并重新钉住 tests/data/synth_digests_v2.json 里该任务的样本摘要
（样本 = 纯函数(任务, 划分, 序号, 该任务的生成器版本)；核对规则见 tests/test_synth_repro.py，
重新生成用 scripts/synth_pin_digests.py）。
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 版本与种子
# ---------------------------------------------------------------------------

# 生成器版本（DEV-0010 起按任务记）。
# - TASK_GENERATOR_VERSIONS[任务]：进入该任务每个样本的种子，也写进样本元数据（meta["generator_version"]）。
# - SYNTH_GENERATOR_VERSION：生成器整体的版本 = 各任务版本的最大值（tests/data/synth_digests_v2.json 顶层记它）。
# 历史：1 = DEV-0007（作废，没有产出过数据，D24）；2 = DEV-0009，跨机器逐位复现的修复；
# 3 = DEV-0010，新增 a、b（3D）。c、d、e 的输出在版本 3 里没有变，仍记 2：版本号进种子，整体加一会改变它们已钉住的
# 180 个摘要，所以改为按任务记版本（任务书 DEV-0010 的后备方案）。
# 4 = DEV-0018（D28-13）：a 加"目标相对深度（起伏）网格"与基线的起伏（src/synth/relief.py），b 放大物体。只改 a、b；
# c、d、e 的输出不变，仍记 2。
# 5 = DEV-0028（D59-4）：b 加起伏网格（与 a 同一定义）、copy_flat 与真值仿射参照线 gt_affine_upper。只改 b；a 的输出不变
# （a 新增的凸起先验基线 copy_convex 不存进样本、评分时现算，所以 a 仍记 4），c、d、e 仍记 2。
TASK_GENERATOR_VERSIONS = {"occlusion": 2, "physics": 2, "imagery": 2, "rotation3d": 4, "viewpoint": 5}
SYNTH_GENERATOR_VERSION = 5
if SYNTH_GENERATOR_VERSION != max(TASK_GENERATOR_VERSIONS.values()):  # pragma: no cover - 常量写错
    raise ImportError("SYNTH_GENERATOR_VERSION 必须等于各任务版本的最大值")

# 种子树的根，固定不变；与版本一起进入 numpy.random.SeedSequence 的 entropy。
SYNTH_SEED_ROOT = 20260930

# 任务名。字母与简报 §7.4 表格一致：a 出平面三维旋转、b 视角变化（第二部分，3D）、c 遮挡补全、d 短时物理、e 文本到意象。
TASKS = ("occlusion", "physics", "imagery", "rotation3d", "viewpoint")
TASK_ALIASES = {"a": "rotation3d", "b": "viewpoint", "c": "occlusion", "d": "physics", "e": "imagery"}
# 任务 → 固定整数，进入种子。不用 hash()（字符串哈希每个进程随机化）。
TASK_CODES = {"rotation3d": 1, "viewpoint": 2, "occlusion": 3, "physics": 4, "imagery": 5}

SPLITS = ("train", "val", "test")
SPLIT_CODES = {"train": 0, "val": 1, "test": 2}
# 场景种子 = SPLIT_CODES[split] * SCENE_SEED_STRIDE + index，0 <= index < SCENE_SEED_STRIDE。
# 三个划分的场景种子落在互不相交的整数区间里，按构造不相交。
SCENE_SEED_STRIDE = 1 << 48

# 单个样本最多尝试的次数（拒绝采样）；超过即抛 SynthGenerationError，不静默降级。
MAX_ATTEMPTS = 400

# d（短时物理）的规范平台（D24）：(platform.machine(), platform.system())。d 的数据只在这里生成，
# 钉住的 d 摘要也只在这里比对。其余任务（a、b、c、e）在任何平台上都必须逐位相同。
CANONICAL_PLATFORM = ("x86_64", "Linux")
CANONICAL_PLATFORM_TASKS = ("physics",)
ANY_PLATFORM_TASKS = tuple(t for t in TASKS if t not in CANONICAL_PLATFORM_TASKS)

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

# ===========================================================================
# 第二部分（DEV-0010）：3D 实体、相机与渲染；a 出平面三维旋转、b 视角变化
# ===========================================================================

# ---------------------------------------------------------------------------
# 3D 实体（程序化低面数网格，solids.py 详述）
# ---------------------------------------------------------------------------

SOLID_FAMILIES = ("box", "sphere", "cylinder", "cone", "torus", "prism", "l_prism")
BOX_DIMS = ((1.0, 1.0, 1.0), (1.0, 0.5, 0.75), (0.6, 1.5, 0.6), (1.6, 0.6, 0.8))  # (x 宽, y 高, z 深)；(1,1,1) 即立方体
SPHERE_SUBDIVISIONS = 2  # 二十面体细分次数：320 个三角面
SPHERE_Y_RATIOS = (1.0, 0.6, 1.5)  # y 方向缩放：1.0 = 球，其余为旋转椭球
CYLINDER_HEIGHT_RATIOS = (0.5, 1.0, 2.0)  # 高 / 底面直径
CONE_HEIGHT_RATIOS = (0.75, 1.5)  # 高 / 底面直径
TORUS_TUBE_RATIOS = (0.25, 0.45)  # 管半径 / 中心圆半径
PRISM_VARIANTS = ((3, 0.6), (3, 1.6), (6, 0.6), (6, 1.6))  # (底面边数, 高 / 底面外接圆直径)
# 拉伸的 L 形：侧面是高 1、宽 L_PRISM_WIDTH 的 L（竖臂在左、横臂在下；两臂不等长，没有旋转对称），
# 变体 = (臂宽, 厚度)，都相对高
L_PRISM_WIDTH = 0.75
L_PRISM_VARIANTS = ((0.30, 0.30), (0.30, 0.60), (0.45, 0.30), (0.45, 0.60))
ROUND_SEGMENTS = 16  # 圆柱、圆锥、圆环的周向段数
TORUS_TUBE_SEGMENTS = 8  # 圆环管截面的段数
# 摆放姿态（局部坐标到世界坐标先做的旋转）：直立；绕 z 轴转 90°；绕 x 轴转 90°。三者分别让局部 y、x、z 方向朝上
SOLID_REST_POSES = ("upright", "side_z", "side_x")
# 实体的外观 = COLORS 里的 10 种纯色（shapes.APPEARANCES 的前 10 个，即图案 solid），平面着色

# ---------------------------------------------------------------------------
# 相机与渲染（render3d.py 详述）
# ---------------------------------------------------------------------------

# 针孔相机，固定内参：焦距 f = CAMERA_FOCAL_FRAC × min(H, W) 像素，主点在画布中心 (W/2, H/2)，无畸变。
# min(H, W) = 512 时 f = 512，短边方向的视场角约 53°。
CAMERA_FOCAL_FRAC = 1.0
# 量化步长（都是 2 的负整数次幂）
SCENE_QUANTUM_M = 2.0**-10  # 场景规格里的位置、距离、尺寸（米）
VERTEX_QUANTUM_M = 2.0**-20  # 相机坐标系里的顶点坐标（米）
DEPTH_QUANTUM_M = 2.0**-12  # 深度图与深度网格（米）；10 m 以内的深度都是 < 2^16 的整数单位
NORMAL_QUANTUM = 2.0**-16  # 面法向与光照方向的分量
# 平行光：相机坐标系（x 右、y 下、z 前）里指向光源的方向（左上方、相机这一侧），归一化后量化到 NORMAL_QUANTUM
LIGHT_DIR_CAM = (-1.0, -2.0, -1.5)
# Lambert 平面着色：亮度 = (SHADE_AMBIENT_256 + SHADE_DIFFUSE_256 · max(0, n·l)) / 256，颜色 = 底色 × 亮度（整数运算）
SHADE_AMBIENT_256 = 96
SHADE_DIFFUSE_256 = 160
# 物体 ID 图（uint8）：0 = 没有表面，k + 1 = 第 k 个物体（obj_id = k），FLOOR_ID = b 的地板
FLOOR_ID = 255

# ---------------------------------------------------------------------------
# a 出平面三维旋转（rotation3d.py 详述）
# ---------------------------------------------------------------------------

# 难度 = 报告的旋转角 |θ|（整数度，按对称性约简后的最小等价角）。前三档 [lo, hi)，最后一档 [90, 180] 两端都含。
ROT3D_BINS = ((15, 30), (30, 60), (60, 90), (90, 180))
# 旋转轴只取相机坐标系的水平轴（x）或竖直轴（y），过物体中心；物体中心在光轴上，所以轴与视线严格垂直（夹角 90°）
ROT3D_AXES = ("horizontal", "vertical")
ROT3D_RADIUS_M = (0.40, 0.65)  # 实体外接半径（中心到最远顶点）
ROT3D_DISTANCE_M = (1.8, 2.8)  # 相机到物体中心
ROT3D_ELEVATION_DEG = (15, 45)  # 相机俯角（整数度，闭区间）；物体中心在世界原点，相机方位角恒为 0，物体偏航角均匀
# 真值拟合的二维仿射上界（warp2d.moment_matched_affine）：A = Σ_gt^{1/2} Q Σ^{-1/2}，正交因子 Q 在 O(2) 上搜索：
# 转角每 AFFINE_SEARCH_COARSE_DEG 度一个（两种镜像），按每隔 AFFINE_SEARCH_STRIDE 像素抽样的 IoU 粗选，
# 在最好的 AFFINE_SEARCH_TOP 个附近按 1° 细化（隔 2 像素抽样），最后按全分辨率 IoU 取最好的。
AFFINE_SEARCH_COARSE_DEG = 10
AFFINE_SEARCH_STRIDE = 4
AFFINE_SEARCH_TOP = 3

# ---------------------------------------------------------------------------
# b 视角变化（viewpoint.py 详述）
# ---------------------------------------------------------------------------

# 相机移动：绕地板中心水平环绕（orbit），或不转向地平移：向右/左（相机的水平轴）、向上/下（竖直）、
# 向前/后（沿视线）。每个样本先以 VIEW_ORBIT_PROB 的概率选环绕，否则在 6 种平移里均匀选。
VIEW_MOVES = ("orbit", "right", "left", "up", "down", "forward", "backward")
VIEW_ORBIT_PROB = 0.5
# 难度档：第 k 档 = 环绕角的第 k 档或平移量的第 k 档（看移动类型）。前三档 [lo, hi)，最后一档两端都含。
VIEW_ORBIT_BINS = ((10, 20), (20, 40), (40, 80), (80, 180))  # 整数度
VIEW_SHIFT_BINS_CM = ((10, 20), (20, 40), (40, 80), (80, 120))  # 整数厘米
# DEV-0018（D28-13，b 的任务版本 3 → 4）：物体放大，让被点名的物体在 32×32 网格上至少约占 100 格。相机（距离、俯角、
# 移动的档位）不变，只把物体与布局放大（版本 3 的值：物体 3–4 个、外接半径 0.30–0.50 m、布局半径 1.1 m、间隙 0.05 m、
# 地板半径 1.7 m；目标在 32×32 网格上的中位数约 15 格），并加一条目标可见面积的下限 VIEW_MIN_TARGET_CELLS。
VIEW_NUM_OBJECTS = (2, 3)  # 闭区间（大物体放不下 4 个；摆不下第 3 个时只有 2 个）
VIEW_RADIUS_M = (0.75, 1.10)  # 实体外接半径
VIEW_PLACE_RADIUS_M = 1.3  # 物体中心离地板中心的最大水平距离（+ 最大外接半径 = 地板半径：物体不伸出地板）
VIEW_MIN_GAP_M = 0.10  # 物体水平外接圆之间的最小间隙（外接圆不相交，所以物体互不穿插）
VIEW_FLOOR_RADIUS_M = 2.4  # 圆形地板：世界原点为中心、y = 0，折线化为 VIEW_FLOOR_SEGMENTS 边形
VIEW_FLOOR_SEGMENTS = 64
VIEW_FLOOR_COLOR = (190, 184, 172)
VIEW_DISTANCE_M = (4.0, 5.0)  # 相机到地板中心（相机看向地板中心）
VIEW_ELEVATION_DEG = (20, 40)  # 相机俯角（整数度，闭区间）
VIEW_MIN_CAMERA_HEIGHT_M = 0.3  # 两个视角里相机都至少这么高
VIEW_MIN_VISIBLE_FRAC = 0.5  # 被点名的物体在两个视角里：可见像素 / 自身完整轮廓的像素 的下限
VIEW_MIN_VISIBLE_PX = 64  # 以及可见像素数的下限
# 以及（DEV-0018）可见面积 ≥ VIEW_MIN_TARGET_CELLS 个"src.constants.GRID_SIZE_E3B × GRID_SIZE_E3B 网格的格子"：
# 可见像素 ≥ ceil(VIEW_MIN_TARGET_CELLS · H · W / GRID_SIZE_E3B²)（512×512 画布上 25,600 像素）。按画布面积定义，与 cfg 的
# grid_size 无关；网格上实际的格数由"格内占比 ≥ 0.5"的规则决定，所以是"约" 100 格（D28-13）。
VIEW_MIN_TARGET_CELLS = 100
# 渲染前的预筛（只省时间）：目标顶点投影的凸包面积（1/256 px 整数单位里精确求）< 下限 × 这个比例的候选直接去掉。
# 像素中心数 ≤ 凸包面积 + 半周长 + 1（Pick 定理），所以 0.9 的余量不会去掉任何能通过像素判定的候选。
VIEW_HULL_PRECHECK_FRAC = (9, 10)
