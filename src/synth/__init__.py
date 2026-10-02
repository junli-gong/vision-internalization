"""E5 合成数据生成器（简报 §7.4、§8.8）。
第一部分（DEV-0007、DEV-0009）：公共框架 + c 遮挡补全、d 短时物理、e 文本到意象；
第二部分（DEV-0010）：a 出平面三维旋转、b 视角变化（程序化 3D 实体 + 自写确定性光栅化）。

只用白名单里的无学习组件（numpy、shapely、Pillow、pymunk 2D 物理；2D 与 3D 光栅化、三角函数都自写，见 raster、render3d、dmath），
只用程序生成的几何体（N5、N7 答复前），不下载任何资产，不用 pyrender / OpenGL / Blender / Kubric。

跨机器复现（D24）：a、b、c、e 在任何平台上逐位相同；d 依赖 pymunk 的浮点行为，规范平台是服务器
（x86_64 Linux），d 的数据只在规范平台上生成。核对规则见 tests/test_synth_repro.py。
生成器版本按任务记（constants.TASK_GENERATOR_VERSIONS；c、d、e 为 2，a、b 为 4）。
DEV-0018（D28-13）：a 加目标相对深度（起伏）网格与基线的起伏（relief），b 放大物体。

入口：
    from src.synth import generate, SynthDataset
    s = generate("occlusion", "train", 0)            # 或任务字母 "a" / "b" / "c" / "d" / "e"
    ds = SynthDataset("physics", "val", 1000, bin=2)  # ds[i] = generate("physics", "val", i, bin=2)

模块：constants（全部 E5 约定）、seeding（种子与划分码）、dmath（确定的三角函数与求平均）、raster（精确整数光栅化）、
shapes（2D 形状族、身份划分、位姿）、scene（2D 场景规格与渲染）、sample（样本、基线、摘要）、
occlusion / physics / imagery（c、d、e）、solids（3D 实体、对称性、实体身份）、render3d（3D 场景、相机、光栅化）、
warp2d（mask 的仿射、矩与矩匹配）、relief（a 的起伏网格与起伏误差）、rotation3d / viewpoint（a、b）、export（可选落盘）。
"""

from src.synth.api import TASK_BINS, SynthDataset, generate, num_bins
from src.synth.sample import Prediction, Sample, SynthGenerationError, sample_digest

__all__ = [
    "TASK_BINS",
    "Prediction",
    "Sample",
    "SynthDataset",
    "SynthGenerationError",
    "generate",
    "num_bins",
    "sample_digest",
]
