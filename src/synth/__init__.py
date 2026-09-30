"""E5 合成数据生成器（简报 §7.4、§8.8），第一部分：公共框架 + c 遮挡补全、d 短时物理、e 文本到意象。

只用白名单里的无学习组件（numpy、scipy、shapely、Pillow、pycocotools 光栅化、pymunk 2D 物理），
只用程序生成的几何体（N5、N7 答复前），不下载任何资产。

入口：
    from src.synth import generate, SynthDataset
    s = generate("occlusion", "train", 0)            # 或任务字母 "c" / "d" / "e"
    ds = SynthDataset("physics", "val", 1000, bin=2)  # ds[i] = generate("physics", "val", i, bin=2)

模块：constants（全部 E5 约定）、seeding（种子与划分码）、shapes（形状族、身份划分、光栅化）、
scene（场景规格与渲染）、sample（样本、基线、摘要）、occlusion / physics / imagery（三类任务）、export（可选落盘）。
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
