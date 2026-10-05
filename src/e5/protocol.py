"""E5 评测的约定（DEV-0018；PREREGISTRATION.md 的 E5 草稿 v0）：任务、评测集与开发集的样本选择、系统、标题口径、分列维度。

只在这里定义，src/e5 的其余模块、scripts/run_e5.py 与测试一律 import。

评测集（eval）与开发集（dev）
- 样本由生成器即时生成（src.synth.generate），不落盘。评测集取 test 划分，开发集取 val 划分（只用于核对格式、截断率与解析失败，
  不看指标；简报 §4 第 7 条：测试集只在决策门处使用）。
- 序号规则：任务 t 有 B 个难度档，每档取 n 条，则取划分里的序号 0, 1, …, B·n − 1，档 = 序号 mod B（生成器的默认轮转，
  src/synth/api.py），即第 b 档的第 k 条是序号 k·B + b。样本顺序 = 序号升序（各档交错），断点续传与逐字节复现都依赖它。
  默认 n：评测集每档 100 条（DEFAULT_N_PER_BIN_EVAL），开发集每档 2 条（DEFAULT_N_PER_BIN_DEV），都可配置。
  加大 n 只在末尾追加序号，已有的样本不变（前缀性质）；但档内的条数与顺序会变，所以 n 记进运行配置，续跑必须相同。
- 样本 id = "<任务>/<划分>/<序号>"（任务用规范名，如 rotation3d/test/12）。
- 画布、网格边长都用生成器的默认值（512×512，GRID_SIZE_E3B = 32）。d（physics）的数据只在规范平台上生成（D24）：
  非规范平台上只允许 --tiny 彩排（runner 检查）。

系统（目录 experiments/E5/<编号>/<系统>/<模型>/<任务>/）
- zeroshot：零样本基座，E1 的数字网格格式扩到 32×32（a、b 用 33 个符号的起伏网格；b 自 DEV-0028），关思考；
- cot：文本思维链，开思考，预算强制收尾（思考 4096 + 作答 1024，D59-5；DEV-0028）；a、b 最后输出 33 符号的起伏网格，
  c–e 的最终回答是 0–1000 坐标的多边形 JSON，光栅化后评分；
- baseline：生成器给出的平凡基线与参照线，直接评分；<模型> 位置是基线名（如 copy）；
- external：外部系统（LISA-7B）给出的 mask，只做评分；<模型> 位置是外部系统名。读入与目录格式在 E5 的 external 模块里，
  只有它允许的几个评分入口可以 import 它（硬约束 1，tests/test_e5_external.py）。

标题口径（D28-13；预注册草稿 §2）
- a：起伏误差，只统计下限 ≥ 30° 的三档；mask IoU 作次要指标（同一口径）。
- b：只用环绕移动（全部四档）；平移作容易档单列。
- c、d、e：全部样本。
分列：每档单列；a 按实体族、b 按移动类型（环绕 / 6 种平移）、c 按目标的形状族、d 按动作、e 按变换类型。
"""

from __future__ import annotations

import platform
from dataclasses import dataclass

from src.constants import GRID_SIZE_E3B
from src.synth.api import TASK_BINS
from src.synth.constants import CANONICAL_PLATFORM, CANONICAL_PLATFORM_TASKS, ROT3D_BINS, TASK_ALIASES, TASKS, VIEW_ORBIT_BINS
from src.synth.seeding import canonical_task

# a–e 的顺序（简报 §7.4 的表）
E5_TASKS: tuple[str, ...] = tuple(TASK_ALIASES[k] for k in sorted(TASK_ALIASES))
if set(E5_TASKS) != set(TASKS):  # pragma: no cover - 常量写错
    raise ImportError("E5_TASKS 与生成器的任务表不一致")
TASK_LETTER = {t: k for k, t in TASK_ALIASES.items()}
E5_GRID_SIZE = GRID_SIZE_E3B

SYSTEMS: tuple[str, ...] = ("zeroshot", "cot", "baseline", "external")
MODEL_SYSTEMS: tuple[str, ...] = ("zeroshot", "cot")
SUBSETS: dict[str, str] = {"eval": "test", "dev": "val"}  # 子集 → 生成器的划分
DEFAULT_N_PER_BIN_EVAL = 100
DEFAULT_N_PER_BIN_DEV = 2

# a 的标题只统计下限 ≥ 30° 的档（D28-13）
A_HEADLINE_MIN_DEG = 30
A_HEADLINE_BINS: tuple[int, ...] = tuple(b for b, (lo, _hi) in enumerate(ROT3D_BINS) if lo >= A_HEADLINE_MIN_DEG)
B_HEADLINE_MOVE = "orbit"
# b 的标题只统计环绕移动里下限 ≥ 20° 的档（D72-1：只统计非真值二维捷径——原样复制或平移——已经不比平面更好地解释起伏的档；
# 同一条规则在 a 上复现 30°。val 上的复核见 RUN-0031）
B_HEADLINE_MIN_DEG = 20
B_HEADLINE_BINS: tuple[int, ...] = tuple(b for b, (lo, _hi) in enumerate(VIEW_ORBIT_BINS) if lo >= B_HEADLINE_MIN_DEG)


@dataclass(frozen=True)
class EvalItem:
    task: str
    split: str
    index: int
    bin: int

    @property
    def id(self) -> str:
        return sample_id(self.task, self.split, self.index)


def sample_id(task: str, split: str, index: int) -> str:
    return f"{canonical_task(task)}/{split}/{int(index)}"


def parse_sample_id(sid: str) -> tuple[str, str, int]:
    task, split, index = sid.split("/")
    return canonical_task(task), split, int(index)


def num_bins(task: str) -> int:
    return len(TASK_BINS[canonical_task(task)])


def eval_items(task: str, subset: str, n_per_bin: int) -> list[EvalItem]:
    """子集（eval / dev）里任务 task 的样本清单，规则见模块说明。"""
    t = canonical_task(task)
    if subset not in SUBSETS:
        raise ValueError(f"subset 必须是 {tuple(SUBSETS)} 之一，收到 {subset!r}")
    if isinstance(n_per_bin, bool) or not isinstance(n_per_bin, int) or n_per_bin < 1:
        raise ValueError(f"n_per_bin 必须是正整数，收到 {n_per_bin!r}")
    b = num_bins(t)
    split = SUBSETS[subset]
    return [EvalItem(t, split, i, i % b) for i in range(b * n_per_bin)]


def default_n_per_bin(subset: str) -> int:
    return DEFAULT_N_PER_BIN_EVAL if subset == "eval" else DEFAULT_N_PER_BIN_DEV


def on_canonical_platform() -> bool:
    return (platform.machine(), platform.system()) == CANONICAL_PLATFORM


def needs_canonical_platform(task: str) -> bool:
    return canonical_task(task) in CANONICAL_PLATFORM_TASKS


# ---------------------------------------------------------------------------
# 分列维度与标题口径（只读样本的元数据）
# ---------------------------------------------------------------------------


def groups_of(task: str, meta: dict) -> dict[str, str]:
    """一条样本在各分列维度上的取值（档另记）。"""
    t = canonical_task(task)
    objs = meta["scene"]["objects"]
    if t == "rotation3d":
        return {"family": objs[0]["family"]}
    if t == "viewpoint":
        mv = meta["move"]["type"]
        return {"move": mv, "move_kind": "orbit" if mv == B_HEADLINE_MOVE else "shift"}
    if t == "occlusion":
        return {"family": objs[int(meta["target_obj_id"])]["family"]}
    if t == "physics":
        return {"action": meta["action"]}
    if t == "imagery":
        return {"transform": meta["transform"]["type"]}
    raise ValueError(t)


def in_headline(task: str, bin_index: int, groups: dict[str, str]) -> bool:
    """这条样本是否计入标题口径（见模块说明）。"""
    t = canonical_task(task)
    if t == "rotation3d":
        return int(bin_index) in A_HEADLINE_BINS
    if t == "viewpoint":
        return groups["move"] == B_HEADLINE_MOVE and int(bin_index) in B_HEADLINE_BINS
    return True


def headline_scope_text(task: str) -> str:
    t = canonical_task(task)
    if t == "rotation3d":
        return f"档 {list(A_HEADLINE_BINS)}（下限 ≥ {A_HEADLINE_MIN_DEG}°）"
    if t == "viewpoint":
        return f"环绕移动，档 {list(B_HEADLINE_BINS)}（下限 ≥ {B_HEADLINE_MIN_DEG}°）"
    return "全部样本"
