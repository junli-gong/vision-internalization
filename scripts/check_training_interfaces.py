#!/usr/bin/env python
"""训练接口的核对仪器（DEV-0021；D21-2、D42-4；简报 §12）：受限 logits 的训练与推理一致率、LoRA 合并前后的一致率。
既是 E3 开工前的 GPU 核对，也是以后每个训练出来的模型评测前的标准合并核对（--checkpoint）。输出一个 JSON。

服务器：
    python scripts/check_training_interfaces.py --model qwen3.5-0.8b --checks restricted merge --n 200 \\
        --data-root $PROJ/data --out <json>
    # 评测前的合并核对（训练出来的检查点；--model 必须与检查点 manifest 的构造方式相同）：
    python scripts/check_training_interfaces.py --model qwen3.5-0.8b --checks merge restricted --n 200 \\
        --data-root $PROJ/data --checkpoint <检查点目录> --out <json>
开发机彩排（小型随机模型、CPU fp32、tests/fixtures/e1 的小数据；这时判据是逐个相同）：
    python scripts/check_training_interfaces.py --tiny --model gemma-4-e2b-it --checks restricted merge --out /tmp/x.json

输入（只读 src/e1，不改 E1 的配置与代码）：E1 grid16 任务的开发子集（三个 RefCOCO 开发子集，src/e1/tasks.select），
按 sha256 排序取 --n 条（src/data/subsets.seeded_sample，用途串 SELECT_PURPOSE），保持选择顺序；提示是 configs/e1_prompts.json
里 grid16 本家族的原文（PromptConfig.render）。两种思考开关各跑一遍，max_new_tokens = MAX_NEW_TOKENS；批配置照 E1 冻结的
batch_protocol（批大小读配置，左填充，选择顺序，连续样本成批；src/e1/runner.batch_plan）。

allowed 集合（受限解码与教师强制共用）：E1 数字网格格式实际用到的 token —— 用本家族的 tokenizer 编码一张 16×16 的示例网格
（src/grid/codec.grid_to_text），编码结果必须恰好由 "0"、"1"、"\\n" 三个单 token 组成，再加本家族的停止 token。不用新增的
grid token：它们是均值初始化，logits 全部相同，只会得到平局（DEV-0020 的提醒）。

检查：
- restricted：每批、每种思考开关调 training.restricted_consistency：受限解码（generate_restricted）与
  generate_batch(allowed_token_ids) 的 token 一致率与第一个分歧位置的分布；教师强制（training_forward + restricted_logits）的
  argmax 与解码 token 的一致率；不一致位置上的 logit 差。给了 --checkpoint 时在 load_trained(merge=True) 的模型上做。
- merge：
  - 不给 --checkpoint：先在 base 上生成并做教师强制（读数 effect_vs_base 的参照），再注入 LoRA（rank LORA_RANK、
    lora_alpha LORA_ALPHA，目标 = lora_target_modules()），按固定种子把 A、B 设成小尺度的随机值（randomize_lora：
    A ~ N(0, 1/in)，B ~ N(0, 1) 再整体缩放，使每个目标模块的 ‖scaling·B·A‖_F = --lora-rel-scale × ‖W‖_F），然后合并前生成 +
    教师强制、合并（merge_lora，在内存里）、合并后生成 + 以合并前的序列为目标做教师强制，逐项比较（checkpoint.merge_side +
    compare_merge_sides）。effect_vs_base 报 LoRA 让输出偏离 base 多少（判据里的前提：LoRA 确实让输出明显不同）。
  - 给 --checkpoint：factory.load_trained(merge=False) 的模型上生成 + 教师强制，释放；再 factory.load_trained(merge=True)，
    同一批开发提示上生成 + 以前者的序列为目标做教师强制，逐项比较。生成一律不受限（全词表 greedy），logits 比较在全词表上
    （training.project，lm_head 先升到 fp32）。
  - 读数：exact_rows、token_agreement、max_abs_logit（及 max_abs_hidden）、argmax 一致率、不一致位置的 logit 差。

不一致位置的 logit 差（gap，判据用它）：两种实现在某个位置选了不同的 token x ≠ y，gap 是参照 logits 上这两个 token 之差：
- restricted 的教师强制：L = 教师强制 logits，gap = L[argmax] − L[解码 token]；
- 生成序列第一个分歧处（restricted 的两种解码、merge 的合并前后）：分歧之前两条序列相同，教师强制 logits 在该位置对两者都有效，
  gap = |L[x] − L[y]|（merge 取合并前后两侧的较大者）；
- merge 的教师强制：gap = max(L_前[x] − L_前[y], L_后[y] − L_后[x])。
另报 top2_gap = 该位置参照 logits 的前两名之差。gap ≥ top2_gap，对方的 token 恰是第二名时两者相同；对方的 token 连第二名都
不是时只有 gap 会暴露，所以判据用 gap。

判据（输出的 criteria 与 pass；提议值，由管理员在 RUN 实测后冻结，D42-4）：每种思考开关分别判，
- GPU（不是 CPU fp32）：一致率（restricted：两种解码的 token_agreement、教师强制 agreement；merge：生成的 token_agreement、
  教师强制 argmax_agreement）≥ MIN_AGREEMENT，不一致位置的 gap 全部 < MAX_GAP；
- CPU fp32（--tiny）：必须逐个相同（exact_rows = token_agreement = agreement = 1，没有不一致的位置）；
- merge 的随机 LoRA 另有前提（两种设备都要求）：effect_vs_base 里生成与 base 逐行相同的比例 ≤ EFFECT_MAX_EXACT_ROWS，
  教师强制 argmax 与 base 相同的比例 ≤ EFFECT_MAX_ARGMAX_AGREEMENT。

输出 JSON 的字段见 build_document（format、argv、written_utc、code、versions、gpu、model、checkpoint、data、generation、
batch_protocol、allowed、criteria、checks、pass、timing_s、error）。
退出码：0 = 跑完且全部判据通过；1 = 跑完但有判据不过（JSON 照写）；2 = 出错（已有的结果与报错先写进 --out）。
argparse 用法错误也是 2。
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src import runlog  # noqa: E402
from src.adapters import checkpoint as CK  # noqa: E402
from src.adapters import factory, registry, training  # noqa: E402
from src.constants import GRID_SIZE_E1  # noqa: E402
from src.data.subsets import seeded_sample  # noqa: E402
from src.e1.prompts import PROMPTS_PATH, load_prompt_config  # noqa: E402
from src.e1.runner import batch_plan, ids_digest, load_images  # noqa: E402
from src.e1.tasks import DataPaths, select  # noqa: E402
from src.grid.codec import grid_to_text  # noqa: E402

FORMAT = "visint-training-interface-check/1"
TASK, SUBSET = "grid16", "dev"  # E1 的 grid16 开发子集与提示
SELECT_PURPOSE = "training_interface_check:grid16_dev"  # seeded_sample 的用途串（取 --n 条）
CHECKS = ("restricted", "merge")
DEFAULT_N = 200
MAX_NEW_TOKENS = 128
THINKING = (False, True)
MODE_KEYS = {False: "thinking_off", True: "thinking_on"}
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
TINY_INIT_STD = 0.05  # 与 scripts/run_e1.py 的 TINY_INIT_STD 相同（tests 钉住）：小模型重取参数，否则 greedy 输出几乎与输入无关
TINY_BATCH_SIZE = 2  # --tiny 的批大小：fixture 只有 3 条，取 2 才有两批（不受冻结的批配置约束，同 run_e1 --tiny）
LORA_RANK = 32  # 简报 §8.4 的默认 rank
LORA_ALPHA = 64  # env_report §7 的 LoRA 行：LoraConfig(r=32, lora_alpha=64)
LORA_REL_SCALE = 0.05  # 随机 LoRA：每个目标模块 ‖scaling·B·A‖_F / ‖W‖_F（依据见 randomize_lora）
LORA_SEED = 0
MIN_AGREEMENT = 0.995  # 提议值（D42-4）
MAX_GAP = 0.125  # 提议值（D42-4）：不一致位置的 logit 差上限 = 2^-3，即 bf16 在 [16, 32) 里的一个 ulp
EFFECT_MAX_EXACT_ROWS = 0.5  # 随机 LoRA 的前提：与 base 逐行相同的生成至多一半
EFFECT_MAX_ARGMAX_AGREEMENT = 0.99  # 随机 LoRA 的前提：教师强制 argmax 与 base 相同的位置至多 99%
MAX_LISTED = 500  # JSON 里逐项列出的不一致位置 / 分歧行的上限（统计量用全部）
TF_CHUNK = 256  # 全词表 logits 比较的分块


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="注册键（src/adapters/registry.py）")
    p.add_argument("--checks", nargs="+", default=list(CHECKS), choices=CHECKS, help="要做的检查（默认两项都做）")
    p.add_argument("--n", type=int, default=DEFAULT_N, help=f"开发样本数（默认 {DEFAULT_N}；样本不够时全取）")
    p.add_argument("--data-root", help="落地目录（默认环境变量 VISINT_DATA_ROOT；--tiny 时取 fixture）")
    p.add_argument("--out", required=True, help="输出 JSON 的路径")
    p.add_argument("--checkpoint", help="检查点目录（save_trained 写的）：合并核对改为 load_trained(merge=False) 对比 "
                                        "load_trained(merge=True)")
    p.add_argument("--tiny", action="store_true", help="小型随机模型 + CPU fp32 + tests/fixtures/e1 的小数据（开发机彩排）")
    p.add_argument("--device", help="默认：cuda，--tiny 时 cpu")
    p.add_argument("--dtype", choices=sorted(DTYPES), help="默认：bf16，--tiny 时 fp32")
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS, help=f"默认 {MAX_NEW_TOKENS}（任务书）")
    p.add_argument("--batch-size", type=int, help="批大小（默认取 configs/e1_prompts.json 冻结的 batch_protocol.batch_size；"
                                                  f"--tiny 默认 {TINY_BATCH_SIZE}）；真模型给别的值要加 --allow-unfrozen-batch")
    p.add_argument("--allow-unfrozen-batch", action="store_true", help="真模型允许冻结值以外的批大小（输出记 frozen = false）")
    p.add_argument("--lora-rel-scale", type=float, default=LORA_REL_SCALE,
                   help=f"随机 LoRA 的尺度 ‖scaling·B·A‖_F / ‖W‖_F（默认 {LORA_REL_SCALE}）")
    p.add_argument("--lora-seed", type=int, default=LORA_SEED, help=f"随机 LoRA 的种子（默认 {LORA_SEED}）")
    p.add_argument("--prompts", default=str(PROMPTS_PATH), help="E1 提示配置（默认 configs/e1_prompts.json）")
    p.add_argument("--dev-subsets", help="开发子集清单（默认 configs/e1_dev_subsets.json；--tiny 时取 fixture）")
    a = p.parse_args(argv)
    if a.model not in registry.MODELS:
        p.error(f"未登记的模型键 {a.model!r}；已登记：{sorted(registry.MODELS)}")
    a.checks = [c for c in CHECKS if c in a.checks]
    a.device = a.device or ("cpu" if a.tiny else "cuda")
    a.dtype = a.dtype or ("fp32" if a.tiny else "bf16")
    if a.tiny and a.batch_size is None:
        a.batch_size = TINY_BATCH_SIZE
    for name in ("n", "max_new_tokens", "batch_size"):
        v = getattr(a, name)
        if v is not None and v < 1:
            p.error(f"--{name.replace('_', '-')} 必须是正整数")
    if not a.lora_rel_scale > 0:
        p.error("--lora-rel-scale 必须是正数")
    if not a.tiny and not (a.data_root or os.environ.get("VISINT_DATA_ROOT")):
        p.error("真模型运行要给 --data-root（或设 VISINT_DATA_ROOT）")
    if a.checkpoint and not Path(a.checkpoint).is_dir():
        p.error(f"--checkpoint {a.checkpoint!r} 不是目录")
    return a


def resolve_batch_size(a, prompts) -> tuple[int, bool]:
    """(批大小, 是否就是冻结值)。真模型默认取冻结值；给了别的值要 --allow-unfrozen-batch（同 run_e1，D36-1）。"""
    frozen = int(prompts.batch_protocol()["batch_size"])
    bs = a.batch_size if a.batch_size is not None else frozen
    if not a.tiny and bs != frozen and not a.allow_unfrozen_batch:
        raise ValueError(f"--batch-size {bs} 不是冻结的批大小 {frozen}（configs/e1_prompts.json 的 batch_protocol）；"
                         "确要换就加 --allow-unfrozen-batch")
    return bs, bs == frozen


# ---------------------------------------------------------------------------
# 数据、allowed 集合、输入
# ---------------------------------------------------------------------------


def data_paths(a) -> DataPaths:
    if a.tiny and not a.data_root:
        base = DataPaths.fixtures()
    else:
        base = DataPaths(root=Path(a.data_root or os.environ["VISINT_DATA_ROOT"]))
    if a.dev_subsets:
        base = DataPaths(root=base.root, dev_subsets=Path(a.dev_subsets), eval_subsets=base.eval_subsets,
                         nyu_pairs=base.nyu_pairs, nyu_cache=base.nyu_cache, strict=base.strict)
    return base


def load_dev(paths: DataPaths, prompts, family: str, n: int) -> tuple[list[Any], list[str], dict[str, Any]]:
    """grid16 开发子集里按 sha256 排序取 n 条（保持选择顺序）与本家族 grid16 提示的渲染结果。"""
    sel = select(TASK, SUBSET, paths)
    ids = [s.id for s in sel.samples]
    chosen = set(seeded_sample(ids, n, SELECT_PURPOSE))
    samples = [s for s in sel.samples if s.id in chosen]
    texts = [prompts.render(TASK, family, s).text for s in samples]
    stats = {"task": TASK, "subset": SUBSET, "n_requested": n, "n_population": len(ids), "n": len(samples),
             "select": f"src/data/subsets.seeded_sample(purpose={SELECT_PURPOSE!r})，保持 src/e1/tasks.select 的顺序",
             "sample_ids_sha256": ids_digest([s.id for s in samples]), "sample_ids": [s.id for s in samples],
             "selection_stats": json.loads(json.dumps(sel.stats, ensure_ascii=False, default=str)),
             "prompt_template": prompts.entry(TASK, family)["template"], "prompt_config_sha256": prompts.sha256,
             "image_order": prompts.order}
    return samples, texts, stats


def grid_allowed(ad: Any) -> dict[str, Any]:
    """E1 数字网格格式实际用到的 token（见模块说明）+ 本家族的停止 token。"""
    tok = ad.tokenizer

    def enc(s: str) -> list[int]:
        return list(tok(s, add_special_tokens=False)["input_ids"])

    probe = np.zeros((GRID_SIZE_E1, GRID_SIZE_E1), dtype=bool)
    probe[GRID_SIZE_E1 // 4: GRID_SIZE_E1 // 2, GRID_SIZE_E1 // 4: 3 * GRID_SIZE_E1 // 4] = True
    text = grid_to_text(probe)
    rows = []
    for role, s in (("digit_0", "0"), ("digit_1", "1"), ("newline", "\n")):
        ids = enc(s)
        if len(ids) != 1:
            raise RuntimeError(f"{s!r} 在本家族的 tokenizer 里不是单个 token：{ids}")
        rows.append({"role": role, "text": s, "id": ids[0], "token": tok.convert_ids_to_tokens(ids[0])})
    fmt = {r["id"] for r in rows}
    probe_ids = enc(text)
    if set(probe_ids) != fmt:
        raise RuntimeError(f"示例网格的编码用到了 {sorted(set(probe_ids))}，不只是 0、1、换行 {sorted(fmt)}：数字或换行与相邻字符合并了，"
                           "allowed 集合要另定")
    stops = [int(s) for s in ad._stop_ids()]
    for s in stops:
        rows.append({"role": "stop", "text": None, "id": s, "token": tok.convert_ids_to_tokens(s)})
    ids = sorted({r["id"] for r in rows})
    if len(ids) != len(rows):
        raise RuntimeError(f"allowed 集合里有重复的 token：{rows}")
    if set(ids) & set(ad.grid_token_ids.values()):
        raise RuntimeError("allowed 集合不能含新增的 grid token")
    return {"ids": ids, "stop_ids": stops, "tokens": rows,
            "probe": {"grid_size": GRID_SIZE_E1, "n_tokens": len(probe_ids), "distinct_ids": sorted(set(probe_ids))},
            "rule": "tokenizer 编码 16×16 示例网格（grid_to_text）必须恰好由 0、1、换行三个单 token 组成，再加本家族的停止 token；"
                    "不用新增的 grid token（均值初始化，logits 全同）"}


def build_batches(ad: Any, samples: Sequence[Any], texts: Sequence[str], order: str,
                  plan: Sequence[tuple[int, int]]) -> list[list[Any]]:
    """每批的 ModelInputs（两种思考开关、两项检查共用）。注入了查询 token 的检查点：先取消查询再建输入（生成不收查询 token），
    建完按原样恢复登记（不碰任何行）。"""
    q = ad.token_table()["query"]
    n_q, mode = q.get("n"), q.get("position_mode")
    if n_q is not None:
        ad.clear_query_tokens()
    try:
        return [[ad.build_inputs(load_images(samples[i]), texts[i], order) for i in range(lo, hi)] for lo, hi in plan]
    finally:
        if n_q is not None:
            ad.inject_query_tokens(int(n_q), mode)


def make_runs(batches: Sequence[Sequence[Any]], plan: Sequence[tuple[int, int]],
              sample_ids: Sequence[str]) -> list[tuple[list[Any], bool, list[str]]]:
    """run = (一批输入, enable_thinking, 这批的样本 id)：先关思考的全部批，再开思考的全部批。"""
    return [(list(b), thinking, list(sample_ids[lo:hi])) for thinking in THINKING for b, (lo, hi) in zip(batches, plan)]


# ---------------------------------------------------------------------------
# 随机 LoRA
# ---------------------------------------------------------------------------


def randomize_lora(ad: Any, *, rel_scale: float, seed: int) -> dict[str, Any]:
    """注入后的 LoRA 设成固定种子的随机值。每个目标模块（lora_target_modules() 的顺序，CPU 生成器依次取）：
    A ~ N(0, 1/in)（Ax 的每个分量与 x 的均方根同量级），B ~ N(0, 1)，再把 B 整体缩放，使 ‖scaling·B·A‖_F = rel_scale·‖W‖_F。

    依据：
    - 尺度按每个模块自身的 ‖W‖ 定，与家族、尺寸、dtype 无关（代码与模型无关）。
    - 默认 0.05 = 每个目标线性层的权重沿一个 rank-32 的随机方向改 5%。量级与 LoRA 微调后常见的 ‖ΔW‖/‖W‖（1e-2 量级）同阶、
      偏大一点，仍是"小尺度"；远大于 bf16 的相对舍入（2^-9 ≈ 0.002），合并进 bf16 权重时增量不会被舍掉。
    - "与 base 明显不同"由 effect_vs_base 实测、判据里作前提（EFFECT_MAX_*）。开发机的小模型（4 层、hidden 64）上标定过：
      0.02 时 Qwen 的教师强制 argmax 与 base 相同 99.2% / 100%（不够）；0.05 时三个家族 98.7%、29%、57–83%，生成逐行都与 base
      不同（Qwen 开思考 1/3 相同）。真模型上的偏离以 RUN 实测为准（effect_vs_base）；不够或过大时用 --lora-rel-scale 调，
      不改代码。"""
    gen = torch.Generator().manual_seed(int(seed))
    mods = dict(ad.model.named_modules())
    ratios, b_std, a_std = [], [], []
    with torch.no_grad():
        for name in ad.lora_target_modules():
            m = mods[name]
            A, B = m.lora_A["default"].weight, m.lora_B["default"].weight
            s = float(m.scaling["default"])
            W = m.base_layer.weight
            a = torch.randn(tuple(A.shape), generator=gen, dtype=torch.float32) / math.sqrt(A.shape[1])
            b = torch.randn(tuple(B.shape), generator=gen, dtype=torch.float32)
            a_d, b_d = a.to(W.device), b.to(W.device)
            w_norm = float(W.float().norm())
            b_d = b_d * (rel_scale * w_norm / float((s * (b_d @ a_d)).norm()))
            A.copy_(a_d.to(A.dtype))
            B.copy_(b_d.to(B.dtype))
            ratios.append(float((s * (B.float() @ A.float())).norm()) / w_norm)
            a_std.append(float(A.float().std()))
            b_std.append(float(B.float().std()))

    def q(v: list[float]) -> dict[str, float]:
        arr = np.asarray(v)
        return {"min": float(arr.min()), "median": float(np.median(arr)), "max": float(arr.max())}

    return {"rel_scale": rel_scale, "seed": int(seed), "n_modules": len(ratios),
            "init": "A ~ N(0, 1/in), B ~ N(0, 1) 再缩放使 ‖scaling·B·A‖_F = rel_scale·‖W‖_F（逐模块；CPU 生成器，"
                    "按 lora_target_modules() 的顺序）",
            "realized_rel_frobenius": q(ratios), "A_std": q(a_std), "B_std": q(b_std)}


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------


def gap_summary(gaps: Sequence[float | None], threshold: float = MAX_GAP) -> dict[str, Any]:
    v = np.asarray([g for g in gaps if g is not None], dtype=float)
    if not v.size:
        return {"n": 0, "median": None, "max": None, "frac_lt_threshold": None, "threshold": threshold}
    return {"n": int(v.size), "median": float(np.median(v)), "max": float(v.max()),
            "frac_lt_threshold": float((v < threshold).mean()), "threshold": threshold}


def position_summary(divs: Sequence[int | None], max_new_tokens: int, width: int = 16) -> dict[str, Any]:
    v = [int(d) for d in divs if d is not None]
    hist = {f"{lo}-{min(lo + width, max_new_tokens) - 1}": 0 for lo in range(0, max_new_tokens, width)}
    for d in v:
        lo = min(d, max_new_tokens - 1) // width * width
        hist[f"{lo}-{min(lo + width, max_new_tokens) - 1}"] += 1
    if not v:
        return {"n": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None, "histogram": hist}
    arr = np.asarray(v, dtype=float)
    return {"n": len(v), "min": int(arr.min()), "p25": float(np.percentile(arr, 25)), "median": float(np.median(arr)),
            "p75": float(np.percentile(arr, 75)), "max": int(arr.max()), "histogram": hist}


class GenAgg:
    """生成序列的逐行比较，按思考开关累计（generation_agreement 的计数相加，分歧行逐行记下）。"""

    def __init__(self, max_new_tokens: int, names: tuple[str, str]):
        self.max_new_tokens, self.names = max_new_tokens, names
        self.n_rows = self.n_exact = self.n_agree = self.n_tok = 0
        self.divs: list[int | None] = []
        self.gaps: list[float | None] = []
        self.top2: list[float | None] = []
        self.rows: list[dict[str, Any]] = []

    def add(self, rep: dict[str, Any], ids: Sequence[str], thinking: bool) -> None:
        self.n_rows += rep["n_rows"]
        self.n_agree += rep["n_agree_tokens"]
        self.n_tok += rep["n_tokens"]
        for r, d in enumerate(rep["first_divergence"]):
            self.divs.append(d)
            if d is None:
                self.n_exact += 1
                continue
            g, t2 = rep["divergence_gaps"][r], rep["divergence_top2_gaps"][r]
            self.gaps.append(g)
            self.top2.append(t2)
            la, lb = rep["lengths"][r]
            self.rows.append({"id": ids[r], "thinking": thinking, "first_divergence": d, f"len_{self.names[0]}": la,
                              f"len_{self.names[1]}": lb, "gap": g, "top2_gap": t2})

    def summary(self) -> dict[str, Any]:
        return {"n_rows": self.n_rows, "exact_rows": self.n_exact / self.n_rows if self.n_rows else None,
                "token_agreement": self.n_agree / self.n_tok if self.n_tok else 1.0, "n_agree_tokens": self.n_agree,
                "n_tokens": self.n_tok, "n_divergent_rows": self.n_rows - self.n_exact,
                "first_divergence": position_summary(self.divs, self.max_new_tokens),
                "n_divergence_without_gap": sum(g is None for g in self.gaps),
                "gap": gap_summary(self.gaps), "top2_gap": gap_summary(self.top2),
                "divergent_rows": self.rows[:MAX_LISTED], "n_listed": min(len(self.rows), MAX_LISTED)}


class TfAgg:
    """教师强制的逐位置比较，按思考开关累计。"""

    def __init__(self, names: tuple[str, str], agreement_key: str):
        self.names, self.key = names, agreement_key
        self.n_tok = self.n_bad = 0
        self.max_abs_logit = self.max_abs_hidden = 0.0
        self.has_abs = False
        self.gaps: list[float] = []
        self.top2: list[float] = []
        self.rows: list[dict[str, Any]] = []

    def add(self, n_tokens: int, disagreements: Sequence[tuple], ids: Sequence[str], thinking: bool,
            max_abs_logit: float | None = None, max_abs_hidden: float | None = None) -> None:
        self.n_tok += n_tokens
        self.n_bad += len(disagreements)
        if max_abs_logit is not None:
            self.has_abs = True
            self.max_abs_logit = max(self.max_abs_logit, max_abs_logit)
            self.max_abs_hidden = max(self.max_abs_hidden, max_abs_hidden or 0.0)
        for row, pos, x, y, g, t2 in disagreements:
            self.gaps.append(g)
            self.top2.append(t2)
            if len(self.rows) < MAX_LISTED:
                self.rows.append({"id": ids[row], "thinking": thinking, "pos": pos, self.names[0]: x, self.names[1]: y,
                                  "gap": g, "top2_gap": t2})

    def summary(self) -> dict[str, Any]:
        out = {"n_tokens": self.n_tok, self.key: 1.0 - self.n_bad / self.n_tok if self.n_tok else 1.0,
               "n_disagreements": self.n_bad, "gap": gap_summary(self.gaps), "top2_gap": gap_summary(self.top2),
               "disagreements": self.rows, "n_listed": len(self.rows)}
        if self.has_abs:
            out.update(max_abs_logit=self.max_abs_logit, max_abs_hidden=self.max_abs_hidden)
        return out


# ---------------------------------------------------------------------------
# 两项检查
# ---------------------------------------------------------------------------


def run_restricted(ad: Any, runs, allowed: dict[str, Any], max_new_tokens: int) -> dict[str, Any]:
    gens = {t: GenAgg(max_new_tokens, ("generate_batch", "restricted")) for t in THINKING}
    tfs = {t: TfAgg(("decoded", "argmax"), "agreement") for t in THINKING}
    for inputs, thinking, ids in runs:
        rep = training.restricted_consistency(ad, inputs, allowed["ids"], max_new_tokens=max_new_tokens,
                                              stop_ids=allowed["stop_ids"], enable_thinking=thinking)
        gens[thinking].add(rep["decode_vs_generate_batch"], ids, thinking)
        tf = rep["teacher_forcing"]
        dis = [(b, p, x, y, g, t2) for (b, p, x, y, g), t2 in zip(tf["disagreements"], tf["disagreement_top2_gaps"])]
        tfs[thinking].add(tf["n_tokens"], dis, ids, thinking)
    return {"on_model": "checkpoint (load_trained merge=True)" if getattr(ad, "lora_state", "none") == "merged" else "base",
            "by_thinking": {MODE_KEYS[t]: {"n_rows": gens[t].n_rows, "decode_vs_generate_batch": gens[t].summary(),
                                           "teacher_forcing": tfs[t].summary()} for t in THINKING}}


def _side_runs(runs) -> list[tuple[list[Any], bool]]:
    return [(inputs, thinking) for inputs, thinking, _ in runs]


def merge_by_thinking(reps: Sequence[dict[str, Any]], runs, max_new_tokens: int) -> dict[str, Any]:
    gens = {t: GenAgg(max_new_tokens, ("before", "after")) for t in THINKING}
    tfs = {t: TfAgg(("before", "after"), "argmax_agreement") for t in THINKING}
    for rep, (_, thinking, ids) in zip(reps, runs):
        gens[thinking].add(rep["generate"], ids, thinking)
        tf = rep["teacher_forcing"]
        tfs[thinking].add(tf["n_tokens"], tf["disagreements"], ids, thinking, tf["max_abs_logit"], tf["max_abs_hidden"])
    return {MODE_KEYS[t]: {"n_rows": gens[t].n_rows, "generate": gens[t].summary(), "teacher_forcing": tfs[t].summary()}
            for t in THINKING}


def effect_by_thinking(base: CK.MergeSide, before: CK.MergeSide, tf_reps: Sequence[dict[str, Any]], runs) -> dict[str, Any]:
    """随机 LoRA 让输出偏离 base 多少：生成（base 与合并前逐行比较）与教师强制（同以 base 的序列为目标）。"""
    out = {}
    for t in THINKING:
        ks = [k for k, (_, thinking, _) in enumerate(runs) if thinking == t]
        g = [training.generation_agreement(base.generations[k], before.generations[k]) for k in ks]
        n_rows = sum(x["n_rows"] for x in g)
        tf = [tf_reps[k]["teacher_forcing"] for k in ks]
        n_tok = sum(x["n_tokens"] for x in tf)
        out[MODE_KEYS[t]] = {
            "generate": {"n_rows": n_rows, "exact_rows": sum(x["exact_rows"] * x["n_rows"] for x in g) / n_rows,
                         "token_agreement": sum(x["n_agree_tokens"] for x in g) / max(1, sum(x["n_tokens"] for x in g))},
            "teacher_forcing": {"n_tokens": n_tok,
                                "argmax_agreement": 1.0 - sum(len(x["disagreements"]) for x in tf) / n_tok if n_tok else 1.0,
                                "max_abs_logit": max(x["max_abs_logit"] for x in tf),
                                "max_abs_hidden": max(x["max_abs_hidden"] for x in tf)}}
    return out


def run_merge_random(ad: Any, runs, max_new_tokens: int, *, rel_scale: float, seed: int,
                     timing: dict[str, float]) -> dict[str, Any]:
    t0 = time.perf_counter()
    side_runs = _side_runs(runs)
    base = CK.merge_side(ad, side_runs, max_new_tokens=max_new_tokens)
    ad.inject_lora(r=LORA_RANK, lora_alpha=LORA_ALPHA, init_lora_weights=False)
    lora = randomize_lora(ad, rel_scale=rel_scale, seed=seed)
    lora.update(r=LORA_RANK, lora_alpha=LORA_ALPHA, n_parameters=ad.lora_parameter_count(),
                targets_sha256=hashlib.sha256("\n".join(sorted(ad.lora_target_modules())).encode()).hexdigest())
    before = CK.merge_side(ad, side_runs, max_new_tokens=max_new_tokens)
    lora_on_base = CK.merge_side(ad, side_runs, max_new_tokens=max_new_tokens, targets=base.targets, generate=False)
    effect = effect_by_thinking(base, before, CK.compare_merge_sides(ad, base, lora_on_base, chunk_size=TF_CHUNK), runs)
    del base, lora_on_base
    timing["merge_before_s"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    ad.merge_lora()
    after = CK.merge_side(ad, side_runs, max_new_tokens=max_new_tokens, targets=before.targets)
    reps = CK.compare_merge_sides(ad, before, after, chunk_size=TF_CHUNK)
    timing["merge_after_s"] = time.perf_counter() - t1
    return {"mode": "random_lora", "lora": lora, "effect_vs_base": effect,
            "by_thinking": merge_by_thinking(reps, runs, max_new_tokens)}


def _release_memory() -> None:
    """调用方先 del 掉引用，再回收（换检查点的第二次加载之前，显存里只留合并前那一侧的隐藏状态）。"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 判据
# ---------------------------------------------------------------------------


def criteria_block(exact: bool) -> dict[str, Any]:
    return {
        "status": "提议值（D42-4）：RUN 实测后由管理员冻结",
        "exact": exact,
        "exact_reason": "CPU fp32：必须逐个相同" if exact else "GPU / bf16：一致率（D21-2）",
        "per_thinking_mode": True,
        "min_agreement": 1.0 if exact else MIN_AGREEMENT,
        "max_gap": None if exact else MAX_GAP,
        "gap_definition": "不一致位置上两种实现所选两个 token 在参照 logits 上之差（见脚本说明）；top2_gap 只报告不判",
        "restricted": ["decode_vs_generate_batch.token_agreement ≥ min_agreement",
                       "teacher_forcing.agreement ≥ min_agreement",
                       "decode_vs_generate_batch.gap.max < max_gap 且 teacher_forcing.gap.max < max_gap"
                       + "（exact 时：exact_rows = 1 且没有不一致的位置）"],
        "merge": ["generate.token_agreement ≥ min_agreement", "teacher_forcing.argmax_agreement ≥ min_agreement",
                  "generate.gap.max < max_gap 且 teacher_forcing.gap.max < max_gap（exact 时：exact_rows = 1 且没有不一致的位置）",
                  f"随机 LoRA 的前提：effect_vs_base.generate.exact_rows ≤ {EFFECT_MAX_EXACT_ROWS}，"
                  f"effect_vs_base.teacher_forcing.argmax_agreement ≤ {EFFECT_MAX_ARGMAX_AGREEMENT}"],
        "effect_max_exact_rows": EFFECT_MAX_EXACT_ROWS,
        "effect_max_argmax_agreement": EFFECT_MAX_ARGMAX_AGREEMENT,
    }


def _judge_pair(where: str, gen: dict[str, Any], tf: dict[str, Any], tf_key: str, crit: dict[str, Any]) -> list[str]:
    bad = []
    if crit["exact"]:
        if gen["exact_rows"] != 1.0 or gen["token_agreement"] != 1.0:
            bad.append(f"{where}：生成不逐个相同（exact_rows {gen['exact_rows']}，token_agreement {gen['token_agreement']}）")
        if tf[tf_key] != 1.0 or tf["n_disagreements"]:
            bad.append(f"{where}：教师强制有 {tf['n_disagreements']} 个位置不同")
        return bad
    if gen["token_agreement"] < crit["min_agreement"]:
        bad.append(f"{where}：生成的 token_agreement {gen['token_agreement']:.4f} < {crit['min_agreement']}")
    if tf[tf_key] < crit["min_agreement"]:
        bad.append(f"{where}：教师强制的 {tf_key} {tf[tf_key]:.4f} < {crit['min_agreement']}")
    if gen["n_divergence_without_gap"]:
        bad.append(f"{where}：{gen['n_divergence_without_gap']} 行的分歧算不出 logit 差")
    for name, s in (("生成分歧处", gen["gap"]), ("教师强制不一致处", tf["gap"])):
        if s["n"] and s["max"] >= crit["max_gap"]:
            bad.append(f"{where}：{name}的 logit 差最大 {s['max']:.4f} ≥ {crit['max_gap']}")
    return bad


def judge(checks: dict[str, Any], crit: dict[str, Any]) -> dict[str, list[str]]:
    out = {}
    if "restricted" in checks:
        out["restricted"] = [m for k, r in checks["restricted"]["by_thinking"].items()
                             for m in _judge_pair(f"restricted/{k}", r["decode_vs_generate_batch"], r["teacher_forcing"],
                                                  "agreement", crit)]
    if "merge" in checks:
        m = checks["merge"]
        bad = [x for k, r in m["by_thinking"].items()
               for x in _judge_pair(f"merge/{k}", r["generate"], r["teacher_forcing"], "argmax_agreement", crit)]
        for k, e in (m.get("effect_vs_base") or {}).items():
            if e["generate"]["exact_rows"] > EFFECT_MAX_EXACT_ROWS:
                bad.append(f"merge/{k}：随机 LoRA 太弱，{e['generate']['exact_rows']:.3f} 的行与 base 相同（> {EFFECT_MAX_EXACT_ROWS}）")
            if e["teacher_forcing"]["argmax_agreement"] > EFFECT_MAX_ARGMAX_AGREEMENT:
                bad.append(f"merge/{k}：随机 LoRA 太弱，教师强制 argmax 与 base 相同的比例 "
                           f"{e['teacher_forcing']['argmax_agreement']:.4f} > {EFFECT_MAX_ARGMAX_AGREEMENT}")
        out["merge"] = bad
    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def model_block(a, entry, ad) -> dict[str, Any]:
    from src.adapters.checkpoint import policy_spec

    return {"key": entry.key, "family": entry.family, "repo": entry.repo, "revision": entry.revision,
            "adapter_model_key": ad.model_key, "tiny": a.tiny, "device": a.device, "dtype": a.dtype,
            "image_policy": policy_spec(ad.image_policy), "tied_word_embeddings": bool(ad.tied_word_embeddings),
            "tiny_init_std": TINY_INIT_STD if a.tiny else None}


def build_document(a, argv) -> dict[str, Any]:
    return {"format": FORMAT, "argv": list(argv), "written_utc": None, "code": runlog.git_info(REPO_ROOT),
            "versions": runlog.package_versions(), "gpu": runlog.gpu_name() if a.device.startswith("cuda") else None,
            "model": None, "checkpoint": None, "data": None,
            "generation": {"max_new_tokens": a.max_new_tokens, "thinking_modes": list(THINKING), "decoding": "greedy",
                           "merge_generation": "全词表（不受限）", "restricted_generation": "allowed 集合"},
            "batch_protocol": None, "allowed": None, "criteria": None, "checks": {}, "failures": {}, "pass": None,
            "timing_s": {}, "error": None}


def write_json(path: Path, doc: dict[str, Any]) -> None:
    doc["written_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def make_base_adapter(a, entry) -> Any:
    kw = {"init_std": TINY_INIT_STD} if a.tiny else {}
    return factory.make_adapter(entry.key, device=a.device, dtype=DTYPES[a.dtype], tiny=a.tiny, **kw)


def load_checkpoint(a, merge: bool) -> tuple[Any, Any]:
    ad, ck = factory.load_trained(a.checkpoint, device=a.device, dtype=DTYPES[a.dtype], merge=merge)
    spec = ck.manifest.get("build_spec") or {}
    if spec.get("key") != a.model or bool(spec.get("tiny")) != a.tiny:
        raise ValueError(f"检查点的构造方式（key {spec.get('key')!r}，tiny {spec.get('tiny')}）与 --model {a.model!r}、"
                         f"--tiny {a.tiny} 不符")
    return ad, ck


def run(a, argv, doc: dict[str, Any]) -> None:
    prompts = load_prompt_config(a.prompts)
    bs, frozen = resolve_batch_size(a, prompts)
    bp = prompts.batch_protocol()
    entry = registry.get_model(a.model)
    paths = data_paths(a)
    samples, texts, dstats = load_dev(paths, prompts, entry.family, a.n)
    plan = batch_plan([None] * len(samples), bs)
    doc["data"] = dstats
    doc["batch_protocol"] = {"batch_size": bs, "frozen": frozen, "frozen_batch_size": int(bp["batch_size"]),
                             "padding_side": bp["padding_side"], "order": bp["order"], "protocol_version": bp["version"],
                             "n_batches": len(plan), "batches": [list(x) for x in plan],
                             "source": "configs/e1_prompts.json batch_protocol（src/e1/runner.batch_plan）"}
    exact = torch.device(a.device).type == "cpu" and DTYPES[a.dtype] == torch.float32
    doc["criteria"] = criteria_block(exact)
    ids = [s.id for s in samples]
    timing = doc["timing_s"]
    t0 = time.perf_counter()
    if a.checkpoint:
        man = CK.read_manifest(a.checkpoint)
        doc["checkpoint"] = {"dir": str(a.checkpoint), "manifest_sha256": runlog.sha256_file(Path(a.checkpoint) / CK.MANIFEST),
                             "lora": man.get("lora"), "query": man.get("query"), "dtype": man.get("dtype"),
                             "build_spec": man.get("build_spec"), "trainable_token_rows": man.get("trainable_token_rows")}
        ad = None
        if "merge" in a.checks:
            ad0, _ = load_checkpoint(a, merge=False)
            doc["model"] = model_block(a, entry, ad0)
            doc["allowed"] = grid_allowed(ad0)
            runs0 = make_runs(build_batches(ad0, samples, texts, prompts.order, plan), plan, ids)
            before = CK.merge_side(ad0, _side_runs(runs0), max_new_tokens=a.max_new_tokens)
            table0 = ad0.token_table()
            timing["merge_before_s"] = time.perf_counter() - t0
            del ad0, runs0
            _release_memory()
            t1 = time.perf_counter()
            ad, _ = load_checkpoint(a, merge=True)
            if ad.token_table() != table0:
                raise RuntimeError("load_trained(merge=True) 与 merge=False 的 token 表不同")
            runs = make_runs(build_batches(ad, samples, texts, prompts.order, plan), plan, ids)
            after = CK.merge_side(ad, _side_runs(runs), max_new_tokens=a.max_new_tokens, targets=before.targets)
            reps = CK.compare_merge_sides(ad, before, after, chunk_size=TF_CHUNK)
            doc["checks"]["merge"] = {"mode": "checkpoint", "compared": "factory.load_trained(merge=False) vs (merge=True)",
                                      "by_thinking": merge_by_thinking(reps, runs, a.max_new_tokens)}
            timing["merge_after_s"] = time.perf_counter() - t1
            del before, after, reps
        if "restricted" in a.checks:
            t2 = time.perf_counter()
            if ad is None:
                ad, _ = load_checkpoint(a, merge=True)
                doc["model"] = model_block(a, entry, ad)
                doc["allowed"] = grid_allowed(ad)
                runs = make_runs(build_batches(ad, samples, texts, prompts.order, plan), plan, ids)
            doc["checks"]["restricted"] = run_restricted(ad, runs, doc["allowed"], a.max_new_tokens)
            timing["restricted_s"] = time.perf_counter() - t2
    else:
        ad = make_base_adapter(a, entry)
        doc["model"] = model_block(a, entry, ad)
        doc["allowed"] = grid_allowed(ad)
        runs = make_runs(build_batches(ad, samples, texts, prompts.order, plan), plan, ids)
        timing["load_and_inputs_s"] = time.perf_counter() - t0
        if "restricted" in a.checks:
            t2 = time.perf_counter()
            doc["checks"]["restricted"] = run_restricted(ad, runs, doc["allowed"], a.max_new_tokens)
            timing["restricted_s"] = time.perf_counter() - t2
        if "merge" in a.checks:
            doc["checks"]["merge"] = run_merge_random(ad, runs, a.max_new_tokens, rel_scale=a.lora_rel_scale,
                                                      seed=a.lora_seed, timing=timing)
    timing["total_s"] = time.perf_counter() - t0
    doc["failures"] = judge(doc["checks"], doc["criteria"])
    for name, bad in doc["failures"].items():
        doc["checks"][name]["pass"] = not bad
    doc["pass"] = all(not bad for bad in doc["failures"].values())


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    a = parse_args(argv)
    out = Path(a.out)
    doc = build_document(a, argv)
    try:
        run(a, argv, doc)
    except Exception as e:  # noqa: BLE001 — 证据先落盘，再以非零码退出
        doc["error"] = {"repr": repr(e), "traceback": traceback.format_exc()[-6000:]}
        doc["pass"] = False
        write_json(out, doc)
        print(f"失败：{e!r}（已写出 {out}）", file=sys.stderr)
        return 2
    write_json(out, doc)
    for name in a.checks:
        bad = doc["failures"].get(name, [])
        print(f"[{a.model} {name}] pass={not bad}" + ("" if not bad else f"：{'; '.join(bad[:4])}"))
    print(f"pass={doc['pass']} -> {out}")
    return 0 if doc["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
