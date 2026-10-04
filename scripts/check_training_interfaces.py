#!/usr/bin/env python
"""训练接口的核对仪器（DEV-0021；D21-2、D42-4；简报 §12）：受限 logits 的训练与推理一致率、LoRA 合并前后的一致率。
既是 E3 开工前的 GPU 核对，也是以后每个训练出来的模型评测前的标准合并核对（--checkpoint）。输出一个 JSON。

服务器：
    python scripts/check_training_interfaces.py --model qwen3.5-0.8b --checks restricted merge --n 200 \\
        --data-root $PROJ/data --out <json>
    # 评测前的合并核对（训练出来的检查点；--model 必须与检查点 manifest 的构造方式相同；带网格 token 的检查点 restricted 用
    # 它自己的网格 token 与 E3b 单轮评测的设置，见下）：
    python scripts/check_training_interfaces.py --model qwen3.5-0.8b --checks merge restricted --n 200 \\
        --data-root $PROJ/data --checkpoint <检查点目录> --out <json> [--e3b-config configs/e3b_train.json]
开发机彩排（小型随机模型、CPU fp32、tests/fixtures/e1 的小数据；这时判据是逐个相同）：
    python scripts/check_training_interfaces.py --tiny --model gemma-4-e2b-it --checks restricted merge --out /tmp/x.json

输入（只读 src/e1，不改 E1 的配置与代码）：E1 grid16 任务的开发子集（三个 RefCOCO 开发子集，src/e1/tasks.select），
按 sha256 排序取 --n 条（src/data/subsets.seeded_sample，用途串 SELECT_PURPOSE），保持选择顺序；提示是 configs/e1_prompts.json
里 grid16 本家族的原文（PromptConfig.render）。两种思考开关各跑一遍，max_new_tokens = MAX_NEW_TOKENS；批配置照 E1 冻结的
batch_protocol（批大小读配置，左填充，选择顺序，连续样本成批；src/e1/runner.batch_plan）。

allowed 集合（受限解码与教师强制共用）：E1 数字网格格式实际用到的 token —— 用本家族的 tokenizer 编码一张 16×16 的示例网格
（src/grid/codec.grid_to_text），编码结果必须恰好由 "0"、"1"、"\\n" 三个单 token 组成（这三个 token 的构造只在
src/e1/forced.grid_format_ids 写一份，本脚本 import 它，D68-4(b)），再加本家族的停止 token。不用新增的
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

带网格 token 的检查点（DEV-0032；D56-5、D65-4(a)）：给了 --checkpoint、检查点又登记了网格 token 时，restricted 核对改用检查点自己的
网格 token 与 E3b 单轮评测的设置（scripts/eval_e3b.py 的第 1 轮，src/e3/evaluate.make_context），merge 不变：
- 样本：同 data（E1 grid16 的开发子集 = 三个 RefCOCO 开发子集，按 sha256 取 --n 条），即分割任务（GRID_TASK）的开发子集；
- 提示：E3b 配置（--e3b-config，默认 configs/e3b_train.json）的 tasks.refseg.grid_template，指令在前，图像用 src/e3/data.load_rgb；
- allowed 集合：tokens.attach_vocab(检查点, 词表).allowed_ids(refseg, allowed_scope)，与 eval_e3b 的单轮生成同一个集合（含停止 token）；
  停止 token = 本家族的回合结束 token；max_new_tokens = 目标序列长度 n(n + 1) + 1（32×32 时 1057；--max-new-tokens 不影响它）；思考开关
  只取 E3b 评测用的那一个（配置的 enable_thinking）。检查点的网格 token 与词表不同就报错（没有定义好的 allowed 集合）。
- restricted.by_thinking：在 load_trained(merge=True)（评测用的模型）上调 training.restricted_consistency，读数与 v1 相同；
- restricted.across_merge：训练时的受限 logits 是在未合并的 LoRA 上算的，评测在合并后的模型上解码。以合并后受限解码的序列为目标，
  在 load_trained(merge=False) 上再做一遍教师强制，两侧只在 allowed 集合上的 logits 逐位置比较（checkpoint.compare_merge_sides）。
  "合并前后网格 token 的 logits 分家"靠这一项查出：合并后的模型内部自洽，by_thinking 看不出来；merge 用 E1 的文本提示、全词表，
  网格 token 从来不是 argmax，也看不出来。两侧不同时驻留：合并后的模型用完释放，再加载未合并的；合并后那一侧的目标隐藏状态留在设备上。
- 判据：by_thinking 照 v1 的 restricted 一节（阈值不变）；across_merge 是合并前后的比较，数值噪声与 merge 相同，照 v1 的 merge 一节的
  教师强制两条判（criteria.restricted_grid；CPU fp32 时逐个相同）。
不带网格 token 的检查点（例如 C3）与不给 --checkpoint 的 base：行为不变（tests 拿 69884b8 的脚本存档逐字节比较输出）。

不一致位置的 logit 差（gap，判据用它）：两种实现在某个位置选了不同的 token x ≠ y，gap 是参照 logits 上这两个 token 之差：
- restricted 的教师强制：L = 教师强制 logits，gap = L[argmax] − L[解码 token]；
- 生成序列第一个分歧处（restricted 的两种解码、merge 的合并前后）：分歧之前两条序列相同，教师强制 logits 在该位置对两者都有效，
  gap = |L[x] − L[y]|（merge 取合并前后两侧的较大者）；
- merge 的教师强制：gap = max(L_前[x] − L_前[y], L_后[y] − L_后[x])。
另报 top2_gap = 该位置参照 logits 的前两名之差。gap ≥ top2_gap，对方的 token 恰是第二名时两者相同；对方的 token 连第二名都
不是时只有 gap 会暴露，所以判据用 gap。

判据 v1（D50-3 冻结，依据 RUN-0021 实测；目的是查出实现错误，数值噪声另记为噪声底；版本号 CRITERIA_VERSION 写进输出的
criteria_version 与 criteria.version）：每种思考开关分别判，
- GPU / bf16：
  - restricted：教师强制 argmax 一致率 teacher_forcing.agreement ≥ RESTRICTED_MIN_TF_AGREEMENT（两种开关）；关思考时受限解码与
    generate_batch 的 exact_rows ≥ RESTRICTED_MIN_EXACT_ROWS_OFF；不一致处的 logit 差最大值 < RESTRICTED_MAX_GAP（教师强制的不一致处
    与两种解码的第一个分歧处都算；分歧算不出 gap 也判不过）。开思考时的 exact_rows、token_agreement 只报不判（自由生成在近似平局处
    分岔，之后逐位不同）。
  - merge：教师强制 argmax 一致率 ≥ MERGE_MIN_TF_AGREEMENT；不一致处 logit 差的中位数 ≤ MERGE_MAX_MEDIAN_GAP（教师强制的不一致处）；
    生成的逐位一致率、exact_rows、gap 与教师强制 gap 的最大值只报不判。
  - 前提（随机 LoRA 才有）：只在开思考时判，effect_vs_base.generate.exact_rows ≤ EFFECT_MAX_EXACT_ROWS；关思考时只报不判
    （输出被限制在几个 token 上，基本不变）；教师强制 argmax 与 base 的一致率只报不判。
- CPU fp32（--tiny）：两项检查都必须逐个相同（exact_rows = token_agreement = agreement = 1，没有不一致的位置）；前提同上。
已有的输出可以用 rejudge_document 按当前判据重判（tests 用 RUN-0021 的 4 份 JSON 钉住：4/4 通过）。

输出 JSON 的字段见 build_document（format、argv、written_utc、code、versions、gpu、model、checkpoint、data、generation、
batch_protocol、allowed、criteria_version、criteria、checks、failures、pass、timing_s、error）；带网格 token 的检查点做 restricted 时
另有 restricted_inputs（E3b 单轮评测的设置），allowed 换成网格 token 的集合，criteria 另有 restricted_grid。
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
from typing import Any, Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src import runlog  # noqa: E402
from src.adapters import checkpoint as CK  # noqa: E402
from src.adapters import factory, registry, training  # noqa: E402
from src.data.subsets import seeded_sample  # noqa: E402
from src.e1.forced import grid_format_ids  # noqa: E402
from src.e1.prompts import PROMPTS_PATH, load_prompt_config  # noqa: E402
from src.e1.runner import batch_plan, ids_digest, load_images  # noqa: E402
from src.e1.tasks import DataPaths, select  # noqa: E402

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
# 判据 v1（D50-3 冻结；规则见模块说明）
CRITERIA_VERSION = "v1"
CRITERIA_SOURCE = "D50-3（依据 RUN-0021 实测冻结，2026-10-02）"
RESTRICTED_MIN_TF_AGREEMENT = 0.999  # restricted：教师强制 argmax 一致率（两种开关）
RESTRICTED_MIN_EXACT_ROWS_OFF = 0.95  # restricted：关思考时受限解码与 generate_batch 逐行相同的比例
RESTRICTED_MAX_GAP = 2.0  # restricted：不一致处 logit 差的最大值上限（超过就逐条查看）
MERGE_MIN_TF_AGREEMENT = 0.98  # merge：教师强制 argmax 一致率
MERGE_MAX_MEDIAN_GAP = 0.5  # merge：不一致处 logit 差的中位数上限
EFFECT_MAX_EXACT_ROWS = 0.5  # 随机 LoRA 的前提（只在开思考时判）：与 base 逐行相同的生成至多一半
PREMISE_MODES = ("thinking_on",)  # 前提只在这些思考开关下判
MAX_LISTED = 500  # JSON 里逐项列出的不一致位置 / 分歧行的上限（统计量用全部）
TF_CHUNK = 256  # 全词表 logits 比较的分块
GRID_TASK = "refseg"  # 带网格 token 的检查点：restricted 用 E3b 单轮评测的分割任务（样本 = data 的开发子集）


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
    p.add_argument("--e3b-config", help="带网格 token 的检查点做 restricted 时读的 E3b 配置（词表、allowed_scope、网格指令、思考开关；"
                                        "默认 configs/e3b_train.json）")
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
    """E1 数字网格格式实际用到的 token（见模块说明）+ 本家族的停止 token。0、1、换行三个 token 的构造只在
    src/e1/forced.grid_format_ids 写一份（D68-4(b)），这里 import 它，再加停止 token；输出与原来的构造逐字节相同（tests 钉住）。"""
    tok = ad.tokenizer
    fmt = grid_format_ids(tok)
    rows = [{"role": r["role"], "text": r["text"], "id": int(r["id"]), "token": r["token"]} for r in fmt["tokens"]]
    probe = fmt["probe"]
    stops = [int(s) for s in ad.stop_token_ids()]
    for s in stops:
        rows.append({"role": "stop", "text": None, "id": s, "token": tok.convert_ids_to_tokens(s)})
    ids = sorted({r["id"] for r in rows})
    if len(ids) != len(rows):
        raise RuntimeError(f"allowed 集合里有重复的 token：{rows}")
    if set(ids) & set(ad.grid_token_ids.values()):
        raise RuntimeError("allowed 集合不能含新增的 grid token")
    return {"ids": ids, "stop_ids": stops, "tokens": rows,
            "probe": {"grid_size": int(probe["grid_size"]), "n_tokens": int(probe["n_tokens"]),
                      "distinct_ids": [int(i) for i in probe["distinct_ids"]]},
            "rule": "tokenizer 编码 16×16 示例网格（grid_to_text）必须恰好由 0、1、换行三个单 token 组成，再加本家族的停止 token；"
                    "不用新增的 grid token（均值初始化，logits 全同）"}


def build_batches(ad: Any, samples: Sequence[Any], texts: Sequence[str], order: str,
                  plan: Sequence[tuple[int, int]], images: Callable[[Any], list[Any]] = load_images) -> list[list[Any]]:
    """每批的 ModelInputs（两种思考开关、两项检查共用）。注入了查询 token 的检查点：先取消查询再建输入（生成不收查询 token），
    建完按原样恢复登记（不碰任何行）。images：样本 → 图像列表（默认 E1 的 load_images；网格的 restricted 用 E3b 评测的读法）。"""
    q = ad.token_table()["query"]
    n_q, mode = q.get("n"), q.get("position_mode")
    if n_q is not None:
        ad.clear_query_tokens()
    try:
        return [[ad.build_inputs(images(samples[i]), texts[i], order) for i in range(lo, hi)] for lo, hi in plan]
    finally:
        if n_q is not None:
            ad.inject_query_tokens(int(n_q), mode)


def make_runs(batches: Sequence[Sequence[Any]], plan: Sequence[tuple[int, int]],
              sample_ids: Sequence[str], modes: Sequence[bool] = THINKING) -> list[tuple[list[Any], bool, list[str]]]:
    """run = (一批输入, enable_thinking, 这批的样本 id)：按 modes 的顺序（默认先关思考的全部批，再开思考的全部批）。"""
    return [(list(b), thinking, list(sample_ids[lo:hi])) for thinking in modes for b, (lo, hi) in zip(batches, plan)]


# ---------------------------------------------------------------------------
# 带网格 token 的检查点：restricted 用 E3b 单轮评测的设置（模块说明）
# ---------------------------------------------------------------------------


class GridSetup:
    """检查点自己的网格 token 与 E3b 单轮评测的设置：cfg（E3Config）、tokens（GridTokens）、ctx（evaluate.TaskContext，分割）。
    （不用 dataclass：本脚本由 src/e3/merge_check.load_cti 按文件路径载入，不在 sys.modules 里。）"""

    def __init__(self, cfg: Any, tokens: Any, ctx: Any) -> None:
        self.cfg, self.tokens, self.ctx = cfg, tokens, ctx

    @property
    def allowed(self) -> list[int]:
        return [int(i) for i in self.ctx.allowed]

    @property
    def stop_ids(self) -> list[int]:
        return [int(self.tokens.stop_id)]

    @property
    def modes(self) -> tuple[bool, ...]:
        return (bool(self.ctx.enable_thinking),)

    @property
    def max_new_tokens(self) -> int:
        return int(self.ctx.max_new_tokens)

    def texts(self, samples: Sequence[Any]) -> list[str]:
        return [self.ctx.instruction(s, correction=False) for s in samples]

    def allowed_doc(self) -> dict[str, Any]:
        return {"ids": self.allowed, "stop_ids": self.stop_ids, "task": self.ctx.task, "scope": self.cfg.allowed_scope,
                "layout": self.tokens.layout(),
                "rule": "检查点自己的网格 token：tokens.attach_vocab(检查点, E3b 词表).allowed_ids(任务, allowed_scope)，与 eval_e3b 的"
                        "单轮生成同一个集合（含本家族的回合结束 token）"}

    def inputs_doc(self) -> dict[str, Any]:
        c = self.ctx
        return {"source": "E3b 单轮评测（src/e3/evaluate.make_context；scripts/eval_e3b.py 的第 1 轮）", "task": c.task,
                "samples": "同 data：E1 grid16 的开发子集（三个 RefCOCO 开发子集）按 sha256 取的样本，即分割任务的开发子集",
                "instruction_template": c.grid_template, "grid_size": c.n, "order": c.order,
                "images": "src/e3/data.load_rgb(sample.images[0])", "max_new_tokens": c.max_new_tokens,
                "thinking_modes": list(self.modes), "stop_ids": self.stop_ids, "allowed_scope": self.cfg.allowed_scope,
                "e3b_config": {"path": self.cfg.path, "sha256": self.cfg.sha256, "version": self.cfg.doc["version"]}}


def grid_setup(a, ad: Any) -> GridSetup:
    """检查点（合并后的适配器）的网格 token 对上 E3b 词表，建分割任务的单轮上下文。对不上就报错。"""
    from src.e3.config import CONFIG_PATH, load_config
    from src.e3.evaluate import make_context
    from src.e3.tokens import GridTokenError, attach_vocab

    cfg = load_config(a.e3b_config or CONFIG_PATH)
    try:
        tokens = attach_vocab(ad, cfg.vocab)
    except GridTokenError as e:
        raise RuntimeError(f"检查点登记了 {len(ad.grid_token_ids)} 个网格 token，与 E3b 词表（{cfg.path} 的 vocab）对不上：{e}；"
                           "restricted 核对没有定义好的 allowed 集合") from None
    return GridSetup(cfg=cfg, tokens=tokens, ctx=make_context(cfg, tokens, GRID_TASK))


def make_grid_runs(ad: Any, g: GridSetup, samples: Sequence[Any], plan: Sequence[tuple[int, int]],
                   sample_ids: Sequence[str]) -> list[tuple[list[Any], bool, list[str]]]:
    from src.e3.data import load_rgb

    batches = build_batches(ad, samples, g.texts(samples), g.ctx.order, plan, images=lambda s: [load_rgb(s.images[0])])
    return make_runs(batches, plan, sample_ids, modes=g.modes)


class RecordingAdapter:
    """restricted_consistency 用的适配器代理：每个调用照原样转给 ad，另按调用顺序记下 generate_restricted 的结果与 training_forward
    的输出（合并后那一侧：受限解码的序列当 across_merge 的目标，教师强制的目标隐藏状态当比较的一侧；不必再解码、再前向一遍）。"""

    def __init__(self, ad: Any) -> None:
        self._ad = ad
        self.restricted: list[list[Any]] = []
        self.forward: list[Any] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ad, name)

    def generate_restricted(self, *args: Any, **kw: Any) -> list[Any]:
        out = self._ad.generate_restricted(*args, **kw)
        self.restricted.append(list(out))
        return out

    def training_forward(self, *args: Any, **kw: Any) -> Any:
        out = self._ad.training_forward(*args, **kw)
        self.forward.append(out)
        return out


def run_restricted_grid(ad: Any, runs, g: GridSetup) -> tuple[dict[str, Any], CK.MergeSide]:
    """合并后的模型上的 restricted（v1 的读数）；返回 (读数, 合并后那一侧的 MergeSide：目标 = 受限解码的序列)。"""
    rec = RecordingAdapter(ad)
    out = run_restricted(rec, runs, {"ids": g.allowed, "stop_ids": g.stop_ids}, g.max_new_tokens, modes=g.modes)
    if len(rec.restricted) != len(runs) or len(rec.forward) != len(runs):
        raise RuntimeError(f"restricted_consistency 的调用次数不对：{len(rec.restricted)} 次受限解码、{len(rec.forward)} 次教师强制，"
                           f"应各 {len(runs)} 次")
    targets = [[list(r.token_ids) for r in res] for res in rec.restricted]
    for k, tf in enumerate(rec.forward):
        if [int(t) for t in tf.target_ids.tolist()] != [int(t) for row in targets[k] for t in row]:
            raise RuntimeError("教师强制的目标不是受限解码的序列")
    side = CK.MergeSide(generations=None, targets=targets, hidden=[tf.target_hidden.detach() for tf in rec.forward],
                        offsets=[tuple(tf.target_offsets) for tf in rec.forward])
    return out, side


def run_restricted_across_merge(ad0: Any, runs0, merged: CK.MergeSide, g: GridSetup) -> dict[str, Any]:
    """未合并的模型（ad0）上以合并后受限解码的序列为目标做教师强制，与合并后那一侧比较（只在 allowed 集合上）。"""
    unmerged = CK.merge_side(ad0, _side_runs(runs0), max_new_tokens=g.max_new_tokens, targets=merged.targets, generate=False)
    reps = CK.compare_merge_sides(ad0, merged, unmerged, allowed_ids=g.allowed, chunk_size=TF_CHUNK)
    tfs = {t: TfAgg(("merged", "unmerged"), "argmax_agreement") for t in g.modes}
    for rep, (_, thinking, ids) in zip(reps, runs0):
        tf = rep["teacher_forcing"]
        tfs[thinking].add(tf["n_tokens"], tf["disagreements"], ids, thinking, tf["max_abs_logit"], tf["max_abs_hidden"])
    return {"compared": "目标 = load_trained(merge=True) 上受限解码（generate_restricted）的序列；教师强制在 merge=True 与 merge=False "
                        "两侧各做一遍，只在 allowed 集合上比较 logits（checkpoint.compare_merge_sides）",
            "by_thinking": {MODE_KEYS[t]: {"n_rows": sum(len(x) for x, th, _ in runs0 if th == t),
                                           "teacher_forcing": tfs[t].summary()} for t in g.modes}}


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
    - "与 base 明显不同"由 effect_vs_base 实测、判据里作前提（EFFECT_MAX_EXACT_ROWS）。开发机的小模型（4 层、hidden 64）上标定过：
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


def gap_summary(gaps: Sequence[float | None], threshold: float = RESTRICTED_MAX_GAP) -> dict[str, Any]:
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


def run_restricted(ad: Any, runs, allowed: dict[str, Any], max_new_tokens: int,
                   modes: Sequence[bool] = THINKING) -> dict[str, Any]:
    gens = {t: GenAgg(max_new_tokens, ("generate_batch", "restricted")) for t in modes}
    tfs = {t: TfAgg(("decoded", "argmax"), "agreement") for t in modes}
    for inputs, thinking, ids in runs:
        rep = training.restricted_consistency(ad, inputs, allowed["ids"], max_new_tokens=max_new_tokens,
                                              stop_ids=allowed["stop_ids"], enable_thinking=thinking)
        gens[thinking].add(rep["decode_vs_generate_batch"], ids, thinking)
        tf = rep["teacher_forcing"]
        dis = [(b, p, x, y, g, t2) for (b, p, x, y, g), t2 in zip(tf["disagreements"], tf["disagreement_top2_gaps"])]
        tfs[thinking].add(tf["n_tokens"], dis, ids, thinking)
    return {"on_model": "checkpoint (load_trained merge=True)" if getattr(ad, "lora_state", "none") == "merged" else "base",
            "by_thinking": {MODE_KEYS[t]: {"n_rows": gens[t].n_rows, "decode_vs_generate_batch": gens[t].summary(),
                                           "teacher_forcing": tfs[t].summary()} for t in modes}}


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
    """判据 v1（D50-3；规则见模块说明）。exact（CPU fp32）时两项检查都要求逐个相同，前提照 v1。"""
    return {
        "version": CRITERIA_VERSION,
        "status": f"冻结：{CRITERIA_SOURCE}",
        "exact": exact,
        "exact_reason": "CPU fp32：两项检查都必须逐个相同" if exact else "GPU / bf16：判据 v1（查实现错误；数值噪声另记为噪声底）",
        "per_thinking_mode": True,
        "gap_definition": "不一致位置上两种实现所选两个 token 在参照 logits 上之差（见脚本说明）；top2_gap 只报告不判",
        "restricted": {
            "min_tf_agreement": RESTRICTED_MIN_TF_AGREEMENT,
            "min_exact_rows_thinking_off": RESTRICTED_MIN_EXACT_ROWS_OFF,
            "max_gap": RESTRICTED_MAX_GAP,
            "rules": [f"teacher_forcing.agreement ≥ {RESTRICTED_MIN_TF_AGREEMENT}（两种开关）",
                      f"thinking_off：decode_vs_generate_batch.exact_rows ≥ {RESTRICTED_MIN_EXACT_ROWS_OFF}",
                      f"teacher_forcing.gap.max < {RESTRICTED_MAX_GAP} 且 decode_vs_generate_batch.gap.max < {RESTRICTED_MAX_GAP}"
                      "（两种开关；分歧算不出 gap 也判不过）"],
            "report_only": ["decode_vs_generate_batch.token_agreement", "thinking_on 的 decode_vs_generate_batch.exact_rows",
                            "top2_gap"],
        },
        "merge": {
            "min_tf_agreement": MERGE_MIN_TF_AGREEMENT,
            "max_median_gap": MERGE_MAX_MEDIAN_GAP,
            "rules": [f"teacher_forcing.argmax_agreement ≥ {MERGE_MIN_TF_AGREEMENT}（两种开关）",
                      f"teacher_forcing.gap.median ≤ {MERGE_MAX_MEDIAN_GAP}（两种开关）"],
            "report_only": ["generate.*（逐位一致率 token_agreement、exact_rows、gap）", "teacher_forcing.gap.max", "top2_gap"],
        },
        "premise": {
            "max_exact_rows_vs_base": EFFECT_MAX_EXACT_ROWS,
            "judged_modes": list(PREMISE_MODES),
            "rules": [f"随机 LoRA：{'、'.join(PREMISE_MODES)} 时 effect_vs_base.generate.exact_rows ≤ {EFFECT_MAX_EXACT_ROWS}"],
            "report_only": ["thinking_off 的 effect_vs_base", "effect_vs_base.teacher_forcing.argmax_agreement"],
        },
        "exact_rules": ["CPU fp32：生成 exact_rows = token_agreement = 1，教师强制没有不一致的位置（restricted 与 merge 都是）"],
    }


def grid_criteria_block(exact: bool) -> dict[str, Any]:
    """带网格 token 的检查点做 restricted 时另附（criteria.restricted_grid；规则见模块说明）。阈值都是 v1（D50-3）已冻结的值。"""
    return {
        "applies_to": "--checkpoint 且检查点登记了网格 token：restricted 用检查点自己的网格 token 与 E3b 单轮评测的设置",
        "by_thinking": "照 v1 的 restricted 一节（criteria.restricted，阈值不变）；只有 E3b 评测用的思考开关（restricted_inputs.thinking_modes）",
        "across_merge": {
            "min_tf_agreement": MERGE_MIN_TF_AGREEMENT,
            "max_median_gap": MERGE_MAX_MEDIAN_GAP,
            "rules": [f"across_merge.teacher_forcing.argmax_agreement ≥ {MERGE_MIN_TF_AGREEMENT}",
                      f"across_merge.teacher_forcing.gap.median ≤ {MERGE_MAX_MEDIAN_GAP}"],
            "report_only": ["across_merge.teacher_forcing.gap.max", "top2_gap", "max_abs_logit", "max_abs_hidden"],
            "basis": "合并前后两侧的比较，数值噪声的来源与 merge 相同，所以照 v1 merge 一节的教师强制两条判（D50-3）",
        },
        "exact": exact,
        "exact_rules": ["CPU fp32：across_merge 的教师强制没有不一致的位置"],
    }


def _judge_exact(where: str, gen: dict[str, Any], tf: dict[str, Any], tf_key: str) -> list[str]:
    bad = []
    if gen["exact_rows"] != 1.0 or gen["token_agreement"] != 1.0:
        bad.append(f"{where}：生成不逐个相同（exact_rows {gen['exact_rows']}，token_agreement {gen['token_agreement']}）")
    if tf[tf_key] != 1.0 or tf["n_disagreements"]:
        bad.append(f"{where}：教师强制有 {tf['n_disagreements']} 个位置不同")
    return bad


def _judge_restricted(where: str, mode: str, gen: dict[str, Any], tf: dict[str, Any], crit: dict[str, Any]) -> list[str]:
    if crit["exact"]:
        return _judge_exact(where, gen, tf, "agreement")
    c, bad = crit["restricted"], []
    if tf["agreement"] < c["min_tf_agreement"]:
        bad.append(f"{where}：教师强制的 agreement {tf['agreement']:.5f} < {c['min_tf_agreement']}")
    if mode == MODE_KEYS[False] and gen["exact_rows"] < c["min_exact_rows_thinking_off"]:
        bad.append(f"{where}：关思考时受限解码与 generate_batch 的 exact_rows {gen['exact_rows']:.4f} < {c['min_exact_rows_thinking_off']}")
    if gen["n_divergence_without_gap"]:
        bad.append(f"{where}：{gen['n_divergence_without_gap']} 行的分歧算不出 logit 差")
    for name, s in (("解码分歧处", gen["gap"]), ("教师强制不一致处", tf["gap"])):
        if s["n"] and s["max"] >= c["max_gap"]:
            bad.append(f"{where}：{name}的 logit 差最大 {s['max']:.4f} ≥ {c['max_gap']}（逐条查看）")
    return bad


def _judge_merge(where: str, gen: dict[str, Any], tf: dict[str, Any], crit: dict[str, Any]) -> list[str]:
    if crit["exact"]:
        return _judge_exact(where, gen, tf, "argmax_agreement")
    c, bad = crit["merge"], []
    if tf["argmax_agreement"] < c["min_tf_agreement"]:
        bad.append(f"{where}：教师强制的 argmax_agreement {tf['argmax_agreement']:.5f} < {c['min_tf_agreement']}")
    s = tf["gap"]
    if s["n"] and s["median"] > c["max_median_gap"]:
        bad.append(f"{where}：教师强制不一致处 logit 差的中位数 {s['median']:.4f} > {c['max_median_gap']}")
    return bad


def _judge_across_merge(where: str, tf: dict[str, Any], crit: dict[str, Any]) -> list[str]:
    """restricted.across_merge（带网格 token 的检查点）：照 v1 merge 一节的教师强制两条判；CPU fp32 时逐个相同。"""
    if crit["exact"]:
        if tf["argmax_agreement"] != 1.0 or tf["n_disagreements"]:
            return [f"{where}：合并前后的受限教师强制有 {tf['n_disagreements']} 个位置不同"]
        return []
    c, bad = crit["merge"], []
    if tf["argmax_agreement"] < c["min_tf_agreement"]:
        bad.append(f"{where}：合并前后受限教师强制的 argmax_agreement {tf['argmax_agreement']:.5f} < {c['min_tf_agreement']}")
    s = tf["gap"]
    if s["n"] and s["median"] > c["max_median_gap"]:
        bad.append(f"{where}：合并前后受限教师强制不一致处 logit 差的中位数 {s['median']:.4f} > {c['max_median_gap']}")
    return bad


def judge(checks: dict[str, Any], crit: dict[str, Any]) -> dict[str, list[str]]:
    out = {}
    if "restricted" in checks:
        out["restricted"] = [m for k, r in checks["restricted"]["by_thinking"].items()
                             for m in _judge_restricted(f"restricted/{k}", k, r["decode_vs_generate_batch"],
                                                        r["teacher_forcing"], crit)]
        for k, r in ((checks["restricted"].get("across_merge") or {}).get("by_thinking") or {}).items():
            out["restricted"] += _judge_across_merge(f"restricted/across_merge/{k}", r["teacher_forcing"], crit)
    if "merge" in checks:
        m = checks["merge"]
        bad = [x for k, r in m["by_thinking"].items()
               for x in _judge_merge(f"merge/{k}", r["generate"], r["teacher_forcing"], crit)]
        p = crit["premise"]
        for k, e in (m.get("effect_vs_base") or {}).items():
            if k in p["judged_modes"] and e["generate"]["exact_rows"] > p["max_exact_rows_vs_base"]:
                bad.append(f"merge/{k}：随机 LoRA 太弱，{e['generate']['exact_rows']:.3f} 的行与 base 相同"
                           f"（> {p['max_exact_rows_vs_base']}）")
        out["merge"] = bad
    return out


def rejudge_document(doc: dict[str, Any]) -> dict[str, Any]:
    """已有的输出（例如 RUN-0021 按提议值判的 JSON）按当前判据重判：exact 取原文档的设置（CPU fp32 与否），checks 原样用。
    返回 {criteria_version, criteria, failures, pass}，不改 doc。"""
    if doc.get("format") != FORMAT:
        raise ValueError(f"不是本仪器的输出（format {doc.get('format')!r} ≠ {FORMAT!r}）")
    if doc.get("error"):
        raise ValueError("这份输出是出错时写的（error 非空），没有可判的读数")
    exact = bool((doc.get("criteria") or {}).get("exact", False))
    crit = criteria_block(exact)
    if "across_merge" in (doc["checks"].get("restricted") or {}):
        crit["restricted_grid"] = grid_criteria_block(exact)
    failures = judge(doc["checks"], crit)
    return {"criteria_version": CRITERIA_VERSION, "criteria": crit, "failures": failures,
            "pass": all(not bad for bad in failures.values())}


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
            "batch_protocol": None, "allowed": None, "criteria_version": CRITERIA_VERSION, "criteria": None, "checks": {},
            "failures": {}, "pass": None, "timing_s": {}, "error": None}


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
                if not ad.grid_token_ids:
                    doc["allowed"] = grid_allowed(ad)
                    runs = make_runs(build_batches(ad, samples, texts, prompts.order, plan), plan, ids)
            if ad.grid_token_ids:  # 带网格 token 的检查点：E3b 单轮评测的设置 + across_merge（模块说明）
                g = grid_setup(a, ad)
                doc["allowed"], doc["restricted_inputs"] = g.allowed_doc(), g.inputs_doc()
                doc["generation"]["restricted_generation"] = "allowed 集合；E3b 单轮评测的设置（见 restricted_inputs）"
                doc["criteria"]["restricted_grid"] = grid_criteria_block(exact)
                res, merged_side = run_restricted_grid(ad, make_grid_runs(ad, g, samples, plan, ids), g)
                table = ad.token_table()
                timing["restricted_s"] = time.perf_counter() - t2
                ad = runs = None  # 合并后的模型用完释放，再加载未合并的（两侧不同时驻留）
                _release_memory()
                t3 = time.perf_counter()
                ad0, _ = load_checkpoint(a, merge=False)
                if ad0.token_table() != table:
                    raise RuntimeError("load_trained(merge=False) 与 merge=True 的 token 表不同")
                res["across_merge"] = run_restricted_across_merge(ad0, make_grid_runs(ad0, g, samples, plan, ids),
                                                                  merged_side, g)
                doc["checks"]["restricted"] = res
                timing["restricted_across_merge_s"] = time.perf_counter() - t3
                del ad0, merged_side
            else:
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
