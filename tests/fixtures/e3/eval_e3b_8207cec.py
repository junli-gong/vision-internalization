#!/usr/bin/env python
"""E3b 像素 token 通道的评测入口（DEV-0030 §1；简报 §8.5、§4 第 11 条、§12）：一个检查点 → 单轮、叠加反馈多轮（K = 1、2、4，
附计算量）与匹配计算量的单轮，分割（RefCOCO val）与深度（NYU 官方 654）。规则见 src/e3/evaluate.py。

服务器（B200；可比的评测一律在 B200 上做，D55、D59-10）：
    python scripts/eval_e3b.py --checkpoint <训练叶目录>/ckpt/final/model --model qwen3.5-0.8b --variant e3b --seed 0 \\
        --data-root $PROJ/data --out experiments/E3/00NN_e3b_eval [--task refseg depth] [--resume] [--limit N]
    # 结果写进 <out>/eval/<变体>/<模型>/seed<k>/<任务>/；评测前的合并核对写进 <out>/eval/<变体>/<模型>/seed<k>/merge_check.json
    # 只跑单轮（DEV-0039，D72-4 (c)：多轮与匹配计算量每个尺寸只做 1 个种子）：加 --k-values 1（只有第 1 轮，不做任何采样）；
    # --k-values 必须是配置 eval.multi_turn.k_values（1 2 4）从 1 开始的前缀（1 或 1 2），否则用法错误。单轮的 predictions.jsonl 与
    # 单轮指标与不给时逐字节相同；config.yaml 的 eval.k_values 记生效的 K，覆盖时另记 eval.k_values_override。续跑时 K 不同拒绝混写。
    # 结构强制解码（DEV-0041 §2，D83）：加 --decode structured（每行格子只允许值 token、行末只允许行分隔、最后只允许停止 token；
    # 单轮、多轮与匹配计算量的采样都用）；结果写进 <任务>__structured/，运行身份与 metrics 记 decode。不给（或 restricted）时与原来逐字节相同。
    # 指令检查（DEV-0041 §1，只用于 refseg、只跑单轮）：加 --instruction-check none|swap [--limit N] [--normal-leaf <正常评测的 refseg 叶目录>]
    # （N 取批大小的整数倍；--task 默认就是 refseg，--k-values 默认就是 1）；结果写进 <任务>__instr_<种类>[__structured]/，正常评测一侧
    # 取已有结果里的同一批 id（默认同一 --out 里同一解码的 refseg/），不重跑。例：
    #   python scripts/eval_e3b.py --checkpoint <叶>/ckpt/final/model --model qwen3.5-2b --seed 0 --data-root $PROJ/data \\
    #       --out experiments/E3/00NN_e3b_instr --instruction-check swap --limit 2048 \\
    #       --normal-leaf experiments/E3/0038_e3_eval_2b_s0/eval/e3b/qwen3.5-2b/seed0/refseg
开发机彩排（小型随机模型、CPU fp32；检查点用 scripts/train_e3b.py --tiny 训出来的；分割用 tests/fixtures/e3/eval_root，
深度用 tests/fixtures/e1/root 的 NYU 测试图）：
    python scripts/train_e3b.py --tiny --model qwen3.5-0.8b --supervision grid --seed 0 --out /tmp/t
    python scripts/eval_e3b.py --tiny --checkpoint /tmp/t/e3b/qwen3.5-0.8b/seed0/ckpt/final/model --model qwen3.5-0.8b \\
        --seed 0 --out /tmp/e

顺序：读检查点 manifest（构造方式必须与 --model、--tiny 相符）→ 评测前的合并核对（scripts/check_training_interfaces.py --checkpoint
的同一套函数，n 取配置的 eval.merge_check.n，可用 --merge-check-n 改；--config 透传给它的 --e3b-config，带网格 token 的检查点做
restricted 时读同一份配置；不通过就不评测）→ factory.load_trained(merge=True) → 逐任务评测。
其他检查点（DEV-0035）：网格格式预热（scripts/train_e3b.py --supervision warmup，LoRA + 网格 token 行，与 E3b 同一种检查点）照常评，
--variant warmup；C1′（--supervision grid --variant c1prime，冻结 LLM、只有网格 token 的行、没有 LoRA）：合并核对跳过 merge 一项、
restricted 照做（含义见 src/e3/merge_check.py），--variant c1prime。批大小默认取配置冻结的 eval.batch_size（64，与 E1 相同）；真模型给别的值要加 --allow-unfrozen-batch。
退出码：0 = 跑完；1 = 合并核对没有通过，拒绝评测；2 = 出错（报错先写进 <out>/eval/<变体>/<模型>/seed<k>/errors/；argparse 用法
错误也是 2）。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from src.adapters import factory, registry  # noqa: E402
from src.adapters.checkpoint import read_manifest  # noqa: E402
from src.e3 import evaluate as EV  # noqa: E402
from src.e3.config import CONFIG_PATH, load_config  # noqa: E402
from src.e3.merge_check import MergeCheckFailed, run_merge_check  # noqa: E402
from src.e3.tokens import GRID_TASKS, attach_vocab  # noqa: E402
from src.runlog import sha256_file  # noqa: E402

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
RUN_DIR_NAME = re.compile(r"[0-9]{4}_[A-Za-z0-9][A-Za-z0-9_.-]*")
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="检查点目录（save_trained 写的：<训练叶目录>/ckpt/final/model）")
    p.add_argument("--model", required=True, help="注册键（必须与检查点 manifest 的构造方式相同）")
    p.add_argument("--variant", default="e3b", help="输出目录里的变体名（默认 e3b）")
    p.add_argument("--seed", type=int, required=True, help="训练的种子（输出目录 seed<k>）")
    p.add_argument("--task", nargs="+", choices=GRID_TASKS, help="默认 refseg depth（--instruction-check 时默认、也只能是 refseg）")
    p.add_argument("--out", required=True, help="本次运行的目录（experiments/E3/<NNNN_slug>）")
    p.add_argument("--resume", action="store_true", help="续跑：已完成的整批跳过")
    p.add_argument("--limit", type=int, help="只取每个评测集的前 N 条（调试）")
    p.add_argument("--tiny", action="store_true", help="小型随机模型 + CPU fp32 + tests/fixtures 的小数据（开发机彩排）")
    p.add_argument("--data-root", help="E1 的落地根（含 refcoco、nyuv2；默认环境变量 VISINT_DATA_ROOT；--tiny 时取 fixture）")
    p.add_argument("--dev-subsets", help="E1 的开发子集清单（默认 configs/e1_dev_subsets.json；--tiny 时取 fixture）")
    p.add_argument("--config", default=str(CONFIG_PATH), help="E3b 配置（默认 configs/e3b_train.json，读其中的 eval 节）")
    p.add_argument("--batch-size", type=int, help="批大小（默认取配置的 eval.batch_size；--tiny 取 eval.tiny.batch_size）")
    p.add_argument("--allow-unfrozen-batch", action="store_true", help="真模型允许配置以外的批大小（运行身份里照记）")
    p.add_argument("--k-values", type=int, nargs="+",
                   help="只跑这些 K（DEV-0039，D72-4 (c)）：必须是配置 eval.multi_turn.k_values 从 1 开始的前缀，如 1 或 1 2；"
                        "匹配计算量的单轮随之只做这些 K（1 = 只跑单轮、不采样）。不给时取配置的全部 K")
    p.add_argument("--decode", choices=EV.DECODE_MODES,
                   help="DEV-0041 §2：restricted（默认，generate_restricted）或 structured（结构强制，D83）；结果目录 <任务>__structured")
    p.add_argument("--instruction-check", choices=EV.INSTRUCTION_CHECKS,
                   help="DEV-0041 §1（只用于 refseg、只跑单轮）：none = 去掉指令（同 E3a 的 no_instruction）；swap = 换成同一张图里"
                        "另一个物体的指代句。结果目录 refseg__instr_<种类>")
    p.add_argument("--normal-leaf", help="--instruction-check 的正常评测一侧：已有的 refseg 叶目录（默认同一 --out 里同一解码的 refseg/）")
    p.add_argument("--merge-check-n", type=int, help="合并核对的开发样本数（默认取配置的 eval.merge_check.n）")
    p.add_argument("--score-workers", type=int, help="评分的进程数（默认取配置的 eval.score_workers；0 = 串行）")
    p.add_argument("--device", help="默认：真模型 cuda，--tiny 时 cpu")
    p.add_argument("--dtype", choices=sorted(DTYPES), help="默认：真模型 bf16，--tiny 时 fp32")
    a = p.parse_args(argv)
    if a.model not in registry.MODELS:
        p.error(f"未登记的模型键 {a.model!r}；已登记：{sorted(registry.MODELS)}")
    if not NAME.fullmatch(a.variant) or ".." in a.variant:
        p.error(f"--variant 只能含字母、数字、_ . -，收到 {a.variant!r}")
    for name in ("limit", "batch_size", "merge_check_n"):
        v = getattr(a, name)
        if v is not None and v < 1:
            p.error(f"--{name.replace('_', '-')} 必须是正整数")
    if a.score_workers is not None and a.score_workers < 0:
        p.error("--score-workers 不能为负")
    if a.instruction_check is not None:  # DEV-0041 §1：只用于 refseg、只跑单轮
        if a.task is not None and a.task != ["refseg"]:
            p.error(f"--instruction-check 只用于 refseg，收到 --task {' '.join(a.task)}")
        if a.k_values is not None and a.k_values != [1]:
            p.error(f"--instruction-check 只跑单轮（--k-values 1），收到 --k-values {' '.join(map(str, a.k_values))}")
        a.task, a.k_values = ["refseg"], [1]
    elif a.normal_leaf is not None:
        p.error("--normal-leaf 只配 --instruction-check 用")
    if a.task is None:
        a.task = list(GRID_TASKS)
    if len(set(a.task)) != len(a.task):
        p.error("--task 有重复")
    a.device = a.device or ("cpu" if a.tiny else "cuda")
    a.dtype = a.dtype or ("fp32" if a.tiny else "bf16")
    if not Path(a.checkpoint).is_dir():
        p.error(f"--checkpoint {a.checkpoint!r} 不是目录")
    if not a.tiny and not (a.data_root or os.environ.get("VISINT_DATA_ROOT")):
        p.error("真模型运行要给 --data-root（或设 VISINT_DATA_ROOT）")
    if not a.tiny and not RUN_DIR_NAME.fullmatch(Path(a.out).name):
        p.error(f"--out 的目录名必须带运行编号，形如 0050_e3b_eval（CLAUDE.md：experiments/<phase>/<NNNN_slug>/），收到 {a.out!r}")
    if a.k_values is not None:  # DEV-0039：取值规则与 resolve_eval 同一个函数（EV.k_values_prefix），不合规是用法错误
        try:
            config_ks = EV.config_k_values(load_config(a.config))
        except Exception as e:  # noqa: BLE001 — 读不了配置就没法核对前缀
            p.error(f"--k-values 要按 --config 的 eval.multi_turn.k_values 核对，读配置失败：{e!r}")
        try:
            EV.k_values_prefix(config_ks, a.k_values)
        except ValueError as e:
            p.error(str(e))
    return a


def eval_paths(a) -> EV.EvalPaths:
    if a.tiny and not a.data_root:
        base = EV.EvalPaths.fixtures()
        return base if not a.dev_subsets else EV.EvalPaths(base.refseg_root, base.depth_root, Path(a.dev_subsets))
    return EV.EvalPaths.single_root(Path(a.data_root or os.environ["VISINT_DATA_ROOT"]), a.dev_subsets)


def checkpoint_block(ck_dir: Path, man: dict) -> dict:
    """运行身份里的检查点：manifest 的 sha256 与几项摘要（目录路径与机器有关，只进运行记录）。"""
    out = {"manifest_sha256": sha256_file(ck_dir / "manifest.json"), "format": man.get("format"),
           "build_spec": man.get("build_spec"), "image_policy": man.get("image_policy"), "dtype": man.get("dtype"),
           "lora": man.get("lora"), "trainable_token_rows": man.get("trainable_token_rows")}
    if man.get("derived") is not None:  # DEV-0041 §3：派生检查点（scripts/scale_lora_ckpt.py）；原生的检查点没有这个键，运行身份不变
        out["derived"] = man["derived"]
    return out


def cmd_run(a, argv) -> int:
    cfg = load_config(a.config)
    settings = EV.resolve_eval(cfg, tiny=a.tiny, batch_size=a.batch_size, score_workers=a.score_workers,
                               merge_check_n=a.merge_check_n, k_values=a.k_values, decode=a.decode)
    if not a.tiny and settings["batch_size"] != settings["batch_size_config"] and not a.allow_unfrozen_batch:
        raise ValueError(f"--batch-size {settings['batch_size']} 不是配置的 eval.batch_size {settings['batch_size_config']}；"
                         "确要换就加 --allow-unfrozen-batch")
    if a.instruction_check is not None and a.limit is not None and a.limit % settings["batch_size"]:
        raise ValueError(f"--instruction-check 的 --limit {a.limit} 必须是批大小 {settings['batch_size']} 的整数倍"
                         "（批组成才与正常评测的前 N 条相同）")
    ck = Path(a.checkpoint)
    man = read_manifest(ck)
    spec_b = man.get("build_spec") or {}
    if spec_b.get("key") != a.model or bool(spec_b.get("tiny")) != a.tiny:
        raise ValueError(f"检查点的构造方式（key {spec_b.get('key')!r}，tiny {spec_b.get('tiny')}）与 --model {a.model!r}、"
                         f"--tiny {a.tiny} 不符")
    entry = registry.get_model(a.model)
    probe = EV.EvalSpec(model_key=entry.key, family=entry.family, repo=entry.repo, revision=entry.revision,
                        adapter_model_key=str(man.get("model_key", entry.key)), tiny=a.tiny, dtype=a.dtype, variant=a.variant,
                        seed=int(a.seed), checkpoint=checkpoint_block(ck, man), limit=a.limit)
    mdir = EV.model_dir(a.out, probe)
    mdir.mkdir(parents=True, exist_ok=True)
    mc = settings["merge_check"]
    summ = run_merge_check(ck, model_key=a.model, tiny=a.tiny, n=mc["n"], checks=mc["checks"], out_json=mdir / EV.MERGE_CHECK,
                           device=a.device, dtype=a.dtype, data_root=a.data_root or os.environ.get("VISINT_DATA_ROOT"),
                           dev_subsets=None, e3b_config=a.config)  # 核对读评测用的同一份配置（D67-4(a)）
    ad, _ = factory.load_trained(ck, device=a.device, dtype=DTYPES[a.dtype], merge=True)
    if man.get("lora") is not None and ad.lora_state != "merged":
        raise RuntimeError(f"load_trained(merge=True) 之后 LoRA 状态是 {ad.lora_state}，应为 merged")
    tokens = attach_vocab(ad, cfg.vocab)
    spec = EV.EvalSpec(**{**probe.__dict__, "adapter_model_key": str(getattr(ad, "model_key", entry.key))})
    paths = eval_paths(a)
    extra = {"checkpoint_dir": str(ck), "paths": paths.as_dict(), "merge_check": summ,
             "gpu": torch.cuda.get_device_name(0) if a.device.startswith("cuda") and torch.cuda.is_available() else None}
    for task in a.task:
        sel = EV.load_eval_set(task, paths)
        ctx = EV.make_context(cfg, tokens, task)
        if a.instruction_check is not None:  # DEV-0041 §1
            m = EV.run_instruction_check(ad, cfg, settings, spec, ctx, sel, a.out, check=a.instruction_check,
                                         normal_leaf=a.normal_leaf, resume=a.resume, argv=argv, run_extra=extra)
            ic = m["instruction_check"]
            for t, b in m["targets"].items():
                pc = b["paired_normal_minus_check"]["cIoU"] or {}
                print(f"{entry.key} {a.variant} seed{a.seed} {task} instr_{ic['kind']} target={t} n={b['check']['cIoU']['n']} "
                      f"(skipped {ic['n_skipped']}) cIoU check={b['check']['cIoU']['value']} normal={b['normal']['cIoU']['value']} "
                      f"normal-check={pc.get('estimate')} [{pc.get('low')}, {pc.get('high')}]")
            continue
        m = EV.run_eval_task(ad, cfg, settings, spec, ctx, sel, a.out, resume=a.resume, argv=argv, run_extra=extra)
        h = m["headline"]
        vals = "  ".join(f"{k}={(h[k] or {}).get('value')}" for k in h if isinstance(h[k], dict))
        print(f"{entry.key} {a.variant} seed{a.seed} {task} n={m['n']} single/{h['upsample']}: {vals}")
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    a = parse_args(argv)
    try:
        return cmd_run(a, argv)
    except MergeCheckFailed as e:
        print(f"拒绝评测：{e}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001 — 证据先落盘，再以非零码退出
        err_dir = Path(a.out) / EV.EVAL_SUB / a.variant / a.model / f"seed{a.seed}" / "errors"
        err_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = err_dir / f"error_{stamp}.json"
        path.write_text(json.dumps({"argv": argv, "error": repr(e), "traceback": traceback.format_exc()[-6000:]},
                                   indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"失败：{e!r}（已写出 {path}）", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
