#!/usr/bin/env python
"""E5 评测框架的入口（DEV-0018）：零样本基座、文本思维链、平凡基线、外部预测（LISA）的评分；另可导出评测集给外部系统。

运行（服务器；--out 是本次运行的目录 experiments/E5/<NNNN_slug>，结果写进 <out>/<系统>/<模型>/<任务>[__dev]/）：
    # 零样本 / 思维链（同一个模型可以一次跑两个系统；--task 给任务名、字母或 all）
    python scripts/run_e5.py --system zeroshot cot --model qwen3.5-0.8b --task all --out experiments/E5/0019_e5_zeroshot \\
        [--subset eval|dev] [--n-per-bin 100] [--batch-size 16] [--resume]
    # 平凡基线（不加载模型；--model 不给时跑该任务的全部基线与参照线）
    python scripts/run_e5.py --system baseline --task all --out experiments/E5/0019_e5_zeroshot
    # 外部预测（只评分；目录格式见 src/e5/external.py）
    python scripts/run_e5.py --system external --model lisa-7b-v1 --external-dir <外部预测目录> --task all --out <dir>
    # 导出评测集的输入给外部系统（图像 PNG + inputs.jsonl）
    python scripts/run_e5.py --export-inputs <目录> --task all [--subset eval] [--n-per-bin 100]
开发机彩排：--tiny（小型随机模型、CPU、fp32；max_new_tokens 默认截到 TINY_MAX_NEW_TOKENS，--limit 取前几条）。
d（physics）的数据只在规范平台上生成（D24）：非规范平台上只允许 --tiny。
退出码：0 = 跑完；2 = 出错（报错写进 <out>/errors/）。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.e5 import external as EXT  # noqa: E402
from src.e5.prompts import PROMPTS_PATH, load_prompt_config  # noqa: E402
from src.e5.protocol import E5_TASKS, MODEL_SYSTEMS, SUBSETS, SYSTEMS, default_n_per_bin  # noqa: E402
from src.e5.runner import (  # noqa: E402
    RunSpec,
    make_sample,
    run_baseline_task,
    run_external_task,
    run_model_task,
    spec_items,
)
from src.synth.seeding import canonical_task  # noqa: E402

TINY_INIT_STD = 0.05  # 与 run_e1 相同：小模型重取参数
TINY_MAX_NEW_TOKENS = 48  # --tiny 彩排的生成上限（小模型几乎不会自己停，按真实上限跑太慢）；记进 config 的 max_new_tokens_override
RUN_DIR_NAME = re.compile(r"[0-9]{4}_[A-Za-z0-9][A-Za-z0-9_.-]*")
DTYPES = ("bf16", "fp32")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--system", nargs="+", choices=SYSTEMS, help="zeroshot / cot（可同时给）/ baseline / external")
    p.add_argument("--export-inputs", help="导出评测集的输入到这个目录（给外部系统），不评分")
    p.add_argument("--model", help="模型注册键（zeroshot、cot）；基线名（baseline，不给 = 全部）；外部系统名（external）")
    p.add_argument("--task", nargs="+", required=True, help=f"任务名、字母（a–e）或 all：{list(E5_TASKS)}")
    p.add_argument("--subset", default="eval", choices=sorted(SUBSETS), help="eval = test 划分；dev = val 划分（只核对格式）")
    p.add_argument("--n-per-bin", type=int, help="每档条数（默认 eval 100、dev 2）")
    p.add_argument("--limit", type=int, help="只取评测清单的前 N 条（彩排、调试）")
    p.add_argument("--out", help="本次运行的目录（experiments/E5/<NNNN_slug>）")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--batch-size", type=int, help="批大小（默认取提示配置的 batching.batch_size；--tiny 默认 2）")
    p.add_argument("--external-dir", help="外部预测目录（--system external）")
    p.add_argument("--prompts", default=str(PROMPTS_PATH))
    p.add_argument("--tiny", action="store_true", help="小型随机模型 + CPU（开发机彩排）")
    p.add_argument("--max-new-tokens", type=int, help="只用于 --tiny：覆盖生成上限（默认 TINY_MAX_NEW_TOKENS）")
    p.add_argument("--n-resamples", type=int, help="bootstrap 次数（默认 src.constants.BOOTSTRAP_N_RESAMPLES；--tiny 默认 50）")
    p.add_argument("--device")
    p.add_argument("--dtype", choices=DTYPES)
    a = p.parse_args(argv)
    if (a.system is None) == (a.export_inputs is None):
        p.error("给 --system 或 --export-inputs 之一")
    a.task = list(E5_TASKS) if a.task == ["all"] else [canonical_task(t) for t in a.task]
    a.n_per_bin = a.n_per_bin or default_n_per_bin(a.subset)
    for name in ("n_per_bin", "limit", "batch_size", "max_new_tokens"):
        v = getattr(a, name)
        if v is not None and v < 1:
            p.error(f"--{name.replace('_', '-')} 必须是正整数")
    if a.max_new_tokens is not None and not a.tiny:
        p.error("--max-new-tokens 只用于 --tiny（正式运行的生成上限冻结在提示配置里）")
    if a.export_inputs:
        return a
    if not a.out:
        p.error("要给 --out")
    systems = set(a.system)
    if systems & set(MODEL_SYSTEMS) and systems - set(MODEL_SYSTEMS):
        p.error("zeroshot / cot 不能与 baseline / external 写在同一条命令里")
    if systems & set(MODEL_SYSTEMS) and not a.model:
        p.error("zeroshot / cot 要给 --model（注册键）")
    if "external" in systems and (not a.model or not a.external_dir):
        p.error("--system external 要给 --model（外部系统名）与 --external-dir")
    if not a.tiny and not RUN_DIR_NAME.fullmatch(Path(a.out).name):
        p.error(f"--out 的目录名必须带运行编号，形如 0019_e5_zeroshot，收到 {a.out!r}")
    if a.tiny:
        a.max_new_tokens = a.max_new_tokens or TINY_MAX_NEW_TOKENS
    a.device = a.device or ("cpu" if a.tiny else "cuda")
    a.dtype = a.dtype or ("fp32" if a.tiny else "bf16")
    if a.n_resamples is None:
        from src.constants import BOOTSTRAP_N_RESAMPLES

        a.n_resamples = 50 if a.tiny else BOOTSTRAP_N_RESAMPLES
    return a


def _headline(m) -> str:
    h = m["headline"]
    parts = [f"iou_grid={h['mask_iou_grid']['strict']}"]
    if "relief_error" in h:
        parts.append(f"relief={h['relief_error']['strict']}")
    g = m.get("gate") or {}
    return " ".join(parts) + f" parse_fail={m['parse']['failure_rate']} imagining={g.get('imagining')}"


def cmd_export(a) -> int:
    for t in a.task:
        spec = RunSpec(system="baseline", model_key="-", task=t, subset=a.subset, n_per_bin=a.n_per_bin, n_resamples=0,
                       limit=a.limit)
        samples = [make_sample(it) for it in spec_items(spec)]
        path = EXT.export_inputs(samples, Path(a.export_inputs) / t)
        print(f"{t}: {len(samples)} 条 -> {path}")
    return 0


def build_adapter(a):
    import torch

    from src.adapters import registry
    from src.adapters.factory import default_image_policy, make_adapter

    entry = registry.get_model(a.model)
    policy = default_image_policy(entry.family)
    kw = {"init_std": TINY_INIT_STD} if a.tiny else {}
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]
    ad = make_adapter(a.model, device=a.device, dtype=dtype, image_policy=policy, tiny=a.tiny, **kw)
    return entry, policy, ad


def cmd_models(a, argv) -> int:
    prompts = load_prompt_config(a.prompts)
    entry, policy, ad = build_adapter(a)
    bs = a.batch_size or (2 if a.tiny else prompts.batch_size())
    for system in a.system:
        for t in a.task:
            spec = RunSpec(system=system, model_key=entry.key, task=t, subset=a.subset, n_per_bin=a.n_per_bin,
                           n_resamples=a.n_resamples, tiny=a.tiny, family=entry.family, repo=entry.repo,
                           revision=entry.revision, adapter_model_key=str(getattr(ad, "model_key", entry.key)),
                           policy={"type": type(policy).__name__, **policy.as_dict()}, batch_size=bs,
                           max_new_tokens_override=a.max_new_tokens if a.tiny else None, limit=a.limit)
            m = run_model_task(ad, spec, prompts, a.out, resume=a.resume, argv=argv)
            print(f"{system} {entry.key} {t}/{a.subset} n={m['n']} {_headline(m)}")
    return 0


def cmd_offline(a, argv) -> int:
    for system in a.system:
        for t in a.task:
            if system == "baseline":
                names = [a.model] if a.model else sorted(make_sample(spec_items(RunSpec(
                    system="baseline", model_key="-", task=t, subset=a.subset, n_per_bin=1, n_resamples=0))[0]).baselines)
                for name in names:
                    spec = RunSpec(system="baseline", model_key=name, task=t, subset=a.subset, n_per_bin=a.n_per_bin,
                                   n_resamples=a.n_resamples, tiny=a.tiny, limit=a.limit)
                    m = run_baseline_task(spec, a.out, resume=a.resume, argv=argv)
                    print(f"baseline {name} {t}/{a.subset} n={m['n']} {_headline(m)}")
            else:
                spec = RunSpec(system="external", model_key=a.model, task=t, subset=a.subset, n_per_bin=a.n_per_bin,
                               n_resamples=a.n_resamples, tiny=a.tiny, external_dir=str(Path(a.external_dir).resolve()),
                               limit=a.limit)
                m = run_external_task(spec, a.out, resume=a.resume, argv=argv)
                print(f"external {a.model} {t}/{a.subset} n={m['n']} {_headline(m)}")
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    a = parse_args(argv)
    try:
        if a.export_inputs:
            return cmd_export(a)
        if set(a.system) & set(MODEL_SYSTEMS):
            return cmd_models(a, argv)
        return cmd_offline(a, argv)
    except Exception as e:  # noqa: BLE001 — 证据先落盘，再以非零码退出
        err_dir = Path(a.out or a.export_inputs) / "errors"
        err_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = err_dir / f"error_{stamp}.json"
        path.write_text(json.dumps({"argv": argv, "error": repr(e), "traceback": traceback.format_exc()[-6000:]},
                                   indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"失败：{e!r}（已写出 {path}）", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
