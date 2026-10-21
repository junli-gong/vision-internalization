#!/usr/bin/env python
"""W1 的入口（DEV-0051；D103；docs/w1_plan.md §3）：一个模型 × 一个任务 × 一种设置 × 一个划分。

运行（服务器；--out 是本次运行的目录 experiments/W1/<NNNN_slug>，结果写进 <out>/<模型>/<任务>/<设置>/<划分>/）：
    python scripts/run_w1.py --model qwen3.5-9b --task refseg --split devcheck --setting full \\
        --data-root $PROJ/data --out experiments/W1/0060_w1_devcheck [--resume] [--limit N]
设置：
- full（W1-1，完整工作台）、no_eye（A1）、no_check（A2）、no_snap（A3：去掉 snap 与 code）：回路（src/w1/loop.py），协议与提示在
  configs/w1.json；批大小默认取 batching.batch_size（--batch-size 可改，记进运行身份）。
- baseline_e1（W1-0，E1 零样本：refseg → refexp_poly，refexp_box，count）：E1 的运行器原样，E1 冻结的批大小（64）。
- baseline_ownbox（"自己的框"，refseg）：读同一个划分上的 W1-0 refexp_box（<out>/<模型>/refexp_box/baseline_e1/<划分>），没写完先跑完。
- baseline_poly_structured（E1 结构强制的多边形，refseg）：E1 事后读数的运行器，样本换成 W1 的划分。
- w1f（W1-F，refseg）：自己的框 → 工作台的同一个 GrabCut（src.w1.workbench.grabcut_mask），没有 LLM（框没写完时先跑框，要模型）。
划分：devtune、devcheck（refseg、refexp_box）；test_refcoco_unc_testA、test_refcoco_unc_testB、test_refcoco+_unc_testA、
test_refcoco+_unc_testB、test_refcocog_umd_test（refseg、refexp_box）；devtune_countbench、test_countbench（count）。清单在
configs/w1_splits.json（src/w1/data.py）。
开发机彩排：加 --tiny（小型随机模型、CPU、tests/fixtures 的小数据，划分按 configs/w1.json 的 splits.tiny 现算，预算与批大小取 tiny 一节）。
冻结保护（DEV-0053；D103 §二.4、§三）：划分是 devcheck 或以 test 开头时，--w1-config 指的文件（默认 configs/w1.json）必须
"frozen": true，否则退出码 2；对全部设置都适用（基线也一样），--tiny 不受此限。--w1-config 可以指向调参的变体（configs/w1_tune/）；
运行身份记这个文件的路径与 sha256。
DEV-0049 / DEV-0050 合入之前，回路的彩排配假适配器与假工作台：--adapter-factory tests._w1_fakes:make_policy_adapter
--workbench tests._w1_fakes:FakeWorkbench（这两个选项只配 --tiny）。
退出码：0 = 跑完；2 = 出错（报错先写进 <out>/<模型>/errors/，argparse 用法错误也是 2）。
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import os
import re
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from src.adapters import registry  # noqa: E402
from src.adapters.factory import default_image_policy, make_adapter  # noqa: E402
from src.e1.prompts import PROMPTS_PATH, load_prompt_config  # noqa: E402
from src.e1.tasks import DataPaths  # noqa: E402
from src.w1 import baselines as B  # noqa: E402
from src.w1.config import BASELINE_TASKS, LOOP_SETTINGS, SETTINGS, W1_CONFIG_PATH, load_w1_config  # noqa: E402
from src.w1.data import W1_SPLITS_PATH, load_splits, split_names, tiny_splits, w1_items  # noqa: E402
from src.w1.loop import TASKS  # noqa: E402
from src.w1.runner import W1RunSpec, run_loop  # noqa: E402

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
TINY_INIT_STD = 0.05  # 与 run_e1 相同
TINY_E1_BATCH_SIZE = 4  # 与 run_e1 的 --tiny 默认相同
RUN_DIR_NAME = re.compile(r"[0-9]{4}_[A-Za-z0-9][A-Za-z0-9_.-]*")
DEFAULT_WORKBENCH = "src.w1.workbench:Workbench"


def needs_frozen(split: str) -> bool:
    """devcheck 与 test_*（开发-检查集与 test）只能用冻结的协议跑（D103 §二.4、§三）；devtune 与 devtune_countbench 不要求。"""
    return split == "devcheck" or split.startswith("test")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="注册键（src/adapters/registry.py）")
    p.add_argument("--task", required=True, choices=TASKS)
    p.add_argument("--split", required=True, help="划分名（configs/w1_splits.json 的 splits）")
    p.add_argument("--setting", required=True, choices=SETTINGS)
    p.add_argument("--out", required=True, help="本次运行的目录（experiments/W1/<NNNN_slug>）")
    p.add_argument("--batch-size", type=int, help="回路：默认 configs/w1.json 的 batching.batch_size；E1 的基线：默认 E1 冻结的 64")
    p.add_argument("--chunk-size", type=int, help="回路与 W1-F 的组大小（续跑粒度；默认 configs/w1.json 的 batching.chunk_size）")
    p.add_argument("--tool-workers", type=int, help="组内跨样本并行执行工具调用的线程数（默认 batching.tool_workers；不进运行身份）")
    p.add_argument("--limit", type=int, help="只取划分的前 N 条（调试）")
    p.add_argument("--resume", action="store_true", help="续跑：跳过 predictions.jsonl 里已完成的 id（身份必须相同）")
    p.add_argument("--tiny", action="store_true", help="小型随机模型 + CPU + tests/fixtures 的小数据（开发机彩排）")
    p.add_argument("--data-root", help="落地目录（默认环境变量 VISINT_DATA_ROOT；--tiny 时取 fixture）")
    p.add_argument("--w1-config", default=str(W1_CONFIG_PATH), help="默认 configs/w1.json")
    p.add_argument("--device", help="默认：真模型 cuda，--tiny 时 cpu")
    p.add_argument("--dtype", choices=sorted(DTYPES), help="默认：真模型 bf16，--tiny 时 fp32")
    p.add_argument("--workbench", default=DEFAULT_WORKBENCH, help="工作台类 MODULE:ATTR（默认真工作台；别的只配 --tiny）")
    p.add_argument("--adapter-factory", help="MODULE:ATTR，返回回路用的适配器（只配 --tiny；DEV-0049 合入前的彩排用假适配器）")
    p.add_argument("--allow-unfrozen-batch", action="store_true", help="E1 的基线允许用冻结值以外的批大小（不是协议口径）")
    a = p.parse_args(argv)
    if a.model not in registry.MODELS:
        p.error(f"未登记的模型键 {a.model!r}")
    if a.setting not in LOOP_SETTINGS and a.task not in BASELINE_TASKS[a.setting]:
        p.error(f"--setting {a.setting} 只用于 {list(BASELINE_TASKS[a.setting])}")
    if not a.tiny and (a.workbench != DEFAULT_WORKBENCH or a.adapter_factory):
        p.error("--workbench（非默认）与 --adapter-factory 只配 --tiny")
    for name in ("batch_size", "chunk_size", "tool_workers", "limit"):
        v = getattr(a, name)
        if v is not None and v < 1:
            p.error(f"--{name.replace('_', '-')} 必须是正整数")
    if not a.tiny and not (a.data_root or os.environ.get("VISINT_DATA_ROOT")):
        p.error("真模型运行要给 --data-root（或设 VISINT_DATA_ROOT）")
    if not a.tiny and not RUN_DIR_NAME.fullmatch(Path(a.out).name):
        p.error(f"--out 的目录名必须带运行编号，形如 0060_w1_devcheck，收到 {a.out!r}")
    if not a.tiny and needs_frozen(a.split):  # 冻结保护：放在最后，其余用法错误先报
        try:
            frozen = load_w1_config(a.w1_config).data["frozen"]
        except (OSError, ValueError) as e:  # W1ConfigError、JSON 写坏都是 ValueError
            p.error(f"--w1-config {a.w1_config} 读不了：{e}")
        if frozen is not True:
            p.error(f"协议未冻结，不能碰开发-检查集与 test（D103 §二.4、§三）：--split {a.split} 要求 {a.w1_config} 的 "
                    f"\"frozen\" 为 true（现在是 {frozen!r}）")
    a.device = a.device or ("cpu" if a.tiny else "cuda")
    a.dtype = a.dtype or ("fp32" if a.tiny else "bf16")
    return a


def _load_attr(spec: str):
    mod, _, attr = spec.partition(":")
    if not mod or not attr:
        raise ValueError(f"应写成 MODULE:ATTR，收到 {spec!r}")
    obj = importlib.import_module(mod)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def data_paths(a) -> DataPaths:
    if a.tiny and not a.data_root:  # RefCOCO 用放大到 48×36 的事后读数 fixture（GrabCut 与放大视图在 12×9 上没有意义）；计数用 e1 的
        return DataPaths.fixtures() if a.task == "count" else DataPaths.posthoc_fixtures()
    return DataPaths(root=Path(a.data_root or os.environ["VISINT_DATA_ROOT"]))


def build_model(a):
    """(registry 条目, 图像口径, 适配器)：真模型或 --tiny 的小型随机模型（E1 冻结的图像口径）。"""
    entry = registry.get_model(a.model)
    policy = default_image_policy(entry.family)
    kw = {"init_std": TINY_INIT_STD} if a.tiny else {}
    ad = make_adapter(a.model, device=a.device, dtype=DTYPES[a.dtype], image_policy=policy, tiny=a.tiny, **kw)
    return entry, policy, ad


def run_spec(a, entry, policy, adapter_key: str) -> W1RunSpec:
    return W1RunSpec(model_key=entry.key, family=entry.family, repo=entry.repo, revision=entry.revision,
                     adapter_model_key=adapter_key, task=a.task, split=a.split, setting=a.setting, tiny=a.tiny, limit=a.limit,
                     image_policy={"tag": "default", "type": type(policy).__name__, **policy.as_dict()})


def e1_batch_size(a, prompts) -> int:
    frozen = prompts.batch_protocol()["batch_size"]
    bs = a.batch_size or (TINY_E1_BATCH_SIZE if a.tiny else frozen)
    if not a.tiny and bs != frozen and not a.allow_unfrozen_batch:
        raise ValueError(f"E1 的基线要用 E1 冻结的批大小 {frozen}（--batch-size {bs}）；确要换就加 --allow-unfrozen-batch")
    return bs


def cmd(a, argv) -> int:
    cfg = load_w1_config(a.w1_config)
    splits_doc = tiny_splits(cfg) if a.tiny else load_splits(W1_SPLITS_PATH)
    if a.split not in split_names(a.task, splits_doc):
        raise ValueError(f"--split {a.split} 不用于 {a.task}；可用：{split_names(a.task, splits_doc)}")
    paths = data_paths(a)
    samples = w1_items(a.task, a.split, paths, splits=splits_doc)
    if a.limit is not None:
        samples = samples[: a.limit]
    bat = cfg.batching(tiny=a.tiny)
    chunk = a.chunk_size or bat["chunk_size"]
    workbench = _load_attr(a.workbench) if a.setting in LOOP_SETTINGS else None
    entry = registry.get_model(a.model)
    policy = default_image_policy(entry.family)
    model_cache: dict[str, tuple] = {}

    def get_model():
        if "m" not in model_cache:
            model_cache["m"] = build_model(a)
        return model_cache["m"]

    def key_of(ad) -> str:
        return str(getattr(ad, "model_key", entry.key))

    if a.setting in LOOP_SETTINGS:
        if a.adapter_factory:
            ad = _load_attr(a.adapter_factory)(entry)
        else:
            _, _, ad = get_model()
        protocol = cfg.protocol(a.setting, tiny=a.tiny)
        spec = run_spec(a, entry, policy, key_of(ad))
        m = run_loop(ad, spec, cfg, protocol, samples, paths, a.out, splits_doc=splits_doc,
                     batch_size=a.batch_size or bat["batch_size"], chunk_size=chunk,
                     tool_workers=a.tool_workers or bat["tool_workers"], workbench_factory=workbench,
                     max_batch_tokens=bat["max_batch_tokens"],
                     workbench_name=a.workbench, resume=a.resume, argv=argv)
    else:
        prompts = load_prompt_config(PROMPTS_PATH)
        if a.setting in ("baseline_e1", "baseline_poly_structured"):
            entry, policy, ad = get_model()
            spec = run_spec(a, entry, policy, key_of(ad))
            run = B.run_baseline_e1 if a.setting == "baseline_e1" else B.run_baseline_poly_structured
            m = run(ad, entry, policy, spec, cfg, samples, paths, a.out, splits_doc=splits_doc,
                    batch_size=e1_batch_size(a, prompts), resume=a.resume, argv=argv, prompts=prompts)
        else:
            akey = f"tiny({entry.key})" if a.tiny else entry.key  # 与 make_adapter 记的 model_key 相同；框已写完时不加载模型
            spec = run_spec(a, entry, policy, akey)
            box_dir, box_recs = B.ensure_box_run(get_model, spec, cfg, samples, paths, a.out, splits_doc=splits_doc,
                                                 batch_size=e1_batch_size(a, prompts), argv=argv, prompts=prompts)
            if a.setting == "baseline_ownbox":
                m = B.run_baseline_ownbox(spec, cfg, samples, paths, a.out, splits_doc=splits_doc, box_dir=box_dir,
                                          box_recs=box_recs, resume=a.resume, argv=argv, prompts=prompts)
            else:
                m = B.run_w1f(spec, cfg, samples, paths, a.out, splits_doc=splits_doc, box_dir=box_dir, box_recs=box_recs,
                              chunk_size=chunk, resume=a.resume, argv=argv, prompts=prompts)
    head = "; ".join(f"{h['name']}={h['value']}" for h in m.get("headline", []))
    print(f"W1 {a.model} {a.task}/{a.setting}/{a.split} n={m['n']} complete={m.get('complete')} {head}")
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    a = parse_args(argv)
    try:
        return cmd(a, argv)
    except Exception as e:  # noqa: BLE001 — 证据先落盘，再以非零码退出
        err_dir = Path(a.out) / a.model / "errors"
        err_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = err_dir / f"error_{stamp}.json"
        path.write_text(json.dumps({"argv": argv, "error": repr(e), "traceback": traceback.format_exc()[-6000:]},
                                   indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"失败：{e!r}（已写出 {path}）", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
