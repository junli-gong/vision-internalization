#!/usr/bin/env python
"""W2 的入口（DEV-0059；D113；docs/w2_plan.md §3）：一个模型 × 一个任务 × 一种设置 × 一个划分。

运行（服务器；--out 是本次运行的目录 experiments/W2/<NNNN_slug>，结果写进 <out>/<模型>/<任务>/<设置>/<划分>/）：
    python scripts/run_w2.py --model qwen3.5-9b --task refseg --split devtune800 --setting w2a \\
        --data-root $PROJ/data --out experiments/W2/0063_w2_tune [--resume] [--limit N]
设置（--setting）：
- w2a（主方法）：refseg 走 提出 → 候选 → 挑选（configs/w2.json）；refexp_box、count 按输出类型路由到 E1 单轮（= baseline_e1 的结果）。
- random_pick、oracle_pick、w1f（refseg）：同一份候选里伪随机挑、用真值挑、交 1 号（= W1-F），不跑模型（候选与框没写完时先跑，要模型）。
- ownbox（refseg）：E1 refexp_box 的框光栅化；baseline_e1：E1 零样本（refseg → refexp_poly、refexp_box、count），E1 冻结的批大小（64）。
"提出"的框 = 同一个 --out 里的 <out>/<模型>/refexp_box/baseline_e1/<划分>，没写完先按 E1 的协议跑完；候选只生成一次
（<out>/<模型>/refseg/candidates/<划分>/candidates.jsonl），w2a、random_pick、oracle_pick、w1f 共用。
划分：devtune800（W1 的 800 句，可以随便用）、devcheck2（新的 500 句，只用于判定）、test_*（同 W1）——refseg、refexp_box；
devtune_countbench、test_countbench——count。清单在 configs/w2_splits.json（src/w2/data.py；devcheck2 由服务器生成）。
冻结保护（同 D108）：划分是 devcheck2 或以 test 开头时，--w2-config（默认 configs/w2.json）必须 "frozen": true，否则退出码 2，什么都不写；
对全部设置都适用；devtune800 不受限；--tiny 不受限。
开发机彩排：加 --tiny（小型随机模型、CPU、tests/fixtures 的小数据，划分按同一套规则现算，挑选的预算与批大小取 configs/w2.json 的 tiny）。
候选生成可以并行：--candidate-workers N（默认 1；按组用 spawn 的进程池，子进程里 OpenCV 与 BLAS 都是 1 个线程；candidates.jsonl 除
info 里的耗时外与单进程相同，不进运行身份）。
候选与渲染的实现（--candidates-module、--render-module，默认 src.w2.candidates、src.w2.render，DEV-0058）与挑选的适配器
（--adapter-factory MODULE:ATTR）可以换成测试的假实现（tests._w2_fakes），只配 --tiny。
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
from src.w1.loop import TASKS  # noqa: E402
from src.w2 import runner as R  # noqa: E402
from src.w2.config import PROPOSE_BOX, SETTING_TASKS, SETTINGS, W2_CONFIG_PATH, load_w2_config  # noqa: E402
from src.w2.data import W2_SPLITS_PATH, load_splits, needs_frozen, split_names, tiny_paths, tiny_splits, w2_items  # noqa: E402

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
TINY_INIT_STD = 0.05  # 与 run_e1、run_w1 相同
TINY_E1_BATCH_SIZE = 4  # 与 run_e1、run_w1 的 --tiny 默认相同
RUN_DIR_NAME = re.compile(r"[0-9]{4}_[A-Za-z0-9][A-Za-z0-9_.-]*")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="注册键（src/adapters/registry.py）")
    p.add_argument("--task", required=True, choices=TASKS)
    p.add_argument("--split", required=True, help="划分名（configs/w2_splits.json 的 splits）")
    p.add_argument("--setting", required=True, choices=SETTINGS)
    p.add_argument("--out", required=True, help="本次运行的目录（experiments/W2/<NNNN_slug>）")
    p.add_argument("--batch-size", type=int, help="挑选：默认 configs/w2.json 的 batching.batch_size；E1 的运行：默认 E1 冻结的 64")
    p.add_argument("--chunk-size", type=int, help="组大小（续跑粒度；默认 batching.chunk_size）")
    p.add_argument("--candidate-workers", type=int, default=1,
                   help="候选生成的进程数（默认 1；>1 时 spawn 进程池、子进程单线程；结果与单进程相同，不进运行身份）")
    p.add_argument("--limit", type=int, help="只取划分的前 N 条（调试）")
    p.add_argument("--resume", action="store_true", help="续跑：跳过已完成的 id（身份必须相同）")
    p.add_argument("--tiny", action="store_true", help="小型随机模型 + CPU + tests/fixtures 的小数据（开发机彩排）")
    p.add_argument("--data-root", help="落地目录（默认环境变量 VISINT_DATA_ROOT；--tiny 时取 fixture）")
    p.add_argument("--w2-config", default=str(W2_CONFIG_PATH), help="默认 configs/w2.json")
    p.add_argument("--device", help="默认：真模型 cuda，--tiny 时 cpu")
    p.add_argument("--dtype", choices=sorted(DTYPES), help="默认：真模型 bf16，--tiny 时 fp32")
    p.add_argument("--candidates-module", default=R.DEFAULT_CANDIDATES_MODULE, help="候选生成器的模块（默认真实现；别的只配 --tiny）")
    p.add_argument("--render-module", default=R.DEFAULT_RENDER_MODULE, help="挑选图渲染的模块（默认真实现；别的只配 --tiny）")
    p.add_argument("--adapter-factory", help="MODULE:ATTR，返回挑选一步的适配器（只配 --tiny）")
    p.add_argument("--allow-unfrozen-batch", action="store_true", help="E1 的运行允许用冻结值以外的批大小（不是协议口径）")
    a = p.parse_args(argv)
    if a.model not in registry.MODELS:
        p.error(f"未登记的模型键 {a.model!r}")
    if a.task not in SETTING_TASKS[a.setting]:
        p.error(f"--setting {a.setting} 只用于 {list(SETTING_TASKS[a.setting])}")
    if not a.tiny and (a.candidates_module != R.DEFAULT_CANDIDATES_MODULE or a.render_module != R.DEFAULT_RENDER_MODULE
                       or a.adapter_factory):
        p.error("--candidates-module、--render-module（非默认）与 --adapter-factory 只配 --tiny")
    for name in ("batch_size", "chunk_size", "limit", "candidate_workers"):
        v = getattr(a, name)
        if v is not None and v < 1:
            p.error(f"--{name.replace('_', '-')} 必须是正整数")
    if not a.tiny and not (a.data_root or os.environ.get("VISINT_DATA_ROOT")):
        p.error("真模型运行要给 --data-root（或设 VISINT_DATA_ROOT）")
    if not a.tiny and not RUN_DIR_NAME.fullmatch(Path(a.out).name):
        p.error(f"--out 的目录名必须带运行编号，形如 0063_w2_tune，收到 {a.out!r}")
    if not a.tiny and needs_frozen(a.split):  # 冻结保护：放在最后，其余用法错误先报
        try:
            frozen = load_w2_config(a.w2_config).data["frozen"]
        except (OSError, ValueError) as e:  # W2ConfigError、JSON 写坏都是 ValueError
            p.error(f"--w2-config {a.w2_config} 读不了：{e}")
        if frozen is not True:
            p.error(f"协议未冻结，不能碰判定集与 test（D113；同 D108）：--split {a.split} 要求 {a.w2_config} 的 \"frozen\" 为 true"
                    f"（现在是 {frozen!r}）")
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
    if a.tiny and not a.data_root:  # RefCOCO 用 tests/fixtures/w2（e1_posthoc 的 48×36 图加两张 val 图）；计数用 e1 的
        return DataPaths.fixtures() if a.task == "count" else tiny_paths()
    return DataPaths(root=Path(a.data_root or os.environ["VISINT_DATA_ROOT"]))


def build_model(a):
    entry = registry.get_model(a.model)
    policy = default_image_policy(entry.family)
    kw = {"init_std": TINY_INIT_STD} if a.tiny else {}
    ad = make_adapter(a.model, device=a.device, dtype=DTYPES[a.dtype], image_policy=policy, tiny=a.tiny, **kw)
    return entry, policy, ad


def run_spec(a, entry, policy, adapter_key: str) -> R.W2RunSpec:
    return R.W2RunSpec(model_key=entry.key, family=entry.family, repo=entry.repo, revision=entry.revision, adapter_model_key=adapter_key,
                       task=a.task, split=a.split, setting=a.setting, tiny=a.tiny, limit=a.limit,
                       image_policy={"tag": "default", "type": type(policy).__name__, **policy.as_dict()})


def e1_batch_size(a, prompts) -> int:
    frozen = prompts.batch_protocol()["batch_size"]
    bs = (a.batch_size if a.setting == "baseline_e1" else None) or (TINY_E1_BATCH_SIZE if a.tiny else frozen)
    if not a.tiny and bs != frozen and not a.allow_unfrozen_batch:
        raise ValueError(f"E1 的运行要用 E1 冻结的批大小 {frozen}（--batch-size {bs}）；确要换就加 --allow-unfrozen-batch")
    return bs


def cmd(a, argv) -> int:
    cfg = load_w2_config(a.w2_config)
    splits_doc = tiny_splits(cfg) if a.tiny else load_splits(W2_SPLITS_PATH)
    if a.split not in split_names(a.task, splits_doc):
        raise ValueError(f"--split {a.split} 不用于 {a.task}；可用：{split_names(a.task, splits_doc)}")
    paths = data_paths(a)
    samples = w2_items(a.task, a.split, paths, splits=splits_doc)
    if a.limit is not None:
        samples = samples[: a.limit]
    bat = cfg.batching(tiny=a.tiny)
    chunk = a.chunk_size or bat["chunk_size"]
    prompts = load_prompt_config(PROMPTS_PATH)
    entry = registry.get_model(a.model)
    policy = default_image_policy(entry.family)
    model_cache: dict[str, tuple] = {}

    def get_model():
        if "m" not in model_cache:
            model_cache["m"] = build_model(a)
        return model_cache["m"]

    akey = f"tiny({entry.key})" if a.tiny else entry.key  # 与 make_adapter 记的 model_key 相同；不跑模型时不加载
    e1_bs = e1_batch_size(a, prompts)
    common = dict(splits_doc=splits_doc, prompts=prompts, argv=argv)
    if a.setting == "baseline_e1":
        entry, policy, ad = get_model()
        spec = run_spec(a, entry, policy, str(getattr(ad, "model_key", entry.key)))
        m = R.run_baseline_e1(ad, entry, policy, spec, cfg, samples, paths, a.out, batch_size=e1_bs, resume=a.resume, **common)
    elif a.setting == "w2a" and a.task != "refseg":  # 按输出类型路由：要框或数字的任务按 E1 单轮作答
        spec = run_spec(a, entry, policy, akey)
        src_dir, _ = R.ensure_e1_run(get_model, spec, a.task, cfg, samples, paths, a.out, batch_size=e1_bs, **common)
        m = R.run_routed(spec, cfg, samples, a.out, splits_doc=splits_doc, src_dir=src_dir, resume=a.resume, argv=argv)
    elif a.setting == "ownbox":
        spec = run_spec(a, entry, policy, akey)
        box_dir, box_recs = R.ensure_e1_run(get_model, spec, "refexp_box", cfg, samples, paths, a.out, batch_size=e1_bs, **common)
        m = R.run_ownbox(spec, cfg, samples, a.out, splits_doc=splits_doc, box_dir=box_dir, box_recs=box_recs, prompts=prompts,
                         resume=a.resume, argv=argv)
    else:  # w2a（refseg）、random_pick、oracle_pick、w1f：同一份候选
        if a.setting == "w2a":
            ad = _load_attr(a.adapter_factory)(entry) if a.adapter_factory else get_model()[2]
            spec = run_spec(a, entry, policy, str(getattr(ad, "model_key", entry.key)))
        else:
            ad, spec = None, run_spec(a, entry, policy, akey)
        mode = PROPOSE_BOX if a.setting == "w1f" else cfg.propose_mode  # w1f 总是 E1 协议的框（= W1-F）
        if mode == PROPOSE_BOX:
            prop_dir, prop_recs = R.ensure_e1_run(get_model, spec, "refexp_box", cfg, samples, paths, a.out, batch_size=e1_bs, **common)
        else:
            prop_dir, prop_recs = R.ensure_points_run(get_model, spec, cfg, samples, paths, a.out, batch_size=e1_bs, **common)
        impl = R.load_impl(a.candidates_module, a.render_module if a.setting == "w2a" else None)
        cand_leaf, cand_rows = R.ensure_candidates(spec, cfg, samples, a.out, mode=mode, prop_dir=prop_dir, prop_recs=prop_recs,
                                                   impl=impl, chunk_size=chunk, workers=a.candidate_workers, **common)
        m = R.run_candidate_setting(ad, spec, cfg, samples, paths, a.out, splits_doc=splits_doc, mode=mode, prop_dir=prop_dir,
                                    prop_recs=prop_recs, cand_leaf=cand_leaf, cand_rows=cand_rows, impl=impl, prompts=prompts,
                                    batch_size=a.batch_size or bat["batch_size"], chunk_size=chunk,
                                    max_batch_tokens=bat["max_batch_tokens"], resume=a.resume, argv=argv)
    head = "; ".join(f"{h['name']}={h.get('value')}" for h in m.get("headline", []))
    print(f"W2 {a.model} {a.task}/{a.setting}/{a.split} n={m['n']} complete={m.get('complete')} {head}")
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
