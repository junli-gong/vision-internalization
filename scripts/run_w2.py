#!/usr/bin/env python
"""W2 的入口（DEV-0059；D113；docs/w2_plan.md §3）：一个模型 × 一个任务 × 一种设置 × 一个划分。

运行（服务器；--out 是本次运行的目录 experiments/W2/<NNNN_slug>，结果写进 <out>/<模型>/<任务>/<设置>/<划分>/）：
    python scripts/run_w2.py --model qwen3.5-9b --task refseg --split devtune800 --setting w2a \\
        --data-root $PROJ/data --out experiments/W2/0063_w2_tune [--resume] [--limit N]
设置（--setting）：
- w2a（主方法）：refseg 走 提出 → 候选 → 挑选（configs/w2.json）；refexp_box、count 按输出类型路由到 E1 单轮（= baseline_e1 的结果）。
- random_pick、oracle_pick、w1f（refseg）：同一份候选里伪随机挑、用真值挑、交 1 号（= W1-F），不跑模型（候选与框没写完时先跑，要模型）。
- ownbox（refseg）：E1 refexp_box 的框光栅化；baseline_e1：E1 零样本（refseg → refexp_poly、refexp_box、count），E1 冻结的批大小（64）。
- w2c（refseg；W2-c，DEV-0061，src/w2/w2c.py）：要给 --checkpoint <W2-c 的训练叶目录>/ckpt/final/model（scripts/train_w2c.py 的输出）。
  先做评测前的合并核对（口径同 E3 的 T 文本评测，写进 <out>/<模型>/w2c_merge_check.json，不通过退出码 1），再 load_trained(merge=True)；
  提出（训练过的模型按 E1 refexp_box 协议重新出框）→ <out>/<模型>/refexp_box/propose_w2c/<划分>（与 base 的 baseline_e1 分开），候选 →
  <out>/<模型>/refseg/candidates_w2c/<划分>，挑选（关思考）→ <out>/<模型>/refseg/w2c/<划分>；候选生成器、渲染、K 用 --w2-config 那一份
  （必须与检查点训练时的 sha256 相同）。config.yaml 记 checkpoint（路径、manifest 的 sha256、合并核对的口径）。判定：
      python scripts/w2_gate.py --main-name W2-c --main <out>/qwen3.5-9b/refseg/w2c/devcheck2 --w1f <base 的 w1f/devcheck2>
"提出"的框 = 同一个 --out 里的 <out>/<模型>/refexp_box/baseline_e1/<划分>，没写完先按 E1 的协议跑完；候选只生成一次
（<out>/<模型>/refseg/candidates/<划分>/candidates.jsonl），w2a、random_pick、oracle_pick、w1f 共用。
划分：devtune800（W1 的 800 句，可以随便用）、devcheck2（新的 500 句，只用于判定）、test_*（同 W1）——refseg、refexp_box；
devtune_countbench、test_countbench——count。清单在 configs/w2_splits.json（src/w2/data.py；devcheck2 由服务器生成）。
冻结保护（同 D108）：划分是 devcheck2 或以 test 开头时，--w2-config（默认 configs/w2.json）必须 "frozen": true，否则退出码 2，什么都不写；
对全部设置都适用；devtune800 不受限；--tiny 不受限。
别的划分清单（DEV-0067）：--splits-file（默认 configs/w2_splits.json；默认时行为与输出逐字节不变），如 configs/explore_splits.json 的
devcheck3_<总体>、explore_val_<总体>（src/data/explore_splits.py；配 --tiny 时在 tests/fixtures/explore 上按同一套规则现算）。运行身份的
splits_config 记这份文件。devcheck3 是**密封**的（D126-4、D128）：必须显式给 --unseal-devcheck3 G-EX（src/data/explore_splits.require_unsealed），
否则退出码 2，什么都不写；--tiny 也一样；也只能用冻结的配置跑（同上）。放行时门名记进运行身份（config.yaml 的 w2.unseal），放行的时间与
argv 追加进 <out>/<模型>/unseal_devcheck3.jsonl（各目录的 logs/run_*.json 另有 argv 与开始时间）。
开发机彩排：加 --tiny（小型随机模型、CPU、tests/fixtures 的小数据，划分按同一套规则现算，挑选的预算与批大小取 configs/w2.json 的 tiny）。
候选生成可以并行：--candidate-workers N（默认 1；按组用 spawn 的进程池，子进程里 OpenCV 与 BLAS 都是 1 个线程；candidates.jsonl 除
info 里的耗时外与单进程相同，不进运行身份）。
候选与渲染的实现（--candidates-module、--render-module，默认 src.w2.candidates、src.w2.render，DEV-0058）与挑选的适配器
（--adapter-factory MODULE:ATTR）可以换成测试的假实现（tests._w2_fakes），只配 --tiny。
退出码：0 = 跑完；1 = --checkpoint 的合并核对没有通过，拒绝评测；2 = 出错（报错先写进 <out>/<模型>/errors/，argparse 用法错误也是 2）。
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
from src.w2 import w2c as W2C  # noqa: E402
from src.w2.config import (  # noqa: E402
    CLI_SETTINGS,
    PROPOSE_BOX,
    SETTING_TASKS,
    W2_CONFIG_PATH,
    W2C_CANDIDATES_SETTING,
    W2C_SETTING,
    load_w2_config,
)
from src.data import explore_splits as EX  # noqa: E402
from src.w2.data import (  # noqa: E402
    W2_SPLITS_PATH,
    load_splits,
    load_splits_file,
    needs_frozen,
    split_names,
    tiny_paths,
    tiny_splits,
    w2_items,
)

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
TINY_INIT_STD = 0.05  # 与 run_e1、run_w1 相同
TINY_E1_BATCH_SIZE = 4  # 与 run_e1、run_w1 的 --tiny 默认相同
RUN_DIR_NAME = re.compile(r"[0-9]{4}_[A-Za-z0-9][A-Za-z0-9_.-]*")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="注册键（src/adapters/registry.py）")
    p.add_argument("--task", required=True, choices=TASKS)
    p.add_argument("--split", required=True, help="划分名（configs/w2_splits.json 的 splits）")
    p.add_argument("--setting", required=True, choices=CLI_SETTINGS)
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
    p.add_argument("--checkpoint", help="--setting w2c 必给：W2-c 的检查点 <训练叶目录>/ckpt/final/model（scripts/train_w2c.py）")
    p.add_argument("--merge-check-n", type=int, help="--checkpoint 的合并核对的开发样本数（默认取 configs/e3b_train.json 的 eval.merge_check.n）")
    p.add_argument("--splits-file", default=str(W2_SPLITS_PATH),
                   help="划分清单（默认 configs/w2_splits.json；如 configs/explore_splits.json，DEV-0067）")
    p.add_argument("--unseal-devcheck3", metavar="GATE", help=f"密封的 devcheck3 只在判定时用：门名必须是 {EX.GATE}（D126-4、D128）")
    a = p.parse_args(argv)
    if a.model not in registry.MODELS:
        p.error(f"未登记的模型键 {a.model!r}")
    if a.task not in SETTING_TASKS[a.setting]:
        p.error(f"--setting {a.setting} 只用于 {list(SETTING_TASKS[a.setting])}")
    if not a.tiny and (a.candidates_module != R.DEFAULT_CANDIDATES_MODULE or a.render_module != R.DEFAULT_RENDER_MODULE
                       or a.adapter_factory):
        p.error("--candidates-module、--render-module（非默认）与 --adapter-factory 只配 --tiny")
    if (a.setting == W2C_SETTING) != (a.checkpoint is not None):
        p.error("--setting w2c 必须给 --checkpoint（W2-c 的检查点），其余设置不能给")
    if a.merge_check_n is not None and a.checkpoint is None:
        p.error("--merge-check-n 只配 --checkpoint 用")
    if a.checkpoint is not None:
        if a.adapter_factory:
            p.error("--setting w2c 的提出与挑选都用加载了检查点的模型，不能配 --adapter-factory")
        if not Path(a.checkpoint).is_dir():
            p.error(f"--checkpoint {a.checkpoint!r} 不是目录")
    for name in ("batch_size", "chunk_size", "limit", "candidate_workers", "merge_check_n"):
        v = getattr(a, name)
        if v is not None and v < 1:
            p.error(f"--{name.replace('_', '-')} 必须是正整数")
    if not a.tiny and not (a.data_root or os.environ.get("VISINT_DATA_ROOT")):
        p.error("真模型运行要给 --data-root（或设 VISINT_DATA_ROOT）")
    if not a.tiny and not RUN_DIR_NAME.fullmatch(Path(a.out).name):
        p.error(f"--out 的目录名必须带运行编号，形如 0063_w2_tune，收到 {a.out!r}")
    if not a.tiny and (needs_frozen(a.split) or EX.needs_frozen(a.split)):  # 冻结保护：放在最后，其余用法错误先报
        try:
            frozen = load_w2_config(a.w2_config).data["frozen"]
        except (OSError, ValueError) as e:  # W2ConfigError、JSON 写坏都是 ValueError
            p.error(f"--w2-config {a.w2_config} 读不了：{e}")
        if frozen is not True:
            p.error(f"协议未冻结，不能碰判定集与 test（D113；同 D108）：--split {a.split} 要求 {a.w2_config} 的 \"frozen\" 为 true"
                    f"（现在是 {frozen!r}）")
    if EX.is_sealed(a.split):  # 密封守卫（DEV-0067）：--tiny 也查
        try:
            EX.require_unsealed(a.split, a.unseal_devcheck3)
        except EX.SealedSplitError as e:
            p.error(str(e))
    elif a.unseal_devcheck3 is not None:
        p.error(f"--unseal-devcheck3 只配密封的 devcheck3 划分（--split {a.split} 不是）")
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


def is_default_splits(a) -> bool:
    return Path(a.splits_file).resolve() == W2_SPLITS_PATH.resolve()


def load_splits_doc(a, cfg):
    """划分清单：默认（configs/w2_splits.json）与原来相同；--splits-file 给别的清单时读它（运行身份记它的路径），配 --tiny 时只认探索
    清单，在 fixture 上现算（src/data/explore_splits.tiny_splits）。"""
    if is_default_splits(a):
        return tiny_splits(cfg) if a.tiny else load_splits(W2_SPLITS_PATH)
    doc = load_splits_file(a.splits_file)
    if a.tiny:
        if doc.get("kind") != EX.KIND:
            raise ValueError(f"--tiny 配 --splits-file 只认探索清单（kind = {EX.KIND}），收到 {a.splits_file}")
        return EX.tiny_splits()
    return doc


def data_paths(a, splits_doc=None) -> DataPaths:
    if a.tiny and not a.data_root:  # RefCOCO 用 tests/fixtures/w2（e1_posthoc 的 48×36 图加两张 val 图）；计数用 e1 的
        if splits_doc is not None and splits_doc.get("kind") == EX.KIND:  # 探索清单的 --tiny：tests/fixtures/explore
            return EX.tiny_paths()
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
                       image_policy={"tag": "default", "type": type(policy).__name__, **policy.as_dict()},
                       unseal=getattr(a, "unseal", None))


def e1_batch_size(a, prompts) -> int:
    frozen = prompts.batch_protocol()["batch_size"]
    bs = (a.batch_size if a.setting == "baseline_e1" else None) or (TINY_E1_BATCH_SIZE if a.tiny else frozen)
    if not a.tiny and bs != frozen and not a.allow_unfrozen_batch:
        raise ValueError(f"E1 的运行要用 E1 冻结的批大小 {frozen}（--batch-size {bs}）；确要换就加 --allow-unfrozen-batch")
    return bs


def cmd(a, argv) -> int:
    cfg = load_w2_config(a.w2_config)
    splits_doc = load_splits_doc(a, cfg)
    if a.split not in split_names(a.task, splits_doc):
        raise ValueError(f"--split {a.split} 不用于 {a.task}；可用：{split_names(a.task, splits_doc)}")
    unseal = EX.require_unsealed(a.split, a.unseal_devcheck3, entry=splits_doc["splits"][a.split])  # 条目标了 sealed 也算
    if unseal is not None:
        if not a.tiny and cfg.data.get("frozen") is not True:
            raise ValueError(f"密封的 {a.split} 只能用冻结的配置跑：{a.w2_config} 的 frozen 不是 true")
        a.unseal = EX.seal_identity(unseal)
        log = Path(a.out) / a.model / "unseal_devcheck3.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps({**unseal, "split": a.split, "task": a.task, "setting": a.setting, "argv": list(argv)},
                               ensure_ascii=False) + "\n")
    paths = data_paths(a, splits_doc)
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
    elif a.setting == W2C_SETTING:  # W2-c（DEV-0061）：训练过的检查点做提出与挑选（src/w2/w2c.py）
        W2C.refuse_mixing(a.out, entry.key, a.split, a.checkpoint)
        ad, ck_info, extra = W2C.load_checkpoint_adapter(
            a.checkpoint, model_key=a.model, tiny=a.tiny, device=a.device, dtype=a.dtype, torch_dtype=DTYPES[a.dtype], w2=cfg,
            merge_check_json=Path(a.out) / a.model / W2C.MERGE_CHECK_JSON, data_root=a.data_root or os.environ.get("VISINT_DATA_ROOT"),
            merge_check_n=a.merge_check_n)
        pol = ad.image_policy
        spec = R.W2RunSpec(model_key=entry.key, family=entry.family, repo=entry.repo, revision=entry.revision,
                           adapter_model_key=str(getattr(ad, "model_key", entry.key)), task=a.task, split=a.split, setting=a.setting,
                           tiny=a.tiny, limit=a.limit, image_policy={"tag": "checkpoint", "type": type(pol).__name__, **pol.as_dict()},
                           unseal=getattr(a, "unseal", None))
        prop_dir, prop_recs = W2C.ensure_proposal(ad, entry, spec, cfg, samples, paths, a.out, checkpoint=ck_info, batch_size=e1_bs,
                                                  run_extra=extra, **common)
        impl = R.load_impl(a.candidates_module, a.render_module)
        cand_leaf, cand_rows = R.ensure_candidates(spec, cfg, samples, a.out, mode=PROPOSE_BOX, prop_dir=prop_dir, prop_recs=prop_recs,
                                                   impl=impl, chunk_size=chunk, workers=a.candidate_workers,
                                                   cand_setting=W2C_CANDIDATES_SETTING, **common)
        m = R.run_candidate_setting(ad, spec, cfg, samples, paths, a.out, splits_doc=splits_doc, mode=PROPOSE_BOX, prop_dir=prop_dir,
                                    prop_recs=prop_recs, cand_leaf=cand_leaf, cand_rows=cand_rows, impl=impl, prompts=prompts,
                                    batch_size=a.batch_size or bat["batch_size"], chunk_size=chunk,
                                    max_batch_tokens=bat["max_batch_tokens"], resume=a.resume, argv=argv, checkpoint=ck_info)
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
    from src.e3.merge_check import MergeCheckFailed

    try:
        return cmd(a, argv)
    except MergeCheckFailed as e:  # --checkpoint：评测前的合并核对没有通过（同 run_e1 --checkpoint）
        print(f"拒绝评测：{e}", file=sys.stderr)
        return 1
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
