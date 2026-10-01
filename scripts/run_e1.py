#!/usr/bin/env python
"""E1 评测框架的入口（DEV-0014）：逐模型、逐任务地跑 T 文本通道零样本基线；另有批处理验收与吞吐估算两种模式。

运行（服务器；--out 是本次运行的目录 experiments/E1/<NNNN_slug>，结果写进 <out>/<模型>/<任务>…/）：
    python scripts/run_e1.py --model qwen3.5-0.8b --task refexp_box --subset eval --batch-size 32 \\
        --data-root $PROJ/data --out experiments/E1/0011_e1_main [--resume] [--limit N]
    # 思考子集：--subset think（预算 4096）/ think8192；附加对照：--subset extra --policy qwen:1024（或 internvl:native）
批处理验收（D28-5；开发子集，batch=1 与 --batch-size 各跑一遍）：
    python scripts/run_e1.py --acceptance --model qwen3.5-0.8b --task all --batch-size 16 --data-root $PROJ/data --out <dir> \\
        [--agreement bitwise|iou90|outcome]
    # 三种一致口径（N15）都算、都写进表；--agreement 只决定 all_pass 用哪一种（默认 bitwise，D32）。
    # 已跑完的验收换口径重算：同一条命令加 --resume（两遍都已完成，不再生成，只重新比较）。
吞吐与算力估算（开发子集，每个 --batch-sizes 各测一遍，外推到各子集的 n）：
    python scripts/run_e1.py --estimate --model qwen3.5-0.8b --task all --batch-sizes 1 8 16 32 --data-root $PROJ/data \\
        --out <dir> [--estimate-think]
开发机彩排（小型随机模型、CPU、tests/fixtures/e1 的小数据）：上面任一条加 --tiny，可省 --data-root。

--task：一个或多个任务名，或 all（refexp_box refexp_poly grid16 count ovd rel_depth_nyu rel_depth_blink spatial_cvbench
spatial_embspatial）。--policy：default（E1 冻结的口径，D28-3）、qwen:1024（Qwen 约 1K token）、internvl:native（InternVL 原生
动态 tile）；后两个只用于 --subset extra 的附加对照（也可用于 --estimate 量它们的吞吐）。
--count-questions：计数的问句来源，heuristic（默认，从 caption 取对象）或 countbenchqa（big_vision 的人工问句，要先落地
countbenchqa_questions；DEV-0015 §4，项目负责人批准前不改默认）。只影响 count。
退出码：0 = 跑完；2 = 出错（报错先写进 <out>/<模型>/errors/，再退出；argparse 用法错误也是 2）。
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

from src.adapters import registry  # noqa: E402
from src.adapters.factory import default_image_policy, make_adapter  # noqa: E402
from src.adapters.image_policy import (  # noqa: E402
    E1_SUBSET_PIXEL_BUDGET_TOKENS,
    INTERNVL_NATIVE_TILES,
    PixelBudget,
)
from src.e1 import acceptance as A  # noqa: E402
from src.e1 import estimate as EST  # noqa: E402
from src.e1.prompts import (  # noqa: E402
    COUNT_QUESTION_SOURCES,
    DEFAULT_COUNT_QUESTIONS,
    PROMPTS_PATH,
    SUBSET_KINDS,
    TASKS,
    load_prompt_config,
)
from src.e1.runner import RunSpec, leaf_dir, policy_dirname, read_jsonl_tolerant, run_task  # noqa: E402
from src.e1.tasks import TASK_SUBSETS, DataPaths, default_nyu_cache, select  # noqa: E402

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}  # 与 scripts/validate_batching.py 的 --dtype 取值相同
POLICIES = ("default", "qwen:1024", "internvl:native")
POLICY_FAMILY = {"qwen:1024": registry.FAMILY_QWEN35, "internvl:native": registry.FAMILY_INTERNVL35}
TINY_INIT_STD = 0.05  # 与 validate_batching 相同：小模型重取参数，否则 greedy 输出几乎与输入无关
DEFAULT_MAP_CI_RESAMPLES = 0  # mAP 的 CI 默认不算：每次重采样都跑 COCOeval，GPU 空等；标题 F1@0.5 的 CI 照常算
RUN_DIR_NAME = re.compile(r"[0-9]{4}_[A-Za-z0-9][A-Za-z0-9_.-]*")


def resolve_policy(tag: str, family: str):
    if tag == "default":
        return default_image_policy(family)
    if POLICY_FAMILY[tag] != family:
        raise ValueError(f"口径 {tag} 只用于 {POLICY_FAMILY[tag]} 家族，这个模型是 {family}")
    return PixelBudget(E1_SUBSET_PIXEL_BUDGET_TOKENS) if tag == "qwen:1024" else INTERNVL_NATIVE_TILES


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--acceptance", action="store_true", help="批处理验收（开发子集，batch=1 与 --batch-size 各一遍）")
    mode.add_argument("--estimate", action="store_true", help="吞吐与算力估算（开发子集，--batch-sizes 各一遍）")
    p.add_argument("--model", required=True, help="注册键（src/adapters/registry.py）")
    p.add_argument("--task", nargs="+", required=True, help=f"任务名或 all：{list(TASKS)}")
    p.add_argument("--subset", default="eval", choices=SUBSET_KINDS, help="运行模式的子集（验收与估算一律用开发子集）")
    p.add_argument("--policy", default="default", choices=POLICIES, help="图像口径；默认 E1 冻结的口径（D28-3）")
    p.add_argument("--batch-size", type=int, help="批大小（默认 16；--tiny 默认 4）；1 = 逐条 generate")
    p.add_argument("--batch-sizes", type=int, nargs="+", help="--estimate 用的批大小（默认 1 8 16 32；--tiny 默认 1 2 4）")
    p.add_argument("--out", required=True, help="本次运行的目录（experiments/E1/<NNNN_slug>）")
    p.add_argument("--resume", action="store_true", help="续跑：跳过 predictions.jsonl 里已完成的 id")
    p.add_argument("--limit", type=int, help="只取每个任务的前 N 条（调试）")
    p.add_argument("--tiny", action="store_true", help="小型随机模型 + CPU + tests/fixtures/e1 的小数据（开发机彩排）")
    p.add_argument("--data-root", help="落地目录（默认环境变量 VISINT_DATA_ROOT；--tiny 时取 fixture）")
    p.add_argument("--prompts", default=str(PROMPTS_PATH), help="提示配置（默认 configs/e1_prompts.json）")
    p.add_argument("--dev-subsets", help="开发子集清单（默认 configs/e1_dev_subsets.json）")
    p.add_argument("--eval-subsets", help="评测子集清单（默认 configs/e1_eval_subsets.json）")
    p.add_argument("--nyu-pairs", help="NYU 点对配置（默认 configs/e1_nyu_pairs.json）")
    p.add_argument("--nyu-cache", help="NYU 点对渲染图的缓存目录（默认 $TMPDIR/visint_nyu_pairs_rendered）")
    p.add_argument("--device", help="默认：真模型 cuda，--tiny 时 cpu")
    p.add_argument("--dtype", choices=sorted(DTYPES), help="默认：真模型 bf16，--tiny 时 fp32")
    p.add_argument("--map-ci-resamples", type=int, help=f"检测补充指标 mAP 的 bootstrap 次数（默认 {DEFAULT_MAP_CI_RESAMPLES} = "
                                                         "不算；标题 F1@0.5 的 CI 总是算）")
    p.add_argument("--estimate-n", type=int, default=64, help="--estimate：每个任务最多用多少条开发样本（默认 64）")
    p.add_argument("--estimate-think", action="store_true", help="--estimate：另测开思考（预算 4096）的吞吐")
    p.add_argument("--estimate-think-n", type=int, default=8, help="--estimate-think 用的开发样本数（默认 8）")
    p.add_argument("--agreement", choices=A.AGREEMENTS,
                   help=f"--acceptance：all_pass 用哪种一致口径（默认 {A.DEFAULT_AGREEMENT}，D32；三种都会算出来）")
    p.add_argument("--count-questions", choices=COUNT_QUESTION_SOURCES, default=DEFAULT_COUNT_QUESTIONS,
                   help=f"计数的问句来源（默认 {DEFAULT_COUNT_QUESTIONS}；countbenchqa = big_vision 的人工问句）")
    a = p.parse_args(argv)
    if a.agreement is not None and not a.acceptance:
        p.error("--agreement 只用于 --acceptance")
    a.agreement = a.agreement or A.DEFAULT_AGREEMENT
    if a.task == ["all"]:
        a.task = list(TASKS)
    bad = [t for t in a.task if t not in TASKS]
    if bad:
        p.error(f"未知任务 {bad}；可用：{list(TASKS)} 或 all")
    if a.model not in registry.MODELS:
        p.error(f"未登记的模型键 {a.model!r}；已登记：{sorted(registry.MODELS)}")
    family = registry.get_model(a.model).family
    if a.policy != "default" and POLICY_FAMILY[a.policy] != family:
        p.error(f"--policy {a.policy} 只用于 {POLICY_FAMILY[a.policy]} 家族")
    if not (a.acceptance or a.estimate):
        unsupported = [t for t in a.task if a.subset not in TASK_SUBSETS[t]]
        if unsupported:
            p.error(f"{unsupported} 不支持 --subset {a.subset}")
        if a.subset == "extra" and a.policy == "default":
            p.error("--subset extra 是附加对照，要配 --policy qwen:1024 或 internvl:native")
        if a.subset != "extra" and a.policy != "default":
            p.error("附加口径只用于 --subset extra（主口径冻结，D28-3）")
    a.device = a.device or ("cpu" if a.tiny else "cuda")
    a.dtype = a.dtype or ("fp32" if a.tiny else "bf16")
    a.batch_size = a.batch_size or (4 if a.tiny else 16)
    a.batch_sizes = a.batch_sizes or ([1, 2, 4] if a.tiny else [1, 8, 16, 32])
    if a.map_ci_resamples is None:
        a.map_ci_resamples = DEFAULT_MAP_CI_RESAMPLES
    for name in ("batch_size", "limit", "estimate_n", "estimate_think_n"):
        v = getattr(a, name)
        if v is not None and v < 1:
            p.error(f"--{name.replace('_', '-')} 必须是正整数")
    if any(b < 1 for b in a.batch_sizes) or a.map_ci_resamples < 0:
        p.error("--batch-sizes 必须都是正整数，--map-ci-resamples 不能为负")
    if a.acceptance and a.batch_size == 1:
        p.error("--acceptance 的 --batch-size 必须 > 1（另一遍固定是 batch=1）")
    if not a.tiny and not (a.data_root or os.environ.get("VISINT_DATA_ROOT")):
        p.error("真模型运行要给 --data-root（或设 VISINT_DATA_ROOT）")
    if not a.tiny and not RUN_DIR_NAME.fullmatch(Path(a.out).name):
        p.error(f"--out 的目录名必须带运行编号，形如 0011_e1_main（CLAUDE.md：experiments/<phase>/<NNNN_slug>/），收到 {a.out!r}")
    return a


def data_paths(a) -> DataPaths:
    cache = Path(a.nyu_cache) if a.nyu_cache else default_nyu_cache()
    base = DataPaths.fixtures(nyu_cache=cache) if (a.tiny and not a.data_root) else DataPaths(
        root=Path(a.data_root or os.environ["VISINT_DATA_ROOT"]), nyu_cache=cache)
    kw = {k: Path(v) for k, v in (("dev_subsets", a.dev_subsets), ("eval_subsets", a.eval_subsets),
                                 ("nyu_pairs", a.nyu_pairs)) if v}
    return DataPaths(root=base.root, dev_subsets=kw.get("dev_subsets", base.dev_subsets),
                     eval_subsets=kw.get("eval_subsets", base.eval_subsets), nyu_pairs=kw.get("nyu_pairs", base.nyu_pairs),
                     nyu_cache=base.nyu_cache, strict=base.strict)


def build_adapter(a):
    entry = registry.get_model(a.model)
    policy = resolve_policy(a.policy, entry.family)
    kw = {"init_std": TINY_INIT_STD} if a.tiny else {}
    ad = make_adapter(a.model, device=a.device, dtype=DTYPES[a.dtype], image_policy=policy, tiny=a.tiny, **kw)
    return entry, policy, ad


def make_spec(a, entry, policy, ad, task: str, subset: str, batch_size: int) -> RunSpec:
    return RunSpec(model_key=entry.key, family=entry.family, repo=entry.repo, revision=entry.revision,
                   adapter_model_key=str(getattr(ad, "model_key", entry.key)), task=task, subset=subset,
                   policy_tag=a.policy, policy={"type": type(policy).__name__, **policy.as_dict()}, batch_size=batch_size,
                   limit=a.limit, tiny=a.tiny, map_ci_resamples=a.map_ci_resamples, count_questions=a.count_questions)


def selection(task: str, subset: str, paths: DataPaths, limit: int | None, count_questions: str = DEFAULT_COUNT_QUESTIONS):
    sel = select(task, subset, paths, count_questions=count_questions)
    if limit is not None:
        sel.samples = sel.samples[:limit]
        sel.stats["limit"] = limit
        sel.stats["n"] = len(sel.samples)
    return sel


def _headline(m) -> str:
    return "; ".join(f"{h['name']}({h['scope']})={h['value']}" for h in m["headline"]) or "（无 GT）"


def cmd_run(a, argv) -> int:
    prompts = load_prompt_config(a.prompts)
    paths = data_paths(a)
    entry, policy, ad = build_adapter(a)
    for task in a.task:
        sel = selection(task, a.subset, paths, a.limit, a.count_questions)
        spec = make_spec(a, entry, policy, ad, task, a.subset, a.batch_size)
        m = run_task(ad, spec, sel, prompts, paths, a.out, resume=a.resume, argv=argv)
        print(f"{entry.key} {task}/{a.subset} n={m['n']} parse_fail={m['parse']['failure_rate']} "
              f"trunc={m['truncation']['rate']} {_headline(m)}")
    return 0


def cmd_acceptance(a, argv) -> int:
    prompts = load_prompt_config(a.prompts)
    paths = data_paths(a)
    entry, policy, ad = build_adapter(a)
    base = Path(a.out) / entry.key / ("acceptance" if a.policy == "default" else f"acceptance__{policy_dirname(a.policy)}")
    rows = []
    for task in a.task:
        sel = selection(task, "dev", paths, a.limit, a.count_questions)
        recs = {}
        for bs in (1, a.batch_size):
            spec = make_spec(a, entry, policy, ad, task, "dev", bs)
            sub = base / f"bs{bs}"
            run_task(ad, spec, sel, prompts, paths, sub, resume=a.resume, argv=argv)
            recs[bs] = read_jsonl_tolerant(leaf_dir(sub, entry.key, task, "dev", a.policy) / "predictions.jsonl",
                                           repair=False)
        row = A.compare(task, sel.samples, recs[1], recs[a.batch_size], has_gt=sel.has_gt, batch_size=a.batch_size,
                        agreement=a.agreement)
        rows.append(row)
        by = " ".join(f"{m}={row['agreement_by_mode'][m]['rate']}/{row['agreement_by_mode'][m]['pass']}" for m in A.AGREEMENTS)
        print(f"[acceptance {entry.key} {task}] agree({a.agreement})={row['answer_agreement']} [{by}] "
              f"raw_equal={row['raw_text_equal_rate']} diff={row['diff']} ci0={row['diff_ci_contains_0']} pass={row['pass']}")
    header = {"model": entry.key, "revision": entry.revision, "policy": a.policy, "device": a.device, "dtype": a.dtype,
              "tiny": a.tiny, "prompt_config_sha256": prompts.sha256, "batch_size": a.batch_size,
              "agreement": a.agreement, "count_questions": a.count_questions}
    by_mode = {m: A.overall(rows, m) for m in A.AGREEMENTS}
    doc = {"header": header,
           "criteria": {"answer_agreement_min": A.AGREEMENT_MIN, "mask_iou_agree": A.AGREE_MASK_IOU,
                        "box_iou_agree": A.AGREE_BOX_IOU, "metric": "配对差的 95% CI 含 0",
                        "agreement": a.agreement, "agreement_rules": dict(A.AGREEMENT_RULES)},
           "agreement": a.agreement, "all_pass": by_mode[a.agreement]["all_pass"],
           "not_evaluated": by_mode[a.agreement]["not_evaluated"], "all_pass_by_agreement": by_mode, "tasks": rows,
           "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    base.mkdir(parents=True, exist_ok=True)
    (base / "acceptance.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (base / "acceptance.md").write_text(A.summary_markdown(entry.key, rows, header=header, agreement=a.agreement),
                                        encoding="utf-8")
    print(f"all_pass({a.agreement})={doc['all_pass']} not_evaluated={doc['not_evaluated']} "
          f"by_agreement={ {m: v['all_pass'] for m, v in by_mode.items()} } -> {base / 'acceptance.json'}")
    return 0


def cmd_estimate(a, argv) -> int:
    prompts = load_prompt_config(a.prompts)
    paths = data_paths(a)
    entry, policy, ad = build_adapter(a)
    base = Path(a.out) / entry.key / ("estimate" if a.policy == "default" else f"estimate__{policy_dirname(a.policy)}")
    rows = []
    for task in a.task:
        rows.append(EST.estimate_task(ad, task, entry.family, prompts, paths, batch_sizes=a.batch_sizes,
                                      n_dev=a.estimate_n, think=a.estimate_think, n_think=a.estimate_think_n,
                                      policy_tag=a.policy, count_questions=a.count_questions,
                                      log=lambda m: print(m, file=sys.stderr, flush=True)))
    header = {"model": entry.key, "revision": entry.revision, "policy": a.policy, "count_questions": a.count_questions,
              "image_policy": {"type": type(policy).__name__, **policy.as_dict()}, "device": a.device, "dtype": a.dtype,
              "tiny": a.tiny, "prompt_config_sha256": prompts.sha256, "batch_sizes": a.batch_sizes,
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() and a.device.startswith("cuda") else None}
    totals = {}
    for k in ("eval", "think", "think8192", "extra"):  # 没有任何一项可外推时记 null，不记 0
        vals = [r["gpu_hours"][k] for r in rows if r["gpu_hours"].get(k) is not None]
        totals[k] = sum(vals) if vals else None
    doc = {"header": header, "mem_headroom": EST.MEM_HEADROOM, "tasks": rows, "gpu_hours_total": totals,
           "argv": list(argv), "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    base.mkdir(parents=True, exist_ok=True)
    (base / "estimate.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (base / "estimate.md").write_text(EST.summary_markdown(entry.key, a.policy, rows, header), encoding="utf-8")
    print(f"gpu_hours_total={totals} -> {base / 'estimate.json'}")
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    a = parse_args(argv)
    try:
        if a.acceptance:
            return cmd_acceptance(a, argv)
        if a.estimate:
            return cmd_estimate(a, argv)
        return cmd_run(a, argv)
    except Exception as e:  # noqa: BLE001 — 证据先落盘，再以非零码退出
        err_dir = Path(a.out) / a.model / "errors"
        err_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = err_dir / f"error_{stamp}.json"
        path.write_text(json.dumps({"argv": argv, "error": repr(e), "traceback": traceback.format_exc()[-6000:]},
                                   indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"失败：{e!r}（已写出 {path}）", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
