#!/usr/bin/env python
"""统一主表的两只"手"（DEV-0074；预注册 U-1、U-2；D154）：在 W2 test 的 5 × 300 句子集（configs/w2_llmseg_subsets.json，与 LLMSeg
的 RUN-0078 同一份、同一个取样函数）上跑
- --system e1_poly_structured：9B 零样本的 E1 结构强制多边形（run_e1.py --subset forced_poly 的 refexp_poly__poly_structured：至少
  3 个点、关思考、贪心、E1 冻结的图像口径与批大小 64；D72-3）；
- --system e3b_grid：9B 的 E3b 像素 token 网格（eval_e3b.py 的 refseg 读法：单轮、结构强制解码、guided 上采样；D83-3；--checkpoint
  先过评测前的合并核对，同 eval_e3b，判据 v1，写进 <out>/<模型>/e3b_merge_check.json；不通过退出码 1）。
推理与解析调用原运行器的函数，一行不改（规则见 src/unified/hands.py）；按 W2 的口径计分，输出 W2 的叶目录
<out>/<模型>/refseg/<u_e1_poly_structured | u_e3b_grid>/<划分>/（scripts/mgr/w2_test_table.preds 直接读），原运行器的输出在它的 raw/ 下。

放行（D154）：这份子集全是 test 句子，必须给 --test-approval D154（恰是这个编号；依据 DECISIONS.md D154-2(2)，同 D147 的做法），
否则退出码 2，什么都不写（--tiny 也一样）。放行时批准编号进运行身份（config.yaml 的 w2.test_unlock；raw 的 selection_stats / stats
里的 w2 一节也有），每个划分追加一行到 <out>/<模型>/test_unlock.jsonl（时间、argv、系统、划分、检查点的 manifest sha256）。

服务器（hpcgpu06，与 RUN-0072、RUN-0092 同节点；--out 的最后一级目录名以四位运行编号开头，否则用法错误）：
    python scripts/run_u_hands.py --system e1_poly_structured --model qwen3.5-9b --split all --test-approval D154 \\
        --data-root $VI/b200/data --out experiments/W2/<NNNN>_u_hands_hpcgpu06 [--resume]
    python scripts/run_u_hands.py --system e3b_grid --model qwen3.5-9b --split all --test-approval D154 \\
        --checkpoint $VI/ckpt/hpcgpu06/0056/e3b/qwen3.5-9b/seed0/ckpt/final/model --seed 0 \\
        --data-root $VI/b200/data --out experiments/W2/<NNNN>_u_hands_hpcgpu06 [--resume]
    # 检查点：RUN-0056 训练的 E3b seed 0（manifest sha256 316d25c7…，RUN-0056 评测的 config.yaml 记的那一个；路径见那次评测的
    #   logs/run_001.json 的 checkpoint_dir）。--split 也可以是单个 test 划分；--limit N 只取每个划分子集的前 N 句（试跑，进运行身份）。
用时（A100-40GB）：
    # e1_poly_structured：W1 的 RUN-0063（hpcgpu07）同一读法 500 句 334 s（批 64，每句约 0.67 s）→ 1,500 句约 17 分钟，加载模型另约 3 分钟。
    # e3b_grid：RUN-0056 同一检查点、同一读法 10,634 句 11,354 s（每句约 1.07 s）→ 1,500 句约 27 分钟；合并核对加载入约 15 分钟（同一步
    #   12,254 s 减去 11,354 s）；评分与导出（guided 重算）约 3–5 分钟。合计约 45–50 分钟。
开发机彩排：--tiny（小型随机模型、CPU、tests/fixtures/w2 的小数据，划分与子集按同一规则现算）；e3b_grid 的 --checkpoint 用
scripts/train_e3b.py --tiny 训出来的小检查点。
退出码：0 = 跑完；1 = 合并核对没有通过，拒绝评测；2 = 出错（报错先写进 <out>/<模型>/errors/；argparse 用法错误、放行不合格也是 2，
什么都不写）。
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import importlib.util
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
from src.data.records import LoadResult  # noqa: E402
from src.e1 import posthoc_runner as PR  # noqa: E402
from src.e1.posthoc import FORCED_POLY_SUBSET, POSTHOC_PATH, load_posthoc_config  # noqa: E402
from src.e1.prompts import PROMPTS_PATH, load_prompt_config  # noqa: E402
from src.e1.runner import read_jsonl_tolerant  # noqa: E402
from src.e1.tasks import DataPaths, Selection  # noqa: E402
from src.llmseg import runner as LR  # noqa: E402
from src.unified import hands as H  # noqa: E402
from src.w1 import baselines as B  # noqa: E402
from src.w1.runner import _rel  # noqa: E402
from src.w2.data import W2_SPLITS_PATH, load_splits, tiny_paths, tiny_splits, w2_items  # noqa: E402
from src.w2.runner import W2RunSpec  # noqa: E402

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
RUN_DIR_NAME = re.compile(r"[0-9]{4}_[A-Za-z0-9][A-Za-z0-9_.-]*")
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
TINY_INIT_STD = 0.05  # 与 run_e1、run_w2、run_llmseg 相同
TINY_E1_BATCH_SIZE = 4  # 与 run_e1、run_w2 的 --tiny 默认相同
ALL = "all"
E3B_ONLY = ("checkpoint", "seed", "merge_check_n", "score_workers")


def _load_eval_e3b():
    """scripts/eval_e3b.py 的 checkpoint_block（运行身份里的检查点，同一个函数，不复制）。scripts 不是包，按路径加载。"""
    spec = importlib.util.spec_from_file_location("_eval_e3b_for_u_hands", REPO_ROOT / "scripts" / "eval_e3b.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--system", required=True, choices=H.SYSTEMS)
    p.add_argument("--model", required=True, help="注册键（src/adapters/registry.py）")
    p.add_argument("--split", required=True, help=f"W2 的 test 划分名，或 {ALL}（5 个依次跑，模型只加载一次）")
    p.add_argument("--out", required=True, help="本次运行的目录（experiments/W2/<NNNN_slug>）")
    p.add_argument("--test-approval", metavar="ID", help=f"负责人的批准编号，必须恰是 {H.TEST_APPROVAL}（D154）")
    p.add_argument("--checkpoint", help="e3b_grid 必给：E3b 的检查点 <训练叶目录>/ckpt/final/model")
    p.add_argument("--seed", type=int, help="e3b_grid 必给：检查点训练的种子（E3b 评测的 seed<k>，进运行身份）")
    p.add_argument("--variant", default="e3b", help="e3b_grid：E3b 评测的变体名（默认 e3b，同 eval_e3b）")
    p.add_argument("--merge-check-n", type=int, help="e3b_grid：合并核对的开发样本数（默认 configs/e3b_train.json 的 eval.merge_check.n）")
    p.add_argument("--score-workers", type=int, help="e3b_grid：E3b 评分的进程数（默认配置的 eval.score_workers；0 = 串行；不改结果）")
    p.add_argument("--data-root", help="落地目录（默认环境变量 VISINT_DATA_ROOT；--tiny 时取 fixture）")
    p.add_argument("--limit", type=int, help="只取每个划分子集的前 N 句（试跑；进运行身份）")
    p.add_argument("--resume", action="store_true", help="续跑：身份必须相同；原运行器的输出按它自己的规则整批续上")
    p.add_argument("--tiny", action="store_true", help="小型随机模型 + CPU + tests/fixtures 的小数据（开发机彩排）")
    p.add_argument("--device", help="默认：真模型 cuda，--tiny 时 cpu")
    p.add_argument("--dtype", choices=sorted(DTYPES), help="默认：真模型 bf16，--tiny 时 fp32")
    a = p.parse_args(argv)
    try:  # 放行（D154）：最先核对，--tiny 也查；不合格是用法错误，什么都不写
        a.test_unlock = H.check_approval(a.test_approval)
    except H.TestApprovalError as e:
        p.error(str(e))
    if a.model not in registry.MODELS:
        p.error(f"未登记的模型键 {a.model!r}")
    splits = LR.test_splits()
    if a.split != ALL and a.split not in splits:
        p.error(f"--split 只能是 {ALL} 或 W2 的 test 划分 {splits}")
    for name in ("limit", "merge_check_n"):
        v = getattr(a, name)
        if v is not None and v < 1:
            p.error(f"--{name.replace('_', '-')} 必须是正整数")
    if a.score_workers is not None and a.score_workers < 0:
        p.error("--score-workers 不能为负")
    if a.system == H.SYSTEM_E1:
        given = [n for n in E3B_ONLY if getattr(a, n) is not None]
        if given or a.variant != "e3b":
            p.error(f"{[('--' + n.replace('_', '-')) for n in given] or ['--variant']} 只配 --system {H.SYSTEM_E3B}"
                    f"（{H.SYSTEM_E1} 是零样本的基座）")
        if registry.get_model(a.model).family not in load_posthoc_config(load_prompt_config(PROMPTS_PATH)).forced_poly["families"]:
            p.error(f"结构强制的多边形只做 Qwen3.5 家族（D72-3），{a.model} 不是")
    else:
        if a.checkpoint is None or a.seed is None:
            p.error(f"--system {H.SYSTEM_E3B} 要给 --checkpoint 与 --seed")
        if not Path(a.checkpoint).is_dir():
            p.error(f"--checkpoint {a.checkpoint!r} 不是目录")
        if a.seed < 0 or not NAME.fullmatch(a.variant) or ".." in a.variant:
            p.error(f"--seed 不能为负，--variant 只能含字母、数字、_ . -（收到 {a.seed}、{a.variant!r}）")
    if not a.tiny and not (a.data_root or os.environ.get("VISINT_DATA_ROOT")):
        p.error("真模型运行要给 --data-root（或设 VISINT_DATA_ROOT）")
    if not a.tiny and not RUN_DIR_NAME.fullmatch(Path(a.out).name):
        p.error(f"--out 的目录名必须带运行编号，形如 0097_u_hands_hpcgpu06（CLAUDE.md：experiments/<phase>/<NNNN_slug>/），收到 {a.out!r}")
    a.device = a.device or ("cpu" if a.tiny else "cuda")
    a.dtype = a.dtype or ("fp32" if a.tiny else "bf16")
    return a


def load_docs(a):
    """(W2 划分清单, 子集清单, 数据路径, 记进 meta.json 的清单文件)：同 scripts/run_llmseg.py（真运行先 check_subsets 按规则重算核对）。"""
    if a.tiny:
        splits_doc = tiny_splits()
        subsets_doc = LR.build_subsets(splits_doc, splits=LR.test_splits(), source={"path": "tiny（src/w2/data.tiny_splits）"})
        return splits_doc, subsets_doc, tiny_paths(), []
    problems = LR.check_subsets()
    if problems:
        raise ValueError("；".join(problems))
    paths = DataPaths(root=Path(a.data_root or os.environ["VISINT_DATA_ROOT"]))
    return load_splits(W2_SPLITS_PATH), LR.load_subsets(), paths, [W2_SPLITS_PATH, LR.SUBSETS_PATH]


def subset_samples(split: str, splits_doc, subsets_doc, paths, limit: int | None):
    """子集的句子：src/llmseg/runner.subset_samples（W2 的取样函数 w2_items 的结果按子集的 id 取，保持 W2 的顺序），再取前 limit 句。"""
    samples = LR.subset_samples(w2_items(H.TASK, split, paths, splits=splits_doc), subsets_doc["splits"][split]["ids"])
    return samples[:limit] if limit is not None else samples


def w2_spec(a, entry, ad, policy_doc) -> W2RunSpec:
    return W2RunSpec(model_key=entry.key, family=entry.family, repo=entry.repo, revision=entry.revision,
                     adapter_model_key=str(getattr(ad, "model_key", entry.key)), task=H.TASK, split="", setting=H.SETTINGS[a.system],
                     tiny=a.tiny, limit=a.limit, image_policy=policy_doc, test_unlock=a.test_unlock)


# ---------------------------------------------------------------------------
# E1 结构强制多边形
# ---------------------------------------------------------------------------


class E1System:
    def __init__(self, a, entry):
        self.a = a
        self.entry = entry
        self.prompts = load_prompt_config(PROMPTS_PATH)
        self.posthoc = load_posthoc_config(self.prompts, POSTHOC_PATH)
        self.policy = default_image_policy(entry.family)  # E1 冻结的图像口径（D28-3）
        kw = {"init_std": TINY_INIT_STD} if a.tiny else {}
        self.ad = make_adapter(a.model, device=a.device, dtype=DTYPES[a.dtype], image_policy=self.policy, tiny=a.tiny, **kw)
        self.batch_size = TINY_E1_BATCH_SIZE if a.tiny else int(self.prompts.batch_protocol()["batch_size"])  # E1 冻结的批大小
        self.checkpoint_sha = None

    def policy_doc(self):
        return {"tag": "default", "type": type(self.policy).__name__, **self.policy.as_dict()}

    def block(self):
        fp = self.posthoc.forced_poly
        v = fp["variants"][H.E1_VARIANT]
        return {"protocol": "E1 结构强制的多边形（src/e1/posthoc_runner.run_forced_poly_task 原样）",
                "posthoc": {"config": _rel(self.posthoc.path), "config_sha256": self.posthoc.sha256, "variant": H.E1_VARIANT,
                            "min_points": int(v["min_points"]), "max_points": int(fp["max_points"]), "rule": v["rule"]},
                "prompts": {"config": _rel(self.prompts.path), "config_sha256": self.prompts.sha256},
                "enable_thinking": bool(fp["enable_thinking"]), "decoding": dict(self.prompts.decoding()), "batch_size": self.batch_size}

    def files(self):
        return [self.prompts.path, self.posthoc.path]

    def run(self, spec: W2RunSpec, leaf: Path, samples, w2, paths, *, argv):
        rs = B.e1_spec(self.entry, self.ad, self.policy, H.E1_TASK, FORCED_POLY_SUBSET, self.batch_size, tiny=self.a.tiny,
                       limit=self.a.limit)
        name = H.subset_name(spec.split)
        sel = Selection(H.E1_TASK, name, list(samples), True, {"w2": dict(w2), "n": len(samples)})
        raw_leaf = leaf / H.RAW
        m = PR.run_forced_poly_task(self.ad, rs, sel, self.prompts, self.posthoc, paths, None, variant=H.E1_VARIANT,
                                    resume=self.a.resume, argv=argv, sample_subset=name, leaf=raw_leaf)
        recs = H.e1_records(spec, samples, read_jsonl_tolerant(raw_leaf / "predictions.jsonl", repair=False))
        extra = {"forced_poly": m["forced_poly"]}
        return recs, raw_leaf, "E1 事后读数（src/e1/posthoc_runner.run_forced_poly_task）", m["headline"], extra, {}


# ---------------------------------------------------------------------------
# E3b 像素 token 网格
# ---------------------------------------------------------------------------


class E3bSystem:
    def __init__(self, a, entry):
        from src.adapters import factory
        from src.adapters.checkpoint import read_manifest
        from src.e3 import evaluate as EV
        from src.e3.config import CONFIG_PATH, load_config
        from src.e3.merge_check import run_merge_check
        from src.e3.tokens import attach_vocab

        self.a, self.entry, self.EV = a, entry, EV
        self.cfg = load_config(CONFIG_PATH)
        self.config_path = CONFIG_PATH
        # 单轮、结构强制解码（同 RUN-0056 的 eval_e3b.py --k-values 1 --decode structured）；判据 v1（默认）；批大小取配置的 eval.batch_size
        self.settings = EV.resolve_eval(self.cfg, tiny=a.tiny, score_workers=a.score_workers, merge_check_n=a.merge_check_n,
                                        k_values=list(H.E3B_K_VALUES), decode=H.E3B_DECODE)
        ck = Path(a.checkpoint)
        man = read_manifest(ck)
        bs = man.get("build_spec") or {}
        if bs.get("key") != a.model or bool(bs.get("tiny")) != a.tiny:
            raise ValueError(f"检查点的构造方式（key {bs.get('key')!r}，tiny {bs.get('tiny')}）与 --model {a.model!r}、--tiny {a.tiny} 不符")
        self.ck_block = _load_eval_e3b().checkpoint_block(ck, man)
        self.checkpoint_sha = self.ck_block["manifest_sha256"]
        mc = self.settings["merge_check"]
        self.merge = run_merge_check(ck, model_key=a.model, tiny=a.tiny, n=mc["n"], checks=mc["checks"],
                                     out_json=Path(a.out) / a.model / "e3b_merge_check.json", device=a.device, dtype=a.dtype,
                                     data_root=a.data_root or os.environ.get("VISINT_DATA_ROOT"), dev_subsets=None,
                                     e3b_config=str(CONFIG_PATH))  # 同 eval_e3b：核对读评测用的同一份配置（D67-4(a)）
        self.ad, _ = factory.load_trained(ck, device=a.device, dtype=DTYPES[a.dtype], merge=True)
        if man.get("lora") is not None and self.ad.lora_state != "merged":
            raise RuntimeError(f"load_trained(merge=True) 之后 LoRA 状态是 {self.ad.lora_state}，应为 merged")
        self.ctx = EV.make_context(self.cfg, attach_vocab(self.ad, self.cfg.vocab), H.E3B_TASK)
        self.espec = EV.EvalSpec(model_key=entry.key, family=entry.family, repo=entry.repo, revision=entry.revision,
                                 adapter_model_key=str(getattr(self.ad, "model_key", entry.key)), tiny=a.tiny, dtype=a.dtype,
                                 variant=a.variant, seed=int(a.seed), checkpoint=self.ck_block, limit=a.limit)
        self.extra = {"checkpoint_dir": str(ck), "merge_check": self.merge,
                      "gpu": torch.cuda.get_device_name(0) if a.device.startswith("cuda") and torch.cuda.is_available() else None}

    def policy_doc(self):
        pol = self.ad.image_policy
        return {"tag": "checkpoint", "type": type(pol).__name__, **pol.as_dict()}

    def upsample(self):
        up = self.settings["upsample"]
        return up["primary"], float(up["threshold"])

    def block(self):
        up, thr = self.upsample()
        return {"protocol": "E3b 评测的单轮、结构强制解码（src/e3/evaluate.run_eval_task 原样）", "variant": self.a.variant,
                "seed": int(self.a.seed), "checkpoint": dict(self.ck_block),
                "e3b_config": {"path": self.cfg.path, "sha256": self.cfg.sha256, "version": self.cfg.doc["version"]},
                "eval": self.EV.identity_settings(self.settings), "context": self.ctx.as_dict(), "upsample": up, "threshold": thr,
                "merge_check": {"criteria_version": self.EV.merge_criteria_version(self.settings), "n": self.settings["merge_check"]["n"],
                                "checks": list(self.settings["merge_check"]["checks"])}}

    def files(self):
        return [self.config_path]

    def run(self, spec: W2RunSpec, leaf: Path, samples, w2, paths, *, argv):
        EV = self.EV
        sel = LoadResult(records=list(samples), stats={"w2": dict(w2), "n": len(samples)})
        raw_out = leaf / H.RAW
        m = EV.run_eval_task(self.ad, self.cfg, self.settings, self.espec, self.ctx, sel, raw_out, resume=self.a.resume, argv=argv,
                             run_extra=self.extra)
        raw_leaf = EV.leaf_dir(raw_out, self.espec, H.E3B_TASK, self.settings)
        up, thr = self.upsample()
        recs = H.e3b_records(spec, samples, read_jsonl_tolerant(raw_leaf / EV.PREDICTIONS, repair=False),
                             read_jsonl_tolerant(raw_leaf / EV.SCORES, repair=False), n=self.ctx.n, upsample=up, threshold=thr)
        head = {k: m["headline"][k] for k in ("procedure", "upsample", "cIoU", "gIoU", "BIoU")}
        extra = {"checkpoint_manifest_sha256": self.checkpoint_sha, "decode": m.get("decode", {}).get("mode"), "upsample": up,
                 "stages": m["stages"]}
        return recs, raw_leaf, "E3b 评测（src/e3/evaluate.run_eval_task）", head, extra, {"checkpoint_dir": self.extra["checkpoint_dir"]}


def cmd(a, argv) -> int:
    a.test_unlock = H.check_approval(getattr(a, "test_approval", None))  # parse_args 已核对；这里再核对一次（直接调用 cmd 时）
    splits_doc, subsets_doc, paths, split_files = load_docs(a)
    entry = registry.get_model(a.model)
    splits = LR.test_splits() if a.split == ALL else [a.split]
    if not a.resume:  # 加载模型之前就拒绝覆盖（_prepare_leaf 之后还会再核对一次）
        busy = [str(lf) for lf in (H.leaf_of(a.out, entry.key, a.system, sp) for sp in splits) if lf.exists() and any(lf.iterdir())]
        if busy:
            raise FileExistsError(f"{busy[0]} 已存在且非空；续跑请加 --resume")
    system = E1System(a, entry) if a.system == H.SYSTEM_E1 else E3bSystem(a, entry)
    H.append_unlock(a.out, entry.key, [H.unlock_line(a.test_unlock, system=a.system, split=sp, model=entry.key, argv=argv,
                                                     checkpoint_manifest_sha256=system.checkpoint_sha) for sp in splits])
    base = w2_spec(a, entry, system.ad, system.policy_doc())
    for split in splits:
        samples = subset_samples(split, splits_doc, subsets_doc, paths, a.limit)
        spec = dataclasses.replace(base, split=split)
        w2 = H.w2_block(spec, splits_doc, subsets_doc, samples)
        leaf = H.leaf_of(a.out, entry.key, a.system, split)
        raw_dir = H.RAW if a.system == H.SYSTEM_E1 else str(system.EV.leaf_dir(Path(H.RAW), system.espec, H.E3B_TASK, system.settings))
        up = system.upsample() if a.system == H.SYSTEM_E3B else (None, None)
        cfg_doc = H.identity(spec, system=a.system, w2=w2, system_block=system.block(), raw_dir=raw_dir,
                             scoring=H.scoring_rule(a.system, upsample=up[0], threshold=up[1]))
        H.prepare(leaf, cfg_doc, [*split_files, *system.files(), *paths.config_files()], resume=a.resume)
        recs, raw_leaf, fmt, raw_head, extra, info = system.run(spec, leaf, samples, w2, paths, argv=argv)
        m = H.write_leaf(leaf, spec, a.system, cfg_doc, recs, n_expected=len(samples), raw_leaf=raw_leaf, raw_format=fmt,
                         raw_headline=raw_head, extra=extra, argv=argv, run_info=info)
        head = "; ".join(f"{h['name']}={h.get('value')}" for h in m["headline"])
        print(f"U {a.system} {entry.key} {split} n={m['n']} complete={m['complete']} {head} failures={m['failures']} "
              f"empty_mask={m['empty_mask']['n']}")
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    a = parse_args(argv)
    from src.e3.merge_check import MergeCheckFailed

    try:
        return cmd(a, argv)
    except MergeCheckFailed as e:
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
