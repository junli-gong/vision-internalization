#!/usr/bin/env python
"""W2-c 的训练数据（DEV-0060；D113 第二节第 4 条；格式见 docs/w2_plan.md §5；规则见 src/w2/w2c_data.py）。

RefCOCO、RefCOCO+、RefCOCOg 的 train 句子（E3 的训练池，按 COCO 图像 id 排除 W2 的全部评测图）里按
hash_rank(salt = w2c_train_v1, purpose = w2c_train) 取前 --n 句（默认 16,000）；base 模型按 E1 refexp_box 协议出框（E1 的运行器原样），
--w2-config 的候选生成器出 K 个候选，真值 IoU 定标签。两步可以拆开跑（--stage propose、--stage candidates），也可以一条命令跑完
（--stage all，默认）。每一步都可续跑：同一条命令再跑一遍就接着做（身份与清单必须相同，不同就拒绝混写）。

服务器上的命令（hpcgpu07；$VI = /home/mail/chenyuma/visint；--out 是本次运行的目录 experiments/W2/<NNNN_slug>，目录名带运行编号）：

  环境：HF_HUB_OFFLINE=1 TMPDIR=$VI/tmp PYTHONHASHSEED=0（同 RUN-0065、RUN-0066）。

  1. 出框（GPU，9B，不要求 W2 配置冻结，可以在 W2-a 调参期间先跑）：
       python scripts/build_w2c_data.py --stage propose --model qwen3.5-9b \\
           --train-data-root $VI/data/train_data --eval-data-root $VI/b200/data --out experiments/W2/<NNNN>_w2c_data
     E1 refexp_box、关思考、贪心、批 64（E1 冻结的批大小）。用时估计：RUN-0063 在 07 上 500 句 92 s（含加载模型），
     约 0.12 s/句，16,000 句约 30–40 分钟；显存峰值约 27 GB（同 RUN-0063 的 27,287 MiB）。
  2. 候选与标签（CPU，要求 --w2-config 已冻结："frozen": true；8 个核，srun -c 8）：
       OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \\
       python scripts/build_w2c_data.py --stage candidates --model qwen3.5-9b \\
           --train-data-root $VI/data/train_data --eval-data-root $VI/b200/data --out experiments/W2/<NNNN>_w2c_data \\
           --candidate-workers 8
     用时估计：DEFAULT_SPEC 每句约 4.5 s 的单核 CPU 时间（RUN-0065 的基准，中位数 4.55 s），8 个进程约 16,000 × 4.5 / 8 ≈ 9,000 s，
     即约 2.5 小时（另加每片写盘与 IoU，几分钟）；每片 1,000 句约 9–10 分钟，写完一片才算数，中断后同一条命令续跑。
     --stop-after-shards K：这一次至多写 K 片就停（分段跑；之后同一条命令接着写）。
  一条命令跑完：--stage all（默认；要求配置已冻结；出框写完就不再跑模型）。

  输出（<out>/）：propose/（E1 格式的运行目录）、w2c_data/manifest.json（总体、去重、抽样各步的条数与 sha256，选中的 id 与它们的图）、
  w2c_data/00000.jsonl…（每片 1,000 句，§5 的格式）、config.yaml（运行身份）、meta.json、metrics.json（出框失败率、标签分布、
  最优候选与 1 号的平均 IoU……）、README.md、logs/（timing.jsonl：每句候选生成的墙钟；run_<k>.json）。
  核对：w2c_data/manifest.json 的 checks.selected_on_w2_eval_images = 0（src/w2/w2c_data.check_manifest 不要数据也能核对）。

开发机彩排：--tiny（小型随机 Qwen、CPU；训练数据 tests/fixtures/w2c/root，评测划分按 W2 的同一套规则在 tests/fixtures 上现算；
默认 --n 10、片大小 4；--tiny 不要求配置冻结）。随机模型给不出框，测试里把提出的记录换成合法的框再跑候选（tests/test_w2c_data.py）。
--candidates-module（非默认）与 --shard-size 只配 --tiny。
退出码：0 = 跑完（--stop-after-shards 停下也是 0，metrics.json 的 complete = false）；2 = 出错（报错写进 <out>/errors/；用法错误、
配置未冻结也是 2，那时什么都不写）。
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
from src.e1.prompts import PROMPTS_PATH, load_prompt_config  # noqa: E402
from src.e1.tasks import DataPaths  # noqa: E402
from src.w2 import runner as R  # noqa: E402
from src.w2 import w2c_data as D  # noqa: E402
from src.w2.config import W2_CONFIG_PATH, load_w2_config  # noqa: E402

STAGES = ("all", "propose", "candidates")
DEFAULT_MODEL = "qwen3.5-9b"  # 判定用 9B（D113）
TINY_MODEL = "qwen3.5-0.8b"  # --tiny 的默认（与 W2 的命令行测试相同：开发机只缓存了它的配置与分词器）
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
TINY_INIT_STD = 0.05  # 与 run_e1、run_w1、run_w2 相同
TINY_E1_BATCH_SIZE = 4  # 与 run_e1、run_w1、run_w2 的 --tiny 默认相同
RUN_DIR_NAME = re.compile(r"[0-9]{4}_[A-Za-z0-9][A-Za-z0-9_.-]*")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=STAGES, default="all", help="propose = 出框（GPU）；candidates = 候选与标签（CPU）；all = 两步")
    p.add_argument("--model", help=f"注册键（src/adapters/registry.py），默认 {DEFAULT_MODEL}（--tiny 默认 {TINY_MODEL}）")
    p.add_argument("--out", required=True, help="本次运行的目录（experiments/W2/<NNNN_slug>）")
    p.add_argument("--train-data-root", help="训练数据的落地根（含 refcoco_train/ 与 dedupe/train_exclusions.json；服务器上 $VI/data/train_data）")
    p.add_argument("--eval-data-root", help="E1 的落地根（W2 的评测划分；默认环境变量 VISINT_DATA_ROOT；服务器上 $VI/b200/data）")
    p.add_argument("--n", type=int, help=f"抽多少句（默认 {D.DEFAULT_N}；--tiny 默认 {D.TINY_N}）")
    p.add_argument("--shard-size", type=int, help=f"每片的句数（固定 {D.SHARD_SIZE}；只配 --tiny，默认 {D.TINY_SHARD_SIZE}）")
    p.add_argument("--candidate-workers", type=int, default=1,
                   help="候选生成的进程数（默认 1；>1 时 spawn 进程池、子进程单线程；结果与单进程逐字节相同，不进运行身份）")
    p.add_argument("--stop-after-shards", type=int, help="这一次至多写 K 片就停（分段跑）")
    p.add_argument("--w2-config", default=str(W2_CONFIG_PATH), help="候选生成器的清单（默认 configs/w2.json；非 --tiny 的候选一步要求已冻结）")
    p.add_argument("--batch-size", type=int, help="出框的批大小（默认 E1 冻结的 64；--tiny 默认 4）")
    p.add_argument("--allow-unfrozen-batch", action="store_true", help="出框允许用 E1 冻结值以外的批大小（不是协议口径）")
    p.add_argument("--tiny", action="store_true", help="小型随机模型 + CPU + tests/fixtures 的小数据（开发机彩排）")
    p.add_argument("--device", help="默认：真模型 cuda，--tiny 时 cpu")
    p.add_argument("--dtype", choices=sorted(DTYPES), help="默认：真模型 bf16，--tiny 时 fp32")
    p.add_argument("--candidates-module", default=R.DEFAULT_CANDIDATES_MODULE, help="候选生成器的模块（默认真实现；别的只配 --tiny）")
    a = p.parse_args(argv)
    a.model = a.model or (TINY_MODEL if a.tiny else DEFAULT_MODEL)
    if a.model not in registry.MODELS:
        p.error(f"未登记的模型键 {a.model!r}")
    for name in ("n", "shard_size", "candidate_workers", "batch_size"):
        v = getattr(a, name)
        if v is not None and v < 1:
            p.error(f"--{name.replace('_', '-')} 必须是正整数")
    if a.stop_after_shards is not None and a.stop_after_shards < 0:
        p.error("--stop-after-shards 不能是负数")
    if not a.tiny:
        if a.candidates_module != R.DEFAULT_CANDIDATES_MODULE or a.shard_size is not None:
            p.error("--candidates-module（非默认）与 --shard-size 只配 --tiny")
        if not a.train_data_root:
            p.error("真模型运行要给 --train-data-root")
        if not (a.eval_data_root or os.environ.get("VISINT_DATA_ROOT")):
            p.error("真模型运行要给 --eval-data-root（或设 VISINT_DATA_ROOT）")
        if not RUN_DIR_NAME.fullmatch(Path(a.out).name):
            p.error(f"--out 的目录名必须带运行编号，形如 0067_w2c_data，收到 {a.out!r}")
    if not a.tiny and a.stage in ("all", "candidates"):  # 冻结保护：放在最后，其余用法错误先报；出框一步不读候选清单，不受限
        try:
            frozen = load_w2_config(a.w2_config).data["frozen"]
        except (OSError, ValueError) as e:  # W2ConfigError、JSON 写坏都是 ValueError
            p.error(f"--w2-config {a.w2_config} 读不了：{e}")
        if frozen is not True:
            p.error(f"W2 的配置未冻结，不能生成 W2-c 的候选与标签（候选生成器要用 W2-a 冻结的那一份）：{a.w2_config} 的 \"frozen\" "
                    f"应为 true（现在是 {frozen!r}）；--stage propose 不受限")
    a.n = a.n or (D.TINY_N if a.tiny else D.DEFAULT_N)
    a.shard_size = a.shard_size or (D.TINY_SHARD_SIZE if a.tiny else D.SHARD_SIZE)
    a.device = a.device or ("cpu" if a.tiny else "cuda")
    a.dtype = a.dtype or ("fp32" if a.tiny else "bf16")
    return a


def w2c_paths(a) -> D.W2CPaths:
    if a.tiny and not a.train_data_root and not a.eval_data_root:
        return D.W2CPaths.tiny()
    if a.tiny:
        raise ValueError("--tiny 用 fixture 的数据，不收 --train-data-root、--eval-data-root")
    return D.W2CPaths(train_root=Path(a.train_data_root), eval=DataPaths(root=Path(a.eval_data_root or os.environ["VISINT_DATA_ROOT"])))


def build_model(a):
    entry = registry.get_model(a.model)
    policy = default_image_policy(entry.family)
    kw = {"init_std": TINY_INIT_STD} if a.tiny else {}
    ad = make_adapter(a.model, device=a.device, dtype=DTYPES[a.dtype], image_policy=policy, tiny=a.tiny, **kw)
    return entry, policy, ad


def e1_batch_size(a, prompts) -> int:
    frozen = prompts.batch_protocol()["batch_size"]
    bs = a.batch_size or (TINY_E1_BATCH_SIZE if a.tiny else frozen)
    if not a.tiny and bs != frozen and not a.allow_unfrozen_batch:
        raise ValueError(f"出框要用 E1 冻结的批大小 {frozen}（--batch-size {bs}）；确要换就加 --allow-unfrozen-batch")
    return bs


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def cmd(a, argv) -> int:
    cfg = load_w2_config(a.w2_config)
    paths = w2c_paths(a)
    out = Path(a.out)
    manifest, samples = D.prepare(cfg, paths, n=a.n, tiny=a.tiny)
    msha = D.ensure_manifest(out, manifest)
    st = manifest["steps"]
    log(f"[W2-c] 总体 {st['population']['n']} 句（E3 的剔除清单去掉 {st['e3_exclusions']['n_sentences_removed']} 句）；W2 评测图 "
        f"{st['w2_eval_images']['n_images']} 张，去掉 {st['w2_eval_exclusion']['n_removed']} 句；取 {st['selected']['n']} 句；"
        f"manifest.json sha256 {msha}")
    prompts = load_prompt_config(PROMPTS_PATH)
    entry = registry.get_model(a.model)
    done = D.propose_complete(out, samples, model_key=entry.key, revision=entry.revision, tiny=a.tiny, manifest_sha256=msha)
    if a.stage in ("all", "propose"):
        if done:
            log(f"[W2-c] 出框已跑完：{D.propose_leaf(out)}")
        else:
            entry, policy, ad = build_model(a)
            m = D.run_propose(ad, entry, policy, samples, out, manifest_sha256=msha, paths=paths.eval, prompts=prompts,
                              batch_size=e1_batch_size(a, prompts), tiny=a.tiny, argv=argv, log=log)
            fails = m.get("parse", {}).get("failures")
            log(f"[W2-c] 出框写完：{D.propose_leaf(out)}；n = {m.get('n')}，解析失败 {fails}")
    if a.stage in ("all", "candidates"):
        m = D.run_candidates(cfg, entry, samples, out, manifest=manifest, manifest_sha256=msha, train_root=paths.train_root,
                             prompts=prompts, tiny=a.tiny, shard_size=a.shard_size, workers=a.candidate_workers,
                             candidates_module=a.candidates_module, eval_files=paths.eval.config_files(),
                             stop_after_shards=a.stop_after_shards, argv=argv, log=log)
        print(f"W2-c {a.model} n={m['n']}/{m['n_expected']} shards={m['n_shards']['done']}/{m['n_shards']['expected']} "
              f"complete={m['complete']} proposal_failure_rate={m['proposal']['failure_rate']} "
              f"best_iou={m['best_iou']['mean_on_rows_with_candidates']} first_iou={m['first_iou']['mean_on_rows_with_candidates']}")
    else:
        print(f"W2-c {a.model} propose n={len(samples)} done")
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    a = parse_args(argv)
    try:
        return cmd(a, argv)
    except Exception as e:  # noqa: BLE001 — 证据先落盘，再以非零码退出
        err_dir = Path(a.out) / "errors"
        err_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = err_dir / f"error_{stamp}.json"
        path.write_text(json.dumps({"argv": argv, "error": repr(e), "traceback": traceback.format_exc()[-6000:]},
                                   indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"失败：{e!r}（已写出 {path}）", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
