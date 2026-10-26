#!/usr/bin/env python
"""W2-c 加回放的回放数据（DEV-0065；D125-2 (b)；规则见 src/w2/w2c_replay.py）：base 9B 自己对 2,000 条通用指令（databricks-dolly-15k，
与健康检查、lang_battery、MMLU-Pro、GSM8K、IFEval 的题不重叠）与 2,000 条计数提示（W2-c 训练图 × 图里有标注的 COCO 类别，E1 计数提示原文；
与 CountBench 的图不重叠）的回答，贪心、关思考；通用的上限 512 个 token，计数 16 个；截断的照实记，训练时不用。

阶段（--stage）：
- fetch：只下载原始文件到 --raw-dir（dolly 与三个评测来源，固定 revision，核对 sha256；要联网，HPC 上在 hpcmgt 跑，经 hf-mirror）；
- plan：选提示、去重与不重叠的核对 → <out>/replay/manifest.json（CPU，逐字节可复现；已有就必须逐字节相同）；
- general、count：base 生成回答（GPU），可续跑；
- all（默认）：plan → general → count → metrics.json、compute.json、README.md。
每个阶段都可以中断后用同一条命令接着跑（运行身份与 manifest 必须相同，不同就拒绝混写）。

服务器上的命令（hpcgpu07，D125-5 的排期：07 上 RUN-0073 之后；$VI = /home/mail/chenyuma/visint；--out 是本次运行的目录，目录名带运行编号；
环境同 RUN-0069：HF_HOME=$VI/hf HF_HUB_CACHE=$VI/hf/hub HF_HUB_OFFLINE=1 TMPDIR=$VI/tmp PYTHONHASHSEED=0 NLTK_DATA=$VI/b200/nltk_data）：

  0. 原始文件（hpcmgt，约 18 MB，1 分钟）：
       env -u HF_HUB_OFFLINE HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \\
       python scripts/build_w2c_replay.py --stage fetch --raw-dir $VI/data/raw/w2c_replay
  1. 回放数据（GPU，9B）：
       python scripts/build_w2c_replay.py --model qwen3.5-9b --raw-dir $VI/data/raw/w2c_replay \\
           --w2c-data experiments/W2/0067_w2c_data_hpcgpu06/w2c_data --train-data-root $VI/data/train_data \\
           --eval-data-root $VI/b200/data --out experiments/W2/<NNNN>_w2c_replay_data
     用时估计：plan 约 1–2 分钟（读 E3 训练池 287,518 句与参照集）；加载模型约 2 分钟；通用 2,000 条 × 至多 512 个 token、批 64（32 批，
     每批至多 512 个解码步）约 10–20 分钟；计数 2,000 条 × 一张原图 × 至多 16 个 token、批 64 约 3–5 分钟（RUN-0063：E1 refexp_box
     约 0.12 s/句）。合计约 20–30 分钟，按 1 GPU 小时预留；显存峰值约 27 GB（E1 批 64 的 9B，RUN-0063、RUN-0069）。
     核对：metrics.json 的 complete = true；replay/manifest.json 的 general.removed.eval_overlap 与 count.countbench 照实报
     （开发机按固定 revision 实测：dolly 15,011 行，too_long 157、structural_marker 0、eval_overlap 15、duplicate 209，剩 14,630 行）。
  2. 训练（scripts/train_w2c.py 的文档字符串；两条回放流，每步 8 + 8 + 3 + 2，权重 1 : 1 : 0.5 : 0.5；约 7–8 GPU 小时）：
       python scripts/train_w2c.py --config configs/w2c_replay_train.json --model qwen3.5-9b \\
           --w2c-data experiments/W2/0067_w2c_data_hpcgpu06/w2c_data --replay-data experiments/W2/<NNNN>_w2c_replay_data \\
           --data-root $VI/data/train_data --out experiments/W2/<NNNN>_w2c_replay_train
     CK=experiments/W2/<NNNN>_w2c_replay_train/w2c/qwen3.5-9b/seed0/ckpt/final/model（与 W2-c 同格式、variant 仍是 w2c）
  3. test 的 5 个划分（W2 预注册 §11；只作描述，与 W2-c 并列；同 RUN-0072 的 test_w2.sh 里 w2c 那一步，--out 换新目录）：
       for SP in test_refcoco_unc_testA test_refcoco_unc_testB test_refcoco+_unc_testA test_refcoco+_unc_testB test_refcocog_umd_test; do
         python scripts/run_w2.py --model qwen3.5-9b --data-root $VI/b200/data --split $SP --w2-config configs/w2.json \\
             --out experiments/W2/<NNNN>_w2c_replay_test --candidate-workers 8 --task refseg --setting w2c --checkpoint $CK
       done
     用时估计：合并核对一次约 10 分钟（之后各划分复用）；每个划分 1,000 句：出框约 4 分钟、候选约 10 分钟（8 进程）、挑选约 5 分钟
     （RUN-0069 的 devcheck2 500 句 673 s）；5 个划分约 2 小时。W1-F 等对照用 RUN-0072 的同一批划分（同一份冻结的 configs/w2.json）。
  4. 健康检查（简报 8.11，协议 v3，固定 200 题；与 W2-c 同一条命令）：
       python scripts/health_check.py --model qwen3.5-9b --checkpoint $CK --data-root $VI/b200/data --out <目录>/health_w2c_replay.json
       python scripts/health_check.py --compare <同一节点的 health_base.json> <目录>/health_w2c_replay.json --out <目录>/health_compare.json
     每次约 18–19 分钟（RUN-0069：base 1,131 s、W2-c 1,065 s）。基座要在同一节点上（D93-1）：W2-c 的基座读数在 06 上，回放变体在 07 上
     训练与评测时，07 上另跑一次 health_check.py --model qwen3.5-9b（不带 --checkpoint）。大题量的语言读数（DEV-0064 的
     scripts/lang_battery.py）同样加 --checkpoint $CK，命令与 W2-c 的相同。
  5. 定位与计数的 T 文本读数（口径同 E3 的 C3 对照表，E1 的 eval 子集）：
       python scripts/run_e1.py --model qwen3.5-9b --task refexp_box count --subset eval --data-root $VI/b200/data \\
           --checkpoint $CK --checkpoint-label t_text/w2c_replay/seed0 --out experiments/W2/<NNNN>_w2c_replay_ttext
     约 1.7 小时（RUN-0069 的 ttext_w2c 5,998 s：refexp_box 56,857 句 + count 442 张）；同一节点上没有 9B 基座的 T 文本读数时，同一个
     --out 去掉 --checkpoint 与 --checkpoint-label 再跑一次基座（约 2.1 小时，RUN-0069 的 ttext_base 7,730 s）。
  合计（07，一个种子）：回放数据约 0.5、训练约 7.5、test 约 2、健康检查约 0.3（同节点基座另 0.3）、T 文本约 1.7（同节点基座另 2.1）GPU
  小时，约 12–15 GPU 小时；在 D125-6 批准的约 25 GPU 小时之内（与 LLMSeg 合计）。

开发机彩排（--tiny：小型随机 Qwen、CPU、fp32；原始文件 tests/fixtures/w2c_replay/raw（测试造的，不是真数据；没有 MMLU-Pro、GSM8K 时只记
missing），训练图与标注 tests/fixtures/e3/root，CountBench tests/fixtures/e1/root；默认 6 + 6 条、上限 8 / 4 个 token、批 4。随机模型不会停，
测试把回答换成以停止 token 结尾再训练，见 tests/test_w2c_replay.py）：
    python -m tests._w2c_fakes <dir>/w2c_data
    python scripts/build_w2c_replay.py --tiny --w2c-data <dir>/w2c_data --out <dir>/replay_run
退出码：0 = 跑完（或 fetch / plan 完成）；2 = 出错（报错写进 <out>/errors/；用法错误也是 2）。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from src.adapters import registry  # noqa: E402
from src.adapters.checkpoint import policy_spec  # noqa: E402
from src.adapters.factory import default_image_policy, make_adapter  # noqa: E402
from src.e1.posthoc_runner import _finish_run_log, _start_run_log  # noqa: E402
from src.e1.prompts import PROMPTS_PATH, load_prompt_config  # noqa: E402
from src.e1.tasks import DataPaths  # noqa: E402
from src.e3.data import FIXTURE_REFSEG_ROOT  # noqa: E402
from src.health import protocol as HP  # noqa: E402
from src.w2 import w2c_data as WD  # noqa: E402
from src.w2 import w2c_replay as RP  # noqa: E402

STAGES = ("all", "fetch", "plan", "general", "count")
DEFAULT_MODEL = "qwen3.5-9b"
TINY_MODEL = "qwen3.5-0.8b"
DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
TINY_INIT_STD = 0.05  # 与 run_e1、build_w2c_data 相同
TINY_BATCH_SIZE = 4  # 与 run_e1、build_w2c_data 的 --tiny 默认相同
RUN_DIR_NAME = re.compile(r"[0-9]{4}_[A-Za-z0-9][A-Za-z0-9_.-]*")
TINY_TRAIN_ROOT = FIXTURE_REFSEG_ROOT  # 与 scripts/train_w2c.py --tiny 的数据根相同（tests/_w2c_fakes 的图）


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=STAGES, default="all")
    p.add_argument("--model", help=f"注册键（默认 {DEFAULT_MODEL}；--tiny 默认 {TINY_MODEL}）")
    p.add_argument("--out", help="本次运行的目录（experiments/W2/<NNNN_slug>；fetch 不要）")
    p.add_argument("--raw-dir", help="原始文件目录（<raw>/<仓库名 / 换成 __>/<文件>；--tiny 默认 tests/fixtures/w2c_replay/raw）")
    p.add_argument("--w2c-data", help="DEV-0060 的 w2c_data 目录（*.jsonl 分片 + manifest.json）")
    p.add_argument("--train-data-root", help="训练数据的落地根（refcoco_train/、dedupe/；W2-c 行的 image 相对它；--tiny 默认 tests/fixtures/e3/root）")
    p.add_argument("--eval-data-root", help="E1 的落地根（countbench/MANIFEST.json；默认环境变量 VISINT_DATA_ROOT；--tiny 默认 tests/fixtures/e1/root）")
    p.add_argument("--n-general", type=int, help=f"通用指令的条数（默认 {RP.DEFAULT_N['general']}；--tiny {RP.TINY_N['general']}）")
    p.add_argument("--n-count", type=int, help=f"计数提示的条数（默认 {RP.DEFAULT_N['count']}；--tiny {RP.TINY_N['count']}）")
    p.add_argument("--batch-size", type=int, help="生成的批大小（默认 E1 冻结的批大小 64；--tiny 4）")
    p.add_argument("--tiny", action="store_true", help="小型随机模型 + CPU + fixtures（开发机彩排）")
    p.add_argument("--device", help="默认：真模型 cuda，--tiny 时 cpu")
    p.add_argument("--dtype", choices=sorted(DTYPES), help="默认：真模型 bf16，--tiny 时 fp32")
    a = p.parse_args(argv)
    a.model = a.model or (TINY_MODEL if a.tiny else DEFAULT_MODEL)
    if a.model not in registry.MODELS:
        p.error(f"未登记的模型键 {a.model!r}")
    for name in ("n_general", "n_count", "batch_size"):
        v = getattr(a, name)
        if v is not None and v < 1:
            p.error(f"--{name.replace('_', '-')} 必须是正整数")
    if a.stage == "fetch":
        if not a.raw_dir:
            p.error("--stage fetch 要给 --raw-dir")
        return a
    if not a.out:
        p.error("要给 --out")
    if not a.tiny:
        for name in ("raw_dir", "w2c_data", "train_data_root"):
            if not getattr(a, name):
                p.error(f"真模型运行要给 --{name.replace('_', '-')}")
        if not (a.eval_data_root or os.environ.get("VISINT_DATA_ROOT")):
            p.error("真模型运行要给 --eval-data-root（或设 VISINT_DATA_ROOT）")
        if not RUN_DIR_NAME.fullmatch(Path(a.out).name):
            p.error(f"--out 的目录名必须带运行编号，形如 0074_w2c_replay_data，收到 {a.out!r}")
    elif not a.w2c_data:
        p.error("--tiny 也要给 --w2c-data（python -m tests._w2c_fakes <dir> 造的假数据）")
    a.raw_dir = a.raw_dir or str(RP.TINY_RAW_DIR)
    a.train_data_root = a.train_data_root or str(TINY_TRAIN_ROOT)
    a.eval_data_root = a.eval_data_root or (str(RP.TINY_EVAL_ROOT) if a.tiny else os.environ["VISINT_DATA_ROOT"])
    n = RP.TINY_N if a.tiny else RP.DEFAULT_N
    a.n_general = a.n_general or n["general"]
    a.n_count = a.n_count or n["count"]
    a.device = a.device or ("cpu" if a.tiny else "cuda")
    a.dtype = a.dtype or ("fp32" if a.tiny else "bf16")
    return a


def cmd_fetch(a) -> int:
    for name, src in RP.all_sources().items():
        p = RP.fetch(src, a.raw_dir, download=True)
        print(f"{name}: {p}（sha256 {RP.sha256_file(p)[:16]}… = 固定值）")
    return 0


def plan(a, family: str, prompts) -> dict:
    """选提示（模块说明）→ manifest 的内容。"""
    tiny = a.tiny
    raw = RP.resolve_raw(a.raw_dir, tiny=tiny)
    refs, ref_info = RP.reference_sets(raw, items_path=REPO_ROOT / HP.ITEMS_PATH, tiny=tiny)
    index = RP.OverlapIndex(refs)
    dolly_rows = RP.read_jsonl(raw["dolly"])
    if not tiny and len(dolly_rows) != RP.DOLLY.n_rows:
        raise RP.ReplayDataError(f"dolly：{len(dolly_rows)} 行，固定 revision 应有 {RP.DOLLY.n_rows} 行")
    general = RP.select_general(dolly_rows, index, n=a.n_general, allow_fewer=tiny)
    w2c_rows, shards = RP.read_w2c_rows(a.w2c_data)
    w2c_info = {"shards": shards, "n_rows": len(w2c_rows), "ids_sha256": RP.ids_digest([str(r["id"]) for r in w2c_rows])}
    image_of = None
    if not tiny:
        chk = RP.check_w2c_manifest(a.w2c_data, w2c_rows)
        image_of = chk.pop("image_of")
        w2c_info["manifest"] = chk
    paths = WD.W2CPaths(train_root=Path(a.train_data_root), eval=DataPaths(root=Path(a.eval_data_root)))
    pool, pool_steps = WD.load_train_pool(paths, real_model=not tiny)
    cb, cb_info = RP.countbench_hashes(a.eval_data_root)
    tpl = prompts.entry(RP.E1_COUNT_TASK, family)["template"]
    count = RP.select_count(w2c_rows, pool, cb, train_root=a.train_data_root, template=tpl, n=a.n_count, allow_fewer=tiny,
                            image_of=image_of)
    sources = {"dolly": RP.source_record(RP.DOLLY, raw["dolly"])}
    return RP.build_manifest(general, count, tiny=tiny, family=family, sources=sources, references=ref_info, w2c=w2c_info,
                             pool_steps=pool_steps, countbench=cb_info,
                             count_template={"config": "configs/e1_prompts.json", "sha256": prompts.sha256, "task": RP.E1_COUNT_TASK,
                                             "family": family, "template": tpl})


def build_model(a):
    entry = registry.get_model(a.model)
    policy = default_image_policy(entry.family)
    kw = {"init_std": TINY_INIT_STD} if a.tiny else {}
    ad = make_adapter(a.model, device=a.device, dtype=DTYPES[a.dtype], image_policy=policy, tiny=a.tiny, **kw)
    return entry, ad


def cmd_run(a, argv) -> int:
    out = Path(a.out)
    entry = registry.get_model(a.model)
    prompts = load_prompt_config(PROMPTS_PATH)
    if a.stage in ("all", "plan"):
        doc = plan(a, entry.family, prompts)
        sha = RP.ensure_manifest(out, doc)
        g, c = doc["general"], doc["count"]
        print(f"plan：通用 {g['selected']['n']} 条（剔除 { {k: v['n'] for k, v in g['removed'].items()} }），计数 {c['selected']['n']} 条"
              f"（{c['selected']['n_images']} 张图；与 CountBench 重合的图 {c['countbench']['n_images_removed']}）；manifest sha256 {sha[:16]}…")
        if a.stage == "plan":
            return 0
    man, man_sha = RP.load_manifest(out)
    if man["family"] != entry.family or bool(man["tiny"]) != bool(a.tiny):
        raise RP.ReplayDataError(f"{RP.manifest_path(out)} 是 {man['family']}（tiny {man['tiny']}）的，本次是 {entry.family}（tiny {a.tiny}）")
    bs = a.batch_size or (TINY_BATCH_SIZE if a.tiny else int(prompts.batch_protocol()["batch_size"]))
    specs = RP.gen_specs(prompts, batch_size=bs, tiny=a.tiny)
    entry, ad = build_model(a)
    ident = RP.run_identity(model=WD.model_record(entry, tiny=a.tiny), image_policy=policy_spec(ad.image_policy), dtype=a.dtype,
                            manifest_sha256=man_sha, specs=specs, prompts=prompts, tiny=a.tiny)
    RP.prepare_root(out, ident, [PROMPTS_PATH, REPO_ROOT / HP.ITEMS_PATH, RP.manifest_path(out)])
    kinds = list(RP.KINDS) if a.stage == "all" else [a.stage]
    run_k, run_path, run_log = _start_run_log(out, argv, stage=a.stage, device=str(ad.device), batch_size=bs)
    t_all = time.perf_counter()
    stats = [RP.run_kind(ad, k, man["items"][k], out, spec=specs[k], data_root=a.train_data_root, run_k=run_k) for k in kinds]
    run_log["kinds"] = stats
    _finish_run_log(run_path, run_log, n_batches=sum(int(s.get("n_batches", 0)) for s in stats), t_all=t_all)
    m = RP.finalize(out, man)
    print(f"{entry.key} 回放数据：" + "；".join(f"{k} {m[k]['n']}/{m[k]['n_expected']}（截断 {m[k]['n_truncated']}）" for k in RP.KINDS)
          + f"；complete = {m['complete']}")
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    a = parse_args(argv)
    try:
        return cmd_fetch(a) if a.stage == "fetch" else cmd_run(a, argv)
    except Exception as e:  # noqa: BLE001 — 证据先落盘，再以非零码退出
        if a.stage == "fetch" or not a.out:
            print(f"失败：{e!r}", file=sys.stderr)
            return 2
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
