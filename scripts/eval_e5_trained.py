#!/usr/bin/env python
"""E5 训练后系统的评测入口（DEV-0037 §4；简报 §8.8）：一个 E5 检查点（起点 + E5 LoRA）→ E5 的 test 划分上单轮受限解码，用零样本的
同一套评分与判据（D59-4、D72-1）。规则见 src/e5/trained_eval.py。

服务器（B200；可比的评测一律在 B200 上做）：
    python scripts/eval_e5_trained.py --checkpoint <E5 训练叶目录>/ckpt/final --model qwen3.5-2b \\
        --out experiments/E5/0046_e5_main_eval [--task all] [--subset eval|dev] [--resume] [--limit N]
    # 结果写进 <out>/trained/<起点>/<模型>/<n 标记>/seed<k>/<任务>[__dev]/；评测前的合并核对写进 <…>/seed<k>/merge_check.json
    # 起点检查点按 <ckpt>/final/e5_stack.json 记下的路径读；搬过家时用 --start-checkpoint 指新路径（manifest 的 sha256 必须相同）
开发机彩排（小型随机模型、CPU fp32；d 的数据只在规范平台上生成，非规范平台上只允许 --tiny）：
    python scripts/eval_e5_trained.py --tiny --checkpoint <dir>/train/base/qwen3.5-0.8b/n10/seed0/ckpt/final --model qwen3.5-0.8b \\
        --out <dir> --limit 2

顺序：读 e5_stack.json 与 E5 检查点 manifest（构造方式必须与 --model、--tiny 相符）→ 建 base、合并起点 LoRA、核对起点（manifest 的 sha256、
权重指纹）→ load_trained(E5 检查点, merge=False) → 评测前的合并核对（不通过就不评测）→ 逐任务评测。
每档条数默认取配置的 eval.n_per_bin（test：每档 100 条，b 200 条，与零样本相同）；--n-per-bin 统一覆盖。批大小默认取配置的 eval.batch_size（64）；
真模型给别的值要加 --allow-unfrozen-batch。
退出码：0 = 跑完；1 = 合并核对没有通过，拒绝评测；2 = 出错（报错先写进 <out>/trained/errors/；argparse 用法错误也是 2）。
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

import torch  # noqa: E402

from src.adapters import registry  # noqa: E402
from src.adapters.checkpoint import read_manifest  # noqa: E402
from src.constants import BOOTSTRAP_N_RESAMPLES  # noqa: E402
from src.e3.merge_check import MergeCheckFailed  # noqa: E402
from src.e3.tokens import attach_vocab  # noqa: E402
from src.e5 import stack as STK  # noqa: E402
from src.e5 import trained_eval as TE  # noqa: E402
from src.e5.protocol import E5_TASKS, SUBSETS  # noqa: E402
from src.e5.token_grid import E5Tokens  # noqa: E402
from src.e5.train_config import CONFIG_PATH, load_train_config  # noqa: E402
from src.runlog import sha256_file  # noqa: E402
from src.synth.seeding import canonical_task  # noqa: E402

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
RUN_DIR_NAME = re.compile(r"[0-9]{4}_[A-Za-z0-9][A-Za-z0-9_.-]*")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="E5 检查点（<训练叶目录>/ckpt/final：含 model/ 与 e5_stack.json）")
    p.add_argument("--model", required=True, help="注册键（必须与检查点 manifest 的构造方式相同）")
    p.add_argument("--out", required=True, help="本次运行的目录（experiments/E5/<NNNN_slug>）")
    p.add_argument("--task", nargs="+", default=["all"], help=f"任务名、字母（a–e）或 all：{list(E5_TASKS)}")
    p.add_argument("--subset", choices=sorted(SUBSETS), help="eval = test 划分（默认取配置的 eval.subset）；dev = val 划分")
    p.add_argument("--n-per-bin", type=int, help="每档条数（统一覆盖配置的 eval.n_per_bin）")
    p.add_argument("--limit", type=int, help="只取每个任务评测清单的前 N 条（彩排、调试）")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--start-checkpoint", help="起点检查点的新路径（默认取 e5_stack.json 记下的；manifest 的 sha256 必须相同）")
    p.add_argument("--batch-size", type=int, help="批大小（默认配置的 eval.batch_size；--tiny 取 eval.tiny.batch_size）")
    p.add_argument("--allow-unfrozen-batch", action="store_true", help="真模型允许配置以外的批大小（运行身份里照记）")
    p.add_argument("--merge-check-n-per-task", type=int, help="合并核对每类的 val 样本数（默认配置的 eval.merge_check.n_per_task）")
    p.add_argument("--n-resamples", type=int, help="bootstrap 次数（默认 src.constants.BOOTSTRAP_N_RESAMPLES；--tiny 取配置的 eval.tiny）")
    p.add_argument("--tiny", action="store_true", help="小型随机模型 + CPU fp32（开发机彩排）")
    p.add_argument("--config", default=str(CONFIG_PATH), help="E5 训练配置（默认 configs/e5_train.json，读其中的 eval 节）")
    p.add_argument("--device", help="默认：真模型 cuda，--tiny 时 cpu")
    p.add_argument("--dtype", choices=sorted(DTYPES), help="默认：真模型 bf16，--tiny 时 fp32")
    a = p.parse_args(argv)
    if a.model not in registry.MODELS:
        p.error(f"未登记的模型键 {a.model!r}；已登记：{sorted(registry.MODELS)}")
    for name in ("n_per_bin", "limit", "batch_size", "merge_check_n_per_task"):
        v = getattr(a, name)
        if v is not None and v < 1:
            p.error(f"--{name.replace('_', '-')} 必须是正整数")
    if a.n_resamples is not None and a.n_resamples < 0:
        p.error("--n-resamples 不能为负")
    a.task = list(E5_TASKS) if a.task == ["all"] else [canonical_task(t) for t in a.task]
    if len(set(a.task)) != len(a.task):
        p.error("--task 有重复")
    a.device = a.device or ("cpu" if a.tiny else "cuda")
    a.dtype = a.dtype or ("fp32" if a.tiny else "bf16")
    if not Path(a.checkpoint).is_dir():
        p.error(f"--checkpoint {a.checkpoint!r} 不是目录")
    if not a.tiny and not RUN_DIR_NAME.fullmatch(Path(a.out).name):
        p.error(f"--out 的目录名必须带运行编号，形如 0046_e5_main_eval（CLAUDE.md：experiments/<phase>/<NNNN_slug>/），收到 {a.out!r}")
    return a


def cmd_run(a, argv) -> int:
    cfg = load_train_config(a.config)
    e = cfg.doc["eval"]
    t = e["tiny"] if a.tiny else {}
    bs = int(a.batch_size or t.get("batch_size", e["batch_size"]))
    if not a.tiny and bs != int(e["batch_size"]) and not a.allow_unfrozen_batch:
        raise ValueError(f"--batch-size {bs} 不是配置的 eval.batch_size {e['batch_size']}；确要换就加 --allow-unfrozen-batch")
    subset = a.subset or e["subset"]
    n_resamples = a.n_resamples if a.n_resamples is not None else int(t.get("n_resamples", BOOTSTRAP_N_RESAMPLES))
    mc_n = int(a.merge_check_n_per_task or t.get("merge_check_n_per_task", e["merge_check"]["n_per_task"]))
    mc_bs = int(t.get("merge_check_batch_size", e["merge_check"]["batch_size"]))
    fd = Path(a.checkpoint)
    stack = STK.read_stack_file(fd)
    man = read_manifest(fd / stack["model_dir"])
    spec_b = man.get("build_spec") or {}
    if spec_b.get("key") != a.model or bool(spec_b.get("tiny")) != a.tiny:
        raise ValueError(f"检查点的构造方式（key {spec_b.get('key')!r}，tiny {spec_b.get('tiny')}）与 --model {a.model!r}、"
                         f"--tiny {a.tiny} 不符")
    if stack["config"]["e3b_config"]["sha256"] != cfg.e3b.sha256:
        raise ValueError(f"E5 检查点训练时的 E3b 配置 sha256 {stack['config']['e3b_config']['sha256'][:16]}… 与本次的 "
                         f"{cfg.e3b.sha256[:16]}… 不同：词表与允许集合要与训练时相同")
    start_kind = stack["start"]["kind"]
    tag = str(stack["tag"])
    seed = int(stack["seed"])
    data_seed = int(stack["data"]["data_seed"])
    mdir = TE.model_dir(a.out, start_kind, a.model, tag, seed, data_seed)
    mdir.mkdir(parents=True, exist_ok=True)
    st = STK.load_stacked(fd, device=a.device, dtype=DTYPES[a.dtype], start_override=a.start_checkpoint,
                          log=lambda m: print(m, file=sys.stderr, flush=True))
    ad = st.adapter
    tokens = attach_vocab(ad, cfg.e3b.vocab)
    e5tok = E5Tokens(tokens, cfg.e3b.allowed_scope)
    if e5tok.layout() != stack["tokens_e5"]:
        raise ValueError("E5 检查点的网格 token 布局（允许集合、起伏映射）与训练时记下的不同")
    prompt = cfg.prompt_format()
    n = cfg.grid_n
    e5_sha = sha256_file(st.model_dir / "manifest.json")
    key = {"e5_manifest_sha256": e5_sha, "start_manifest_sha256": stack["start"]["manifest_sha256"], "n_per_task": mc_n,
           "tasks": list(E5_TASKS), "tiny": a.tiny, "dtype": a.dtype, "batch_size": mc_bs, "config_sha256": cfg.sha256}
    exact = torch.device(a.device).type == "cpu" and DTYPES[a.dtype] == torch.float32
    try:
        mc = TE.stacked_merge_check(ad, e5tok, prompt, n=n, n_per_task=mc_n, tasks=E5_TASKS, batch_size=mc_bs, tiny=a.tiny,
                                    exact=exact, key=key, out_json=mdir / TE.MERGE_CHECK)
    except MergeCheckFailed as err:
        print(f"拒绝评测：{err}", file=sys.stderr)
        return 1
    checkpoint = {"manifest_sha256": e5_sha, "train_identity_sha256": stack["identity_sha256"],
                  "start": {"kind": start_kind, "manifest_sha256": stack["start"]["manifest_sha256"]},
                  "lora": man.get("lora"), "build_spec": man.get("build_spec"), "data": stack["data"], "seed": seed}
    extra = {"checkpoint_dir": str(fd), "start_checkpoint": None if st.start_checkpoint is None else str(st.start_checkpoint),
             "merge_check": mc, "fingerprint": st.fingerprint,
             "gpu": torch.cuda.get_device_name(0) if a.device.startswith("cuda") and torch.cuda.is_available() else None}
    for task in a.task:
        npb = int(a.n_per_bin or e["n_per_bin"][task])
        spec = TE.EvalSpec(start=start_kind, model_key=a.model, tag=tag, seed=seed, task=task, subset=subset, n_per_bin=npb,
                           n_resamples=n_resamples, batch_size=bs, tiny=a.tiny, dtype=a.dtype, limit=a.limit,
                           data_seed=data_seed)
        m = TE.run_task(ad, e5tok, prompt, cfg, spec, a.out, checkpoint=checkpoint, n=n, resume=a.resume, argv=argv,
                        run_extra=extra)
        h = m["headline"]
        parts = [f"iou_grid={h['mask_iou_grid']['strict']}"]
        if "relief_error" in h:
            parts.append(f"relief={h['relief_error']['strict']}")
        print(f"trained {start_kind} {a.model} {tag} seed{seed} {task}/{subset} n={m['n']} {' '.join(parts)} "
              f"parse_fail={m['parse']['failure_rate']} imagining={(m.get('gate') or {}).get('imagining')}")
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    a = parse_args(argv)
    try:
        return cmd_run(a, argv)
    except Exception as e:  # noqa: BLE001 — 证据先落盘，再以非零码退出
        err_dir = Path(a.out) / TE.EVAL_SUB / "errors"
        err_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = err_dir / f"error_{stamp}.json"
        path.write_text(json.dumps({"argv": argv, "error": repr(e), "traceback": traceback.format_exc()[-6000:]},
                                   indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"失败：{e!r}（已写出 {path}）", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
