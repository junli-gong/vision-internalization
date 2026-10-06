#!/usr/bin/env python
"""E5 主比较的训练入口（DEV-0037 §3；简报 §8.8）：一个起点 × 一个模型 × 一个数据量档（× 任务子集）× 一个种子。规则见 src/e5/train.py。

运行（B200；--out 是本次运行的目录 experiments/E5/<NNNN_slug>；结果写进 <out>/train/<起点>/<模型>/<n 标记>/seed<k>/）：
    python scripts/train_e5.py --init e3b --start-checkpoint <E3b 训练叶目录>/ckpt/final/model --model qwen3.5-2b \\
        --n-train 1000 --seed 0 --data-root $PROJ/data --out experiments/E5/0045_e5_main [--resume] [--stop-after-steps N]
    # --init base_warmup：DEV-0035 的预热检查点（scripts/train_e3b.py --supervision warmup 的 ckpt/final/model）
    # --init e3a：E3a 检查点（scripts/train_e3a.py --variant full 的 ckpt/final/model；先加上与 E3b 同一套网格 token 行）
    # --init base：基座，加网格 token 行，不预热（只作诊断；不给 --start-checkpoint）
    # --tasks a b c d：只用 a–d 训练（图像到文本的迁移，在 e 上测；每类条数与五类全训相同，叶目录 n<数据量>_abcd）
    # --data-root：起点检查点的合并核对要 E1 的 grid16 开发子集（同 eval_e3b 的评测前核对；默认环境变量 VISINT_DATA_ROOT）；base 不需要
开发机彩排（小型随机模型、CPU、fp32；起点用 scripts/train_e3b.py --tiny / scripts/train_e3a.py --tiny 的小检查点；数据量档取配置的
data.tiny.tiers；d 的数据只在规范平台上生成，非规范平台上只允许 --tiny）：
    python scripts/train_e5.py --tiny --init base --model qwen3.5-0.8b --n-train 10 --seed 0 --out <dir>

--n-train 必须是配置的数据量档之一（--tiny 时是 data.tiny.tiers 之一）；--seed 必须在配置的 training.seeds 里；--data-seed 默认等于 --seed。
小档（DEV-0038，D77-3）：training.min_steps 所列的档（1K、4K；--tiny 时 10 条的档）步数 = max(按 epoch 的步数, 下限)，并在 val 划分上
选检查点（selection 一节；src/e5/val_select.py），final/ 就是 val 上最好的那一步；16K、64K 照旧固定 epoch、取最后一步。
配种子（D77-3）：起点检查点必须来自 E3 的同一个种子（它训练叶目录 config.yaml 的 seed = --seed），不符或核对不了就拒绝训练（退出码 2）；
--allow-seed-mismatch 放开，记进运行身份（config.yaml 的 seed_pairing）。
评测：scripts/eval_e5_trained.py --checkpoint <叶目录>/ckpt/final。
退出码：0 = 跑完或按 --stop-after-steps 停下；1 = 起点检查点的合并核对没有通过，拒绝训练；2 = 出错（报错先写进
<out>/train/<起点>/<模型>/errors/；argparse 用法错误也是 2）。
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
from src.adapters.checkpoint import policy_spec, read_manifest  # noqa: E402
from src.adapters.factory import make_adapter  # noqa: E402
from src.adapters.image_policy import parse_policy  # noqa: E402
from src.e3.merge_check import MergeCheckFailed  # noqa: E402
from src.e5 import stack as STK  # noqa: E402
from src.e5.train import RunSpec, run_e5_training  # noqa: E402
from src.e5.train_config import CONFIG_PATH, STARTS, load_train_config, resolve_run  # noqa: E402
from src.e5.train_data import TrainSet  # noqa: E402

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
RUN_DIR_NAME = re.compile(r"[0-9]{4}_[A-Za-z0-9][A-Za-z0-9_.-]*")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--init", required=True, choices=STARTS, help="起点：base_warmup / e3b / e3a / base（诊断）")
    p.add_argument("--start-checkpoint", help="起点检查点目录（<训练叶目录>/ckpt/final/model）；base 不给")
    p.add_argument("--model", required=True, help="注册键（必须与起点检查点 manifest 的构造方式相同）")
    p.add_argument("--n-train", type=int, required=True, help="数据量档（五类合计；配置的 data.tiers 之一）")
    p.add_argument("--tasks", nargs="+", default=["all"], help="任务子集（任务名或字母 a–e；默认 all）")
    p.add_argument("--seed", type=int, required=True, help="种子（必须在配置的 training.seeds 里）")
    p.add_argument("--data-seed", type=int, help="数据种子（默认等于 --seed）")
    p.add_argument("--allow-seed-mismatch", action="store_true",
                   help="起点检查点的种子与 --seed 不同（或核对不了）时也训练（D77-3 的配种子放开；记进运行身份）")
    p.add_argument("--out", required=True, help="本次运行的目录（experiments/E5/<NNNN_slug>）")
    p.add_argument("--resume", action="store_true", help="续跑：从 <ckpt>/latest 接着训练")
    p.add_argument("--ckpt-root", help="检查点的根目录（默认 <叶目录>/ckpt；检查点不入库）")
    p.add_argument("--stop-after-steps", type=int, help="本次只跑 N 步，写检查点后退出（下次 --resume）")
    p.add_argument("--keep-optimizer-state", action="store_true", help="训练完成后保留 <ckpt>/latest（默认删掉）")
    p.add_argument("--tiny", action="store_true", help="小型随机模型 + CPU + fp32（开发机彩排）")
    p.add_argument("--data-root", help="E1 的落地根（起点检查点的合并核对用；默认环境变量 VISINT_DATA_ROOT；--tiny 时取 fixture）")
    p.add_argument("--config", default=str(CONFIG_PATH), help="E5 训练配置（默认 configs/e5_train.json）")
    p.add_argument("--device", help="默认：真模型 cuda，--tiny 时 cpu")
    p.add_argument("--dtype", choices=sorted(DTYPES), help="默认：真模型取 E3b 配置的 model.dtype（bf16），--tiny 时 fp32")
    a = p.parse_args(argv)
    if a.model not in registry.MODELS:
        p.error(f"未登记的模型键 {a.model!r}；已登记：{sorted(registry.MODELS)}")
    if a.stop_after_steps is not None and a.stop_after_steps < 1:
        p.error("--stop-after-steps 必须是正整数")
    if (a.init == "base") != (a.start_checkpoint is None):
        p.error("--init base 不给 --start-checkpoint；其余起点必须给")
    if a.start_checkpoint is not None and not Path(a.start_checkpoint).is_dir():
        p.error(f"--start-checkpoint {a.start_checkpoint!r} 不是目录")
    a.device = a.device or ("cpu" if a.tiny else "cuda")
    a.data_root = a.data_root or os.environ.get("VISINT_DATA_ROOT")
    if not a.tiny and a.init != "base" and not a.data_root:
        p.error("非 base 的起点要给 --data-root（或设 VISINT_DATA_ROOT）：起点检查点的合并核对用 E1 的 grid16 开发子集")
    if not a.tiny and not RUN_DIR_NAME.fullmatch(Path(a.out).name):
        p.error(f"--out 的目录名必须带运行编号，形如 0045_e5_main（CLAUDE.md：experiments/<phase>/<NNNN_slug>/），收到 {a.out!r}")
    a.data_seed = a.seed if a.data_seed is None else a.data_seed
    return a


def build_adapter(a, cfg):
    """起点有检查点时按它 manifest 的构造方式建 base（与起点同一个 base）；base 起点同 scripts/train_e3b.py。"""
    entry = registry.get_model(a.model)
    dtype = a.dtype or ("fp32" if a.tiny else cfg.e3b.doc["model"]["dtype"])
    if a.start_checkpoint is not None:
        spec = read_manifest(a.start_checkpoint).get("build_spec") or {}
        if spec.get("key") != a.model or bool(spec.get("tiny")) != a.tiny:
            raise ValueError(f"起点检查点的构造方式（key {spec.get('key')!r}，tiny {spec.get('tiny')}）与 --model {a.model!r}、"
                             f"--tiny {a.tiny} 不符")
        ad = make_adapter(spec["key"], device=a.device, dtype=DTYPES[dtype], image_policy=parse_policy(spec["image_policy"]),
                          tiny=bool(spec["tiny"]), **dict(spec["family_kw"]))
    elif a.tiny:
        t = cfg.e3b.doc["tiny"]
        ad = make_adapter(a.model, device=a.device, dtype=DTYPES[dtype], image_policy=parse_policy(t["image_policies"][entry.family]),
                          tiny=True, init_std=float(t["init_std"]))
    else:
        ad = make_adapter(a.model, device=a.device, dtype=DTYPES[dtype])
    return entry, ad, dtype


def cmd_run(a, argv) -> int:
    cfg = load_train_config(a.config)
    if int(a.seed) not in cfg.seeds:
        raise ValueError(f"--seed {a.seed} 不在配置的 training.seeds {list(cfg.seeds)} 里")
    tiers = cfg.tiers(a.tiny)
    if int(a.n_train) not in tiers:
        raise ValueError(f"--n-train {a.n_train} 不是配置的数据量档 {list(tiers)}" + ("（--tiny 用 data.tiny.tiers）" if a.tiny else ""))
    ts = TrainSet.build(cfg, a.n_train, a.tasks, a.data_seed)
    start = STK.resolve_start(cfg, a.init, a.start_checkpoint and Path(a.start_checkpoint).resolve(), model_key=a.model,
                              tiny=a.tiny)
    pairing = STK.check_seed_pairing(start, int(a.seed), allow_mismatch=a.allow_seed_mismatch)
    if pairing["status"] not in ("match", "not_applicable"):
        print(f"注意：配种子放开（{pairing['status']}：起点的种子 {pairing['start_seed']}，--seed {a.seed}），已记进运行身份",
              file=sys.stderr)
    rc = resolve_run(cfg, n_train=int(a.n_train), n_examples=len(ts), tiny=a.tiny)
    entry, ad, dtype = build_adapter(a, cfg)
    spec = RunSpec(model_key=entry.key, family=entry.family, repo=entry.repo, revision=entry.revision,
                   adapter_model_key=str(getattr(ad, "model_key", entry.key)), tiny=a.tiny, image_policy=policy_spec(ad.image_policy),
                   model_dtype=dtype, seed=int(a.seed), start=start, trainset=ts, seed_pairing=pairing)
    try:
        s = run_e5_training(ad, cfg, spec, rc, a.out, resume=a.resume, argv=argv, ckpt_root=a.ckpt_root,
                            stop_after_steps=a.stop_after_steps, keep_optimizer_state=a.keep_optimizer_state, data_root=a.data_root,
                            device=a.device, dtype_name=dtype)
    except MergeCheckFailed as e:
        print(f"拒绝训练：{e}", file=sys.stderr)
        return 1
    if s is None:
        print(f"{entry.key} {a.init} {ts.tag} seed{a.seed}：按 --stop-after-steps 停下，检查点已写；加 --resume 接着跑")
        return 0
    print(f"{entry.key} {a.init} {ts.tag} seed{a.seed} steps={s['steps']} selected_step={s.get('selected_step')} "
          f"final_loss={s['final_loss']:.4f} ckpt={s['final_checkpoint']} ({s['final_checkpoint_bytes']} B)")
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    a = parse_args(argv)
    try:
        return cmd_run(a, argv)
    except Exception as e:  # noqa: BLE001 — 证据先落盘，再以非零码退出
        err_dir = Path(a.out) / "train" / a.init / a.model / "errors"
        err_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = err_dir / f"error_{stamp}.json"
        path.write_text(json.dumps({"argv": argv, "error": repr(e), "traceback": traceback.format_exc()[-6000:]},
                                   indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"失败：{e!r}（已写出 {path}）", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
