#!/usr/bin/env python
"""E3b 像素 token 通道的训练入口（DEV-0025 第一部分；简报 §8.5）：一个模型 × 一种监督 × 一个种子。

运行（--out 是本次运行的目录 experiments/E3/<NNNN_slug>；结果写进 <out>/<变体>/<模型>/seed<k>/，变体 e3b = 网格监督，c3 = 稀疏文本）：
    python scripts/train_e3b.py --model qwen3.5-0.8b --supervision grid --seed 0 --data-root $DATA \\
        --out experiments/E3/0040_e3b_train [--resume] [--stop-after-steps N] [--ckpt-root DIR] [--keep-optimizer-state]
    python scripts/train_e3b.py --model qwen3.5-0.8b --supervision text --seed 0 --data-root $DATA --out experiments/E3/0040_e3b_train
    # --data-root：DEV-0022 的落地根（含 dedupe/train_exclusions.json；refcoco_train 与 nyu_depth_train 都在这里）
开发机彩排（小型随机模型、CPU、fp32、tests/fixtures 的小数据；配置的 tiny 节只改规模参数）：
    python scripts/train_e3b.py --tiny --model qwen3.5-0.8b --supervision grid --seed 0 --out <dir>
    --model 可以给多个（三家族一起彩排：qwen3.5-0.8b gemma-4-e2b-it internvl3.5-2b-hf）。

--seed 必须是配置 seeds 里的一个（简报 §8.4：每个配置 3 个种子）。--stop-after-steps N：本次只跑 N 步就写检查点退出（分段运行；
下次加 --resume 接着跑）。去重（硬约束 7）：训练加载器没有剔除清单就报错；清单还必须比较过配置 dedupe.required_eval_sets 的评测集，
旧格式的清单只在 --tiny 时照读。
退出码：0 = 跑完或按 --stop-after-steps 停下；2 = 出错（报错先写进 <out>/<变体>/<模型>/errors/，再退出；argparse 用法错误也是 2）。
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
from src.adapters.checkpoint import policy_spec  # noqa: E402
from src.adapters.factory import make_adapter  # noqa: E402
from src.adapters.image_policy import parse_policy  # noqa: E402
from src.e3.config import CONFIG_PATH, SUPERVISIONS, VARIANT_OF, load_config, resolve_run  # noqa: E402
from src.e3.data import E3Paths  # noqa: E402
from src.e3.train import RunSpec, run_training  # noqa: E402

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
RUN_DIR_NAME = re.compile(r"[0-9]{4}_[A-Za-z0-9][A-Za-z0-9_.-]*")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", nargs="+", required=True, help="注册键（src/adapters/registry.py），可多个（依次跑）")
    p.add_argument("--supervision", required=True, choices=SUPERVISIONS, help="grid = E3b（GT 网格 + 修正对）；text = C3（稀疏文本）")
    p.add_argument("--seed", type=int, required=True, help="种子（必须在配置的 seeds 里）")
    p.add_argument("--out", required=True, help="本次运行的目录（experiments/E3/<NNNN_slug>）")
    p.add_argument("--resume", action="store_true", help="续跑：从 <ckpt>/latest 接着训练")
    p.add_argument("--ckpt-root", help="检查点的根目录（默认 <叶目录>/ckpt；检查点不入库）")
    p.add_argument("--stop-after-steps", type=int, help="本次只跑 N 步，写检查点后退出（下次 --resume）")
    p.add_argument("--keep-optimizer-state", action="store_true", help="训练完成后保留 <ckpt>/latest（续跑状态；默认删掉）")
    p.add_argument("--tiny", action="store_true", help="小型随机模型 + CPU + fp32 + tests/fixtures 的小数据（开发机彩排）")
    p.add_argument("--data-root", help="DEV-0022 的落地根（默认环境变量 VISINT_DATA_ROOT；--tiny 时取 fixture）")
    p.add_argument("--config", default=str(CONFIG_PATH), help="E3b 训练配置（默认 configs/e3b_train.json）")
    p.add_argument("--device", help="默认：真模型 cuda，--tiny 时 cpu")
    p.add_argument("--dtype", choices=sorted(DTYPES), help="默认：真模型取配置的 model.dtype（bf16），--tiny 时 fp32")
    a = p.parse_args(argv)
    bad = [m for m in a.model if m not in registry.MODELS]
    if bad:
        p.error(f"未登记的模型键 {bad}；已登记：{sorted(registry.MODELS)}")
    if len(set(a.model)) != len(a.model):
        p.error("--model 有重复")
    if a.stop_after_steps is not None and a.stop_after_steps < 1:
        p.error("--stop-after-steps 必须是正整数")
    a.device = a.device or ("cpu" if a.tiny else "cuda")
    if not a.tiny and not (a.data_root or os.environ.get("VISINT_DATA_ROOT")):
        p.error("真模型运行要给 --data-root（或设 VISINT_DATA_ROOT）")
    if not a.tiny and not RUN_DIR_NAME.fullmatch(Path(a.out).name):
        p.error(f"--out 的目录名必须带运行编号，形如 0040_e3b_train（CLAUDE.md：experiments/<phase>/<NNNN_slug>/），收到 {a.out!r}")
    return a


def data_paths(a) -> E3Paths:
    if a.tiny and not a.data_root:
        return E3Paths.fixtures()
    return E3Paths.single_root(Path(a.data_root or os.environ["VISINT_DATA_ROOT"]))


def build_adapter(a, cfg, key: str):
    entry = registry.get_model(key)
    dtype = a.dtype or ("fp32" if a.tiny else cfg.doc["model"]["dtype"])
    if a.tiny:
        t = cfg.doc["tiny"]
        ad = make_adapter(key, device=a.device, dtype=DTYPES[dtype], image_policy=parse_policy(t["image_policies"][entry.family]),
                          tiny=True, init_std=float(t["init_std"]))
    else:
        ad = make_adapter(key, device=a.device, dtype=DTYPES[dtype])
    return entry, ad, dtype


def make_spec(a, entry, ad, dtype: str) -> RunSpec:
    return RunSpec(model_key=entry.key, family=entry.family, repo=entry.repo, revision=entry.revision,
                   adapter_model_key=str(getattr(ad, "model_key", entry.key)), tiny=a.tiny,
                   image_policy=policy_spec(ad.image_policy), model_dtype=dtype, supervision=a.supervision, seed=int(a.seed))


def cmd_run(a, argv) -> int:
    cfg = load_config(a.config)
    if int(a.seed) not in cfg.seeds:
        raise ValueError(f"--seed {a.seed} 不在配置的 seeds {list(cfg.seeds)} 里")
    rc = resolve_run(cfg, supervision=a.supervision, tiny=a.tiny)
    paths = data_paths(a)
    for key in a.model:
        entry, ad, dtype = build_adapter(a, cfg, key)
        spec = make_spec(a, entry, ad, dtype)
        s = run_training(ad, cfg, spec, rc, paths, a.out, resume=a.resume, argv=argv, ckpt_root=a.ckpt_root,
                         stop_after_steps=a.stop_after_steps, keep_optimizer_state=a.keep_optimizer_state)
        if s is None:
            print(f"{entry.key} {spec.variant} seed{spec.seed}：按 --stop-after-steps 停下，检查点已写；加 --resume 接着跑"
                  "（后面的模型没有开始）")
            return 0
        print(f"{entry.key} {spec.variant} seed{spec.seed} steps={s['steps']} final_loss={s['final_loss']:.4f} "
              f"ckpt={s['final_checkpoint']} ({s['final_checkpoint_bytes']} B)")
        del ad
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    a = parse_args(argv)
    try:
        return cmd_run(a, argv)
    except Exception as e:  # noqa: BLE001 — 证据先落盘，再以非零码退出
        err_dir = Path(a.out) / VARIANT_OF[a.supervision] / a.model[0] / "errors"
        err_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = err_dir / f"error_{stamp}.json"
        path.write_text(json.dumps({"argv": argv, "error": repr(e), "traceback": traceback.format_exc()[-6000:]},
                                   indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"失败：{e!r}（已写出 {path}）", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
