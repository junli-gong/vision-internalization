#!/usr/bin/env python
"""E3a 稠密读出的训练入口（DEV-0034 §2；简报 §8.4）：一个模型 × 一个变体 × 一个种子。规则见 src/readout/train.py。

运行（--out 是本次运行的目录 experiments/E3/<NNNN_slug>；结果写进 <out>/e3a/<变体>/<模型>/seed<k>/）：
    python scripts/train_e3a.py --model qwen3.5-0.8b --variant full --seed 0 --data-root $DATA \\
        --out experiments/E3/0050_e3a_train [--resume] [--stop-after-steps N] [--ckpt-root DIR] [--keep-optimizer-state]
    # --variant full（完整方法）| c1（冻结 LLM，只训练查询、混合、解码器与各头）| c2（解码器读视觉编码器的输出，绕过 LLM）
    # --tasks refseg depth：只用指代分割与深度（E3b 的同一组数据，简报 §8.5 的同预算对照）；不给时取配置的 mix.weights。
    #   生效的混合进运行身份（training.mix），叶目录不带任务子集：与全任务的运行放在不同的 --out（续跑时身份不同会拒绝混写）
    # --data-root：DEV-0022 的落地根（含 dedupe/train_exclusions.json；refcoco_train、nyu_depth_train、ade20k、nyu_normals 都在这里）
E3a+C3（DEV-0044：每步 = C3 同一种子的文本流 8 条 + E3a full 同一种子的稠密流 8 条，8000 步，读出损失与语言损失 1 : 1；规则见配置的
mixed 节）：
    python scripts/train_e3a.py --model qwen3.5-2b --variant full --with-text --seed 0 --data-root $DATA \\
        --out experiments/E3/0060_e3a_c3
    # 结果写进 <out>/e3a/e3a_c3/<模型>/seed<k>/；只配 --variant full，不能与 --tasks 同用。检查点同 E3a full：评测用
    #   scripts/eval_e3a.py --variant full --checkpoint <叶目录>/ckpt/final（readout.json 的结构变体是 full），T 文本用
    #   scripts/run_e1.py --checkpoint <叶目录>/ckpt/final/model
开发机彩排（小型随机模型、CPU、fp32、tests/fixtures 的小数据；配置的 tiny 节只改规模参数）：
    python scripts/train_e3a.py --tiny --model qwen3.5-0.8b gemma-4-e2b-it internvl3.5-2b-hf --variant full --seed 0 --out <dir>

--seed 必须是配置 seeds 里的一个（简报 §8.4：每个配置 3 个种子，由命令行给出）。--stop-after-steps N：本次只跑 N 步就写检查点退出
（吞吐估算用 20；下次加 --resume 接着跑）。去重（硬约束 7）：训练加载器没有剔除清单就报错；清单还必须比较过配置
dedupe.required_eval_sets 的评测集，旧格式的清单只在 --tiny 时照读。
检查点：<叶目录>/ckpt/final/{model,readout}；model/ 可以直接给 scripts/run_e1.py --checkpoint（T 文本评测），整个 final/ 给
scripts/eval_e3a.py --checkpoint。
--seed 在 E3a+C3 下还必须在文本流配置（configs/e3b_train.json）的 seeds 里（load_config 核对两份的种子相容）。
退出码：0 = 跑完或按 --stop-after-steps 停下；2 = 出错（报错先写进 <out>/e3a/<变体>/<模型>/errors/，再退出；argparse 用法错误也是 2）。
"""

from __future__ import annotations

import argparse
import copy
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
from src.readout.config import CONFIG_PATH, E3A_C3, MIXED_BASE, E3aConfig, load_config, resolve_run  # noqa: E402
from src.readout.data import E3aPaths  # noqa: E402
from src.readout.model import TASKS, VARIANTS  # noqa: E402
from src.readout.train import EXPERIMENT_SUB, RunSpec, run_training  # noqa: E402

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
RUN_DIR_NAME = re.compile(r"[0-9]{4}_[A-Za-z0-9][A-Za-z0-9_.-]*")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", nargs="+", required=True, help="注册键（src/adapters/registry.py），可多个（依次跑）")
    p.add_argument("--variant", required=True, choices=VARIANTS, help="full = 完整方法；c1 = 冻结 LLM；c2 = 读视觉编码器的输出")
    p.add_argument("--seed", type=int, required=True, help="种子（必须在配置的 seeds 里）")
    p.add_argument("--out", required=True, help="本次运行的目录（experiments/E3/<NNNN_slug>）")
    p.add_argument("--tasks", nargs="+", choices=TASKS, help="只用这些任务（各权重 1）；不给时取配置的 mix.weights")
    p.add_argument("--with-text", action="store_true",
                   help=f"E3a+C3（DEV-0044）：另加 C3 的文本样本流与语言损失（配置的 mixed 节）；只配 --variant {MIXED_BASE}，输出变体 {E3A_C3}")
    p.add_argument("--resume", action="store_true", help="续跑：从 <ckpt>/latest 接着训练")
    p.add_argument("--ckpt-root", help="检查点的根目录（默认 <叶目录>/ckpt；检查点不入库）")
    p.add_argument("--stop-after-steps", type=int, help="本次只跑 N 步，写检查点后退出（下次 --resume）")
    p.add_argument("--keep-optimizer-state", action="store_true", help="训练完成后保留 <ckpt>/latest（续跑状态；默认删掉）")
    p.add_argument("--tiny", action="store_true", help="小型随机模型 + CPU + fp32 + tests/fixtures 的小数据（开发机彩排）")
    p.add_argument("--data-root", help="DEV-0022 的落地根（默认环境变量 VISINT_DATA_ROOT；--tiny 时取 fixture）")
    p.add_argument("--config", default=str(CONFIG_PATH), help="E3a 训练配置（默认 configs/e3a_train.json）")
    p.add_argument("--device", help="默认：真模型 cuda，--tiny 时 cpu")
    p.add_argument("--dtype", choices=sorted(DTYPES), help="默认：真模型取配置的 model.dtype（bf16），--tiny 时 fp32")
    a = p.parse_args(argv)
    bad = [m for m in a.model if m not in registry.MODELS]
    if bad:
        p.error(f"未登记的模型键 {bad}；已登记：{sorted(registry.MODELS)}")
    if len(set(a.model)) != len(a.model):
        p.error("--model 有重复")
    if a.tasks is not None and len(set(a.tasks)) != len(a.tasks):
        p.error("--tasks 有重复")
    if a.with_text and a.variant != MIXED_BASE:
        p.error(f"--with-text 只配 --variant {MIXED_BASE}（E3a+C3 的读出、LoRA、查询同 {MIXED_BASE}），收到 {a.variant}")
    if a.with_text and a.tasks is not None:
        p.error("--with-text 不能与 --tasks 同用（E3a+C3 的稠密流照配置的 mix.weights，四个任务）")
    if a.stop_after_steps is not None and a.stop_after_steps < 1:
        p.error("--stop-after-steps 必须是正整数")
    a.device = a.device or ("cpu" if a.tiny else "cuda")
    if not a.tiny and not (a.data_root or os.environ.get("VISINT_DATA_ROOT")):
        p.error("真模型运行要给 --data-root（或设 VISINT_DATA_ROOT）")
    if not a.tiny and not RUN_DIR_NAME.fullmatch(Path(a.out).name):
        p.error(f"--out 的目录名必须带运行编号，形如 0050_e3a_train（CLAUDE.md：experiments/<phase>/<NNNN_slug>/），收到 {a.out!r}")
    return a


def with_tasks(cfg: E3aConfig, tasks) -> E3aConfig:
    """--tasks：mix.weights 换成这些任务各 1（其余 0）；配置的 sha256 不变，生效的混合写进运行身份（training.mix）。"""
    if not tasks:
        return cfg
    doc = copy.deepcopy(dict(cfg.doc))
    doc["mix"] = {**doc["mix"], "weights": {t: (1 if t in tasks else 0) for t in TASKS}}
    return E3aConfig(path=cfg.path, sha256=cfg.sha256, doc=doc)


def run_variant(a) -> str:
    """输出的变体名：--with-text 时 e3a_c3，否则就是 --variant。"""
    return E3A_C3 if getattr(a, "with_text", False) else a.variant


def data_paths(a) -> E3aPaths:
    if a.tiny and not a.data_root:
        return E3aPaths.fixtures()
    return E3aPaths.single_root(Path(a.data_root or os.environ["VISINT_DATA_ROOT"]))


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
                   image_policy=policy_spec(ad.image_policy), model_dtype=dtype, variant=run_variant(a), seed=int(a.seed))


def cmd_run(a, argv) -> int:
    cfg = with_tasks(load_config(a.config), a.tasks)
    if int(a.seed) not in cfg.seeds:
        raise ValueError(f"--seed {a.seed} 不在配置的 seeds {list(cfg.seeds)} 里")
    rc = resolve_run(cfg, variant=run_variant(a), tiny=a.tiny)
    paths = data_paths(a)
    for key in a.model:
        entry, ad, dtype = build_adapter(a, cfg, key)
        spec = make_spec(a, entry, ad, dtype)
        s = run_training(ad, cfg, spec, rc, paths, a.out, resume=a.resume, argv=argv, ckpt_root=a.ckpt_root,
                         stop_after_steps=a.stop_after_steps, keep_optimizer_state=a.keep_optimizer_state)
        if s is None:
            print(f"{entry.key} {run_variant(a)} seed{a.seed}：按 --stop-after-steps 停下，检查点已写；加 --resume 接着跑"
                  "（后面的模型没有开始）")
            return 0
        print(f"{entry.key} {run_variant(a)} seed{a.seed} steps={s['steps']} final_loss={s['final_loss']:.4f} "
              f"ckpt={s['final_checkpoint']} ({s['final_checkpoint_bytes']} B) "
              f"readout={s['readout_params']['decoder_budget_total']} params")
        del ad
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    a = parse_args(argv)
    try:
        return cmd_run(a, argv)
    except Exception as e:  # noqa: BLE001 — 证据先落盘，再以非零码退出
        err_dir = Path(a.out) / EXPERIMENT_SUB / run_variant(a) / a.model[0] / "errors"
        err_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = err_dir / f"error_{stamp}.json"
        path.write_text(json.dumps({"argv": argv, "error": repr(e), "traceback": traceback.format_exc()[-6000:]},
                                   indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"失败：{e!r}（已写出 {path}）", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
