#!/usr/bin/env python
"""E3a 稠密读出的评测入口（DEV-0034 §3；简报 §8.4；在 B200 上运行，D55、D59-10）：一个检查点 → 指代分割（RefCOCO val）、深度
（NYU 654）、语义分割（ADE20K val）、法向（NYU 法向 test 654）的稠密指标（guided 与 bilinear 两种上采样，bootstrap CI，逐样本统计）
与两项泄漏检查（批内打乱、去掉指令）。规则见 src/readout/evaluate.py。

服务器：
    python scripts/eval_e3a.py --checkpoint <训练叶目录>/ckpt/final --model qwen3.5-0.8b --variant full --seed 0 \\
        --data-root $PROJ/data [--train-data-root DIR] --out experiments/E3/00NN_e3a_eval [--task ...] [--resume] [--limit N]
    # --data-root：E1 的落地根（refcoco、nyuv2）；--train-data-root：DEV-0022 的落地根（ade20k/val、nyu_normals/test），默认同 --data-root
    # 结果写进 <out>/eval/e3a/<变体>/<模型>/seed<k>/<任务>/；full 的合并核对写进 <out>/eval/e3a/<变体>/<模型>/seed<k>/merge_check.json
开发机彩排（小型随机模型、CPU fp32；检查点用 scripts/train_e3a.py --tiny 训出来的；评测数据取 tests/fixtures）：
    python scripts/train_e3a.py --tiny --model qwen3.5-0.8b --variant full --seed 0 --out /tmp/t
    python scripts/eval_e3a.py --tiny --checkpoint /tmp/t/e3a/full/qwen3.5-0.8b/seed0/ckpt/final --model qwen3.5-0.8b \\
        --variant full --seed 0 --out /tmp/e

顺序：读两份 manifest（model/ 的构造方式必须与 --model、--tiny 相符；readout/ 的变体必须是 --variant）→ full：评测前的合并核对
（src/e3/merge_check 的 merge 一项作用于 <检查点>/model，n 取配置的 eval.merge_check.n，可用 --merge-check-n 改；不通过就不评测）；
c1、c2 没有 LoRA，跳过 → 加载（load_trained(merge=True) + 读出）→ 逐任务评测。批大小默认取配置的 eval.batch_size；真模型给别的值
要加 --allow-unfrozen-batch。
退出码：0 = 跑完；1 = 合并核对没有通过，拒绝评测；2 = 出错（报错先写进 <out>/eval/e3a/<变体>/<模型>/seed<k>/errors/；argparse 用法
错误也是 2）。
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
from src.adapters.checkpoint import read_manifest  # noqa: E402
from src.e3.merge_check import MergeCheckFailed, run_merge_check  # noqa: E402
from src.readout import evaluate as EV  # noqa: E402
from src.readout.config import CONFIG_PATH, load_config  # noqa: E402
from src.readout.model import TASKS, VARIANTS  # noqa: E402
from src.readout.train import MODEL_SUB, READOUT_SUB, read_readout_manifest  # noqa: E402
from src.runlog import sha256_file  # noqa: E402

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
RUN_DIR_NAME = re.compile(r"[0-9]{4}_[A-Za-z0-9][A-Za-z0-9_.-]*")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="检查点目录（含 model/ 与 readout/：<训练叶目录>/ckpt/final）")
    p.add_argument("--model", required=True, help="注册键（必须与检查点 manifest 的构造方式相同）")
    p.add_argument("--variant", required=True, choices=VARIANTS, help="必须与检查点 readout.json 的变体相同")
    p.add_argument("--seed", type=int, required=True, help="训练的种子（输出目录 seed<k>）")
    p.add_argument("--task", nargs="+", default=list(TASKS), choices=TASKS, help="默认四个任务")
    p.add_argument("--out", required=True, help="本次运行的目录（experiments/E3/<NNNN_slug>）")
    p.add_argument("--resume", action="store_true", help="续跑：已完成的整批跳过")
    p.add_argument("--limit", type=int, help="只取每个评测集的前 N 条（调试；至少 2）")
    p.add_argument("--tiny", action="store_true", help="小型随机模型 + CPU fp32 + tests/fixtures 的小数据（开发机彩排）")
    p.add_argument("--data-root", help="E1 的落地根（refcoco、nyuv2；默认环境变量 VISINT_DATA_ROOT；--tiny 时取 fixture）")
    p.add_argument("--train-data-root", help="DEV-0022 的落地根（ade20k/val、nyu_normals/test；默认同 --data-root）")
    p.add_argument("--dev-subsets", help="E1 的开发子集清单（默认 configs/e1_dev_subsets.json；--tiny 时取 fixture）")
    p.add_argument("--config", default=str(CONFIG_PATH), help="E3a 配置（默认 configs/e3a_train.json，读其中的 eval 节）")
    p.add_argument("--batch-size", type=int, help="批大小（默认取配置的 eval.batch_size；--tiny 取 eval.tiny.batch_size）")
    p.add_argument("--allow-unfrozen-batch", action="store_true", help="真模型允许配置以外的批大小（运行身份里照记）")
    p.add_argument("--merge-check-n", type=int, help="合并核对的开发样本数（默认取配置的 eval.merge_check.n）")
    p.add_argument("--score-workers", type=int, help="评分的进程数（默认取配置的 eval.score_workers；0 = 串行）")
    p.add_argument("--device", help="默认：真模型 cuda，--tiny 时 cpu")
    p.add_argument("--dtype", choices=sorted(DTYPES), help="默认：真模型 bf16，--tiny 时 fp32")
    a = p.parse_args(argv)
    if a.model not in registry.MODELS:
        p.error(f"未登记的模型键 {a.model!r}；已登记：{sorted(registry.MODELS)}")
    for name in ("batch_size", "merge_check_n"):
        v = getattr(a, name)
        if v is not None and v < 1:
            p.error(f"--{name.replace('_', '-')} 必须是正整数")
    if a.limit is not None and a.limit < 2:
        p.error("--limit 至少 2（打乱检查要求每批至少 2 条）")
    if a.score_workers is not None and a.score_workers < 0:
        p.error("--score-workers 不能为负")
    if len(set(a.task)) != len(a.task):
        p.error("--task 有重复")
    a.device = a.device or ("cpu" if a.tiny else "cuda")
    a.dtype = a.dtype or ("fp32" if a.tiny else "bf16")
    ck = Path(a.checkpoint)
    if not (ck / MODEL_SUB).is_dir() or not (ck / READOUT_SUB).is_dir():
        p.error(f"--checkpoint {a.checkpoint!r} 里要有 {MODEL_SUB}/ 与 {READOUT_SUB}/")
    if not a.tiny and not (a.data_root or os.environ.get("VISINT_DATA_ROOT")):
        p.error("真模型运行要给 --data-root（或设 VISINT_DATA_ROOT）")
    if not a.tiny and not RUN_DIR_NAME.fullmatch(Path(a.out).name):
        p.error(f"--out 的目录名必须带运行编号，形如 0051_e3a_eval（CLAUDE.md：experiments/<phase>/<NNNN_slug>/），收到 {a.out!r}")
    return a


def eval_paths(a) -> EV.EvalPaths:
    if a.tiny and not a.data_root:
        base = EV.EvalPaths.fixtures()
        if not a.dev_subsets:
            return base
        return EV.EvalPaths(base.refseg_root, base.depth_root, base.semseg_root, base.normals_root, Path(a.dev_subsets))
    root = Path(a.data_root or os.environ["VISINT_DATA_ROOT"])
    return EV.EvalPaths.roots(root, a.train_data_root, a.dev_subsets)


def cmd_run(a, argv) -> int:
    cfg = load_config(a.config)
    settings = EV.resolve_eval(cfg, tiny=a.tiny, batch_size=a.batch_size, score_workers=a.score_workers,
                               merge_check_n=a.merge_check_n)
    if not a.tiny and settings["batch_size"] != settings["batch_size_config"] and not a.allow_unfrozen_batch:
        raise ValueError(f"--batch-size {settings['batch_size']} 不是配置的 eval.batch_size {settings['batch_size_config']}；"
                         "确要换就加 --allow-unfrozen-batch")
    ck = Path(a.checkpoint)
    man = read_manifest(ck / MODEL_SUB)
    rman = read_readout_manifest(ck / READOUT_SUB)
    spec_b = man.get("build_spec") or {}
    if spec_b.get("key") != a.model or bool(spec_b.get("tiny")) != a.tiny:
        raise ValueError(f"检查点的构造方式（key {spec_b.get('key')!r}，tiny {spec_b.get('tiny')}）与 --model {a.model!r}、"
                         f"--tiny {a.tiny} 不符")
    if rman["spec"]["variant"] != a.variant:
        raise ValueError(f"检查点的变体是 {rman['spec']['variant']}，--variant 给的是 {a.variant}")
    if (man.get("lora") is not None) != (a.variant == "full"):
        raise ValueError(f"{a.variant} 的检查点{'应' if a.variant == 'full' else '不应'}有 LoRA")
    entry = registry.get_model(a.model)
    ck_block = {"model_manifest_sha256": sha256_file(ck / MODEL_SUB / "manifest.json"),
                "readout_manifest_sha256": sha256_file(ck / READOUT_SUB / "readout.json"),
                "build_spec": man.get("build_spec"), "image_policy": man.get("image_policy"), "dtype": man.get("dtype"),
                "lora": man.get("lora"), "readout_params": rman.get("params")}
    probe = EV.EvalSpec(model_key=entry.key, family=entry.family, repo=entry.repo, revision=entry.revision,
                        adapter_model_key=str(man.get("model_key", entry.key)), tiny=a.tiny, dtype=a.dtype, variant=a.variant,
                        seed=int(a.seed), checkpoint=ck_block, limit=a.limit)
    mdir = EV.model_dir(a.out, probe)
    mdir.mkdir(parents=True, exist_ok=True)
    summ = None
    if a.variant == "full":
        mc = settings["merge_check"]
        summ = run_merge_check(ck / MODEL_SUB, model_key=a.model, tiny=a.tiny, n=mc["n"], checks=mc["checks"],
                               out_json=mdir / EV.MERGE_CHECK, device=a.device, dtype=a.dtype,
                               data_root=a.data_root or os.environ.get("VISINT_DATA_ROOT"), dev_subsets=a.dev_subsets)
    model = EV.EvalModel.from_checkpoint(ck, cfg, device=a.device, dtype=DTYPES[a.dtype])
    if man.get("lora") is not None and model.ad.lora_state != "merged":
        raise RuntimeError(f"load_trained(merge=True) 之后 LoRA 状态是 {model.ad.lora_state}，应为 merged")
    spec = EV.EvalSpec(**{**probe.__dict__, "adapter_model_key": str(getattr(model.ad, "model_key", entry.key))})
    paths = eval_paths(a)
    extra = {"checkpoint_dir": str(ck), "paths": paths.as_dict(), "merge_check": summ if summ is not None else "skipped（没有 LoRA）",
             "gpu": torch.cuda.get_device_name(0) if a.device.startswith("cuda") and torch.cuda.is_available() else None}
    for task in a.task:
        sel = EV.load_eval_set(task, paths)
        m = EV.run_eval_task(model, cfg, settings, spec, task, sel, a.out, resume=a.resume, argv=argv, run_extra=extra)
        h = m["headline"]
        vals = "  ".join(f"{k}={(h[k] or {}).get('value')}" for k in h if isinstance(h[k], dict))
        sh = m["leak_checks"]["shuffle"]["metrics"]
        print(f"{entry.key} e3a/{a.variant} seed{a.seed} {task} n={m['n']} {h['upsample']}: {vals} | shuffle: "
              + "  ".join(f"{k}={(v or {}).get('value')}" for k, v in EV.headline(task, sh).items()))
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    a = parse_args(argv)
    try:
        return cmd_run(a, argv)
    except MergeCheckFailed as e:
        print(f"拒绝评测：{e}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001 — 证据先落盘，再以非零码退出
        err_dir = Path(a.out) / EV.EVAL_SUB / EV.EXPERIMENT / a.variant / a.model / f"seed{a.seed}" / "errors"
        err_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = err_dir / f"error_{stamp}.json"
        path.write_text(json.dumps({"argv": argv, "error": repr(e), "traceback": traceback.format_exc()[-6000:]},
                                   indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"失败：{e!r}（已写出 {path}）", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
