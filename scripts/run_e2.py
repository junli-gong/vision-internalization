#!/usr/bin/env python
"""E2 探针框架的入口（DEV-0023；简报 §8.3）：冻结模型，在线训练逐层线性探针与两层卷积探针，最后在评测集上评一次。

运行（训练数据在 A800，D47、D49；--out 是本次运行的目录 experiments/E2/<NNNN_slug>，结果写进 <out>/<模型>/<任务>/）：
    python scripts/run_e2.py --model qwen3.5-0.8b --task all --data-root $DATA [--eval-root $E1_DATA] \\
        --out experiments/E2/0030_e2_probes [--layers all | --layers every 2] [--reference post_proj pre_proj] \\
        [--resume] [--ckpt-root DIR] [--stop-after-steps N]
    # --data-root：DEV-0022 的落地根（含 dedupe/train_exclusions.json；ade20k、nyu_normals 的评测划分也在这里）
    # --eval-root：E1 的落地根（nyuv2 的官方 654、refcoco 的 val），默认与 --data-root 相同
    # 法向：D59-6 许可已确认（Bae 2021 / GeoNet；仅限非商业研究、不分发），--task all 含法向，不再需要 --enable-normals（D63-4）
吞吐与算力估算（E2 预注册用；每个任务实测几步训练与几条评测，按配置外推 GPU 小时）：
    python scripts/run_e2.py --estimate --model qwen3.5-9b --task all --data-root $DATA --out <dir> [--reference post_proj pre_proj]
    # 写 <out>/<模型>/estimate/estimate.json 与 estimate.md；表头写明层、参考线、是否含法向、运行时的卷积与矩阵乘精度设置
    # （只记录、不改，D64-4），每行写明上采样实现；
    # 给了 --reference pre_proj 时，pre_proj 那组探针（探针数、步时间）与视觉塔的额外前向（前向开销）都在实测里
开发机彩排（小型随机模型、CPU、tests/fixtures/e2 与 tests/fixtures/e1 的小数据）：上面任一条加 --tiny，可省 --data-root。

--task：depth semseg normals refseg 中的一个或多个，或 all。--layers 不给时取配置（configs/e2_probes.json 的 layers：9B 每隔一层，
其余全部层）；配置的 layers.pre_norm_last 为真时另加特征键 L{N}_pre（最后一层归一化之前，单列；D53-3）。--reference post_proj 就是
第 0 层（D21-3），只核对、不另训；pre_proj 用适配器的 pre_projection_features（三个家族都已实现，D59-8、D60；各家族取的位置写在适配器里）。
--stop-after-steps N：本次只跑 N 步训练就写检查点退出（分段运行；下次加 --resume 接着跑）。
去重（硬约束 7；DEV-0024）：训练池的剔除清单必须有这个训练集的键、且 eval_sets 含这个任务的评测集，否则开跑前报错；旧格式的
清单（没有 eval_sets）只在 --tiny 时照读。
退出码：0 = 跑完或按 --stop-after-steps 停下；2 = 出错（报错先写进 <out>/<模型>/errors/，再退出；argparse 用法错误也是 2）。
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
from src.e2.config import (  # noqa: E402
    CONFIG_PATH,
    REFERENCES,
    TASKS,
    expand_layers,
    load_config,
    parse_layer_spec,
    pre_norm_key,
    resolve_task,
)
from src.e2.bilinear import audits as upsample_audits  # noqa: E402
from src.e2.data import E2Paths  # noqa: E402
from src.e2.estimate import estimate_task, summary_markdown  # noqa: E402
from src.e2.runner import RunSpec, backend_precision, run_task  # noqa: E402

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
TINY_INIT_STD = 0.05  # 与 run_e1 / validate_batching 相同：小模型重取参数，隐藏状态才随输入变化
RUN_DIR_NAME = re.compile(r"[0-9]{4}_[A-Za-z0-9][A-Za-z0-9_.-]*")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--estimate", action="store_true", help="吞吐与算力估算（每个任务实测几步，按配置外推）")
    p.add_argument("--model", required=True, help="注册键（src/adapters/registry.py）")
    p.add_argument("--task", nargs="+", required=True, help=f"任务名或 all：{list(TASKS)}")
    p.add_argument("--layers", nargs="+", help="all 或 every k（默认取配置的 layers）")
    p.add_argument("--reference", nargs="+", default=[], choices=REFERENCES, help="参考线：post_proj（= 第 0 层）、pre_proj")
    p.add_argument("--enable-normals", action="store_true",
                   help="无作用，只为兼容旧命令：D59-6 已确认法向数据的许可（Bae 2021 / GeoNet），真模型跑 normals 不再需要它（D63-4）")
    p.add_argument("--out", required=True, help="本次运行的目录（experiments/E2/<NNNN_slug>）")
    p.add_argument("--resume", action="store_true", help="续跑：从检查点接着训练")
    p.add_argument("--ckpt-root", help="检查点的根目录（默认 <叶目录>/ckpt；检查点不入库）")
    p.add_argument("--stop-after-steps", type=int, help="本次只跑 N 步训练，写检查点后退出（下次 --resume）")
    p.add_argument("--keep-optimizer-state", action="store_true",
                   help="任务完成后保留 ckpt/state.pt（优化器状态；默认删掉，探针权重留在 probes_final.pt）")
    p.add_argument("--eval-limit", type=int, help="评测集只取前 N 条（调试；进运行身份，正式运行不用）")
    p.add_argument("--tiny", action="store_true", help="小型随机模型 + CPU + fixture 的小数据（开发机彩排）")
    p.add_argument("--data-root", help="DEV-0022 的落地根（默认环境变量 VISINT_DATA_ROOT；--tiny 时取 fixture）")
    p.add_argument("--eval-root", help="E1 的落地根（nyuv2、refcoco；默认与 --data-root 相同）")
    p.add_argument("--dev-subsets", help="E1 的开发子集清单（默认 configs/e1_dev_subsets.json；--tiny 取 fixture）")
    p.add_argument("--config", default=str(CONFIG_PATH), help="E2 配置（默认 configs/e2_probes.json）")
    p.add_argument("--device", help="默认：真模型 cuda，--tiny 时 cpu")
    p.add_argument("--dtype", choices=sorted(DTYPES), help="默认：真模型 bf16，--tiny 时 fp32")
    a = p.parse_args(argv)
    if a.task == ["all"]:
        a.task = list(TASKS)
    bad = [t for t in a.task if t not in TASKS]
    if bad:
        p.error(f"未知任务 {bad}；可用：{list(TASKS)} 或 all")
    if len(set(a.task)) != len(a.task):
        p.error("--task 有重复")
    if a.model not in registry.MODELS:
        p.error(f"未登记的模型键 {a.model!r}；已登记：{sorted(registry.MODELS)}")
    if a.layers is not None:
        try:
            a.layers = parse_layer_spec(a.layers)
        except ValueError as e:
            p.error(str(e))
    if len(set(a.reference)) != len(a.reference):
        p.error("--reference 有重复")
    for name in ("stop_after_steps", "eval_limit"):
        v = getattr(a, name)
        if v is not None and v < 1:
            p.error(f"--{name.replace('_', '-')} 必须是正整数")
    if a.estimate and (a.resume or a.stop_after_steps):
        p.error("--estimate 不接受 --resume / --stop-after-steps")
    a.device = a.device or ("cpu" if a.tiny else "cuda")
    a.dtype = a.dtype or ("fp32" if a.tiny else "bf16")
    if not a.tiny and not (a.data_root or os.environ.get("VISINT_DATA_ROOT")):
        p.error("真模型运行要给 --data-root（或设 VISINT_DATA_ROOT）")
    if not a.tiny and not RUN_DIR_NAME.fullmatch(Path(a.out).name):
        p.error(f"--out 的目录名必须带运行编号，形如 0030_e2_probes（CLAUDE.md：experiments/<phase>/<NNNN_slug>/），收到 {a.out!r}")
    return a


def data_paths(a) -> E2Paths:
    if a.tiny and not a.data_root:
        base = E2Paths.fixtures()
    else:
        root = Path(a.data_root or os.environ["VISINT_DATA_ROOT"])
        base = E2Paths(train_root=root, eval_root=Path(a.eval_root) if a.eval_root else root)
    if a.eval_root and a.tiny and not a.data_root:
        base = E2Paths(train_root=base.train_root, eval_root=Path(a.eval_root), dev_subsets=base.dev_subsets)
    if a.dev_subsets:
        base = E2Paths(train_root=base.train_root, eval_root=base.eval_root, dev_subsets=Path(a.dev_subsets))
    return base


def build_adapter(a):
    entry = registry.get_model(a.model)
    kw = {"init_std": TINY_INIT_STD} if a.tiny else {}
    ad = make_adapter(a.model, device=a.device, dtype=DTYPES[a.dtype], tiny=a.tiny, **kw)
    return entry, ad


def make_spec(a, cfg, entry, ad, task: str) -> RunSpec:
    policy = default_image_policy(entry.family)
    layer_spec = a.layers or cfg.default_layer_spec(entry.key)
    layers = expand_layers(layer_spec, ad.num_layers(), always_include_last=bool(cfg.doc["layers"]["always_include_last"]))
    if "post_proj" in a.reference and 0 not in layers:
        layers = [0] + layers
    return RunSpec(model_key=entry.key, family=entry.family, repo=entry.repo, revision=entry.revision,
                   adapter_model_key=str(getattr(ad, "model_key", entry.key)), tiny=a.tiny,
                   policy={"type": type(policy).__name__, **policy.as_dict()}, model_dtype=a.dtype, task=task,
                   layer_spec=layer_spec, layers=tuple(layers), references=tuple(a.reference),
                   num_layers=int(ad.num_layers()), hidden_size=int(ad.hidden_size()), eval_limit=a.eval_limit,
                   pre_norm_last=cfg.pre_norm_last)


def _headline(m) -> str:
    out = []
    for h in m["headline"][:2]:
        best = None
        for name, p in m["probes"].items():
            v = p["eval"].get(h["metric"])
            if v is None:
                continue
            if best is None or (v < best[1] if h["better"] == "↓" else v > best[1]):
                best = (name, v)
        if best:
            out.append(f"{h['metric']} 最好 {best[1]:.4f}（{best[0]}）")
    return "；".join(out)


def reference_notice(a) -> None:
    if "post_proj" in a.reference:
        print("注意：post_proj 就是第 0 层（D21-3：vision_features 与第 0 层在图像位置上逐位相同）；只核对、不另训一组探针，"
              "G2 解读时不能算作第二条证据", file=sys.stderr, flush=True)


def cmd_run(a, argv) -> int:
    cfg = load_config(a.config)
    paths = data_paths(a)
    reference_notice(a)
    entry, ad = build_adapter(a)
    for task in a.task:
        spec = make_spec(a, cfg, entry, ad, task)
        tc = resolve_task(cfg, task, tiny=a.tiny)
        m = run_task(ad, cfg, spec, tc, paths, a.out, resume=a.resume, argv=argv, ckpt_root=a.ckpt_root,
                     stop_after_steps=a.stop_after_steps, keep_optimizer_state=a.keep_optimizer_state)
        if m is None:
            print(f"{entry.key} {task}：按 --stop-after-steps 停下，检查点已写；加 --resume 接着跑（后面的任务没有开始）")
            return 0
        print(f"{entry.key} {task} n_eval={m['n_eval']} probes={len(m['probes'])} {_headline(m)}")
    return 0


def cmd_estimate(a, argv) -> int:
    cfg = load_config(a.config)
    paths = data_paths(a)
    reference_notice(a)
    entry, ad = build_adapter(a)
    est = cfg.doc["estimate"]
    rows = []
    spec0 = None
    for task in a.task:
        spec = spec0 = make_spec(a, cfg, entry, ad, task)
        tc = resolve_task(cfg, task, tiny=a.tiny)
        rows.append(estimate_task(ad, cfg, spec, tc, paths, train_steps=int(est["train_steps"]),
                                  eval_samples=int(est["eval_samples"]),
                                  log=lambda m: print(m, file=sys.stderr, flush=True)))
    dev = torch.device(a.device)
    header = {"model": entry.key, "revision": entry.revision, "device": a.device, "dtype": a.dtype, "tiny": a.tiny,
              "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None, "config_sha256": cfg.sha256,
              "layer_spec": spec0.layer_spec, "n_layers": len(spec0.layers),
              "pre_norm_last": pre_norm_key(spec0.num_layers) if spec0.pre_norm_last else None,
              "references": list(spec0.references), "tasks": list(a.task), "includes_normals": "normals" in a.task,
              "upsample_impls": sorted({r["upsample_impl"] for r in rows}), "upsample_audit": upsample_audits(),
              "precision": backend_precision()}
    doc = {"header": header, "tasks": rows, "gpu_hours_total": sum(r["gpu_hours"] for r in rows), "argv": list(argv),
           "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    base = Path(a.out) / entry.key / "estimate"
    base.mkdir(parents=True, exist_ok=True)
    (base / "estimate.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (base / "estimate.md").write_text(summary_markdown(entry.key, rows, header), encoding="utf-8")
    print(f"gpu_hours_total={doc['gpu_hours_total']:.3f} -> {base / 'estimate.json'}")
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    a = parse_args(argv)
    try:
        return cmd_estimate(a, argv) if a.estimate else cmd_run(a, argv)
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
