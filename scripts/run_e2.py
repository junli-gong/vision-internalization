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
其余全部层）；--layers L24 L8 …（DEV-0038）只训练给定层上的探针（不补最后一层；要 L{N}_pre 就写进列表），探针设置不变；配置的 layers.pre_norm_last 为真时另加特征键 L{N}_pre（最后一层归一化之前，单列；D53-3）。--reference post_proj 就是
第 0 层（D21-3），只核对、不另训；pre_proj 用适配器的 pre_projection_features（三个家族都已实现，D59-8、D60；各家族取的位置写在适配器里）。
--stop-after-steps N：本次只跑 N 步训练就写检查点退出（分段运行；下次加 --resume 接着跑）。
去重（硬约束 7；DEV-0024）：训练池的剔除清单必须有这个训练集的键、且 eval_sets 含这个任务的评测集，否则开跑前报错；旧格式的
清单（没有 eval_sets）只在 --tiny 时照读。
训练后的检查点（DEV-0038 §2；D77-2(b)：E3 的机制测量，A800）：加 --checkpoint <检查点> --checkpoint-label <标签>，只用于正式运行：
    python scripts/run_e2.py --model qwen3.5-2b --task depth refseg --layers L24 L8 --data-root $DATA --eval-root $E1_DATA \
        --checkpoint <E3b 训练叶目录>/ckpt/final/model --checkpoint-label e3b/seed0 --out experiments/E3/00NN_e3_probes
    # E3b、C3、C1′、预热：<叶>/ckpt/final/model；E3a：<叶>/ckpt/final（或 final/model），只用 LoRA，不用查询与读出；C1、C2 拒绝
    # （LLM 权重没动：引用 E2 的 base 读数）。先做合并核对（src/e3/merge_check，判据 v1；输出 <out>/<标签>/<模型>/merge_check.json；
    # 用 --eval-root（没有就 --data-root）下 E1 的 grid16 开发子集），再 load_trained(merge=True)；结果写进 <out>/<标签>/<模型>/<任务>/，
    # config.yaml 另记 checkpoint（manifest 的 sha256 等）。规则见 src/e2/checkpoint.py。
退出码：0 = 跑完或按 --stop-after-steps 停下；1 = --checkpoint 的合并核对没有通过，拒绝运行；2 = 出错（含被拒的检查点，如 C1、C2；
报错先写进 <out>/<模型>/errors/，再退出；argparse 用法错误也是 2）。
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
    explicit_pre_norm,
    load_config,
    parse_layer_spec,
    pre_norm_key,
    resolve_task,
)
from src.e2 import checkpoint as E2CK  # noqa: E402
from src.e3.merge_check import MergeCheckFailed  # noqa: E402
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
    p.add_argument("--layers", nargs="+", help="all、every k，或指定层 L24 L8 …（默认取配置的 layers）")
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
    p.add_argument("--checkpoint", help="训练后的检查点（DEV-0038）：合并核对 → load_trained(merge=True) 替代 base；规则见 src/e2/checkpoint.py")
    p.add_argument("--checkpoint-label", help="--checkpoint 时必给：结果放在 <out>/<标签>/<模型>/<任务>/（如 e3b/seed0）")
    p.add_argument("--merge-check-n", type=int, help="--checkpoint 的合并核对的开发样本数（默认取 configs/e3b_train.json 的 "
                                                     "eval.merge_check.n；--tiny 取 eval.tiny.merge_check_n）")
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
    if a.checkpoint is not None:  # DEV-0038
        if a.estimate:
            p.error("--checkpoint 只用于正式运行（不能配 --estimate）")
        lab = a.checkpoint_label
        if not lab or not E2CK.CHECKPOINT_LABEL.fullmatch(lab) or ".." in lab:
            p.error(f"--checkpoint 要配 --checkpoint-label（相对路径，字母、数字、_ . -，用 / 分隔），收到 {lab!r}")
        if not Path(a.checkpoint).is_dir():
            p.error(f"--checkpoint {a.checkpoint!r} 不是目录")
        if a.merge_check_n is not None and a.merge_check_n < 1:
            p.error("--merge-check-n 必须是正整数")
    elif a.checkpoint_label is not None or a.merge_check_n is not None:
        p.error("--checkpoint-label、--merge-check-n 只配 --checkpoint 用")
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


def build_checkpoint_adapter(a):
    """--checkpoint（DEV-0038；src/e2/checkpoint.py）：认变体（C1、C2 拒绝）→ 构造方式核对 → 合并核对（不通过抛 MergeCheckFailed）
    → load_trained(merge=True)。返回 (entry, 适配器, config.yaml 的 checkpoint, 运行记录的附加项)。"""
    from src.adapters.checkpoint import policy_spec
    from src.e3.config import load_config as load_e3b_config
    from src.e3.merge_check import run_merge_check, settings_from_config

    entry = registry.get_model(a.model)
    ref = E2CK.resolve_checkpoint(a.checkpoint)
    spec = ref.manifest.get("build_spec") or {}
    if spec.get("key") != a.model or bool(spec.get("tiny")) != a.tiny:
        raise ValueError(f"检查点的构造方式（key {spec.get('key')!r}，tiny {spec.get('tiny')}）与 --model {a.model!r}、--tiny {a.tiny} 不符")
    if not a.tiny and ref.manifest.get("image_policy") != policy_spec(default_image_policy(entry.family)):
        raise ValueError(f"检查点的图像口径 {ref.manifest.get('image_policy')} 不是 E2 的冻结口径（同 E1，D28-3）：探针设置不变，拒绝")
    mc = settings_from_config(load_e3b_config().doc, tiny=a.tiny, n=a.merge_check_n, purpose=E2CK.MERGE_CHECK_PURPOSE)
    out_json = Path(a.out) / a.checkpoint_label / entry.key / E2CK.MERGE_CHECK_FILE
    summ = run_merge_check(ref.model_dir, model_key=a.model, tiny=a.tiny, n=mc["n"], checks=mc["checks"], out_json=out_json,
                           device=a.device, dtype=a.dtype, data_root=a.eval_root or a.data_root or os.environ.get("VISINT_DATA_ROOT"),
                           dev_subsets=a.dev_subsets)
    ad = E2CK.load_for_probes(ref, device=a.device, dtype=DTYPES[a.dtype])
    info = E2CK.checkpoint_record(ref, label=a.checkpoint_label, merge_check=summ)
    return entry, ad, info, {"checkpoint_dir": str(ref.model_dir), "merge_check": summ}


def make_spec(a, cfg, entry, ad, task: str, checkpoint=None) -> RunSpec:
    policy = default_image_policy(entry.family) if checkpoint is None else ad.image_policy  # 检查点：照记它自己的口径（真模型已核对相同）
    layer_spec = a.layers or cfg.default_layer_spec(entry.key)
    layers = expand_layers(layer_spec, ad.num_layers(), always_include_last=bool(cfg.doc["layers"]["always_include_last"]))
    pre_norm = explicit_pre_norm(layer_spec, ad.num_layers())  # 指定层（DEV-0038）：列表里有 L{N}_pre 才加；否则取配置
    if "post_proj" in a.reference and 0 not in layers:
        layers = [0] + layers
    return RunSpec(model_key=entry.key, family=entry.family, repo=entry.repo, revision=entry.revision,
                   adapter_model_key=str(getattr(ad, "model_key", entry.key)), tiny=a.tiny,
                   policy={"type": type(policy).__name__, **policy.as_dict()}, model_dtype=a.dtype, task=task,
                   layer_spec=layer_spec, layers=tuple(layers), references=tuple(a.reference),
                   num_layers=int(ad.num_layers()), hidden_size=int(ad.hidden_size()), eval_limit=a.eval_limit,
                   pre_norm_last=cfg.pre_norm_last if pre_norm is None else pre_norm, checkpoint=checkpoint)


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
    out, ck_info, extra = Path(a.out), None, None
    if a.checkpoint:  # DEV-0038：训练后的检查点，结果放在 <out>/<标签>/<模型>/<任务>/
        entry, ad, ck_info, extra = build_checkpoint_adapter(a)
        out = out / a.checkpoint_label
    else:
        entry, ad = build_adapter(a)
    for task in a.task:
        spec = make_spec(a, cfg, entry, ad, task, checkpoint=ck_info)
        tc = resolve_task(cfg, task, tiny=a.tiny)
        m = run_task(ad, cfg, spec, tc, paths, out, resume=a.resume, argv=argv, ckpt_root=a.ckpt_root,
                     stop_after_steps=a.stop_after_steps, keep_optimizer_state=a.keep_optimizer_state, run_extra=extra)
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
    except MergeCheckFailed as e:  # --checkpoint：合并核对没有通过（DEV-0038）
        print(f"拒绝运行：{e}", file=sys.stderr)
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
