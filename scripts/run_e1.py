#!/usr/bin/env python
"""E1 评测框架的入口（DEV-0014；协议 v2：DEV-0017）：逐模型、逐任务地跑 T 文本通道零样本基线；另有批处理验收与吞吐估算两种模式。

运行（服务器；--out 是本次运行的目录 experiments/E1/<NNNN_slug>，结果写进 <out>/<模型>/<任务>…/）：
    python scripts/run_e1.py --model qwen3.5-0.8b --task refexp_box --subset eval \\
        --data-root $PROJ/data --out experiments/E1/0011_e1_main [--resume] [--limit N]
    # 思考子集：--subset think（预算强制收尾，D59-1：思考 4096、作答 1024，adapter.generate_batch_budgeted；think8192 已取消）；
    # 附加对照：--subset extra --policy qwen:1024（或 internvl:native）
    # 选择题与点对任务的 eval、dev 是受限字母读数（1 个 token）；自由作答的诊断：--subset free（think 子集的 id，关思考）
批配置冻结（configs/e1_prompts.json 的 batch_protocol，D36-1）：--batch-size 默认取冻结值（64，D43）；真模型给别的值要加
--allow-unfrozen-batch（config.yaml 记 frozen = false）。--tiny 默认 4，不受此限。
批处理验收 v3（D36-1；开发子集；A1、A2 = 冻结的批配置跑两遍，B = composition_b 的批大小跑一遍；规则见配置的 acceptance）：
    python scripts/run_e1.py --acceptance --model qwen3.5-0.8b --task all --data-root $PROJ/data --out <dir> [--resume]
    # 写 <out>/<模型>/acceptance/{A1,A2,B}/… 与 acceptance.json、acceptance.md；检测要先落地 coco_train2014_e1dev。
    # 已跑完的验收重新比较：同一条命令加 --resume（三遍都已完成，不再生成）。--alt-batch-size 只用于 --tiny 或配
    # --allow-unfrozen-batch。
吞吐与算力估算（开发子集，每个 --batch-sizes 各测一遍，外推到各子集的 n；外推用冻结的批大小，没测时退回推荐值）：
    python scripts/run_e1.py --estimate --model qwen3.5-0.8b --task all --batch-sizes 1 8 16 32 --data-root $PROJ/data \\
        --out <dir> [--estimate-think]
开发机彩排（小型随机模型、CPU、tests/fixtures/e1 的小数据）：上面任一条加 --tiny，可省 --data-root。
训练后模型的 T 文本评测（DEV-0030 §2；跨通道迁移）：加 --checkpoint <检查点目录> --checkpoint-label <标签>，只用于正式运行模式
（不能配 --acceptance、--estimate、附加口径）：
    python scripts/run_e1.py --model qwen3.5-0.8b --task all --subset eval --data-root $PROJ/data \\
        --checkpoint <训练叶目录>/ckpt/final/model --checkpoint-label t_text/e3b/seed0 --out experiments/E3/00NN_e3b_ttext
    # 结果写进 <out>/<标签>/<模型>/<任务>…/；先跑评测前的合并核对（src/e3/merge_check：check_training_interfaces --checkpoint 的同一套
    # 函数，n 取 configs/e3b_train.json 的 eval.merge_check，可用 --merge-check-n 改），写进 <out>/<标签>/<模型>/merge_check.json，
    # 不通过就不评测（退出码 1）。base 换成 factory.load_trained(merge=True)，其余协议不变（冻结口径、批配置、三列）；检查点的图像
    # 口径必须就是 E1 冻结的口径（--tiny 时照记检查点自己的口径）。config.yaml 另记 checkpoint（manifest 的 sha256 等）；metrics.json
    # 的 diagnostics 另记自由生成里冒出网格 token 的条数（grid_tokens_in_output）。

事后读数（DEV-0033；D66；E1 预注册 §10，写于运行之前；口径在 configs/e1_posthoc.json，主评测的配置与输出不变）：
    # 去偏字母读数（§10.1，N22 → (b)）：四个选择题与点对任务的全量评测集，k 种循环移位各读一次（option_logits），合法字母上 softmax、
    # 映射回选项内容取平均；同一次运行顺带给出原字母读数（s = 0，批组成同主评测）。--subset debiased_dev 读开发子集（验收用）。
    python scripts/run_e1.py --model qwen3.5-4b --task all --subset debiased --data-root $PROJ/data --out experiments/E1/<NNNN_slug>
    # 格式强制诊断（§10.2，只做 Qwen）：思考子集的 id，关思考；refexp_poly 预填 "[["，grid16 受限解码固定 271 个 token（主读法 restricted
    # 与结构强制 structured 两个目录）。
    python scripts/run_e1.py --model qwen3.5-4b --task all --subset forced --data-root $PROJ/data --out experiments/E1/<NNNN_slug>
    # 写 <out>/<模型>/<任务>__debiased[_dev]/ 与 <任务>__forced_<读法>/；--task all 取这种读法支持的任务；--resume、--limit、--tiny 同上
    # （--tiny 用 tests/fixtures/e1_posthoc 的小数据：BLINK、CV-Bench 的提示是真实格式，RefCOCO 的图放大到 48×36）。不能配 --acceptance、--estimate、
    # --checkpoint、附加口径。有效性检查与配对汇总：scripts/summarize_e1_posthoc.py。

--task：一个或多个任务名，或 all（refexp_box refexp_poly grid16 count ovd rel_depth_nyu rel_depth_blink spatial_cvbench
spatial_embspatial）。--policy：default（E1 冻结的口径，D28-3）、qwen:1024（Qwen 约 1K token）、internvl:native（InternVL 原生
动态 tile）；后两个只用于 --subset extra 的附加对照（也可用于 --estimate 量它们的吞吐）。
--count-questions：计数的问句来源，countbenchqa（默认，D36-2：big_vision 的人工问句，出自 PaliGemma，Gemma 家族可能见过；
要先落地 countbenchqa_questions）或 heuristic（从 caption 取对象）。只影响 count。
退出码：0 = 跑完；1 = --checkpoint 的合并核对没有通过，拒绝评测；2 = 出错（报错先写进 <out>/<模型>/errors/，再退出；argparse 用法
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
from src.adapters.factory import default_image_policy, make_adapter  # noqa: E402
from src.adapters.image_policy import (  # noqa: E402
    E1_SUBSET_PIXEL_BUDGET_TOKENS,
    INTERNVL_NATIVE_TILES,
    PixelBudget,
)
from src.e1 import acceptance as A  # noqa: E402
from src.e1 import estimate as EST  # noqa: E402
from src.e1.prompts import (  # noqa: E402
    COUNT_QUESTION_SOURCES,
    DEFAULT_COUNT_QUESTIONS,
    PROMPTS_PATH,
    SUBSET_KINDS,
    TASKS,
    load_prompt_config,
)
from src.e1.posthoc import DEBIASED_SUBSETS, FORCED_SUBSET, POSTHOC_PATH, POSTHOC_SUBSETS  # noqa: E402
from src.e1.runner import RunSpec, leaf_dir, policy_dirname, read_jsonl_tolerant, run_task  # noqa: E402
from src.e1.tasks import CHOICE_TASKS, TASK_SUBSETS, DataPaths, default_nyu_cache, select  # noqa: E402

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}  # 与 scripts/validate_batching.py 的 --dtype 取值相同
POLICIES = ("default", "qwen:1024", "internvl:native")
POLICY_FAMILY = {"qwen:1024": registry.FAMILY_QWEN35, "internvl:native": registry.FAMILY_INTERNVL35}
TINY_INIT_STD = 0.05  # 与 validate_batching 相同：小模型重取参数，否则 greedy 输出几乎与输入无关
DEFAULT_MAP_CI_RESAMPLES = 0  # mAP 的 CI 默认不算：每次重采样都跑 COCOeval，GPU 空等；标题 F1@0.5 的 CI 照常算
RUN_DIR_NAME = re.compile(r"[0-9]{4}_[A-Za-z0-9][A-Za-z0-9_.-]*")
CHECKPOINT_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*(/[A-Za-z0-9][A-Za-z0-9_.-]*)*")  # --checkpoint-label：相对路径，不含 ..
TINY_BATCH_SIZE = 4  # --tiny 的默认批大小（fixture 每个任务只有几条；不受冻结的批配置约束）
TINY_ALT_BATCH_SIZE = 3  # --tiny 验收的 B 遍批大小（与 4 不同，fixture 上也能换出不同的批组成）
# 事后读数（DEV-0033）各自支持的任务（--task all 展开成它们；运行时再与 configs/e1_posthoc.json 核对）
POSTHOC_TASKS = {**{s: CHOICE_TASKS for s in DEBIASED_SUBSETS}, FORCED_SUBSET: ("refexp_poly", "grid16")}


def resolve_policy(tag: str, family: str):
    if tag == "default":
        return default_image_policy(family)
    if POLICY_FAMILY[tag] != family:
        raise ValueError(f"口径 {tag} 只用于 {POLICY_FAMILY[tag]} 家族，这个模型是 {family}")
    return PixelBudget(E1_SUBSET_PIXEL_BUDGET_TOKENS) if tag == "qwen:1024" else INTERNVL_NATIVE_TILES


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--acceptance", action="store_true",
                      help="批处理验收 v3（开发子集；冻结的批配置跑两遍 A1、A2，composition_b 跑一遍 B）")
    mode.add_argument("--estimate", action="store_true", help="吞吐与算力估算（开发子集，--batch-sizes 各一遍）")
    p.add_argument("--model", required=True, help="注册键（src/adapters/registry.py）")
    p.add_argument("--task", nargs="+", required=True, help=f"任务名或 all：{list(TASKS)}")
    p.add_argument("--subset", default="eval", choices=SUBSET_KINDS + POSTHOC_SUBSETS,
                   help="运行模式的子集（验收与估算一律用开发子集）；debiased、debiased_dev、forced 是事后读数（DEV-0033，见上）")
    p.add_argument("--posthoc-config", default=str(POSTHOC_PATH), help="事后读数的配置（默认 configs/e1_posthoc.json）")
    p.add_argument("--policy", default="default", choices=POLICIES, help="图像口径；默认 E1 冻结的口径（D28-3）")
    p.add_argument("--batch-size", type=int, help=f"批大小（默认取冻结的 batch_protocol.batch_size；--tiny 默认 {TINY_BATCH_SIZE}）；"
                                                  "1 = 逐条 generate。真模型给别的值要加 --allow-unfrozen-batch")
    p.add_argument("--alt-batch-size", type=int, help="--acceptance 的 B 遍批大小（默认取 acceptance.composition_b.batch_size；"
                                                      f"--tiny 默认 {TINY_ALT_BATCH_SIZE}）")
    p.add_argument("--allow-unfrozen-batch", action="store_true",
                   help="真模型运行允许用冻结值以外的批大小（config.yaml 记 frozen = false；不是协议口径）")
    p.add_argument("--batch-sizes", type=int, nargs="+", help="--estimate 用的批大小（默认 1 8 16 32；--tiny 默认 1 2 4）")
    p.add_argument("--out", required=True, help="本次运行的目录（experiments/E1/<NNNN_slug>）")
    p.add_argument("--resume", action="store_true", help="续跑：跳过 predictions.jsonl 里已完成的 id")
    p.add_argument("--limit", type=int, help="只取每个任务的前 N 条（调试）")
    p.add_argument("--tiny", action="store_true", help="小型随机模型 + CPU + tests/fixtures/e1 的小数据（开发机彩排）")
    p.add_argument("--data-root", help="落地目录（默认环境变量 VISINT_DATA_ROOT；--tiny 时取 fixture）")
    p.add_argument("--prompts", default=str(PROMPTS_PATH), help="提示配置（默认 configs/e1_prompts.json）")
    p.add_argument("--dev-subsets", help="开发子集清单（默认 configs/e1_dev_subsets.json）")
    p.add_argument("--eval-subsets", help="评测子集清单（默认 configs/e1_eval_subsets.json）")
    p.add_argument("--nyu-pairs", help="NYU 点对配置（默认 configs/e1_nyu_pairs.json）")
    p.add_argument("--nyu-cache", help="NYU 点对渲染图的缓存目录（默认 $TMPDIR/visint_nyu_pairs_rendered）")
    p.add_argument("--device", help="默认：真模型 cuda，--tiny 时 cpu")
    p.add_argument("--dtype", choices=sorted(DTYPES), help="默认：真模型 bf16，--tiny 时 fp32")
    p.add_argument("--map-ci-resamples", type=int, help=f"检测补充指标 mAP 的 bootstrap 次数（默认 {DEFAULT_MAP_CI_RESAMPLES} = "
                                                         "不算；标题 F1@0.5 的 CI 总是算）")
    p.add_argument("--estimate-n", type=int, default=64, help="--estimate：每个任务最多用多少条开发样本（默认 64）")
    p.add_argument("--estimate-think", action="store_true", help="--estimate：另测思考子集（预算强制收尾，思考 4096 + 作答 1024，D59-1）的吞吐")
    p.add_argument("--estimate-think-n", type=int, default=8, help="--estimate-think 用的开发样本数（默认 8）")
    p.add_argument("--count-questions", choices=COUNT_QUESTION_SOURCES, default=DEFAULT_COUNT_QUESTIONS,
                   help=f"计数的问句来源（默认 {DEFAULT_COUNT_QUESTIONS}：big_vision 的人工问句，D36-2；heuristic = 从 caption 取对象）")
    p.add_argument("--checkpoint", help="训练后的检查点目录（DEV-0030 §2）：load_trained(merge=True) 替代 base，其余协议不变；"
                                        "评测前先跑合并核对")
    p.add_argument("--checkpoint-label", help="--checkpoint 时必给：结果放在 <out>/<标签>/<模型>/…（如 t_text/e3b/seed0）")
    p.add_argument("--merge-check-n", type=int, help="--checkpoint 的合并核对的开发样本数（默认取 configs/e3b_train.json 的 "
                                                     "eval.merge_check.n）")
    a = p.parse_args(argv)
    if a.checkpoint is not None:
        if a.acceptance or a.estimate:
            p.error("--checkpoint 只用于正式运行模式（不能配 --acceptance / --estimate）")
        if a.policy != "default":
            p.error("--checkpoint 只用 E1 冻结的口径（检查点的图像口径写在 manifest 里）；附加口径不适用")
        if not a.checkpoint_label or not CHECKPOINT_LABEL.fullmatch(a.checkpoint_label) or ".." in a.checkpoint_label:
            p.error(f"--checkpoint 要配 --checkpoint-label（相对路径，字母、数字、_ . -，用 / 分隔），收到 {a.checkpoint_label!r}")
        if not Path(a.checkpoint).is_dir():
            p.error(f"--checkpoint {a.checkpoint!r} 不是目录")
    elif a.checkpoint_label is not None or a.merge_check_n is not None:
        p.error("--checkpoint-label、--merge-check-n 只配 --checkpoint 用")
    if a.merge_check_n is not None and a.merge_check_n < 1:
        p.error("--merge-check-n 必须是正整数")
    if a.alt_batch_size is not None and not a.acceptance:
        p.error("--alt-batch-size 只用于 --acceptance")
    posthoc = a.subset in POSTHOC_SUBSETS
    if posthoc and (a.acceptance or a.estimate or a.checkpoint is not None or a.policy != "default"):
        p.error(f"--subset {a.subset}（事后读数）不能配 --acceptance、--estimate、--checkpoint 或附加口径")
    if a.task == ["all"]:
        a.task = list(POSTHOC_TASKS[a.subset]) if posthoc else list(TASKS)
    bad = [t for t in a.task if t not in TASKS]
    if bad:
        p.error(f"未知任务 {bad}；可用：{list(TASKS)} 或 all")
    if posthoc and any(t not in POSTHOC_TASKS[a.subset] for t in a.task):
        p.error(f"--subset {a.subset} 只支持 {list(POSTHOC_TASKS[a.subset])}，收到 {a.task}")
    if a.model not in registry.MODELS:
        p.error(f"未登记的模型键 {a.model!r}；已登记：{sorted(registry.MODELS)}")
    family = registry.get_model(a.model).family
    if a.policy != "default" and POLICY_FAMILY[a.policy] != family:
        p.error(f"--policy {a.policy} 只用于 {POLICY_FAMILY[a.policy]} 家族")
    if not (a.acceptance or a.estimate or posthoc):
        unsupported = [t for t in a.task if a.subset not in TASK_SUBSETS[t]]
        if unsupported:
            p.error(f"{unsupported} 不支持 --subset {a.subset}")
        if a.subset == "extra" and a.policy == "default":
            p.error("--subset extra 是附加对照，要配 --policy qwen:1024 或 internvl:native")
        if a.subset != "extra" and a.policy != "default":
            p.error("附加口径只用于 --subset extra（主口径冻结，D28-3）")
    a.device = a.device or ("cpu" if a.tiny else "cuda")
    a.dtype = a.dtype or ("fp32" if a.tiny else "bf16")
    if a.tiny:  # 真模型的默认批大小取配置里冻结的值（resolve_batch_sizes，在读配置之后）
        a.batch_size = a.batch_size or TINY_BATCH_SIZE
        if a.acceptance:
            a.alt_batch_size = a.alt_batch_size or TINY_ALT_BATCH_SIZE
    a.batch_sizes = a.batch_sizes or ([1, 2, 4] if a.tiny else [1, 8, 16, 32])
    if a.map_ci_resamples is None:
        a.map_ci_resamples = DEFAULT_MAP_CI_RESAMPLES
    for name in ("batch_size", "alt_batch_size", "limit", "estimate_n", "estimate_think_n"):
        v = getattr(a, name)
        if v is not None and v < 1:
            p.error(f"--{name.replace('_', '-')} 必须是正整数")
    if any(b < 1 for b in a.batch_sizes) or a.map_ci_resamples < 0:
        p.error("--batch-sizes 必须都是正整数，--map-ci-resamples 不能为负")
    if a.acceptance and 1 in (a.batch_size, a.alt_batch_size):
        p.error("--acceptance 的批大小必须 > 1（A、B 两种批组成都走 generate_batch，与正式运行同一条路径）")
    if a.acceptance and a.batch_size is not None and a.batch_size == a.alt_batch_size:
        p.error("--acceptance 的 B 遍批大小必须与 A 遍不同（否则两种批组成相同）")
    if not a.tiny and not (a.data_root or os.environ.get("VISINT_DATA_ROOT")):
        p.error("真模型运行要给 --data-root（或设 VISINT_DATA_ROOT）")
    if not a.tiny and not RUN_DIR_NAME.fullmatch(Path(a.out).name):
        p.error(f"--out 的目录名必须带运行编号，形如 0011_e1_main（CLAUDE.md：experiments/<phase>/<NNNN_slug>/），收到 {a.out!r}")
    return a


def data_paths(a) -> DataPaths:
    cache = Path(a.nyu_cache) if a.nyu_cache else default_nyu_cache()
    if a.tiny and not a.data_root:  # 事后读数的 --tiny 用 tests/fixtures/e1_posthoc（DEV-0033；见 tasks.FIXTURE_POSTHOC_DIR）
        fx = DataPaths.posthoc_fixtures if a.subset in POSTHOC_SUBSETS else DataPaths.fixtures
        base = fx(nyu_cache=cache)
    else:
        base = DataPaths(root=Path(a.data_root or os.environ["VISINT_DATA_ROOT"]), nyu_cache=cache)
    kw = {k: Path(v) for k, v in (("dev_subsets", a.dev_subsets), ("eval_subsets", a.eval_subsets),
                                 ("nyu_pairs", a.nyu_pairs)) if v}
    return DataPaths(root=base.root, dev_subsets=kw.get("dev_subsets", base.dev_subsets),
                     eval_subsets=kw.get("eval_subsets", base.eval_subsets), nyu_pairs=kw.get("nyu_pairs", base.nyu_pairs),
                     nyu_cache=base.nyu_cache, strict=base.strict)


def build_adapter(a):
    entry = registry.get_model(a.model)
    policy = resolve_policy(a.policy, entry.family)
    kw = {"init_std": TINY_INIT_STD} if a.tiny else {}
    ad = make_adapter(a.model, device=a.device, dtype=DTYPES[a.dtype], image_policy=policy, tiny=a.tiny, **kw)
    return entry, policy, ad


def build_checkpoint_adapter(a):
    """--checkpoint（DEV-0030 §2）：构造方式核对 → 评测前的合并核对（不通过抛 MergeCheckFailed）→ load_trained(merge=True)。
    返回 (entry, 实际的图像口径, 适配器, config.yaml 的 checkpoint, 运行记录的附加项)。"""
    from src.adapters.checkpoint import policy_spec, read_manifest
    from src.adapters.factory import load_trained
    from src.e3.config import load_config
    from src.e3.merge_check import run_merge_check, settings_from_config
    from src.runlog import sha256_file

    entry = registry.get_model(a.model)
    ck = Path(a.checkpoint)
    man = read_manifest(ck)
    spec = man.get("build_spec") or {}
    if spec.get("key") != a.model or bool(spec.get("tiny")) != a.tiny:
        raise ValueError(f"检查点的构造方式（key {spec.get('key')!r}，tiny {spec.get('tiny')}）与 --model {a.model!r}、"
                         f"--tiny {a.tiny} 不符")
    mc = settings_from_config(load_config().doc, tiny=a.tiny, n=a.merge_check_n)
    summ = run_merge_check(ck, model_key=a.model, tiny=a.tiny, n=mc["n"], checks=mc["checks"],
                           out_json=Path(a.out) / a.checkpoint_label / entry.key / "merge_check.json", device=a.device,
                           dtype=a.dtype, data_root=a.data_root or os.environ.get("VISINT_DATA_ROOT"), dev_subsets=a.dev_subsets)
    ad, _ = load_trained(ck, device=a.device, dtype=DTYPES[a.dtype], merge=True)
    if man.get("lora") is not None and ad.lora_state != "merged":
        raise RuntimeError(f"load_trained(merge=True) 之后 LoRA 状态是 {ad.lora_state}，应为 merged")
    policy = ad.image_policy
    frozen = default_image_policy(entry.family)
    if not a.tiny and policy_spec(policy) != policy_spec(frozen):
        raise ValueError(f"检查点的图像口径 {policy_spec(policy)} 不是 E1 冻结的口径 {policy_spec(frozen)}（D28-3）：协议不变，拒绝评测")
    info = {"manifest_sha256": sha256_file(ck / "manifest.json"), "label": a.checkpoint_label,
            "build_spec": man.get("build_spec"), "image_policy": man.get("image_policy"), "dtype": man.get("dtype"),
            "lora": man.get("lora"), "trainable_token_rows": man.get("trainable_token_rows"),
            "n_grid_tokens": len(ad.grid_token_ids), "merge_check": {"criteria_version": summ["criteria_version"],
                                                                     "n": summ["n"], "checks": summ["checks"]}}
    return entry, policy, ad, info, {"checkpoint_dir": str(ck), "merge_check": summ}


def make_spec(a, entry, policy, ad, task: str, subset: str, batch_size: int, checkpoint=None) -> RunSpec:
    return RunSpec(model_key=entry.key, family=entry.family, repo=entry.repo, revision=entry.revision,
                   adapter_model_key=str(getattr(ad, "model_key", entry.key)), task=task, subset=subset,
                   policy_tag=a.policy, policy={"type": type(policy).__name__, **policy.as_dict()}, batch_size=batch_size,
                   limit=a.limit, tiny=a.tiny, map_ci_resamples=a.map_ci_resamples, count_questions=a.count_questions,
                   checkpoint=checkpoint)


def selection(task: str, subset: str, paths: DataPaths, limit: int | None, count_questions: str = DEFAULT_COUNT_QUESTIONS):
    sel = select(task, subset, paths, count_questions=count_questions)
    if limit is not None:
        sel.samples = sel.samples[:limit]
        sel.stats["limit"] = limit
        sel.stats["n"] = len(sel.samples)
    return sel


def _headline(m) -> str:
    return "; ".join(f"{h['name']}({h['scope']})={h['value']}" for h in m["headline"]) or "（无 GT）"


def resolve_batch_sizes(a, prompts) -> tuple[int, int | None]:
    """(A 遍 / 正式运行的批大小, 验收 B 遍的批大小)。真模型：默认取冻结值（batch_protocol、acceptance.composition_b）；
    给了别的值要 --allow-unfrozen-batch，否则报错（D36-1：批配置是协议的一部分）。--tiny 用 parse_args 定的小值。"""
    frozen = prompts.batch_protocol()["batch_size"]
    alt_cfg = prompts.acceptance()["composition_b"]["batch_size"]
    bs = a.batch_size if a.batch_size is not None else frozen
    alt = (a.alt_batch_size if a.alt_batch_size is not None else alt_cfg) if a.acceptance else None
    if not a.tiny and not a.allow_unfrozen_batch:
        if bs != frozen:
            raise ValueError(f"--batch-size {bs} 不是冻结的批大小 {frozen}（configs/e1_prompts.json 的 batch_protocol）；"
                             "确要换就加 --allow-unfrozen-batch")
        if a.acceptance and alt != alt_cfg:
            raise ValueError(f"--alt-batch-size {alt} 不是协议的 composition_b.batch_size {alt_cfg}；确要换就加 --allow-unfrozen-batch")
    if a.acceptance and (alt == bs or 1 in (bs, alt)):
        raise ValueError(f"验收的两种批大小必须都 > 1 且不同，收到 A = {bs}、B = {alt}")
    return bs, alt


def cmd_run(a, argv) -> int:
    prompts = load_prompt_config(a.prompts)
    bs, _ = resolve_batch_sizes(a, prompts)
    paths = data_paths(a)
    ck_info, extra, out = None, None, Path(a.out)
    if a.checkpoint:
        entry, policy, ad, ck_info, extra = build_checkpoint_adapter(a)
        out = out / a.checkpoint_label
    else:
        entry, policy, ad = build_adapter(a)
    for task in a.task:
        sel = selection(task, a.subset, paths, a.limit, a.count_questions)
        spec = make_spec(a, entry, policy, ad, task, a.subset, bs, checkpoint=ck_info)
        m = run_task(ad, spec, sel, prompts, paths, out, resume=a.resume, argv=argv, run_extra=extra)
        print(f"{entry.key} {task}/{a.subset} n={m['n']} parse_fail={m['parse']['failure_rate']} "
              f"trunc={m['truncation']['rate']} valid={m['valid']['rate']} {_headline(m)}")
    return 0


def cmd_posthoc(a, argv) -> int:
    """事后读数（DEV-0033）：--subset debiased | debiased_dev（去偏字母读数）或 forced（格式强制，每个任务的各种读法各一个目录）。
    口径、批配置与提示配置的核对见 configs/e1_posthoc.json 与 src/e1/posthoc_runner.py。"""
    from src.e1.posthoc import load_posthoc_config
    from src.e1.posthoc_runner import forced_subset_name, run_debiased_task, run_forced_task

    prompts = load_prompt_config(a.prompts)
    posthoc = load_posthoc_config(prompts, a.posthoc_config)
    bad = [t for t in a.task if t not in posthoc.tasks_for(a.subset)]
    if bad:
        raise ValueError(f"{bad} 不在 {a.posthoc_config} 给 {a.subset} 的任务 {posthoc.tasks_for(a.subset)} 里")
    bs, _ = resolve_batch_sizes(a, prompts)
    paths = data_paths(a)
    entry, policy, ad = build_adapter(a)
    for task in a.task:
        sel = selection(task, posthoc.base_subset(a.subset), paths, a.limit, a.count_questions)
        spec = make_spec(a, entry, policy, ad, task, a.subset, bs)
        if a.subset in DEBIASED_SUBSETS:
            m = run_debiased_task(ad, spec, sel, prompts, posthoc, paths, a.out, resume=a.resume, argv=argv)
            d = m["debiased"]
            orig = (d.get("original_letter") or {}).get("headline") or [{}]
            print(f"{entry.key} {task}/{a.subset} n={m['n']} {_headline(m)} original={orig[0].get('value')} "
                  f"random={d['random_level']}")
            continue
        for variant in posthoc.forced_variants(task):
            m = run_forced_task(ad, spec, sel, prompts, posthoc, paths, a.out, variant=variant, resume=a.resume, argv=argv)
            print(f"{entry.key} {task}/{forced_subset_name(variant)} n={m['n']} parse_fail={m['parse']['failure_rate']} "
                  f"trunc={m['truncation']['rate']} valid={m['valid']['rate']} {_headline(m)}")
    return 0


def cmd_acceptance(a, argv) -> int:
    """批处理验收 v3（规则见 configs/e1_prompts.json 的 acceptance 与 src/e1/acceptance 的 v3 一节）。"""
    prompts = load_prompt_config(a.prompts)
    bs_a, bs_b = resolve_batch_sizes(a, prompts)
    protocol, bp = prompts.acceptance(), prompts.batch_protocol()
    paths = data_paths(a)
    entry, policy, ad = build_adapter(a)
    base = Path(a.out) / entry.key / ("acceptance" if a.policy == "default" else f"acceptance__{policy_dirname(a.policy)}")
    sels = {}
    for task in a.task:
        sel = selection(task, "dev", paths, a.limit, a.count_questions)
        if not sel.has_gt:
            raise RuntimeError(f"{task} 的开发核对没有 GT，验收 v3 要按判定结果比："
                               f"{sel.stats.get('note', '')}")
        sel.samples, st = A.take_samples(task, sel.samples, protocol)
        sel.stats.update(acceptance_take=st, n=len(sel.samples))
        sels[task] = sel
    passes = dict(zip(A.PASSES_V3, (bs_a, bs_a, bs_b)))
    recs: dict[str, dict[str, list]] = {p: {} for p in passes}
    task_rows = []
    for task in a.task:
        files = {}
        for p, bs in passes.items():
            spec = make_spec(a, entry, policy, ad, task, "dev", bs)
            run_task(ad, spec, sels[task], prompts, paths, base / p, resume=a.resume, argv=argv)
            files[p] = leaf_dir(base / p, entry.key, task, "dev", a.policy) / "predictions.jsonl"
            recs[p][task] = read_jsonl_tolerant(files[p], repair=False)
        sel = sels[task]
        rerun = A.rerun_identical(task, recs["A1"][task], recs["A2"][task], sha_a1=A.file_sha256(files["A1"]),
                                  sha_a2=A.file_sha256(files["A2"]))
        ref = A.reference_columns(task, sel.samples, recs["A1"][task], recs["B"][task], has_gt=sel.has_gt, batch_size_b=bs_b)
        task_rows.append({"task": task, "group": A.group_of(task, protocol)["name"], "n": len(sel.samples),
                          "take": sel.stats["acceptance_take"], "rerun": rerun, "reference": ref})
        print(f"[acceptance {entry.key} {task}] n={len(sel.samples)} rerun_identical={rerun['identical']} "
              f"(diff {rerun['n_diff']}) A1-B diff={ref['diff']} agree={ref['agreement_rate']}")
    group_rows = []
    for g in protocol["groups"]:
        tasks = [t for t in g["tasks"] if t in sels]
        if not tasks:
            continue
        row = A.noise_floor(g, A.group_delta(g),
                            [(t, sels[t].samples, recs["A1"][t]) for t in tasks],
                            [(t, sels[t].samples, recs["B"][t]) for t in tasks])
        row["tasks_missing"] = [t for t in g["tasks"] if t not in sels]
        group_rows.append(row)
        print(f"[acceptance {entry.key} group {g['name']}] n={row['n']} diff={row['diff']} ci={row['diff_ci']} "
              f"within±{row['delta']}={row['within_delta']}")
    overall = A.overall_v3(task_rows, group_rows)
    header = {"model": entry.key, "revision": entry.revision, "policy": a.policy, "device": a.device, "dtype": a.dtype,
              "tiny": a.tiny, "prompt_config_sha256": prompts.sha256, "batch_size_a": bs_a, "batch_size_b": bs_b,
              "frozen": bs_a == bp["batch_size"] and bs_b == protocol["composition_b"]["batch_size"],
              "limit": a.limit, "count_questions": a.count_questions,
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() and a.device.startswith("cuda") else None}
    doc = {"version": protocol["version"], "header": header, "protocol": protocol, "batch_protocol": bp, **overall,
           "groups": group_rows, "tasks": task_rows, "argv": list(argv),
           "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    base.mkdir(parents=True, exist_ok=True)
    (base / "acceptance.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (base / "acceptance.md").write_text(A.summary_markdown_v3(entry.key, task_rows, group_rows, header=header,
                                                              protocol=protocol, batch_protocol=bp), encoding="utf-8")
    print(f"all_pass={overall['all_pass']} rerun_all_identical={overall['rerun_all_identical']} "
          f"groups_failed={overall['groups_failed']} not_evaluated={overall['groups_not_evaluated']} -> {base / 'acceptance.json'}")
    return 0


def cmd_estimate(a, argv) -> int:
    prompts = load_prompt_config(a.prompts)
    paths = data_paths(a)
    entry, policy, ad = build_adapter(a)
    base = Path(a.out) / entry.key / ("estimate" if a.policy == "default" else f"estimate__{policy_dirname(a.policy)}")
    rows = []
    for task in a.task:
        rows.append(EST.estimate_task(ad, task, entry.family, prompts, paths, batch_sizes=a.batch_sizes,
                                      n_dev=a.estimate_n, think=a.estimate_think, n_think=a.estimate_think_n,
                                      policy_tag=a.policy, count_questions=a.count_questions,
                                      log=lambda m: print(m, file=sys.stderr, flush=True)))
    header = {"model": entry.key, "revision": entry.revision, "policy": a.policy, "count_questions": a.count_questions,
              "image_policy": {"type": type(policy).__name__, **policy.as_dict()}, "device": a.device, "dtype": a.dtype,
              "tiny": a.tiny, "prompt_config_sha256": prompts.sha256, "batch_sizes": a.batch_sizes,
              "frozen_batch_size": prompts.batch_protocol()["batch_size"],
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() and a.device.startswith("cuda") else None}
    totals = {}
    for k in EST.SUBSET_TOTALS:  # 没有任何一项可外推时记 null，不记 0
        vals = [r["gpu_hours"][k] for r in rows if r["gpu_hours"].get(k) is not None]
        totals[k] = sum(vals) if vals else None
    doc = {"header": header, "mem_headroom": EST.MEM_HEADROOM, "tasks": rows, "gpu_hours_total": totals,
           "argv": list(argv), "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    base.mkdir(parents=True, exist_ok=True)
    (base / "estimate.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (base / "estimate.md").write_text(EST.summary_markdown(entry.key, a.policy, rows, header), encoding="utf-8")
    print(f"gpu_hours_total={totals} -> {base / 'estimate.json'}")
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    a = parse_args(argv)
    from src.e3.merge_check import MergeCheckFailed

    try:
        if a.acceptance:
            return cmd_acceptance(a, argv)
        if a.estimate:
            return cmd_estimate(a, argv)
        if a.subset in POSTHOC_SUBSETS:
            return cmd_posthoc(a, argv)
        return cmd_run(a, argv)
    except MergeCheckFailed as e:  # --checkpoint：评测前的合并核对没有通过（DEV-0030 §2）
        print(f"拒绝评测：{e}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001 — 证据先落盘，再以非零码退出
        err_dir = Path(a.out) / a.model / "errors"
        err_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = err_dir / f"error_{stamp}.json"
        path.write_text(json.dumps({"argv": argv, "error": repr(e), "traceback": traceback.format_exc()[-6000:]},
                                   indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"失败：{e!r}（已写出 {path}）", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
