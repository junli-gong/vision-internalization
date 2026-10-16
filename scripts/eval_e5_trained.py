#!/usr/bin/env python
"""E5 训练后系统的评测入口（DEV-0037 §4；简报 §8.8）：一个 E5 检查点（起点 + E5 LoRA）→ E5 的 test 划分上单轮解码，用零样本的
同一套评分与判据（D59-4、D72-1）。规则见 src/e5/trained_eval.py。

解码（DEV-0043；E5 预注册 §8a 第 5 项，D83-3）：默认取 configs/e5_train.json 的 decode 一节，v3 的正式值是结构强制（structured：每个格子
位置只允许本任务的值 token，行末只允许行分隔，最后只允许停止 token）；结果写进 <任务>[__dev]__structured/，运行身份与 metrics 记 decode。
--decode restricted 作诊断（单轮受限解码，v2 的口径）：结果写进 <任务>[__dev]/，输出与改动前逐字节相同。E5 LoRA 合并前后的核对与解码
方式无关，两种解码共用 <…>/seed<k>/merge_check.json。
v3 主比较的起点（D83-5）：base_warmup、e3b_c3、c3_warmup。e3b、e3a、base 的起点（以及从它们训练的 E5 检查点）照常能评，不属于 v3 主比较，
运行时在 stderr 提示。

服务器（B200；可比的评测一律在 B200 上做）：
    python scripts/eval_e5_trained.py --checkpoint <E5 训练叶目录>/ckpt/final --model qwen3.5-2b \\
        --out experiments/E5/0046_e5_main_eval [--task all] [--subset eval|dev] [--resume] [--limit N]
    # 结果写进 <out>/trained/<起点>/<模型>/<n 标记>/seed<k>/<任务>[__dev]/；评测前的合并核对写进 <…>/seed<k>/merge_check.json
    # 起点检查点按 <ckpt>/final/e5_stack.json 记下的路径读；搬过家时用 --start-checkpoint 指新路径（manifest 的 sha256 必须相同）
    # 结构强制（默认）的结果在 <任务>[__dev]__structured/；--decode restricted 的在 <任务>[__dev]/
开发机彩排（小型随机模型、CPU fp32；d 的数据只在规范平台上生成，非规范平台上只允许 --tiny）：
    python scripts/eval_e5_trained.py --tiny --checkpoint <dir>/train/base/qwen3.5-0.8b/n10/seed0/ckpt/final --model qwen3.5-0.8b \\
        --out <dir> --limit 2

顺序：读 e5_stack.json 与 E5 检查点 manifest（构造方式必须与 --model、--tiny 相符）→ 建 base、合并起点 LoRA、核对起点（manifest 的 sha256、
权重指纹）→ load_trained(E5 检查点, merge=False) → 评测前的合并核对（不通过就不评测）→ 逐任务评测。
评测前的合并核对（DEV-0046；规则见 src/e5/trained_eval.py 的模块说明）：判据 e5-v2（D96-1 = N34 的 (b′)）——判允许集合上的合并前后一致
（restricted.across_merge）、受限教师强制与所有比较里分歧处的 gap；merge 的全词表教师强制一致率、decode_vs_generate_batch 的 exact_rows 等
只报告；v1 的判定照算、只报告（merge_check.json 的 v1_report_only）。同一次核对里另做指标层面的合并前后比较（metric_level，D96-2，只报告、
不影响判定）：同一批 val 样本，E5 LoRA 未合并与合并后各跑一次结构强制解码，E5 的正式评分逐行比较；总体差与两个标记在 stderr 打一行（出错时
打一行警告）。已有的 merge_check.json 判据版本不同（v1）时先改名留存为 merge_check.v1.json（重名加 .1、.2…），再写新的。

只做核对（DEV-0046；--check-only，只用于 --checkpoint）：
    python scripts/eval_e5_trained.py --check-only --checkpoint <E5 训练叶目录>/ckpt/final --model qwen3.5-2b --out experiments/E5/<NNNN_slug>
    # 加载 → 合并核对（e5-v2 + 指标层面）→ 写 <…>/seed<k>/merge_check.json → stdout 打一行摘要；不评任何任务、不建任何任务目录，已有的
    # 任务输出一字节都不动。--task、--subset、--limit、--resume、--decode 等评测参数不起作用。退出码：通过 0，不过 1（与 --task 无关），出错 2。
    # 用途：判据 v1 下已经评过 test 的运行补出 e5-v2 与指标层面的读数，不重评 test。
小档（DEV-0038）：E5 检查点的 final/ 就是训练时 val 上选中的那一步；e5_stack.json 的 selection（规则、选中的步、分数）照记进
config.yaml 的 checkpoint.selection。test 只在这个检查点上评一次。
每档条数默认取配置的 eval.n_per_bin（test：每档 100 条，b 200 条，与零样本相同）；--n-per-bin 统一覆盖。批大小默认取配置的 eval.batch_size（64）；
真模型给别的值要加 --allow-unfrozen-batch。

只评起点（DEV-0040，D81-1；N = 0：起点检查点不经 E5 训练，直接评；与 --checkpoint 二选一；规则见 src/e5/trained_eval.py 的模块说明）：
    python scripts/eval_e5_trained.py --start-only --init {base_warmup,e3b_c3,c3_warmup} --start-checkpoint <起点训练叶目录>/ckpt/final/model \\
        --model qwen3.5-2b --seed <k> --data-root $PROJ/data --out experiments/E5/<NNNN_slug> [--task all] [--subset eval|dev] [--resume] [--limit N]
    # 结果写进 <out>/trained/<起点>/<模型>/n0/seed<k>/<任务>[__dev]__structured/（--decode restricted：<任务>[__dev]/）；起点检查点的合并
    # 核对与权重指纹写进 <out>/train/<起点>/<模型>/start_checks/（与 scripts/train_e5.py 开训前的核对同一个位置，已通过的复用）
    # --init 只收网格 token 行训练过的起点：v3 的 base_warmup、e3b_c3、c3_warmup（DEV-0043），另保留 e3b（不属于 v3 主比较，消融的 N = 0
    # 读数）；e3a 的网格 token 行没训练过，base 没有起点检查点。--seed 必须等于起点训练叶目录 config.yaml 的 seed
    #（同 train_e5 的配种子；--allow-seed-mismatch 放开，记进运行身份）。B200 的训练队列把最终检查点按叶目录结构搬到 $CK_FINAL/<叶>/ckpt/final，
    # 旁边放 config.yaml，给 <新位置>/ckpt/final/model 即可
    # --data-root：起点检查点的合并核对用 E1 的 grid16 开发子集（默认环境变量 VISINT_DATA_ROOT；--tiny 时取 fixture）
顺序：resolve_start 与核对（构造方式、manifest、网格 token、变体）→ 配种子 → 起点检查点的合并核对 → 权重指纹的参照 → 建 base、合并起点 LoRA、
恢复起点的网格 token 行（src/e5/stack.load_start_only）→ 权重指纹必须等于参照 → 逐任务评测（与 E5 检查点的评测同一个 run_task）。
没有 E5 LoRA，"E5 LoRA 合并前后"的核对跳过（写进运行身份与 README）。

退出码：0 = 跑完（--check-only：核对通过）；1 = 合并核对没有通过，拒绝评测（--check-only：核对没有通过）；2 = 出错（报错先写进
<out>/trained/errors/；argparse 用法错误也是 2）。
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
from src.constants import BOOTSTRAP_N_RESAMPLES  # noqa: E402
from src.e3.merge_check import MergeCheckFailed  # noqa: E402
from src.e3.tokens import attach_vocab  # noqa: E402
from src.e5 import stack as STK  # noqa: E402
from src.e5 import trained_eval as TE  # noqa: E402
from src.e5.protocol import E5_TASKS, SUBSETS  # noqa: E402
from src.e5.token_grid import DECODE_MODES, STRUCTURED, E5Tokens  # noqa: E402
from src.e5.train_config import CONFIG_PATH, STARTS, load_train_config  # noqa: E402
from src.runlog import sha256_file  # noqa: E402
from src.synth.seeding import canonical_task  # noqa: E402

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
RUN_DIR_NAME = re.compile(r"[0-9]{4}_[A-Za-z0-9][A-Za-z0-9_.-]*")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", help="E5 检查点（<训练叶目录>/ckpt/final：含 model/ 与 e5_stack.json）；与 --start-only 二选一")
    p.add_argument("--start-only", action="store_true",
                   help="只评起点（N = 0，DEV-0040）：--start-checkpoint 给的起点检查点不经 E5 训练，直接评；与 --checkpoint 二选一")
    p.add_argument("--init", choices=STARTS,
                   help=f"--start-only 的起点，只收 {TE.START_ONLY_KINDS}（v3 主比较是 base_warmup、e3b_c3、c3_warmup；e3a 的网格 token "
                        "行没训练过，base 没有起点检查点）")
    p.add_argument("--seed", type=int, help="--start-only：种子（在配置的 training.seeds 里，且等于起点训练叶目录 config.yaml 的 seed）")
    p.add_argument("--allow-seed-mismatch", action="store_true",
                   help="--start-only：起点检查点的种子与 --seed 不同（或核对不了）时也评（配种子放开，记进运行身份）")
    p.add_argument("--data-root", help="--start-only：E1 的落地根（起点检查点的合并核对用；默认环境变量 VISINT_DATA_ROOT；--tiny 时取 fixture）")
    p.add_argument("--model", required=True, help="注册键（必须与检查点 manifest 的构造方式相同）")
    p.add_argument("--out", required=True, help="本次运行的目录（experiments/E5/<NNNN_slug>）")
    p.add_argument("--task", nargs="+", default=["all"], help=f"任务名、字母（a–e）或 all：{list(E5_TASKS)}")
    p.add_argument("--subset", choices=sorted(SUBSETS), help="eval = test 划分（默认取配置的 eval.subset）；dev = val 划分")
    p.add_argument("--n-per-bin", type=int, help="每档条数（统一覆盖配置的 eval.n_per_bin）")
    p.add_argument("--limit", type=int, help="只取每个任务评测清单的前 N 条（彩排、调试）")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--start-checkpoint", help="--checkpoint：起点检查点的新路径（默认取 e5_stack.json 记下的；manifest 的 sha256 必须相同）；"
                                              "--start-only：要评的起点检查点（<起点训练叶目录>/ckpt/final/model，必须给）")
    p.add_argument("--batch-size", type=int, help="批大小（默认配置的 eval.batch_size；--tiny 取 eval.tiny.batch_size）")
    p.add_argument("--allow-unfrozen-batch", action="store_true", help="真模型允许配置以外的批大小（运行身份里照记）")
    p.add_argument("--merge-check-n-per-task", type=int, help="合并核对每类的 val 样本数（默认配置的 eval.merge_check.n_per_task）")
    p.add_argument("--n-resamples", type=int, help="bootstrap 次数（默认 src.constants.BOOTSTRAP_N_RESAMPLES；--tiny 取配置的 eval.tiny）")
    p.add_argument("--decode", choices=DECODE_MODES,
                   help="解码方式（默认取配置的 decode.mode：v3 是 structured，D83-3）；restricted 作诊断（与改动前逐字节相同）")
    p.add_argument("--check-only", action="store_true",
                   help="只做评测前的合并核对（判据 e5-v2 与指标层面的比较，DEV-0046）：写 merge_check.json、打印摘要，不评任何任务、不建任务"
                        "目录；通过退出 0、不过退出 1（与 --task 无关）。只用于 --checkpoint")
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
    if a.start_only == (a.checkpoint is not None):
        p.error("--checkpoint（E5 检查点的评测）与 --start-only（只评起点，N = 0）二选一，必须且只能给一个")
    if a.start_only:  # DEV-0040
        if a.init is None:
            p.error(f"--start-only 要给 --init（{' / '.join(TE.START_ONLY_KINDS)}）")
        if a.init not in TE.START_ONLY_KINDS:
            why = {"e3a": "e3a 的网格 token 行是 E5 开训时新加的、没训练过", "base": "base 没有起点检查点"}.get(a.init, "")
            p.error(f"--start-only 只收 --init {' / '.join(TE.START_ONLY_KINDS)}（D81-1），收到 {a.init!r}" + (f"：{why}" if why else ""))
        if a.start_checkpoint is None or not Path(a.start_checkpoint).is_dir():
            p.error(f"--start-only 要给起点检查点目录 --start-checkpoint（<起点训练叶目录>/ckpt/final/model），收到 {a.start_checkpoint!r}")
        if a.seed is None:
            p.error("--start-only 要给 --seed（起点训练叶目录 config.yaml 的 seed）")
        if a.merge_check_n_per_task is not None:
            p.error("--merge-check-n-per-task 只用于 --checkpoint：只评起点没有 E5 LoRA，不做 E5 LoRA 合并前后的核对")
        if a.check_only:
            p.error("--check-only 只用于 --checkpoint：只评起点没有 E5 LoRA，不做 E5 LoRA 合并前后的核对")
        a.data_root = a.data_root or os.environ.get("VISINT_DATA_ROOT")
        if not a.tiny and not a.data_root:
            p.error("--start-only 要给 --data-root（或设 VISINT_DATA_ROOT）：起点检查点的合并核对用 E1 的 grid16 开发子集")
    else:
        for flag, given in (("--init", a.init is not None), ("--seed", a.seed is not None), ("--data-root", a.data_root is not None),
                            ("--allow-seed-mismatch", a.allow_seed_mismatch)):
            if given:
                p.error(f"{flag} 只用于 --start-only（E5 检查点的起点、种子取 e5_stack.json）")
        if not Path(a.checkpoint).is_dir():
            p.error(f"--checkpoint {a.checkpoint!r} 不是目录")
    if not a.tiny and not RUN_DIR_NAME.fullmatch(Path(a.out).name):
        p.error(f"--out 的目录名必须带运行编号，形如 0046_e5_main_eval（CLAUDE.md：experiments/<phase>/<NNNN_slug>/），收到 {a.out!r}")
    return a


def eval_settings(a, cfg):
    """配置的 eval 节、--tiny 的覆盖、批大小、子集、bootstrap 次数（E5 检查点的评测与只评起点同一套取法）。"""
    e = cfg.doc["eval"]
    t = e["tiny"] if a.tiny else {}
    bs = int(a.batch_size or t.get("batch_size", e["batch_size"]))
    if not a.tiny and bs != int(e["batch_size"]) and not a.allow_unfrozen_batch:
        raise ValueError(f"--batch-size {bs} 不是配置的 eval.batch_size {e['batch_size']}；确要换就加 --allow-unfrozen-batch")
    subset = a.subset or e["subset"]
    n_resamples = a.n_resamples if a.n_resamples is not None else int(t.get("n_resamples", BOOTSTRAP_N_RESAMPLES))
    return e, t, bs, subset, n_resamples


def decode_prefix(prefix: str, decode: str) -> str:
    """打印行的前缀：结构强制时加 decode=structured；restricted 与改动前相同。"""
    return prefix + (f" decode={decode}" if decode == STRUCTURED else "")


def v3_note(cfg, kind: str) -> None:
    """起点不属于 E5 v3 主比较时在 stderr 提示（D83-5；配置的 starts.<起点>.in_v3）。"""
    if not cfg.in_v3(kind):
        print(f"注意：起点 {kind} 不属于 E5 v3 主比较（D83-5；E5 预注册 §8a：v3 的起点是 "
              f"{'、'.join(k for k in STARTS if cfg.in_v3(k))}）", file=sys.stderr)


def report(prefix: str, task: str, subset: str, m) -> None:
    """一个任务跑完后打印的一行（标题读数、解析失败率、imagining）。"""
    h = m["headline"]
    parts = [f"iou_grid={h['mask_iou_grid']['strict']}"]
    if "relief_error" in h:
        parts.append(f"relief={h['relief_error']['strict']}")
    print(f"{prefix} {task}/{subset} n={m['n']} {' '.join(parts)} "
          f"parse_fail={m['parse']['failure_rate']} imagining={(m.get('gate') or {}).get('imagining')}")


def check_line(prefix: str, summ) -> str:
    """--check-only 打印的一行（stdout）：判据、是否通过、v1 的判定（只报告）、是否复用、指标层面的读数（只报告）、核对文件。"""
    ml = summ.get("metric_level") or {}
    fl = ml.get("flags") or {}

    def f(x) -> str:
        return "None" if x is None else f"{x:+.4f}"

    prev = summ.get("previous") or {}
    return (f"check-only {prefix} criteria={summ.get('criteria_version')} pass={summ.get('pass')} v1_pass={summ.get('v1_pass')} "
            f"reused={summ.get('reused')} metric_level: iou_grid_diff={f(ml.get('iou_grid_diff'))} "
            f"relief_error_diff={f(ml.get('relief_error_diff'))} identical={ml.get('n_identical')}/{ml.get('n')} "
            f"flags.iou={fl.get('iou')} flags.relief={fl.get('relief')} error={ml.get('error') is not None} "
            f"previous={prev.get('renamed_to')} file={summ.get('file')}")


def cmd_run(a, argv) -> int:
    cfg = load_train_config(a.config)
    e, t, bs, subset, n_resamples = eval_settings(a, cfg)
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
    decode = cfg.decode_mode(a.decode)
    v3_note(cfg, start_kind)
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
    prefix = f"trained {start_kind} {a.model} {tag} seed{seed}"
    try:
        mc = TE.stacked_merge_check(ad, e5tok, prompt, n=n, n_per_task=mc_n, tasks=E5_TASKS, batch_size=mc_bs, tiny=a.tiny,
                                    exact=exact, key=key, out_json=mdir / TE.MERGE_CHECK)
    except MergeCheckFailed as err:
        print(TE.metric_level_line(err.summary.get("metric_level")), file=sys.stderr)
        if a.check_only:  # DEV-0046：只做核对
            print(check_line(prefix, err.summary))
            print(f"合并核对没有通过（--check-only）：{err}", file=sys.stderr)
            return 1
        print(f"拒绝评测：{err}", file=sys.stderr)
        return 1
    print(TE.metric_level_line(mc.get("metric_level")), file=sys.stderr)  # DEV-0046：指标层面的比较（只报告）
    if a.check_only:  # DEV-0046：只做核对，不评任何任务、不建任务目录
        print(check_line(prefix, mc))
        return 0
    checkpoint = {"manifest_sha256": e5_sha, "train_identity_sha256": stack["identity_sha256"],
                  "start": {"kind": start_kind, "manifest_sha256": stack["start"]["manifest_sha256"]},
                  "lora": man.get("lora"), "build_spec": man.get("build_spec"), "data": stack["data"], "seed": seed}
    if stack.get("selection"):  # DEV-0038：val 选中的那一步（小档）；test 只在它上面评这一次。没有选择的检查点，运行身份与原来相同
        sel = stack["selection"]
        checkpoint["selection"] = {k: sel[k] for k in ("rule", "split", "selected_step", "selected_score", "steps", "val_every",
                                                       "n_per_task")}
        if sel.get("decode") is not None:  # DEV-0043：val 选择用结构强制时记下（restricted 选的检查点与原来相同）
            checkpoint["selection"]["decode"] = sel["decode"]
    extra = {"checkpoint_dir": str(fd), "start_checkpoint": None if st.start_checkpoint is None else str(st.start_checkpoint),
             "merge_check": mc, "fingerprint": st.fingerprint,
             "gpu": torch.cuda.get_device_name(0) if a.device.startswith("cuda") and torch.cuda.is_available() else None}
    for task in a.task:
        npb = int(a.n_per_bin or e["n_per_bin"][task])
        spec = TE.EvalSpec(start=start_kind, model_key=a.model, tag=tag, seed=seed, task=task, subset=subset, n_per_bin=npb,
                           n_resamples=n_resamples, batch_size=bs, tiny=a.tiny, dtype=a.dtype, limit=a.limit,
                           data_seed=data_seed, decode=decode)
        m = TE.run_task(ad, e5tok, prompt, cfg, spec, a.out, checkpoint=checkpoint, n=n, resume=a.resume, argv=argv,
                        run_extra=extra)
        report(decode_prefix(prefix, decode), task, subset, m)
    return 0


def cmd_start_only(a, argv) -> int:
    """只评起点（DEV-0040；模块说明）。批大小、子集、bootstrap 次数、每档条数的取法与 cmd_run 相同。"""
    log = lambda m: print(m, file=sys.stderr, flush=True)  # noqa: E731
    cfg = load_train_config(a.config)
    e, _, bs, subset, n_resamples = eval_settings(a, cfg)
    seed = int(a.seed)
    decode = cfg.decode_mode(a.decode)
    v3_note(cfg, a.init)
    if seed not in cfg.seeds:  # 同 scripts/train_e5.py：n0 与各数据量档同一组种子
        raise ValueError(f"--seed {seed} 不在配置的 training.seeds {list(cfg.seeds)} 里")
    start = STK.resolve_start(cfg, a.init, Path(a.start_checkpoint).resolve(), model_key=a.model, tiny=a.tiny)
    pairing = STK.check_seed_pairing(start, seed, allow_mismatch=a.allow_seed_mismatch)
    if pairing["status"] != "match":
        log(f"注意：配种子放开（{pairing['status']}：起点的种子 {pairing['start_seed']}，--seed {seed}），已记进运行身份")
    # 评测前的核对（同 src/e5/train.run_e5_training 开训前的两项；结果放在训练的同一个位置，已通过的复用）
    checks_dir = STK.start_checks_dir(a.out, start.kind, a.model)
    checks_dir.mkdir(parents=True, exist_ok=True)
    try:
        start_check = STK.start_merge_check(start, cfg, model_key=a.model, tiny=a.tiny, device=a.device, dtype=a.dtype,
                                            out_dir=checks_dir, data_root=a.data_root, log=log)
    except MergeCheckFailed as err:
        print(f"拒绝评测：起点检查点的合并核对没有通过：{err}", file=sys.stderr)
        return 1
    reference = STK.reference_fingerprint(start, device=a.device, dtype=DTYPES[a.dtype], out_dir=checks_dir, log=log)
    ad, tokens, info = STK.load_start_only(start, cfg, device=a.device, dtype=DTYPES[a.dtype])
    fp = info["fingerprint"]
    if fp != reference["fingerprint"]:
        raise STK.StartError(f"合并起点 LoRA 之后的权重指纹 {fp[:16]}… 与 factory.load_trained(起点, merge=True) 的 "
                             f"{reference['fingerprint'][:16]}… 不同：评测的模型与起点检查点合并评测时的不一致")
    e5tok = E5Tokens(tokens, cfg.e3b.allowed_scope)
    prompt = cfg.prompt_format()
    n = cfg.grid_n
    log(f"[只评起点] 跳过的核对：{TE.START_ONLY_SKIPPED}")
    checkpoint = TE.start_only_checkpoint(start, seed=seed, seed_pairing=pairing, fingerprint=fp, device_key=STK.device_key(ad),
                                          start_check=start_check)
    extra = {"mode": "start_only", "start_checkpoint": str(start.checkpoint), "start_leaf_config": start.leaf_config,
             "start_merge_check": start_check,
             "reference_fingerprint": {k: reference[k] for k in ("fingerprint", "device_key", "manifest_sha256")},
             "setup": info, "skipped_checks": dict(TE.START_ONLY_SKIPPED),
             "gpu": torch.cuda.get_device_name(0) if a.device.startswith("cuda") and torch.cuda.is_available() else None}
    for task in a.task:
        npb = int(a.n_per_bin or e["n_per_bin"][task])
        spec = TE.EvalSpec(start=start.kind, model_key=a.model, tag=TE.START_ONLY_TAG, seed=seed, task=task, subset=subset,
                           n_per_bin=npb, n_resamples=n_resamples, batch_size=bs, tiny=a.tiny, dtype=a.dtype, limit=a.limit,
                           start_only=True, decode=decode)
        m = TE.run_task(ad, e5tok, prompt, cfg, spec, a.out, checkpoint=checkpoint, n=n, resume=a.resume, argv=argv,
                        run_extra=extra)
        report(decode_prefix(f"start-only {start.kind} {a.model} {TE.START_ONLY_TAG} seed{seed}", decode), task, subset, m)
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    a = parse_args(argv)
    try:
        return cmd_start_only(a, argv) if a.start_only else cmd_run(a, argv)
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
