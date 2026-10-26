#!/usr/bin/env python
"""W2-c 的训练入口（DEV-0061；D113 第二节第 4 条；docs/w2_plan.md §5）：一个模型 × 一个种子，只有文本的 LoRA 监督
（提出 = E1 refexp_box 提示 + 原图 → GT 框；挑选 = 挑选图 + 挑选提示 → 编号）。规则见 src/w2/w2c_train.py，配置 configs/w2c_train.json。

服务器（--out 是本次运行的目录 experiments/W2/<NNNN_slug>；结果写进 <out>/w2c/<模型>/seed<k>/，检查点 <叶目录>/ckpt/final/model）：
    python scripts/train_w2c.py --model qwen3.5-9b --w2c-data <DEV-0060 的输出>/w2c_data --data-root $DATA \\
        --out experiments/W2/00NN_w2c_train [--seed 0] [--steps N] [--resume] [--stop-after-steps N] [--ckpt-root DIR]
    # --w2c-data：DEV-0060 的 <out>/w2c_data（*.jsonl 分片 + manifest.json）；--data-root：数据行里 image 的相对根（DEV-0060 用的同一个）
    # 前提：configs/w2.json 已冻结（frozen = true），configs/w2c_train.json 的 w2_config.sha256 已由管理员填成它的 sha256（占位 null 时拒跑）
    # 用时（按 9B 的 C3 实测：每步约 7.3 s、显存峰值约 28.6 GB；W2-c 每步 16 条、同样的微批 4 × 累积 4，提出与 C3 的框样本同形（一张图、
    # 约 50 个目标 token），挑选是一张 grid 拼图（约 300 个视觉 token）+ 约 70 个 token 的提示 + 2 个目标 token；C3 里还有多边形（目标至多
    # 约 400 个 token）与点对样本，所以每步与显存峰值应不超过 C3）：2000 步约 3.5–4 小时，显存峰值 ≤ 约 29 GB；layout = separate 时挑选
    # 样本有 K + 1 张图（约 2,700 个视觉 token），每步可能约 2 倍。检查点每 250 步一次（约 30 分钟）。可以分段：--stop-after-steps N 写
    # 检查点退出，下次同一条命令加 --resume 接着跑（CPU 上与一次跑完逐位一致）。

训练之后（同一台机器上，与基座配对；--checkpoint 都是 <叶目录>/ckpt/final/model；每条都先跑评测前的合并核对，口径同 E3 的 T 文本评测：
src/e3/merge_check 的 t_text，n = configs/e3b_train.json 的 eval.merge_check.n = 200，不通过就拒绝评测、退出码 1）：
  1. 判定（W2-c 的主配置，最晚 11-01；提出与挑选都用训练过的模型、都关思考；候选、渲染、K 用 W2-a 冻结的那一份）：
     python scripts/run_w2.py --model qwen3.5-9b --task refseg --split devcheck2 --setting w2c --checkpoint $CK \\
         --data-root $DATA --out experiments/W2/00NN_w2c_eval --candidate-workers 8
     # 写 <out>/qwen3.5-9b/refexp_box/propose_w2c/devcheck2（训练过的模型按 E1 协议重新出框，与 base 的 baseline_e1 分开）、
     # <out>/qwen3.5-9b/refseg/candidates_w2c/devcheck2、<out>/qwen3.5-9b/refseg/w2c/devcheck2；合并核对在 <out>/qwen3.5-9b/w2c_merge_check.json
     # 用时（9B、A100）：合并核对约 10 分钟，出框 500 句约 3 分钟，候选 500 × 4.5 s / 8 进程约 5 分钟，挑选约 3 分钟
     python scripts/w2_gate.py --main-name W2-c --main experiments/W2/00NN_w2c_eval/qwen3.5-9b/refseg/w2c/devcheck2 \\
         --w1f <W2-a 判定那次运行>/qwen3.5-9b/refseg/w1f/devcheck2 --out-dir <目录>
     # w1f 是 base 9B 的 W1-F（同一个 devcheck2、同一份冻结的 configs/w2.json）；gate 核对 W2-c 必须带检查点
  2. 健康检查（简报 8.11；协议 v3；固定 200 题；scripts/health_check.py 的 --checkpoint 是 DEV-0061 加的最小改动，加载方式同 run_e1）：
     python scripts/health_check.py --model qwen3.5-9b --checkpoint $CK --data-root $DATA --out <目录>/health_w2c.json
     python scripts/health_check.py --model qwen3.5-9b --out <目录>/health_base.json        # 基座，同一台机器（跨机器不可比，D55、D93-1）
     python scripts/health_check.py --compare <目录>/health_base.json <目录>/health_w2c.json --out <目录>/health_compare.json
     # --data-root 只给合并核对读 E1 grid16 的开发子集（写进 <目录>/health_w2c_merge_check.json）；任何一项比基座低超过 5 个点标红
     # （--compare 给出）；每次约 10–15 分钟（RUN-0007 的基座 6–13 分钟）；要 NLTK_DATA 里的 punkt_tab（同 RUN-0017）。
     # 基座的 v3 参照（RUN-0017）是在已退役的 B200 上跑的，要在评 W2-c 的同一节点上重跑基座
  3. 定位与计数的 T 文本读数（口径同 E3 的 C3 对照表：scripts/run_e1.py --checkpoint 原样，E1 冻结的提示、口径、批配置，eval 子集；
     RUN-0056 的 eval.sh 里 ttext 的同一条命令，只是 --task 换成这两个）：
     python scripts/run_e1.py --model qwen3.5-9b --task refexp_box count --subset eval --data-root $DATA \\
         --checkpoint $CK --checkpoint-label t_text/w2c/seed0 --out experiments/W2/00NN_w2c_ttext
     python scripts/run_e1.py --model qwen3.5-9b --task refexp_box count --subset eval --data-root $DATA \\
         --out experiments/W2/00NN_w2c_ttext                                                  # 基座，同一台机器
     # 结果：<out>/t_text/w2c/seed0/qwen3.5-9b/{refexp_box,count}/metrics.json 对 <out>/qwen3.5-9b/{refexp_box,count}/metrics.json；
     # 合并核对在 <out>/t_text/w2c/seed0/qwen3.5-9b/merge_check.json。同一节点上已有的 9B 基座 T 文本（如 RUN-0056 在 hpcgpu06 上的
     # ttext base）可以直接配对，不必重跑基座；换了节点就在同一节点上重跑（D93-1：效应都在机内算）

W2-c 加回放（DEV-0065；D125-2 (b)；配置 configs/w2c_replay_train.json = W2-c 的配置 + replay 一节；规则见 src/w2/w2c_train.py 的
"回放"一段与 src/w2/w2c_replay.py；回放数据由 scripts/build_w2c_replay.py 生成，那里的文档字符串有全流程的命令与用时）：
    python scripts/train_w2c.py --config configs/w2c_replay_train.json --model qwen3.5-9b \
        --w2c-data experiments/W2/0067_w2c_data_hpcgpu06/w2c_data --replay-data experiments/W2/<NNNN>_w2c_replay_data \
        --data-root $VI/data/train_data --out experiments/W2/<NNNN>_w2c_replay_train [--resume] [--stop-after-steps N]
    # --replay-data：build_w2c_replay.py 的 --out（回放数据必须由同一个 9B 生成；截断的行不用）；--data-root 同 W2-c（计数回放的图
    #   也相对它）。叶目录、检查点、记账与 W2-c 相同（<out>/w2c/qwen3.5-9b/seed0/，config.yaml 的 variant 仍是 w2c、多 replay 一节），
    #   所以训练之后的 1–3 三条评测命令原样可用（--checkpoint 换成这次的；--out 换新目录，run_w2 拒绝同一个 --out 混用两个检查点）。
    # 回放：两条流（管理员裁定），每步 8 + 8 + 3 + 2 = 21 条——replay_general 3 条通用、replay_count 2 条计数，四条流各自按本流的监督
    #   token 数平均，权重 1 : 1 : 0.5 : 0.5（回放总权重 1/3，计数与通用各占一半）。
    # 用时：W2-c（RUN-0069，06）每步中位 10.9 s、显存峰值 26.5 GB；回放的 3 条通用是纯文本（提示约 100–1,000 个 token、回答至多 512），
    #   2 条计数是一张原图 + 约 2 个目标 token（与提出的样本同形），按样本的计算量估每步约 13–14 s，2000 步约 7–8 小时；回放每步两个
    #   微批（3 条、2 条），都比挑选的微批（4 × 约 2,800 个 token）短，显存峰值预计不变（≤ 约 29 GB）。
    # 没有 replay 一节的配置不收 --replay-data；有 replay 一节时必须给 --replay-data（或配置的 replay.data.path）。

探索 X-2（DEV-0069；配置 configs/x2_train.json = W2-c 加回放的配置 + x2_data、spline、correct 三节；规则见 src/w2/w2c_train.py 的
"X-2 的两条流"一段与 src/explore/x2_data.py；数据由 scripts/build_x2_data.py 在 CPU 上生成）：
    python scripts/train_w2c.py --config configs/x2_train.json --model qwen3.5-9b \
        --w2c-data experiments/W2/0067_w2c_data_hpcgpu06/w2c_data --x2-data experiments/X/<NNNN>_x2_data \
        --replay-data experiments/W2/<NNNN>_w2c_replay_data --data-root $VI/data/train_data \
        --out experiments/X/<NNNN>_x2_train [--steps 1500] [--resume] [--stop-after-steps N]
    # --x2-data：build_x2_data.py 的 --out（必须由同一份 --w2c-data 生成，参数等于配置的 spline、correct 两节）；--replay-data 同 W2-c 加回放
    #   （回放数据由同一个 9B 生成）。叶目录、检查点、记账与 W2-c 相同（<out>/w2c/qwen3.5-9b/seed0/，variant 仍是 w2c，多 x2、replay 两节），
    #   所以 run_w2.py --setting w2c、health_check.py、run_e1.py --checkpoint 原样可用；X-2 的管线用 scripts/run_x2.py。
    # 每步 8 + 8 + 6 + 6 + 3 + 2 = 33 条：提出、挑选、样条、修正、回放通用、回放计数，六条流各自按本流的监督 token 数平均，权重
    #   1 : 1 : 1 : 1 : 0.5 : 0.5；微批 4：样条、修正各两个微批（4 + 2）。
    # 用时（按 RUN-0069 的每步中位 10.9 s 折算；16 条 = 8 提出 + 8 挑选）：
    #   - 按条数等比折算：10.9 / 16 × 33 ≈ 22.5 s/步，2,000 步 ≈ 12.5 GPU 小时；
    #   - 按样本的计算量：挑选的样本是 K + 1 = 5 张图（约 1,500 个视觉 token），提出、样条、修正都是 1 张图（约 300 个视觉 token）+ 约 50–160
    #     个目标 token（样条 16 点约 160 个、修正约 50 个）；设挑选每条约是提出的 3 倍，10.9 s = 8p + 24p → p ≈ 0.34 s；样条、修正 12 条约
    #     4–5.5 s，回放约 2.5–3 s（W2-c 加回放的估计 13–14 s/步）→ 约 17.5–19.5 s/步，2,000 步 ≈ 9.7–10.8 GPU 小时。
    #   - 上界超过 12 GPU 小时，所以备选 1,500 步（--steps 1500；样条 9,000 条 ≈ 0.6 个 epoch、修正 9,000 条 ≈ 0.4 个 epoch）：约 7.3–9.4 GPU 小时。
    #   - 显存峰值：样条的微批（4 × 约 600 个 token）、修正的微批（4 × 约 500 个 token）都比挑选的微批（4 × 约 2,800 个 token）短，预计不变
    #     （≤ 约 29 GB）。前 50 步跑完先报每步的中位耗时与显存峰值，外推超过 12 小时就停下来定步数。
    # 没有 spline、correct 两节的配置不收 --x2-data；有这两节时必须给 --x2-data（或配置的 x2_data.path）。

开发机彩排（小型随机模型、CPU、fp32；数据用测试造的假数据，图像在 tests/fixtures/e3/root）：
    python -m tests._w2c_fakes <dir>/w2c_data            # 按当前的 configs/w2.json 造 §5 格式的假数据
    python scripts/train_w2c.py --tiny --model qwen3.5-0.8b --w2c-data <dir>/w2c_data --out <dir>/t
--tiny 时可以用 --w2-config 换一份 W2 配置、--candidates-module / --render-module 换候选类与渲染（只配 --tiny）。
退出码：0 = 跑完或按 --stop-after-steps 停下；2 = 出错（报错先写进 <out>/w2c/<模型>/errors/，再退出；argparse 用法错误也是 2）。
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
from src.e3.data import FIXTURE_REFSEG_ROOT  # noqa: E402
from src.explore import x2_data as XD  # noqa: E402
from src.w2 import runner as R  # noqa: E402
from src.w2 import w2c_replay as RP  # noqa: E402
from src.w2 import w2c_train as T  # noqa: E402

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}
RUN_DIR_NAME = re.compile(r"[0-9]{4}_[A-Za-z0-9][A-Za-z0-9_.-]*")
TINY_DATA_ROOT = FIXTURE_REFSEG_ROOT  # 假数据的图像（tests/_w2c_fakes 用 E3 的训练池 fixture）


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="注册键（默认配置的 model.default_key，即 qwen3.5-9b）")
    p.add_argument("--w2c-data", required=True, help="DEV-0060 的 w2c_data 目录（*.jsonl 分片 + manifest.json）")
    p.add_argument("--data-root", help="数据行里 image 的相对根（默认环境变量 VISINT_DATA_ROOT；--tiny 时取 tests/fixtures/e3/root）")
    p.add_argument("--out", required=True, help="本次运行的目录（experiments/W2/<NNNN_slug>）")
    p.add_argument("--seed", type=int, help="种子（默认配置的 seed = 0）")
    p.add_argument("--steps", type=int, help="步数（默认配置的 steps；给了就覆盖，运行身份照记）")
    p.add_argument("--resume", action="store_true", help="续跑：从 <ckpt>/latest 接着训练")
    p.add_argument("--ckpt-root", help="检查点的根目录（默认 <叶目录>/ckpt）")
    p.add_argument("--stop-after-steps", type=int, help="本次只跑 N 步，写检查点后退出（下次 --resume）")
    p.add_argument("--keep-optimizer-state", action="store_true", help="训练完成后保留 <ckpt>/latest（默认删掉）")
    p.add_argument("--tiny", action="store_true", help="小型随机模型 + CPU + fp32（开发机彩排）")
    p.add_argument("--config", default=str(T.W2C_CONFIG_PATH), help="默认 configs/w2c_train.json（加回放：configs/w2c_replay_train.json）")
    p.add_argument("--replay-data", help="DEV-0065：回放数据（scripts/build_w2c_replay.py 的 --out）；只配有 replay 一节的配置")
    p.add_argument("--x2-data", help="DEV-0069：X-2 的数据（scripts/build_x2_data.py 的 --out）；只配有 spline、correct 两节的配置")
    p.add_argument("--w2-config", help="换一份 W2 配置（只配 --tiny；正式运行用 configs/w2c_train.json 写死的那一份）")
    p.add_argument("--candidates-module", default=R.DEFAULT_CANDIDATES_MODULE, help="Candidate 类所在的模块（非默认只配 --tiny）")
    p.add_argument("--render-module", default=R.DEFAULT_RENDER_MODULE, help="render_selection 所在的模块（非默认只配 --tiny）")
    p.add_argument("--device", help="默认：真模型 cuda，--tiny 时 cpu")
    p.add_argument("--dtype", choices=sorted(DTYPES), help="默认：真模型取配置的 model.dtype（bf16），--tiny 时 fp32")
    a = p.parse_args(argv)
    if a.model is not None and a.model not in registry.MODELS:
        p.error(f"未登记的模型键 {a.model!r}；已登记：{sorted(registry.MODELS)}")
    for name in ("stop_after_steps", "steps"):
        v = getattr(a, name)
        if v is not None and v < 1:
            p.error(f"--{name.replace('_', '-')} 必须是正整数")
    if a.seed is not None and a.seed < 0:
        p.error("--seed 不能为负")
    if not a.tiny and (a.w2_config or a.candidates_module != R.DEFAULT_CANDIDATES_MODULE or a.render_module != R.DEFAULT_RENDER_MODULE):
        p.error("--w2-config、--candidates-module、--render-module（非默认）只配 --tiny")
    if not a.tiny and not (a.data_root or os.environ.get("VISINT_DATA_ROOT")):
        p.error("真模型运行要给 --data-root（或设 VISINT_DATA_ROOT）")
    if not a.tiny and not RUN_DIR_NAME.fullmatch(Path(a.out).name):
        p.error(f"--out 的目录名必须带运行编号，形如 0070_w2c_train（CLAUDE.md：experiments/<phase>/<NNNN_slug>/），收到 {a.out!r}")
    a.device = a.device or ("cpu" if a.tiny else "cuda")
    return a


def data_root(a) -> Path:
    if a.data_root:
        return Path(a.data_root)
    if a.tiny:
        return TINY_DATA_ROOT
    return Path(os.environ["VISINT_DATA_ROOT"])


def build_adapter(a, cfg: T.W2CConfig, key: str):
    entry = registry.get_model(key)
    dtype = a.dtype or ("fp32" if a.tiny else cfg.doc["model"]["dtype"])
    if a.tiny:
        t = cfg.doc["tiny"]
        ad = make_adapter(key, device=a.device, dtype=DTYPES[dtype], image_policy=parse_policy(t["image_policies"][entry.family]),
                          tiny=True, init_std=float(t["init_std"]))
    else:
        ad = make_adapter(key, device=a.device, dtype=DTYPES[dtype])
    return entry, ad, dtype


def load_replay(a, cfg: T.W2CConfig, rc, key: str, root: Path):
    """DEV-0065：配置有 replay 一节时读回放数据（--replay-data 优先，其次配置的 replay.data.path）；没有这一节时不收 --replay-data。"""
    if rc.get("replay") is None:
        if a.replay_data is not None:
            raise T.W2CConfigError(f"{cfg.path} 没有 replay 一节，不收 --replay-data")
        return None
    path = a.replay_data or cfg.doc["replay"]["data"]["path"]
    if path is None:
        raise T.W2CConfigError(f"{cfg.path} 的 replay.data.path 是占位（null）：要给 --replay-data")
    p = Path(path)
    if a.replay_data is None and not p.is_absolute():
        p = REPO_ROOT / p
    entry = registry.get_model(key)
    streams = {s: v["kinds"] for s, v in rc["replay"]["streams"].items()}
    return RP.load_replay_data(p, streams=streams, model_key=entry.key, revision=entry.revision, tiny=a.tiny, data_root=root)


def load_x2(a, cfg: T.W2CConfig, rc, root: Path):
    """DEV-0069：配置有 spline、correct 两节时读 X-2 的数据（--x2-data 优先，其次配置的 x2_data.path）；没有这两节时不收 --x2-data。
    核对：参数等于配置的；来源与 --w2c-data 是同一份 w2c_data（非 --tiny）；图像都在 --data-root 下。"""
    if rc.get("x2") is None:
        if a.x2_data is not None:
            raise T.W2CConfigError(f"{cfg.path} 没有 spline、correct 两节，不收 --x2-data")
        return None
    path = a.x2_data or cfg.doc["x2_data"]["path"]
    if path is None:
        raise T.W2CConfigError(f"{cfg.path} 的 x2_data.path 是占位（null）：要给 --x2-data")
    p = Path(path)
    if a.x2_data is None and not p.is_absolute():
        p = REPO_ROOT / p
    source = None if a.tiny else XD.source_identity(a.w2c_data, require_manifest=True)
    return XD.load_x2_data(p, params=rc["x2"]["data_params"], source=source, data_root=root)


def cmd_run(a, argv) -> int:
    cfg = T.load_w2c_config(a.config)
    key = a.model or cfg.doc["model"]["default_key"]
    seed = cfg.seed if a.seed is None else int(a.seed)
    w2, w2_info = T.resolve_w2_config(cfg, tiny=a.tiny, override=a.w2_config)
    impl = R.load_impl(a.candidates_module, a.render_module)
    root = data_root(a)
    data = T.load_w2c_data(a.w2c_data, w2=w2, sources=cfg.doc["data"]["sources"], data_root=root, require_manifest=not a.tiny)
    rc = T.resolve_w2c_run(cfg, tiny=a.tiny, steps=a.steps)
    replay = load_replay(a, cfg, rc, key, root)
    x2 = load_x2(a, cfg, rc, root)
    entry, ad, dtype = build_adapter(a, cfg, key)
    spec = T.W2CRunSpec(model_key=entry.key, family=entry.family, repo=entry.repo, revision=entry.revision,
                        adapter_model_key=str(getattr(ad, "model_key", entry.key)), tiny=a.tiny, image_policy=policy_spec(ad.image_policy),
                        model_dtype=dtype, seed=seed)
    s = T.run_w2c_training(ad, cfg, spec, rc, data, w2=w2, w2_info=w2_info, impl=impl, data_root=root, out_dir=a.out, resume=a.resume,
                           argv=argv, ckpt_root=a.ckpt_root, stop_after_steps=a.stop_after_steps,
                           keep_optimizer_state=a.keep_optimizer_state, **({} if replay is None else {"replay": replay}),
                           **({} if x2 is None else {"x2": x2}))
    if s is None:
        print(f"{entry.key} w2c seed{seed}：按 --stop-after-steps 停下，检查点已写；加 --resume 接着跑")
        return 0
    print(f"{entry.key} w2c seed{seed} steps={s['steps']} final_loss={s['final_loss']:.4f} by_stream={s['final_loss_by_stream']} "
          f"ckpt={s['final_checkpoint']} ({s['final_checkpoint_bytes']} B)")
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    a = parse_args(argv)
    try:
        return cmd_run(a, argv)
    except Exception as e:  # noqa: BLE001 — 证据先落盘，再以非零码退出
        err_dir = Path(a.out) / T.VARIANT / (a.model or "default") / "errors"
        err_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = err_dir / f"error_{stamp}.json"
        path.write_text(json.dumps({"argv": argv, "error": repr(e), "traceback": traceback.format_exc()[-6000:]},
                                   indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"失败：{e!r}（已写出 {path}）", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
