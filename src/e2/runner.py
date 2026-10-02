"""E2 的运行器：一个模型 × 一个任务 → <out>/<模型>/<任务>/（DEV-0023；简报 §8.3、§11.1）。

流程（TaskRun）：
1. 数据：训练池（必须有去重剔除清单）→ 留出验证子集与标准化样本（src/e2/data.split_pool）；评测集（全部，不抽样）。
2. 标准化：在标准化样本上逐条前向，按特征来源算逐通道统计量，冻结（src/e2/features）。算完立刻写一次检查点（第 0 步）。
3. 探针：每个（特征来源, 种类）一个探针与优化器（src/e2/probes.ProbeBank）。
4. 训练：每步一批 batch_size 个样本；每个样本一次前向，标准化后的特征喂给全部探针，探针按 probe_chunk 分块上采样、算损失、
   反传（各块的损失除以 batch_size，梯度在批内累加）；批末每个探针各自 step。GT 没有有效像素的样本跳过（计数）。
5. 曲线：完成第 eval_every·k 步（以及最后一步）后，在验证子集上评一次全部探针（无 CI），连同这段时间的平均训练损失、当前学习率
   写进 curves.jsonl。
6. 评测：训练完成后在评测集上评一次全部探针，带 bootstrap CI → metrics.json、eval_samples.npz（逐样本统计）、README.md。
   另按光栅位置（上半 / 下半、token 光栅下标的四分位；src/e2/positions.py）各报一遍标题指标（点估计，metrics.json 的
   probes.<探针>.eval_by_position 与 README 的表；DEV-0024，D53-3 的 I2），标题指标不变。
   特征键：整数层 L0 … L{N} 是逐层曲线；参考线 pre_proj 与 L{N}_pre（最后一层归一化之前，D53-3 的 I1）单列（metrics.json 的
   curve_keys / single_keys）。
开跑前（load_data）：训练池用的去重剔除清单必须比较过这个任务的评测集（src/e2/data.check_dedupe_coverage；旧格式的清单真模型
不跑），清单的 max_hamming 与 eval_sets 记进 config.yaml 的 data.dedupe。

输出（叶目录 <out>/<模型>/<任务>/）：
- config.yaml：运行身份（模型与 revision、图像口径、任务、输入顺序与指令、层与参考线、探针、训练参数、种子、配置 sha256、
  训练 / 验证 / 标准化 / 评测样本的 id 指纹）；不含时间、机器路径。meta.json：git、主机、包版本、数据清单的 sha256（src/runlog）。
- curves.jsonl、metrics.json、eval_samples.npz、README.md：只由训练与评测本身决定，不含墙钟；CPU 上续跑与一次跑完逐字节相同
  （tests/test_e2_cli.py 钉住）。GPU 上上采样的反向用原子加，不保证逐位可复现（见 RUN-0025 一类的噪声底）。
- accounting.json（各次运行合计）与 logs/run_<k>.json（每次调用一份）：GPU 墙钟（分阶段）、前向次数、视觉 token 与像素数、
  显存峰值（torch.cuda.max_memory_allocated；CPU 上为 null）。
- 检查点：<ckpt_dir>/state.pt（默认 <叶目录>/ckpt/，可用 --ckpt-root 放到别处；不入库）：探针、优化器、调度器、标准化统计量、
  训练损失的累计量、步数，原子写入；每 ckpt_every 步与最后一步写一次。训练完成后另写 probes_final.pt（探针权重 + 标准化统计量）；
  整个任务完成后删掉 state.pt（优化器状态只为续跑；9B 约 2 GB），--keep-optimizer-state 保留。

断点续传（--resume）：叶目录非空时必须给 --resume 且 config.yaml 与本次完全相同；从检查点的步数接着训练，curves.jsonl 里检查点
之后的行截掉重算；训练已完成而评测没完成时，评测整段重做（确定性的）。metrics.json 已标记完成就直接返回。
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import yaml

from src.data.land.nyuv2 import npz_bytes
from src.data.records import LoadResult, Sample
from src.e2 import positions as POS
from src.e2.config import PROBE_KINDS, E2Config, curve_steps, layer_label
from src.e2.data import (
    E2Paths,
    Split,
    batch_indices,
    check_dedupe_coverage,
    ids_digest,
    load_eval_set,
    load_train_pool,
    split_pool,
)
from src.e2.features import (
    ChannelStats,
    FeatureExtractor,
    FeatureKey,
    ForwardCounters,
    Normalizer,
    compute_stats,
)
from src.e2.probes import ProbeBank
from src.e2.tasks import E2Task, PositionAccumulator, make_task, upsample
from src.runlog import write_run_metadata

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = "config.yaml"
CURVES = "curves.jsonl"
METRICS = "metrics.json"
SAMPLES = "eval_samples.npz"
README = "README.md"
ACCOUNTING = "accounting.json"
STATE = "state.pt"
FINAL_PROBES = "probes_final.pt"


@dataclass(frozen=True)
class RunSpec:
    model_key: str
    family: str
    repo: str
    revision: str
    adapter_model_key: str
    tiny: bool
    policy: Mapping[str, Any]
    model_dtype: str
    task: str
    layer_spec: str
    layers: tuple[int, ...]
    references: tuple[str, ...]
    num_layers: int
    hidden_size: int
    eval_limit: int | None = None
    pre_norm_last: bool = False  # 加特征键 L{N}_pre（最后一层归一化之前；configs/e2_probes.json 的 layers.pre_norm_last）


def leaf_dir(out: str | Path, model_key: str, task: str) -> Path:
    return Path(out) / model_key / task


_MACHINE_KEYS = ("dedupe_file", "coco_json")  # 机器相关的路径，不进运行身份（剔除清单本身按 sha256 进）


def _scalars(d: Mapping[str, Any]) -> dict[str, Any]:
    """只留下标量项（剔除清单之类的长列表不进运行身份，只留计数），去掉机器相关的路径。"""
    return {k: v for k, v in d.items()
            if k not in _MACHINE_KEYS and (isinstance(v, (int, float, str, bool)) or v is None)}


def _file_sha256(path: str | Path | None) -> str | None:
    if not path or not Path(path).exists():
        return None
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _identity(cfg: Mapping[str, Any]) -> dict[str, Any]:
    return yaml.safe_load(yaml.safe_dump(dict(cfg), sort_keys=False, allow_unicode=True))


def identity_sha256(cfg: Mapping[str, Any]) -> str:
    return hashlib.sha256(yaml.safe_dump(_identity(cfg), sort_keys=True, allow_unicode=True).encode("utf-8")).hexdigest()


def _sync(device: Any) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)


def _atomic_torch_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    torch.save(obj, tmp)
    tmp.replace(path)


def _jsonl(rec: Mapping[str, Any]) -> str:
    return json.dumps(rec, ensure_ascii=False, sort_keys=False) + "\n"


def _fmt(v: Any, nd: int = 4) -> str:
    return "—" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


# ---------------------------------------------------------------------------
# 一个任务的状态与基本操作（正式运行与 --estimate 共用）
# ---------------------------------------------------------------------------


class TaskRun:
    def __init__(self, adapter: Any, cfg: E2Config, spec: RunSpec, task_cfg: Mapping[str, Any], paths: E2Paths, *,
                 log: Callable[[str], None] | None = None) -> None:
        self.adapter = adapter
        self.cfg = cfg
        self.spec = spec
        self.tc = dict(task_cfg)
        self.paths = paths
        self.log = log or (lambda m: print(m, file=sys.stderr, flush=True))
        self.task: E2Task = make_task(self.tc)
        self.device = adapter.device
        self.counters = ForwardCounters()
        self.extractor = FeatureExtractor(adapter, cfg, spec.task, spec.layers, spec.references, self.counters,
                                          pre_norm_last=spec.pre_norm_last)
        self.bank: ProbeBank | None = None
        self.norm: Normalizer | None = None
        self.stats: dict[FeatureKey, ChannelStats] | None = None
        self.n_skipped = 0
        self.pool: LoadResult | None = None
        self.split: Split | None = None
        self.eval_set: LoadResult | None = None
        self.dedupe: dict[str, Any] | None = None

    # ----- 数据 -----
    @staticmethod
    def _drop_empty_expressions(res: LoadResult) -> LoadResult:
        """refseg：指代句去掉首尾空白后为空的样本没法组成指令，剔除并计数（不静默）。"""
        keep = [s for s in res.records if str(s.fields.get("expression", "")).strip()]
        return LoadResult(keep, {**res.stats, "n_empty_expression": len(res.records) - len(keep)})

    def load_data(self) -> None:
        self.pool = load_train_pool(self.spec.task, self.paths)
        # 开跑前：剔除清单比较过这个任务的评测集（旧格式的清单真模型不跑；DEV-0024）
        self.dedupe = check_dedupe_coverage(self.spec.task, self.pool.stats.get("dedupe_file"), real_model=not self.spec.tiny)
        if self.spec.task == "refseg":
            self.pool = self._drop_empty_expressions(self.pool)
        self.split = split_pool(self.spec.task, self.pool.records, self.tc["holdout"],
                                n_stats=int(self.tc["feature_norm"]["n_samples"]))
        ev = load_eval_set(self.spec.task, self.paths, eval_subset=self.tc.get("eval_subset"))
        if self.spec.task == "refseg":
            ev = self._drop_empty_expressions(ev)
        if self.spec.eval_limit is not None:
            ev = LoadResult(ev.records[: self.spec.eval_limit], {**ev.stats, "eval_limit": self.spec.eval_limit})
        elif self.tc.get("n_eval_expected") is not None and len(ev) != int(self.tc["n_eval_expected"]):
            raise ValueError(f"{self.spec.task} 的评测集有 {len(ev)} 条，配置写的是 {self.tc['n_eval_expected']}：落地不全或划分不对；"
                             "不抽样（任务书），先补齐数据")
        if not ev.records:
            raise ValueError(f"{self.spec.task} 的评测集是空的")
        self.eval_set = ev

    # ----- 标准化与探针 -----
    def set_stats(self, stats: Mapping[FeatureKey, ChannelStats]) -> None:
        missing = [k for k in self.extractor.keys if k not in stats]
        if missing:
            raise ValueError(f"标准化统计量缺少 {missing}")
        self.stats = dict(stats)
        self.norm = Normalizer.from_stats(self.stats, eps=float(self.tc["feature_norm"]["eps"]), device=self.device)
        dims = {k: int(self.stats[k].mean.shape[0]) for k in self.extractor.keys}
        self.bank = ProbeBank(self.tc, dims, device=self.device, base_seed=self.cfg.seed)

    def probe_audit(self) -> dict[str, Any] | None:
        """运行期审计（进运行记录）：探针参数的设备与 dtype 集合、可训练参数总数。"""
        if self.bank is None:
            return None
        ps = [p for e in self.bank.entries.values() for p in e.module.parameters()]
        return {"devices": sorted({str(p.device) for p in ps}), "dtypes": sorted({str(p.dtype) for p in ps}),
                "n_params": int(sum(p.numel() for p in ps)), "n_probes": len(self.bank.entries)}

    def compute_stats(self, samples: Sequence[Sample]) -> dict[FeatureKey, ChannelStats]:
        return compute_stats(self.extractor, samples, log=self.log)

    def features(self, s: Sample, phase: str) -> dict[FeatureKey, torch.Tensor]:
        raw, _ = self.extractor.raw(s, phase=phase)
        return {k: self.norm.apply(k, g) for k, g in raw.items()}

    def _chunks(self, feats: Mapping[FeatureKey, torch.Tensor], hw: tuple[int, int]) -> list[list[str]]:
        """一块至多 probe_chunk 个探针，且上采样后的 P·C·H·W ≤ probe_chunk_max_elements（至少 1 个）。只由配置与 GT 尺寸决定。"""
        per = int(self.task.out_channels) * int(hw[0]) * int(hw[1])
        size = max(1, min(int(self.tc["probe_chunk"]), int(self.tc["probe_chunk_max_elements"]) // max(1, per)))
        return self.bank.chunks({k: tuple(v.shape[1:]) for k, v in feats.items()}, size)

    # ----- 训练一步 -----
    def train_step(self, step: int, train: Sequence[Sample]) -> tuple[dict[str, float], int]:
        """第 step 步（0 起）。返回 ({探针: 本批各样本损失之和}, 用到的样本数)。"""
        bs = int(self.tc["batch_size"])
        batch = [train[i] for i in batch_indices(len(train), step, bs, seed=self.cfg.seed, task=self.spec.task)]
        self.bank.train()
        self.bank.zero_grad()
        sums = {n: 0.0 for n in self.bank.names}
        pending: list[tuple[list[str], torch.Tensor]] = []  # 损失留在设备上，批末一次取回（少同步）
        used = 0
        for s in batch:
            feats = self.features(s, "train")
            tgt = self.task.target(s, self.device)
            if tgt.n_valid == 0:
                self.n_skipped += 1
                continue
            for chunk in self._chunks(feats, tgt.hw):
                up = upsample(self.bank.predict(chunk, feats), tgt.hw, self.tc["upsample"])
                losses = self.task.loss(up, tgt)
                (losses.sum() / bs).backward()
                pending.append((chunk, losses.detach()))
                del up, losses
            used += 1
        self.bank.step()
        for chunk, losses in pending:
            for n, v in zip(chunk, losses.double().cpu().tolist()):
                sums[n] += v
        return sums, used

    # ----- 评一遍 -----
    def collect(self, samples: Sequence[Sample], phase: str, *,
                positions: PositionAccumulator | None = None) -> dict[str, list[Any]]:
        """全部探针在 samples 上的逐样本统计（task.sample_stats），按探针名。给了 positions 时另按光栅位置分开累计
        （task.position_stats；区域按 LLM 层的 token 网格划分，src/e2/positions.py）。"""
        self.bank.eval()
        out: dict[str, list[Any]] = {n: [] for n in self.bank.names}
        with torch.no_grad():
            for i, s in enumerate(samples):
                feats = self.features(s, phase)
                tgt = self.task.target(s, self.device)
                grid = tuple(int(v) for v in feats[self.extractor.layers[0]].shape[1:])
                if positions is not None:
                    positions.add_sample(tgt)
                for chunk in self._chunks(feats, tgt.hw):
                    up = upsample(self.bank.predict(chunk, feats), tgt.hw, self.tc["upsample"])
                    if positions is None:
                        whole = self.task.sample_stats(up, tgt)
                    else:
                        whole, pos = self.task.eval_stats(up, tgt, grid)
                        for n, ps in zip(chunk, pos):
                            positions.add(n, ps)
                    for n, st in zip(chunk, whole):
                        out[n].append(st)
                    del up
                if phase == "eval" and (i + 1 == len(samples) or (i + 1) % 200 == 0):
                    self.log(f"[{self.spec.model_key} {self.spec.task} eval] {i + 1}/{len(samples)}")
        self.bank.train()
        return out


# ---------------------------------------------------------------------------
# 运行身份
# ---------------------------------------------------------------------------


def run_config(run: TaskRun, cfg: E2Config) -> dict[str, Any]:
    spec, tc, split, ev, pool = run.spec, run.tc, run.split, run.eval_set, run.pool
    return {
        "experiment": "E2",
        "framework": "src/e2（DEV-0023）",
        "model": {"key": spec.model_key, "family": spec.family, "repo": spec.repo, "revision": spec.revision,
                  "adapter_model_key": spec.adapter_model_key, "tiny": spec.tiny, "dtype": spec.model_dtype},
        "image_policy": dict(spec.policy),
        "task": spec.task,
        "input": {"order": tc["order"],
                  "instruction": cfg.refseg_template if spec.task == "refseg" else cfg.dense_prompt},
        "layers": {"spec": spec.layer_spec, "list": list(spec.layers), "num_layers": spec.num_layers,
                   "hidden_size": spec.hidden_size, "pre_norm_last": run.extractor.pre_norm,
                   "convention": "0 = 视觉特征并入之后的嵌入输出；i = 第 i 个 decoder 层输出；num_layers = 最终归一化之后（D19）；"
                                 "L{N}_pre = 最后一个 decoder 层的原始输出（最终归一化之前，单列；D53-3）"},
        "references": list(spec.references),
        "feature_keys": [layer_label(k) for k in run.extractor.keys],
        "training": {k: tc[k] for k in ("steps", "batch_size", "eval_every", "ckpt_every", "probe_chunk",
                                         "probe_chunk_max_elements", "out_channels", "holdout", "loss", "optim",
                                         "feature_norm", "probes", "upsample", "bootstrap_resamples")},
        "seed": cfg.seed,
        "config": {"path": cfg.path, "sha256": cfg.sha256, "version": cfg.doc["version"]},
        "data": {
            "train_pool": _scalars(pool.stats),
            "train_exclusions_sha256": _file_sha256(pool.stats.get("dedupe_file")),
            "dedupe": {k: run.dedupe[k] for k in ("schema", "legacy", "max_hamming", "eval_sets", "required_eval_sets")},
            "holdout": {k: v for k, v in split.info.items() if k not in ("val_ids", "stats_ids")},
            "train_ids_sha256": ids_digest([s.id for s in split.train]),
            "val_ids": list(split.info["val_ids"]),
            "stats_ids_sha256": ids_digest(list(split.info["stats_ids"])),
            "eval": {"n": len(ev), "ids_sha256": ids_digest([s.id for s in ev.records]), "stats": _scalars(ev.stats),
                     "subset": tc.get("eval_subset")},
        },
        "eval_limit": spec.eval_limit,
    }


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------


def _probe_meta(bank: ProbeBank) -> dict[str, dict[str, Any]]:
    counts = bank.param_counts()
    return {n: {"feature": layer_label(e.key), "kind": e.kind, "n_params": counts[n], "on_curve": isinstance(e.key, int)}
            for n, e in bank.entries.items()}


def _readme_probe_table(run: TaskRun, metrics: Mapping[str, Any], keys: Sequence[FeatureKey]) -> list[str]:
    hl = run.task.headline
    head = "| 特征 | " + " | ".join(f"{k} {lab} {d}" for k in PROBE_KINDS for lab, d in hl) + " |"
    lines = [head, "|---|" + "---|" * (len(PROBE_KINDS) * len(hl))]
    for key in keys:
        cells = []
        for kind in PROBE_KINDS:
            m = metrics["probes"][f"{layer_label(key)}/{kind}"]["eval"]
            for lab, _ in hl:
                ci = m.get(f"{lab}_ci")
                cells.append(f"{_fmt(m.get(lab))} [{_fmt(ci['low'])}, {_fmt(ci['high'])}]" if ci else _fmt(m.get(lab)))
        lines.append(f"| {layer_label(key)} | " + " | ".join(cells) + " |")
    return lines


def _readme_position_tables(run: TaskRun, metrics: Mapping[str, Any]) -> list[str]:
    regs = [(s, r) for s, rs in POS.SPLITS.items() for r in rs]
    pos = metrics["positions"]
    lines = ["", "## 按光栅位置分开（评测集，点估计，无 CI；D53-3 的 I2）", "",
             f"- 划分：{POS.DEFINITION}",
             f"- 评测样本 {pos['n_samples']} 条" + (f"，其中 {pos['n_unassigned']} 条 GT 掩码为空、不进任何区域" if pos["n_unassigned"] else ""),
             "- 标题指标不变（上面的表）；括号里是区域内参与的样本数（像素划分：区域里有评测像素的样本）", ""]
    for lab, d in run.task.headline:
        lines += [f"### {lab} {d}", "", "| 探针 | 全图 | " + " | ".join(POS.REGION_TITLES[r] for _, r in regs) + " |",
                  "|---|---|" + "---|" * len(regs)]
        for name in run.bank.names:
            p = metrics["probes"][name]
            cells = [_fmt(p["eval"].get(lab))]
            for s, r in regs:
                m = p["eval_by_position"][s][r]
                n = m.get("n_used", m.get("n"))
                cells.append(f"{_fmt(m.get(lab))} ({n})")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def readme_text(run: TaskRun, cfg_doc: Mapping[str, Any], metrics: Mapping[str, Any], curves: Sequence[Mapping[str, Any]]) -> str:
    spec, task = run.spec, run.task
    tr = cfg_doc["training"]
    lines = [f"# E2 · {spec.model_key} · {spec.task}", "",
             f"- 模型：{spec.repo}@{spec.revision}（{spec.family}；适配器 {spec.adapter_model_key}；{spec.model_dtype}；冻结）",
             f"- 图像口径：{cfg_doc['image_policy']}（E1 冻结口径，D28-3）",
             f"- 输入：{cfg_doc['input']['order']}；指令 `{cfg_doc['input']['instruction']}`",
             f"- 层：{spec.layer_spec} → {list(spec.layers)}（num_layers = {spec.num_layers}；最后一层是最终归一化之后的值，D19）"
             + (f"；另有 {run.extractor.pre_norm} = 最后一个 decoder 层的原始输出（最终归一化之前），单列" if run.extractor.pre_norm else ""),
             f"- 参考线：{list(spec.references) or '无'}" + ("；post_proj 就是第 0 层（D21-3），不是第二条证据" if "post_proj" in spec.references else ""),
             f"- 探针：linear = 1×1 卷积；conv = 3×3 → {tr['probes']['conv']['hidden']}、GELU、3×3；双线性上采样到 GT 分辨率",
             f"- 训练：{tr['steps']} 步 × 批 {tr['batch_size']}，AdamW lr {tr['optim']['lr']}，warmup {tr['optim']['warmup_steps']} + 余弦，"
             f"种子 {cfg_doc['seed']}；损失 {tr['loss']['kind']}；按通道标准化（{tr['feature_norm']['n_samples']} 条训练样本上的统计量）",
             f"- 验证子集（训练集留出，只记曲线）：{metrics['n_val']} 条；评测集：{metrics['n_eval']} 条（全部，不抽样）；"
             f"bootstrap {tr['bootstrap_resamples']} 次，95%",
             f"- GT 无有效像素而跳过的训练样本：{metrics['n_train_skipped']}"]
    pc = metrics.get("post_proj_check")
    if pc:
        lines.append(f"- post_proj 核对（评测集第一条 {pc['sample']}）：vision_features 与第 0 层逐位相同 = {pc['equal']}")
    lines.append("")
    hl = task.headline
    curve = [k for k in run.extractor.keys if isinstance(k, int)]
    single = [k for k in run.extractor.keys if not isinstance(k, int)]
    lines += [f"## 逐层探针（评测集，95% CI；逐层曲线，L{spec.num_layers} 是最终归一化之后）", ""]
    lines += _readme_probe_table(run, metrics, curve)
    if single:
        lines += ["", "## 单列（不与上面的逐层曲线连成一条线：参考线、最后一层归一化之前）", ""]
        lines += _readme_probe_table(run, metrics, single)
    lines += _readme_position_tables(run, metrics)
    first, d = hl[0]
    steps = sorted({c["step"] for c in curves})
    lines += ["", f"## 训练曲线（验证子集上的 {first} {d}；括号里是这段的平均训练损失）", ""]
    lines += ["| 探针 | " + " | ".join(f"步 {s}" for s in steps) + " |", "|---|" + "---|" * len(steps)]
    by = {(c["probe"], c["step"]): c for c in curves}
    for name in run.bank.names:
        cells = []
        for s in steps:
            c = by.get((name, s))
            cells.append("—" if c is None else f"{_fmt(c['val'].get(first))} ({_fmt(c['train_loss'], 3)})")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "## 文件", "",
              "- metrics.json：每个探针的评测指标与 CI、验证子集的最终指标；eval_samples.npz：逐样本统计（配对比较用）；",
              "- curves.jsonl：每个曲线点 × 探针一行；config.yaml：运行身份；meta.json：git、主机、包版本；",
              "- accounting.json：GPU 墙钟、前向次数、显存峰值（各次运行合计）；logs/run_<k>.json：每次调用一份。", ""]
    return "\n".join(lines)


def write_samples_npz(path: Path, run: TaskRun, stats: Mapping[str, Sequence[Any]], ids: Sequence[str]) -> None:
    arrays: dict[str, np.ndarray] = {"sample_ids": np.array(list(ids), dtype=str)}
    for name in run.bank.names:
        e = run.bank.entries[name]
        for f, a in run.task.pack(stats[name]).items():
            arrays[f"{layer_label(e.key)}__{e.kind}__{f}"] = a
    _atomic_write_bytes(path, npz_bytes(**arrays))


# ---------------------------------------------------------------------------
# 正式运行
# ---------------------------------------------------------------------------


class StopRequested(Exception):
    """--stop-after-steps 到了：检查点已写，下次 --resume 接着跑。"""


def _read_curves(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    keep = raw.rfind(b"\n") + 1
    return [json.loads(l) for l in raw[:keep].decode("utf-8").splitlines() if l.strip()]


def _write_curves(path: Path, recs: Sequence[Mapping[str, Any]]) -> None:
    _atomic_write_bytes(path, "".join(_jsonl(r) for r in recs).encode("utf-8"))


def _append(path: Path, lines: Sequence[str]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write("".join(lines))
        f.flush()
        os.fsync(f.fileno())


def accounting_doc(leaf: Path) -> dict[str, Any]:
    """logs/run_*.json 的合计：各阶段前向次数与耗时相加，显存峰值取最大。"""
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((leaf / "logs").glob("run_*.json"))]
    phases: dict[str, dict[str, float]] = {}
    wall = 0.0
    peaks = []
    for r in runs:
        wall += float(r.get("wall_s") or 0.0)
        if r.get("peak_mem_bytes") is not None:
            peaks.append(int(r["peak_mem_bytes"]))
        for ph, d in (r.get("phases") or {}).items():
            acc = phases.setdefault(ph, {})
            for k, v in d.items():
                acc[k] = acc.get(k, 0) + v
    n_fwd = int(sum(d.get("n_forward", 0) for d in phases.values()))
    return {"n_runs": len(runs), "wall_s": wall, "gpu_hours": wall / 3600.0, "n_forward": n_fwd, "phases": phases,
            "peak_mem_bytes": max(peaks) if peaks else None, "device": runs[-1].get("device") if runs else None,
            "gpu": runs[-1].get("gpu") if runs else None,
            "note": "wall_s = 各次调用的墙钟之和（单卡独占时即 GPU 墙钟）；phases 里 forward_s 只含 hidden_states 前向（CUDA 前后同步），"
                    "wall_s 另含探针、上采样、损失、指标与 bootstrap"}


def run_task(adapter: Any, cfg: E2Config, spec: RunSpec, task_cfg: Mapping[str, Any], paths: E2Paths, out_dir: str | Path, *,
             resume: bool = False, argv: Sequence[str] = (), ckpt_root: str | Path | None = None,
             stop_after_steps: int | None = None, log: Callable[[str], None] | None = None,
             keep_optimizer_state: bool = False, _fail_after_step: int | None = None) -> dict[str, Any] | None:
    """跑一个任务；返回 metrics（stop_after_steps 到了时返回 None）。规则见模块说明。"""
    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    run = TaskRun(adapter, cfg, spec, task_cfg, paths, log=log)
    run.load_data()
    ident = run_config(run, cfg)
    ident_sha = identity_sha256(ident)
    leaf = leaf_dir(out_dir, spec.model_key, spec.task)
    ckpt_dir = (Path(ckpt_root) / spec.model_key / spec.task) if ckpt_root else leaf / "ckpt"
    if leaf.exists() and any(leaf.iterdir()):
        if not resume:
            raise FileExistsError(f"{leaf} 已存在且非空；续跑请加 --resume")
        if not (leaf / CONFIG).exists():
            raise FileExistsError(f"{leaf} 里没有 {CONFIG}，不能续跑")
        old = yaml.safe_load((leaf / CONFIG).read_text(encoding="utf-8"))
        if _identity(old) != _identity(ident):
            diff = sorted(k for k in set(_identity(old)) | set(_identity(ident)) if _identity(old).get(k) != _identity(ident).get(k))
            raise ValueError(f"{leaf}：续跑的配置与已有的不同（{diff}），拒绝混写")
        if (leaf / METRICS).exists():
            m = json.loads((leaf / METRICS).read_text(encoding="utf-8"))
            if m.get("complete"):
                log(f"[{spec.model_key} {spec.task}] 已完成，跳过（{leaf / METRICS}）")
                return m
    else:
        leaf.mkdir(parents=True, exist_ok=True)
        files = [REPO_ROOT / cfg.path if not Path(cfg.path).is_absolute() else Path(cfg.path)]
        excl = paths.train_root / "dedupe" / "train_exclusions.json"
        files += [p for p in (excl, Path(paths.dev_subsets)) if p.exists()]
        write_run_metadata(leaf, config=ident, seed=cfg.seed, data_files=files, repo_root=REPO_ROOT)
    (leaf / "logs").mkdir(exist_ok=True)

    dev = torch.device(adapter.device)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    run_k = len(list((leaf / "logs").glob("run_*.json"))) + 1
    run_path = leaf / "logs" / f"run_{run_k:03d}.json"
    run_log: dict[str, Any] = {"argv": list(argv), "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                               "device": str(dev), "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
                               "paths": paths.as_dict(), "ckpt_dir": str(ckpt_dir), "finished_utc": None}
    run_path.write_text(json.dumps(run_log, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    walls: dict[str, float] = {}
    done_box = [0]  # 本次调用跑了几步（finish_log 用）
    t_all = time.perf_counter()

    def finish_log(status: str) -> None:
        _sync(dev)
        run_log.update(status=status, finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                       wall_s=round(time.perf_counter() - t_all, 3),
                       phases={ph: {**d, "wall_s": round(walls.get(ph, 0.0), 3)} for ph, d in run.counters.by_phase.items()},
                       peak_mem_bytes=int(torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else None,
                       post_proj_checks=run.extractor.post_proj_checks, n_train_skipped=run.n_skipped,
                       steps_this_run=done_box[0], audit={"features": run.extractor.audit, "probes": run.probe_audit()})
        run_path.write_text(json.dumps(run_log, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        (leaf / ACCOUNTING).write_text(json.dumps(accounting_doc(leaf), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    def timed(phase: str, fn: Callable[[], Any]) -> Any:
        _sync(dev)
        t0 = time.perf_counter()
        r = fn()
        _sync(dev)
        walls[phase] = walls.get(phase, 0.0) + time.perf_counter() - t0
        return r

    steps = int(task_cfg["steps"])
    cpoints = set(curve_steps(steps, int(task_cfg["eval_every"])))
    ckpt_every = int(task_cfg["ckpt_every"])
    state_path = ckpt_dir / STATE
    loss_acc: dict[str, list[float]] = {}
    start = 0
    try:
        st = torch.load(state_path, map_location="cpu", weights_only=False) if state_path.exists() else None
        if st is not None:
            if st["identity_sha256"] != ident_sha:
                raise ValueError(f"{state_path} 的运行身份与本次不同，拒绝续跑")
            run.set_stats({k: ChannelStats.from_state(v) for k, v in st["stats"].items()})
            run.bank.load_state_dict(st["bank"])
            start, loss_acc, run.n_skipped = int(st["step"]), {k: list(v) for k, v in st["loss_acc"].items()}, int(st["n_skipped"])
            log(f"[{spec.model_key} {spec.task}] 从检查点第 {start} 步续跑")
        else:
            stats = timed("stats", lambda: run.compute_stats(run.split.stats))
            run.set_stats(stats)
            loss_acc = {n: [0.0, 0.0] for n in run.bank.names}

        def save(step: int) -> None:
            _atomic_torch_save({"identity_sha256": ident_sha, "step": step, "bank": run.bank.state_dict(),
                                "stats": {k: v.state() for k, v in run.stats.items()}, "loss_acc": loss_acc,
                                "n_skipped": run.n_skipped}, state_path)

        if st is None:
            save(0)
        # 曲线：检查点之后的行截掉（中断时可能已写了下一段）
        curves = [c for c in _read_curves(leaf / CURVES) if c["step"] <= start]
        _write_curves(leaf / CURVES, curves)
        done_this_run = 0
        every = max(1, steps // 60)
        t_train = time.perf_counter()
        for step in range(start, steps):
            sums, used = timed("train", lambda: run.train_step(step, run.split.train))
            if (step + 1) % every == 0 and (step + 1) not in cpoints:
                rate = (time.perf_counter() - t_train) / (step + 1 - start)
                log(f"[{spec.model_key} {spec.task}] 第 {step + 1}/{steps} 步（{rate:.2f} s/步，含曲线）")
            for n in run.bank.names:
                loss_acc[n][0] += sums[n]
                loss_acc[n][1] += used
            s1 = step + 1
            if s1 in cpoints:
                vstats = timed("val", lambda: run.collect(run.split.val, "val"))
                lrs = run.bank.lrs()
                lines = []
                for n in run.bank.names:
                    e = run.bank.entries[n]
                    agg = run.task.aggregate(vstats[n], n_resamples=0, ci=False)
                    tl = loss_acc[n][0] / loss_acc[n][1] if loss_acc[n][1] else None
                    lines.append(_jsonl({"step": s1, "probe": n, "feature": layer_label(e.key), "kind": e.kind,
                                         "train_loss": tl, "lr_next": lrs[n], "val": agg}))
                    loss_acc[n] = [0.0, 0.0]
                _append(leaf / CURVES, lines)
                log(f"[{spec.model_key} {spec.task}] 第 {s1}/{steps} 步，曲线已记")
            done_this_run += 1
            done_box[0] = done_this_run
            stop = stop_after_steps is not None and done_this_run >= stop_after_steps and s1 < steps
            if s1 % ckpt_every == 0 or s1 == steps or stop:
                save(s1)
            if _fail_after_step is not None and s1 == _fail_after_step:
                raise RuntimeError(f"测试注入的中断（第 {s1} 步之后）")
            if stop:
                raise StopRequested(f"{spec.model_key} {spec.task}：本次跑了 {done_this_run} 步（到第 {s1}/{steps} 步），检查点已写")
        if start < steps or not (ckpt_dir / FINAL_PROBES).exists():
            _atomic_torch_save({"identity_sha256": ident_sha, "probes": run.bank.module_state(),
                                "stats": {k: v.state() for k, v in run.stats.items()}}, ckpt_dir / FINAL_PROBES)

        # 评测（post_proj 在评测集第一条上再核一次：这一条的记录与是否续跑无关）；另按光栅位置分开累计（I2）
        run.extractor.request_post_proj_check()
        pos_acc = PositionAccumulator(run.task, run.bank.names)
        estats = timed("eval", lambda: run.collect(run.eval_set.records, "eval", positions=pos_acc))
        vstats = timed("val_final", lambda: run.collect(run.split.val, "val_final"))
        n_res = int(task_cfg["bootstrap_resamples"])
        probes_doc = {}
        meta = _probe_meta(run.bank)
        by_pos = pos_acc.result()
        for n in run.bank.names:
            probes_doc[n] = {**meta[n], "eval": run.task.aggregate(estats[n], n_resamples=n_res, ci=True),
                             "eval_by_position": by_pos[n],
                             "val_final": run.task.aggregate(vstats[n], n_resamples=0, ci=False)}
        metrics = {"experiment": "E2", "task": spec.task, "model": spec.model_key, "family": spec.family,
                   "revision": spec.revision, "feature_keys": [layer_label(k) for k in run.extractor.keys],
                   "curve_keys": [layer_label(k) for k in run.extractor.keys if isinstance(k, int)],
                   "single_keys": [layer_label(k) for k in run.extractor.keys if not isinstance(k, int)],
                   "layers": list(spec.layers), "references": list(spec.references), "pre_norm_last": run.extractor.pre_norm,
                   "headline": [{"metric": m, "better": d} for m, d in run.task.headline],
                   "positions": {"definition": POS.DEFINITION, "splits": {s: list(r) for s, r in POS.SPLITS.items()},
                                 "unit": run.task.position_unit, "ci": None, "n_samples": pos_acc.n_samples,
                                 "n_unassigned": pos_acc.n_unassigned},
                   "n_eval": len(run.eval_set), "n_val": len(run.split.val), "n_train": len(run.split.train),
                   "n_train_skipped": run.n_skipped, "post_proj_check": run.extractor.post_proj_check("eval"),
                   "probes": probes_doc, "complete": True}
        write_samples_npz(leaf / SAMPLES, run, estats, [s.id for s in run.eval_set.records])
        (leaf / README).write_text(readme_text(run, ident, metrics, _read_curves(leaf / CURVES)), encoding="utf-8")
        # metrics.json 最后写：它的 complete 是"整个任务已完成"的标记
        _atomic_write_bytes(leaf / METRICS, (json.dumps(metrics, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))
        if not keep_optimizer_state and state_path.exists():
            state_path.unlink()  # 优化器状态只为续跑训练；完成后探针权重与统计量留在 probes_final.pt
        finish_log("complete")
        return metrics
    except StopRequested as e:
        log(f"[停止] {e}")
        finish_log("stopped")
        return None
    except BaseException:
        finish_log("error")
        raise


__all__ = [
    "ACCOUNTING",
    "CONFIG",
    "CURVES",
    "FINAL_PROBES",
    "METRICS",
    "README",
    "RunSpec",
    "SAMPLES",
    "STATE",
    "TaskRun",
    "accounting_doc",
    "identity_sha256",
    "leaf_dir",
    "readme_text",
    "run_config",
    "run_task",
]
