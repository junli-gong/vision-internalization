"""E5 主比较的训练（DEV-0037 §3）：一个起点 × 一个模型 × 一个数据量档（× 任务子集）× 一个种子 →
<out>/train/<起点>/<模型>/<n 标记>/seed<k>/（n 标记 = n<数据量>，部分任务时加 _<字母>，见 src/e5/train_data.TrainSet.tag）。

结构借自 E3b（src/e3/train.py）：训练器是 src/e3/train.Trainer 的子类（E5Trainer），一步优化、token 行的 fp32 主副本、AdamW 两个参数组、
warmup + 余弦、梯度检查点、续跑状态、记账都原样复用；只换三处：
- 搭建（setup_fresh / setup_resume）：先按起点的规则叠加（src/e5/stack.setup_start：合并起点 LoRA，网格 token 恢复或新加），再
  torch.manual_seed(种子) → 注入一份新的 LoRA（r、alpha、dropout 同 E3b）→ trainable_token_rows（67 个网格 token 行继续训练）。
  续跑：合并起点 LoRA → load_trained(<latest>/model, merge=False)（E5 训练过的网格 token 行与 LoRA）→ 优化器、调度器、随机数状态。
- 损失（_loss）：受限交叉熵，按 E5 的任务分组，允许集合取 src/e5/token_grid.E5Tokens.allowed_ids(任务)（训练与受限解码同一个集合），
  只在网格 token 上算（training_forward 只返回目标位置）；一步之内全部监督 token 的平均（同 E3b）。
- 数据：src/e5/train_data（训练集、固定 epoch 的日程、样本构造），没有训练池与去重清单（合成数据按场景种子与物体身份划分，D23-2）。

**开训前的核对**（src/e5/stack 的模块说明）：起点检查点的合并核对（判据 v1，D50-3；不过抛 MergeCheckFailed，调用方拒绝训练）；
合并起点 LoRA 之后的权重指纹必须与 factory.load_trained(起点, merge=True) 的相同（不同抛 StartError）。两项的结果放在
<out>/train/<起点>/<模型>/start_checks/，同一个起点检查点的各数据量档与种子复用。

**检查点**（D34）：<ckpt>/latest/ 同 E3b（model/ + resume_state.pt）；训练完成后 <ckpt>/final/model/（save_trained：新 LoRA、网格 token 行、
manifest，起点有检查点时 extra.safetensors 里存起点 manifest 的 sha256）+ <ckpt>/final/e5_stack.json（起点的引用、叠加后的权重指纹、
训练集与种子）。评测：scripts/eval_e5_trained.py --checkpoint <ckpt>/final（src/e5/trained_eval.py）。<ckpt> 默认是 <叶目录>/ckpt。

**小档的修正**（DEV-0038，D77-3；configs/e5_train.json 的 training.min_steps 与 selection）：
- 步数下限：1K、4K 档步数 = max(按 epoch 的步数, 1,000)，样本日程的总条数 = 步数 × 批（src/e5/train_config.plan_steps、
  src/e5/train_data.Schedule 的 total）；warmup 与余弦按实际步数排。
- val 选检查点（只对下限所列的档；src/e5/val_select.py）：完成第 val_every·k 步（以及最后一步）后在 val 划分上评一次（eval 模式、no_grad、
  随机数状态前后不变），一行写进 val_selection.jsonl；分数刷新最好时先把当前模型存成候选 <ckpt>/best_step<k>/（save_trained），再存
  latest（评测步一律存 latest，续跑状态里带全部 val 历史），再删掉别的候选。续跑时历史以 latest 的续跑状态为准，之后的候选作废。
  训练完成后选中的候选原样拷成 <ckpt>/final/model/；train_summary.json 与 e5_stack.json 记 selection（规则、选中的步、分数、val 曲线）。
  其余档照旧取最后一步。
- 配种子：起点检查点训练叶目录 config.yaml 的 seed 必须等于 --seed（src/e5/stack.check_seed_pairing；scripts/train_e5.py
  --allow-seed-mismatch 放开），记录进运行身份的 seed_pairing。

**记账**：同 E3b（train_log.jsonl 每步的确定性记录，CPU 上续跑与一次跑完逐字节相同；logs/run_<k>.json、logs/timing_run_<k>.jsonl；
accounting.json、compute.json）。train_log 的 loss_by_task、n_supervised_by_task 按 E5 的任务名。
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F
import yaml

from src.accounting import merge, write_compute_json
from src.adapters.checkpoint import read_manifest
from src.adapters.training import restricted_targets
from src.e3.targets import NoSupervision
from src.e3.train import (
    ACCOUNTING,
    COMPUTE,
    CONFIG,
    FINAL,
    LATEST,
    MODEL_SUB,
    README,
    RESUME_STATE,
    SUMMARY,
    TOKENS_INIT,
    TRAIN_LOG,
    StopRequested,
    Trainer,
    accounting_doc,
    identity_sha256,
    token_identity,
)
from src.e3.train import _append, _atomic_write_text, _dir_bytes, _identity, _jsonl, _read_log, _sync
from src.e5 import stack as STK
from src.e5.protocol import E5_TASKS, TASK_LETTER
from src.e5.token_grid import E5Tokens, PromptFormat
from src.e5.train_config import E5TrainConfig
from src.e5 import val_select as VS
from src.e5.train_data import ExampleBuilder, Schedule, TrainSet
from src.runlog import write_run_metadata

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SUB = "train"


@dataclass(frozen=True)
class RunSpec:
    model_key: str
    family: str
    repo: str
    revision: str
    adapter_model_key: str
    tiny: bool
    image_policy: str
    model_dtype: str
    seed: int
    start: STK.StartRef
    trainset: TrainSet
    seed_pairing: Mapping[str, Any] | None = None  # src/e5/stack.check_seed_pairing 的记录（DEV-0038；进运行身份）


def seed_dir_name(seed: int, data_seed: int) -> str:
    """seed<k>；数据种子不等于运行种子时另加 _d<数据种子>（两者都决定训练内容，叶目录要分开）。"""
    return f"seed{int(seed)}" + ("" if int(data_seed) == int(seed) else f"_d{int(data_seed)}")


def leaf_dir(out: str | Path, spec: RunSpec) -> Path:
    return Path(out) / TRAIN_SUB / spec.start.kind / spec.model_key / spec.trainset.tag / seed_dir_name(spec.seed,
                                                                                                          spec.trainset.data_seed)


class E5Trainer(Trainer):
    """E3b 训练器的 E5 版（模块说明）：搭建按起点叠加，损失按 E5 的任务与允许集合。"""

    def __init__(self, adapter: Any, cfg: E5TrainConfig, run_cfg: Mapping[str, Any], *, seed: int, start: STK.StartRef) -> None:
        super().__init__(adapter, cfg.e3b, run_cfg, supervision="grid", seed=seed)
        self.e5cfg = cfg
        self.start = start
        self.e5: E5Tokens | None = None
        self.stack_info: dict[str, Any] = {}

    def setup_fresh(self) -> None:
        self.tokens, self.stack_info = STK.setup_start(self.ad, self.start, self.e5cfg)  # 合并在 manual_seed 之前（模块说明）
        torch.manual_seed(self.seed)
        lo = self.rc["lora"]
        self.ad.inject_lora(r=int(lo["r"]), lora_alpha=float(lo["alpha"]), lora_dropout=float(lo["dropout"]))
        self.e5 = E5Tokens(self.tokens, str(self.rc["allowed_scope"]))
        self._finish_setup()

    def setup_resume(self, model_dir: Path) -> None:
        info: dict[str, Any] = {"kind": self.start.kind, "merged": None}
        if self.start.has_checkpoint:
            info["merged"] = STK.merge_start_lora(self.ad, self.start.checkpoint)
        info["fingerprint"] = STK.stack_fingerprint(self.ad)
        info["tokens"] = "load_trained(latest, merge=False)：E5 训练过的网格 token 行"
        self.stack_info = info
        super().setup_resume(model_dir)
        self.e5 = E5Tokens(self.tokens, str(self.rc["allowed_scope"]))

    def _loss(self, out: Any, group: Sequence[Any]) -> tuple[torch.Tensor, dict[str, list[float]], list[int]]:
        ad = self.ad
        th, tid, trow = out.target_hidden, out.target_ids, out.target_rows
        w = torch.tensor([x for e in group for x in e.weights], dtype=torch.float32, device=th.device)
        total = th.new_zeros((), dtype=torch.float32)
        per: dict[str, list[float]] = {}
        acc = [0, 0]
        for task in E5_TASKS:
            rows = [b for b, e in enumerate(group) if e.plan.task == task]
            if not rows:
                continue
            sel = torch.isin(trow, torch.tensor(rows, dtype=torch.long, device=trow.device)).nonzero().flatten()
            h, t, ws = th.index_select(0, sel), tid.index_select(0, sel), w.index_select(0, sel)
            allowed = self.e5.allowed_ids(task)
            lg = ad.restricted_logits(h, allowed)
            tt = restricted_targets(t, allowed)
            ls = (F.cross_entropy(lg, tt, reduction="none") * ws).sum()
            with torch.no_grad():
                hit = (lg.argmax(-1) == tt) & (ws > 0)
                acc[0] += int(hit.sum())
                acc[1] += int((ws > 0).sum())
            total = total + ls
            per[task] = [float(ls.detach()), float(ws.sum())]
        return total, per, acc


# ---------------------------------------------------------------------------
# 运行身份与输出
# ---------------------------------------------------------------------------


def run_identity(spec: RunSpec, cfg: E5TrainConfig, run_cfg: Mapping[str, Any], trainer: E5Trainer,
                 prompt: PromptFormat, selector: VS.ValSelector | None = None) -> dict[str, Any]:
    return {
        "experiment": "E5",
        "framework": "src/e5（DEV-0037：E5 主比较的训练）",
        "model": {"key": spec.model_key, "family": spec.family, "repo": spec.repo, "revision": spec.revision,
                  "adapter_model_key": spec.adapter_model_key, "tiny": spec.tiny, "dtype": spec.model_dtype},
        "image_policy": spec.image_policy,
        "start": spec.start.identity(),
        "seed": spec.seed,
        "training": dict(run_cfg),
        "grid_size": cfg.grid_n,
        "tokens": {**token_identity(trainer.ad, trainer.tokens, run_cfg["allowed_scope"]), "e5": trainer.e5.layout()},
        "stop_id": trainer.stop_id,
        "prompt": prompt.as_dict(),
        "config": cfg.identity(),
        "data": spec.trainset.identity(),
        "seed_pairing": None if spec.seed_pairing is None else dict(spec.seed_pairing),
        "selection": None if selector is None else selector.identity(),
    }


def steps_text(tr: Mapping[str, Any]) -> str:
    """README 的步数说明：固定 epoch，或步数下限生效（DEV-0038）。"""
    if tr.get("min_steps_binding"):
        ep = tr["n_samples"] / max(1, int(tr["n_examples"]))
        return (f"步数下限 {tr['min_steps']}（按 {tr['epochs']} 个 epoch 是 {tr['steps_by_epochs']} 步）= {tr['steps']} 步，"
                f"共 {tr['n_samples']} 条（{ep:g} 个 epoch）")
    return f"{tr['epochs']} 个 epoch = {tr['steps']} 步"


def selection_text(sel: Mapping[str, Any] | None) -> str:
    if not sel:
        return "- 检查点的选择：取最后一步（这一档不做 val 选择）"
    curve = "，".join(f"{h['step']}: {h['score']:.4f}" for h in sel["history"])
    return (f"- 检查点的选择（val 划分，每类 {sel['n_per_task']} 条，每 {sel['val_every']} 步一次；规则 {sel['rule']}）：选中第 "
            f"{sel['selected_step']} 步（分数 {sel['selected_score']:.4f}）；val 曲线 {curve}（全部读数在 {sel['val_log']}）")


def readme_text(spec: RunSpec, ident: Mapping[str, Any], summary: Mapping[str, Any], log: Sequence[Mapping[str, Any]]) -> str:
    tr = ident["training"]
    d = ident["data"]
    first, last = (log[0], log[-1]) if log else ({}, {})
    letters = "".join(TASK_LETTER[t] for t in d["tasks"])
    lines = [f"# E5 训练 · 起点 {spec.start.kind} · {spec.model_key} · {spec.trainset.tag} · seed {spec.seed}", "",
             f"- 模型：{spec.repo}@{spec.revision}（{spec.family}；{spec.model_dtype}）；图像口径 {spec.image_policy}",
             f"- 起点：{spec.start.kind}（{spec.start.grid_tokens}；起点检查点 manifest "
             f"{(spec.start.manifest_sha256 or '—')[:16]}；起点 LoRA 合并进基座后注入新 LoRA）",
             f"- 数据：train 划分，数据量档 {d['n_train']}，任务 {letters}（每类 {d['per_task']} 条，共 {d['n_examples']} 条），"
             f"数据种子 {d['data_seed']}；清单 sha256 {d['items_sha256'][:16]}…",
             f"- 训练：{steps_text(tr)}；每步 {tr['batch_size']} 条（微批 {tr['micro_batch_size']} × 累积 "
             f"{tr['grad_accum_steps']}），LoRA r {tr['lora']['r']} / alpha {tr['lora']['alpha']} / dropout {tr['lora']['dropout']}，"
             f"AdamW lr {tr['optim']['lora']['lr']}（token 行 {tr['optim']['token_rows']['lr']}，{tr['optim']['token_rows']['mode']}），"
             f"warmup {tr['warmup_steps']} 步 + 余弦；受限交叉熵（允许集合见 config.yaml 的 tokens.e5）",
             f"- 第一步损失 {first.get('loss')}，最后一步损失 {last.get('loss')}（{len(log)} 步）",
             f"- 配种子：{(ident.get('seed_pairing') or {}).get('status')}（起点的种子 {(ident.get('seed_pairing') or {}).get('start_seed')}，"
             f"--allow-seed-mismatch {(ident.get('seed_pairing') or {}).get('allow_mismatch')}）",
             selection_text(summary.get("selection")),
             f"- 检查点：{summary.get('final_checkpoint')}（{summary.get('final_checkpoint_bytes')} 字节；新 LoRA、网格 token 行、"
             f"manifest；起点的引用在 {STK.STACK_FILE}）", "",
             "## 文件", "",
             "- train_log.jsonl：每步的确定性记录；logs/：每次调用的记录与逐步计时；accounting.json、compute.json：记账；",
             "- config.yaml：运行身份；meta.json：git、主机、包版本；tokens_init.json：开训时的 token 表（含行哈希）。", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 正式运行
# ---------------------------------------------------------------------------


def run_e5_training(adapter: Any, cfg: E5TrainConfig, spec: RunSpec, run_cfg: Mapping[str, Any], out_dir: str | Path, *,
                    resume: bool = False, argv: Sequence[str] = (), ckpt_root: str | Path | None = None,
                    stop_after_steps: int | None = None, keep_optimizer_state: bool = False,
                    data_root: str | Path | None = None, device: str | None = None, dtype_name: str | None = None,
                    log: Callable[[str], None] | None = None, _fail_after_step: int | None = None,
                    _val_eval: Callable[[int], Mapping[str, Any]] | None = None) -> dict[str, Any] | None:
    """跑一次 E5 训练；返回 train_summary（stop_after_steps 到了时返回 None）。adapter 必须是刚挂好的 base。
    起点检查点的合并核对不过时抛 src/e3/merge_check.MergeCheckFailed（叶目录不建）。
    _val_eval（只给测试）：替换 val 选择的一次评测（完成的步数 → {score, by_task}），用来核对选择与存盘的流程。"""
    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    rc = dict(run_cfg)
    if int(rc["n_examples"]) != len(spec.trainset):
        raise ValueError(f"run_cfg 的 n_examples {rc['n_examples']} 与训练集 {len(spec.trainset)} 条不符")
    leaf = leaf_dir(out_dir, spec)
    ckpt = (Path(ckpt_root) / spec.start.kind / spec.model_key / spec.trainset.tag / leaf.name) if ckpt_root else leaf / "ckpt"
    latest = ckpt / LATEST
    exists = leaf.exists() and any(leaf.iterdir())
    if exists and not resume:
        raise FileExistsError(f"{leaf} 已存在且非空；续跑请加 --resume")
    if exists and (leaf / SUMMARY).exists():
        s = json.loads((leaf / SUMMARY).read_text(encoding="utf-8"))
        if s.get("complete"):
            old = yaml.safe_load((leaf / CONFIG).read_text(encoding="utf-8"))
            now = {"start": spec.start.identity(), "data": spec.trainset.identity(), "training": _identity(dict(rc)),
                   "seed": spec.seed, "config": cfg.identity(),
                   "seed_pairing": None if spec.seed_pairing is None else dict(spec.seed_pairing)}
            bad = sorted(k for k, v in now.items() if _identity({k: old.get(k)}) != _identity({k: v}))
            if bad:
                raise ValueError(f"{leaf} 已完成，但运行身份与本次不同（{bad}），拒绝混写")
            log(f"[E5 {spec.start.kind} {spec.model_key} {spec.trainset.tag} seed{spec.seed}] 已完成，跳过（{leaf / SUMMARY}）")
            return s

    # 开训前的两项核对（模块说明）
    dev_name = device or str(adapter.device)
    dt_name = dtype_name or str(adapter.dtype).removeprefix("torch.").replace("bfloat16", "bf16").replace("float32", "fp32")
    checks_dir = STK.start_checks_dir(out_dir, spec.start.kind, spec.model_key)
    start_check = None
    reference = None
    if spec.start.has_checkpoint:
        checks_dir.mkdir(parents=True, exist_ok=True)
        start_check = STK.start_merge_check(spec.start, cfg, model_key=spec.model_key, tiny=spec.tiny, device=dev_name,
                                            dtype=dt_name, out_dir=checks_dir, data_root=data_root, log=log)
        reference = STK.reference_fingerprint(spec.start, device=dev_name, dtype=adapter.dtype, out_dir=checks_dir, log=log)

    trainer = E5Trainer(adapter, cfg, rc, seed=spec.seed, start=spec.start)
    st = None
    if exists and (latest / RESUME_STATE).exists():
        trainer.setup_resume(latest / MODEL_SUB)
        st = torch.load(latest / RESUME_STATE, map_location="cpu", weights_only=False)
    else:
        trainer.setup_fresh()
    fp = trainer.stack_info["fingerprint"]
    if reference is not None and fp != reference["fingerprint"]:
        raise STK.StartError(f"合并起点 LoRA 之后的权重指纹 {fp[:16]}… 与 factory.load_trained(起点, merge=True) 的 "
                             f"{reference['fingerprint'][:16]}… 不同：叠加后的模型与检查点合并评测时的不一致")
    prompt = cfg.prompt_format()
    n = cfg.grid_n
    builder = ExampleBuilder(spec.trainset, tokens=trainer.e5, prompt=prompt, n=n, tiny=spec.tiny)
    bs = int(rc["batch_size"])
    schedule = Schedule(spec.trainset, seed=spec.seed, epochs=int(rc["epochs"]), batch_size=bs, total=int(rc["n_samples"]))
    want_total = int(rc["steps"]) * bs if rc.get("min_steps_binding") else int(rc["epochs"]) * len(spec.trainset)
    if schedule.total != want_total or -(-schedule.total // bs) != int(rc["steps"]):
        raise RuntimeError("日程的总条数与步数对不上 run_cfg")
    sel_cfg = rc.get("selection")
    selector = VS.ValSelector.build(sel_cfg, spec.trainset.tasks, tiny=spec.tiny) if sel_cfg else None  # DEV-0038（val_select）
    ident = run_identity(spec, cfg, rc, trainer, prompt, selector)
    ident_sha = identity_sha256(ident)
    if exists:
        if not (leaf / CONFIG).exists():
            raise FileExistsError(f"{leaf} 里没有 {CONFIG}，不能续跑")
        old = yaml.safe_load((leaf / CONFIG).read_text(encoding="utf-8"))
        if _identity(old) != _identity(ident):
            diff = sorted(k for k in set(_identity(old)) | set(_identity(ident)) if _identity(old).get(k) != _identity(ident).get(k))
            raise ValueError(f"{leaf}：续跑的运行身份与已有的不同（{diff}），拒绝混写")
    else:
        leaf.mkdir(parents=True, exist_ok=True)
        files = [REPO_ROOT / cfg.path if not Path(cfg.path).is_absolute() else Path(cfg.path),
                 REPO_ROOT / cfg.e3b.path if not Path(cfg.e3b.path).is_absolute() else Path(cfg.e3b.path)]
        write_run_metadata(leaf, config=ident, seed=spec.seed, data_files=files, repo_root=REPO_ROOT)
        (leaf / TOKENS_INIT).write_text(json.dumps(adapter.token_table(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    start = 0
    if st is not None:
        if st["identity_sha256"] != ident_sha:
            raise ValueError(f"{latest / RESUME_STATE} 的运行身份与本次不同，拒绝续跑")
        trainer.load_state_dict(st)
        start = int(st["step"])
        log(f"[E5 {spec.start.kind} {spec.model_key} {spec.trainset.tag} seed{spec.seed}] 从第 {start} 步续跑")
    tl = [r for r in _read_log(leaf / TRAIN_LOG) if r["step"] < start]
    _atomic_write_text(leaf / TRAIN_LOG, "".join(_jsonl(r) for r in tl))
    # val 选择（DEV-0038）：历史以续跑状态里的为准（第 s 步做完之后的评测记 step = s；保留 step ≤ start 的），val_selection.jsonl 按它重写；
    # 候选检查点在 <ckpt>/best_step<k>/，只留历史里最好的那一个（之后的、没被 latest 记下的作废）
    history: list[dict[str, Any]] = []
    if selector is not None:
        if st is not None:
            history = [dict(h) for h in (st.get("selection") or {}).get("history", [])]
            if any(int(h["step"]) > start for h in history):
                raise RuntimeError(f"{latest / RESUME_STATE} 的 val 历史有第 {start} 步之后的记录")
        _atomic_write_text(leaf / VS.VAL_LOG, "".join(_jsonl(r) for r in history))
        _prune_best(ckpt, VS.best_of(history))

    (leaf / "logs").mkdir(exist_ok=True)
    dev = torch.device(adapter.device)
    run_k = len(list((leaf / "logs").glob("run_*.json"))) + 1
    run_path = leaf / "logs" / f"run_{run_k:03d}.json"
    timing_path = leaf / "logs" / f"timing_run_{run_k:03d}.jsonl"
    stack_rec = {"start": spec.start.record(), "start_merge_check": start_check,
                 "reference_fingerprint": None if reference is None else
                 {k: reference[k] for k in ("fingerprint", "device_key", "manifest_sha256")},
                 "fingerprint": {"value": fp, "device_key": STK.device_key(adapter)}, "setup": trainer.stack_info}
    run_log: dict[str, Any] = {"argv": list(argv), "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                               "device": str(dev), "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
                               "ckpt_dir": str(ckpt), "start_step": start, "finished_utc": None, "audit": trainer.audit,
                               "stack": stack_rec}
    run_path.write_text(json.dumps(run_log, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    t_all = time.perf_counter()
    done_box = [0]

    def finish_log(status: str) -> None:
        _sync(dev)
        run_log.update(status=status, finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                       wall_s=round(time.perf_counter() - t_all, 3), steps_this_run=done_box[0])
        run_path.write_text(json.dumps(run_log, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        acc, comp = accounting_doc(leaf)
        (leaf / ACCOUNTING).write_text(json.dumps(acc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        write_compute_json(leaf / COMPUTE, comp)

    def save_latest(step_done: int) -> None:
        tmp = ckpt / (LATEST + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        adapter.save_trained(tmp / MODEL_SUB, extra=STK.marker_extra(spec.start))
        extra_state = {} if selector is None else {"selection": {"history": [dict(h) for h in history]}}
        torch.save({"identity_sha256": ident_sha, "step": step_done, **trainer.state_dict(), **extra_state}, tmp / RESUME_STATE)
        old = ckpt / (LATEST + ".old")
        if old.exists():
            shutil.rmtree(old)
        if latest.exists():
            latest.replace(old)
        tmp.replace(latest)
        if old.exists():
            shutil.rmtree(old)

    steps = int(rc["steps"])
    every = int(rc["ckpt_every"])
    workers = int(rc["data_workers"])

    def val_round(step_done: int) -> dict[str, Any]:
        """一次 val 评测（val_select 的模块说明）：eval 模式、no_grad，随机数状态前后不变（有没有评测，训练轨迹逐位相同）。"""
        t_v = time.perf_counter()
        devices = [dev] if dev.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            adapter.eval_mode()
            try:
                with torch.no_grad():
                    ev = dict(_val_eval(step_done)) if _val_eval is not None else \
                        selector.evaluate(adapter, trainer.e5, prompt, n, log=log)
            finally:
                adapter.train_mode()
        _sync(dev)
        run_log.setdefault("val_evals", []).append({"step": step_done, "wall_s": round(time.perf_counter() - t_v, 3)})
        return {"step": step_done, "score": float(ev["score"]), "by_task": ev["by_task"]}
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="e5-data") if workers > 0 else None

    def build_one(plan: Any) -> tuple[Any, dict[str, Any] | None]:
        try:
            return builder.build(plan), None
        except NoSupervision as e:
            return None, {"k": plan.k, "id": plan.sample_id, "reason": str(e)}

    def submit(step: int) -> list[Any]:
        plans = schedule.step_plans(step)
        return [pool.submit(build_one, p) for p in plans] if pool is not None else plans

    def collect(handle: list[Any]) -> tuple[list[Any], list[dict[str, Any]]]:
        res = [h.result() for h in handle] if pool is not None else [build_one(p) for p in handle]
        return [e for e, _ in res if e is not None], [s for _, s in res if s is not None]

    try:
        adapter.train_mode()
        done = 0
        pending = submit(start) if start < steps else None
        for step in range(start, steps):
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats(dev)
            t0 = time.perf_counter()
            examples, skipped = collect(pending)
            pending = submit(step + 1) if step + 1 < steps else None
            t_data = time.perf_counter() - t0
            rec, times, comps = trainer.optimize(examples, device=dev)
            step_s = time.perf_counter() - t0
            nn = len(examples)
            comps = [c.as_sample(wall_time_s=step_s / nn) for c in comps]
            if step == start:
                trainer.audit["first_step_examples"] = [{"id": e.plan.sample_id, "task": e.plan.task, "n_images": len(e.images),
                                                         "n_target": len(e.target_ids), "n_supervised": e.n_supervised}
                                                        for e in examples[:4]]
            ep = sorted({p.epoch for p in schedule.step_plans(step)})
            line = {"step": step, **rec, "epochs": ep, "n_skipped": len(skipped), "skipped": skipped}
            _append(leaf / TRAIN_LOG, [_jsonl(line)])
            _append(timing_path, [_jsonl({"step": step, "data_s": t_data, **times, "step_s": step_s,
                                          "peak_mem_bytes": int(torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else None,
                                          "compute": {k: v for k, v in merge(comps).to_json_dict().items()
                                                      if k not in ("schema_version", "per_sample_mean")}})])
            done += 1
            done_box[0] = done
            s1 = step + 1
            if s1 % 10 == 0 or s1 == steps:
                log(f"[E5 {spec.start.kind} {spec.model_key} {spec.trainset.tag} seed{spec.seed}] 第 {s1}/{steps} 步 "
                    f"loss {rec['loss']:.4f}（{step_s:.2f} s/步）")
            stop = stop_after_steps is not None and done >= stop_after_steps and s1 < steps
            evaluated = selector is not None and selector.is_eval_step(s1, steps)
            if evaluated:  # val 选择（DEV-0038）：评测 → 新的最好就先存候选 → latest 记下历史 → 删掉别的候选
                ev_rec = val_round(s1)
                prev = VS.best_of(history)
                history.append(ev_rec)
                best = VS.best_of(history)
                ev_rec["best_step"] = int(best["step"])
                _append(leaf / VS.VAL_LOG, [_jsonl(ev_rec)])
                if prev is None or int(best["step"]) != int(prev["step"]):
                    bdir = _best_dir(ckpt, s1)
                    if bdir.exists():
                        shutil.rmtree(bdir)
                    bdir.parent.mkdir(parents=True, exist_ok=True)
                    adapter.save_trained(bdir, extra=STK.marker_extra(spec.start))
                log(f"[E5 {spec.start.kind} {spec.model_key} {spec.trainset.tag} seed{spec.seed}] 第 {s1} 步 val 分数 "
                    f"{ev_rec['score']:.4f}（目前最好：第 {best['step']} 步 {best['score']:.4f}）")
            if (s1 % every == 0 and s1 < steps) or stop or (evaluated and s1 < steps):
                save_latest(s1)
            if evaluated:
                _prune_best(ckpt, VS.best_of(history))
            if _fail_after_step is not None and s1 == _fail_after_step:
                raise RuntimeError(f"测试注入的中断（第 {s1} 步之后）")
            if stop:
                raise StopRequested(f"本次跑了 {done} 步（到第 {s1}/{steps} 步），检查点已写")
        adapter.eval_mode()
        final_dir = ckpt / FINAL
        final = final_dir / MODEL_SUB
        if final_dir.exists():
            shutil.rmtree(final_dir)
        final_dir.mkdir(parents=True)
        selection = None
        if selector is None:
            adapter.save_trained(final, extra=STK.marker_extra(spec.start))
        else:  # val 选中的那一步（DEV-0038）：候选检查点原样拷成 final/model
            best = VS.best_of(history)
            if best is None or [int(h["step"]) for h in history] != selector.eval_steps(steps):
                raise RuntimeError(f"val 历史的步 {[h['step'] for h in history]} 不是应评的 {selector.eval_steps(steps)}")
            bdir = _best_dir(ckpt, int(best["step"]))
            if not (bdir / "manifest.json").is_file():
                raise RuntimeError(f"选中的第 {best['step']} 步的检查点不在 {bdir}")
            shutil.copytree(bdir, final)
            read_manifest(final)  # 逐个核对文件的 sha256 与字节数
            selection = {"rule": selector.rule, "rule_text": VS.RULE_TEXT, "split": VS.VAL_SPLIT, "selected_step": int(best["step"]),
                         "selected_score": float(best["score"]), "steps": steps, "val_every": selector.val_every,
                         "n_per_task": selector.n_per_task, "n_evals": len(history), "val_log": VS.VAL_LOG,
                         "history": [{"step": int(h["step"]), "score": float(h["score"]),
                                      "by_task": {t: v["h"] for t, v in h["by_task"].items()}} for h in history]}
        tlog = _read_log(leaf / TRAIN_LOG)
        stack_doc = {"model_dir": MODEL_SUB, "start": spec.start.record(), "fingerprint": stack_rec["fingerprint"],
                     "model": ident["model"], "seed": spec.seed, "tag": spec.trainset.tag, "data": spec.trainset.identity(),
                     "identity_sha256": ident_sha,
                     "config": cfg.identity(), "train_leaf": str(leaf), "tokens_e5": trainer.e5.layout(),
                     "steps": steps, "selection": selection, "seed_pairing": ident["seed_pairing"]}
        STK.write_stack_file(final_dir, stack_doc)
        _prune_best(ckpt, None)
        if not keep_optimizer_state and latest.exists():
            shutil.rmtree(latest)
        man = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
        summary = {"complete": True, "steps": steps, "epochs": int(rc["epochs"]), "min_steps": rc.get("min_steps"),
                   "n_samples": int(rc["n_samples"]), "selected_step": steps if selection is None else selection["selected_step"],
                   "selection": None if selection is None else {**selection, "history_full": history},
                   "start": spec.start.kind,
                   "model": spec.model_key, "tag": spec.trainset.tag, "seed": spec.seed, "identity_sha256": ident_sha,
                   "final_loss": tlog[-1]["loss"] if tlog else None,
                   "final_checkpoint": str(final_dir.relative_to(leaf)) if final_dir.is_relative_to(leaf) else str(final_dir),
                   "final_checkpoint_bytes": _dir_bytes(final_dir),
                   "final_checkpoint_files": {k: v["bytes"] for k, v in man["files"].items()},
                   "lora": man.get("lora"), "trainable_token_rows": man.get("trainable_token_rows"),
                   "start_merge_check": start_check, "stack_fingerprint": fp,
                   "n_skipped": sum(int(r.get("n_skipped", 0)) for r in tlog)}
        (leaf / README).write_text(readme_text(spec, ident, summary, tlog), encoding="utf-8")
        _atomic_write_text(leaf / SUMMARY, json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
        finish_log("complete")
        return summary
    except StopRequested as e:
        log(f"[停止] {e}")
        adapter.eval_mode()
        finish_log("stopped")
        return None
    except BaseException:
        finish_log("error")
        raise
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)


BEST_PREFIX = "best_step"


def _best_dir(ckpt: Path, step: int) -> Path:
    """val 选择的候选检查点：<ckpt>/best_step<步，6 位>/（save_trained 的目录）。"""
    return ckpt / f"{BEST_PREFIX}{int(step):06d}"


def _prune_best(ckpt: Path, keep: Mapping[str, Any] | None) -> None:
    """删掉 <ckpt>/best_step* 里除 keep（历史里最好的那一条）以外的候选；keep 为 None 时全删。"""
    if not ckpt.exists():
        return
    want = None if keep is None else _best_dir(ckpt, int(keep["step"])).name
    for d in ckpt.glob(f"{BEST_PREFIX}*"):
        if d.name != want:
            shutil.rmtree(d)
    if want is not None and not (ckpt / want / "manifest.json").is_file():
        raise RuntimeError(f"val 历史里最好的第 {keep['step']} 步的候选检查点不在 {ckpt / want}")


__all__ = ["BEST_PREFIX", "E5Trainer", "RunSpec", "TRAIN_SUB", "leaf_dir", "run_e5_training", "run_identity", "seed_dir_name"]
