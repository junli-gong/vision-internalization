"""E5 训练后系统的评测（DEV-0037 §4）：一个 E5 检查点 × 一个任务 → <out>/trained/<起点>/<模型>/<n 标记>/seed<k>/<任务>[__dev]/。

**加载**（scripts/eval_e5_trained.py；src/e5/stack.load_stacked）：按 E5 检查点 manifest 的构造方式建 base → 合并起点 LoRA → 核对起点
manifest 的 sha256 与权重指纹 → load_trained(E5 检查点, merge=False)。

**评测前的合并核对**（stacked_merge_check；判据 v1 = D50-3，scripts/check_training_interfaces.py 的同一套判定函数 judge、criteria_block、
grid_criteria_block 与累计器 GenAgg、TfAgg）：起点 LoRA 合并之后，E5 LoRA 合并前（A）对合并后（B），在原地合并（同一个适配器，不加载两份）：
- 样本：E5 val 划分每类 n_per_task 条（configs/e5_train.json 的 eval.merge_check；序号规则同评测集，src/e5/protocol.eval_items 的 dev），
  E5 训练后的提示与允许集合；按格式（二值 c、d、e / 起伏 a、b）分组，组内按 merge_check.batch_size 成批（同一批同一个允许集合），关思考。
- A（未合并）：受限解码 generate_restricted（评测实际走的路径）+ 以它为目标的教师强制；merge_lora；B（合并后）：
  training.restricted_consistency（generate_restricted 对 generate_batch(allowed) 对以 B 的受限解码为目标的教师强制）+ 以 A 的序列为目标的
  教师强制。每批 3 次整段生成。比较（checkpoint.compare_merge_sides，两侧的生成都是 generate_restricted、教师强制的目标都是 A 的序列）：
  * merge：受限解码逐位一致率（只报不判；CPU fp32 时要逐个相同）与教师强制在全词表上的 argmax 一致率、不一致处的 logit 差（v1 的 merge 一节）；
  * restricted.across_merge：同一批隐藏状态只在允许集合上比较（v1 的 merge 一节，同 E3b 带网格 token 的检查点）；
  * restricted.by_thinking：B 上 restricted_consistency 的读数（v1 的 restricted 一节）。
  与 E3b 的检查点核对（脚本）的差别：across_merge 的公共目标取合并前的序列（脚本取合并后受限解码的序列；原地合并之后合并前的模型已不在）。
- 输出 <模型目录>/merge_check.json；不通过（或出错）抛 src/e3/merge_check.MergeCheckFailed，调用方拒绝评测（不写任何任务目录）。同一个
  E5 检查点、同一起点、同一组设置已通过的结果直接复用（只做 merge_lora）。

**单轮**（唯一的流程；多轮在 E5 不做）：样本 = src/e5/protocol.eval_items(任务, 子集, 每档条数)（与零样本完全相同的清单）；提示 = 指令原文 +
E3b 网格指令的结尾句（src/e5/token_grid.PromptFormat），指令在前；adapter.generate_restricted(允许集合 = E5Tokens.allowed_ids(任务)，
max_new_tokens = n(n + 1) + 1 = 1057，停止 token = 本家族的回合结束 token，关思考，贪心)；按样本顺序连续成批（批大小进运行身份）。
解析与评分：token 序列 → E5 的格子表示 → src/e5/scoring.score（与零样本同一套评分）；汇总 scoring.aggregate、门槛 scoring.gate（三条基线的
逐样本配对 bootstrap CI，D59-4）；标题口径与判定的统计范围就是 src/e5/protocol.in_headline（与零样本同一个函数；D72-1 / D73：b 统计
环绕的 [20,180] 三档），这里不另写。

**输出**：predictions.jsonl（每条：样本字段、提示、停止、记账字段、预测的网格 / 起伏文本、解析失败原因（失败时另存 token id）、逐条得分；不含
墙钟）、metrics.json（与零样本同一结构：headline / overall / by_bin / by_group / parse / gate / baselines_headline）、compute.json、
config.yaml（运行身份）、meta.json、README.md、logs/timing.jsonl、logs/run_<k>.json。
**续跑**（--resume）：config.yaml 必须相同；已完成的 id 必须是样本顺序的前缀，退回到整批的边界接着跑；metrics 只由 predictions 与生成器算出，
续跑后逐字节相同。
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from src.accounting import Compute, merge, write_compute_json
from src.constants import BOOTSTRAP_SEED
from src.e3.merge_check import MergeCheckFailed, load_cti
from src.e5 import scoring as SC
from src.e5.protocol import TASK_LETTER, eval_items, headline_scope_text, num_bins
from src.e5.runner import check_platform, make_sample, read_jsonl_tolerant
from src.e5.token_grid import E5Tokens, PromptFormat, max_new_tokens, sample_images, task_format
from src.e5.train_config import E5TrainConfig
from src.runlog import write_run_metadata
from src.synth.seeding import canonical_task

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_SUB = "trained"
PREDICTIONS = "predictions.jsonl"
METRICS = "metrics.json"
COMPUTE = "compute.json"
CONFIG = "config.yaml"
README = "README.md"
TIMING = "logs/timing.jsonl"
MERGE_CHECK = "merge_check.json"
MERGE_CHECK_FORMAT = "visint-e5-stacked-merge-check/1"
SYSTEM = "trained"


def model_dir(out: str | Path, start: str, model_key: str, tag: str, seed: int, data_seed: int | None = None) -> Path:
    """<out>/trained/<起点>/<模型>/<n 标记>/seed<k>（数据种子不等于种子时 seed<k>_d<数据种子>，同训练的叶目录，src/e5/train.seed_dir_name）。"""
    from src.e5.train import seed_dir_name

    return Path(out) / EVAL_SUB / start / model_key / tag / seed_dir_name(seed, seed if data_seed is None else data_seed)


@dataclass(frozen=True)
class EvalSpec:
    start: str
    model_key: str
    tag: str
    seed: int
    task: str
    subset: str
    n_per_bin: int
    n_resamples: int
    batch_size: int
    tiny: bool
    dtype: str
    limit: int | None = None
    data_seed: int | None = None

    def items(self) -> list[Any]:
        its = eval_items(self.task, self.subset, self.n_per_bin)
        return its if self.limit is None else its[: self.limit]

    def leaf(self, out: str | Path) -> Path:
        name = canonical_task(self.task) + ("" if self.subset == "eval" else f"__{self.subset}")
        return model_dir(out, self.start, self.model_key, self.tag, self.seed, self.data_seed) / name


# ---------------------------------------------------------------------------
# 评测前的合并核对（模块说明）
# ---------------------------------------------------------------------------


def merge_check_items(n_per_task: int, tasks: Sequence[str]) -> list[Any]:
    """val 划分每类前 n_per_task 条（eval_items 的 dev 清单，档轮转）。"""
    out = []
    for t in tasks:
        k = math.ceil(n_per_task / num_bins(t))
        out += eval_items(t, "dev", k)[:n_per_task]
    return out


def _reusable(doc: Mapping[str, Any], key: Mapping[str, Any], criteria_version: str) -> bool:
    try:
        return bool(doc.get("format") == MERGE_CHECK_FORMAT and doc.get("pass") is True and not doc.get("error")
                    and doc["key"] == dict(key) and doc["criteria_version"] == criteria_version)
    except (KeyError, TypeError):
        return False


def _summary(doc: Mapping[str, Any], path: Path, *, reused: bool) -> dict[str, Any]:
    import hashlib

    return {"file": str(path), "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "pass": bool(doc.get("pass")),
            "criteria_version": doc.get("criteria_version"), "n": (doc.get("data") or {}).get("n"),
            "checks": sorted(doc.get("checks") or {}), "reused": reused, "failures": doc.get("failures"),
            "error": (doc.get("error") or {}).get("repr") if doc.get("error") else None}


def stacked_merge_check(ad: Any, e5tok: E5Tokens, prompt: PromptFormat, *, n: int, n_per_task: int, tasks: Sequence[str],
                        batch_size: int, tiny: bool, exact: bool, key: Mapping[str, Any], out_json: str | Path,
                        enable_thinking: bool = False, log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """起点合并之后的 E5 LoRA 合并核对（模块说明）。ad 必须挂着未合并的 E5 LoRA；返回时已合并。通过返回摘要，否则抛 MergeCheckFailed。"""
    import traceback

    import torch

    from src.adapters import checkpoint as CK
    from src.adapters import training

    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    cti = load_cti()
    out = Path(out_json)
    if out.exists():
        try:
            old = json.loads(out.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            old = {}
        if _reusable(old, key, cti.CRITERIA_VERSION):
            log(f"[合并核对] 复用已通过的 {out}")
            ad.merge_lora()
            return _summary(old, out, reused=True)
    if ad.lora_state != "injected":
        raise RuntimeError(f"合并核对要挂着未合并的 E5 LoRA（当前 {ad.lora_state}）")
    mode = cti.MODE_KEYS[bool(enable_thinking)]
    mnt = max_new_tokens(n)
    doc: dict[str, Any] = {"format": MERGE_CHECK_FORMAT, "written_utc": None, "key": dict(key),
                           "criteria_version": cti.CRITERIA_VERSION, "criteria": None, "checks": {}, "failures": {},
                           "pass": None, "data": None, "generation": {"max_new_tokens": mnt, "enable_thinking": bool(enable_thinking),
                                                                    "batch_size": int(batch_size)},
                           "timing_s": {}, "error": None,
                           "rule": "src/e5/trained_eval.py 的模块说明：起点 LoRA 合并之后，E5 LoRA 合并前（A）对合并后（B），原地合并；"
                                   "判据 v1（D50-3）= scripts/check_training_interfaces.py 的 judge"}
    t0 = time.perf_counter()
    try:
        for t in tasks:
            check_platform(t, tiny)
        items = merge_check_items(n_per_task, tasks)
        samples = [make_sample(it) for it in items]
        runs = []  # (inputs, ids, 格式)
        for fmt in ("binary", "relief"):
            group = [s for s in samples if task_format(s.task) == fmt]
            for lo in range(0, len(group), int(batch_size)):
                chunk = group[lo: lo + int(batch_size)]
                runs.append(([ad.build_inputs(sample_images(s), prompt.render(s, n), prompt.order) for s in chunk],
                             [f"{s.task}/{s.split}/{s.index}" for s in chunk], fmt))
        doc["data"] = {"subset": "dev", "split": "val", "n_per_task": int(n_per_task), "tasks": list(tasks), "n": len(samples),
                       "sample_ids": [f"{s.task}/{s.split}/{s.index}" for s in samples], "n_runs": len(runs)}
        th = bool(enable_thinking)
        sides_a = []
        for inp, _, fmt in runs:  # A：未合并，受限解码 + 教师强制
            res = ad.generate_restricted(list(inp), e5tok.allowed_ids_of(fmt), max_new_tokens=mnt, stop_ids=[e5tok.stop_id],
                                         enable_thinking=th)
            tgt = [list(r.token_ids) for r in res]
            with torch.no_grad():
                tf = ad.training_forward(list(inp), tgt, enable_thinking=th)
            sides_a.append(CK.MergeSide(generations=[list(res)], targets=[tgt], hidden=[tf.target_hidden.detach()],
                                        offsets=[tuple(tf.target_offsets)]))
            del tf
        doc["timing_s"]["unmerged_s"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        ad.merge_lora()
        gen = cti.GenAgg(mnt, ("before", "after"))
        tf_full = cti.TfAgg(("before", "after"), "argmax_agreement")
        tf_allowed = cti.TfAgg(("unmerged", "merged"), "argmax_agreement")
        rgen = cti.GenAgg(mnt, ("generate_batch", "restricted"))
        rtf = cti.TfAgg(("decoded", "argmax"), "agreement")
        for (inp, ids, fmt), a in zip(runs, sides_a):  # B：合并后
            allowed = e5tok.allowed_ids_of(fmt)
            rec = cti.RecordingAdapter(ad)
            rep = training.restricted_consistency(rec, inp, allowed, max_new_tokens=mnt, stop_ids=[e5tok.stop_id],
                                                  enable_thinking=th)
            rgen.add(rep["decode_vs_generate_batch"], ids, th)
            tf = rep["teacher_forcing"]
            dis = [(b_, p, x, y, g, t2_) for (b_, p, x, y, g), t2_ in zip(tf["disagreements"], tf["disagreement_top2_gaps"])]
            rtf.add(tf["n_tokens"], dis, ids, th)
            if len(rec.restricted) != 1:
                raise RuntimeError(f"restricted_consistency 应调一次 generate_restricted，实际 {len(rec.restricted)} 次")
            on_a = CK.merge_side(ad, [(inp, th)], max_new_tokens=mnt, targets=a.targets, generate=False)
            b = CK.MergeSide(generations=[list(rec.restricted[0])], targets=a.targets, hidden=on_a.hidden, offsets=on_a.offsets)
            (rep,) = CK.compare_merge_sides(ad, a, b, allowed_ids=allowed, chunk_size=cti.TF_CHUNK)
            gen.add(rep["generate"], ids, th)
            tf = rep["teacher_forcing"]
            tf_allowed.add(tf["n_tokens"], tf["disagreements"], ids, th, tf["max_abs_logit"], tf["max_abs_hidden"])
            (repf,) = CK.compare_merge_sides(ad, a, b, allowed_ids=None, chunk_size=cti.TF_CHUNK)
            tf = repf["teacher_forcing"]
            tf_full.add(tf["n_tokens"], tf["disagreements"], ids, th, tf["max_abs_logit"], tf["max_abs_hidden"])
            del a, b, on_a, rec
        doc["timing_s"]["merged_s"] = time.perf_counter() - t1
        n_rows = len(samples)
        doc["checks"] = {
            "merge": {"mode": "e5_stacked", "compared": "E5 LoRA 未合并（起点已合并）vs 合并后；生成 = 两侧各自的 generate_restricted，"
                                                        "教师强制的目标 = 合并前受限解码的序列，logits 在全词表上比较",
                      "by_thinking": {mode: {"n_rows": n_rows, "generate": gen.summary(), "teacher_forcing": tf_full.summary()}}},
            "restricted": {"on_model": "合并后（起点 LoRA 与 E5 LoRA 都已合并）",
                           "by_thinking": {mode: {"n_rows": n_rows, "decode_vs_generate_batch": rgen.summary(),
                                                  "teacher_forcing": rtf.summary()}},
                           "across_merge": {"compared": "同 merge 的两侧隐藏状态，只在允许集合上比较 logits",
                                            "by_thinking": {mode: {"n_rows": n_rows, "teacher_forcing": tf_allowed.summary()}}}},
        }
        crit = cti.criteria_block(exact)
        crit["restricted_grid"] = cti.grid_criteria_block(exact)
        doc["criteria"] = crit
        doc["failures"] = cti.judge(doc["checks"], crit)
        for name, bad in doc["failures"].items():
            doc["checks"][name]["pass"] = not bad
        doc["pass"] = all(not bad for bad in doc["failures"].values())
        doc["allowed"] = e5tok.layout()
    except Exception as e:  # noqa: BLE001 — 证据先落盘
        doc["error"] = {"repr": repr(e), "traceback": traceback.format_exc()[-6000:]}
        doc["pass"] = False
    doc["timing_s"]["total_s"] = time.perf_counter() - t0
    doc["written_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    summ = _summary(doc, out, reused=False)
    if not doc["pass"]:
        raise MergeCheckFailed(f"评测前的合并核对没有通过：{summ['failures'] or summ['error']}；拒绝评测（{out}）", summ)
    return summ


# ---------------------------------------------------------------------------
# 单轮评测
# ---------------------------------------------------------------------------


def ids_digest(ids: Sequence[str]) -> str:
    import hashlib

    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def run_identity(spec: EvalSpec, items: Sequence[Any], e5tok: E5Tokens, prompt: PromptFormat, cfg: E5TrainConfig,
                 checkpoint: Mapping[str, Any], n: int) -> dict[str, Any]:
    t = canonical_task(spec.task)
    ids = [it.id for it in items]
    from src.synth.constants import SYNTH_GENERATOR_VERSION, TASK_GENERATOR_VERSIONS

    return {
        "experiment": "E5",
        "framework": "src/e5（DEV-0037：训练后系统的评测）",
        "system": SYSTEM,
        "task": t,
        "task_letter": TASK_LETTER[t],
        "subset": spec.subset,
        "data": {"generator": "src.synth.generate", "split": "test" if spec.subset == "eval" else "val",
                 "n_per_bin": spec.n_per_bin, "n_samples": len(ids), "sample_ids_sha256": ids_digest(ids), "limit": spec.limit,
                 "index_rule": "序号 0..B·n−1，档 = 序号 mod B（src/e5/protocol.py；与零样本相同）",
                 "task_generator_version": TASK_GENERATOR_VERSIONS[t], "synth_generator_version": SYNTH_GENERATOR_VERSION},
        "n_resamples": spec.n_resamples,
        "tiny": spec.tiny,
        "dtype": spec.dtype,
        "headline_scope": headline_scope_text(t),
        "gate": {"rule": "D59-4：配对 bootstrap 95% CI，判定条件同时胜过 " + "、".join(SC.JUDGMENT_BASELINES)},
        "checkpoint": dict(checkpoint),
        "decoding": {"path": "generate_restricted", "allowed_ids": e5tok.allowed_ids(t), "value_ids": list(e5tok.value_ids(t)),
                     "format": task_format(t), "max_new_tokens": max_new_tokens(n), "stop_ids": [e5tok.stop_id],
                     "enable_thinking": False, "greedy": True, "grid_size": int(n)},
        "batching": {"batch_size": spec.batch_size, "order": "样本顺序，连续成批，左填充（适配器）"},
        "prompt": prompt.as_dict(),
        "config": cfg.identity(),
    }


def _prepare(leaf: Path, ident: Mapping[str, Any], resume: bool, cfg: E5TrainConfig) -> None:
    def norm(x: Mapping[str, Any]) -> dict[str, Any]:
        return yaml.safe_load(yaml.safe_dump(dict(x), sort_keys=False, allow_unicode=True))

    if leaf.exists() and any(leaf.iterdir()):
        if not resume:
            raise FileExistsError(f"{leaf} 已存在且非空；续跑请加 --resume")
        if not (leaf / CONFIG).exists():
            raise FileExistsError(f"{leaf} 里没有 {CONFIG}，不能续跑")
        old, new = norm(yaml.safe_load((leaf / CONFIG).read_text(encoding="utf-8"))), norm(ident)
        if old != new:
            diff = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
            raise ValueError(f"{leaf}：续跑的配置与已有的不同（{diff}），拒绝混写")
    else:
        leaf.mkdir(parents=True, exist_ok=True)
        files = [REPO_ROOT / cfg.path if not Path(cfg.path).is_absolute() else Path(cfg.path)]
        write_run_metadata(leaf, config=dict(ident), seed=BOOTSTRAP_SEED, data_files=files, repo_root=REPO_ROOT)
    (leaf / "logs").mkdir(exist_ok=True)


def _append(path: Path, lines: Sequence[str]) -> None:
    if not lines:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write("".join(lines))
        f.flush()
        os.fsync(f.fileno())


def _truncate_lines(path: Path, k: int) -> None:
    raw = path.read_bytes()
    pos = 0
    for _ in range(k):
        pos = raw.index(b"\n", pos) + 1
    with open(path, "r+b") as f:
        f.truncate(pos)


def _jsonl(rec: Mapping[str, Any]) -> str:
    return json.dumps(rec, ensure_ascii=False) + "\n"


def record(sample: Any, prompt_text: str, res: Any, e5tok: E5Tokens, n: int) -> dict[str, Any]:
    """一条生成 → predictions.jsonl 的一行（与零样本的记录同一套字段：样本字段、记账、解析、得分）。"""
    h, w = sample.target_mask.shape
    pred = e5tok.pred(sample.task, res.token_ids, n, h, w)
    rec = SC.sample_fields(sample)
    rec.update({"prompt": prompt_text, "stopped": res.stopped, "truncated": res.stopped is False, "thinking_unclosed": False,
                "n_generated_tokens": int(res.n_generated_tokens), "n_prompt_tokens": int(res.n_prompt_tokens),
                "n_visual_tokens": int(res.n_visual_tokens), "n_image_pixels": int(res.n_image_pixels or 0),
                "valid": pred.valid, "failure": pred.failure, "pred": SC.pred_record(pred, store_rle=False),
                "score": SC.score(sample, pred, relief_expected=task_format(sample.task) == "relief")})
    if pred.failure is not None:
        rec["token_ids"] = [int(t) for t in res.token_ids]
    return rec


def readme_text(spec: EvalSpec, ident: Mapping[str, Any], m: Mapping[str, Any]) -> str:
    from src.e5.runner import _block_lines, _ci_text, _fmt

    ck = ident["checkpoint"]
    lines = [f"# E5 · 训练后（{spec.start} → E5）· {spec.model_key} · {spec.tag} · seed {spec.seed} · {m['task']}"
             f"（{TASK_LETTER[m['task']]}）· {spec.subset}", "",
             f"- 样本：{ident['data']['split']} 划分，每档 {spec.n_per_bin} 条，n = {m['n']}（id 指纹 "
             f"{ident['data']['sample_ids_sha256'][:16]}…，与零样本相同的清单）；生成器任务版本 {ident['data']['task_generator_version']}",
             f"- 检查点：E5 manifest {str(ck.get('manifest_sha256'))[:16]}…；起点 {ck.get('start', {}).get('kind')}（manifest "
             f"{str(ck.get('start', {}).get('manifest_sha256'))[:16]}…）；训练运行身份 {str(ck.get('train_identity_sha256'))[:16]}…",
             f"- 解码：受限（{ident['decoding']['format']}，{len(ident['decoding']['allowed_ids'])} 个允许 token），单轮，"
             f"max_new_tokens = {ident['decoding']['max_new_tokens']}，关思考，贪心，批大小 {spec.batch_size}",
             f"- 标题口径：{m['headline_scope']}", "", "## 标题（严格 / 有效率 / 条件读数）", ""]
    lines += _block_lines(m["headline"])
    g = m.get("gate") or {}
    pre, jud = g.get("prerequisite"), g.get("judgment")
    lines += ["", "## 两级门槛（D59-4：逐样本配对 bootstrap 95% CI）", ""]
    if pre:
        lines.append(f"- 先决条件（{g['iou_key']}）：系统 {_fmt(pre['system'])} vs 最强非真值基线 {pre['strongest']} "
                     f"{_fmt(pre['value'])}；差 {_fmt(pre['diff'])}，CI {_ci_text(pre['diff_ci'])} → pass={pre['pass']}")
    if jud:
        lines.append(f"- 判定条件：{jud['status']}" + (f"，系统起伏误差 {_fmt(jud.get('system'))}，n = {jud.get('n')} → "
                                                     f"pass={jud.get('pass')}" if jud["status"] == "evaluated" else ""))
        for name, c in (jud.get("baselines") or {}).items():
            lines.append(f"  - vs {name}：基线 {_fmt(c['value'])}；差 {_fmt(c['diff'])}，CI {_ci_text(c['diff_ci'])} → 胜过={c['beats']}")
    lines.append(f"- imagining：{g.get('imagining')}")
    p = m["parse"]
    lines += ["", "## 解析", "", f"- 解析失败率：{_fmt(p['failure_rate'])}（{p['n_failed']}/{p['n']}）；分原因：{p['failures']}",
              f"- 截断率：{_fmt((m.get('truncation') or {}).get('rate'))}", "",
              "## 文件", "", "- predictions.jsonl、metrics.json（含每档、分列与门槛）、compute.json、config.yaml、meta.json、logs/", ""]
    return "\n".join(lines)


def finalize(leaf: Path, spec: EvalSpec, items: Sequence[Any], ident: Mapping[str, Any]) -> dict[str, Any]:
    recs = read_jsonl_tolerant(leaf / PREDICTIONS, repair=False)
    ids = [it.id for it in items]
    if [r["id"] for r in recs] != ids[: len(recs)]:
        raise RuntimeError(f"{leaf / PREDICTIONS} 的 id 不是样本顺序的前缀")
    samples = [make_sample(it) for it in items[: len(recs)]]
    body = SC.aggregate(spec.task, recs, n_resamples=spec.n_resamples, enable_thinking=False)
    base_recs, roles = SC.baseline_records(samples)
    body["gate"] = SC.gate(spec.task, recs, base_recs, roles, n_resamples=spec.n_resamples)
    body["baselines_headline"] = {nm: SC.blocks([r for r in v if r["headline"]], 0) for nm, v in sorted(base_recs.items())}
    body["baseline_roles"] = dict(sorted(roles.items()))
    body["complete"] = len(recs) == len(ids)
    body["n_expected"] = len(ids)
    metrics = {"system": SYSTEM, "start": spec.start, "model": spec.model_key, "tag": spec.tag, "seed": spec.seed,
               "subset": spec.subset, **body}
    (leaf / METRICS).write_text(json.dumps(metrics, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    timing = {t["id"]: t for t in read_jsonl_tolerant(leaf / TIMING, repair=False)}
    comp = []
    for r in recs:
        wall = float(timing.get(r["id"], {}).get("wall_time_s", 0.0))
        c = Compute.call(generated_tokens=r["n_generated_tokens"], prompt_tokens=r["n_prompt_tokens"],
                         visual_tokens=r["n_visual_tokens"], image_pixels=r["n_image_pixels"], call_time_s=wall, new_round=True)
        comp.append(c.as_sample(wall_time_s=wall))
    write_compute_json(leaf / COMPUTE, merge(comp))
    (leaf / README).write_text(readme_text(spec, ident, metrics), encoding="utf-8")
    return metrics


def run_task(ad: Any, e5tok: E5Tokens, prompt: PromptFormat, cfg: E5TrainConfig, spec: EvalSpec, out: str | Path, *,
             checkpoint: Mapping[str, Any], n: int, resume: bool = False, argv: Sequence[str] = (),
             run_extra: Mapping[str, Any] | None = None, log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """一个任务的单轮评测（模块说明）。ad 必须是合并后的模型。"""
    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    if ad.lora_state not in ("merged", "none"):
        raise RuntimeError(f"评测要在合并之后（当前 LoRA 状态 {ad.lora_state}）")
    t = canonical_task(spec.task)
    check_platform(t, spec.tiny)
    items = spec.items()
    leaf = spec.leaf(out)
    ident = run_identity(spec, items, e5tok, prompt, cfg, checkpoint, n)
    _prepare(leaf, ident, resume, cfg)
    path = leaf / PREDICTIONS
    done = read_jsonl_tolerant(path, repair=True)
    ids = [it.id for it in items]
    if [r["id"] for r in done] != ids[: len(done)]:
        raise RuntimeError(f"{path} 里已完成的 id 不是样本顺序的前缀，不能续跑")
    bs = int(spec.batch_size)
    keep = len(done) if len(done) == len(ids) else (len(done) // bs) * bs
    if keep < len(done):
        _truncate_lines(path, keep)
        log(f"[{t}] 已完成 {len(done)} 条不在批的边界上，退回到 {keep} 条整批重跑")
    run_k = len(list((leaf / "logs").glob("run_*.json"))) + 1
    run_path = leaf / "logs" / f"run_{run_k:03d}.json"
    run_log = {"argv": list(argv), "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               "n_done_before": keep, "n_remaining": len(ids) - keep, "batch_size": bs, "finished_utc": None, "run": run_k,
               **dict(run_extra or {})}
    run_path.write_text(json.dumps(run_log, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    t_all = time.perf_counter()
    allowed = e5tok.allowed_ids(t)
    mnt = max_new_tokens(n)
    n_batches = 0
    for lo in range(keep, len(items), bs):
        batch = [make_sample(it) for it in items[lo: lo + bs]]
        t0 = time.perf_counter()
        texts = [prompt.render(s, n) for s in batch]
        inputs = [ad.build_inputs(sample_images(s), txt, prompt.order) for s, txt in zip(batch, texts)]
        prep = time.perf_counter() - t0
        results = ad.generate_restricted(inputs, allowed, max_new_tokens=mnt, stop_ids=[e5tok.stop_id], enable_thinking=False)
        if len(results) != len(batch):
            raise RuntimeError("generate_restricted 返回的行数与输入不同")
        lines, tlines = [], []
        for s, txt, res in zip(batch, texts, results):
            rec = record(s, txt, res, e5tok, n)
            lines.append(_jsonl(rec))
            tlines.append(_jsonl({"id": rec["id"], "run": run_k, "batch_index": n_batches, "batch_rows": len(batch),
                                  "wall_time_s": float(res.wall_time_s), "batch_wall_time_s": float(res.batch_wall_time_s or 0.0),
                                  "prep_s": prep / len(batch)}))
        _append(leaf / TIMING, tlines)
        _append(path, lines)
        n_batches += 1
        hi = lo + len(batch)
        if n_batches == 1 or n_batches % 20 == 0 or hi == len(ids):
            log(f"[E5 trained {spec.start} {spec.model_key} {spec.tag} seed{spec.seed} {t}/{spec.subset}] {hi}/{len(ids)}")
    metrics = finalize(leaf, spec, items, ident)
    run_log.update(finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                   wall_s=round(time.perf_counter() - t_all, 3), n_batches=n_batches)
    run_path.write_text(json.dumps(run_log, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return metrics


__all__ = ["EVAL_SUB", "EvalSpec", "MERGE_CHECK", "MERGE_CHECK_FORMAT", "finalize", "merge_check_items", "model_dir", "record",
           "run_task", "stacked_merge_check"]
