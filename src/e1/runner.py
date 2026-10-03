"""E1 的运行器：一个模型 × 一个任务 × 一个子集 → 输出目录（DEV-0014 任务书 §2；简报 §11.1；协议 v2 见 DEV-0017）。

流程：加载器选样本（src/e1/tasks.select）→ 按提示配置组装输入（build_inputs，图在前）→ generate（batch = 1）或 generate_batch
→ 解析 → 评分（src/e1/scoring）。每批写完才落盘：先写 logs/timing.jsonl，再写 predictions.jsonl（flush + fsync）。

思考子集（think；D59-1，DEV-0027）：生成走 adapter.generate_batch_budgeted（预算强制收尾：思考 thinking_budget、作答 answer_budget，
取自配置的 generation.think_budget）；解析照旧读 text。predictions.jsonl 另记思考段的结束方式（thinking_end：natural、forced、
not_opened、unclosed）、各段 token 数、补的 token 与 answer_text；metrics.json 按结束方式分开报（scoring.aggregate 的 thinking_end）；
compute.json 的 prompt_tokens 含补的 token（它们是送进模型的输入，不是模型生成的）。其余子集的记录与指标与原来逐字节相同。

读答案的两种方式（prompts.readout；DEV-0017 §1）：
- text：自由生成后按任务条目的 parse 解析（原来的路径）；
- letter：受限字母读数（选择题与点对任务的 eval、dev）：只生成 1 个 token，allowed_token_ids = 本题选项字母的全部写法
  （src/e1/letters.py 的字母表，写进 config.yaml），生成的 token 按表映射回字母（scoring.score_letter），不算截断。

批（configs/e1_prompts.json 的 batch_protocol；D36-1）：batch_plan 在整份样本上一次划好——按样本顺序连续取，每批至多
batch_size 条；受限字母读数时字母集合一变就另起一批（generate_batch 的 allowed_token_ids 对整批相同）。

输出（<out>/<模型>/<任务>[__<子集>][__<口径>]/）：
- predictions.jsonl：每条样本一行（id、instruction、raw_text、text、读法、解析结果、失败原因、是否截断、思考段是否未闭合、
  生成 token 数、视觉 token 数与像素数、逐条得分、坐标诊断）。不含墙钟，所以与续跑无关、逐字节可复现。
- metrics.json：只由 predictions.jsonl 与样本 GT 算出（scoring.aggregate），续跑后再算一遍逐字节相同。
- compute.json：src/accounting.py（schema 3）；墙钟来自 logs/timing.jsonl（同一 id 取最后一条）。
- config.yaml（运行的身份：模型、revision、口径、任务、子集、提示配置 sha256 与本家族的提示原文、读法与字母表、批配置、样本
  id 指纹……；不含时间戳与 git 状态）与 meta.json（src/runlog.write_run_metadata：git、主机、包版本、数据清单的 sha256、时间）。
- README.md：由 metrics.json 生成（含三列与拒答按人物分列的表）。logs/：timing.jsonl、每次调用一份 run_<k>.json。

训练后的模型（DEV-0030 §2，run_e1 --checkpoint）：适配器换成 factory.load_trained(merge=True) 的结果，协议不变（冻结口径、批配置、
三列）。RunSpec.checkpoint（manifest 的 sha256 与摘要）写进 config.yaml 的 checkpoint；适配器登记了网格 token 时，predictions 每条另记
n_grid_tokens（生成的 token 里网格 token 的个数），metrics.json 的 diagnostics 另有 grid_tokens_in_output。base 运行两者都没有，
输出与原来逐字节相同。

断点续传（--resume）：目录里已有 predictions.jsonl 时必须给 --resume，并且 config.yaml 与本次完全相同（DEV-0017 起批大小也是
运行身份的一部分），否则报错；末尾不完整的一行截掉；已完成的 id 必须是样本顺序的前缀。已完成的条数不在批的边界上时（上次
中断在一批中间），截回到那一批的开头、整批重跑，所以续跑与一次跑完的批组成相同。没有 --resume 而目录非空：拒绝覆盖。
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
from typing import Any, Callable, Hashable, Mapping, Sequence

import yaml
from PIL import Image

from src.accounting import Compute, merge, write_compute_json
from src.constants import BOOTSTRAP_SEED
from src.data.records import Sample
from src.e1.letters import LetterTable, build_letter_table
from src.e1.prompts import COUNTBENCHQA_FIELD, DEFAULT_COUNT_QUESTIONS, THINK_METHOD, PromptConfig, Rendered
from src.e1.scoring import aggregate, score_generation, score_letter
from src.e1.tasks import DataPaths, Selection
from src.runlog import write_run_metadata

REPO_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS = "predictions.jsonl"
METRICS = "metrics.json"
COMPUTE = "compute.json"
CONFIG = "config.yaml"
README = "README.md"
TIMING = "logs/timing.jsonl"
NOT_IDENTITY_KEYS: tuple[str, ...] = ()  # DEV-0017：批配置是协议的一部分，批大小也进运行身份（D36-1）


@dataclass(frozen=True)
class RunSpec:
    model_key: str
    family: str
    repo: str
    revision: str
    adapter_model_key: str
    task: str
    subset: str
    policy_tag: str
    policy: Mapping[str, Any]
    batch_size: int
    limit: int | None
    tiny: bool
    map_ci_resamples: int
    count_questions: str = DEFAULT_COUNT_QUESTIONS  # 计数的问句来源（DEV-0015 §4；D36-2 起默认 countbenchqa）；别的任务不用
    # 训练后的检查点（DEV-0030 §2，run_e1 --checkpoint）：manifest 的 sha256 与摘要，写进 config.yaml 的 checkpoint；base 运行为 None
    # （config.yaml 没有这个键，与原来逐字节相同）
    checkpoint: Mapping[str, Any] | None = None


def policy_dirname(tag: str) -> str:
    return tag.replace(":", "-")


def leaf_dir(out: str | Path, model_key: str, task: str, subset: str, policy_tag: str) -> Path:
    """<out>/<模型>/<任务>，非主口径时加 __<子集>、__<口径>（如 refexp_box__extra__qwen-1024）。"""
    name = task
    if subset != "eval":
        name += f"__{subset}"
    if policy_tag != "default":
        name += f"__{policy_dirname(policy_tag)}"
    return Path(out) / model_key / name


def _jsonl_line(rec: Mapping[str, Any]) -> str:
    return json.dumps(rec, ensure_ascii=False) + "\n"


def read_jsonl_tolerant(path: Path, *, repair: bool) -> list[dict[str, Any]]:
    """读 jsonl；末尾不完整的一行（中断时写了一半）丢掉，repair=True 时把文件截到最后一个完整行。"""
    if not path.exists():
        return []
    raw = path.read_bytes()
    keep = raw.rfind(b"\n") + 1
    recs = []
    for line in raw[:keep].decode("utf-8").splitlines():
        if line.strip():
            recs.append(json.loads(line))
    if repair and keep != len(raw):
        with open(path, "r+b") as f:
            f.truncate(keep)
    return recs


def _truncate_lines(path: Path, k: int) -> None:
    """把 jsonl 截到前 k 行（续跑时退回批的边界）。"""
    raw = path.read_bytes()
    pos = 0
    for _ in range(k):
        pos = raw.index(b"\n", pos) + 1
    with open(path, "r+b") as f:
        f.truncate(pos)


def _append(path: Path, lines: Sequence[str]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write("".join(lines))
        f.flush()
        os.fsync(f.fileno())


def ids_digest(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def batch_plan(keys: Sequence[Hashable], batch_size: int) -> list[tuple[int, int]]:
    """样本下标 [0, n) 的批划分（batch_protocol）：按顺序连续取，每批至多 batch_size 条；keys[i] 与本批第一条不同时另起一批
    （受限字母读数：keys = 每条的字母集合；其余任务 keys 全相同）。返回 [(起, 止)]，止不含。"""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"batch_size 必须是正整数，收到 {batch_size!r}")
    n = len(keys)
    out, start = [], 0
    for i in range(1, n + 1):
        if i == n or i - start == batch_size or keys[i] != keys[start]:
            out.append((start, i))
            start = i
    return out


def readout_info(adapter: Any, prompts: PromptConfig, spec: RunSpec,
                 rendered: Sequence[Rendered]) -> tuple[dict[str, Any], LetterTable | None]:
    """这次运行怎么读答案（写进 config.yaml 的 generation.readout）与受限字母读数的字母表（text 读法为 None）。
    字母表按全部样本的字母并集、用适配器的分词器构造（src/e1/letters.py），续跑时同一份。"""
    kind = prompts.readout(spec.task, spec.subset)
    info: dict[str, Any] = {"kind": kind, "parse": prompts.parser(spec.task, spec.family, spec.subset)}
    if kind != "letter":
        return info, None
    letters = sorted({lt for r in rendered for lt in r.extras["letters"]})
    info["spellings"] = list(prompts.letter_spellings())
    if not letters:
        info["table"] = None
        return info, None
    table = build_letter_table(adapter.tokenizer, letters, prompts.letter_spellings())
    info["table"] = table.as_dict()
    return info, table


def batch_protocol_record(prompts: PromptConfig, batch_size: int) -> dict[str, Any]:
    """config.yaml 的 batch_protocol：这次实际用的批配置，以及它是否就是冻结的那一份（frozen）。"""
    bp = prompts.batch_protocol()
    return {"protocol_version": bp["version"], "batch_size": batch_size, "padding_side": bp["padding_side"],
            "order": bp["order"], "letter_batches": bp["letter_batches"], "frozen": batch_size == bp["batch_size"],
            "frozen_batch_size": bp["batch_size"]}


def run_config(spec: RunSpec, sel: Selection, prompts: PromptConfig, paths: DataPaths,
               readout: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """config.yaml 的内容：运行的身份（不含时间戳、git 状态与机器相关的路径）。readout 默认按配置给出读法（不含字母表）。"""
    e = prompts.entry(spec.task, spec.family)
    ids = [s.id for s in sel.samples]
    ro = dict(readout) if readout is not None else {"kind": prompts.readout(spec.task, spec.subset),
                                                   "parse": prompts.parser(spec.task, spec.family, spec.subset)}
    cfg = {
        "experiment": "E1",
        "framework": "src/e1（DEV-0014；协议 v2：DEV-0017）",
        "model": {"key": spec.model_key, "family": spec.family, "repo": spec.repo, "revision": spec.revision,
                  "adapter_model_key": spec.adapter_model_key, "tiny": spec.tiny},
        "image_policy": {"tag": spec.policy_tag, **dict(spec.policy)},
        "task": spec.task,
        "subset": spec.subset,
        "generation": generation_record(prompts, spec, ro),
        "prompt": {"config": os.path.relpath(prompts.path, REPO_ROOT) if Path(prompts.path).is_relative_to(REPO_ROOT)
                   else prompts.path, "config_sha256": prompts.sha256,
                   "entry": {k: e[k] for k in ("template", "native", "provenance", "source_url", "parse",
                                               "max_new_tokens") if k in e},
                   **({"count_questions": spec.count_questions} if spec.task == "count" else {})},
        "data": {"n_samples": len(ids), "sample_ids_sha256": ids_digest(ids), "has_gt": sel.has_gt,
                 "selection_stats": json.loads(json.dumps(sel.stats, ensure_ascii=False)),
                 "config_sha256": {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                   for p in paths.config_files()}},
        "limit": spec.limit,
        "map_ci_resamples": spec.map_ci_resamples,
        "batch_size": spec.batch_size,
        "batch_protocol": batch_protocol_record(prompts, spec.batch_size),
    }
    if spec.checkpoint is not None:  # DEV-0030 §2：训练后的模型（base 运行没有这个键）
        cfg["checkpoint"] = json.loads(json.dumps(dict(spec.checkpoint), ensure_ascii=False))
    return cfg


def generation_record(prompts: PromptConfig, spec: RunSpec, readout: Mapping[str, Any]) -> dict[str, Any]:
    """config.yaml 的 generation 一节。思考子集另加 think_budget（D59-1）；其余子集与原来逐字节相同（键与值都不变）。"""
    gen = {"enable_thinking": prompts.enable_thinking(spec.subset),
           "max_new_tokens": prompts.max_new_tokens(spec.task, spec.family, spec.subset),
           "decoding": prompts.decoding(), "order": prompts.order, "readout": dict(readout)}
    if prompts.uses_think_budget(spec.subset):
        tb, ab = prompts.think_budget()
        gen["think_budget"] = {"thinking_budget": tb, "answer_budget": ab, "method": THINK_METHOD,
                               "max_new_tokens_note": "max_new_tokens = thinking_budget + answer_budget（模型生成的 token 数上限，补的 token 不计入）"}
    return gen


def _identity(cfg: Mapping[str, Any]) -> dict[str, Any]:
    norm = yaml.safe_load(yaml.safe_dump(dict(cfg), sort_keys=False, allow_unicode=True))
    return {k: v for k, v in norm.items() if k not in NOT_IDENTITY_KEYS}


def load_images(sample: Sample) -> list[Image.Image]:
    out = []
    for p in sample.images:
        with Image.open(p) as im:
            out.append(im.convert("RGB"))
    return out


def _metrics_doc(spec: RunSpec, cfg: Mapping[str, Any], body: Mapping[str, Any], prompts: PromptConfig) -> dict[str, Any]:
    doc = {"task": spec.task, "subset": spec.subset, "model": spec.model_key, "family": spec.family,
           "revision": spec.revision, "image_policy": cfg["image_policy"], "generation": cfg["generation"],
           "batch_protocol": cfg["batch_protocol"], "prompt_config_sha256": cfg["prompt"]["config_sha256"]}
    if spec.task == "count":
        doc["count_questions"] = {"source": spec.count_questions,
                                  "note": prompts.countbenchqa_note if spec.count_questions == "countbenchqa" else None}
    return {**doc, **body}


def _fmt(v: Any) -> str:
    return "null" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def _ci_s(ci: Mapping[str, Any] | None) -> str:
    return f"[{_fmt(ci['low'])}, {_fmt(ci['high'])}]" if ci else "—"


def _think_marker_line(tm: Mapping[str, Any] | None) -> str:
    if tm is None:
        return "不适用（开思考）"
    return f"{tm['n']} 条（其中思考段未闭合 {tm['n_thinking_unclosed']} 条）"


def _three_column_lines(metrics: Mapping[str, Any]) -> list[str]:
    lines = ["## 三列：严格读数 / 有效率 / 条件读数（规则见 configs/e1_prompts.json 的 report_columns）", ""]
    cols = metrics.get("three_columns") or []
    if not cols:
        return lines + ["（没有 GT 或空集合，不算指标）", ""]
    lines += ["| 指标 | 范围 | 严格读数 | 95% CI | 有效率 | 有效 / 全部 | 条件读数 | 95% CI |", "|---|---|---|---|---|---|---|---|"]
    for c in cols:
        s, cd = c["strict"], c["conditional"]
        lines.append(f"| {c['name']} | {c['scope']} | {_fmt(s['value'])} | {_ci_s(s['ci'])} | {_fmt(c['valid_rate'])} | "
                     f"{c['n_valid']} / {c['n']} | {_fmt(cd['value'])} | {_ci_s(cd['ci'])} |")
    v = metrics["valid"]
    lines += ["", f"- 无效 {v['n_invalid']} 条 = 拒答 {v['n_refusal']} + 其他解析失败 {v['n_other_failure']}", ""]
    rbp = metrics.get("refusal_by_person")
    if rbp:
        lines += [f"## 拒答按被指代物体分列（person / 非 person；指标 {rbp['metric']}）", "",
                  "| 组 | n | 拒答 | 拒答率 | 有效率 | 严格读数 | 条件读数 | 条件读数 95% CI |", "|---|---|---|---|---|---|---|---|"]
        for g in ("person", "non_person", "unknown"):
            if g in rbp:
                r = rbp[g]
                lines.append(f"| {g} | {r['n']} | {r['n_refusal']} | {_fmt(r['refusal_rate'])} | {_fmt(r['valid_rate'])} | "
                             f"{_fmt(r['strict']['value'])} | {_fmt(r['conditional']['value'])} | {_ci_s(r['conditional']['ci'])} |")
        lines.append("")
    return lines


def _thinking_end_lines(te: Mapping[str, Any] | None) -> list[str]:
    """思考子集：按思考段的结束方式分开的条数与读数（scoring.thinking_end_breakdown）。其余子集为空。"""
    if not te:
        return []
    names = {"natural": "自然收尾", "forced": "强制收尾", "not_opened": "没写思考段（直接作答）", "unclosed": "思考段里就停了"}
    lines = ["## 思考段的结束方式（预算强制收尾，D59-1）", "",
             "| 结束方式 | 条数 | 解析失败率 | 截断 | 思考 token 中位数 | 严格读数 |", "|---|---|---|---|---|---|"]
    for k, n in te["counts"].items():
        e = te["by_kind"].get(k)
        if e is None:
            lines.append(f"| {names.get(k, k)} | 0 | — | — | — | — |")
            continue
        if "strict" in e:
            val = "；".join(f"{x['name']} {_fmt(x['value'])}" for x in e["strict"]) or "—"
        else:
            val = f"F1@0.5（合并）{_fmt((e.get('F1@0.5_pooled') or {}).get('value'))}" if "F1@0.5_pooled" in e else "—"
        med = (e["thinking_tokens"] or {}).get("p50")
        lines.append(f"| {names.get(k, k)} | {n} | {_fmt(e['parse_failure_rate'])} | {e['n_truncated']} | {_fmt(med)} | {val} |")
    return lines + [""]


def readme_text(spec: RunSpec, cfg: Mapping[str, Any], metrics: Mapping[str, Any]) -> str:
    gen, bp = cfg["generation"], cfg["batch_protocol"]
    lines = [f"# E1 · {spec.model_key} · {spec.task} · {spec.subset}", "",
             f"- 模型：{spec.repo}@{spec.revision}（{spec.family}；适配器 {spec.adapter_model_key}）",
             f"- 图像口径：{cfg['image_policy']}",
             f"- 生成：enable_thinking={gen['enable_thinking']}，max_new_tokens={gen['max_new_tokens']}，greedy，{gen['order']}；"
             f"读法 {gen['readout']['kind']}（解析 {gen['readout']['parse']}）",
             f"- 批配置：batch_size={bp['batch_size']}、{bp['padding_side']} 填充、顺序 {bp['order']}、{bp['letter_batches']}；"
             f"{'冻结的协议配置' if bp['frozen'] else '不是冻结的协议配置（冻结值 ' + str(bp['frozen_batch_size']) + '）'}",
             f"- 提示配置 sha256：{cfg['prompt']['config_sha256']}；模板：`{cfg['prompt']['entry']['template']!r}`",
             f"- 样本数 n = {metrics['n']}（id 指纹 {cfg['data']['sample_ids_sha256'][:16]}…）；有 GT：{metrics['has_gt']}"]
    if metrics.get("count_questions"):
        cq = metrics["count_questions"]
        lines.append(f"- 计数问句：{cq['source']}" + (f"。{cq['note']}" if cq["note"] else ""))
    if gen.get("think_budget"):
        tb = gen["think_budget"]
        lines.append(f"- 预算强制收尾（D59-1）：思考 {tb['thinking_budget']} token，作答 {tb['answer_budget']} token"
                     f"（{tb['method']}，规则见 src/adapters/budget.py）")
    lines += ["", "## 标题指标（严格读数）", ""]
    if not metrics["headline"]:
        lines.append("（没有 GT 或空集合，不算指标）")
    for h in metrics["headline"]:
        lines.append(f"- {h['name']}（{h['scope']}）：{_fmt(h['value'])}，95% CI {_ci_s(h.get('ci'))}，n = {h['n']}")
    lines += [""] + _three_column_lines(metrics)
    lines += _thinking_end_lines(metrics.get("thinking_end"))
    p = metrics["parse"]
    lines += ["## 解析与截断", "",
              f"- 解析失败率：{_fmt(p['failure_rate'])}（{p['n_failed']}/{p['n']}）；分原因：{p['failures']}",
              f"- 截断率：{_fmt(metrics['truncation']['rate'])}；思考段未闭合：{_fmt(metrics['thinking_unclosed']['rate'])}",
              "", "## 失败的诊断（只作诊断，失败原因不变；规则见 src/e1/scoring.failure_detail）", "",
              f"- 失败的样子：{metrics['diagnostics']['failure_detail']['counts'] or '（无）'}",
              f"- 关思考时自发写出思考标记：{_think_marker_line(metrics['diagnostics']['think_marker_when_off'])}",
              "", "## 文件", "",
              "- predictions.jsonl：逐条输出、解析与得分；metrics.json：全部指标（由 predictions.jsonl 算出）；",
              "- compute.json：计算记账（src/accounting.py）；config.yaml：运行身份；meta.json：git、主机、包版本；logs/：墙钟与每次调用的记录。",
              ""]
    return "\n".join(lines)


def finalize(leaf: Path, spec: RunSpec, sel: Selection, cfg: Mapping[str, Any], prompts: PromptConfig) -> dict[str, Any]:
    """由 predictions.jsonl 与 logs/timing.jsonl 写 metrics.json、compute.json、README.md。返回 metrics。"""
    recs = read_jsonl_tolerant(leaf / PREDICTIONS, repair=False)
    ids = [s.id for s in sel.samples]
    if [r["id"] for r in recs] != ids[: len(recs)]:
        raise RuntimeError(f"{leaf / PREDICTIONS} 的 id 不是样本顺序的前缀")
    samples = sel.samples[: len(recs)]
    body = aggregate(spec.task, samples, recs, has_gt=sel.has_gt, map_ci_resamples=spec.map_ci_resamples,
                     enable_thinking=bool(cfg["generation"]["enable_thinking"]))
    gt_diag = grid_token_diagnostic(recs)
    if gt_diag is not None:  # DEV-0030 §2：只有带网格 token 的检查点才有这一项（base 运行的 metrics.json 不变）
        body["diagnostics"]["grid_tokens_in_output"] = gt_diag
    body["complete"] = len(recs) == len(ids)
    body["n_expected"] = len(ids)
    metrics = _metrics_doc(spec, cfg, body, prompts)
    (leaf / METRICS).write_text(json.dumps(metrics, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    timing: dict[str, dict[str, Any]] = {}
    for t in read_jsonl_tolerant(leaf / TIMING, repair=False):
        timing[t["id"]] = t
    comp = []
    for r in recs:
        wall = float(timing.get(r["id"], {}).get("wall_time_s", 0.0))
        # 预算强制收尾补的 token 是送进模型的输入（与提示 token 同类），记进 prompt_tokens；其余子集的记录没有这个键（加 0）
        comp.append(Compute.call(generated_tokens=r["n_generated_tokens"],
                                 prompt_tokens=r["n_prompt_tokens"] + int(r.get("n_inserted_tokens", 0)),
                                 visual_tokens=r["n_visual_tokens"], image_pixels=r["n_image_pixels"], call_time_s=wall,
                                 new_round=True).as_sample(wall_time_s=wall))
    write_compute_json(leaf / COMPUTE, merge(comp))
    (leaf / README).write_text(readme_text(spec, cfg, metrics), encoding="utf-8")
    return metrics


def grid_token_ids_of(adapter: Any) -> frozenset[int]:
    """适配器登记的网格 token（训练后的 E3b 检查点才有；base 为空）。"""
    return frozenset(int(v) for v in (getattr(adapter, "grid_token_ids", None) or {}).values())


def grid_token_diagnostic(recs: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """自由生成里冒出网格 token 的条数（DEV-0030 §2，DEV-0025 的建议）：记录里有 n_grid_tokens（适配器有网格 token）才算。"""
    if not any("n_grid_tokens" in r for r in recs):
        return None
    hit = [r for r in recs if int(r.get("n_grid_tokens", 0)) > 0]
    return {"n_rows": len(hit), "n": len(recs), "rate": (len(hit) / len(recs)) if recs else None,
            "n_tokens": int(sum(int(r.get("n_grid_tokens", 0)) for r in recs)),
            "note": "生成的 token（不含预算强制收尾补的 token）里属于检查点网格 token 的个数；n_rows 是至少有一个的条数"}


def _common_record(s: Sample, rendered: Any, image_size: tuple[int, int], res: Any, readout: str) -> dict[str, Any]:
    if res.stopped is None or res.n_image_pixels is None:
        raise RuntimeError(f"{s.id}：适配器没有填 stopped / n_image_pixels（D29：三个适配器一律填写）")
    return {"id": s.id, "source": s.source, "split": s.split, "instruction": rendered.text, "extras": rendered.extras,
            "image_size": list(image_size), "readout": readout, "raw_text": res.raw_text, "text": res.text,
            "stopped": res.stopped,
            # 受限字母读数只生成 1 个 token、停止 token 不在允许集合里，stopped 总是 False；那 1 个 token 就是全部回答，不算截断
            "truncated": res.stopped is False and readout != "letter",
            "thinking_unclosed": res.thinking_unclosed,
            "n_generated_tokens": res.n_generated_tokens, "n_prompt_tokens": res.n_prompt_tokens,
            "n_visual_tokens": res.n_visual_tokens, "n_image_pixels": res.n_image_pixels,
            **budget_fields(res)}


def budget_fields(res: Any) -> dict[str, Any]:
    """预算强制收尾的记录字段（D59-1；只有 generate_batch_budgeted 的结果有，其余子集返回空字典，记录与原来逐字节相同）。
    inserted_text 是补的那段文字（raw_text[inserted_span] 恰是它）。"""
    if getattr(res, "thinking_end", None) is None:
        return {}
    span = res.inserted_span
    return {"thinking_end": res.thinking_end, "forced_close": res.forced_close,
            "n_thinking_tokens": res.n_thinking_tokens, "n_answer_tokens": res.n_answer_tokens,
            "n_inserted_tokens": res.n_inserted_tokens, "inserted_token_ids": list(res.inserted_token_ids),
            "inserted_at": res.inserted_at, "inserted_span": None if span is None else list(span),
            "inserted_text": None if span is None else res.raw_text[span[0]:span[1]], "answer_text": res.answer_text}


def generate_rows(adapter: Any, inputs: Sequence[Any], *, batch_size: int, enable_thinking: bool, max_new_tokens: int,
                  decoding: Mapping[str, Any], allowed_token_ids: Sequence[int] | None = None,
                  think_budget: tuple[int, int] | None = None) -> list[Any]:
    """batch_size = 1 走 generate（逐条 greedy 的路径）；否则 generate_batch（即使这一批只有一行）。
    allowed_token_ids 只在受限字母读数时给（整批相同）。
    think_budget = (thinking_budget, answer_budget) 时（思考子集，D59-1）一律走 generate_batch_budgeted（batch_size = 1 时是一行的批）：
    它固定 greedy、开思考，所以 decoding 必须是 greedy、enable_thinking 必须为 True，max_new_tokens 必须是两个预算之和。"""
    if think_budget is not None:
        tb, ab = think_budget
        if not enable_thinking or decoding.get("do_sample") is not False or max_new_tokens != tb + ab:
            raise ValueError(f"预算强制收尾要求开思考、greedy、max_new_tokens = {tb} + {ab}；收到 enable_thinking={enable_thinking}、"
                             f"do_sample={decoding.get('do_sample')!r}、max_new_tokens={max_new_tokens}")
        if batch_size == 1 and len(inputs) != 1:
            raise ValueError("batch_size = 1 时每批只能有一条")
        return adapter.generate_batch_budgeted(list(inputs), thinking_budget=tb, answer_budget=ab,
                                               allowed_token_ids=None if allowed_token_ids is None else list(allowed_token_ids))
    kw = dict(decoding)
    if allowed_token_ids is not None:
        kw["allowed_token_ids"] = list(allowed_token_ids)
    if batch_size == 1:
        if len(inputs) != 1:
            raise ValueError("batch_size = 1 时每批只能有一条")
        return [adapter.generate(inputs[0], enable_thinking=enable_thinking, max_new_tokens=max_new_tokens, **kw)]
    return adapter.generate_batch(list(inputs), enable_thinking=enable_thinking, max_new_tokens=max_new_tokens, **kw)


def batch_keys(readout: str, rendered: Sequence[Rendered]) -> list[Hashable]:
    """batch_plan 的分批键：受限字母读数取每条的字母集合，其余任务全相同。"""
    if readout == "letter":
        return [tuple(r.extras["letters"]) for r in rendered]
    return [None] * len(rendered)


def run_task(adapter: Any, spec: RunSpec, sel: Selection, prompts: PromptConfig, paths: DataPaths, out_dir: str | Path,
             *, resume: bool = False, argv: Sequence[str] = (), log: Callable[[str], None] | None = None,
             warmup: bool = True, run_extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """跑一个任务，返回 metrics（规则见模块说明）。sel 的样本已按 limit 截好。run_extra 并进 logs/run_<k>.json（DEV-0030：
    --checkpoint 时的合并核对摘要与检查点路径）。"""
    grid_ids = grid_token_ids_of(adapter)
    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    if isinstance(spec.batch_size, bool) or not isinstance(spec.batch_size, int) or spec.batch_size < 1:
        raise ValueError(f"batch_size 必须是正整数，收到 {spec.batch_size!r}")
    if spec.task == "count":  # 问句来源由样本决定（tasks.select），这里核对与 RunSpec 一致，防止静默混用
        has = [COUNTBENCHQA_FIELD in s.fields for s in sel.samples]
        mixed = any(has) and not all(has)
        if has and (mixed or any(has) != (spec.count_questions == "countbenchqa")):
            raise ValueError(f"count_questions={spec.count_questions!r}，而样本里带人工问句的有 {sum(has)}/{len(has)} 条："
                             "select 与 RunSpec 的问句来源不一致")
    rendered_all = [prompts.render(spec.task, spec.family, s) for s in sel.samples]
    ro, table = readout_info(adapter, prompts, spec, rendered_all)
    plan = batch_plan(batch_keys(ro["kind"], rendered_all), spec.batch_size)
    leaf = leaf_dir(out_dir, spec.model_key, spec.task, spec.subset, spec.policy_tag)
    cfg = run_config(spec, sel, prompts, paths, readout=ro)
    pred_path = leaf / PREDICTIONS
    if leaf.exists() and any(leaf.iterdir()):
        if not resume:
            raise FileExistsError(f"{leaf} 已存在且非空；续跑请加 --resume")
        if not (leaf / CONFIG).exists():
            raise FileExistsError(f"{leaf} 里没有 {CONFIG}，不能续跑")
        old = yaml.safe_load((leaf / CONFIG).read_text(encoding="utf-8"))
        if _identity(old) != _identity(cfg):
            diff = sorted(k for k in set(_identity(old)) | set(_identity(cfg)) if _identity(old).get(k) != _identity(cfg).get(k))
            raise ValueError(f"{leaf}：续跑的配置与已有的不同（{diff}），拒绝混写")
    else:
        leaf.mkdir(parents=True, exist_ok=True)
        (leaf / "logs").mkdir(exist_ok=True)
        files = [prompts.path, *paths.config_files()]
        write_run_metadata(leaf, config=cfg, seed=BOOTSTRAP_SEED, data_files=files, repo_root=REPO_ROOT)
    (leaf / "logs").mkdir(exist_ok=True)
    done = read_jsonl_tolerant(pred_path, repair=True)
    ids = [s.id for s in sel.samples]
    if [r["id"] for r in done] != ids[: len(done)]:
        raise RuntimeError(f"{pred_path} 里已完成的 id 不是样本顺序的前缀，不能续跑")
    bounds = [b for b, _ in plan] + [len(ids)]
    keep = max(b for b in bounds if b <= len(done))
    trimmed = None
    if keep < len(done):  # 上次中断在一批中间：退回这一批的开头，整批重跑（批组成与一次跑完相同）
        _truncate_lines(pred_path, keep)
        trimmed = {"from": len(done), "to": keep}
        log(f"[{spec.model_key} {spec.task}/{spec.subset}] 已完成 {len(done)} 条不在批的边界上，退回到 {keep} 条整批重跑")
        done = done[:keep]
    todo = [(a, b) for a, b in plan if a >= keep]
    remaining = sel.samples[keep:]
    run_k = len(list((leaf / "logs").glob("run_*.json"))) + 1
    run_path = leaf / "logs" / f"run_{run_k:03d}.json"
    run_log: dict[str, Any] = {"argv": list(argv), "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                               "n_done_before": len(done), "n_remaining": len(remaining), "batch_size": spec.batch_size,
                               "trimmed_to_batch_boundary": trimmed, "finished_utc": None}
    if run_extra:
        run_log.update(dict(run_extra))
    run_path.write_text(json.dumps(run_log, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")  # 先占位：中断也留下记录
    enable_thinking = prompts.enable_thinking(spec.subset)
    max_new = prompts.max_new_tokens(spec.task, spec.family, spec.subset)
    decoding = prompts.decoding()
    think_budget = prompts.think_budget() if prompts.uses_think_budget(spec.subset) else None
    parser = ro["parse"]
    t_all = time.perf_counter()
    if remaining and warmup:
        run_log["warmup_wall_s"] = adapter.warmup(load_images(remaining[0]), batch_size=min(spec.batch_size, len(remaining)))
    n_batches = 0
    for a, b in todo:
        batch, rendered = sel.samples[a:b], rendered_all[a:b]
        t0 = time.perf_counter()
        imgs = [load_images(s) for s in batch]
        inputs = [adapter.build_inputs(im, r.text, prompts.order) for im, r in zip(imgs, rendered)]
        build_s = time.perf_counter() - t0
        allowed = table.allowed_ids(rendered[0].extras["letters"]) if table is not None else None
        results = generate_rows(adapter, inputs, batch_size=spec.batch_size, enable_thinking=enable_thinking,
                                max_new_tokens=max_new, decoding=decoding, allowed_token_ids=allowed,
                                think_budget=think_budget)
        if len(results) != len(batch):
            raise RuntimeError("generate_batch 返回的行数与输入不同")
        lines, tlines = [], []
        for s, r, im, res in zip(batch, rendered, imgs, results):
            size = im[0].size
            rec = _common_record(s, r, size, res, ro["kind"])
            if table is not None:
                rec.update(score_letter(spec.task, s, res, table=table, letters=r.extras["letters"]))
            else:
                rec.update(score_generation(spec.task, parser, s, res, adapter=adapter, image_size=size, extras=r.extras))
            if grid_ids:  # DEV-0030 §2：只有带网格 token 的检查点才记（base 的记录逐字节不变）
                rec["n_grid_tokens"] = sum(1 for t in res.token_ids if int(t) in grid_ids)
            lines.append(_jsonl_line(rec))
            tlines.append(_jsonl_line({"id": s.id, "run": run_k, "batch_index": n_batches, "batch_rows": len(batch),
                                       "wall_time_s": res.wall_time_s, "batch_wall_time_s": res.batch_wall_time_s,
                                       "build_inputs_s": build_s / len(batch)}))
        _append(leaf / TIMING, tlines)
        _append(pred_path, lines)
        n_batches += 1
        if n_batches == 1 or n_batches % 20 == 0 or b == len(ids):
            log(f"[{spec.model_key} {spec.task}/{spec.subset}] {b}/{len(ids)}")
    metrics = finalize(leaf, spec, sel, cfg, prompts)
    run_log.update(finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), n_batches=n_batches,
                   wall_s=round(time.perf_counter() - t_all, 3))
    run_path.write_text(json.dumps(run_log, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return metrics


__all__ = [
    "COMPUTE",
    "CONFIG",
    "METRICS",
    "PREDICTIONS",
    "README",
    "RunSpec",
    "TIMING",
    "batch_keys",
    "batch_plan",
    "batch_protocol_record",
    "budget_fields",
    "finalize",
    "generate_rows",
    "generation_record",
    "grid_token_diagnostic",
    "grid_token_ids_of",
    "ids_digest",
    "leaf_dir",
    "load_images",
    "policy_dirname",
    "read_jsonl_tolerant",
    "readout_info",
    "run_config",
    "run_task",
]
