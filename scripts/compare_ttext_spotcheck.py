#!/usr/bin/env python
"""冻结 LLM 的对照（C1、C2、C1′）的 T 文本抽查比对（DEV-0039 §2；D72-4）。只读两份 run_e1 的输出目录，不跑模型。

背景（D72-4、E3 预注册 §9）：C1、C2 的 LLM 没动，C1′ 只多了网格 token 的行、T 文本解码时屏蔽它们（run_e1 --mask-grid-tokens）。
这三种对照不重跑 T 文本评测，直接引用基座的读数（RUN-0027 的主评测，experiments/E1/0027_e1_main/<模型>/<任务>/）；为此抽查几百条，
确认与基座相同。抽查的运行与比对：
    # 服务器（B200；与基座同一台机器、同一套批配置）。C1′ 另加 --mask-grid-tokens。N 的取法见下面"批组成"：CV-Bench 的 320 不是
    # 批边界，所以它单独用 321 跑（或者全部任务都用 --limit 192）
    python scripts/run_e1.py --model qwen3.5-2b --task refexp_box refexp_poly grid16 count ovd rel_depth_nyu rel_depth_blink \\
        spatial_embspatial --data-root $PROJ/data --checkpoint <训练叶目录>/ckpt/final/model --checkpoint-label t_text/c1/seed0 \\
        --limit 320 --out experiments/E3/<NNNN_slug>
    python scripts/run_e1.py --model qwen3.5-2b --task spatial_cvbench --data-root $PROJ/data --checkpoint <同上> \\
        --checkpoint-label t_text/c1/seed0 --limit 321 --out experiments/E3/<NNNN_slug>
    python scripts/compare_ttext_spotcheck.py --base experiments/E1/0027_e1_main/qwen3.5-2b \\
        --spot experiments/E3/<NNNN_slug>/t_text/c1/seed0/qwen3.5-2b --out experiments/E3/<NNNN_slug>/spotcheck_c1_seed0.json \\
        [--task count grid16 …] [--data-root $PROJ/data]
    # 开发机彩排：两边都是 run_e1 --tiny 的输出；--tiny 时检测的 GT 取 tests/fixtures/e1
任务：默认两边都有的任务目录（<任务>[__<子集>]，含 config.yaml 与 predictions.jsonl 或 predictions.jsonl.gz）；--task 给的必须两边都有。
只比 run_e1 正式运行模式的输出（eval、free 等子集；读法 text 或 letter，每条有 raw_text）；事后读数（debiased、letter_swap、forced）的
记录没有原始输出，遇到就判不可比。
大文件：只有 predictions.jsonl.gz 时解压来读；同目录有 LARGE_FILES.sha256（RUN-0027 的定位）就核对解压后的 sha256 与字节数。

**先核对**（任何一条不过：判"不可比"，退出码 2，不给判定）：
1. 每个任务两边的 config.yaml 相同的项：模型（model 一节）、任务、子集、图像口径（image_policy）、批配置（batch_size 与 batch_protocol）、
   生成设置（generation；受限字母读数的字母表按本次全部样本的字母并集构造，所以只要求抽查那边的字母是基座的子集、每个字母的写法与 token
   相同）、提示条目（prompt.entry、计数的问句来源）、有没有 GT。
2. 提示配置的 sha256 相同；或者是已裁定等价的一对（EQUIVALENT_PROMPT_CONFIGS）且子集在等价范围内。RUN-0027 用的是 5de1aec7…，之后的
   代码是 f3353de0…，两者在 eval、dev、free、extra 上的提示渲染、生成参数与解析逐字节相同（D60；tests/test_e1_d59.py 核对）。等价时照记进
   报告（prompt_config.equivalence），并且下面第 3 条的逐条指令原文核对一样要过。
3. 条目：抽查的 N 条（抽查运行必须完整：predictions 的行数 = config.yaml 的 n_samples）是基座评测清单（基座 predictions 的顺序，即选择
   顺序）的前 N 条，id 逐条相同；每条的指令原文、extras（选项、字母）与原图尺寸也逐条相同（输入相同）。基座也必须完整。
4. 批组成：N 必须是基座全量运行的一个批边界（src/e1/runner.batch_plan，分批键 runner.batch_keys，批大小取 config.yaml）。
   **为什么 N 取批大小（64）的整数倍**：批配置是协议的一部分（D36-1、D36-7、D43：批大小 64、左填充、按选择顺序连续组成，受限字母读数时
   字母集合一变就另起一批）。一批里的各行一起前向，左填充的长度与矩阵的形状由整批决定，bf16 下批组成一变，数值就可能变（D36-1 的"数值
   噪声底"）。批按样本顺序连续组成，所以前 N 条的批组成与全量运行相同，当且仅当 N 落在全量运行的批边界上：自由生成与字母集合不变的任务，
   边界就是 64 的整数倍；否则最后一批只有一部分，它的组成与全量不同，这一批的输出不同就不能归到模型上。
   **例外**：字母集合会变的任务，边界不在 64 的整数倍上。RUN-0027 的 CV-Bench（spatial_cvbench：2、4、5、6 个选项混排）2588 条里，64 的
   整数倍只有 64、192、832、960 是边界；300 附近的边界有 299、301、315、321 等（各模型同一组边界，由数据的顺序决定）。其余任务（NYU 与
   BLINK 点对、EmbSpatial 的字母集合不变）的边界都是 64 的整数倍。所以一次 --task all 的 N 只能取两者共同的边界（64、192、832、960）；
   要 300 条左右，CV-Bench 单独用另一个 --limit 跑（如 321）。N 不在边界上时报错并列出最近的边界。

**逐任务输出**：比对条数 n；原始输出（raw_text）逐条相同的比例；解析结果（failure 与 parsed）相同的比例；逐条得分（score）相同的比例；
两边在这 N 条上的标题指标（src/e1/scoring.per_sample_metric 的那一个：定位 acc@0.5、多边形 cIoU、网格 gIoU、其余准确率；检测是前 N 张图
合并的 F1@0.5，要 GT：给 --data-root（或环境变量 VISINT_DATA_ROOT；--tiny 时取 tests/fixtures/e1）才算，否则记 null 并注明）与 95% CI；
配对差（抽查 − 基座）与配对 bootstrap 95% CI（src/stats）；两边的有效率；抽查输出里的网格 token 个数（C1′ 屏蔽之后应为 0）。

**判定**（VERDICT_RULE）：
- "相同"：所有任务的原始输出逐条相同（raw_text 逐字节相同）。管理员据此引用基座的读数。
- 否则"不完全相同"：逐任务列出原始输出不同的条目 id（每个任务最多 20 条）与解析结果的一致率。管理员据此决定能不能引用基座的读数。
退出码：0 = 相同；1 = 不完全相同；2 = 不可比（先核对不过）或出错。

输出：--out 是 JSON 的路径（以 .json 结尾），同名的 .md 写在旁边。两份都只由输入决定（不含时间戳），同样的输入逐字节相同。
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from src.e1.prompts import DEFAULT_COUNT_QUESTIONS, TASKS  # noqa: E402
from src.e1.runner import CONFIG, PREDICTIONS, batch_keys, batch_plan, ids_digest  # noqa: E402
from src.e1.scoring import f1_from_counts, metric_of, ovd_per_image_counts, per_sample_metric, validity  # noqa: E402
from src.stats import f1_statistic, paired_bootstrap_ci  # noqa: E402

TOOL = "scripts/compare_ttext_spotcheck.py（DEV-0039）"
LARGE_FILES = "LARGE_FILES.sha256"  # RUN-0027：超过 20 MB 的 predictions.jsonl 压成 .gz，这里记原文件的 sha256 与字节数
MAX_LISTED_IDS = 20
VERDICT_SAME, VERDICT_DIFF, VERDICT_NA = "相同", "不完全相同", "不可比"
VERDICT_RULE = ("相同 = 所有任务的原始输出（predictions 的 raw_text）逐条逐字节相同，管理员据此引用基座的读数；否则不完全相同：逐任务列出原始输出"
                f"不同的条目 id（每个任务最多 {MAX_LISTED_IDS} 条）与解析结果的一致率，管理员据此决定能不能引用基座的读数。先核对（模型、提示配置"
                " sha256、图像口径、批配置、前 N 条 id 与输入、批边界）不过时判不可比，不给判定")
BATCH_RULE = ("N 必须是基座全量运行的批边界（src/e1/runner.batch_plan）：批按样本顺序连续组成（D36-7、D43：批大小 64、左填充、按选择顺序，"
              "字母读数时字母集合一变就另起一批），一批里的各行一起前向，bf16 下批组成一变数值就可能变；前 N 条的批组成与全量运行相同，当且仅当 N 落在"
              "全量运行的批边界上。自由生成与字母集合不变的任务，边界就是 64 的整数倍；字母集合会变的任务（如 CV-Bench）另选 N")
# 已裁定等价的提示配置（只在列出的子集上）：RUN-0027 主评测用 5de1aec7…（D43），DEV-0027 改思考子集后是 f3353de0…（D60）；两者在这些子集上
# 的提示渲染、生成参数与解析逐字节相同（tests/test_e1_d59.py 的 SAME_SUBSETS；旧配置存档在 tests/fixtures/e1/e1_prompts_5de1aec7.json）
EQUIVALENT_PROMPT_CONFIGS: tuple[dict[str, Any], ...] = (
    {"sha256": ("5de1aec711fad0140db0cf27094d66e460cf2999ffcd5456f1964bc5b9851158",
                "f3353de0d53de22b8f2226e07afef94b643d02ec106868eaf6bf2766a45bf452"),
     "subsets": ("eval", "dev", "free", "extra"),
     "basis": "D60：只改了思考子集与验收 δ；eval、dev、free、extra 的提示渲染、生成参数与解析逐字节相同（tests/test_e1_d59.py）"},
)
SAME_KEYS = ("model", "task", "subset", "image_policy", "batch_size", "batch_protocol")  # config.yaml 里必须逐项相同的顶层键
READOUTS = ("text", "letter")  # run_e1 正式运行模式的两种读法（src/e1/runner）；事后读数的记录没有 raw_text，不比


class NotComparable(ValueError):
    """两边不能比：缺文件、运行身份不同、条目对不上、N 不在批边界上、运行不完整。"""


# ---------------------------------------------------------------------------
# 读入
# ---------------------------------------------------------------------------


def task_dirs(model_dir: Path) -> dict[str, Path]:
    """<模型目录> 下的任务叶目录（有 config.yaml 与 predictions.jsonl[.gz]）。"""
    if not model_dir.is_dir():
        raise NotComparable(f"{model_dir} 不是目录")
    out = {}
    for p in sorted(model_dir.iterdir()):
        if p.is_dir() and (p / CONFIG).is_file() and ((p / PREDICTIONS).is_file() or (p / (PREDICTIONS + ".gz")).is_file()):
            out[p.name] = p
    return out


def _large_file_entry(leaf: Path) -> tuple[str, int] | None:
    """LARGE_FILES.sha256 里 predictions.jsonl 那一行：(原文件的 sha256, 字节数)。"""
    f = leaf / LARGE_FILES
    if not f.is_file():
        return None
    for line in f.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[2] == PREDICTIONS:
            return parts[0], int(parts[1])
    raise NotComparable(f"{f} 里没有 {PREDICTIONS} 那一行")


def read_predictions(leaf: Path) -> list[dict[str, Any]]:
    """<leaf>/predictions.jsonl，没有时读 .jsonl.gz（有 LARGE_FILES.sha256 就核对解压后的 sha256 与字节数）。末行必须完整。"""
    if (leaf / PREDICTIONS).is_file():
        raw = (leaf / PREDICTIONS).read_bytes()
    else:
        raw = gzip.decompress((leaf / (PREDICTIONS + ".gz")).read_bytes())
        ent = _large_file_entry(leaf)
        if ent is not None and (hashlib.sha256(raw).hexdigest(), len(raw)) != ent:
            raise NotComparable(f"{leaf / (PREDICTIONS + '.gz')} 解压后的 sha256 / 字节数与 {LARGE_FILES} 记的 {ent} 不符")
    if raw and not raw.endswith(b"\n"):
        raise NotComparable(f"{leaf} 的 predictions 末行不完整（运行中断？）")
    return [json.loads(x) for x in raw.decode("utf-8").splitlines() if x.strip()]


def read_config(leaf: Path) -> dict[str, Any]:
    return yaml.safe_load((leaf / CONFIG).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 先核对
# ---------------------------------------------------------------------------


def prompt_equivalence(sha_base: str | None, sha_spot: str | None, subset: str) -> dict[str, Any] | None:
    """提示配置：相同返回 None；是已裁定等价的一对且子集在范围内返回那一条；否则 NotComparable。"""
    if sha_base == sha_spot and sha_base:
        return None
    for eq in EQUIVALENT_PROMPT_CONFIGS:
        if {sha_base, sha_spot} == set(eq["sha256"]) and subset in eq["subsets"]:
            return {"base_sha256": sha_base, "spot_sha256": sha_spot, "subsets": list(eq["subsets"]), "basis": eq["basis"]}
    raise NotComparable(f"提示配置的 sha256 不同：基座 {sha_base}，抽查 {sha_spot}（子集 {subset}），也不是已裁定等价的一对")


def _letter_table_ok(tb: Mapping[str, Any] | None, ts: Mapping[str, Any] | None) -> bool:
    """字母表：抽查那边的字母是基座的子集，每个字母的写法（spelling、token id、decode）相同；不是单 token 的写法也一致。"""
    if tb is None or ts is None:
        return tb is None and ts is None
    letters = list(ts.get("letters") or [])
    if not set(letters) <= set(tb.get("letters") or []):
        return False
    if any((ts.get("entries") or {}).get(lt) != (tb.get("entries") or {}).get(lt) for lt in letters):
        return False
    return [x for x in (tb.get("skipped") or []) if x.get("letter") in letters] == list(ts.get("skipped") or [])


def check_configs(name: str, cb: Mapping[str, Any], cs: Mapping[str, Any]) -> dict[str, Any] | None:
    """先核对的第 1、2 条（模块说明）。不过就 NotComparable；返回提示配置的等价记录（相同时 None）。"""
    bad = [f"{k}：基座 {cb.get(k)!r}，抽查 {cs.get(k)!r}" for k in SAME_KEYS if cb.get(k) != cs.get(k)]
    gb, gs = copy.deepcopy(cb.get("generation") or {}), copy.deepcopy(cs.get("generation") or {})
    tb, ts = (gb.get("readout") or {}).pop("table", None), (gs.get("readout") or {}).pop("table", None)
    if gb != gs:
        bad.append(f"generation：基座 {gb!r}，抽查 {gs!r}")
    if not _letter_table_ok(tb, ts):
        bad.append("generation.readout.table：抽查的字母不是基座的子集，或同一字母的写法 / token 不同")
    pb, ps = cb.get("prompt") or {}, cs.get("prompt") or {}
    bad += [f"prompt.{k}：基座 {pb.get(k)!r}，抽查 {ps.get(k)!r}" for k in ("entry", "count_questions") if pb.get(k) != ps.get(k)]
    hb, hs = (cb.get("data") or {}).get("has_gt"), (cs.get("data") or {}).get("has_gt")
    if hb != hs:
        bad.append(f"data.has_gt：基座 {hb!r}，抽查 {hs!r}")
    if bad:
        raise NotComparable(f"{name}：运行身份不同：" + "；".join(bad))
    return prompt_equivalence(pb.get("config_sha256"), ps.get("config_sha256"), str(cb.get("subset")))


def batch_boundaries(readout_kind: str, recs: Sequence[Mapping[str, Any]], batch_size: int) -> list[int]:
    """基座全量运行的批边界（runner.batch_plan，分批键与 runner.batch_keys 同一个函数），含末尾 len(recs)。"""
    rendered = [SimpleNamespace(extras=r.get("extras") or {}) for r in recs]
    return [b for _, b in batch_plan(batch_keys(readout_kind, rendered), int(batch_size))]


def check_items(name: str, cb: Mapping[str, Any], cs: Mapping[str, Any], rb_all: Sequence[Mapping[str, Any]],
                rs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """先核对的第 3、4 条（模块说明）。返回批组成的记录。"""
    nb, ns = len(rb_all), len(rs)
    kind = str(((cb.get("generation") or {}).get("readout") or {}).get("kind"))
    if kind not in READOUTS or any("raw_text" not in r for r in (*rb_all, *rs)):
        raise NotComparable(f"{name}：只比 run_e1 正式运行模式的输出（读法 {READOUTS}、每条有 raw_text），收到读法 {kind!r}；"
                            "事后读数（debiased、letter_swap、forced）的记录没有原始输出，不在本工具的范围内")
    if (cb.get("data") or {}).get("n_samples") != nb or (cb.get("data") or {}).get("sample_ids_sha256") != ids_digest(
            [r["id"] for r in rb_all]):
        raise NotComparable(f"{name}：基座不完整或与自己的 config.yaml 不符（predictions {nb} 条，n_samples "
                            f"{(cb.get('data') or {}).get('n_samples')}）")
    if ns == 0 or (cs.get("data") or {}).get("n_samples") != ns:
        raise NotComparable(f"{name}：抽查运行不完整（predictions {ns} 条，config.yaml 的 n_samples {(cs.get('data') or {}).get('n_samples')}）")
    if ns > nb:
        raise NotComparable(f"{name}：抽查 {ns} 条，多于基座的 {nb} 条")
    ids_b, ids_s = [r["id"] for r in rb_all[:ns]], [r["id"] for r in rs]
    if ids_s != ids_b:
        i = next(k for k, (a, b) in enumerate(zip(ids_b, ids_s)) if a != b)
        raise NotComparable(f"{name}：抽查的条目不是基座评测清单的前 {ns} 条：第 {i} 条基座 {ids_b[i]!r}，抽查 {ids_s[i]!r}")
    if (cs.get("data") or {}).get("sample_ids_sha256") != ids_digest(ids_s):
        raise NotComparable(f"{name}：抽查的 config.yaml 的 id 指纹与它的 predictions 不符")
    diff_in = [r["id"] for r, s in zip(rb_all, rs)
               if (r.get("instruction"), r.get("extras"), r.get("image_size")) != (s.get("instruction"), s.get("extras"), s.get("image_size"))]
    if diff_in:
        raise NotComparable(f"{name}：{len(diff_in)} 条的输入（指令原文、extras、原图尺寸）不同，如 {diff_in[:5]}")
    bounds = batch_boundaries(kind, rb_all, int(cb["batch_size"]))
    if ns not in bounds:
        lo = max([b for b in bounds if b < ns], default=0)
        hi = min(b for b in bounds if b > ns)
        raise NotComparable(f"{name}：N = {ns} 不在基座全量运行的批边界上（批大小 {cb['batch_size']}，读法 {kind}）：前 N 条的批组成与全量运行"
                            f"不同。最近的边界是 {lo} 与 {hi}。{BATCH_RULE}")
    return {"n_base": nb, "batch_size": int(cb["batch_size"]), "readout": kind, "n_on_batch_boundary": True,
            "n_batches_compared": bounds.index(ns) + 1, "n_batches_base": len(bounds),
            "full_batches_only": all(b % int(cb["batch_size"]) == 0 for b in bounds[: bounds.index(ns) + 1] if b != nb)}


# ---------------------------------------------------------------------------
# 比对
# ---------------------------------------------------------------------------


def _ci(x: Any) -> dict[str, Any] | None:
    if x is None:
        return None
    return {"estimate": x.estimate, "low": x.low, "high": x.high, "level": x.level, "n_resamples": x.n_resamples}


def _paired(a: np.ndarray, b: np.ndarray, statistic: Any = None) -> dict[str, Any] | None:
    try:
        return _ci(paired_bootstrap_ci(a, b, statistic=statistic))
    except ValueError:  # 重采样上统计量无定义（如累计并为 0）
        return None


def task_metric(task: str, rb: Sequence[Mapping[str, Any]], rs: Sequence[Mapping[str, Any]],
                gt_samples: Sequence[Any] | None) -> dict[str, Any]:
    """两边在这 N 条上的标题指标与配对差（抽查 − 基座）。"""
    if task == "ovd":
        if gt_samples is None:
            return {"name": "F1@0.5", "base": None, "spot": None, "diff": None,
                    "note": "检测的 F1@0.5 要 GT：给 --data-root（或设 VISINT_DATA_ROOT；--tiny 时取 tests/fixtures/e1）才算"}
        cb, cs = ovd_per_image_counts(gt_samples, rb), ovd_per_image_counts(gt_samples, rs)
        return {"name": "F1@0.5", "scope": f"前 {len(rb)} 张图合并（不按数据集分）", "base": f1_from_counts(cb),
                "spot": f1_from_counts(cs),
                "diff": _paired(np.asarray(cs, dtype=np.float64), np.asarray(cb, dtype=np.float64), f1_statistic)}
    name, ab, stat = per_sample_metric(task, rb)
    _, a_s, _ = per_sample_metric(task, rs)
    return {"name": name, "base": metric_of(name, rb), "spot": metric_of(name, rs), "diff": _paired(a_s, ab, stat)}


def compare_task(name: str, base_leaf: Path, spot_leaf: Path, *, gt_paths: Any = None) -> dict[str, Any]:
    cb, cs = read_config(base_leaf), read_config(spot_leaf)
    equiv = check_configs(name, cb, cs)
    rb_all, rs = read_predictions(base_leaf), read_predictions(spot_leaf)
    batch = check_items(name, cb, cs, rb_all, rs)
    rb = rb_all[: len(rs)]
    raw = [b.get("raw_text") == s.get("raw_text") for b, s in zip(rb, rs)]
    parsed = [(b.get("failure"), b.get("parsed")) == (s.get("failure"), s.get("parsed")) for b, s in zip(rb, rs)]
    score = [b.get("score") == s.get("score") for b, s in zip(rb, rs)]
    task = str(cb["task"])
    gt = None
    if task == "ovd" and gt_paths is not None:
        from src.e1.tasks import select

        by_id = {s.id: s for s in select(task, str(cb["subset"]), gt_paths, count_questions=DEFAULT_COUNT_QUESTIONS).samples}
        missing = [r["id"] for r in rb if r["id"] not in by_id]
        if missing:
            raise NotComparable(f"{name}：GT 里找不到这些图：{missing[:5]}")
        gt = [by_id[r["id"]] for r in rb]
    n = len(rs)
    differing = [b["id"] for b, ok in zip(rb, raw) if not ok]
    grid = [int(s["n_grid_tokens"]) for s in rs if "n_grid_tokens" in s]
    return {
        "task": task, "subset": cb["subset"], "n": n,
        "raw_identical": {"n": sum(raw), "rate": sum(raw) / n},
        "parsed_identical": {"n": sum(parsed), "rate": sum(parsed) / n},
        "score_identical": {"n": sum(score), "rate": sum(score) / n},
        "identical": all(raw),
        "differing_ids": differing[:MAX_LISTED_IDS], "n_differing": len(differing),
        "metric": task_metric(task, rb, rs, gt),
        "valid_rate": {"base": validity(task, rb)["rate"], "spot": validity(task, rs)["rate"]},
        "batch": batch,
        "prompt_config": {"base_sha256": (cb.get("prompt") or {}).get("config_sha256"),
                          "spot_sha256": (cs.get("prompt") or {}).get("config_sha256"), "equivalence": equiv},
        "spot_grid_tokens_in_output": None if not grid else {"n_rows": sum(1 for g in grid if g > 0), "n_tokens": sum(grid)},
        "spot_checkpoint": _ckpt_summary(cs),
        "base_limit": cb.get("limit"), "spot_limit": cs.get("limit"),
    }


def _ckpt_summary(cs: Mapping[str, Any]) -> dict[str, Any] | None:
    ck = cs.get("checkpoint")
    mg = cs.get("mask_grid_tokens")
    if ck is None and mg is None:
        return None
    ck = ck or {}
    return {"manifest_sha256": ck.get("manifest_sha256"), "label": ck.get("label"), "lora": ck.get("lora"),
            "n_grid_tokens": ck.get("n_grid_tokens"),
            "mask_grid_tokens": None if mg is None else {"enabled": mg.get("enabled"), "applied": mg.get("applied")}}


def _task_order(name: str) -> tuple[int, str]:
    head = name.split("__")[0]
    return (TASKS.index(head) if head in TASKS else len(TASKS), name)


def compare(base: str | Path, spot: str | Path, *, tasks: Sequence[str] | None = None, gt_paths: Any = None) -> dict[str, Any]:
    """两份 run_e1 的模型目录 → 报告（规则见模块说明）。先核对不过抛 NotComparable。"""
    db, ds = task_dirs(Path(base)), task_dirs(Path(spot))
    if tasks:
        miss = [t for t in tasks if t not in db or t not in ds]
        if miss:
            raise NotComparable(f"--task {miss} 不是两边都有的任务目录（基座有 {sorted(db)}，抽查有 {sorted(ds)}）")
        names = sorted(dict.fromkeys(tasks), key=_task_order)
    else:
        names = sorted(set(db) & set(ds), key=_task_order)
    if not names:
        raise NotComparable(f"两边没有共同的任务目录（基座 {sorted(db)}，抽查 {sorted(ds)}）")
    rows = {t: compare_task(t, db[t], ds[t], gt_paths=gt_paths) for t in names}
    same = all(r["identical"] for r in rows.values())
    n_all = sum(r["n"] for r in rows.values())
    n_raw = sum(r["raw_identical"]["n"] for r in rows.values())
    equivs = [r["prompt_config"]["equivalence"] for r in rows.values() if r["prompt_config"]["equivalence"]]
    return {
        "tool": TOOL, "verdict": VERDICT_SAME if same else VERDICT_DIFF, "verdict_rule": VERDICT_RULE, "batch_rule": BATCH_RULE,
        "base": str(base), "spot": str(spot), "model": read_config(db[names[0]])["model"],
        "tasks_compared": names, "tasks_only_in_base": sorted(set(db) - set(ds)), "tasks_only_in_spot": sorted(set(ds) - set(db)),
        "summary": {"n_tasks": len(names), "n_items": n_all, "n_raw_identical": n_raw, "raw_identical_rate": n_raw / n_all,
                    "tasks_not_identical": [t for t, r in rows.items() if not r["identical"]],
                    "prompt_config_equivalence": equivs[0] if equivs else None},
        "tasks": rows,
    }


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------


def _f(v: Any, nd: int = 4) -> str:
    return "—" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def _val(m: Mapping[str, Any] | None) -> str:
    if not m or m.get("value") is None:
        return "—"
    ci = m.get("ci")
    return _f(m["value"]) + (f" [{_f(ci['low'])}, {_f(ci['high'])}]" if ci else "")


def _diff(d: Mapping[str, Any] | None) -> str:
    return "—" if not d else f"{_f(d['estimate'])} [{_f(d['low'])}, {_f(d['high'])}]"


def markdown(rep: Mapping[str, Any]) -> str:
    if rep["verdict"] == VERDICT_NA:
        return "\n".join([f"# 冻结对照的 T 文本抽查：{VERDICT_NA}", "", f"- 基座：{rep.get('base')}", f"- 抽查：{rep.get('spot')}",
                          f"- 原因：{rep.get('reason')}", "", f"- 判定规则：{VERDICT_RULE}", f"- 批组成：{BATCH_RULE}", ""])
    s = rep["summary"]
    m = rep["model"]
    lines = [f"# 冻结对照的 T 文本抽查：{rep['verdict']}", "",
             f"- 基座：{rep['base']}", f"- 抽查：{rep['spot']}",
             f"- 模型：{m.get('key')}（{m.get('repo')}@{m.get('revision')}；tiny = {m.get('tiny')}）",
             f"- {s['n_tasks']} 个任务、{s['n_items']} 条；原始输出逐条相同 {s['n_raw_identical']}/{s['n_items']}"
             f"（{_f(s['raw_identical_rate'])}）"]
    if s["tasks_not_identical"]:
        lines.append(f"- 原始输出不完全相同的任务：{', '.join(s['tasks_not_identical'])}")
    if s["prompt_config_equivalence"]:
        e = s["prompt_config_equivalence"]
        lines.append(f"- 提示配置 sha256 不同但已裁定等价：基座 {e['base_sha256'][:8]}…，抽查 {e['spot_sha256'][:8]}…（{e['basis']}）")
    ck = next((r["spot_checkpoint"] for r in rep["tasks"].values() if r["spot_checkpoint"]), None)
    if ck:
        lines.append(f"- 抽查的检查点：manifest {str(ck['manifest_sha256'])[:16]}…，标签 {ck['label']}，LoRA {'无' if ck['lora'] is None else '有'}，"
                     f"网格 token {ck['n_grid_tokens']} 个，屏蔽网格 token {ck['mask_grid_tokens'] or '否'}")
    if rep["tasks_only_in_base"] or rep["tasks_only_in_spot"]:
        lines.append(f"- 只有基座有的任务（没比）：{rep['tasks_only_in_base'] or '无'}；只有抽查有的：{rep['tasks_only_in_spot'] or '无'}")
    lines += ["", "| 任务 | n | 原始输出相同 | 解析相同 | 得分相同 | 指标 | 基座 | 抽查 | 差（抽查 − 基座）[95% CI] | 有效率 基座 / 抽查 | 批 |",
              "|---|---:|---:|---:|---:|---|---|---|---|---|---:|"]
    for t, r in rep["tasks"].items():
        mt = r["metric"]
        lines.append(f"| {t} | {r['n']} | {_f(r['raw_identical']['rate'])} | {_f(r['parsed_identical']['rate'])} | "
                     f"{_f(r['score_identical']['rate'])} | {mt['name']} | {_val(mt['base'])} | {_val(mt['spot'])} | {_diff(mt['diff'])} | "
                     f"{_f(r['valid_rate']['base'])} / {_f(r['valid_rate']['spot'])} | {r['batch']['n_batches_compared']} |")
    notes = [f"- {t}：{r['metric']['note']}" for t, r in rep["tasks"].items() if r["metric"].get("note")]
    if notes:
        lines += ["", *notes]
    bad = {t: r for t, r in rep["tasks"].items() if not r["identical"]}
    if bad:
        lines += ["", "## 原始输出不同的条目", ""]
        for t, r in bad.items():
            more = f"（共 {r['n_differing']} 条，列前 {MAX_LISTED_IDS} 条）" if r["n_differing"] > MAX_LISTED_IDS else ""
            lines.append(f"- {t}：{r['n_differing']}/{r['n']} 条不同{more}；解析结果一致率 {_f(r['parsed_identical']['rate'])}："
                         + ", ".join(r["differing_ids"]))
    lines += ["", "## 规则", "", f"- 判定：{VERDICT_RULE}", f"- 批组成：{BATCH_RULE}", ""]
    return "\n".join(lines)


def write_report(rep: Mapping[str, Any], out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    md = out.with_suffix(".md")
    md.write_text(markdown(rep), encoding="utf-8")
    return md


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, help="基座主评测的模型目录（RUN-0027：experiments/E1/0027_e1_main/<模型>）")
    p.add_argument("--spot", required=True, help="抽查运行的模型目录（run_e1 --checkpoint … --limit N 的 <out>/<标签>/<模型>）")
    p.add_argument("--task", nargs="+", help="任务目录名（<任务>[__<子集>]）；默认两边都有的任务")
    p.add_argument("--out", required=True, help="报告 JSON 的路径（以 .json 结尾）；同名 .md 写在旁边")
    p.add_argument("--data-root", help="检测的 F1 要 GT：E1 的落地根（默认环境变量 VISINT_DATA_ROOT；都没有时检测的指标记 null）")
    p.add_argument("--tiny", action="store_true", help="开发机彩排：检测的 GT 取 tests/fixtures/e1（与 run_e1 --tiny 同一份小数据）")
    a = p.parse_args(argv)
    if not a.out.endswith(".json"):
        p.error(f"--out 必须以 .json 结尾，收到 {a.out!r}")
    out = Path(a.out)
    try:
        from src.e1.tasks import DataPaths

        root = a.data_root or os.environ.get("VISINT_DATA_ROOT")
        gt_paths = DataPaths.fixtures() if a.tiny and not a.data_root else (DataPaths(root=Path(root)) if root else None)
        rep = compare(a.base, a.spot, tasks=a.task, gt_paths=gt_paths)
    except Exception as e:  # noqa: BLE001 — 不可比（先核对不过）与出错都是退出码 2，原因写进报告
        reason = str(e) if isinstance(e, NotComparable) else f"出错：{e!r}"
        rep = {"tool": TOOL, "verdict": VERDICT_NA, "reason": reason, "base": a.base, "spot": a.spot}
        if not isinstance(e, NotComparable):
            rep["traceback"] = traceback.format_exc()[-6000:]
        write_report(rep, out)
        print(f"{VERDICT_NA}：{reason}", file=sys.stderr)
        return 2
    md = write_report(rep, out)
    print(md.read_text(encoding="utf-8").split("\n## 规则")[0])
    print(f"判定：{rep['verdict']}（{out}）")
    return 0 if rep["verdict"] == VERDICT_SAME else 1


if __name__ == "__main__":
    sys.exit(main())
