#!/usr/bin/env python
"""探索候选的判定（G-EX；DEV-0067）：照 D126-4（负责人原文 docs/lead_answers/2026-10-25-explore-directive.md 第四节）与 D126-5、D128
原样实现，规则写死，不留参数；在看到任何数据之前写好，与预注册一起固定。判定在 11-02。

    python scripts/gate_explore.py --candidate-name <候选名> --unseal-devcheck3 G-EX \\
        --reference <W2-c 的 out>/qwen3.5-9b/refseg/w2c/devcheck3_refcoco_unc_val <…>/devcheck3_refcoco+_unc_val <…>/devcheck3_refcocog_umd_val \\
        --candidate <候选在同样三个划分上的输出目录（3 个）> \\
        [--health-base <基座的 health_check.json> --health-candidate <候选的 health_check.json> | --health-compare <--compare 的输出>] \\
        --out-dir <目录>     # 写 <目录>/gate_explore.json 与 <目录>/gate_explore.md

输入：参照（W2-c，scripts/run_w2.py --setting w2c 的输出）与候选在 devcheck3 三个划分上的结果目录，都是 W2 格式的 predictions.jsonl
（每句一行，id 是样本 id，score 里有整数的 inter 与 union）；目录里有 W2 的 config.yaml 时另核对身份（写完、划分与它的 sha256、密封的门名）。
可选：候选与基座同机配对的健康检查（scripts/health_check.py 的两份输出 JSON，或 --compare 的输出）。最多 2 个候选：一次判一个，多个就分别
调用；不做多重比较校正（负责人的规则里没有）。读的是密封的 devcheck3，所以要显式给 --unseal-devcheck3 G-EX（src/data/explore_splits
的守卫）。

先核对（不满足就拒绝判定，退出码 2）：清单（--splits-file，默认 configs/explore_splits.json）已生成；两边的句子集合都恰是 devcheck3 的
500 句（id 与清单的 sha256 一致），没有重复，任何一边缺句或多句都拒绝；参照是 W2-c（设置 w2c、带训练的检查点、身份齐全）。

读数：500 句合并的 cIoU（ratio-of-sums，Σinter / Σunion），两边各一个；差值 = 候选 − W2-c。CI：配对 bootstrap，按句重抽 10,000 次，
97.5% 的双侧 CI，种子 src/constants.BOOTSTRAP_SEED——即 scripts/w2_gate.decide（同一个函数，不复制；样本按清单里组的样本 id 顺序排）。

规则（D126-4），满足其一即通过：
- (i) 差值 CI 下界 > 0；
- (ii) 差值 CI 下界 > −0.02，并且健康检查没有任何一项标红。标红口径即 scripts/health_check.py --compare（src/health/scoring.compare：
  标题读数的标题口径比基座低超过 5 个百分点；简报 §8.11，G-W2 §4 同一口径）。没给健康检查记为"无法判定"，不算通过；健康检查不完整
  （--limit、有项没评分、口径不可比、主机不同、不是一次跑完的运行、基座带检查点）也记为"无法判定"。
另附 gIoU（逐句 IoU 的平均，配对 bootstrap 同一水平与次数）与各层的 cIoU 作诊断，不入规则。
判定点核对（另行报出；不满足时照算，但结论标为"彩排，不作数"）：清单不是 --tiny、n = 500、两边都不是 --tiny 也没有 --limit、两边同一个
模型键、两边同一台机器（目录里 meta.json 的 hostname，D126-5：同机、同一时段配对）、参照的 W2 配置已冻结、两边（有 W2 身份的）都由
--unseal-devcheck3 G-EX 放行。
退出码：0 = 判完（通过与否不影响）；2 = 不能判定（输入核对不过、门名不对）。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src import constants as C  # noqa: E402
from src.data import explore_splits as EX  # noqa: E402
from src.e1.runner import CONFIG, PREDICTIONS, read_jsonl_tolerant  # noqa: E402
from src.health import protocol as HP  # noqa: E402
from src.health.scoring import compare as health_compare  # noqa: E402
from src.data.subsets import population_digest  # noqa: E402
from src.stats import paired_bootstrap_ci  # noqa: E402
from src.w1.scoring import exact_ciou  # noqa: E402


def _load_w2_gate():
    """scripts/w2_gate.py（同一套判定函数：decide、load_run 与写死的水平和次数）。scripts 不是包，按路径加载。"""
    spec = importlib.util.spec_from_file_location("_w2_gate_for_gate_explore", REPO_ROOT / "scripts" / "w2_gate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


W2G = _load_w2_gate()

# D126-4（负责人原文第四节；不得修改）
LEVEL = W2G.LEVEL  # 0.975
N_RESAMPLES = W2G.N_RESAMPLES  # 10,000
NONINFERIORITY_LOW = -0.02  # 规则 (ii)：差值 CI 下界 > −0.02（严格）
GROUP = EX.DEVCHECK3
GATE_N = EX.group_target_n(EX.DEVCHECK3)  # 500
REFERENCE_NAME = "W2-c"
REFERENCE_SETTING = "w2c"
PASS, FAIL, UNDECIDABLE = W2G.PASS, W2G.FAIL, "无法判定"
RULE_I = ("(i) 候选 − W2-c 的 cIoU 差（devcheck3 三层 500 句合并的 Σinter / Σunion），配对 bootstrap 97.5% CI（按句重抽 10,000 次，"
          "种子 src/constants.BOOTSTRAP_SEED），下界 > 0")
RULE_II = ("(ii) 同一 CI 的下界 > −0.02，并且健康检查没有任何一项标红（scripts/health_check.py --compare 的口径：标题读数比基座低超过 "
           f"{HP.RED_FLAG_DROP_PP} 个百分点）；没给健康检查或健康检查不完整记为无法判定")


class GateInputError(ValueError):
    """不能判定（清单没生成、句子集合不对、参照不是 W2-c、健康检查的输入不对……）。"""


# ---------------------------------------------------------------------------
# 读结果目录
# ---------------------------------------------------------------------------


def _w2_section(cfg: dict | None) -> dict | None:
    if not cfg:
        return None
    return cfg.get("w2") or ((cfg.get("data") or {}).get("selection_stats") or {}).get("w2")


def _check_records(recs: list, what: str) -> None:
    for r in recs:
        sc = r.get("score") if isinstance(r, dict) else None
        ok = isinstance(sc, dict) and all(isinstance(sc.get(k), int) and not isinstance(sc.get(k), bool) and sc[k] >= 0
                                          for k in ("inter", "union"))
        if not isinstance(r, dict) or not isinstance(r.get("id"), str) or not ok or sc["inter"] > sc["union"]:
            raise GateInputError(f"{what}：记录 {str(r)[:120]} 不是 W2 格式（id 与 score 里非负整数的 inter ≤ union）")


def load_dir(path: str | Path, what: str) -> dict:
    """一个结果目录（或它的 predictions.jsonl）：记录，以及 W2 的身份（有 config.yaml 且有 W2 一节时；写完的核对同 w2_gate.load_run）、
    密封的门名与主机名（meta.json）。"""
    p = Path(path)
    leaf = p.parent if p.name == PREDICTIONS else p
    if not (leaf / PREDICTIONS).exists():
        raise GateInputError(f"{what}：{leaf} 里没有 {PREDICTIONS}")
    cfg = yaml.safe_load((leaf / CONFIG).read_text(encoding="utf-8")) if (leaf / CONFIG).exists() else None
    w2 = _w2_section(cfg)
    if w2:
        ident, recs = W2G.load_run(leaf, what)
    else:
        ident, recs = None, read_jsonl_tolerant(leaf / PREDICTIONS, repair=False)
    _check_records(recs, f"{what}（{leaf}）")
    meta = json.loads((leaf / "meta.json").read_text(encoding="utf-8")) if (leaf / "meta.json").exists() else {}
    return {"leaf": str(leaf), "identity": ident, "unseal": (w2 or {}).get("unseal"), "host": meta.get("hostname"), "records": recs}


def merge_side(dirs: list[dict], doc: dict, what: str) -> tuple[list[dict], list[str]]:
    """一边的各目录合起来：样本 id 不重复，集合恰是 devcheck3 的 500 句（清单的 sha256）；有 W2 身份的目录，划分属于 devcheck3、
    sha256 与清单相同、记录都在这个划分里。返回 (按组的样本 id 顺序排好的记录, 组的样本 id)。"""
    want = EX.group_ids(doc, GROUP)
    split_of = EX.split_of_sample(doc, GROUP)
    by_id: dict[str, dict] = {}
    for d in dirs:
        ident = d["identity"]
        if ident is not None:
            if ident["split"] not in EX.group_split_names(GROUP):
                raise GateInputError(f"{what}：{d['leaf']} 的划分是 {ident['split']}，不是 devcheck3 的三个划分之一")
            if ident["split_sha256"] != doc["splits"][ident["split"]]["sha256"]:
                raise GateInputError(f"{what}：{d['leaf']} 的划分 sha256 与清单的 {ident['split']} 不同")
        for r in d["records"]:
            rid = r["id"]
            if rid in by_id:
                raise GateInputError(f"{what}：样本 {rid} 重复出现")
            if ident is not None and split_of.get(rid) != ident["split"]:
                raise GateInputError(f"{what}：{d['leaf']}（划分 {ident['split']}）里有不属于它的样本 {rid}")
            by_id[rid] = r
    missing = sorted(set(want) - set(by_id))
    extra = sorted(set(by_id) - set(want))
    if missing or extra:
        raise GateInputError(f"{what}：句子集合不是 devcheck3 的 {len(want)} 句——缺 {len(missing)} 句（如 {missing[:3]}），"
                             f"多 {len(extra)} 句（如 {extra[:3]}）；拒绝判定")
    if population_digest(list(by_id)) != doc["groups"][GROUP]["sha256"]:
        raise GateInputError(f"{what}：样本 id 的 sha256 与清单的组 {GROUP} 不同")
    return [by_id[i] for i in want], want


# ---------------------------------------------------------------------------
# 健康检查（标红口径 = src/health/scoring.compare，即 scripts/health_check.py --compare）
# ---------------------------------------------------------------------------


def _host_of(report: dict) -> str | None:
    return ((report.get("run") or {}).get("meta") or {}).get("hostname")


def health_status(base: str | None = None, candidate: str | None = None, compare_json: str | None = None,
                  candidate_checkpoints: list | None = None, reference_model: str | None = None) -> dict:
    """健康检查的读数与是否完整。given = False（没给）时 (ii) 无法判定；valid = False 时也无法判定（problems 写明）。"""
    if base is None and candidate is None and compare_json is None:
        return {"given": False, "valid": None, "any_flag": None, "flagged": None, "problems": ["没给健康检查"]}
    problems: list[str] = []
    hosts = None
    if compare_json is not None:
        if base is not None or candidate is not None:
            raise GateInputError("--health-compare 与 --health-base / --health-candidate 二选一")
        res = json.loads(Path(compare_json).read_text(encoding="utf-8"))
        if not isinstance(res, dict) or "rows" not in res or "any_flag" not in res:
            raise GateInputError(f"{compare_json} 不是 scripts/health_check.py --compare 的输出")
        if res.get("threshold_pp") != HP.RED_FLAG_DROP_PP:
            problems.append(f"--compare 的标红阈值 {res.get('threshold_pp')} 不是现在的 {HP.RED_FLAG_DROP_PP}")
        source = {"kind": "compare", "path": str(compare_json), "note": "同机未核对（--compare 的输出不带主机名）"}
    else:
        if base is None or candidate is None:
            raise GateInputError("--health-base 与 --health-candidate 要一起给")
        rb = json.loads(Path(base).read_text(encoding="utf-8"))
        rc = json.loads(Path(candidate).read_text(encoding="utf-8"))
        for nm, r in (("基座", rb), ("候选", rc)):
            if r.get("ok") is not True:
                problems.append(f"{nm}的健康检查不是一次跑完的运行（ok = {r.get('ok')!r}）")
        if rb.get("checkpoint") is not None:
            problems.append("基座的健康检查带了训练的检查点")
        if reference_model is not None and (rb.get("model") or {}).get("key") not in (None, reference_model):
            problems.append(f"基座的健康检查是 {(rb.get('model') or {}).get('key')}，不是 {reference_model}")
        ck = (rc.get("checkpoint") or {}).get("manifest_sha256")
        known = [c.get("manifest_sha256") for c in (candidate_checkpoints or []) if c]
        if ck is not None and known and any(k != ck for k in known):
            raise GateInputError("候选的健康检查用的检查点与候选在 devcheck3 上的运行不同（manifest sha256）")
        hosts = {"base": _host_of(rb), "candidate": _host_of(rc)}
        if None not in hosts.values() and hosts["base"] != hosts["candidate"]:
            problems.append(f"健康检查不是同机配对（{hosts}）")
        res = health_compare(rb, rc)
        source = {"kind": "pair", "base": str(base), "candidate": str(candidate)}
    if res.get("partial_run"):
        problems.append("健康检查用了 --limit（不完整）")
    if res.get("unscored"):
        problems.append(f"健康检查有项没评分：{res['unscored']}")
    heads = [r for r in res["rows"] if r.get("headline")]
    if res.get("incomparable") or any(not r.get("comparable") for r in heads):
        problems.append(f"健康检查有口径不可比的读数：{res.get('incomparable')}")
    flagged = [r["reading"] for r in res["rows"] if r.get("flag")]
    if bool(flagged) != bool(res["any_flag"]):
        raise GateInputError("健康检查的 any_flag 与逐行的 flag 不一致")
    return {"given": True, "valid": not problems, "problems": problems, "source": source, "hosts": hosts,
            "any_flag": bool(flagged), "flagged": flagged, "threshold_pp": res.get("threshold_pp"),
            "headline_rows": [{k: r.get(k) for k in ("reading", "metric", "base_pct", "merged_pct", "delta_pp", "ci95_pp", "flag")}
                              for r in heads]}


# ---------------------------------------------------------------------------
# 判定
# ---------------------------------------------------------------------------


def decide(candidate: list[dict], reference: list[dict], health: dict) -> dict:
    """两边按同一顺序排好的记录 → 读数、CI 与 (i)、(ii) 的结论。CI 即 scripts/w2_gate.decide（主配置 = 候选，W1-F 的位置 = W2-c）。"""
    d = W2G.decide(candidate, reference)
    rule_i = bool(d["ci_low"] > 0)
    if not health["given"] or not health["valid"]:
        rule_ii, why = UNDECIDABLE, "；".join(health["problems"])
    else:
        ok = d["ci_low"] > NONINFERIORITY_LOW and not health["any_flag"]
        rule_ii = PASS if ok else FAIL
        why = (f"CI 下界 {d['ci_low']:.4f} {'>' if d['ci_low'] > NONINFERIORITY_LOW else '≤'} {NONINFERIORITY_LOW}；"
               f"标红 {health['flagged'] or '无'}")
    passed = rule_i or rule_ii == PASS
    return {"n": d["n"], "ciou": {"candidate": d["ciou"]["main"], "reference": d["ciou"]["w1f"]}, "diff": d["diff"],
            "diff_exact": d["diff_exact"], "ci_low": d["ci_low"], "ci_high": d["ci_high"], "level": d["level"],
            "n_resamples": d["n_resamples"], "seed": d["seed"],
            "rules": {"i": {"rule": RULE_I, "pass": rule_i, "result": PASS if rule_i else FAIL},
                      "ii": {"rule": RULE_II, "result": rule_ii, "why": why, "noninferiority_low": NONINFERIORITY_LOW}},
            "pass": bool(passed), "decision": PASS if passed else FAIL}


def diagnostics(candidate: list[dict], reference: list[dict], split_of: dict[str, str]) -> dict:
    """不入规则：gIoU（逐句 IoU 的平均，配对 bootstrap 同一水平与次数）与各层的 cIoU。"""
    out: dict = {"note": "诊断，不入规则"}
    if all(isinstance((r.get("score") or {}).get("iou"), (int, float)) for r in candidate + reference):
        ic = np.array([float(r["score"]["iou"]) for r in candidate])
        ir = np.array([float(r["score"]["iou"]) for r in reference])
        ci = paired_bootstrap_ci(ic, ir, n_resamples=N_RESAMPLES, level=LEVEL, seed=C.BOOTSTRAP_SEED)
        out["giou"] = {"candidate": float(ic.mean()), "reference": float(ir.mean()), "diff": ci.estimate, "ci_low": ci.low,
                       "ci_high": ci.high, "level": ci.level, "n_resamples": ci.n_resamples}
    else:
        out["giou"] = None
    per = {}
    for nm in EX.group_split_names(GROUP):
        c = [r for r in candidate if split_of[r["id"]] == nm]
        r_ = [r for r in reference if split_of[r["id"]] == nm]
        cc, cr = exact_ciou(c), exact_ciou(r_)
        per[nm] = {"n": len(c), "candidate": float(cc), "reference": float(cr), "diff": float(cc - cr)}
    out["ciou_by_split"] = per
    return out


def evaluate(reference_dirs: list, candidate_dirs: list, candidate_name: str, *, splits_file: str | Path = EX.EXPLORE_SPLITS_PATH,
             unseal: str | None = None, health_base: str | None = None, health_candidate: str | None = None,
             health_compare_json: str | None = None) -> dict:
    record = EX.require_unsealed(GROUP, unseal)  # 读的是密封的 devcheck3：门名必须是 G-EX（SealedSplitError）
    doc = EX.load_explore_splits(splits_file)
    if EX.status(doc) != EX.STATUS_BUILT:
        raise GateInputError(f"{splits_file} 的 devcheck3 还没生成（status = pending）")
    ref_dirs = [load_dir(p, f"参照（{REFERENCE_NAME}）") for p in reference_dirs]
    cand_dirs = [load_dir(p, f"候选（{candidate_name}）") for p in candidate_dirs]
    for d in ref_dirs:
        i = d["identity"]
        if i is None:
            raise GateInputError(f"参照：{d['leaf']} 没有 W2 的身份（config.yaml）；参照必须是 scripts/run_w2.py --setting w2c 的输出")
        if i["task"] != W2G.GATE_TASK or i["setting"] != REFERENCE_SETTING or i["checkpoint"] is None:
            raise GateInputError(f"参照：{d['leaf']} 应是带检查点的 {W2G.GATE_TASK}/{REFERENCE_SETTING}，收到 {i['task']}/{i['setting']}、"
                                 f"checkpoint = {i['checkpoint']!r}")
    for key in ("model", "revision", "checkpoint"):
        vals = {json.dumps(d["identity"][key], sort_keys=True, default=str) for d in ref_dirs}
        if len(vals) != 1:
            raise GateInputError(f"参照的三个目录的 {key} 不同")
    ref_recs, ids = merge_side(ref_dirs, doc, f"参照（{REFERENCE_NAME}）")
    cand_recs, _ = merge_side(cand_dirs, doc, f"候选（{candidate_name}）")
    ref_ident = ref_dirs[0]["identity"]
    cand_idents = [d["identity"] for d in cand_dirs if d["identity"] is not None]
    health = health_status(health_base, health_candidate, health_compare_json,
                           candidate_checkpoints=[i["checkpoint"] for i in cand_idents], reference_model=ref_ident["model"])
    out = {"candidate_name": candidate_name, "reference_name": REFERENCE_NAME, **decide(cand_recs, ref_recs, health)}
    idents = [d["identity"] for d in ref_dirs + cand_dirs if d["identity"] is not None]
    hosts = {d["host"] for d in ref_dirs + cand_dirs}
    w2c = ref_ident.get("w2_config") or {}
    checks = {
        "manifest_not_tiny": not doc.get("tiny"),
        "n_is_500": out["n"] == GATE_N,
        "not_tiny": all(not i["tiny"] for i in idents),
        "no_limit": all(i["limit"] is None for i in idents),
        "same_model": all(i["model"] == ref_ident["model"] for i in idents),
        "same_host": None not in hosts and len(hosts) == 1,
        "reference_w2_config_frozen": w2c.get("frozen") is True,
        "unsealed_with_G-EX": all((d["unseal"] or {}).get("gate") == EX.GATE for d in ref_dirs + cand_dirs if d["identity"] is not None),
    }
    out["gate_point"] = {"checks": checks, "is_gate_point": all(checks.values()), "hosts": sorted(str(h) for h in hosts)}
    out["verdict"] = out["decision"] if all(checks.values()) else f"彩排，不作数（规则给出：{out['decision']}）"
    out["health"] = health
    out["diagnostics"] = diagnostics(cand_recs, ref_recs, EX.split_of_sample(doc, GROUP))
    out["manifest"] = {"path": str(splits_file), "sha256": EX.splits_sha256(doc), "group": GROUP, "group_sha256": doc["groups"][GROUP]["sha256"],
                       "n": len(ids)}
    out["runs"] = {"reference": [{"leaf": d["leaf"], "split": (d["identity"] or {}).get("split"), "host": d["host"],
                                  "checkpoint": (d["identity"] or {}).get("checkpoint")} for d in ref_dirs],
                   "candidate": [{"leaf": d["leaf"], "split": (d["identity"] or {}).get("split"), "host": d["host"],
                                  "setting": (d["identity"] or {}).get("setting"), "checkpoint": (d["identity"] or {}).get("checkpoint")}
                                 for d in cand_dirs]}
    out["identity"] = {k: ref_ident[k] for k in ("model", "revision", "tiny", "limit")}
    out["unseal"] = record
    return out


def _f(v) -> str:
    return "—" if v is None else f"{v:.4f}"


def markdown(doc: dict) -> str:
    r1, r2 = doc["rules"]["i"], doc["rules"]["ii"]
    h = doc["health"]
    g = doc["diagnostics"]["giou"]
    lines = [
        f"# 探索候选的判定 G-EX：{doc['candidate_name']} 对 {doc['reference_name']}（D126-4，规则写死）", "",
        f"- devcheck3（密封；{doc['manifest']['group_sha256'][:12]}…），n = {doc['n']}；模型 {doc['identity']['model']}；"
        f"判定点核对：{doc['gate_point']['checks']}", "",
        "| 读数（500 句合并） | 值 |", "|---|---|",
        f"| {doc['candidate_name']} cIoU | {doc['ciou']['candidate']:.4f} |",
        f"| {doc['reference_name']} cIoU | {doc['ciou']['reference']:.4f} |",
        f"| 差（候选 − W2-c） | {doc['diff']:.4f}（{doc['diff_exact']}） |",
        f"| {doc['level'] * 100:.1f}% CI | [{doc['ci_low']:.4f}, {doc['ci_high']:.4f}]（{doc['n_resamples']} 次，种子 {doc['seed']}） |", "",
        f"- {r1['rule']} → **{r1['result']}**",
        f"- {r2['rule']} → **{r2['result']}**（{r2['why']}）", "",
        f"**结论：{doc['verdict']}**", "",
        "健康检查：" + ("没给" if not h["given"] else
                    f"{'完整' if h['valid'] else '不完整：' + '；'.join(h['problems'])}；标红 {h['flagged'] or '无'}"),
        "",
        "诊断（不入规则）：",
        ("- gIoU：没有逐句的 iou" if g is None else
         f"- gIoU：候选 {g['candidate']:.4f}，W2-c {g['reference']:.4f}，差 {g['diff']:.4f}，{g['level'] * 100:.1f}% CI "
         f"[{_f(g['ci_low'])}, {_f(g['ci_high'])}]"),
    ]
    for nm, v in doc["diagnostics"]["ciou_by_split"].items():
        lines.append(f"- {nm}（n = {v['n']}）：cIoU 候选 {v['candidate']:.4f}，W2-c {v['reference']:.4f}，差 {v['diff']:+.4f}")
    lines += ["", "最多 2 个候选，各判一次；不做多重比较校正（负责人的规则里没有）。", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidate-name", required=True, help="候选的名字（写进报告）")
    p.add_argument("--reference", nargs="+", required=True, help="W2-c 在 devcheck3 三个划分上的输出目录（refseg/w2c/<划分>）")
    p.add_argument("--candidate", nargs="+", required=True, help="候选在 devcheck3 三个划分上的输出目录（W2 格式的 predictions.jsonl）")
    p.add_argument("--health-base", help="基座的健康检查（scripts/health_check.py 的输出 JSON）")
    p.add_argument("--health-candidate", help="候选的健康检查（与基座同机配对）")
    p.add_argument("--health-compare", help="或：scripts/health_check.py --compare 的输出")
    p.add_argument("--splits-file", default=str(EX.EXPLORE_SPLITS_PATH), help="默认 configs/explore_splits.json")
    p.add_argument("--unseal-devcheck3", metavar="GATE", help=f"读密封的 devcheck3：门名必须是 {EX.GATE}")
    p.add_argument("--out-dir", help="写 gate_explore.json 与 gate_explore.md 的目录")
    a = p.parse_args(argv)
    try:
        doc = evaluate(a.reference, a.candidate, a.candidate_name, splits_file=a.splits_file, unseal=a.unseal_devcheck3,
                       health_base=a.health_base, health_candidate=a.health_candidate, health_compare_json=a.health_compare)
    except (GateInputError, EX.SealedSplitError, ValueError) as e:  # W2G.GateInputError、SplitsError 都是 ValueError
        print(f"不能判定：{e}", file=sys.stderr)
        return 2
    md = markdown(doc)
    if a.out_dir:
        d = Path(a.out_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "gate_explore.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        (d / "gate_explore.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
