#!/usr/bin/env python
"""跨机器训练一致性的比对（DEV-0038 §3；D72-4：开训前 A800 与 B200 同一个种子、同一批数据各跑 50 步，损失曲线应在 bf16 噪声以内一致）。

输入两份训练叶目录（或其中的 train_log.jsonl；同目录下要有 config.yaml）：同一个运行身份、同一个种子、不同机器。适用于
scripts/train_e3b.py、train_e3a.py、train_e5.py 的输出（三者的 train_log.jsonl 每步一行，键 step、loss、grad_norm 与数据计数）。
    python scripts/compare_train_logs.py <A800 的叶目录> <B200 的叶目录> [--n-steps 50] [--out report.json] [--md report.md]

**先核对运行身份**：两边 config.yaml 逐键比较（展开成点分路径），除机器字段外必须相同，否则判"不可比"（退出码 2）。机器字段 =
路径的某一段在 MACHINE_FIELDS 里的键（主机、GPU、设备、本机路径；各训练脚本的运行身份本来就不含它们，它们在 meta.json 与 logs/），
另可用 --ignore-key 点分路径追加，追加的键照记进报告。种子另单独核对。meta.json 的 git 提交只报告、不判。

**逐步比较**（窗口 = 前 n_steps 步，两边都必须有第 0 … n_steps−1 步）：
- 数据与日程：每步除模型输出以外的字段（OUTPUT_FIELDS 以外的全部：样本数、各类 token 数、像素数、跳过的样本、学习率……）必须逐个相同，
  不同就说明两边喂的不是同一批数据或调度不同，判"不一致"；
- loss：相对差 = |a − b| / ((|a| + |b|) / 2)（两边都是 0 时记 0）；grad_norm 同样算，只报不判。

**判据**（CRITERIA；v1，暂定，管理员审）：
1. 窗口内每步 loss 的相对差 ≤ 2%，中位数 ≤ 0.5%；
2. 窗口最后 10 步的平均 loss 相对差 ≤ 1%（窗口不足 10 步时取整个窗口，只在 --tiny 彩排时出现）；
3. 两份日志的全部步里，loss、各任务 loss、grad_norm 没有 NaN 或 Inf。
全部满足判"一致"（退出码 0），否则"不一致"（退出码 1）。

**阈值的依据**（THRESHOLD_BASIS；暂定）：bf16 训练里不同 GPU（Ampere 的 A800 对 Blackwell 的 B200）的 GEMM、注意力内核的累加顺序不同，
结果只在舍入上有差别，但会随步数累积；正常应远小于"数据不同"或"学习率不同"造成的差（那是批与批之间的损失起伏，量级是百分之几到几十）。
开发机上的实测（DEV-0038；E5 训练器、base 起点、小模型、同一个种子与同一批数据、50 步）：
- 只换 GEMM 的归约顺序（同一台机器，CPU 线程数 1 对 6，fp32）：loss 相对差最大 1.7e-4、中位数 4.9e-5，最后 10 步平均 7.7e-5，随步数
  缓慢累积；grad_norm 最大 3%（所以 grad_norm 只报不判）。bf16 下 CPU 换线程数逐位相同（CPU 的 bf16 内核与线程数无关，量不出）。
- 对照，一边换了精度（同一台机器 fp32 对 bf16，比内核差异大得多）：loss 相对差最大 2.5%、中位数 1.2%。
2% / 0.5% / 1% 比"只换归约顺序"的噪声高两个数量级；中位数 0.5% 低于"精度不一致"的 1.2%，能抓出一边的精度设置不同一类的问题；
数据或学习率不同时，差是批与批之间的损失起伏（百分之几到几十），由逐步的数据字段与单步上限抓。小模型不能代表 2B 在两种 GPU 上的真实
bf16 噪声，所以阈值标为暂定：A800 与 B200 的第一次实测之后由管理员定稿（D72-4）。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

TRAIN_LOG = "train_log.jsonl"
CONFIG = "config.yaml"
META = "meta.json"
CRITERIA_VERSION = "v1-provisional"
CRITERIA = {"per_step_max": 0.02, "per_step_median": 0.005, "last_k": 10, "last_k_mean": 0.01}
THRESHOLD_BASIS = ("暂定（管理员审）：bf16 下不同 GPU 的内核累加顺序不同，差只在舍入上并随步数累积。开发机实测（小模型、50 步）：只换 GEMM 归约"
                   "顺序（CPU 线程数 1 对 6，fp32）loss 相对差最大 1.7e-4、中位数 4.9e-5；一边换精度（fp32 对 bf16）最大 2.5%、中位数 1.2%。"
                   "阈值比前者高两个数量级，中位数判据低于后者；数据或学习率不同时差是百分之几以上。A800 与 B200 第一次实测后定稿（D72-4）")
OUTPUT_FIELDS = ("loss", "loss_by_task", "grad_norm", "grid_token_acc", "mix")  # 由模型输出决定的字段；其余每步字段必须逐个相同
MACHINE_FIELDS = frozenset({"host", "hostname", "node", "gpu", "gpu_name", "gpus", "device", "devices", "cuda", "driver", "paths",
                            "ckpt_dir", "checkpoint_dir", "data_root", "out_dir", "started_utc", "finished_utc", "argv"})
VERDICT_OK, VERDICT_BAD, VERDICT_NA = "一致", "不一致", "不可比"


class NotComparable(ValueError):
    """两份日志不能比：运行身份不同、种子不同、步数不够或不连续、文件缺失。"""


def _leaf(p: str | Path) -> tuple[Path, Path]:
    p = Path(p)
    log = p / TRAIN_LOG if p.is_dir() else p
    if not log.is_file():
        raise NotComparable(f"{log} 不存在")
    leaf = log.parent
    if not (leaf / CONFIG).is_file():
        raise NotComparable(f"{leaf} 里没有 {CONFIG}（运行身份）")
    return leaf, log


def read_log(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    keep = raw.rfind(b"\n") + 1  # 末行不完整（中断）时丢掉
    return [json.loads(l) for l in raw[:keep].decode("utf-8").splitlines() if l.strip()]


def flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    """嵌套的映射 → {点分路径: 叶子值}；列表按整体比较。"""
    if isinstance(d, Mapping):
        out: dict[str, Any] = {}
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
        return out
    return {prefix: d}


def is_machine_key(path: str, extra: Sequence[str] = ()) -> bool:
    parts = path.split(".")
    return any(p in MACHINE_FIELDS for p in parts) or any(path == e or path.startswith(e + ".") for e in extra)


def identity_diff(a: Mapping[str, Any], b: Mapping[str, Any], extra: Sequence[str] = ()) -> tuple[list[str], list[str]]:
    """(挡住比较的不同键, 当作机器字段忽略的不同键)。"""
    fa, fb = flatten(a), flatten(b)
    diff = sorted(k for k in set(fa) | set(fb) if fa.get(k, "<缺>") != fb.get(k, "<缺>"))
    return [k for k in diff if not is_machine_key(k, extra)], [k for k in diff if is_machine_key(k, extra)]


def rel_diff(a: float, b: float) -> float:
    a, b = float(a), float(b)
    den = (abs(a) + abs(b)) / 2.0
    return 0.0 if den == 0.0 else abs(a - b) / den


def _finite_all(rec: Mapping[str, Any]) -> list[str]:
    """一步里 loss、各任务 loss、grad_norm 的非有限值（键名）。"""
    bad = []
    for k in ("loss", "grad_norm"):
        v = rec.get(k)
        if v is None or not math.isfinite(float(v)):
            bad.append(k)
    for t, v in (rec.get("loss_by_task") or {}).items():
        if v is not None and not math.isfinite(float(v)):
            bad.append(f"loss_by_task.{t}")
    return bad


def _stats(v: Sequence[float]) -> dict[str, float | None]:
    return {"max": max(v) if v else None, "median": statistics.median(v) if v else None,
            "mean": (sum(v) / len(v)) if v else None}


def compare(a: str | Path, b: str | Path, *, n_steps: int = 50, ignore_keys: Sequence[str] = ()) -> dict[str, Any]:
    """两份训练日志 → 报告（verdict = 一致 / 不一致；不可比时抛 NotComparable）。"""
    (la, pa), (lb, pb) = _leaf(a), _leaf(b)
    ca, cb = (yaml.safe_load((l / CONFIG).read_text(encoding="utf-8")) or {} for l in (la, lb))
    blocking, ignored = identity_diff(ca, cb, ignore_keys)
    if blocking:
        raise NotComparable(f"两边的运行身份（config.yaml）除机器字段外不同：{blocking[:12]}" + ("…" if len(blocking) > 12 else ""))
    if ca.get("seed") != cb.get("seed"):
        raise NotComparable(f"种子不同：{ca.get('seed')} 对 {cb.get('seed')}")
    ra, rb = read_log(pa), read_log(pb)
    for name, recs in (("A", ra), ("B", rb)):
        steps = [int(r["step"]) for r in recs]
        if steps[:n_steps] != list(range(n_steps)):
            raise NotComparable(f"{name} 的 train_log 没有连续的第 0 … {n_steps - 1} 步（有 {len(steps)} 步）")
    window = list(zip(ra[:n_steps], rb[:n_steps]))
    rows, data_mismatch = [], []
    for x, y in window:
        da = {k: v for k, v in x.items() if k not in OUTPUT_FIELDS}
        db = {k: v for k, v in y.items() if k not in OUTPUT_FIELDS}
        if da != db:
            data_mismatch.append({"step": x["step"], "keys": sorted(k for k in set(da) | set(db) if da.get(k) != db.get(k))})
        ok = not _finite_all(x) and not _finite_all(y)
        rows.append({"step": int(x["step"]), "loss_a": x.get("loss"), "loss_b": y.get("loss"),
                     "loss_rel": rel_diff(x["loss"], y["loss"]) if ok else None,
                     "grad_norm_a": x.get("grad_norm"), "grad_norm_b": y.get("grad_norm"),
                     "grad_norm_rel": rel_diff(x["grad_norm"], y["grad_norm"]) if ok else None})
    nonfinite = [{"log": n, "step": int(r["step"]), "keys": _finite_all(r)} for n, recs in (("A", ra), ("B", rb)) for r in recs
                 if _finite_all(r)]
    lr = [r["loss_rel"] for r in rows if r["loss_rel"] is not None]
    gr = [r["grad_norm_rel"] for r in rows if r["grad_norm_rel"] is not None]
    k = min(int(CRITERIA["last_k"]), n_steps)  # 窗口不足 10 步（--tiny 彩排）时取整个窗口，报告里记 k
    tail = window[-k:]
    last_a = sum(float(x["loss"]) for x, _ in tail) / len(tail)
    last_b = sum(float(y["loss"]) for _, y in tail) / len(tail)
    last_rel = rel_diff(last_a, last_b) if math.isfinite(last_a) and math.isfinite(last_b) else None
    checks = {
        "data_identical": {"pass": not data_mismatch, "mismatches": data_mismatch[:20], "n_mismatch": len(data_mismatch),
                           "rule": f"每步除 {list(OUTPUT_FIELDS)} 以外的字段逐个相同（同一批数据、同一个调度）"},
        "per_step_max": {"pass": bool(lr) and len(lr) == len(rows) and max(lr) <= CRITERIA["per_step_max"],
                         "value": max(lr) if lr else None, "threshold": CRITERIA["per_step_max"]},
        "per_step_median": {"pass": bool(lr) and statistics.median(lr) <= CRITERIA["per_step_median"],
                            "value": statistics.median(lr) if lr else None, "threshold": CRITERIA["per_step_median"]},
        "last_k_mean": {"pass": last_rel is not None and last_rel <= CRITERIA["last_k_mean"], "value": last_rel,
                        "threshold": CRITERIA["last_k_mean"], "k": k, "mean_a": last_a, "mean_b": last_b},
        "finite": {"pass": not nonfinite, "nonfinite": nonfinite[:20], "n_nonfinite": len(nonfinite)},
    }
    verdict = VERDICT_OK if all(c["pass"] for c in checks.values()) else VERDICT_BAD
    meta = {}
    for name, l in (("a", la), ("b", lb)):
        try:
            m = json.loads((l / META).read_text(encoding="utf-8"))
            meta[name] = {"git": m.get("git"), "host": m.get("host") or m.get("hostname")}
        except (OSError, ValueError):
            meta[name] = None
    return {"verdict": verdict, "criteria_version": CRITERIA_VERSION, "criteria": dict(CRITERIA), "threshold_basis": THRESHOLD_BASIS,
            "a": str(pa), "b": str(pb), "seed": ca.get("seed"), "n_steps": n_steps,
            "identity": {"ignored_machine_keys": ignored, "extra_ignore_keys": list(ignore_keys)},
            "checks": checks, "loss_rel": _stats(lr), "grad_norm_rel": {**_stats(gr), "note": "只报不判"},
            "steps_total": {"a": len(ra), "b": len(rb)}, "meta": meta, "rows": rows}


def markdown(rep: Mapping[str, Any]) -> str:
    def f(v: Any, nd: int = 6) -> str:
        return "—" if v is None else (f"{v:.{nd}g}" if isinstance(v, float) else str(v))

    c = rep["checks"]
    lines = [f"# 跨机器训练一致性：{rep['verdict']}", "",
             f"- A：{rep['a']}", f"- B：{rep['b']}", f"- 种子 {rep['seed']}；窗口 = 前 {rep['n_steps']} 步；判据 {rep['criteria_version']}",
             f"- 忽略的机器字段：{rep['identity']['ignored_machine_keys'] or '无'}",
             f"- 数据与日程逐步相同：{c['data_identical']['pass']}（不同的步 {c['data_identical']['n_mismatch']}）",
             f"- 每步 loss 相对差：最大 {f(c['per_step_max']['value'])}（≤ {CRITERIA['per_step_max']}：{c['per_step_max']['pass']}），"
             f"中位数 {f(c['per_step_median']['value'])}（≤ {CRITERIA['per_step_median']}：{c['per_step_median']['pass']}）",
             f"- 最后 {c['last_k_mean']['k']} 步平均 loss 的相对差 {f(c['last_k_mean']['value'])}（≤ {CRITERIA['last_k_mean']}："
             f"{c['last_k_mean']['pass']}）",
             f"- NaN / Inf：{c['finite']['n_nonfinite']} 处（{c['finite']['pass']}）",
             f"- grad_norm 相对差（只报不判）：最大 {f(rep['grad_norm_rel']['max'])}，中位数 {f(rep['grad_norm_rel']['median'])}",
             f"- 阈值的依据：{rep['threshold_basis']}", "",
             "| 步 | loss A | loss B | 相对差 | grad_norm A | grad_norm B | 相对差 |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rep["rows"]:
        lines.append(f"| {r['step']} | {f(r['loss_a'])} | {f(r['loss_b'])} | {f(r['loss_rel'], 3)} | {f(r['grad_norm_a'])} | "
                     f"{f(r['grad_norm_b'])} | {f(r['grad_norm_rel'], 3)} |")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("a", help="A 的训练叶目录（或其中的 train_log.jsonl）")
    p.add_argument("b", help="B 的训练叶目录（或其中的 train_log.jsonl）")
    p.add_argument("--n-steps", type=int, default=50, help="比较的窗口：前 N 步（默认 50，D72-4；--tiny 的日志只有几步，彩排时给小值）")
    p.add_argument("--ignore-key", action="append", default=[], help="另当作机器字段的 config.yaml 点分路径（可重复；照记进报告）")
    p.add_argument("--out", help="报告 JSON 的路径")
    p.add_argument("--md", help="报告 Markdown 的路径（默认只打印）")
    a = p.parse_args(argv)
    if a.n_steps < 1:
        p.error("--n-steps 必须是正整数")
    try:
        rep = compare(a.a, a.b, n_steps=a.n_steps, ignore_keys=a.ignore_key)
    except NotComparable as e:
        print(f"{VERDICT_NA}：{e}", file=sys.stderr)
        if a.out:
            Path(a.out).write_text(json.dumps({"verdict": VERDICT_NA, "reason": str(e)}, ensure_ascii=False, indent=1) + "\n",
                                   encoding="utf-8")
        return 2
    md = markdown(rep)
    if a.out:
        Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if a.md:
        Path(a.md).write_text(md, encoding="utf-8")
    print(md.split("\n| 步 |")[0])
    print(f"判定：{rep['verdict']}")
    return 0 if rep["verdict"] == VERDICT_OK else 1


if __name__ == "__main__":
    sys.exit(main())
