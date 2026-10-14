#!/usr/bin/env python3
"""RUN-0051 n0_all（N = 0 的 11 个读数）的事实表：起点合并核对、test 评测的完整性与有效率/解析失败率、墙钟。

只读结果目录，打印 Markdown。不打印任何分数（IoU、起伏误差、门判定等一律不读出）。
用法：python3 p51_n0.py [结果目录]（默认 ../0051_e5_main_hpcgpu07）
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parent / "0051_e5_main_hpcgpu07"
MODEL = "qwen3.5-2b"
SEEDS = (0, 1, 2)
STARTS = ("e3b_c3", "c3_warmup", "base_warmup")
E3B_SEEDS = (0, 2)
TASKS = ("rotation3d", "viewpoint", "occlusion", "physics", "imagery")  # 评测顺序（driver 日志）
METRIC_KEYS = ("mask_iou_grid", "mask_iou_full")  # 只读其中的 n / n_valid / valid_rate
# relief_error 的 valid_rate 不读：它的无效原因是 no_overlap（预测与目标不重叠），随预测质量变，按分数处理


def readouts():
    for s in SEEDS:
        for init in STARTS:
            yield init, s
        if s in E3B_SEEDS:
            yield "e3b", s


def load(p: Path):
    try:
        return json.loads(p.read_text())
    except FileNotFoundError:
        return None


def get(d, *path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def rel(p) -> str:
    p = str(p)
    for root in ("/home/mail/chenyuma/visint/repo/experiments/E5/0051_e5_main_hpcgpu07/", str(RES) + "/"):
        if p.startswith(root):
            return p[len(root):]
    return p


def fmt(x, nd=4):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, float):
        return f"{x:.{nd}g}" if abs(x) < 1e4 else f"{x:.1f}"
    return str(x)


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def task_dir(init, s, t):
    return RES / "trained" / init / MODEL / "n0" / f"seed{s}" / f"{t}__structured"


def run_jsons(d: Path):
    return sorted((d / "logs").glob("run_*.json"))


def driver_log(init, s):
    return RES / "driver" / "logs" / f"n0_{init}_s{s}.log"


def main():
    done = set((RES / "driver" / "done.txt").read_text().split()) if (RES / "driver" / "done.txt").exists() else set()
    gpu_s = (RES / "driver" / "gpu_seconds").read_text().strip() if (RES / "driver" / "gpu_seconds").exists() else None
    out = []
    w = out.append
    w(f"# RUN-0051 n0_all 事实表（{RES.name}）\n")
    try:
        snap = subprocess.run(["git", "-C", str(RES), "log", "-1", "--format=%h %cI %s", "--", "."],
                              capture_output=True, text=True, timeout=30).stdout.strip() or "—"
    except Exception as e:  # noqa: BLE001
        snap = f"git 不可用（{e}）"
    w(f"- 本地快照：最后一次改动这个目录的 commit `{snap}`。")
    w(f"- 结果目录：`{RES.relative_to(RES.parents[2]) if len(RES.parents) > 2 else RES}`；读数 11 个（种子 0–2 × e3b_c3、c3_warmup、base_warmup；e3b 只种子 0、2）。")
    w(f"- `driver/done.txt` 行数 {len(done)}；`driver/gpu_seconds` = {gpu_s}。")
    w("- 不含任何分数（IoU、起伏误差、门判定均不读出）。\n")

    # ---------- 1. 起点合并核对 ----------
    w("## 1. 起点合并核对\n")
    w("| 读数 | 核对文件 | pass | failures | restricted thinking_off exact_rows（n_divergent_rows / n_rows） | 判据下限 | timing_s.total_s | written_utc | run json 的 reused | 驱动日志 |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    all_checks = sorted((RES / "train").glob(f"*/{MODEL}/start_checks/merge_check_*.json"))
    referenced = set()
    for init, s in readouts():
        refs = {}
        reused = set()
        for t in TASKS:
            for rj in run_jsons(task_dir(init, s, t)):
                smc = get(load(rj), "start_merge_check") or {}
                if smc.get("file"):
                    refs.setdefault(smc["file"], []).append(t)
                    reused.add(smc.get("reused"))
        if len(refs) == 1:
            f = next(iter(refs))
            cpath = RES / rel(f)
            src = "run json"
        else:
            # 退路：按检查点目录匹配
            cpath, src = None, "按检查点目录匹配"
            want = f"/{'warmup' if init == 'base_warmup' else init}/{MODEL}/seed{s}/ckpt/final/model"
            for c in all_checks:
                if c.parts[-4] == init and str(get(load(c), "checkpoint", "dir") or "").endswith(want):
                    cpath = c
        referenced.add(cpath)
        doc = load(cpath) if cpath else None
        log = driver_log(init, s)
        if log.exists():
            txt = log.read_text(errors="replace")
            if "复用已通过" in txt:
                lg = "复用已通过的核对"
            elif "[合并核对] check_training_interfaces" in txt:
                lg = "当场跑了核对"
            else:
                lg = "日志里没有核对行"
        else:
            lg = "没有驱动日志"
        if doc is None:
            w(f"| {init} s{s} | 缺 | — | — | — | — | — | — | — | {lg} |")
            continue
        fails = doc.get("failures")
        fails_s = "; ".join(f"{k}: {len(v) if isinstance(v, list) else v}" for k, v in (fails or {}).items()) if isinstance(fails, dict) else fmt(fails)
        dvg = get(doc, "checks", "restricted", "by_thinking", "thinking_off", "decode_vs_generate_batch") or {}
        er = f"{fmt(dvg.get('exact_rows'))}（{fmt(dvg.get('n_divergent_rows'))} / {fmt(dvg.get('n_rows'))}）" if dvg else "—"
        thr = get(doc, "criteria", "restricted", "min_exact_rows_thinking_off")
        note = "" if len(refs) <= 1 else f"（run json 引用了 {len(refs)} 个不同文件）"
        if len(refs) == 1:
            tasks_ref = next(iter(refs.values()))
            note += f"（{len(tasks_ref)} 个任务的 run json 引用）"
        else:
            note += f"（{src}）"
        rs = ",".join(sorted(str(x) for x in reused)) if reused else "—"
        w(f"| {init} s{s} | `{rel(cpath)}`{note} | {fmt(doc.get('pass'))} | {fails_s} | {er} | ≥ {fmt(thr)} | "
          f"{fmt(get(doc, 'timing_s', 'total_s'), 5)} | {doc.get('written_utc')} | {rs} | {lg} |")
    extra = [c for c in all_checks if c not in referenced]
    w("")
    w(f"- `train/*/{MODEL}/start_checks/` 下 merge_check_*.json 共 {len(all_checks)} 个；未被上表引用的：" + (", ".join(f"`{rel(c)}`" for c in extra) if extra else "无") + "。")
    w("- failures 一栏是各项核对的失败条数（`failures.<核对>` 列表长度）。\n")

    # ---------- 2. test 评测：完整性、有效率与解析失败 ----------
    w("## 2. test 评测：完整性、有效率与解析失败率\n")
    w("字段（metrics.json）：`subset`、`n`、`n_expected`、`complete`；`parse.{n_ok,n_failed,failure_rate}`；`refusal.n`；"
      "`truncation.{n,rate}`；`thinking_unclosed.{n,rate}`；`overall.<指标>.{n_valid,n,valid_rate}`、`headline.<指标>.{n_valid,n,valid_rate}`。"
      "pred 行 = predictions.jsonl 的行数。\n")
    missing = []
    for t in TASKS:
        w(f"### {t}\n")
        mk_present = []
        for init, s in readouts():
            m = load(task_dir(init, s, t) / "metrics.json")
            if m:
                for k in METRIC_KEYS:
                    if k in (m.get("overall") or {}) and k not in mk_present:
                        mk_present.append(k)
        hdr = ["读数", "subset", "n", "n_expected", "complete", "pred 行", "parse ok/failed", "parse failure_rate",
               "refusal n", "truncation n (rate)", "thinking_unclosed n (rate)"]
        hdr += [f"overall {k} valid" for k in mk_present] + [f"headline {k} valid" for k in mk_present]
        w("| " + " | ".join(hdr) + " |")
        w("|" + "---|" * len(hdr))
        for init, s in readouts():
            d = task_dir(init, s, t)
            m = load(d / "metrics.json")
            pred = d / "predictions.jsonl"
            nrows = sum(1 for _ in pred.open()) if pred.exists() else None
            if m is None:
                missing.append(f"{init} s{s} {t}：" + ("目录在、没有 metrics.json" if d.exists() else "目录不存在")
                               + f"（目录里的文件：{', '.join(sorted(p.name for p in d.iterdir())) if d.exists() else '—'}）")
                w("| " + " | ".join([f"{init} s{s}", "缺 metrics.json"] + ["—"] * (len(hdr) - 2)) + " |")
                continue
            p = m.get("parse") or {}
            row = [f"{init} s{s}", fmt(m.get("subset")), fmt(m.get("n")), fmt(m.get("n_expected")), fmt(m.get("complete")),
                   fmt(nrows), f"{fmt(p.get('n_ok'))}/{fmt(p.get('n_failed'))}", fmt(p.get("failure_rate")),
                   fmt(get(m, "refusal", "n")),
                   f"{fmt(get(m, 'truncation', 'n'))} ({fmt(get(m, 'truncation', 'rate'))})",
                   f"{fmt(get(m, 'thinking_unclosed', 'n'))} ({fmt(get(m, 'thinking_unclosed', 'rate'))})"]
            for sec in ("overall", "headline"):
                for k in mk_present:
                    x = get(m, sec, k) or {}
                    row.append(f"{fmt(x.get('n_valid'))}/{fmt(x.get('n'))} ({fmt(x.get('valid_rate'))})" if x else "—")
            w("| " + " | ".join(row) + " |")
            if p.get("failures"):
                missing.append(f"{init} s{s} {t}：parse.failures 非空（{len(p['failures'])} 项）")
        w("")
    w("### 缺失与异常（第 2 节）\n")
    w("\n".join(f"- {x}" for x in missing) if missing else "- 无")
    w("")

    # ---------- 3. 墙钟 ----------
    w("## 3. 每个读数的墙钟\n")
    w("驱动日志第一行 `=== <时刻> chunk rc=<rc> wall_s=<s>`；各任务 `logs/run_*.json` 的 `started_utc`、`finished_utc`、`wall_s`（跨度 = 最早 started 到最晚 finished）。\n")
    w("| 读数 | done.txt | 驱动 rc | 驱动 wall_s | 驱动结束时刻 | 任务 run json 数 | 最早 started_utc | 最晚 finished_utc | 跨度 s | 各任务 wall_s 之和 | 各任务 wall_s（" + "/".join(TASKS) + "） |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    unfinished = []
    for init, s in readouts():
        log = driver_log(init, s)
        rc = wall = when = None
        if log.exists():
            mm = re.search(r"^=== (\S+) chunk rc=(\S+) wall_s=(\S+)", log.read_text(errors="replace"), re.M)
            if mm:
                when, rc, wall = mm.group(1), mm.group(2), mm.group(3)
        starts, fins, walls, nrun = [], [], [], 0
        per = []
        for t in TASKS:
            rjs = run_jsons(task_dir(init, s, t))
            nrun += len(rjs)
            tw = 0.0
            have = False
            for rj in rjs:
                r = load(rj) or {}
                if r.get("started_utc"):
                    starts.append(ts(r["started_utc"]))
                if r.get("finished_utc"):
                    fins.append(ts(r["finished_utc"]))
                else:
                    unfinished.append(f"{init} s{s} {t}：`{rel(rj)}` started_utc={r.get('started_utc')}，finished_utc={r.get('finished_utc')}")
                if r.get("wall_s") is not None:
                    tw += r["wall_s"]
                    have = True
            per.append(f"{tw:.0f}" if have else "—")
            if have:
                walls.append(tw)
        span = (max(fins) - min(starts)).total_seconds() if starts and fins else None
        w(f"| {init} s{s} | {'是' if f'n0_{init}_s{s}' in done else '否'} | {fmt(rc)} | {fmt(wall)} | {fmt(when)} | {nrun} | "
          f"{min(starts).isoformat() if starts else '—'} | {max(fins).isoformat() if fins else '—'} | "
          f"{fmt(span, 6)} | {sum(walls):.0f} | {'/'.join(per)} |")
    w("")
    w("- 没有 finished_utc 的 run json：" + ("；".join(unfinished) if unfinished else "无") + "。")
    nolog = [f"n0_{i}_s{k}" for i, k in readouts() if not driver_log(i, k).exists()]
    w("- 本地没有驱动日志 `driver/logs/n0_*.log` 的读数：" + ("、".join(nolog) if nolog else "无") + "。")
    w("- 跨度只算已结束的任务；任务有缺的读数，跨度与 wall_s 之和不是整个读数的墙钟。")
    w("")
    print("\n".join(out))


if __name__ == "__main__":
    main()
