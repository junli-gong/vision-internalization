"""RUN-0050 report helper: read the pushed results of eval.sh a / queue.sh / eval.sh b and print the report tables (no hand-typed numbers).
usage: python3 p50_tables.py <repo root>
Facts only: every number below is read from a file; nothing is compared across arms."""
import json, re, statistics, sys
from datetime import datetime
from pathlib import Path

R = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[3]
EV = R / "experiments/E3/0050_hpc_eval_08b_hpcgpu07"
TR = R / "experiments/E3/0050_hpc_e3b_hpcgpu07"
M = "qwen3.5-0.8b"
SEEDS = (0, 1, 2)
TASKS = ("count", "grid16", "ovd", "refexp_box", "refexp_poly", "rel_depth_blink", "rel_depth_nyu", "spatial_cvbench", "spatial_embspatial")
SUBSETS = ("eval", "debiased", "free", "letter_swap")
NA = "metrics.json 里没有"

ts = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))
J = lambda p: json.loads(Path(p).read_text())
jl = lambda p: [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def get(d, path, default=None):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def f4(x):
    return "—" if x is None else (f"{x:.4f}" if isinstance(x, float) else str(x))


def vci(o):  # {"value", "ci": {low, high}, "n"} -> "0.1234 [0.1000, 0.1500]"
    if not isinstance(o, dict):
        return NA
    s = f4(o.get("value"))
    ci = o.get("ci")
    if isinstance(ci, dict):
        s += f" [{f4(ci.get('low'))}, {f4(ci.get('high'))}]"
    return s


# ---------------------------------------------------------------- steps.log
def steps(path):
    segs, cur = {}, None
    for line in Path(path).read_text().splitlines():
        m = re.match(r"(\S+) start phase (\S+) commit (\S+) node (\S+)", line)
        if m:
            cur = m[2]; segs[cur] = {"start": ts(m[1]), "commit": m[3], "node": m[4], "steps": [], "done": None, "notes": []}; continue
        m = re.match(r"(\S+) (\S+) rc=(\d+) wall_s=(\d+)", line)
        if m and cur:
            segs[cur]["steps"].append((m[2], int(m[3]), int(m[4]), m[1])); continue
        m = re.match(r"(\S+) done phase (\S+); failed steps: (\d+)", line)
        if m:
            segs[m[2]]["done"] = (ts(m[1]), int(m[3])); continue
        if cur and ("检查点不在" in line):
            segs[cur]["notes"].append(line)
    return segs


SEG = steps(EV / "steps.log")
print("## steps.log（eval.sh a、b）\n")
for ph, s in SEG.items():
    print(f"### 段 {ph}：start {s['start']:%Y-%m-%dT%H:%M:%SZ}，commit `{s['commit']}`，node {s['node']}\n")
    print("| 步 | rc | wall_s | 结束（UTC） |")
    print("|---|---|---:|---|")
    for name, rc, w, end in s["steps"]:
        print(f"| {name} | {rc} | {w:,} | {end} |")
    tot = sum(w for _, _, w, _ in s["steps"])
    nz = [n for n, rc, _, _ in s["steps"] if rc != 0]
    span = (s["done"][0] - s["start"]).total_seconds() if s["done"] else None
    print(f"\n- 段 {ph}：{len(s['steps'])} 步，rc≠0 的步：{', '.join(nz) if nz else '无'}；各步 wall_s 合计 {tot:,} s（{tot / 3600:.2f} h）")
    if s["done"]:
        print(f"- `done phase {ph}; failed steps: {s['done'][1]}`，{s['done'][0]:%Y-%m-%dT%H:%M:%SZ}；start 到 done 的跨度 {span:,.0f} s（{span / 3600:.2f} h）")
    for n in s["notes"]:
        print(f"- 原文：`{n}`")
    print()

# ---------------------------------------------------------------- merge checks
print("## 合并核对（全部 merge_check*.json）\n")
MC = sorted(EV.rglob("merge_check*.json"))
print("| 文件（相对结果目录） | 检查点目录 | manifest_sha256 | checks | pass | failures | merge TF argmax_agreement（off / on） | merge TF gap.median（off / on） | merge generate exact_rows（off / on，只报告） | restricted decode_vs_generate_batch exact_rows | restricted TF agreement | restricted across_merge TF argmax_agreement | timing_s（各项） | total_s |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---:|")
for f in MC:
    d = J(f)
    a = d.get("argv", [])
    chk = []
    if "--checks" in a:
        for x in a[a.index("--checks") + 1:]:
            if x.startswith("--"):
                break
            chk.append(x)
    mb = get(d, "checks.merge.by_thinking", {}) or {}
    rb = get(d, "checks.restricted.by_thinking", {}) or {}
    ab = get(d, "checks.restricted.across_merge.by_thinking", {}) or {}
    pair = lambda p: " / ".join(f4(get(mb.get(t, {}), p)) for t in ("thinking_off", "thinking_on"))
    rex = "; ".join(f"{t}: {f4(get(v, 'decode_vs_generate_batch.exact_rows'))}" for t, v in rb.items()) or "（没有 restricted）"
    rtf = "; ".join(f"{t}: {f4(get(v, 'teacher_forcing.agreement'))}" for t, v in rb.items()) or "—"
    am = "; ".join(f"{t}: {f4(get(v, 'teacher_forcing.argmax_agreement'))}" for t, v in ab.items()) or "—"
    tm = d.get("timing_s", {}) or {}
    tms = ", ".join(f"{k} {v:.1f}" for k, v in tm.items() if k != "total_s")
    fails = json.dumps(d.get("failures"), ensure_ascii=False)
    print(f"| `{f.relative_to(EV)}` | `{get(d, 'checkpoint.dir')}` | `{(get(d, 'checkpoint.manifest_sha256') or '')[:16]}…` | {' '.join(chk)} | {d.get('pass')} | {fails} | "
          f"{pair('teacher_forcing.argmax_agreement')} | {pair('teacher_forcing.gap.median')} | {pair('generate.exact_rows')} | {rex} | {rtf} | {am} | {tms} | {tm.get('total_s', 0):.1f} |")
errs = [str(f.relative_to(EV)) for f in MC if J(f).get("error")]
print(f"\n- 共 {len(MC)} 份；error 非空的：{', '.join(errs) if errs else '无'}")
print("- 0.8B 基座没有检查点，没有合并核对文件。\n")

# ---------------------------------------------------------------- T-text
ARMS = [("e3b_c3", s, EV / f"t_text/e3b_c3/seed{s}/{M}") for s in SEEDS] + \
       [("c3", s, EV / f"t_text/c3/seed{s}/{M}") for s in SEEDS] + [("base", "—", EV / M)]


def tdir(root, task, sub):
    return root / (task if sub == "eval" else f"{task}__{sub}")


print("## T 文本：E1 各任务的严格、有效、条件三列（`three_columns`，原样取自 metrics.json）\n")
print("- 严格 = `three_columns[i].strict`（值 [95% CI]，n）；有效 = `three_columns[i].n_valid` / `three_columns[i].n`（`valid_rate`）；条件 = `three_columns[i].conditional`（值 [95% CI]，n）。")
print("- 子集 eval 的目录不带后缀；debiased、free、letter_swap 的目录是 `<任务>__<子集>`。某任务在某子集下没有目录，就不列。\n")
incomplete = []
for sub in SUBSETS:
    rows = []
    for task in TASKS:
        for arm, s, root in ARMS:
            p = tdir(root, task, sub) / "metrics.json"
            if not p.exists():
                continue
            d = J(p)
            if d.get("complete") is not True or d.get("n") != d.get("n_expected"):
                incomplete.append(f"{p.relative_to(EV)}（complete={d.get('complete')}, n={d.get('n')}, n_expected={d.get('n_expected')}）")
            tc = d.get("three_columns")
            if not tc:
                rows.append((task, NA, arm, s, NA, NA, NA)); continue
            for c in tc:
                st, cd = c.get("strict"), c.get("conditional")
                rows.append((task, f"{c.get('name')} / {c.get('scope')}", arm, s,
                             f"{vci(st)}，n={st.get('n') if isinstance(st, dict) else NA}",
                             f"{c.get('n_valid', NA)} / {c.get('n', NA)}（{f4(c.get('valid_rate'))}）",
                             f"{vci(cd)}，n={cd.get('n') if isinstance(cd, dict) else NA}"))
    if not rows:
        continue
    print(f"### 子集 {sub}\n")
    print("| 任务 | 指标 / scope | 变体 | 种子 | 严格 | 有效 | 条件 |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")
    print()

print("### 去偏读数（子集 debiased，`debiased` 一节）\n")
print("| 任务 | 变体 | 种子 | 原字母读数 accuracy（`debiased.original_letter.headline`） | 去偏读数 accuracy（`headline`） | `paired_diff`（去偏 − 原字母，同一次运行） | 改答案 / 错→对 / 对→错 | random_level | mean_p_gold（去偏 / 原） | k_counts |")
print("|---|---|---|---|---|---|---|---|---|---|")
for task in TASKS:
    for arm, s, root in ARMS:
        p = tdir(root, task, "debiased") / "metrics.json"
        if not p.exists():
            continue
        d = J(p); x = d.get("debiased")
        if not isinstance(x, dict):
            print(f"| {task} | {arm} | {s} | {NA} | | | | | | |"); continue
        ol = (get(x, "original_letter.headline") or [{}])[0]
        hl = (d.get("headline") or [{}])[0]
        pd = x.get("paired_diff", {})
        mp = x.get("mean_p_gold", {})
        print(f"| {task} | {arm} | {s} | {vci(ol)} | {vci(hl)} | {vci(pd) if pd else NA} | {pd.get('n_answer_changed', NA)} / {pd.get('n_wrong_to_right', NA)} / {pd.get('n_right_to_wrong', NA)} | "
              f"{f4(x.get('random_level'))} | {f4(mp.get('debiased'))} / {f4(mp.get('original'))} | {json.dumps(x.get('k_counts'))} |")
print()

print("### NYU 对调标记（子集 letter_swap，`letter_swap` 一节）\n")
print("| 变体 | 种子 | 读数 accuracy（`headline`） | 原图 accuracy（per_view original） | 对调图 accuracy（per_view labels_swapped） | 原图最常见字母份额 | 对调图最常见字母份额 | `paired_diff`（letter_swap 读数 − 原字母读数，同一次运行） | 改答案 / 错→对 / 对→错 | mean_p_gold（debiased / original 键） |")
print("|---|---|---|---|---|---|---|---|---|---|")
for arm, s, root in ARMS:
    p = tdir(root, "rel_depth_nyu", "letter_swap") / "metrics.json"
    if not p.exists():
        print(f"| {arm} | {s} | 目录不在 | | | | | | | |"); continue
    d = J(p); x = d.get("letter_swap")
    if not isinstance(x, dict):
        print(f"| {arm} | {s} | {NA} | | | | | | | |"); continue
    pv = {v.get("view"): v for v in x.get("per_view", [])}
    hl = (d.get("headline") or [{}])[0]
    pd = x.get("paired_diff", {}); mp = x.get("mean_p_gold", {})
    o, w = pv.get("original", {}), pv.get("labels_swapped", {})
    print(f"| {arm} | {s} | {vci(hl)} | {vci(o.get('accuracy')) if o else NA} | {vci(w.get('accuracy')) if w else NA} | {f4(o.get('most_common_letter_share'))}（{o.get('most_common_letter')}） | "
          f"{f4(w.get('most_common_letter_share'))}（{w.get('most_common_letter')}） | {vci(pd) if pd else NA} | {pd.get('n_answer_changed', NA)} / {pd.get('n_wrong_to_right', NA)} / {pd.get('n_right_to_wrong', NA)} | "
          f"{f4(mp.get('debiased'))} / {f4(mp.get('original'))} |")
print()
print(f"- complete 不是 true 或 n ≠ n_expected 的 metrics.json：{'; '.join(incomplete) if incomplete else '无'}\n")

# ---------------------------------------------------------------- structured grid
print("## 网格（结构强制，单轮，`single.guided` 与 `single.bilinear`）\n")
print("| 变体 | 种子 | 任务 | n | headline upsample | 上采样 | n_valid（valid_rate） | 指标 |")
print("|---|---|---|---|---|---|---|---|")
for v in ("e3b_c3", "e3b"):
    for s in SEEDS:
        for task, keys in (("refseg", ("cIoU", "gIoU")), ("depth", ("abs_rel", "d1", "d1_strict"))):
            p = EV / f"eval/{v}/{M}/seed{s}/{task}__structured/metrics.json"
            if not p.exists():
                print(f"| {v} | {s} | {task} | 目录不在 | | | | |"); continue
            d = J(p)
            for up in ("guided", "bilinear"):
                b = get(d, f"single.{up}")
                if not isinstance(b, dict):
                    print(f"| {v} | {s} | {task} | {d.get('n')} | {get(d, 'headline.upsample')} | {up} | {NA} | |"); continue
                ms = "；".join(f"{k} {vci(b.get(k))}" if k in b else f"{k} {NA}" for k in keys)
                print(f"| {v} | {s} | {task} | {d.get('n')} | {get(d, 'headline.upsample')} | {up} | {b.get('n_valid')}（{f4(b.get('valid_rate'))}） | {ms} |")
gc = []
for p in sorted(EV.glob(f"eval/*/{M}/seed*/*__structured/metrics.json")):
    d = J(p)
    gc.append(f"`{p.relative_to(EV)}`：complete={d.get('complete')}，checkpoint_manifest_sha256 `{(d.get('checkpoint_manifest_sha256') or '')[:16]}…`，config_sha256 `{(d.get('config_sha256') or '')[:16]}…`")
print("\n- " + "\n- ".join(gc) + "\n")

# per grid command: step wall (steps.log) vs evaluation run (logs/run_001.json); step start = previous steps.log line time
print("### 网格命令的墙钟拆分（steps.log 与 `<任务>__structured/logs/run_001.json`）\n")
print("| 步 | 命令 wall_s | 命令开始 → 评测运行 started_utc 的间隔 s | 评测运行 wall_s | 其中 score_wall_s | warmup_wall_s |")
print("|---|---:|---:|---:|---:|---:|")
for ph, sg in SEG.items():
    prev = sg["start"]
    for name, rc, w, end in sg["steps"]:
        m = re.match(r"grid_(e3b_c3|e3b)_s(\d)_(refseg|depth)$", name)
        if m:
            rj = EV / f"eval/{m[1]}/{M}/seed{m[2]}/{m[3]}__structured/logs/run_001.json"
            if rj.exists():
                r = J(rj)
                gap = (ts(r["started_utc"]) - prev).total_seconds()
                print(f"| {name} | {w:,} | {gap:,.0f} | {r.get('wall_s', 0):.1f} | {r.get('score_wall_s', 0):.1f} | {r.get('warmup_wall_s', 0):.1f} |")
            else:
                print(f"| {name} | {w:,} | run_001.json 不在 | | | |")
        prev = ts(end)
print("\n- 命令开始取 steps.log 上一行的时刻（段内第一步取 start 行）；refseg 命令含合并核对，depth 命令复用同目录已通过的 merge_check.json（见 `logs/grid_*_depth.log` 第一行）。\n")

# ---------------------------------------------------------------- E3b training
print("## E3b 训练（0.8B，`--supervision grid`）\n")
led = {}
lt = (TR / "ckpt_ledger.tsv").read_text().splitlines()
hdr = lt[0].split("\t")
for line in lt[1:]:
    if line.strip():
        r = dict(zip(hdr, line.split("\t"))); led[r["name"]] = r
print("| 种子 | 步数（train_summary.steps，complete） | 段数（驱动日志 chunk 行） | 各段 rc | 各段 wall_s 合计 | accounting wall_s | accounting gpu_hours | 首段开始 → 末段结束（run_*.json） | 每步中位 s（timing_run_*.jsonl，全部步） | 每步平均 s | 输入构建 / 前向 / 反向中位 s | 显存峰值（accounting peak_mem_bytes） | 最终 loss（train_log.jsonl 末行，step 键从 0 起） | 检查点（ckpt_ledger.tsv） |")
print("|---|---|---|---|---:|---:|---:|---|---:|---:|---|---|---|---|")
for s in SEEDS:
    leaf = TR / f"e3b/{M}/seed{s}"
    name = f"e3b_s{s}"
    summ = J(leaf / "train_summary.json") if (leaf / "train_summary.json").exists() else {}
    acc = J(leaf / "accounting.json") if (leaf / "accounting.json").exists() else {}
    dl = TR / f"driver/logs/{name}.log"
    chunks = re.findall(r"=== (\S+) chunk rc=(\d+) wall_s=(\d+)", dl.read_text()) if dl.exists() else []
    tm = [r for f in sorted((leaf / "logs").glob("timing_run_*.jsonl")) for r in jl(f)]
    runs = sorted((leaf / "logs").glob("run_*.json"))
    span = "—"
    if runs:
        r0, r1 = J(runs[0]), J(runs[-1])
        span = f"{r0.get('started_utc')} → {r1.get('finished_utc')}（{len(runs)} 份 run json）"
    last = jl(leaf / "train_log.jsonl")[-1] if (leaf / "train_log.jsonl").exists() else {}
    med = lambda k: statistics.median(r[k] for r in tm) if tm else None
    pk = acc.get("peak_mem_bytes")
    lg = led.get(name, {})
    print(f"| {s} | {summ.get('steps')}（{summ.get('complete')}） | {len(chunks)} | {','.join(c[1] for c in chunks)} | {sum(int(c[2]) for c in chunks):,} | {acc.get('wall_s'):,.1f} | {acc.get('gpu_hours'):.3f} | {span} | "
          f"{med('step_s'):.3f}（{len(tm)} 步） | {statistics.mean(r['step_s'] for r in tm):.3f} | {med('build_inputs_s'):.3f} / {med('fwd_s'):.3f} / {med('bwd_s'):.3f} | "
          f"{pk:,} B（{pk / 1e9:.2f} GB） | 第 {last.get('step')} 步 {f4(last.get('loss'))}（train_summary.final_loss {f4(summ.get('final_loss'))}） | `{lg.get('checkpoint')}`，{lg.get('kbytes')} KB，manifest_sha256 `{(lg.get('manifest_sha256') or '')[:16]}…` |")
print()
drv = (TR / "driver/driver.log").read_text().splitlines()
print("驱动日志（`driver/driver.log`，原文，不含“不推送”行）：\n\n```")
for line in drv:
    if "不推送" not in line:
        print(line)
print("```\n")
for fn in ("done.txt", "failed.txt"):
    p = TR / "driver" / fn
    print(f"- `driver/{fn}`：" + (", ".join(x for x in p.read_text().split() if x) if p.exists() else "不存在"))
print()

# ---------------------------------------------------------------- usage
print("## GPU 用量（文件里的数）\n")
gs = TR / "driver/gpu_seconds"
gsv = int(gs.read_text().strip()) if gs.exists() else None
tot = 0
for ph, s in SEG.items():
    w = sum(x[2] for x in s["steps"]); tot += w
    print(f"- 段 {ph}：steps.log 各步 wall_s 合计 {w:,} s（{w / 3600:.2f} h）")
if gsv is not None:
    print(f"- 训练：`driver/gpu_seconds` = {gsv:,} s（{gsv / 3600:.2f} h；驱动累计各段墙钟，不算暂停与等待）")
    tot += gsv
else:
    w = sum(int(c[2]) for s in SEEDS for c in re.findall(r"chunk rc=\d+ wall_s=(\d+)", (TR / f"driver/logs/e3b_s{s}.log").read_text()))
    print(f"- 训练：没有 gpu_seconds 文件；各段 wall_s 合计 {w:,} s"); tot += w
print(f"- 三项合计 {tot:,} s（{tot / 3600:.2f} h）")
