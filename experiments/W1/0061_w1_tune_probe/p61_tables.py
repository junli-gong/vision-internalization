"""RUN-0061 report helper: per-arm loop statistics, baseline headlines, gate texts and GPU memory peaks (attempt 2).
usage: python p61_tables.py <OUT dir = experiments/W1/0061_w1_tune_hpcgpu07>"""
import collections as C, csv, json, re, statistics as st, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

O = Path(sys.argv[1])
M = "qwen3.5-9b"
J = lambda p: json.loads(Path(p).read_text())


def dist(xs):
    c = C.Counter(xs)
    return ", ".join(f"{k}: {v}" for k, v in sorted(c.items(), key=lambda kv: (kv[0] is None, str(kv[0]) if not isinstance(kv[0], int) else f"{kv[0]:06d}")))


def headline(d):
    m = J(d / "metrics.json")
    return "; ".join(f"{h['name']} {h['value']:.4f} [{h['ci']['low']:.4f}, {h['ci']['high']:.4f}] n={h['n']}" for h in m.get("headline", []))


print("## 各臂（refseg / full / devtune，n = 96）\n")
for k in ("T1", "T2", "T3", "T4"):
    D = O / f"0061_{k}" / M / "refseg/full/devtune"
    rows = [json.loads(x) for x in (D / "predictions.jsonl").read_text().splitlines() if x.strip()]
    per_round, refused, counted_calls = [], C.Counter(), []
    for r in rows:
        nc = 0
        for rd in r.get("rounds") or []:
            tcs = rd.get("tool_calls") or []
            per_round.append(len(tcs))
            for tc in tcs:
                if tc.get("refused"):
                    refused[tc["refused"]] += 1
                if tc.get("counted"):
                    nc += 1
        counted_calls.append(nc)
    gen = [(r.get("tokens") or {}).get("generated") for r in rows]
    gen = [g for g in gen if g is not None]
    meta = J(D / "metrics.json")
    print(f"### {k}\n")
    print("| 项 | 值 |\n|---|---|")
    print(f"| 句数 | {len(rows)} |")
    print(f"| 提交（ended_by = submission） | {sum(r.get('ended_by') == 'submission' for r in rows)} |")
    print(f"| ended_by 分布 | {dist([r.get('ended_by') for r in rows])} |")
    print(f"| failure 分布 | {dist([r.get('failure') for r in rows])} |")
    print(f"| 轮数分布 | {dist([r.get('n_rounds') for r in rows])} |")
    print(f"| n_tool_calls 分布 | {dist([r.get('n_tool_calls') for r in rows])} |")
    print(f"| 计数调用（counted = true）个数分布 | {dist(counted_calls)} |")
    print(f"| 每轮调用个数的分布（轮数） | {dist(per_round)} |")
    print(f"| 被拒调用按原因 | {dict(refused) if refused else '无'}（n_refused 合计 {sum(r.get('n_refused') or 0 for r in rows)}） |")
    print(f"| 工具调用出错（n_tool_call_errors 合计） | {sum(r.get('n_tool_call_errors') or 0 for r in rows)} |")
    print(f"| 每句生成 token 中位 / 最大 | {st.median(gen):.1f} / {max(gen)} |")
    print(f"| 标题读数 | {headline(D)} |")
    print(f"| tools_sha256 | {meta.get('tools_sha256', '')[:8]}… |\n")

print("## 基线（devtune，n = 96）\n")
print("| 步 | 标题读数 |\n|---|---|")
for task, setting in (("refexp_box", "baseline_e1"), ("refseg", "baseline_ownbox"), ("refseg", "baseline_poly_structured"), ("refseg", "w1f")):
    D = O / "0061_base" / M / task / setting / "devtune"
    print(f"| {task} / {setting} | {headline(D)} |")

steps, prev = {}, None
txt = (O / "steps.log").read_text()
txt = txt[txt.index("start commit 00fa15b1") - 21:]
for line in txt.splitlines():
    m = re.match(r"(\S+) start commit", line)
    if m:
        prev = datetime.fromisoformat(m[1].replace("Z", "+00:00")); continue
    m = re.match(r"(\S+) (\S+) rc=(\d+) wall_s=(\d+)", line)
    if m:
        end = datetime.fromisoformat(m[1].replace("Z", "+00:00")); steps[m[2]] = (prev, end, int(m[3]), int(m[4])); prev = end
CST = timezone(timedelta(hours=8))
mem = []
for row in csv.reader((O / "gpu_mem.csv").read_text().splitlines()):
    try:
        mem.append((datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=CST), int(row[1].strip().split()[0])))
    except (ValueError, IndexError):
        continue
print(f"\n## 显存（gpu_mem.csv：{len(mem)} 行，全程最大 {max(v for _, v in mem)} MiB）\n")
print("| 步 | rc | 墙钟 s | 窗口内显存峰值 MiB |\n|---|---|---|---|")
for n, (s, e, rc, w) in steps.items():
    win = [v for t, v in mem if s <= t <= e]
    print(f"| {n} | {rc} | {w} | {max(win) if win else 'n/a'} |")

for k in ("T1", "T2", "T3", "T4"):
    print(f"\n## gate_tune_{k}.md 原文\n")
    print((O / f"gate_tune_{k}.md").read_text())
