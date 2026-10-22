"""RUN-0063 report helper: loop statistics for the full-workbench runs, headline readings of every output,
GPU memory peaks per step, and gate.md verbatim.
usage: python p63_tables.py <OUT dir = experiments/W1/0063_w1_devcheck_hpcgpu07>"""
import collections as C, csv, json, re, statistics as st, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

O = Path(sys.argv[1])
M = O / "qwen3.5-9b"
J = lambda p: json.loads(Path(p).read_text())


def dist(xs):
    c = C.Counter(xs)
    return ", ".join(f"{k}: {v}" for k, v in sorted(c.items(), key=lambda kv: (kv[0] is None, str(kv[0]) if not isinstance(kv[0], int) else f"{kv[0]:06d}")))


def headline(d):
    m = J(d / "metrics.json")
    return "; ".join(f"{h['name']} {h['value']:.4f} [{h['ci']['low']:.4f}, {h['ci']['high']:.4f}] n={h['n']}" for h in m.get("headline", []))


print("## 回路各次（完整工作台）\n")
for rel, title in (("refseg/full/devcheck", "W1-1：refseg / full / devcheck"), ("refexp_box/full/devtune", "定位：refexp_box / full / devtune"),
                   ("count/full/devtune_countbench", "计数：count / full / devtune_countbench")):
    D = M / rel
    rows = [json.loads(x) for x in (D / "predictions.jsonl").read_text().splitlines() if x.strip()]
    per_round, refused = [], C.Counter()
    for r in rows:
        for rd in r.get("rounds") or []:
            tcs = rd.get("tool_calls") or []
            per_round.append(len(tcs))
            for tc in tcs:
                if tc.get("refused"):
                    refused[tc["refused"]] += 1
    gen = [g for g in ((r.get("tokens") or {}).get("generated") for r in rows) if g is not None]
    print(f"### {title}\n")
    print("| 项 | 值 |\n|---|---|")
    print(f"| 句数 | {len(rows)} |")
    print(f"| 提交（ended_by = submission） | {sum(r.get('ended_by') == 'submission' for r in rows)} |")
    print(f"| ended_by 分布 | {dist([r.get('ended_by') for r in rows])} |")
    print(f"| failure 分布 | {dist([r.get('failure') for r in rows])} |")
    print(f"| 轮数分布 | {dist([r.get('n_rounds') for r in rows])} |")
    print(f"| n_tool_calls 分布 | {dist([r.get('n_tool_calls') for r in rows])} |")
    print(f"| 每轮调用个数的分布（轮数） | {dist(per_round)} |")
    print(f"| 被拒调用按原因 | {dict(refused) if refused else '无'}（n_refused 合计 {sum(r.get('n_refused') or 0 for r in rows)}） |")
    print(f"| 工具调用出错（n_tool_call_errors 合计） | {sum(r.get('n_tool_call_errors') or 0 for r in rows)} |")
    print(f"| 每句生成 token 中位 / 最大 | {st.median(gen):.1f} / {max(gen)} |")
    print(f"| 标题读数 | {headline(D)} |")
    print(f"| tools_sha256 | {J(D / 'metrics.json').get('tools_sha256', '')[:8]}… |\n")

print("## 各输出的标题读数\n")
print("| 输出 | 标题读数 |\n|---|---|")
for d in sorted(p.parent for p in M.glob("*/*/*/metrics.json")):
    print(f"| {d.relative_to(M)} | {headline(d)} |")

steps, prev = {}, None
for line in (O / "steps.log").read_text().splitlines():
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

print("\n## gate.md 原文\n")
print((O / "gate.md").read_text())
