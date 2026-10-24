"""RUN-0070 报告用的表：各臂标题读数、选择统计、候选统计、显存峰值（只读结果文件）。
用法：python p70_tables.py experiments/W2/0070_w2_tune_r2_hpcgpu06（或 07）"""
import csv, json, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

O = Path(sys.argv[1])
CST = timezone(timedelta(hours=8))
M = "qwen3.5-9b"


def load(p):
    return json.loads(p.read_text()) if p.exists() else None


def hl(m, name):
    for h in m.get("headline") or []:
        if h["name"] == name:
            return f"{h['value']:.4f} [{h['ci']['low']:.4f}, {h['ci']['high']:.4f}]"
    return "—"


def rate(m, k):
    x = (m.get("pick") or {}).get(k)
    return f"{x['rate']:.3f}（{x['n']}/{x['of']}）" if x else "—"


def tok(m, k):
    t = (m.get("tokens") or {}).get(k)
    return f"{t['p50']:.0f} / {t['max']}" if t else "—"


print("## 分割臂（refseg，devtune800）\n")
print("| 运行目录 | 叶 | cIoU [95% CI] | gIoU | 挑中最优 | 挑 1 号 | none | 解析失败 | prompt token 中位/最大 | 生成 token 中位/最大 | 出框失败 | 模型调用均值 |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for d in sorted(O.glob("0070_sep_*")):
    for leaf in sorted((d / M / "refseg").glob("*/devtune800/metrics.json")):
        m = load(leaf)
        if not m.get("headline"):
            continue
        f = {k: v for k, v in (m.get("failures") or {}).items() if k != "ok"}
        print(f"| {d.name} | {leaf.parent.parent.name} | {hl(m, 'cIoU')} | {hl(m, 'gIoU').split(' ')[0]} | {rate(m, 'final_is_best')} | {rate(m, 'final_is_1')} | {rate(m, 'none')} | {rate(m, 'parse_failure')} | {tok(m, 'prompt')} | {tok(m, 'generated')} | {f} | {(m.get('n_model_calls') or {}).get('mean', '—')} |")

print("\n## 选择阶段（metrics.json 的 select，只有 w2a 臂有）\n")
for d in sorted(O.glob("0070_sep_*")):
    m = load(d / M / "refseg/w2a/devtune800/metrics.json")
    if m and m.get("select"):
        s = m["select"]
        print(f"- {d.name}：answer {s.get('answer')}；parse_failure {s.get('parse_failure')}；stopped {s.get('stopped')}；thinking_end {s.get('thinking_end')}；final_index {m.get('final_index')}；best_index {m.get('best_index')}；oracle cIoU（有候选） {m.get('oracle_cIoU_on_samples_with_candidates'):.4f}；config sha256 {m.get('w2_config_sha256', '')[:12]}…")

print("\n## 候选（candidates*/metrics.json）\n")
for d in sorted(O.glob("0070_sep_*")):
    for leaf in sorted((d / M / "refseg").glob("candidates*/devtune800/metrics.json")):
        m = load(leaf)
        print(f"- {d.name}/{leaf.parent.parent.name}：k {m.get('k')}，n_with_candidates {m.get('n_with_candidates')}，names {m.get('names')}，duplicate_rate {m.get('duplicate_rate'):.4f}（{m.get('n_duplicates')}），not_ok_by_index {m.get('not_ok_by_index')}")

print("\n## 出框（refexp_box/baseline_e1）与 box_points 出点\n")
for d in sorted(O.glob("0070_sep_*")):
    m = load(d / M / "refexp_box/baseline_e1/devtune800/metrics.json")
    if m:
        print(f"- {d.name}：acc@0.5 {hl(m, 'acc@0.5')}，n {m.get('n')}")
    m = load(d / M / "refseg/propose_box_points/devtune800/metrics.json")
    if m:
        print(f"- {d.name}/propose_box_points：failures {m.get('failures')}，fg_points {json.dumps(m.get('fg_points'), ensure_ascii=False)[:200]}，bg_points {json.dumps(m.get('bg_points'), ensure_ascii=False)[:200]}，n_points_dropped {m.get('n_points_dropped')}，complete {m.get('complete')}")

# 显存：gpu_mem.csv 是节点本地时间（+08:00），steps.log 是 UTC；本运行可能没有这个文件
if not (O / "gpu_mem.csv").exists():
    print("\n## 显存\n\n- 结果目录里没有 gpu_mem.csv")
    sys.exit(0)
mem = []
for row in csv.reader((O / "gpu_mem.csv").read_text().splitlines()):
    try:
        mem.append((datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=CST), int(row[1].strip().split()[0])))
    except (ValueError, IndexError):
        pass
print(f"\n## 显存（gpu_mem.csv：{len(mem)} 行，{mem[0][0].astimezone(timezone.utc):%H:%M:%SZ} – {mem[-1][0].astimezone(timezone.utc):%H:%M:%SZ}，全程最大 {max(v for _, v in mem)} MiB）\n")
prev = None
for line in (O / "steps.log").read_text().splitlines():
    parts = line.split()
    if len(parts) >= 3 and parts[2].startswith("rc="):
        t = datetime.strptime(parts[0], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        w = int(parts[3].split("=")[1])
        win = [v for ts, v in mem if t - timedelta(seconds=w) <= ts <= t]
        print(f"- {parts[1]}：wall_s {w}，窗口内最大 {max(win) if win else '—'} MiB（{len(win)} 行）")
