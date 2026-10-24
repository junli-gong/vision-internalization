"""RUN-0068 报告用的表：各设置标题读数、w2a 的挑选统计、候选统计、显存峰值（只读结果文件）。
用法：python p68_tables.py experiments/W2/0068_w2a_devcheck2_hpcgpu07"""
import csv, json, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

O = Path(sys.argv[1])
CST = timezone(timedelta(hours=8))
M, SPLIT = "qwen3.5-9b", "devcheck2"


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


print("## 各设置（devcheck2）\n")
m = load(O / M / "refexp_box/baseline_e1" / SPLIT / "metrics.json")
if m:
    print(f"- refexp_box/baseline_e1：acc@0.5 {hl(m, 'acc@0.5')}，n {m.get('n')}\n")
print("| 设置 | cIoU [95% CI] | gIoU [95% CI] | 挑中最优 | 挑 1 号 | none | 解析失败 | prompt token 中位/最大 | 生成 token 中位/最大 | 出框失败 | 模型调用均值 |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for s in ["w1f", "w2a", "ownbox", "random_pick", "oracle_pick"]:
    m = load(O / M / "refseg" / s / SPLIT / "metrics.json")
    if not m:
        print(f"| {s} | 缺 | | | | | | | | | |")
        continue
    f = {k: v for k, v in (m.get("failures") or {}).items() if k != "ok"}
    print(f"| {s} | {hl(m, 'cIoU')} | {hl(m, 'gIoU')} | {rate(m, 'final_is_best')} | {rate(m, 'final_is_1')} | {rate(m, 'none')} | {rate(m, 'parse_failure')} | {tok(m, 'prompt')} | {tok(m, 'generated')} | {f} | {(m.get('n_model_calls') or {}).get('mean', '—')} |")

m = load(O / M / "refseg/w2a" / SPLIT / "metrics.json")
if m and m.get("select"):
    s = m["select"]
    print(f"\n## w2a 的挑选（metrics.json 的 pick 与 select）\n")
    pk = m.get("pick") or {}
    wrong = None
    if "final_is_best" in pk:
        wrong = pk["final_is_best"]["of"] - pk["final_is_best"]["n"]
    print(f"- 挑中最优 {rate(m, 'final_is_best')}；挑 1 号 {rate(m, 'final_is_1')}；none {rate(m, 'none')}；解析失败 {rate(m, 'parse_failure')}")
    print(f"- answer {s.get('answer')}；parse_failure {s.get('parse_failure')}；stopped {s.get('stopped')}；thinking_end {s.get('thinking_end')}")
    print(f"- final_index {m.get('final_index')}；best_index {m.get('best_index')}；oracle cIoU（有候选） {m.get('oracle_cIoU_on_samples_with_candidates'):.4f}；n_with_candidates {m.get('n_with_candidates')}；config sha256 {m.get('w2_config_sha256')}")
    print(f"- pick 的规则：{pk.get('rule')}")

m = load(O / M / "refseg/candidates" / SPLIT / "metrics.json")
if m:
    print(f"\n## 候选\n\n- k {m.get('k')}，n_with_candidates {m.get('n_with_candidates')}，names {m.get('names')}，duplicate_rate {m.get('duplicate_rate'):.4f}（{m.get('n_duplicates')}），not_ok_by_index {m.get('not_ok_by_index')}")

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
for line in (O / "steps.log").read_text().splitlines():
    parts = line.split()
    if len(parts) >= 4 and parts[2].startswith("rc="):
        t = datetime.strptime(parts[0], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        w = int(parts[3].split("=")[1])
        win = [v for ts, v in mem if t - timedelta(seconds=w) <= ts <= t]
        print(f"- {parts[1]}：wall_s {w}，窗口内最大 {max(win) if win else '—'} MiB（{len(win)} 行）")
