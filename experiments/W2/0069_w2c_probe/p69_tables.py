"""RUN-0069 报告用的表：数据、训练、判定、健康检查、T 文本各自的标题读数与显存峰值（只读结果文件）。
用法：python p69_tables.py experiments/W2/0069_w2c_hpcgpu06 experiments/W2/0067_w2c_data_hpcgpu06"""
import csv, json, statistics, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

O, D = Path(sys.argv[1]), Path(sys.argv[2])
CST = timezone(timedelta(hours=8))
M, SPLIT = "qwen3.5-9b", "devcheck2"


def load(p):
    return json.loads(p.read_text()) if p.exists() else None


def hl(m, name=None):
    out = []
    for h in m.get("headline") or []:
        if name is None or h["name"] == name:
            out.append(f"{h['name']} {h['value']:.4f} [{h['ci']['low']:.4f}, {h['ci']['high']:.4f}]（n {h.get('n')}）")
    return "；".join(out) or "—"


def rate(m, k):
    x = (m.get("pick") or {}).get(k)
    return f"{x['rate']:.3f}（{x['n']}/{x['of']}）" if x else "—"


# 数据
m = load(D / "metrics.json")
print("## 数据（RUN-0067 数据目录的 metrics.json）\n")
print(f"- n {m['n']}，n_with_candidates {m['n_with_candidates']}，k {m['k']}，names {m['names']}，complete {m['complete']}，分片 {m['n_shards']}")
print(f"- 出框失败 {m['proposal']['n_failed']}（{m['proposal']['failure_rate']:.4f}）：{m['proposal']['failures']}")
print(f"- 标签分布 counts {m['label']['counts']}；占有候选的比例 " + "，".join(f"{k}: {v:.4f}" for k, v in m['label']['rate_of_with_candidates'].items()))
print(f"- 最优候选平均 IoU（有候选） {m['best_iou']['mean_on_rows_with_candidates']:.4f}；1 号平均 IoU（有候选） {m['first_iou']['mean_on_rows_with_candidates']:.4f}，（全部句子、失败按 0） {m['first_iou']['mean_on_all_rows_failure_as_0']:.4f}；最优 − 1 号 {m['best_minus_first_mean']:.4f}")
print(f"- 各编号平均 IoU：" + "；".join(f"{k} {v['name']} {v['mean_iou']:.4f}（not_ok {v['n_not_ok']}）" for k, v in m['per_index'].items()))
print(f"- 重复候选 {m['duplicates']['n']}（{m['duplicates']['rate']:.4f}）；w2_config_sha256 {m['w2_config_sha256']}；manifest_sha256 {m['manifest_sha256']}")

# 训练
T = O / "0069_train/w2c" / M / "seed0"
s, a = load(T / "train_summary.json"), load(T / "accounting.json")
tim = [json.loads(l) for l in (T / "logs/timing_run_001.jsonl").read_text().splitlines() if l.strip()]
print("\n## 训练（0069_train/w2c/qwen3.5-9b/seed0）\n")
print(f"- train_summary：complete {s['complete']}，steps {s['steps']}，final_loss {s['final_loss']:.4f}（propose {s['final_loss_by_stream']['propose']:.4f}，select {s['final_loss_by_stream']['select']:.4f}），final_checkpoint_bytes {s['final_checkpoint_bytes']:,}，LoRA r {s['lora']['r']} alpha {s['lora']['lora_alpha']} n_parameters {s['lora']['n_parameters']:,}")
print(f"- accounting：n_runs {a['n_runs']}，wall_s {a['wall_s']:,.0f}，gpu_hours {a['gpu_hours']:.2f}，n_samples {a['n_samples']:,}，peak_mem_bytes {a['peak_mem_bytes']:,}")
st = [r["step_s"] for r in tim]
print(f"- timing_run_001.jsonl：{len(tim)} 步，step_s 中位 {statistics.median(st):.3f} s，第一步 {st[0]:.2f} s；fwd 中位 {statistics.median(r['fwd_s'] for r in tim):.3f} s，bwd 中位 {statistics.median(r['bwd_s'] for r in tim):.3f} s；peak_mem_bytes 最大 {max(r['peak_mem_bytes'] for r in tim):,}")

# 判定
print("\n## 判定（devcheck2，0069_eval）\n")
for leaf in ["refexp_box/baseline_e1", "refexp_box/propose_w2c"]:
    m = load(O / "0069_eval" / M / leaf / SPLIT / "metrics.json")
    if m:
        print(f"- {leaf}：{hl(m)}")
print("\n| 设置 | cIoU | gIoU | 挑中最优 | 挑 1 号 | 都不对（none） | 解析失败 | 出框失败 |")
print("|---|---|---|---|---|---|---|---|")
for st_ in ["w1f", "w2c"]:
    m = load(O / "0069_eval" / M / "refseg" / st_ / SPLIT / "metrics.json")
    f = {k: v for k, v in (m.get("failures") or {}).items() if k != "ok"}
    print(f"| {st_} | {hl(m, 'cIoU').split('（')[0].replace('cIoU ', '')} | {hl(m, 'gIoU').split('（')[0].replace('gIoU ', '')} | {rate(m, 'final_is_best')} | {rate(m, 'final_is_1')} | {rate(m, 'none')} | {rate(m, 'parse_failure')} | {f} |")
m = load(O / "0069_eval" / M / "refseg/w2c" / SPLIT / "metrics.json")
if m.get("select"):
    sel = m["select"]
    print(f"\n- w2c 的 select：answer {sel.get('answer')}；parse_failure {sel.get('parse_failure')}；stopped {sel.get('stopped')}；thinking_end {sel.get('thinking_end')}；final_index {m.get('final_index')}；best_index {m.get('best_index')}；oracle cIoU（有候选） {m.get('oracle_cIoU_on_samples_with_candidates'):.4f}")
for leaf in ["candidates", "candidates_w2c"]:
    c = load(O / "0069_eval" / M / "refseg" / leaf / SPLIT / "metrics.json")
    if c:
        print(f"- {leaf}：k {c.get('k')}，n_with_candidates {c.get('n_with_candidates')}，duplicate_rate {c.get('duplicate_rate'):.4f}（{c.get('n_duplicates')}），not_ok_by_index {c.get('not_ok_by_index')}")

# 健康检查
print("\n## 健康检查（health/）\n")
mc = load(O / "health/health_w2c_merge_check.json")
if mc:
    print(f"- health_w2c_merge_check.json：pass {mc.get('pass')}，criteria_version {mc.get('criteria_version')}，failures {json.dumps(mc.get('failures'), ensure_ascii=False)[:400]}")
for n in ["health_base", "health_w2c"]:
    h = load(O / "health" / f"{n}.json")
    print(f"- {n}.json：ok {h.get('ok')}，protocol_version {h.get('protocol_version')}，wall_time_s {h.get('wall_time_s')}，headline {json.dumps(h.get('headline'), ensure_ascii=False)[:600]}")
c = load(O / "health/health_compare.json")
print("\n| reading | metric | headline | n | base % | W2-c % | Δ pp | 95% CI pp | flag |")
print("|---|---|---|---|---|---|---|---|---|")
for r in c["rows"]:
    print(f"| {r['reading']} | {r['metric']} | {r['headline']} | {r['n']} | {r['base_pct']} | {r['merged_pct']} | {r['delta_pp']} | {r['ci95_pp']} | {r['flag']} |")

# T 文本
print("\n## T 文本读数（0069_ttext，subset eval）\n")
for task in ["refexp_box", "count"]:
    b = load(O / "0069_ttext" / M / task / "metrics.json")
    w = load(O / "0069_ttext/t_text/w2c/seed0" / M / task / "metrics.json")
    print(f"- {task}：基座 {hl(b)}；W2-c {hl(w)}")

# 显存
mem = []
for row in csv.reader((O / "gpu_mem.csv").read_text().splitlines()):
    try:
        mem.append((datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=CST), int(row[1].strip().split()[0])))
    except (ValueError, IndexError):
        pass
print(f"\n## 显存（gpu_mem.csv：{len(mem)} 行，{mem[0][0].astimezone(timezone.utc):%m-%d %H:%M:%SZ} – {mem[-1][0].astimezone(timezone.utc):%m-%d %H:%M:%SZ}，全程最大 {max(v for _, v in mem)} MiB）\n")
for line in (O / "steps.log").read_text().splitlines():
    parts = line.split()
    if len(parts) >= 4 and parts[2].startswith("rc="):
        t = datetime.strptime(parts[0], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        w = int(parts[3].split("=")[1])
        win = [v for ts, v in mem if t - timedelta(seconds=w) <= ts <= t]
        print(f"- {parts[1]}：wall_s {w:,}，窗口内最大 {max(win) if win else '—'} MiB（{len(win)} 行）")
