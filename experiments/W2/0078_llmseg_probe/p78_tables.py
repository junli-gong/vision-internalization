"""RUN-0078 报告用的表：5 个划分的 seg / rec 读数、状态、每图调用、截断、格式失败、坐标、gpu_summary（只读结果文件）。
用法：python p78_tables.py experiments/W2/0078_llmseg_hpcgpu06"""
import json, sys
from pathlib import Path

O = Path(sys.argv[1]) / "qwen3.5-9b/refseg/llmseg"
SPLITS = ["test_refcoco_unc_testA", "test_refcoco_unc_testB", "test_refcoco+_unc_testA", "test_refcoco+_unc_testB", "test_refcocog_umd_test"]
L = {s: json.loads((O / s / "metrics.json").read_text()) for s in SPLITS}
G = {s: json.loads((O / s / "logs/gpu_summary.json").read_text()) if (O / s / "logs/gpu_summary.json").exists() else None for s in SPLITS}


def h(m, r, k):
    x = {e["name"]: e for e in m["readouts"][r]["headline"]}[k]
    return f"{x['value']:.4f} [{x['ci']['low']:.4f}, {x['ci']['high']:.4f}]"


print("### 读数（n 300 / 划分；rec = 修正后，标题；seg = 修正前）\n")
print("| 划分 | seg cIoU | seg gIoU | rec cIoU | rec gIoU | ok / failed | complete |")
print("|---|---|---|---|---|---|---|")
for s in SPLITS:
    m = L[s]
    print(f"| {s} | {h(m,'seg','cIoU')} | {h(m,'seg','gIoU')} | {h(m,'rec','cIoU')} | {h(m,'rec','gIoU')} | {m['status'].get('ok')} / {m['status'].get('failed')} | {m['complete']} |")

print("\n### 计算记账\n")
print("| 划分 | 每图调用 均值 / p50 / max | 各阶段调用（detect / segment / focus_boxes / focus_splines / box_adjust） | 每图 prompt / 生成 token 均值 | 截断（全部） | 格式失败（全部） | 坐标 > 1000（值 / 调用） | 未解析调用 | max_abs |")
print("|---|---|---|---|---|---|---|---|---|")
for s in SPLITS:
    m = L[s]; c = m["calls_per_image"]; b = m["calls_by_stage"]; t = m["truncation"]["all"]; f = m["format_failure"]["all"]; k = m["coords"]
    cpi = m["readouts"]["rec"]["compute_per_image"]
    print(f"| {s} | {c['mean']:.2f} / {c['p50']:.0f} / {c['max']} | {b['detect']} / {b['segment']} / {b['focus_boxes']} / {b['focus_splines']} / {b['box_adjust']} | {cpi['prompt_tokens']:.0f} / {cpi['generated_tokens']:.0f} | {t['n']}/{t['of']} = {t['rate']:.4f} | {f['n']}/{f['of']} = {f['rate']:.4f} | {k['n_gt_1000']}/{k['n_values']} / {k['n_calls_gt_1000']} | {k['n_calls_unparsed']} | {k['max_abs']:.0f} |")

print("\n### 各阶段的截断率与格式失败率\n")
print("| 划分 | 截断 detect / segment / focus_boxes / focus_splines | 格式失败 detect / segment / focus_boxes / focus_splines |")
print("|---|---|---|")
for s in SPLITS:
    m = L[s]; st = ["detect", "segment", "focus_boxes", "focus_splines"]
    tr = " / ".join(f"{m['truncation'][x]['rate']:.4f}" for x in st); ff = " / ".join(f"{m['format_failure'][x]['rate']:.4f}" for x in st)
    print(f"| {s} | {tr} | {ff} |")

print("\n### gpu_summary.json（每个划分）\n")
print("| 划分 | 子批 | generate_wall_s | s / 条 | 外推 1,500 条 GPU 小时 | 阶段墙钟 detect / segment / focus_boxes / focus_splines（s） |")
print("|---|---|---|---|---|---|")
for s in SPLITS:
    g = G[s]
    if not g:
        print(f"| {s} | — | — | — | — | — |"); continue
    bs = g["by_stage"]
    print(f"| {s} | {g['n_sub_batches']} | {g['generate_wall_s']:,.0f} | {g['generate_wall_per_item_s']:.2f} | {g['extrapolate']['gpu_hours']:.2f} | " + " / ".join(f"{bs[x]['wall_s']:,.0f}" for x in ["detect", "segment", "focus_boxes", "focus_splines"]) + " |")

print("\n### 其他\n")
for s in SPLITS:
    m = L[s]
    print(f"- {s}：pipeline_events {m['pipeline_events']}；refine converged {m['refine']['converged']['n']}/{m['refine']['converged']['of']}，iterations p50 {m['refine']['iterations']['p50']:.0f}；n_empty_reply {m['n_empty_reply']}；n_input_rejected {m['n_input_rejected']}；生成 token / 图 p50 {m['tokens']['generated']['p50']:.0f}、max {m['tokens']['generated']['max']}")
