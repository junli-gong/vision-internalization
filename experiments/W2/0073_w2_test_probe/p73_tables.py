"""RUN-0073 报告用的表：两个模型 × 5 个 test 划分的各设置标题读数与 w2a 的挑选统计（只读结果文件）。
用法：python p73_tables.py experiments/W2/0073_w2_test_hpcgpu07"""
import json, sys
from pathlib import Path

O = Path(sys.argv[1])
SPLITS = ["test_refcoco_unc_testA", "test_refcoco_unc_testB", "test_refcoco+_unc_testA", "test_refcoco+_unc_testB", "test_refcocog_umd_test"]


def load(p):
    return json.loads(p.read_text()) if p.exists() else None


def hl(m, name):
    for h in (m or {}).get("headline") or []:
        if h["name"] == name:
            return f"{h['value']:.4f} [{h['ci']['low']:.4f}, {h['ci']['high']:.4f}]"
    return "—"


def rate(m, k):
    x = ((m or {}).get("pick") or {}).get(k)
    return f"{x['rate']:.3f}（{x['n']}/{x['of']}）" if x else "—"


for M in ["qwen3.5-2b", "qwen3.5-4b"]:
    print(f"## {M}\n")
    print("| 划分 | E1 框 acc@0.5 | ownbox cIoU | w1f cIoU | w1f gIoU | w2a cIoU | w2a gIoU | n | complete |")
    print("|---|---|---|---|---|---|---|---|---|")
    for sp in SPLITS:
        e1 = load(O / M / "refexp_box/baseline_e1" / sp / "metrics.json")
        ob = load(O / M / "refseg/ownbox" / sp / "metrics.json")
        w1 = load(O / M / "refseg/w1f" / sp / "metrics.json")
        w2 = load(O / M / "refseg/w2a" / sp / "metrics.json")
        print(f"| {sp} | {hl(e1, 'acc@0.5')} | {hl(ob, 'cIoU')} | {hl(w1, 'cIoU')} | {hl(w1, 'gIoU')} | {hl(w2, 'cIoU')} | {hl(w2, 'gIoU')} | {(w2 or {}).get('n')} | {(w2 or {}).get('complete')} |")
    print("\n### w2a 的挑选统计（分母 = 有候选的句子；都不对 = pick.none）\n")
    print("| 划分 | 挑中最优 | 挑 1 号 | 都不对 | 解析失败 | 出框失败 | thinking_end | parse_failure |")
    print("|---|---|---|---|---|---|---|---|")
    for sp in SPLITS:
        w2 = load(O / M / "refseg/w2a" / sp / "metrics.json") or {}
        f = {k: v for k, v in (w2.get("failures") or {}).items() if k != "ok"}
        s = w2.get("select") or {}
        print(f"| {sp} | {rate(w2, 'final_is_best')} | {rate(w2, 'final_is_1')} | {rate(w2, 'none')} | {rate(w2, 'parse_failure')} | {f} | {s.get('thinking_end')} | {s.get('parse_failure')} |")
    print()
