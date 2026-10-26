"""RUN-0072 报告用的表：9B × 5 个 test 划分的各设置标题读数、w2a / w2c 的挑选统计、出框读数（只读结果文件）。
用法：python p72_tables.py experiments/W2/0072_w2_test_hpcgpu06"""
import json, sys
from pathlib import Path

O = Path(sys.argv[1]) / "qwen3.5-9b"
SPLITS = ["test_refcoco_unc_testA", "test_refcoco_unc_testB", "test_refcoco+_unc_testA", "test_refcoco+_unc_testB", "test_refcocog_umd_test"]
SETTINGS = ["w1f", "w2a", "w2c", "ownbox", "random_pick", "oracle_pick", "baseline_e1"]


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


for sp in SPLITS:
    print(f"### {sp}\n")
    print("| 设置 | cIoU [95% CI] | gIoU [95% CI] | n | complete |")
    print("|---|---|---|---|---|")
    for s in SETTINGS:
        m = load(O / "refseg" / s / sp / "metrics.json")
        print(f"| {s} | {hl(m, 'cIoU')} | {hl(m, 'gIoU')} | {(m or {}).get('n')} | {(m or {}).get('complete')} |")
    for b in ["baseline_e1", "propose_w2c"]:
        m = load(O / "refexp_box" / b / sp / "metrics.json")
        print(f"| 出框 {b} acc@0.5 | {hl(m, 'acc@0.5')} | | {(m or {}).get('n')} | {(m or {}).get('complete')} |")
    m = load(O / "refseg/baseline_e1" / sp / "metrics.json") or {}
    print(f"\n- baseline_e1（多边形）parse：{json.dumps(m.get('parse'), ensure_ascii=False)}")
    print()

print("### w2a 与 w2c 的挑选统计（分母 = 有候选的句子；都不对 = pick.none）\n")
print("| 划分 | 设置 | 挑中最优 | 挑 1 号 | 都不对 | 解析失败 | 出框失败 | thinking_end | parse_failure |")
print("|---|---|---|---|---|---|---|---|---|")
for sp in SPLITS:
    for s in ["w2a", "w2c"]:
        m = load(O / "refseg" / s / sp / "metrics.json") or {}
        f = {k: v for k, v in (m.get("failures") or {}).items() if k != "ok"}
        sel = m.get("select") or {}
        print(f"| {sp} | {s} | {rate(m, 'final_is_best')} | {rate(m, 'final_is_1')} | {rate(m, 'none')} | {rate(m, 'parse_failure')} | {f} | {sel.get('thinking_end')} | {sel.get('parse_failure')} |")
