"""计数错误按物体个数分档（D125-2 (c)：确认 W2-c 的"数多"集中在哪些情况）。管理员的分析脚本，只读已有的 T 文本计数预测，不跑模型。

输入：CountBench 评测 442 条的 E1 格式 predictions.jsonl（score.gt、parsed.count、failure）。每个系统与它同节点的基座配对：
- W2-c、W2-c+回放（RUN-0076 补充 3）对 RUN-0069 的基座（0069_ttext）；
- E3b+C3、C3 对 RUN-0056 的基座（0056 的 qwen3.5-9b/count）。
按真值个数分档（2–10 各一档，CountBench 的范围），每档报：条数、准确率、多数率（答的 > 真值）、少数率、解析失败率、
带符号误差的均值；与配对基座的准确率差与配对 bootstrap 95% CI（按条重抽，10,000 次，种子 0），以及"基座对 → 系统错"
里多数与少数各占多少。另按 2–4、5–7、8–10 三个大档汇总。

用法：python scripts/mgr/count_errors.py [--out docs/analysis/count_errors_9b]  → <out>.json 与 <out>.md
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SYSTEMS = {
    # 名称: (预测文件, 配对的基座名)
    "base@0069": ("experiments/W2/0069_w2c_hpcgpu06/0069_ttext/qwen3.5-9b/count/predictions.jsonl", None),
    "W2-c": ("experiments/W2/0069_w2c_hpcgpu06/0069_ttext/t_text/w2c/seed0/qwen3.5-9b/count/predictions.jsonl", "base@0069"),
    "W2-c+回放": ("experiments/W2/0076_w2c_replay_ttext_hpcgpu06/t_text/w2c_replay/seed0/qwen3.5-9b/count/predictions.jsonl", "base@0069"),
    "base@0056": ("experiments/E3/0056_hpc_9b_eval_hpcgpu06/qwen3.5-9b/count/predictions.jsonl", None),
    "E3b+C3": ("experiments/E3/0056_hpc_9b_eval_hpcgpu06/t_text/e3b_c3/seed0/qwen3.5-9b/count/predictions.jsonl", "base@0056"),
    "C3": ("experiments/E3/0056_hpc_9b_eval_hpcgpu06/t_text/c3/seed0/qwen3.5-9b/count/predictions.jsonl", "base@0056"),
}
COARSE = {"2–4": range(2, 5), "5–7": range(5, 8), "8–10": range(8, 11)}
N_BOOT = 10_000


def load(path: str) -> dict[str, dict]:
    out = {}
    for line in (ROOT / path).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        gt = int(r["score"]["gt"])
        pred = (r.get("parsed") or {}).get("count") if r.get("failure") is None else None
        out[r["id"]] = {"gt": gt, "pred": None if pred is None else int(pred), "correct": bool(r["score"].get("correct"))}
    return out


def kind(x: dict) -> str:
    if x["pred"] is None:
        return "fail"
    return "ok" if x["pred"] == x["gt"] else ("over" if x["pred"] > x["gt"] else "under")


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    ks = [kind(x) for x in rows]
    err = [x["pred"] - x["gt"] for x in rows if x["pred"] is not None]
    return {"n": n, "acc": sum(k == "ok" for k in ks) / n if n else None,
            "over": sum(k == "over" for k in ks) / n if n else None,
            "under": sum(k == "under" for k in ks) / n if n else None,
            "fail": sum(k == "fail" for k in ks) / n if n else None,
            "mean_signed_err": sum(err) / len(err) if err else None}


def paired(sys_rows: list[dict], base_rows: list[dict], seed: int = 0) -> dict:
    n = len(sys_rows)
    d = [(kind(s) == "ok") - (kind(b) == "ok") for s, b in zip(sys_rows, base_rows)]
    rng = random.Random(seed)
    boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(N_BOOT))
    lost = [(s, b) for s, b in zip(sys_rows, base_rows) if kind(b) == "ok" and kind(s) != "ok"]
    gained = sum(1 for s, b in zip(sys_rows, base_rows) if kind(b) != "ok" and kind(s) == "ok")
    return {"d_acc": sum(d) / n, "ci95": [boots[int(0.025 * N_BOOT)], boots[int(0.975 * N_BOOT) - 1]],
            "base_ok_sys_wrong": len(lost), "of_which_over": sum(kind(s) == "over" for s, _ in lost),
            "of_which_under": sum(kind(s) == "under" for s, _ in lost), "of_which_fail": sum(kind(s) == "fail" for s, _ in lost),
            "base_wrong_sys_ok": gained}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="docs/analysis/count_errors_9b")
    a = ap.parse_args(argv)
    data = {k: load(p) for k, (p, _) in SYSTEMS.items()}
    ids = sorted(data["base@0069"], key=lambda s: int(s.rsplit("/", 1)[1]))
    for k, v in data.items():
        if set(v) != set(ids):
            raise SystemExit(f"{k}：条目与基座不同")
    bins = {str(g): [i for i in ids if data["base@0069"][i]["gt"] == g] for g in range(2, 11)}
    bins.update({k: [i for i in ids if data["base@0069"][i]["gt"] in r] for k, r in COARSE.items()})
    bins["all"] = ids
    base_agree = sum(data["base@0069"][i]["pred"] == data["base@0056"][i]["pred"] for i in ids) / len(ids)
    doc = {"source": {k: p for k, (p, _) in SYSTEMS.items()}, "n_boot": N_BOOT, "seed": 0,
           "base_0069_vs_0056_same_answer_rate": base_agree, "systems": {}}
    for k, (_, base) in SYSTEMS.items():
        per = {}
        for b, bid in bins.items():
            e = summarize([data[k][i] for i in bid])
            if base is not None and bid:
                e["vs_base"] = paired([data[k][i] for i in bid], [data[base][i] for i in bid])
            per[b] = e
        doc["systems"][k] = {"paired_base": base, "bins": per}
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    def f(x, p=3):
        return "—" if x is None else f"{x:.{p}f}"
    lines = ["# 9B 计数错误按真值个数分档（CountBench 442 条；`scripts/mgr/count_errors.py` 生成）", "",
             f"基座两次读数（RUN-0069 与 RUN-0056）逐条答案相同的比例：{base_agree:.3f}。", ""]
    for k, (_, base) in SYSTEMS.items():
        lines += [f"## {k}" + (f"（配对 {base}）" if base else ""), "",
                  "| 真值 | n | 准确率 | 多数 | 少数 | 失败 | 带符号误差 | Δ准确率 [95% CI] | 基座对→错（多/少/失败） | 基座错→对 |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for b, e in doc["systems"][k]["bins"].items():
            v = e.get("vs_base")
            dv = f"{v['d_acc']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]" if v else ""
            lost = f"{v['base_ok_sys_wrong']}（{v['of_which_over']}/{v['of_which_under']}/{v['of_which_fail']}）" if v else ""
            lines.append(f"| {b} | {e['n']} | {f(e['acc'])} | {f(e['over'])} | {f(e['under'])} | {f(e['fail'])} | "
                         f"{f(e['mean_signed_err'], 2)} | {dv} | {lost} | {v['base_wrong_sys_ok'] if v else ''} |")
        lines.append("")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out.with_suffix(".md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
