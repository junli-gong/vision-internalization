"""RUN-0044 §7–§8 report helper (from RUN-0034's p35_tput_table.py): per training run on the HPC A100s, the driver's rc and command wall
(logs/0044_<node>/driver_<phase>.log), accounting.json (wall, steps, peak memory) and per-step medians from logs/timing_run_*.jsonl
(step_s, build_inputs_s, fwd_s, bwd_s). Then the per-run estimate for the formal training: median step_s x configured steps (C3: e3b_train.json
steps; E3b+C3: mixed.steps), 3 seeds, with and without a +10 % allowance for loading, checkpoints and the per-segment restarts.
usage: python p44_tput_table.py <experiments/E3 dir> <logs dir ($VI/logs)> <configs/e3b_train.json> <out prefix>"""
import json, re, statistics, sys
from pathlib import Path

NODES = {"hpcgpu06": "qwen3.5-4b", "hpcgpu07": "qwen3.5-0.8b"}


def driver_rc(logs, node):
    rc = {}
    for ph in ("consistency", "throughput"):
        p = Path(logs) / f"0044_{node}" / f"driver_{ph}.log"
        if p.exists():
            for line in p.read_text().splitlines():
                g = re.search(r"=== (\S+) rc=(\d+) (\d+)s", line)
                if g:
                    rc[g[1]] = (int(g[2]), int(g[3]))
    return rc


def leaf(d):
    a = json.loads((d / "accounting.json").read_text())
    t = [json.loads(x) for f in sorted((d / "logs").glob("timing_run_*.jsonl")) for x in f.read_text().splitlines() if x.strip()]
    med = lambda k: statistics.median(r[k] for r in t if k in r)
    return a, {k: med(k) for k in ("step_s", "build_inputs_s", "fwd_s", "bwd_s")}


def main(e3, logs, cfg, prefix):
    e3, c = Path(e3), json.loads(Path(cfg).read_text())
    steps = {"c3": c["steps"], "e3b_c3": c["mixed"]["steps"]}
    rows, md = [], ["# RUN-0044 吞吐与一致性：每条训练命令（HPC A100-PCIE-40GB）", "",
                    "| 节点 | 运行 | 模型 | rc | 命令墙钟 s（driver，含加载） | accounting wall_s | 步数（计时 / 记录） | 每步中位 s | 输入构建中位 s | 前向中位 s | 反向中位 s | 每步平均 s | 显存峰值 GB | 目录 |",
                    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for node, m in NODES.items():
        rc = driver_rc(logs, node)
        runs = [(f"consistency_e3b_qwen3.5-2b", e3 / f"0044_hpc_consistency_{node}/e3b/qwen3.5-2b/seed0"),
                (f"consistency_e3a_qwen3.5-2b", e3 / f"0044_hpc_consistency_{node}/e3a/full/qwen3.5-2b/seed0"),
                (f"tput_c3_{m}", e3 / f"0044_hpc_throughput_{node}/c3/{m}/seed0"),
                (f"tput_e3b_c3_{m}", e3 / f"0044_hpc_throughput_{node}/e3b_c3/{m}/seed0"),
                (f"tput_shm_e3b_c3_{m}", e3 / f"0044_hpc_throughput_shm_{node}/e3b_c3/{m}/seed0")]
        for name, d in runs:
            if not (d / "accounting.json").exists():
                md.append(f"| {node} | {name} | {m} | {rc.get(name, (None,))[0]} | 缺 accounting.json：`{d.relative_to(e3)}` |||||||||")
                continue
            a, med = leaf(d)
            r, w = rc.get(name, (None, None))
            row = {"node": node, "run": name, "model": m if not name.startswith("consistency") else "qwen3.5-2b", "rc": r, "cmd_wall_s": w,
                   "accounting_wall_s": a["wall_s"], "n_steps_timed": a["n_steps_timed"], "n_steps_logged": a["n_steps_logged"],
                   **{f"{k}_median": v for k, v in med.items()}, "step_s_mean": a["phases_s"]["step_s"] / a["n_steps_timed"],
                   "peak_mem_bytes": a["peak_mem_bytes"], "gpu": a.get("gpu"), "dir": str(d.relative_to(e3))}
            rows.append(row)
            md.append(f"| {node} | {name} | {row['model']} | {r} | {w} | {a['wall_s']:.1f} | {a['n_steps_timed']} / {a['n_steps_logged']} | "
                      f"{med['step_s']:.3f} | {med['build_inputs_s']:.3f} | {med['fwd_s']:.3f} | {med['bwd_s']:.3f} | {row['step_s_mean']:.3f} | "
                      f"{a['peak_mem_bytes'] / 1e9:.2f} | `{row['dir']}` |")
    md += ["", "## 正式训练的逐项预估（每步中位 × 配置步数；3 个种子；另列 +10 % 的加载、检查点与分段重启余量）", "",
           f"步数：C3 = e3b_train.json `steps` = {steps['c3']}；E3b+C3 = `mixed.steps` = {steps['e3b_c3']}。", "",
           "| 节点 | 模型 | 变体 | 每步中位 s（Lustre） | 单个种子 h | 3 个种子 h | 3 个种子 h（+10 %） |", "|---|---|---|---|---|---|---|"]
    est, tot = [], {}
    for r in rows:
        g = re.match(r"tput_(c3|e3b_c3)_(.+)", r["run"])
        if not g:
            continue
        v = g[1]; one = r["step_s_median"] * steps[v] / 3600
        est.append({"node": r["node"], "model": r["model"], "variant": v, "step_s_median": r["step_s_median"], "h_one": one, "h_three": 3 * one, "h_three_plus10": 3.3 * one})
        tot[r["node"]] = tot.get(r["node"], 0) + 3 * one
        md.append(f"| {r['node']} | {r['model']} | {v} | {r['step_s_median']:.3f} | {one:.2f} | {3 * one:.2f} | {3.3 * one:.2f} |")
    for n, h in tot.items():
        md.append(f"| {n} | 合计（C3 × 3 + E3b+C3 × 3） | | | | {h:.2f} | {h * 1.1:.2f} |")
    Path(prefix + ".json").write_text(json.dumps({"runs": rows, "estimates": est, "steps": steps}, indent=1, ensure_ascii=False) + "\n")
    Path(prefix + ".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(*sys.argv[1:5])
