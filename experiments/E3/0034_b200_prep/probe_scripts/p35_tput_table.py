"""RUN-0034 §5 report helper: per training run, the driver's GPU wall and rc (logs/driver.log), accounting.json (wall_s, steps, peak memory,
phase sums) and the per-step median from logs/timing_run_001.jsonl (accounting.json has sums only, no median).
usage: python p35_tput_table.py <run dir> <driver.log> <out prefix>"""
import json, re, statistics, sys
from pathlib import Path
RUNS = [("consistency_e3b", "0034_e3_consistency/e3b"), ("consistency_e3a", "0034_e3_consistency/e3a/full"),
        ("tput_e3a_full", "0034_e3_throughput/e3a/full"), ("tput_e3a_c1", "0034_e3_throughput/e3a/c1"), ("tput_e3a_c2", "0034_e3_throughput/e3a/c2"),
        ("tput_e3b", "0034_e3_throughput/e3b"), ("tput_c1prime", "0034_e3_throughput/c1prime"), ("tput_c3", "0034_e3_throughput/c3")]


def main(run, driver_log, prefix):
    run = Path(run)
    rc = {}
    for line in Path(driver_log).read_text().splitlines():
        g = re.match(r"=== ((consistency|tput)_\S+) rc=(\d+) \S+ \((\d+)s; gpu=1\)", line)
        if g:
            rc[g[1]] = (int(g[3]), int(g[4]))
    out, md = [], ["# RUN-0034 吞吐与一致性：每条训练命令", "",
                   "| 运行 | 模型 | rc | 命令墙钟 s（driver，含加载） | accounting wall_s | 步数（计时 / 记录） | 每步中位 s（timing） | 每步平均 s（accounting） | 前向中位 s | 反向中位 s | 显存峰值 GB（accounting） | gpu_hours（accounting） | 目录 |",
                   "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, sub in RUNS:
        for m in ("qwen3.5-2b", "qwen3.5-9b"):
            d = run / sub / m / "seed0"
            if not (d / "accounting.json").exists():
                continue
            a = json.loads((d / "accounting.json").read_text())
            t = [json.loads(x) for x in (d / "logs" / "timing_run_001.jsonl").read_text().splitlines() if x.strip()]
            med = lambda k: statistics.median(r[k] for r in t)
            r, w = rc.get(f"{name}_{m}", (None, None))
            row = {"run": name, "model": m, "rc": r, "cmd_wall_s": w, "accounting_wall_s": a["wall_s"], "n_steps_timed": a["n_steps_timed"],
                   "n_steps_logged": a["n_steps_logged"], "step_s_median": med("step_s"), "step_s_mean": a["phases_s"]["step_s"] / a["n_steps_timed"],
                   "fwd_s_median": med("fwd_s"), "bwd_s_median": med("bwd_s"), "peak_mem_bytes": a["peak_mem_bytes"], "gpu_hours": a["gpu_hours"],
                   "dir": str(d.relative_to(run))}
            out.append(row)
            md.append(f"| {name} | {m} | {r} | {w} | {a['wall_s']:.1f} | {a['n_steps_timed']} / {a['n_steps_logged']} | {row['step_s_median']:.3f} | "
                      f"{row['step_s_mean']:.3f} | {row['fwd_s_median']:.3f} | {row['bwd_s_median']:.3f} | {a['peak_mem_bytes'] / 1e9:.2f} | {a['gpu_hours']:.4f} | `{row['dir']}` |")
    tot = sum(r["cmd_wall_s"] or 0 for r in out)
    md += ["", f"命令墙钟合计：{tot} s（{tot / 3600:.3f} h）。"]
    Path(prefix + ".json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    Path(prefix + ".md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:4])
