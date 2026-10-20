"""RUN-0055 report helper: timing and memory tables for the 9B measurement on hpcgpu06 (no metric readings).
usage: python p55_tables.py <experiments/E3 root>"""
import json, re, statistics as st, sys
from datetime import datetime
from pathlib import Path

R = Path(sys.argv[1])
D = R / "0055_hpc_measure_9b_hpcgpu06"
E1 = R / "0055_hpc_measure_9b_e1_hpcgpu06"
J = lambda p: json.loads(Path(p).read_text())
jl = lambda p: [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
ts = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))
GB = lambda b: f"{b / 1e9:.2f}"

steps, prev = {}, None
for line in (D / "steps.log").read_text().splitlines():
    m = re.match(r"(\S+) start commit", line)
    if m:
        prev = ts(m[1]); continue
    m = re.match(r"(\S+) (\S+) rc=(\d+) wall_s=(\d+)", line)
    if m:
        end = ts(m[1]); steps[m[2]] = {"start": prev, "end": end, "rc": int(m[3]), "wall_s": int(m[4])}; prev = end

print("## 训练 20 步（种子 0，微批 4 × 累积 4）\n")
print("| 臂 | 步 rc | 命令墙钟 s | 命令开始到训练开始 s | 第 1 步 step_s | 20 步每步中位 s | 第 2–20 步每步中位 s | 输入构建中位 s | 前向中位 s | 反向中位 s | peak_mem_bytes 最大（allocated）GB | reserved |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for arm in ("e3b_c3", "e3b", "c3"):
    L = D / arm / "qwen3.5-9b/seed0"
    run = J(L / "logs/run_001.json"); tm = jl(L / "logs/timing_run_001.jsonl"); s = steps[f"train20_{arm}_micro4"]
    t0 = (ts(run["started_utc"]) - s["start"]).total_seconds()
    med = lambda k, rows=tm: st.median(r[k] for r in rows)
    print(f"| {arm} | {s['rc']} | {s['wall_s']} | {t0:.0f} | {tm[0]['step_s']:.2f} | {med('step_s'):.3f} | {med('step_s', tm[1:]):.3f} | "
          f"{med('build_inputs_s'):.3f} | {med('fwd_s'):.3f} | {med('bwd_s'):.3f} | {GB(max(r['peak_mem_bytes'] for r in tm))} | 日志里没有 |")
print("\n- 三臂都在微批 4 下跑完，没有走微批 2 的补测（`steps.log` 里没有 micro2 的步）。")

print("\n## 网格评测（E3b+C3 的 20 步检查点，结构强制）\n")
mc = J(D / "eval/e3b_c3/qwen3.5-9b/seed0/merge_check.json")
print(f"- 合并核对（两条共用，`eval/e3b_c3/qwen3.5-9b/seed0/merge_check.json`，写于 {mc.get('written_utc')}）：pass = {mc['pass']}，failures = "
      f"{json.dumps(mc['failures'], ensure_ascii=False)}，criteria_version = {mc.get('criteria_version')}；耗时 "
      + "、".join(f"{k} {v:.1f} s" for k, v in mc["timing_s"].items()) + "\n")
print("| 步 | rc | 命令墙钟 s | 条数 | 解码 call_time_s | 解码 wall_time_s | 每条 wall_time_s | 批大小 | 每批耗时 | 显存峰值 |")
print("|---|---|---|---|---|---|---|---|---|---|")
for step, task in (("grid_e3b_c3_refseg100", "refseg"), ("grid_e3b_c3_depth50", "depth")):
    G = D / f"eval/e3b_c3/qwen3.5-9b/seed0/{task}__structured"; c = J(G / "compute.json"); s = steps[step]
    bs = re.search(r"^\s*batch_size:\s*(\d+)", (G / "config.yaml").read_text(), re.M)
    print(f"| {step} | {s['rc']} | {s['wall_s']} | {c['n_samples']} | {c['call_time_s']:.1f} | {c['wall_time_s']:.1f} | "
          f"{c['per_sample_mean']['wall_time_s']:.3f} | {bs[1] if bs else '日志里没有'} | 日志里没有 | 日志里没有 |")

print("\n## T 文本\n")
for step, base, suffix in (("e1_base_dev20", E1 / "qwen3.5-9b", "__dev"), ("e1_base_free10", E1 / "qwen3.5-9b", "__free"),
                           ("ttext_e3b_c3_dev20", E1 / "t_text/e3b_c3/seed0/qwen3.5-9b", "__dev")):
    s = steps[step]; rows, gen = [], 0.0
    for d in sorted(base.glob(f"*{suffix}")):
        c = J(d / "compute.json"); run = J(d / "logs/run_001.json"); gen += float(run["wall_s"])
        rows.append(f"| {d.name} | {c['n_samples']} | {run.get('batch_size')} | {run.get('n_batches')} | {float(run['wall_s']):.1f} | "
                    f"{c['per_sample_mean']['wall_time_s']:.3f} | {c['generated_tokens']} |")
    extra = ""
    mcp = base / "merge_check.json"
    if mcp.exists():
        m = J(mcp); extra = (f"；合并核对 pass = {m['pass']}，failures = {json.dumps(m['failures'], ensure_ascii=False)}，耗时 "
                             + "、".join(f"{k} {v:.1f} s" for k, v in m["timing_s"].items()))
    print(f"### {step}：rc={s['rc']}，命令墙钟 {s['wall_s']} s；各任务评测运行 wall_s 合计 {gen:.1f} s；加载与核对等 {s['wall_s'] - gen:.0f} s（命令墙钟 − 各任务合计）{extra}；显存峰值：日志里没有\n")
    print("| 任务 | 条数 | 批大小 | 批数 | 评测运行 wall_s | 每条 s（compute.json） | 生成 token |"); print("|---|---|---|---|---|---|---|")
    print("\n".join(rows) + "\n")
