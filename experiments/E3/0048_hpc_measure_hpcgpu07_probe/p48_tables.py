"""RUN-0048 report helper: read the pushed results of measure.sh and measure_e1.sh and print the report tables (no hand-typed numbers).
usage: python p48_tables.py <repo root>"""
import json, re, statistics, sys
from datetime import datetime
from pathlib import Path

R = Path(sys.argv[1])
E3 = R / "experiments/E3/0048_hpc_measure_hpcgpu07"
E5 = R / "experiments/E5/0048_hpc_measure_hpcgpu07"
E1 = R / "experiments/E3/0048_hpc_measure_e1_hpcgpu07"
ts = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))
J = lambda p: json.loads(Path(p).read_text())
jl = lambda p: [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def steps(path):  # name -> (end time, rc, wall_s); start = previous end (first one = 'start' line)
    out, prev = {}, None
    for line in Path(path).read_text().splitlines():
        m = re.match(r"(\S+) start commit", line)
        if m:
            prev = ts(m[1]); continue
        m = re.match(r"(\S+) (\S+) rc=(\d+) wall_s=(\d+)", line)
        if m:
            end = ts(m[1]); out[m[2]] = {"end": end, "rc": int(m[3]), "wall_s": int(m[4]), "start": prev}; prev = end
    return out


S = steps(E3 / "steps.log")
print("## E5 训练 20 步（2B，各起点）\n")
print("| 起点 | 命令 rc | 命令墙钟 s | 合并核对 pass | 合并核对失败项 | 合并核对耗时 s（merge 前 / 后 / restricted / across_merge / 合计） | 命令开始到训练开始 s | 每步中位 s | 每步平均 s | 显存峰值 GB |")
print("|---|---|---|---|---|---|---|---|---|---|")
for start, step in [("e3b_c3", "e5_train20_e3b_c3"), ("c3_warmup", "e5_train20_c3_warmup"), ("base_warmup", "e5_train20_base_warmup"), ("e3b（种子 1）", "e5_train20_e3b_s1")]:
    key = start.split("（")[0]
    mc = sorted((E5 / f"train/{key}/qwen3.5-2b/start_checks").glob("merge_check_*.json"))
    m = J(mc[0]) if mc else None
    t = m["timing_s"] if m else {}
    tim = " / ".join(f"{t.get(k, 0):.1f}" for k in ("merge_before_s", "merge_after_s", "restricted_s", "restricted_across_merge_s", "total_s")) if m else "—"
    fails = "; ".join(x for v in (m or {}).get("failures", {}).values() for x in v) if m else "—"
    leaf = E5 / f"train/{key}/qwen3.5-2b/n1000/seed0"
    st = S.get(step, {})
    if (leaf / "accounting.json").exists() and key != "e3b":
        a = J(leaf / "accounting.json"); run = J(leaf / "logs/run_001.json"); tm = jl(leaf / "logs/timing_run_001.jsonl")
        t2 = (ts(run["started_utc"]) - st["start"]).total_seconds() if st.get("start") else None
        print(f"| {start} | {st.get('rc')} | {st.get('wall_s')} | {m['pass'] if m else '—'} | {fails or '无'} | {tim} | {t2:.0f} | "
              f"{statistics.median(r['step_s'] for r in tm):.3f} | {a['phases_s']['step_s'] / a['n_steps_timed']:.3f} | {a['peak_mem_bytes'] / 1e9:.2f} |")
    else:
        print(f"| {start} | {st.get('rc')} | {st.get('wall_s')} | {m['pass'] if m else '—'} | {fails or '无'} | {tim} | 没有开训 | — | — | — |")
for key in ("e3b_c3", "c3_warmup", "base_warmup", "e3b"):
    for f in sorted((E5 / f"train/{key}/qwen3.5-2b/start_checks").glob("merge_check_*.json")):
        m = J(f)
        print(f"\n### 合并核对原文摘要：{key}（`{f.relative_to(R)}`）\n")
        print(f"- checkpoint：`{m['checkpoint']['dir']}`，manifest_sha256 `{m['checkpoint']['manifest_sha256'][:16]}…`；pass = {m['pass']}；failures = {json.dumps(m['failures'], ensure_ascii=False)}")
        for ck, v in m["checks"].items():
            flat = {}
            def walk(x, p):
                if isinstance(x, dict):
                    for k2, v2 in x.items(): walk(v2, f"{p}.{k2}" if p else k2)
                elif isinstance(x, (int, float, bool)) and not isinstance(x, bool) or isinstance(x, bool):
                    flat[p] = x
            walk(v, "")
            keep = {k: v for k, v in flat.items() if re.search(r"(pass|agreement|exact_rows|n_disagree|median|max)$", k) and "criteria" not in k}
            print(f"- {ck}：" + "，".join(f"`{k}`={(round(v, 4) if isinstance(v, float) else v)}" for k, v in list(keep.items())[:24]))

print("\n## E5 val 一次（e3b_c3 起点，只评起点，结构强制）\n")
st = S.get("e5_val64_e3b_c3", {})
print(f"- 命令 rc={st.get('rc')}，命令墙钟 {st.get('wall_s')} s。起点合并核对：复用 e3b_c3 训练时写的 `merge_check_bf73188b….json`，没有重算（日志第 1 行）。\n")
print("| 类 | 条数 | 生成墙钟 s（compute.json wall_time_s，不含加载） | 每条 s | 批数 | 每批墙钟 s |")
print("|---|---|---|---|---|---|")
for d in sorted((E5 / "trained/e3b_c3/qwen3.5-2b/n0/seed0").glob("*__dev__structured")):
    c = J(d / "compute.json"); tm = jl(d / "logs/timing.jsonl")
    b = {}
    for r in tm: b.setdefault(r.get("batch_index", r.get("chunk")), r["batch_wall_time_s"])
    print(f"| {d.name.split('__')[0]} | {c['n_samples']} | {c['wall_time_s']:.1f} | {c['per_sample_mean']['wall_time_s']:.3f} | {len(b)} | {', '.join(f'{v:.1f}' for v in b.values())} |")

print("\n## 0.8B 的 E3b 训练 20 步\n")
leaf = E3 / "e3b/qwen3.5-0.8b/seed0"; a = J(leaf / "accounting.json"); tm = jl(leaf / "logs/timing_run_001.jsonl"); st = S.get("e3b_train20_qwen3.5-0.8b", {})
print(f"- 命令 rc={st.get('rc')}，墙钟 {st.get('wall_s')} s；accounting wall_s {a['wall_s']:.1f}；每步中位 {statistics.median(r['step_s'] for r in tm):.3f} s，"
      f"输入构建中位 {statistics.median(r['build_inputs_s'] for r in tm):.3f} s，前向中位 {statistics.median(r['fwd_s'] for r in tm):.3f} s，反向中位 {statistics.median(r['bwd_s'] for r in tm):.3f} s；显存峰值 {a['peak_mem_bytes'] / 1e9:.2f} GB。")

print("\n## 0.8B 网格评测（E3b+C3 种子 0，结构强制，单轮）\n")
mc = J(E3 / "eval/e3b_c3/qwen3.5-0.8b/seed0/merge_check.json")
print(f"- 合并核对（两条命令共用一份，`eval/e3b_c3/qwen3.5-0.8b/seed0/merge_check.json`）：pass = {mc['pass']}，failures = {json.dumps(mc['failures'], ensure_ascii=False)}；耗时 "
      + "、".join(f"{k} {v:.1f} s" for k, v in mc["timing_s"].items()) + "\n")
print("| 命令 | rc | 命令墙钟 s | 评测运行 wall_s（run_001） | 加载与合并核对 s（命令墙钟 − 评测运行） | 条数 | 生成墙钟 s | 每条 s | 评分 s |")
print("|---|---|---|---|---|---|---|---|---|")
for step, task in [("grid_e3b_c3_s0_refseg100", "refseg"), ("grid_e3b_c3_s0_depth50", "depth")]:
    d = E3 / f"eval/e3b_c3/qwen3.5-0.8b/seed0/{task}__structured"; c = J(d / "compute.json"); run = J(d / "logs/run_001.json"); st = S[step]
    print(f"| {step} | {st['rc']} | {st['wall_s']} | {run['wall_s']:.1f} | {st['wall_s'] - run['wall_s']:.0f} | {c['n_samples']} | {c['wall_time_s']:.1f} | "
          f"{c['per_sample_mean']['wall_time_s']:.3f} | {run.get('score_wall_s', '日志里没有')} |")

S1 = steps(E1 / "steps.log")
print("\n## 补充 1：T 文本与基座 E1（measure_e1.sh）\n")
for step, base in [("ttext_e3b_c3_s0_dev20", E1 / "t_text/e3b_c3/seed0/qwen3.5-0.8b"), ("ttext_e3b_c3_s0_free10", E1 / "t_text/e3b_c3/seed0/qwen3.5-0.8b"), ("e1_base_dev20", E1 / "qwen3.5-0.8b")]:
    st = S1[step]; suffix = "__free" if "free" in step else "__dev"
    rows, gen = [], 0.0
    for d in sorted(base.glob(f"*{suffix}")):
        c = J(d / "compute.json"); run = J(d / "logs/run_001.json"); gen += float(run["wall_s"])
        rows.append(f"| {d.name} | {c['n_samples']} | {float(run['wall_s']):.1f} | {c['per_sample_mean']['wall_time_s']:.3f} | {c['generated_tokens']} |")
    extra = ""
    mcp = base / "merge_check.json"
    if mcp.exists() and "ttext" in step:
        m = J(mcp); extra = f"；合并核对 pass = {m['pass']}，耗时 {m['timing_s']['total_s']:.1f} s（`{mcp.relative_to(R)}`，dev 与 free 共用）"
    print(f"### {step}：rc={st['rc']}，命令墙钟 {st['wall_s']} s；各任务评测运行 wall_s 合计 {gen:.1f} s；加载与核对等 {st['wall_s'] - gen:.0f} s（命令墙钟 − 各任务合计）{extra}\n")
    print("| 任务 | 条数 | 评测运行 wall_s | 每条 s（compute.json） | 生成 token |"); print("|---|---|---|---|---|")
    print("\n".join(rows) + "\n")
