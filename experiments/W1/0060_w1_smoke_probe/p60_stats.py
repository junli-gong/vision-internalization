"""RUN-0060 report helper: per-step loop statistics and GPU memory peaks from the W1 smoke outputs.
usage: python p60_stats.py <OUT dir = experiments/W1/0060_w1_smoke_hpcgpu07> [step ...]"""
import collections as C, csv, json, re, statistics as st, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

O = Path(sys.argv[1])
want = set(sys.argv[2:])
steps, prev = {}, None
for line in (O / "steps.log").read_text().splitlines():
    m = re.match(r"(\S+) start commit", line)
    if m:
        prev = datetime.fromisoformat(m[1].replace("Z", "+00:00")); continue
    m = re.match(r"(\S+) (\S+) rc=(\d+) wall_s=(\d+)", line)
    if m:
        end = datetime.fromisoformat(m[1].replace("Z", "+00:00"))
        steps[m[2]] = {"start": prev, "end": end, "rc": int(m[3]), "wall_s": int(m[4])}; prev = end

CST = timezone(timedelta(hours=8))
mem = []
for row in csv.reader((O / "gpu_mem.csv").read_text().splitlines()):
    try:
        t = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=CST)
        mem.append((t, int(row[1].strip().split()[0])))
    except (ValueError, IndexError):
        continue
print(f"gpu_mem.csv: {len(mem)} samples; overall max {max(v for _, v in mem)} MiB" if mem else "gpu_mem.csv: no samples")


def dist(xs):
    return ", ".join(f"{k}: {v}" for k, v in sorted(C.Counter(xs).items(), key=lambda kv: str(kv[0])))


for pred in sorted(O.glob("qwen3.5-9b/*/*/*/predictions.jsonl")):
    task, setting, split = pred.parts[-4], pred.parts[-3], pred.parts[-2]
    rows = [json.loads(x) for x in pred.read_text().splitlines() if x.strip()]
    print(f"\n### {task}/{setting}/{split}: n={len(rows)}")
    print(f"- failure: {dist([r.get('failure') for r in rows])}")
    print(f"- ended_by: {dist([r.get('ended_by') for r in rows])}")
    if any("n_rounds" in r for r in rows):
        print(f"- n_rounds: {dist([r.get('n_rounds') for r in rows])}")
        print(f"- n_tool_calls: {dist([r.get('n_tool_calls') for r in rows])}")
        print(f"- n_refused total {sum(r.get('n_refused') or 0 for r in rows)}; n_tool_call_errors total {sum(r.get('n_tool_call_errors') or 0 for r in rows)}")
        gen = [(r.get("tokens") or {}).get("generated") for r in rows if (r.get("tokens") or {}).get("generated") is not None]
        if gen:
            print(f"- generated tokens per sentence: median {st.median(gen):.0f}, max {max(gen)}")
        te = [e for r in rows for e in (r.get("thinking_end") or [])]
        if te:
            print(f"- thinking_end over rounds: {dist(te)}")
        tools = C.Counter()
        for r in rows:
            for rd in r.get("rounds") or []:
                for tc in re.findall(r"<function=(\w+)>", rd.get("text") or ""):
                    tools[tc] += 1
        if tools:
            print(f"- tool names called (from round text): {dict(tools.most_common())}")
        br = [rd.get("batch_rows") for r in rows for rd in (r.get("rounds") or []) if rd.get("batch_rows") is not None]
        if br:
            print(f"- batch_rows over (sentence, round): {dist(br)}")
    tj = pred.parent / "logs/timing.jsonl"
    if tj.exists():
        tm = [json.loads(x) for x in tj.read_text().splitlines() if x.strip()]
        print(f"- per-sentence wall_time_s: median {st.median(t['wall_time_s'] for t in tm):.1f}, max {max(t['wall_time_s'] for t in tm):.1f}; "
              f"tool_time_s median {st.median(t.get('tool_time_s', 0) for t in tm):.2f}; chunks {sorted({t.get('chunk_index') for t in tm})}")
    cj = pred.parent / "compute.json"
    if cj.exists():
        c = json.loads(cj.read_text())
        print(f"- compute.json: n_calls {c.get('n_calls')}, generated_tokens {c.get('generated_tokens')}, prompt_tokens {c.get('prompt_tokens')}, "
              f"call_time_s {c.get('call_time_s'):.1f}, wall_time_s {c.get('wall_time_s'):.1f}")

print()
for name, s in steps.items():
    if want and name not in want:
        continue
    w = [v for t, v in mem if s["start"] <= t <= s["end"]]
    print(f"- {name}: rc={s['rc']} wall_s={s['wall_s']}; gpu_mem peak in window {max(w) if w else 'n/a'} MiB ({len(w)} samples)")
