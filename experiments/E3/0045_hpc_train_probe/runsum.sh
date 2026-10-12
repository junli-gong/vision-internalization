VI=/home/mail/chenyuma/visint
bash $VI/probe/0044/patrol_0045.sh
echo "quota delta KB: $(( $(timeout 60 lfs quota -u chenyuma /home | awk 'NR==3{print $2}') - 7059403936 ))"
for h in hpcgpu07 hpcgpu06; do O=$VI/repo/experiments/E3/0045_hpc_train_$h; echo "### $h"; cat $VI/logs/train_0045-$h.driver.log | grep -E "完成|失败|FAILED|没有完成|跑完|die|blocked|cap"
  cat $O/ckpt_ledger.tsv; cat $O/driver/gpu_seconds 2>/dev/null
  $VI/env/bin/python - "$O" <<'PY'
import json, sys, statistics, re
from pathlib import Path
o = Path(sys.argv[1])
for acc in sorted(o.glob("*/*/seed*/accounting.json")):
    d = acc.parent; a = json.loads(acc.read_text())
    l = [json.loads(x) for x in (d / "train_log.jsonl").read_text().splitlines() if x.strip()]
    t = [json.loads(x) for f in sorted((d / "logs").glob("timing_run_*.jsonl")) for x in f.read_text().splitlines() if x.strip()]
    run = {"c3": "c3", "e3b_c3": "e3b_c3"}.get(d.parts[-3], d.parts[-3]) + "_s" + d.parts[-1][-1]
    log = (o / "driver" / "logs" / f"{run}.log")
    walls = [int(x) for x in re.findall(r"chunk rc=0 wall_s=(\d+)", log.read_text())] if log.exists() else []
    print(f"{d.relative_to(o)} steps={len(l)} last_step={l[-1]['step']} last_loss={l[-1]['loss']:.4f} step_med={statistics.median(r['step_s'] for r in t):.3f} "
          f"peak_GB={a['peak_mem_bytes']/1e9:.2f} gpu_h={a['gpu_hours']:.3f} chunks={len(walls)} wall_sum_s={sum(walls)} ({sum(walls)/3600:.2f} h)")
PY
done
