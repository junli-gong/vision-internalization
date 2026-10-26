"""RUN-0077 报告用：各划分 predictions.jsonl 里局部 GrabCut 退回 direct 的计数（info.cor.fallback_direct）与 failure 计数。
用法：python p77_counts.py experiments/X/0077_ceiling_hpcgpu07"""
import collections, json, sys
from pathlib import Path

O = Path(sys.argv[1])
for split in ["devtune800", "devcheck2"]:
    by_kind = collections.defaultdict(lambda: collections.Counter())
    for line in (O / split / "predictions.jsonl").read_text().splitlines():
        r = json.loads(line)
        c = by_kind[r.get("box_kind")]
        c["jobs"] += 1
        c["failure"] += bool(r.get("failure"))
        fd = ((r.get("info") or {}).get("cor") or {}).get("fallback_direct")
        if fd is None:
            c["no_cor_info"] += 1
        else:
            c["fallback_direct_cells"] += fd
            c["jobs_with_fallback"] += fd > 0
    tot = collections.Counter()
    for k, c in sorted(by_kind.items()):
        tot.update(c)
        print(f"{split} box_kind={k}: jobs {c['jobs']}, failure {c['failure']}, 没有 info.cor {c['no_cor_info']}, "
              f"有退回的 job {c['jobs_with_fallback']}, 退回 direct 的格点合计 {c['fallback_direct_cells']}")
    print(f"{split} 合计: jobs {tot['jobs']}, failure {tot['failure']}, 没有 info.cor {tot['no_cor_info']}, "
          f"有退回的 job {tot['jobs_with_fallback']}, 退回 direct 的格点合计 {tot['fallback_direct_cells']}")
