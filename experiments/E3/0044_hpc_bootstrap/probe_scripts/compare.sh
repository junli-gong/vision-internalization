#!/bin/bash
# RUN-0044 §7: compare_train_logs (ea49e15 version) — each HPC node vs B200 RUN-0034, and hpcgpu06 vs hpcgpu07
VI=/home/mail/chenyuma/visint; WT=$VI/wt/wt0044; PY=$VI/env/bin/python
B=$WT/experiments/E3/0034_b200_prep/0034_e3_consistency; H=$VI/repo/experiments/E3; O=$H/0044_hpc_consistency_compare; mkdir -p $O
cd $WT
for v in "e3b/qwen3.5-2b/seed0:e3b" "e3a/full/qwen3.5-2b/seed0:e3a"; do p=${v%%:*}; n=${v##*:}
  for node in hpcgpu06 hpcgpu07; do $PY scripts/compare_train_logs.py $H/0044_hpc_consistency_$node/$p $B/$p --out $O/${n}_${node}_vs_b200.json --md $O/${n}_${node}_vs_b200.md > /dev/null 2>&1; echo "$n $node vs B200 rc=$?"; done
  $PY scripts/compare_train_logs.py $H/0044_hpc_consistency_hpcgpu06/$p $H/0044_hpc_consistency_hpcgpu07/$p --out $O/${n}_06_vs_07.json --md $O/${n}_06_vs_07.md > /dev/null 2>&1; echo "$n 06 vs 07 rc=$?"
done
for f in $O/*.md; do echo "=================== $(basename $f)"; cat $f; done
