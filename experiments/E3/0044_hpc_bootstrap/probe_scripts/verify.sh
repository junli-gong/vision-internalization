#!/bin/bash
# RUN-0044 §4 data checks (CPU, hpcgpu07): (1) recompute sha256 on disk vs each source's MANIFEST; (2) MANIFEST vs A800 (RUN-0034 p33, unchanged);
# (3) MANIFEST vs B200 (p44); (4) train_exclusions.json vs B200 and A800 (RUN-0034 p34, unchanged) + sha256.
export VI=/home/mail/chenyuma/visint
export PATH=$VI/env/bin:$PATH
export NPROC=16
TR=$VI/data/train_data; WT=$VI/wt/wt0044; P=$VI/probe/0044; O=$VI/logs/0044_verify; mkdir -p $O
cd $WT || exit 1
echo "== $(date -u +%FT%TZ) host=$(hostname -s)"
t0=$(date +%s); python $P/p44_disk_sha.py $TR $O/disk_sha refcoco_train nyu_depth_train nyu_normals ade20k; echo "=== disk_sha rc=$? $(( $(date +%s)-t0 ))s"
python $WT/experiments/E3/0034_b200_prep/probe_scripts/p33_manifest_compare.py $TR $WT $O/land_compare_a800; echo "=== p33 vs A800 rc=$?"; cat $O/land_compare_a800.md
python $P/p44_manifest_vs_b200.py $TR $WT $O/land_compare_b200; echo "=== p44 vs B200 rc=$?"
python $WT/experiments/E3/0034_b200_prep/probe_scripts/p34_exclusions_compare.py $TR/dedupe/train_exclusions.json $WT/experiments/E3/0034_b200_prep/dedupe/train_exclusions.json $O/dedupe_compare_b200; echo "=== p34 vs B200 rc=$?"; cat $O/dedupe_compare_b200.md
python $WT/experiments/E3/0034_b200_prep/probe_scripts/p34_exclusions_compare.py $TR/dedupe/train_exclusions.json $WT/experiments/P0/0026_a800_delta/results_0026/dedupe/train_exclusions.json $O/dedupe_compare_a800; echo "=== p34 vs A800 rc=$?"; cat $O/dedupe_compare_a800.md
sha256sum $TR/dedupe/train_exclusions.json $WT/experiments/E3/0034_b200_prep/dedupe/train_exclusions.json $WT/experiments/P0/0026_a800_delta/results_0026/dedupe/train_exclusions.json
echo "== done $(date -u +%FT%TZ)"
