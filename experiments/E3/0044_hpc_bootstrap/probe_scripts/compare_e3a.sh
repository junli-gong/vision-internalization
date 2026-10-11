#!/bin/bash
# RUN-0044 §7: e3a vs B200 with --ignore-key class_names_check.file (only differing identity key: a local path; its sha256 is identical on both sides)
VI=/home/mail/chenyuma/visint; WT=$VI/wt/wt0044; PY=$VI/env/bin/python
B=$WT/experiments/E3/0034_b200_prep/0034_e3_consistency; H=$VI/repo/experiments/E3; O=$H/0044_hpc_consistency_compare
cd $WT; p=e3a/full/qwen3.5-2b/seed0
for node in hpcgpu06 hpcgpu07; do $PY scripts/compare_train_logs.py $H/0044_hpc_consistency_$node/$p $B/$p --ignore-key class_names_check.file --out $O/e3a_${node}_vs_b200.json --md $O/e3a_${node}_vs_b200.md > /dev/null 2>&1; echo "e3a $node vs B200 (--ignore-key class_names_check.file) rc=$?"; grep -E "^# |^- (忽略|数据|每步|最后|NaN|grad)" $O/e3a_${node}_vs_b200.md; done
