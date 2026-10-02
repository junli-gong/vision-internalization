#!/bin/bash
# RUN-0017 rerun driver for the last 4 models: same steps as run_health_v3.sh, but executed inside a git worktree pinned at c3f6983
# (the commit the first 4 models ran on), so that later pulls of the main checkout cannot change the code under the run.
# Outputs go to the main repo's run directory (absolute paths). bash run_health_v3_wt.sh <model key> [...]
source /home/ubuntu/workspace/wzg/gjl/vision-internalization/scripts/env.sh
export NLTK_DATA=$PROJ/nltk_data
L=$PROJ/logs/0017; DC=$PROJ/scripts/diskcheck17.sh; CAP=10800
WT=/dev/shm/visint/tmp/wt0017
H=$REPO/experiments/P0/0017_health_v3_base
VER=$REPO/experiments/P0/0009_ifeval_rescore/probe_scripts/p12_verify_weights.py
cd $WT
echo "=== worktree $(git rev-parse HEAD) dirty_files=$(git status --short | wc -l)" | tee -a $L/driver.log
for M in "$@"; do
  el=$(( $(date +%s) - $(cat $L/start_epoch.txt) ))
  if [ $el -gt $CAP ]; then echo "=== elapsed ${el}s > 3h; not starting $M" | tee -a $L/driver.log; touch $L/cap_reached; break; fi
  echo "=== $M start (worktree c3f6983) $(date -Is) (elapsed ${el}s)" | tee -a $L/driver.log
  $DC "before download $M" | tee -a $L/driver.log; [ ${PIPESTATUS[0]} -eq 3 ] && { echo "DISK RED LINE" | tee -a $L/driver.log; exit 3; }
  python $VER "$M" --download 2>&1 | grep -vE "Warn|it/s\]" | tee -a $L/driver.log
  echo "=== $M weights verify rc=${PIPESTATUS[0]}" | tee -a $L/driver.log
  $DC "after download $M" | tee -a $L/driver.log; [ ${PIPESTATUS[0]} -eq 3 ] && { echo "DISK RED LINE" | tee -a $L/driver.log; exit 3; }
  ( time python scripts/health_check.py --model "$M" --device cuda --dtype bf16 --out $H/health_$M.json ) > $L/health_$M.log 2>&1
  echo "=== $M health rc=$? $(date -Is)" | tee -a $L/driver.log
  python scripts/health_check.py --compare $REPO/experiments/P0/0011_health_v2_base/health_$M.json $H/health_$M.json \
      --out $H/compare_v2_v3_$M.json > $L/compare_$M.log 2>&1
  echo "=== $M compare rc=$?" | tee -a $L/driver.log
  ( time python $WT/experiments/P0/0017_health_v3_base/probe_scripts/audit_letter.py "$M" $H/health_$M.json $H/audit_letter_$M.json 100 ) > $L/audit_$M.log 2>&1
  echo "=== $M audit rc=$? $(date -Is) $(grep 'same_token' $L/audit_$M.log)" | tee -a $L/driver.log
  $DC "after health+audit $M" | tee -a $L/driver.log
  python - "$M" <<'PY' 2>&1 | grep -v Warn | tee -a $L/driver.log
import sys; sys.path.insert(0, '.')
from src.adapters import registry
from huggingface_hub import scan_cache_dir
e = registry.get_model(sys.argv[1]); ci = scan_cache_dir('/dev/shm/visint/hf-hub')
s = ci.delete_revisions(e.revision); print('delete', e.repo, e.revision, 'expected_freed', s.expected_freed_size); s.execute()
print('remaining', [(r.repo_id, r.size_on_disk) for r in scan_cache_dir('/dev/shm/visint/hf-hub').repos])
PY
  $DC "after delete $M" | tee -a $L/driver.log
done
echo "=== driver done $(date -Is)" | tee -a $L/driver.log
