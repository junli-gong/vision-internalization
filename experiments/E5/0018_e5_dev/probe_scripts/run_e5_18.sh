#!/bin/bash
# RUN-0018 driver, executed inside a git worktree pinned at the task's start commit (/dev/shm/visint/tmp/wt0018), so that pulls of the
# main checkout during the run cannot change the code; outputs go to the main repo's run directory (absolute --out).
# 1) trivial baselines (CPU); 2) per model (not started if cumulative time > 4 h): ledger -> download + verify -> run_e5 zeroshot+cot
#    -> ledger -> delete from /dev/shm (except qwen3.5-0.8b) -> ledger.   usage: bash run_e5_18.sh <model key> [...]
source /home/ubuntu/workspace/wzg/gjl/vision-internalization/scripts/env.sh
export TMPDIR=/dev/shm/visint/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=$PROJ/logs/0018; DC=$PROJ/scripts/diskcheck18.sh; CAP=14400
WT=/dev/shm/visint/tmp/wt0018; OUT=$REPO/experiments/E5/0018_e5_dev
VER=$REPO/experiments/P0/0009_ifeval_rescore/probe_scripts/p12_verify_weights.py
cd $WT
echo "=== worktree $(git rev-parse HEAD) dirty_files=$(git status --short | wc -l)" | tee -a $L/driver.log
( time python scripts/run_e5.py --system baseline --task all --subset dev --n-per-bin 4 --out $OUT ) > $L/baseline.log 2>&1
echo "=== baseline rc=$? $(date -Is)" | tee -a $L/driver.log
for M in "$@"; do
  el=$(( $(date +%s) - $(cat $L/start_epoch.txt) ))
  if [ $el -gt $CAP ]; then echo "=== elapsed ${el}s > 4h; not starting $M" | tee -a $L/driver.log; touch $L/cap_reached; break; fi
  echo "=== $M start $(date -Is) (elapsed ${el}s)" | tee -a $L/driver.log
  $DC "before download $M" | tee -a $L/driver.log; [ ${PIPESTATUS[0]} -eq 3 ] && { echo "DISK RED LINE" | tee -a $L/driver.log; exit 3; }
  python $VER "$M" --download 2>&1 | grep -vE "Warn|it/s\]" | tee -a $L/driver.log
  echo "=== $M weights verify rc=${PIPESTATUS[0]}" | tee -a $L/driver.log
  $DC "after download $M" | tee -a $L/driver.log; [ ${PIPESTATUS[0]} -eq 3 ] && { echo "DISK RED LINE" | tee -a $L/driver.log; exit 3; }
  ( time python scripts/run_e5.py --system zeroshot cot --model "$M" --task all --subset dev --n-per-bin 4 --out $OUT ) > $L/e5_$M.log 2>&1
  echo "=== $M run_e5 rc=$? $(date -Is)" | tee -a $L/driver.log
  $DC "after run $M" | tee -a $L/driver.log
  if [ "$M" != "qwen3.5-0.8b" ]; then
    python - "$M" <<'PY' 2>&1 | grep -v Warn | tee -a $L/driver.log
import sys; sys.path.insert(0, '.')
from src.adapters import registry
from huggingface_hub import scan_cache_dir
e = registry.get_model(sys.argv[1]); ci = scan_cache_dir('/dev/shm/visint/hf-hub')
s = ci.delete_revisions(e.revision); print('delete', e.repo, e.revision, 'expected_freed', s.expected_freed_size); s.execute()
print('remaining', [(r.repo_id, r.size_on_disk) for r in scan_cache_dir('/dev/shm/visint/hf-hub').repos])
PY
    $DC "after delete $M" | tee -a $L/driver.log
  fi
done
echo "=== driver done $(date -Is)" | tee -a $L/driver.log
