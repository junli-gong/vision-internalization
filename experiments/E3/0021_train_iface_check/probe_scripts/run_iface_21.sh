#!/bin/bash
# RUN-0021 driver, executed inside a git worktree pinned at the task's start commit (/dev/shm/visint/tmp/wt0021); outputs go to the main
# repo's run directory (absolute --out). Per model (not started if cumulative time > 1.5 h): ledger -> download + verify ->
# check_training_interfaces.py --checks restricted merge --n 200 -> ledger -> delete from /dev/shm (except qwen3.5-0.8b) -> ledger.
source /home/ubuntu/workspace/wzg/gjl/vision-internalization/scripts/env.sh
export VISINT_DATA_ROOT=$PROJ/data
export TMPDIR=/dev/shm/visint/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=$PROJ/logs/0021; DC=$PROJ/scripts/diskcheck21.sh; CAP=5400
WT=/dev/shm/visint/tmp/wt0021; OUT=$REPO/experiments/E3/0021_train_iface_check
VER=$REPO/experiments/P0/0009_ifeval_rescore/probe_scripts/p12_verify_weights.py
cd $WT
echo "=== worktree $(git rev-parse HEAD) dirty_files=$(git status --short | wc -l)" | tee -a $L/driver.log
for M in "$@"; do
  el=$(( $(date +%s) - $(cat $L/start_epoch.txt) ))
  if [ $el -gt $CAP ]; then echo "=== elapsed ${el}s > 1.5h; not starting $M" | tee -a $L/driver.log; touch $L/cap_reached; break; fi
  echo "=== $M start $(date -Is) (elapsed ${el}s)" | tee -a $L/driver.log
  $DC "before download $M" | tee -a $L/driver.log; [ ${PIPESTATUS[0]} -eq 3 ] && { echo "DISK RED LINE" | tee -a $L/driver.log; exit 3; }
  python $VER "$M" --download 2>&1 | grep -vE "Warn|it/s\]" | tee -a $L/driver.log
  echo "=== $M weights verify rc=${PIPESTATUS[0]}" | tee -a $L/driver.log
  $DC "after download $M" | tee -a $L/driver.log; [ ${PIPESTATUS[0]} -eq 3 ] && { echo "DISK RED LINE" | tee -a $L/driver.log; exit 3; }
  ( time python scripts/check_training_interfaces.py --model "$M" --checks restricted merge --n 200 \
      --data-root $PROJ/data --out $OUT/$M.json ) > $L/iface_$M.log 2>&1
  echo "=== $M check rc=$? $(date -Is)" | tee -a $L/driver.log
  $DC "after check $M" | tee -a $L/driver.log
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
