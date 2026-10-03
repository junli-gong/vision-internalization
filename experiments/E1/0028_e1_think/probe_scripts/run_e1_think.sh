#!/bin/bash
# RUN-0028 driver (think subset; one command per model) (worktree pinned at the task's start commit, /dev/shm/visint/tmp/wt0028; --out = main repo's run dir, absolute).
# Per model: ledger -> download + verify -> think subset (all tasks, budget-forced 4096 + 1024) -> ledger -> delete from /dev/shm (except qwen3.5-0.8b) -> ledger -> touch logs/0028/done_<model>.
# Before every command: stop if cumulative time > 22 h (> 15 h, task book §4; writes cap_reached). Disk red line: stop.
source /home/ubuntu/workspace/wzg/gjl/vision-internalization/scripts/env.sh
export VISINT_DATA_ROOT=$PROJ/data
export NLTK_DATA=$PROJ/nltk_data
export TMPDIR=/dev/shm/visint/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=$PROJ/logs/0028; DC=$PROJ/scripts/diskcheck28.sh; CAP=54000
WT=/dev/shm/visint/tmp/wt0028; OUT=$REPO/experiments/E1/0028_e1_think
VER=$REPO/experiments/P0/0009_ifeval_rescore/probe_scripts/p12_verify_weights.py
cd $WT
echo "=== worktree $(git rev-parse HEAD) dirty_files=$(git status --short | wc -l)" | tee -a $L/driver.log
cap() { el=$(( $(date +%s) - $(cat $L/start_epoch.txt) )); if [ $el -gt $CAP ]; then echo "=== elapsed ${el}s > 15h; stopping before $1" | tee -a $L/driver.log; touch $L/cap_reached; echo "=== driver stopped $(date -Is)" | tee -a $L/driver.log; exit 0; fi; }
disk() { $DC "$1" | tee -a $L/driver.log; if [ ${PIPESTATUS[0]} -eq 3 ]; then echo "DISK RED LINE" | tee -a $L/driver.log; touch $L/disk_red; exit 3; fi; }
step() { name=$1; shift; cap "$name"; echo "=== $name start $(date -Is) (elapsed $(( $(date +%s) - $(cat $L/start_epoch.txt) ))s)" | tee -a $L/driver.log
         ( time "$@" ) > $L/$name.log 2>&1; rc=$?; echo "=== $name rc=$rc $(date -Is)" | tee -a $L/driver.log; disk "after $name"; }
for M in "$@"; do
  cap "$M"
  echo "=== $M model start $(date -Is)" | tee -a $L/driver.log
  disk "before download $M"
  python $VER "$M" --download 2>&1 | grep -vE "Warn|it/s\]" | tee -a $L/driver.log
  echo "=== $M weights verify rc=${PIPESTATUS[0]}" | tee -a $L/driver.log
  disk "after download $M"
  step "${M}__think" python scripts/run_e1.py --model "$M" --task all --subset think --data-root $PROJ/data --out $OUT
  if [ "$M" != "qwen3.5-0.8b" ]; then
    python - "$M" <<'PY' 2>&1 | grep -v Warn | tee -a $L/driver.log
import sys; sys.path.insert(0, '.')
from src.adapters import registry
from huggingface_hub import scan_cache_dir
e = registry.get_model(sys.argv[1]); ci = scan_cache_dir('/dev/shm/visint/hf-hub')
s = ci.delete_revisions(e.revision); print('delete', e.repo, e.revision, 'expected_freed', s.expected_freed_size); s.execute()
print('remaining', [(r.repo_id, r.size_on_disk) for r in scan_cache_dir('/dev/shm/visint/hf-hub').repos])
PY
    disk "after delete $M"
  fi
  touch $L/done_$M
  echo "=== $M model done $(date -Is)" | tee -a $L/driver.log
done
echo "=== driver done $(date -Is)" | tee -a $L/driver.log
