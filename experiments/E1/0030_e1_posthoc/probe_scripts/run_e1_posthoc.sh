#!/bin/bash
# RUN-0030 driver (E1 post-hoc readouts) (worktree pinned at the task's start commit, /dev/shm/visint/tmp/wt0030; --out = main repo's run dir, absolute).
# Per model: ledger -> download + verify -> debiased (all tasks) -> [Qwen only: forced (all tasks)] -> ledger -> delete from /dev/shm (except qwen3.5-0.8b) -> ledger -> touch logs/0030/done_<model>.
# Before each model: stop if cumulative run_e1 wall time > 3 h (task book §4: finish the current model, then stop; writes cap_reached). Disk red line: stop.
source /home/ubuntu/workspace/wzg/gjl/vision-internalization/scripts/env.sh
export VISINT_DATA_ROOT=$PROJ/data
export NLTK_DATA=$PROJ/nltk_data
export TMPDIR=/dev/shm/visint/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=$PROJ/logs/0030; DC=$PROJ/scripts/diskcheck30.sh; CAP=10800
WT=/dev/shm/visint/tmp/wt0030; OUT=$REPO/experiments/E1/0030_e1_posthoc
VER=$REPO/experiments/P0/0009_ifeval_rescore/probe_scripts/p12_verify_weights.py
cd $WT
echo "=== worktree $(git rev-parse HEAD) dirty_files=$(git status --short | wc -l)" | tee -a $L/driver.log
[ -f $L/gpu_s.txt ] || echo 0 > $L/gpu_s.txt
cap() { g=$(cat $L/gpu_s.txt); if [ $g -gt $CAP ]; then echo "=== cumulative run_e1 wall ${g}s > 3h; stopping before $1" | tee -a $L/driver.log; touch $L/cap_reached; echo "=== driver stopped $(date -Is)" | tee -a $L/driver.log; exit 0; fi; }
disk() { $DC "$1" | tee -a $L/driver.log; if [ ${PIPESTATUS[0]} -eq 3 ]; then echo "DISK RED LINE" | tee -a $L/driver.log; touch $L/disk_red; exit 3; fi; }
step() { name=$1; shift; t0=$(date +%s); echo "=== $name start $(date -Is) (run_e1 cumulative $(cat $L/gpu_s.txt)s)" | tee -a $L/driver.log
         ( time "$@" ) > $L/$name.log 2>&1; rc=$?; t1=$(date +%s); echo $(( $(cat $L/gpu_s.txt) + t1 - t0 )) > $L/gpu_s.txt
         echo "=== $name rc=$rc $(date -Is) ($((t1 - t0))s; cumulative $(cat $L/gpu_s.txt)s)" | tee -a $L/driver.log; disk "after $name"; }
for M in "$@"; do
  cap "$M"
  echo "=== $M model start $(date -Is)" | tee -a $L/driver.log
  disk "before download $M"
  python $VER "$M" --download 2>&1 | grep -vE "Warn|it/s\]" | tee -a $L/driver.log
  echo "=== $M weights verify rc=${PIPESTATUS[0]}" | tee -a $L/driver.log
  disk "after download $M"
  step "${M}__debiased" python scripts/run_e1.py --model "$M" --task all --subset debiased --data-root $PROJ/data --out $OUT
  case "$M" in qwen3.5-*)
    step "${M}__forced" python scripts/run_e1.py --model "$M" --task all --subset forced --data-root $PROJ/data --out $OUT ;;
  esac
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
