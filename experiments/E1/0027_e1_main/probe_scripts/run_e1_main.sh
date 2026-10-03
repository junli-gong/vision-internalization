#!/bin/bash
# RUN-0027 driver (worktree pinned at the task's start commit, /dev/shm/visint/tmp/wt0027; --out = main repo's run dir, absolute).
# Per model: ledger -> download + verify -> eval (all tasks) -> free (4 letter tasks) -> extra (refexp_box ovd, if the model has an
# extra policy) -> ledger -> delete from /dev/shm (except qwen3.5-0.8b) -> ledger -> touch logs/0027/done_<model>.
# Before every command: stop if cumulative time > 22 h (task book §4; writes cap_reached). Disk red line: stop.
source /home/ubuntu/workspace/wzg/gjl/vision-internalization/scripts/env.sh
export VISINT_DATA_ROOT=$PROJ/data
export NLTK_DATA=$PROJ/nltk_data
export TMPDIR=/dev/shm/visint/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=$PROJ/logs/0027; DC=$PROJ/scripts/diskcheck27.sh; CAP=79200
WT=/dev/shm/visint/tmp/wt0027; OUT=$REPO/experiments/E1/0027_e1_main
VER=$REPO/experiments/P0/0009_ifeval_rescore/probe_scripts/p12_verify_weights.py
cd $WT
echo "=== worktree $(git rev-parse HEAD) dirty_files=$(git status --short | wc -l)" | tee -a $L/driver.log
cap() { el=$(( $(date +%s) - $(cat $L/start_epoch.txt) )); if [ $el -gt $CAP ]; then echo "=== elapsed ${el}s > 22h; stopping before $1" | tee -a $L/driver.log; touch $L/cap_reached; echo "=== driver stopped $(date -Is)" | tee -a $L/driver.log; exit 0; fi; }
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
  step "${M}__eval" python scripts/run_e1.py --model "$M" --task all --subset eval --data-root $PROJ/data --out $OUT
  step "${M}__free" python scripts/run_e1.py --model "$M" --task rel_depth_nyu rel_depth_blink spatial_cvbench spatial_embspatial --subset free --data-root $PROJ/data --out $OUT
  case "$M" in
    qwen3.5-0.8b|qwen3.5-9b) POL=qwen:1024 ;;
    internvl3.5-2b-hf|internvl3.5-4b-hf) POL=internvl:native ;;
    *) POL= ;;
  esac
  if [ -n "$POL" ]; then step "${M}__extra" python scripts/run_e1.py --model "$M" --task refexp_box ovd --subset extra --policy "$POL" --data-root $PROJ/data --out $OUT; fi
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
