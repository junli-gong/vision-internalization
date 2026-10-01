#!/bin/bash
# RUN-0013 §3 driver: bash run_accept13.sh <model key> [...]. Per model (not started if cumulative time > 8 h, task book §5):
# ledger -> download + verify -> acceptance (bitwise) -> --resume iou90 -> --resume outcome (each summary copied to acceptance__<mode>.*)
# -> same three for --task count --count-questions countbenchqa (separate --out, since the leaf dir does not encode the question source)
# -> ledger -> delete from /dev/shm (except qwen3.5-0.8b) -> ledger.
source /home/ubuntu/workspace/wzg/gjl/vision-internalization/scripts/env.sh
export VISINT_DATA_ROOT=$PROJ/data
export NLTK_DATA=$PROJ/nltk_data
export TMPDIR=/dev/shm/visint/tmp
L=$PROJ/logs/0013; DC=$PROJ/scripts/diskcheck13.sh; CAP=28800
VER=$REPO/experiments/P0/0009_ifeval_rescore/probe_scripts/p12_verify_weights.py
OUT=experiments/E1/0013_acceptance; OUTC=experiments/E1/0013_acceptance_countbenchqa
[ -f $L/start_epoch.txt ] || date +%s > $L/start_epoch.txt
cd $REPO
keep() {  # keep() <base dir> <mode>: copy the summary just written so the next recompute does not overwrite it
  cp "$1/acceptance.json" "$1/acceptance__$2.json"; cp "$1/acceptance.md" "$1/acceptance__$2.md"; }
for M in "$@"; do
  el=$(( $(date +%s) - $(cat $L/start_epoch.txt) ))
  if [ $el -gt $CAP ]; then echo "=== elapsed ${el}s > 8h; not starting acceptance $M" | tee -a $L/driver.log; touch $L/cap_reached; break; fi
  echo "=== $M accept start $(date -Is) (elapsed ${el}s)" | tee -a $L/driver.log
  $DC "before download $M" | tee -a $L/driver.log; [ ${PIPESTATUS[0]} -eq 3 ] && { echo "DISK RED LINE" | tee -a $L/driver.log; exit 3; }
  python $VER "$M" --download 2>&1 | grep -vE "Warn|it/s\]" | tee -a $L/driver.log
  echo "=== $M weights verify rc=${PIPESTATUS[0]}" | tee -a $L/driver.log
  $DC "after download $M" | tee -a $L/driver.log; [ ${PIPESTATUS[0]} -eq 3 ] && { echo "DISK RED LINE" | tee -a $L/driver.log; exit 3; }
  ( time python scripts/run_e1.py --acceptance --model "$M" --task all --batch-size 16 --limit 100 --agreement bitwise --out $OUT ) > $L/accept_$M.log 2>&1
  echo "=== $M accept bitwise rc=$? $(date -Is)" | tee -a $L/driver.log; keep $OUT/$M/acceptance bitwise
  for MODE in iou90 outcome; do
    ( time python scripts/run_e1.py --acceptance --model "$M" --task all --batch-size 16 --limit 100 --resume --agreement $MODE --out $OUT ) > $L/accept_${M}__$MODE.log 2>&1
    echo "=== $M accept $MODE (resume) rc=$? $(date -Is)" | tee -a $L/driver.log; keep $OUT/$M/acceptance $MODE
  done
  ( time python scripts/run_e1.py --acceptance --model "$M" --task count --count-questions countbenchqa --batch-size 16 --limit 100 --agreement bitwise --out $OUTC ) > $L/acceptc_$M.log 2>&1
  echo "=== $M accept count/countbenchqa bitwise rc=$? $(date -Is)" | tee -a $L/driver.log; keep $OUTC/$M/acceptance bitwise
  for MODE in iou90 outcome; do
    ( time python scripts/run_e1.py --acceptance --model "$M" --task count --count-questions countbenchqa --batch-size 16 --limit 100 --resume --agreement $MODE --out $OUTC ) > $L/acceptc_${M}__$MODE.log 2>&1
    echo "=== $M accept count/countbenchqa $MODE (resume) rc=$? $(date -Is)" | tee -a $L/driver.log; keep $OUTC/$M/acceptance $MODE
  done
  $DC "after accept $M" | tee -a $L/driver.log
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
echo "=== accept driver done $(date -Is)" | tee -a $L/driver.log
