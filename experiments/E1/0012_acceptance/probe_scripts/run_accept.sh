#!/bin/bash
# RUN-0012 §3 driver: bash run_accept.sh <model key> [...]. Per model: ledger -> download + verify (RUN-0009 manifest) -> run_e1 --acceptance
# -> ledger -> delete from /dev/shm (except qwen3.5-0.8b) -> ledger; stops after the current model once cumulative time > 6 h (task book §5).
source /home/ubuntu/workspace/wzg/gjl/vision-internalization/scripts/env.sh
export VISINT_DATA_ROOT=$PROJ/data
export NLTK_DATA=$PROJ/nltk_data
export TMPDIR=/dev/shm/visint/tmp
L=$PROJ/logs/0012; DC=$PROJ/scripts/diskcheck12.sh
VER=$REPO/experiments/P0/0009_ifeval_rescore/probe_scripts/p12_verify_weights.py
[ -f $L/start_epoch.txt ] || date +%s > $L/start_epoch.txt
cd $REPO
for M in "$@"; do
  echo "=== $M accept start $(date -Is)" | tee -a $L/driver.log
  $DC "before download $M" | tee -a $L/driver.log; [ ${PIPESTATUS[0]} -eq 3 ] && { echo "DISK RED LINE" | tee -a $L/driver.log; exit 3; }
  python $VER "$M" --download 2>&1 | grep -vE "Warn|it/s\]" | tee -a $L/driver.log
  echo "=== $M weights verify rc=${PIPESTATUS[0]}" | tee -a $L/driver.log
  $DC "after download $M" | tee -a $L/driver.log; [ ${PIPESTATUS[0]} -eq 3 ] && { echo "DISK RED LINE" | tee -a $L/driver.log; exit 3; }
  ( time python scripts/run_e1.py --acceptance --model "$M" --task all --batch-size 16 --limit 100 \
      --out experiments/E1/0012_acceptance ) > $L/accept_$M.log 2>&1
  echo "=== $M accept rc=$? $(date -Is)" | tee -a $L/driver.log
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
  el=$(( $(date +%s) - $(cat $L/start_epoch.txt) )); echo "=== elapsed ${el}s" | tee -a $L/driver.log
  if [ $el -gt 21600 ]; then echo "=== 6h budget exceeded; stopping after $M" | tee -a $L/driver.log; break; fi
done
echo "=== driver done $(date -Is)" | tee -a $L/driver.log
