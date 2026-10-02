#!/bin/bash
# RUN-0013 §4 driver: bash run_estimate.sh <model key> [...]. Per model (skipped if cumulative time > 8 h before it starts, task book §5):
# ledger -> download + verify -> run_e1 --estimate (default policy, all tasks) -> extra policy (refexp_box ovd) for qwen3.5-0.8b/9b
# (qwen:1024) and internvl3.5-2b/4b (internvl:native) -> ledger -> delete from /dev/shm (except qwen3.5-0.8b) -> ledger.
source /home/ubuntu/workspace/wzg/gjl/vision-internalization/scripts/env.sh
export VISINT_DATA_ROOT=$PROJ/data
export NLTK_DATA=$PROJ/nltk_data
export TMPDIR=/dev/shm/visint/tmp
L=$PROJ/logs/0013; DC=$PROJ/scripts/diskcheck13.sh
VER=$REPO/experiments/P0/0009_ifeval_rescore/probe_scripts/p12_verify_weights.py
cd $REPO
for M in "$@"; do
  el=$(( $(date +%s) - $(cat $L/start_epoch.txt) ))
  if [ $el -gt 28800 ]; then echo "=== elapsed ${el}s > 8h; not starting estimate $M" | tee -a $L/driver.log; break; fi
  echo "=== $M estimate start $(date -Is) (elapsed ${el}s)" | tee -a $L/driver.log
  $DC "before download $M (estimate)" | tee -a $L/driver.log; [ ${PIPESTATUS[0]} -eq 3 ] && { echo "DISK RED LINE" | tee -a $L/driver.log; exit 3; }
  python $VER "$M" --download 2>&1 | grep -vE "Warn|it/s\]" | tee -a $L/driver.log
  echo "=== $M weights verify rc=${PIPESTATUS[0]}" | tee -a $L/driver.log
  $DC "after download $M (estimate)" | tee -a $L/driver.log; [ ${PIPESTATUS[0]} -eq 3 ] && { echo "DISK RED LINE" | tee -a $L/driver.log; exit 3; }
  ( time python scripts/run_e1.py --estimate --model "$M" --task all --batch-sizes 8 16 32 --estimate-think \
      --out experiments/E1/0013_estimate ) > $L/estimate_$M.log 2>&1
  echo "=== $M estimate rc=$? $(date -Is)" | tee -a $L/driver.log
  case "$M" in
    qwen3.5-0.8b|qwen3.5-9b) POL=qwen:1024 ;;
    internvl3.5-2b-hf|internvl3.5-4b-hf) POL=internvl:native ;;
    *) POL= ;;
  esac
  if [ -n "$POL" ]; then
    ( time python scripts/run_e1.py --estimate --model "$M" --task refexp_box ovd --batch-sizes 8 16 32 --estimate-think \
        --policy "$POL" --out experiments/E1/0013_estimate ) > $L/estimate_${M}__${POL/:/_}.log 2>&1
    echo "=== $M estimate policy $POL rc=$? $(date -Is)" | tee -a $L/driver.log
  fi
  $DC "after estimate $M" | tee -a $L/driver.log
  if [ "$M" != "qwen3.5-0.8b" ]; then
    python - "$M" <<'PY' 2>&1 | grep -v Warn | tee -a $L/driver.log
import sys; sys.path.insert(0, '.')
from src.adapters import registry
from huggingface_hub import scan_cache_dir
e = registry.get_model(sys.argv[1]); ci = scan_cache_dir('/dev/shm/visint/hf-hub')
s = ci.delete_revisions(e.revision); print('delete', e.repo, e.revision, 'expected_freed', s.expected_freed_size); s.execute()
print('remaining', [(r.repo_id, r.size_on_disk) for r in scan_cache_dir('/dev/shm/visint/hf-hub').repos])
PY
    $DC "after delete $M (estimate)" | tee -a $L/driver.log
  fi
done
echo "=== estimate driver done $(date -Is)" | tee -a $L/driver.log
