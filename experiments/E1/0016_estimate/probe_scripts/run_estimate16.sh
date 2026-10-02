#!/bin/bash
# RUN-0016 §3 driver: bash run_estimate16.sh <model key> [...]. Per model (not started if cumulative time since task start > 6 h, task book §4):
# ledger -> download + verify (RUN-0009 manifest) -> run_e1 --estimate (default policy, all tasks, bs 16 32 64, think n 64)
# -> extra policy (refexp_box ovd, bs 16 32 64, no think) for qwen3.5-9b/0.8b (qwen:1024) and internvl3.5-4b/2b (internvl:native)
# -> ledger -> delete from /dev/shm (except qwen3.5-0.8b) -> ledger.
source /home/ubuntu/workspace/wzg/gjl/vision-internalization/scripts/env.sh
export VISINT_DATA_ROOT=$PROJ/data
export NLTK_DATA=$PROJ/nltk_data
export TMPDIR=/dev/shm/visint/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=$PROJ/logs/0016; DC=$PROJ/scripts/diskcheck16.sh; CAP=21600
VER=$REPO/experiments/P0/0009_ifeval_rescore/probe_scripts/p12_verify_weights.py
OUT=experiments/E1/0016_estimate
cd $REPO
for M in "$@"; do
  el=$(( $(date +%s) - $(cat $L/start_epoch.txt) ))
  if [ $el -gt $CAP ]; then echo "=== elapsed ${el}s > 6h; not starting $M" | tee -a $L/driver.log; touch $L/cap_reached; break; fi
  echo "=== $M estimate start $(date -Is) (elapsed ${el}s)" | tee -a $L/driver.log
  $DC "before download $M" | tee -a $L/driver.log; [ ${PIPESTATUS[0]} -eq 3 ] && { echo "DISK RED LINE" | tee -a $L/driver.log; exit 3; }
  python $VER "$M" --download 2>&1 | grep -vE "Warn|it/s\]" | tee -a $L/driver.log
  echo "=== $M weights verify rc=${PIPESTATUS[0]}" | tee -a $L/driver.log
  $DC "after download $M" | tee -a $L/driver.log; [ ${PIPESTATUS[0]} -eq 3 ] && { echo "DISK RED LINE" | tee -a $L/driver.log; exit 3; }
  ( time python scripts/run_e1.py --estimate --model "$M" --task all --batch-sizes 16 32 64 \
      --estimate-n 64 --estimate-think --estimate-think-n 64 --out $OUT ) > $L/estimate_$M.log 2>&1
  echo "=== $M estimate rc=$? $(date -Is)" | tee -a $L/driver.log
  case "$M" in
    qwen3.5-0.8b|qwen3.5-9b) POL=qwen:1024 ;;
    internvl3.5-2b-hf|internvl3.5-4b-hf) POL=internvl:native ;;
    *) POL= ;;
  esac
  if [ -n "$POL" ]; then
    ( time python scripts/run_e1.py --estimate --model "$M" --task refexp_box ovd --batch-sizes 16 32 64 \
        --policy "$POL" --out $OUT ) > $L/estimate_${M}__${POL/:/_}.log 2>&1
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
    $DC "after delete $M" | tee -a $L/driver.log
  fi
  [ "$M" = "qwen3.5-9b" ] && touch $L/large4_done
done
echo "=== estimate driver done $(date -Is)" | tee -a $L/driver.log
