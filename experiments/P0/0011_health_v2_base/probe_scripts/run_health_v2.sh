#!/bin/bash
# RUN-0011 driver: bash run_health_v2.sh <model key> [...]. Per model: ledger -> download + verify against RUN-0009 weights_manifest
# -> health_check.py v2 (--audit-fmt32 100) -> compare with RUN-0009 v1 -> ledger -> delete from /dev/shm (except qwen3.5-0.8b) -> ledger.
source /home/ubuntu/workspace/wzg/gjl/vision-internalization/scripts/env.sh
export NLTK_DATA=$PROJ/nltk_data
E=$REPO/experiments/P0/0011_health_v2_base; L=$PROJ/logs/0011; DC=$PROJ/scripts/diskcheck11.sh
VER=$REPO/experiments/P0/0009_ifeval_rescore/probe_scripts/p12_verify_weights.py
cd $REPO
for M in "$@"; do
  echo "=== $M start $(date -Is)" | tee -a $L/driver.log
  $DC "before download $M" | tee -a $L/driver.log || { echo "DISK RED LINE" | tee -a $L/driver.log; exit 3; }
  python $VER "$M" --download 2>&1 | grep -vE "Warn|it/s\]" | tee -a $L/driver.log
  echo "=== $M weights verify rc=${PIPESTATUS[0]}" | tee -a $L/driver.log
  $DC "after download $M" | tee -a $L/driver.log || { echo "DISK RED LINE" | tee -a $L/driver.log; exit 3; }
  ( time python scripts/health_check.py --model "$M" --device cuda --dtype bf16 --audit-fmt32 100 \
      --out experiments/P0/0011_health_v2_base/health_$M.json ) > $L/health_$M.log 2>&1
  echo "=== $M health rc=$? $(date -Is)" | tee -a $L/driver.log
  python scripts/health_check.py --compare experiments/P0/0009_ifeval_rescore/health_$M.json \
      experiments/P0/0011_health_v2_base/health_$M.json --out experiments/P0/0011_health_v2_base/compare_v1_v2_$M.json > $L/compare_$M.log 2>&1
  echo "=== $M compare rc=$?" | tee -a $L/driver.log
  $DC "after health $M" | tee -a $L/driver.log
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
