#!/bin/bash
# RUN-0007 driver: bash run_health.sh <model key> [...]. Per model: ledger -> snapshot_download(pinned revision) -> ledger ->
# scripts/health_check.py (task-book command) -> ledger -> delete from /dev/shm via scan_cache_dir().delete_revisions()
# (except qwen3.5-0.8b) -> ledger. 8 h cumulative budget measured from the first model start (health_start_epoch.txt).
source /home/ubuntu/workspace/wzg/gjl/vision-internalization/scripts/env.sh
E=$REPO/experiments/P0/0007_health_base; L=$PROJ/logs/0007; DC=$PROJ/scripts/diskcheck7.sh
cd $REPO
[ -f $L/health_start_epoch.txt ] || date +%s > $L/health_start_epoch.txt
for M in "$@"; do
  echo "=== $M start $(date -Is)" | tee -a $L/driver.log
  $DC "before download $M" | tee -a $L/driver.log || { echo "DISK RED LINE" | tee -a $L/driver.log; exit 3; }
  python - "$M" <<'PY' 2>&1 | grep -vE "Warn|it/s\]" | tee -a $L/driver.log
import sys; sys.path.insert(0, '.')
from src.adapters import registry
from huggingface_hub import snapshot_download
e = registry.get_model(sys.argv[1]); print('download', e.repo, e.revision, snapshot_download(e.repo, revision=e.revision))
PY
  $DC "after download $M" | tee -a $L/driver.log || { echo "DISK RED LINE" | tee -a $L/driver.log; exit 3; }
  ( time python scripts/health_check.py --model "$M" --device cuda --dtype bf16 --skip-ifeval-scoring \
      --out experiments/P0/0007_health_base/health_$M.json ) > $L/health_$M.log 2>&1
  RC=$?; echo "=== $M health rc=$RC $(date -Is)" | tee -a $L/driver.log
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
  EL=$(( $(date +%s) - $(cat $L/health_start_epoch.txt) ))
  if [ $EL -gt 28800 ]; then echo "=== 8h budget exceeded after $M (elapsed ${EL}s)" | tee -a $L/driver.log; echo "budget after $M" > $L/driver.stop; exit 4; fi
done
echo "=== driver done $(date -Is)" | tee -a $L/driver.log
