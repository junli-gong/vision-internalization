#!/bin/bash
# RUN-0005 driver: bash run_models.sh <model key> [...]. Per model: ledger -> download (pinned revision) -> ledger -> probe -> ledger
# -> delete from /dev/shm via scan_cache_dir().delete_revisions() (except qwen3.5-0.8b) -> ledger. Budget stop (rc=4) ends the loop.
source /home/ubuntu/workspace/wzg/gjl/vision-internalization/scripts/env.sh
E=$REPO/experiments/P0/0005_thinking_probe; L=$PROJ/logs/0005; DC=$PROJ/scripts/diskcheck5.sh
IMG_A=$PROJ/tmp/000000039769.jpg; IMG_B=$PROJ/tmp/coco/000000000139.jpg
cd $REPO
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
  python experiments/P0/0005_thinking_probe/probe_scripts/p9_thinking_probe.py --model "$M" --image-a $IMG_A --image-b $IMG_B --out-dir $E > $L/probe_$M.log 2>&1
  RC=$?; echo "=== $M probe rc=$RC $(date -Is)" | tee -a $L/driver.log
  $DC "after probe $M" | tee -a $L/driver.log
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
  if [ $RC -eq 4 ]; then echo "=== budget stop after $M" | tee -a $L/driver.log; echo "budget" > $L/driver.stop; exit 4; fi
done
echo "=== driver done $(date -Is)" | tee -a $L/driver.log
