#!/bin/bash
# RUN-0025 driver (worktree pinned at the task's start commit, /dev/shm/visint/tmp/wt0025; outputs to the main repo's run dir).
# Each setting runs in its own new process. S1 = RUN-0024 env; S2 = S1 in 3 new processes x 3 passes; S3 = S1 + CUBLAS_WORKSPACE_CONFIG=:4096:8
# + torch.use_deterministic_algorithms(True, warn_only=True); S4 = S3 + FLA_CACHE_MODE=default. Then the full refexp_box dev subset
# (run_e1.py --acceptance --task refexp_box) under S1 and under S3. Stops before a step if cumulative time > 3 h.
source /home/ubuntu/workspace/wzg/gjl/vision-internalization/scripts/env.sh
export VISINT_DATA_ROOT=$PROJ/data
export TMPDIR=/dev/shm/visint/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=$PROJ/logs/0025; DC=$PROJ/scripts/diskcheck25.sh; CAP=10800
WT=/dev/shm/visint/tmp/wt0025; R=$REPO/experiments/E1/0025_determinism_probe; PS=$WT/experiments/E1/0025_determinism_probe/probe_scripts
VER=$REPO/experiments/P0/0009_ifeval_rescore/probe_scripts/p12_verify_weights.py
cd $WT
cap() { el=$(( $(date +%s) - $(cat $L/start_epoch.txt) )); if [ $el -gt $CAP ]; then echo "=== elapsed ${el}s > 3h; stopping before $1" | tee -a $L/driver.log; exit 0; fi; }
echo "=== worktree $(git rev-parse HEAD)" | tee -a $L/driver.log
$DC "before download qwen3.5-2b" | tee -a $L/driver.log
python $VER qwen3.5-2b --download 2>&1 | grep -vE "Warn|it/s\]" | tee -a $L/driver.log
$DC "after download qwen3.5-2b" | tee -a $L/driver.log
run() { name=$1; shift; cap $name; echo "=== $name start $(date -Is)" | tee -a $L/driver.log; ( time "$@" ) > $L/$name.log 2>&1; echo "=== $name rc=$? $(date -Is) $(grep -h 'pid .*passes' $L/$name.log | tail -1)" | tee -a $L/driver.log; }
run S1 python $PS/rerun_batch.py --setting S1 --passes 20 --out $R/probe_out/S1.json
for k in a b c; do run S2$k python $PS/rerun_batch.py --setting S2$k --passes 3 --out $R/probe_out/S2$k.json; done
run S3 env CUBLAS_WORKSPACE_CONFIG=:4096:8 python $PS/rerun_batch.py --setting S3 --passes 20 --deterministic --out $R/probe_out/S3.json
run S4 env CUBLAS_WORKSPACE_CONFIG=:4096:8 FLA_CACHE_MODE=default python $PS/rerun_batch.py --setting S4 --passes 20 --deterministic --out $R/probe_out/S4.json
run full_S1 python scripts/run_e1.py --acceptance --model qwen3.5-2b --task refexp_box --data-root $PROJ/data --out $R/0025_full_S1
run full_S3 env CUBLAS_WORKSPACE_CONFIG=:4096:8 python $PS/run_e1_det.py --warn-out $R/probe_out/full_S3_warnings.json -- --acceptance --model qwen3.5-2b --task refexp_box --data-root $PROJ/data --out $R/0025_full_S3
$DC "after probes" | tee -a $L/driver.log
python - <<'PY' 2>&1 | grep -v Warn | tee -a $L/driver.log
import sys; sys.path.insert(0, '.')
from src.adapters import registry
from huggingface_hub import scan_cache_dir
e = registry.get_model('qwen3.5-2b'); ci = scan_cache_dir('/dev/shm/visint/hf-hub')
s = ci.delete_revisions(e.revision); print('delete', e.repo, e.revision, 'expected_freed', s.expected_freed_size); s.execute()
print('remaining', [(r.repo_id, r.size_on_disk) for r in scan_cache_dir('/dev/shm/visint/hf-hub').repos])
PY
$DC "after delete qwen3.5-2b" | tee -a $L/driver.log
echo "=== driver done $(date -Is)" | tee -a $L/driver.log
