#!/bin/bash
# RUN-0039 §3 (server CC): E3a query-instruction probe, detached. Writes the pause file, waits until no train_*.status is running and the GPU
# has no process, runs scripts/diag/e3a_query_instruction_probe.py in the pinned worktree (58ffef4), writes e3a_probe.txt (main repo), removes
# the pause file (trap). Inputs: $PROJ/logs/0039/probe_inputs.txt (selected from swap_pairs.jsonl).
PROJ=/home/ubuntu/workspace/wzg/gjl/vision-internalization; ROOT=/home/ubuntu/workspace/wzg/gjl; L=$PROJ/logs/0039
export HF_HOME=$ROOT/.hf HF_HUB_CACHE=/dev/shm/visint/hf-hub HF_HUB_OFFLINE=1 TMPDIR=/dev/shm/visint/tmp PIP_NO_CACHE_DIR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
OUTF=$PROJ/repo/experiments/E3/0039_e3b_diag_2b_s0/e3a_probe.txt
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $L/probe_run.log; }
log "probe runner pid $$"
echo "0039-b200-e3b-diag.md (E3a probe)" > $PROJ/logs/train.pause; trap 'rm -f $PROJ/logs/train.pause; log "pause file removed"' EXIT
log "pause file written"
t0=$(date +%s)
while :; do
  run=$(for st in $PROJ/logs/train_*.status; do case "$(cut -d' ' -f2- $st)" in running*) echo $st;; esac; done)
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | grep -v '^$' | paste -sd, -)
  [ -z "$run" ] && [ -z "$pids" ] && break
  [ $(( $(date +%s) - t0 )) -ge 10800 ] && { log "gpu not free after 3 h; giving up"; exit 1; }
  sleep 30
done
log "gpu free after $(( $(date +%s) - t0 )) s; train_0036.status: $(cat $PROJ/logs/train_0036.status)"
cd /dev/shm/visint/tmp/wt0039 || exit 2
log "worktree $(git rev-parse HEAD)"
t1=$(date +%s)
$PROJ/venv/bin/python scripts/diag/e3a_query_instruction_probe.py --checkpoint $PROJ/ckpt/0036/e3a/full/qwen3.5-2b/seed0/ckpt/final \
  --device cuda --dtype bf16 \
  --image $PROJ/data/refcoco/images/COCO_train2014_000000574870.jpg $PROJ/data/refcoco/images/COCO_train2014_000000578805.jpg \
  --expr-a "woman on left" --expr-b "man right" --expr-c "woman on right" > $OUTF 2>&1
log "probe rc=$? wall_s=$(( $(date +%s) - t1 )) -> $OUTF"
