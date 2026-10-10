#!/bin/bash
# RUN-0044 §7–§8 driver, one node per invocation, launched inside the job's allocation (srun --overlap; RUN-0044 §6 decides from where).
#   bash driver_0044.sh consistency   # 2B e3b + e3a, 50 steps, worktree pinned at 6e77ab0 (RUN-0034's commit)
#   bash driver_0044.sh throughput    # this node's size: C3 and E3b+C3, 20 steps (Lustre data), then E3b+C3 20 steps from a /dev/shm copy (A/B)
# TINY=1 CODE=<repo> OUTROOT=<dir>: dev-machine rehearsal (tiny random model, CPU, repo fixtures), same command lines.
# Only writes results (under $VI/repo/experiments/E3/) and logs; git is done on hpcmgt (D89-8). Never touches Slurm.
# Stop lines (task book §9, D89-8): MemAvailable < 100 GB or our RSS > 150 GB -> stop our own process group, status "blocked".
set -uo pipefail
PHASE=${1:?consistency|throughput}
TINY=${TINY:-0}
NODE=$(hostname -s)
export VI=/home/mail/chenyuma/visint
if [ "$TINY" = 1 ]; then
  WT_CONS=${CODE:?}; WT_HEAD=${CODE:?}; OUT=${OUTROOT:?}; CK=$OUTROOT/ckpt; L=$OUTROOT/logs; TRAIN=; SHM=
  TFLAG=(--tiny); M_CONS=qwen3.5-0.8b; [ "$PHASE" = throughput ] && M_TPUT=qwen3.5-0.8b
else
  WT_CONS=$VI/wt/wt0044_6e77ab0; WT_HEAD=$VI/wt/wt0044
  OUT=$VI/repo/experiments/E3; CK=$VI/ckpt/$NODE/0044; L=$VI/logs/0044_$NODE; TRAIN=$VI/data/train_data
  SHM=/dev/shm/visint-chenyuma/train_data
  TFLAG=(); M_CONS=qwen3.5-2b
  case $NODE in hpcgpu06) M_TPUT=qwen3.5-4b ;; hpcgpu07) M_TPUT=qwen3.5-0.8b ;; *) echo "unknown node $NODE"; exit 64 ;; esac
  export HF_HOME=$VI/hf
  export HF_HUB_CACHE=$VI/hf/hub
  export HF_HUB_OFFLINE=1
  export HF_HUB_DISABLE_TELEMETRY=1
  export HF_XET_CACHE=$VI/tmp/xet
  export HF_XET_CHUNK_CACHE_SIZE_BYTES=0
  export HF_DATASETS_CACHE=$VI/tmp/hf-datasets
  export TMPDIR=$VI/tmp
  export PIP_NO_CACHE_DIR=1
  export XDG_CACHE_HOME=$VI/cache/$NODE/xdg
  export TRITON_CACHE_DIR=$VI/cache/$NODE/triton
  export TORCHINDUCTOR_CACHE_DIR=$VI/cache/$NODE/inductor
  export PATH=$VI/env/bin:$PATH
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export PYTHONHASHSEED=0
mkdir -p "$L" "$CK"
ST=$L/status_$PHASE.txt
beat() { echo "$(date -u +%FT%TZ) $*" > "$ST"; }
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$L/driver_$PHASE.log"; }
PG=$(ps -o pgid= $$ | tr -d ' ')
log "=== start phase=$PHASE node=$NODE pid=$$ pgid=$PG CVD=${CUDA_VISIBLE_DEVICES:-unset} nproc=$(nproc) step=${SLURM_JOB_ID:-}.${SLURM_STEP_ID:-}"
echo "$$ $PG" > "$L/driver_$PHASE.pid"

# samplers: GPU every 10 s, node load every 30 s; watchdog every 30 s (RSS of our process group, MemAvailable)
if [ "$TINY" != 1 ]; then
  ( while kill -0 $$ 2>/dev/null; do echo "$(date -u +%FT%TZ),$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits | tr -d ' ')" >> "$L/gpu_$PHASE.csv"; sleep 10; done ) &
  ( while kill -0 $$ 2>/dev/null; do echo "$(date -u +%FT%TZ) $(cut -d' ' -f1-3 /proc/loadavg) memavail_kB=$(awk '/MemAvailable/{print $2}' /proc/meminfo)" >> "$L/load_$PHASE.txt"; sleep 30; done ) &
  ( trap '' TERM; while kill -0 $$ 2>/dev/null; do
      ma=$(awk '/MemAvailable/{print $2*1024}' /proc/meminfo); rss=$(ps -e -o pgid=,rss= | awk -v pg="$PG" '$1==pg{s+=$2} END {print s*1024}')
      if [ "$ma" -lt 100000000000 ] || [ "$rss" -gt 150000000000 ]; then
        echo "$(date -u +%FT%TZ) STOP LINE: MemAvailable=$ma B, our RSS=$rss B; stopping our process group $PG" | tee -a "$L/driver_$PHASE.log"
        echo "$(date -u +%FT%TZ) blocked stop-line memavail=$ma rss=$rss" > "$ST"; kill -TERM -- -"$PG"; exit 0; fi
      sleep 30; done ) &
fi

# trn <name> <worktree> <script> args... : one training run, rc and wall time logged; data root / tiny flag / --resume appended
trn() { local name=$1 wt=$2; shift 2; local t0; t0=$(date +%s); beat "running $PHASE $name"
  local DA=(); [ -n "${DATA_ROOT:-}" ] && DA=(--data-root "$DATA_ROOT")
  ( cd "$wt" && python "$@" ${DA[@]+"${DA[@]}"} ${TFLAG[@]+"${TFLAG[@]}"} --resume ) > "$L/$name.log" 2>&1; local rc=$?
  log "=== $name rc=$rc $(( $(date +%s) - t0 ))s"; return $rc; }

if [ "$PHASE" = consistency ]; then
  [ "$TINY" != 1 ] && log "worktree $(git -C "$WT_CONS" rev-parse HEAD) dirty=$(git -C "$WT_CONS" status --short | wc -l)"
  O=$OUT/0044_hpc_consistency_$NODE; DATA_ROOT=$TRAIN
  trn consistency_e3b_$M_CONS "$WT_CONS" scripts/train_e3b.py --model "$M_CONS" --supervision grid --seed 0 \
    --out "$O" --ckpt-root "$CK/consistency" --stop-after-steps 50
  trn consistency_e3a_$M_CONS "$WT_CONS" scripts/train_e3a.py --model "$M_CONS" --variant full --seed 0 \
    --out "$O" --ckpt-root "$CK/consistency" --stop-after-steps 50
  rm -rf "$CK/consistency"
elif [ "$PHASE" = throughput ]; then
  [ "$TINY" != 1 ] && log "worktree $(git -C "$WT_HEAD" rev-parse HEAD) dirty=$(git -C "$WT_HEAD" status --short | wc -l)"
  O=$OUT/0044_hpc_throughput_$NODE; DATA_ROOT=$TRAIN
  trn tput_c3_$M_TPUT "$WT_HEAD" scripts/train_e3b.py --model "$M_TPUT" --supervision text --seed 0 \
    --out "$O" --ckpt-root "$CK/throughput" --stop-after-steps 20
  trn tput_e3b_c3_$M_TPUT "$WT_HEAD" scripts/train_e3b.py --model "$M_TPUT" --supervision grid+text --seed 0 \
    --out "$O" --ckpt-root "$CK/throughput" --stop-after-steps 20
  if [ "$TINY" != 1 ]; then  # A/B: same E3b+C3 20-step command, data from a /dev/shm copy
    beat "copy train data to $SHM"; t0=$(date +%s); mkdir -p "$SHM"
    for s in refcoco_train nyu_depth_train nyu_normals ade20k dedupe; do cp -a "$TRAIN/$s" "$SHM/"; done
    log "=== copy to shm rc=$? $(( $(date +%s) - t0 ))s $(du -sb "$SHM" | cut -f1) B"
    DATA_ROOT=$SHM
    trn tput_shm_e3b_c3_$M_TPUT "$WT_HEAD" scripts/train_e3b.py --model "$M_TPUT" --supervision grid+text --seed 0 \
      --out "$OUT/0044_hpc_throughput_shm_$NODE" --ckpt-root "$CK/throughput_shm" --stop-after-steps 20
    rm -rf /dev/shm/visint-chenyuma; log "=== removed /dev/shm/visint-chenyuma"
  fi
  rm -rf "$CK/throughput" "$CK/throughput_shm"
fi
log "=== done phase=$PHASE"
beat "done $PHASE"
