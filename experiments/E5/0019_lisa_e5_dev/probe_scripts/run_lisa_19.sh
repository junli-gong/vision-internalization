#!/bin/bash
# RUN-0019 §3.2/§3.3: LISA E5 inference in the LISA venv, from the worktree pinned at 60fba2b. usage: bash run_lisa_19.sh <out subdir> [--limit N] [--resume]
source /home/ubuntu/workspace/wzg/gjl/vision-internalization/scripts/env.sh
export TMPDIR=$PROJ/tmp
export PIP_NO_CACHE_DIR=1
export HF_HOME=$ROOT/.hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd /dev/shm/visint/tmp/wt0019
OUTSUB=$1; shift
$PROJ/ext/lisa/venv/bin/python scripts/external/lisa_e5_infer.py --lisa-src $PROJ/ext/lisa/LISA \
  --inputs $PROJ/tmp/e5_inputs_dev_0019 --out /home/ubuntu/workspace/wzg/gjl/vision-internalization/repo/experiments/E5/0019_lisa_e5_dev/$OUTSUB \
  --lisa-weights /dev/shm/visint/hf-hub/models--xinlai--LISA-7B-v1/snapshots/43c754eef75871fcc11c84d3930a402f7b0a754f --clip-weights /dev/shm/visint/hf-hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41 "$@"
