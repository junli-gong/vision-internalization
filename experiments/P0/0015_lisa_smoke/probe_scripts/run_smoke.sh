#!/bin/bash
# RUN-0015 §3: runs p19_lisa_smoke.py in the isolated LISA venv, offline (weights already on the RAM disk), cwd = LISA repo.
P=/home/ubuntu/workspace/wzg/gjl/vision-internalization
export HF_HOME=/home/ubuntu/workspace/wzg/gjl/.hf
export HF_HUB_CACHE=/dev/shm/visint/hf-hub
export TMPDIR=$P/tmp
export PIP_NO_CACHE_DIR=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR=$P/tmp/triton
export TORCHINDUCTOR_CACHE_DIR=$P/tmp/inductor
export XDG_CACHE_HOME=$P/tmp/xdg
E=$P/repo/experiments/P0/0015_lisa_smoke
cd $P/ext/lisa/LISA
( time $P/ext/lisa/venv/bin/python $E/probe_scripts/p19_lisa_smoke.py \
    /dev/shm/visint/hf-hub/models--xinlai--LISA-7B-v1/snapshots/43c754eef75871fcc11c84d3930a402f7b0a754f \
    /dev/shm/visint/hf-hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41 \
    $P/tmp/000000039769.jpg "Can you segment the cat on the left in this image?" $E/smoke_output ) 2>&1
