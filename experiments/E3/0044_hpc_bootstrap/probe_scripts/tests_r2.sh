#!/bin/bash
# RUN-0044 §2 round 2 (after: HF cache for gemma/internvl tokenizer+config copied from the dev machine, git in env): full test suites on hpcgpu06 (CPU; GPU hidden, same condition as the dev-machine D87 count). Plain and server-parity, separate basetemp, env python directly.
export VI=/home/mail/chenyuma/visint
export TMPDIR=$VI/tmp
export XDG_CACHE_HOME=$VI/cache/hpcgpu06/xdg
export TRITON_CACHE_DIR=$VI/cache/hpcgpu06/triton
export HF_HOME=$VI/hf
export HF_HUB_CACHE=$VI/hf/hub
export HF_HUB_OFFLINE=1
export PIP_NO_CACHE_DIR=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES=
export PATH=$VI/env/bin:$PATH
cd $VI/wt/wt0044 || exit 1
L=$VI/logs/0044_tests_r2; mkdir -p $L $VI/tmp/pt_plain_r2 $VI/tmp/pt_parity_r2
echo "== $(date -u +%FT%TZ) host=$(hostname -s)"
( t0=$(date +%s); python -m pytest tests/ -q -p no:cacheprovider --basetemp=$VI/tmp/pt_plain_r2 > $L/plain.log 2>&1; echo "plain rc=$? $(( $(date +%s)-t0 ))s" >> $L/summary.txt ) &
( t0=$(date +%s); PYTHON=$VI/env/bin/python bash scripts/pytest_server_parity.sh -q -p no:cacheprovider --basetemp=$VI/tmp/pt_parity_r2 > $L/parity.log 2>&1; echo "parity rc=$? $(( $(date +%s)-t0 ))s" >> $L/summary.txt ) &
wait; tail -3 $L/plain.log $L/parity.log; cat $L/summary.txt
echo "== done $(date -u +%FT%TZ)"
