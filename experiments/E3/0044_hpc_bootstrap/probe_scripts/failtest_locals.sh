#!/bin/bash
# RUN-0044: full failure output (with locals, -l) of the one non-timing failing test, CPU, same env as tests_r3.sh
export VI=/home/mail/chenyuma/visint
export TMPDIR=$VI/tmp
export XDG_CACHE_HOME=$VI/cache/hpcgpu06/xdg
export TRITON_CACHE_DIR=$VI/cache/hpcgpu06/triton
export HF_HOME=$VI/hf
export HF_HUB_CACHE=$VI/hf/hub
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES=
export PATH=$VI/env/bin:$PATH
cd $VI/wt/wt0044 || exit 1
echo "== $(date -u +%FT%TZ) $(hostname -s) load: $(cat /proc/loadavg)"
echo "== $(date -u +%FT%TZ) $(hostname -s)"
python -m pytest -q -p no:cacheprovider --basetemp=$VI/tmp/pt_failtest -rf -l --tb=long "tests/test_check_training_interfaces.py::test_grid_checkpoint_split_across_merge_fails_restricted" 2>&1
echo "== rc=$? $(date -u +%FT%TZ)"
