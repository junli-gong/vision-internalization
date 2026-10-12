#!/bin/bash
# RUN-0049: merge-check diagnosis + two test groups, CPU only (command content as in the task book §3)
export VI=/home/mail/chenyuma/visint
export TMPDIR=$VI/tmp
export HF_HOME=$VI/hf
export HF_HUB_CACHE=$VI/hf/hub
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES=
export PATH=$VI/env/bin:/usr/bin:/bin
cd $VI/wt/wt0049 || exit 1
echo "== $(date -u +%FT%TZ) $(hostname -s) wt=$(git rev-parse HEAD) lscpu: $(lscpu | grep 'Model name' | sed 's/  */ /g')"
python scripts/diag/merge_split_probe.py --out $VI/tmp/msp_0049_$(hostname -s).json --compare scripts/diag/merge_split_probe.dev.json; echo diag_rc=$?
python -m pytest -q -p no:cacheprovider --basetemp=$VI/tmp/pt_0049_$(hostname -s) -rfEs tests/test_check_training_interfaces.py "tests/test_e3_eval.py::test_bad_checkpoint_refused"; echo pytest_rc=$?
echo "== done $(date -u +%FT%TZ)"
