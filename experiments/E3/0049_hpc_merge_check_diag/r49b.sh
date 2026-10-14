#!/bin/bash
# RUN-0049 (06) completion: rerun the one test that failed only because wt0049 had been made sparse (experiments/E3/0021_train_iface_check missing)
VI=/home/mail/chenyuma/visint; O=$VI/logs/0049b_06.txt; W=$VI/wt/wt0049b
{ echo "## squeue $(date -u +%FT%TZ) [0049b before]"; squeue -h -j 157296,157297 -o "%i %T %M %N"
git -C $VI/repo worktree add -q --detach $W 59f086f8f02731531a2e903491eedc404c07fde4; echo "worktree add rc=$?"
git -C $W sparse-checkout set --no-cone '/*' '!/experiments/*/' '/experiments/P0/' '/experiments/E3/0021_train_iface_check/'; echo "sparse rc=$?"
echo "HEAD $(git -C $W rev-parse HEAD) status_lines=$(git -C $W status --short | wc -l) 0021 files: $(ls $W/experiments/E3/0021_train_iface_check | wc -l)"
cat > $VI/tmp/r49b_node.sh <<'IN'
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
cd $VI/wt/wt0049b || exit 1
echo "== $(date -u +%FT%TZ) $(hostname -s) wt=$(git rev-parse HEAD)"
python -m pytest -q -p no:cacheprovider --basetemp=$VI/tmp/pt_0049b_$(hostname -s) -rfEs "tests/test_check_training_interfaces.py::test_v1_rejudges_run0021_all_pass"; echo pytest_rc=$?
IN
timeout 1200 ssh -n hpcgpu06 "srun --jobid=157296 --overlap -n1 bash $VI/tmp/r49b_node.sh < /dev/null" 2>&1 | tail -6
rm -f $VI/tmp/r49b_node.sh; rm -rf $VI/tmp/pt_0049b_hpcgpu06
git -C $VI/repo worktree remove $W && echo "removed wt0049b"
echo "## squeue $(date -u +%FT%TZ) [0049b after]"; squeue -h -j 157296,157297 -o "%i %T %M %N"; echo "== done"; } > $O 2>&1
