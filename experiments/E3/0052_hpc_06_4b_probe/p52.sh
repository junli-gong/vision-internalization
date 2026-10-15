#!/bin/bash
VI=/home/mail/chenyuma/visint; O=$VI/logs/0052_prep.txt
{ echo "## squeue $(date -u +%FT%TZ) [0052 prep]"; squeue -h -j 157296,157297 -o "%i %T %M %N"
t0=$(date +%s); timeout 600 flock -w 300 $VI/logs/git.lock -c "git -C $VI/repo fetch -q origin main"; echo "fetch rc=$? $(( $(date +%s)-t0 ))s"
git -C $VI/repo log --oneline -1 origin/main
git -C $VI/repo merge-base --is-ancestor ac1a3bad origin/main && echo "ac1a3bad in origin/main: yes" || echo "ac1a3bad NOT in origin/main"
[ -e $VI/wt/wt0052 ] || { t0=$(date +%s); git -C $VI/repo worktree add -q --detach $VI/wt/wt0052 origin/main; echo "worktree add rc=$? $(( $(date +%s)-t0 ))s"; }
echo "wt0052 HEAD $(git -C $VI/wt/wt0052 rev-parse HEAD) status_lines=$(git -C $VI/wt/wt0052 status --short | wc -l)"
cd $VI/wt/wt0052 && sha256sum prompts/server_hpc/0052-hpc-06-4b/run.sh prompts/server_hpc/0052-hpc-06-4b/eval.sh prompts/server_hpc/0052-hpc-06-4b/queue.sh scripts/server/eval_lib.sh scripts/server/train_queue.sh configs/e3b_train.json configs/e1_prompts.json configs/e1_posthoc.json
echo "## squeue $(date -u +%FT%TZ) [0052 prep after]"; squeue -h -j 157296,157297 -o "%i %T %M %N"; echo "== done"; } > $O 2>&1
