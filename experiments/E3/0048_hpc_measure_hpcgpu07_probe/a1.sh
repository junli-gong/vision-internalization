set -u
VI=/home/mail/chenyuma/visint
echo "## squeue $(date -u +%FT%TZ) [0048a before]"; squeue -h -j 157296,157297 -o "%i %T %M %N"
[ $(squeue -h -j 157296,157297 -t RUNNING | wc -l) -eq 2 ] || { echo "NOT BOTH RUNNING - STOP"; exit 9; }
grep -q "done 0048" $VI/logs/eval_0048.status || { echo "measure.sh not done - STOP"; exit 8; }
t0=$(date +%s); timeout 600 flock -w 300 $VI/logs/git.lock -c "git -C $VI/repo fetch -q origin main"; echo "fetch rc=$? $(( $(date +%s)-t0 ))s"
git -C $VI/repo log --oneline -1 origin/main
git -C $VI/repo merge-base --is-ancestor fa221598 origin/main && echo "fa221598 in origin/main: yes" || { echo "fa221598 NOT in origin/main - STOP"; exit 7; }
[ -e $VI/wt/wt0048a ] || { t0=$(date +%s); git -C $VI/repo worktree add -q --detach $VI/wt/wt0048a origin/main; echo "worktree add rc=$? $(( $(date +%s)-t0 ))s"; }
echo "wt0048a HEAD $(git -C $VI/wt/wt0048a rev-parse HEAD) status_lines=$(git -C $VI/wt/wt0048a status --short | wc -l)"
cd $VI/wt/wt0048a && sha256sum prompts/server_hpc/0048-hpc-measure-07/measure_e1.sh scripts/server/eval_lib.sh
