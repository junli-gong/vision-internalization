set -u
VI=/home/mail/chenyuma/visint
echo "## squeue $(date -u +%FT%TZ) [0048 prep]"; squeue -h -j 157296,157297 -o "%i %T %M %N"
t0=$(date +%s); timeout 600 flock -w 300 $VI/logs/git.lock -c "git -C $VI/repo fetch -q origin main"; echo "fetch rc=$? $(( $(date +%s)-t0 ))s"
git -C $VI/repo log --oneline -1 origin/main
git -C $VI/repo merge-base --is-ancestor f0a699b0 origin/main && echo "f0a699b0 in origin/main: yes" || echo "f0a699b0 NOT in origin/main"
[ -e $VI/wt/wt0048 ] && echo "wt0048 exists" || { t0=$(date +%s); git -C $VI/repo worktree add -q --detach $VI/wt/wt0048 origin/main; echo "worktree add rc=$? $(( $(date +%s)-t0 ))s"; }
echo "wt0048 HEAD $(git -C $VI/wt/wt0048 rev-parse HEAD) status_lines=$(git -C $VI/wt/wt0048 status --short | wc -l)"
cd $VI/wt/wt0048 && sha256sum prompts/server_hpc/0048-hpc-measure-07/measure.sh scripts/server/eval_lib.sh
ls -la $VI/logs/train_0045-hpcgpu07.done; cat $VI/logs/train_0045-hpcgpu07.status
for f in $VI/logs/train.pause $VI/logs/train.pause.hpcgpu07 $VI/logs/HOLD; do [ -e $f ] && echo "EXISTS $f" || echo "absent $f"; done
