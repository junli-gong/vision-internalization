set -u
VI=/home/mail/chenyuma/visint
echo "## squeue $(date -u +%FT%TZ) [0049 before]"; squeue -h -j 157296,157297 -o "%i %T %M %N"
t0=$(date +%s); timeout 900 flock $VI/logs/git.lock -c "git -C $VI/repo fetch -q origin main"; echo "fetch rc=$? $(( $(date +%s)-t0 ))s"
git -C $VI/repo log --oneline -1 origin/main
for c in 71e2ab3 e374d70; do git -C $VI/repo merge-base --is-ancestor $c origin/main && echo "$c in origin/main: yes" || echo "$c NOT in origin/main"; done
[ -e $VI/wt/wt0049 ] || git -C $VI/repo worktree add -q --detach $VI/wt/wt0049 origin/main
echo "wt0049 HEAD $(git -C $VI/wt/wt0049 rev-parse HEAD) status_lines=$(git -C $VI/wt/wt0049 status --short | wc -l)"
ls $VI/wt/wt0049/scripts/diag/merge_split_probe.py $VI/wt/wt0049/scripts/diag/merge_split_probe.dev.json
