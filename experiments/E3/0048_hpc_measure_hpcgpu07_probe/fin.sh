set -u
VI=/home/mail/chenyuma/visint
echo "## squeue $(date -u +%FT%TZ) [0048 finish before]"; squeue -h -j 157296,157297 -o "%i %T %M %N"
echo "tmp0048: $(ls -d $VI/ckpt/hpcgpu07/tmp0048 2>&1 | tail -1)"
echo "sync_0048 status: $(cat $VI/logs/sync_0048.status)"
# all results already in the push worktree?
for d in experiments/E3/0048_hpc_measure_hpcgpu07 experiments/E5/0048_hpc_measure_hpcgpu07 experiments/E3/0048_hpc_measure_e1_hpcgpu07; do
  n=$(rsync -rcn --exclude='*.cur' --out-format='%n' $VI/repo/$d/ $VI/wt/push/$d/ | grep -v '/$' | wc -l); echo "$d: files differing from push worktree: $n"; done
git -C $VI/wt/push status --short | wc -l; git -C $VI/wt/push log --oneline -1
P=$(cut -d= -f2 $VI/logs/sync_0048.pid); C=$(ps -o pid=,args= --ppid $P | awk '{print $1, $2}'); echo "sync_0048 pid $P child: $C"
case "$C" in *sleep*) kill $P ${C%% *}; sleep 2; ps -p $P -o pid= >/dev/null && echo "still alive" || echo "0048 sync loop stopped";; *) echo "not sleeping - not killed";; esac
echo "$(date -u +%FT%TZ) stopped by server CC after final push (RUN-0048 done)" > $VI/logs/sync_0048.status
echo "sync_0045 alive: $(kill -0 $(cut -d= -f2 $VI/logs/sync_0045.pid) 2>/dev/null && echo yes || echo NO); $(cat $VI/logs/sync_0045.status)"
st=$(git -C $VI/wt/wt0048a status --short | wc -l); u="$(pgrep -u chenyuma -f "$VI/wt/wt0048[a]/" | wc -l)/$(ssh -n hpcgpu07 "pgrep -u chenyuma -f '$VI/wt/wt0048[a]/' | wc -l")/$(ssh -n hpcgpu06 "pgrep -u chenyuma -f '$VI/wt/wt0048[a]/' | wc -l")"
echo "wt0048a status_lines=$st users=$u"; [ "$st" = 0 ] && [ "$u" = "0/0/0" ] && git -C $VI/repo worktree remove $VI/wt/wt0048a && echo "removed wt0048a"
echo "quota delta KB: $(( $(timeout 60 lfs quota -u chenyuma /home | awk 'NR==3{print $2}') - 7059403936 ))"
echo "## squeue $(date -u +%FT%TZ) [0048 finish after]"; squeue -h -j 157296,157297 -o "%i %T %M %N"
