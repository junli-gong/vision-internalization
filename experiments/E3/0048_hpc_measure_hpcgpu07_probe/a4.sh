set -u
VI=/home/mail/chenyuma/visint
echo "## squeue $(date -u +%FT%TZ) [0048a launch before]"; squeue -h -j 157296,157297 -o "%i %T %M %N"
[ $(squeue -h -j 157296,157297 -t RUNNING | wc -l) -eq 2 ] || { echo "NOT BOTH RUNNING - STOP"; exit 9; }
timeout 60 ssh -n -o BatchMode=yes hpcgpu07 "bash $VI/probe/0044/launch.sh 157297 $VI/logs/eval_0048e1.driver.log env PATH=$VI/env/bin:/usr/bin:/bin bash $VI/wt/wt0048a/prompts/server_hpc/0048-hpc-measure-07/measure_e1.sh > /dev/null 2>&1 < /dev/null; tail -1 $VI/logs/0044_launched.tsv"
# restart 0048 sync loop: stop old one only while sleeping
P=$(cut -d= -f2 $VI/logs/sync_0048.pid); echo "old sync pid $P: $(ps -o args= -p $P | cut -c1-80)"
C=$(ps -o pid=,args= --ppid $P | awk '{print $1, $2}'); echo "child: $C"
case "$C" in *sleep*) kill $P ${C%% *}; sleep 2; ps -p $P -o pid= >/dev/null && echo "old loop still alive" || echo "old loop stopped";;
  *) echo "old loop not sleeping (child: $C) - not killed; retry later";; esac
ps -p $P -o pid= >/dev/null || bash $VI/probe/0044/start_sync_0048b.sh > /dev/null 2>&1 < /dev/null; cat $VI/logs/sync_0048.pid
# remove wt0048
st=$(git -C $VI/wt/wt0048 status --short | wc -l); users="$(pgrep -u chenyuma -f "$VI/wt/wt0048/" | wc -l)/$(ssh -n hpcgpu07 "pgrep -u chenyuma -f '$VI/wt/wt0048/' | wc -l")/$(ssh -n hpcgpu06 "pgrep -u chenyuma -f '$VI/wt/wt0048/' | wc -l")"
echo "wt0048 status_lines=$st users(mgt/07/06)=$users"; [ "$st" = 0 ] && [ "$users" = "0/0/0" ] && git -C $VI/repo worktree remove $VI/wt/wt0048 && echo "removed wt0048"
sleep 30; squeue -s -j 157297 -o "%i %j %M"; cat $VI/logs/eval_0048e1.status 2>/dev/null; cat $VI/logs/sync_0048.status
echo "## squeue $(date -u +%FT%TZ) [0048a launch after]"; squeue -h -j 157296,157297 -o "%i %T %M %N"
