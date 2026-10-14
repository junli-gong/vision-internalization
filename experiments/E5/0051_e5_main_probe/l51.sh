#!/bin/bash
# RUN-0051 §3 launch on hpcgpu07 + v3 push loop 0051; then stop 0050 loop and remove wt0050
VI=/home/mail/chenyuma/visint; O=$VI/logs/0051_launch07.txt; L=$VI/logs/git.lock; V3=$VI/wt/sync3/scripts/server/sync_push.sh; W=$VI/wt/wt0051
{ echo "## squeue $(date -u +%FT%TZ) [0051 launch before]"; squeue -h -j 157296,157297 -o "%i %T %M %N"
[ $(squeue -h -j 157296,157297 -t RUNNING | wc -l) -eq 2 ] || { echo "NOT BOTH RUNNING - STOP"; exit 9; }
echo "run_0050.log last: $(tail -1 $VI/logs/run_0050.log)"; tail -1 $VI/logs/run_0050.log | grep -qi done || { echo "RUN-0050 not done - STOP"; exit 8; }
echo "07 card apps: [$(ssh -n hpcgpu07 "nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_memory --format=csv,noheader | grep 65:00")]"
echo "07 our train/eval procs: [$(ssh -n hpcgpu07 "pgrep -u chenyuma -af '[s]cripts/(train_e|eval_e|run_e)' | head -3")]"
for f in $VI/logs/train.pause.hpcgpu07 $VI/logs/HOLD $VI/logs/train.pause; do [ -e $f ] && echo "EXISTS $f" || echo "absent $f"; done
[ -e $VI/logs/e5_claims_0051 ] && { echo "claims dir exists:"; ls -la $VI/logs/e5_claims_0051; } || echo "claims dir absent"
echo "wt0051 HEAD $(git -C $W rev-parse HEAD) status_lines=$(git -C $W status --short | wc -l)"
sha256sum $W/prompts/server_hpc/0051-hpc-e5-main/queue.sh $W/scripts/server/train_queue.sh $W/configs/e5_train.json $W/configs/e5_prompts.json
a=$(sha256sum $W/prompts/server_hpc/0051-hpc-e5-main/queue.sh | cut -d' ' -f1); [ "$a" = 8f33ad28c0f7b95757c0a2555b96f3c01f4c4d840824f0a2b45e39cb9a5ab769 ] || { echo "queue.sh sha mismatch - STOP"; exit 7; }
timeout 60 ssh -n -o BatchMode=yes hpcgpu07 "bash $VI/probe/0044/launch.sh 157297 $VI/logs/train_0051-hpcgpu07.driver.log env NODE=hpcgpu07 PATH=$VI/env/bin:/usr/bin:/bin QUEUE_CAP_H=170 bash $W/prompts/server_hpc/0051-hpc-e5-main/queue.sh > /dev/null 2>&1 < /dev/null; tail -1 $VI/logs/0044_launched.tsv"
cd $VI && setsid nohup bash $V3 $VI/repo $L 1800 0051 experiments/E5/0051_e5_main_hpcgpu07 > $VI/logs/sync_0051.log 2>&1 < /dev/null & echo "sync_pid=$!" > $VI/logs/sync_0051.pid; echo "$(date -u +%FT%TZ) v3 0051 started $(cat $VI/logs/sync_0051.pid)"
sleep 60; squeue -s -j 157297 -o "%i %j %M"; echo "status: $(cat $VI/logs/train_0051-hpcgpu07.status 2>/dev/null)"; head -5 $VI/repo/experiments/E5/0051_e5_main_hpcgpu07/driver/driver.log 2>/dev/null; ls $VI/logs/e5_claims_0051 2>&1
echo "sync_0051: $(cat $VI/logs/sync_0051.status 2>/dev/null)"
echo "## squeue $(date -u +%FT%TZ) [0051 launch after]"; squeue -h -j 157296,157297 -o "%i %T %M %N"
echo "=========== stop 0050 v3 loop after a round newer than RUN-0050 done"
done_t=$(date -u -d "$(tail -1 $VI/logs/run_0050.log | cut -d' ' -f1)" +%s); P=$(cut -d= -f2 $VI/logs/sync_0050.pid)
for i in $(seq 1 150); do s=$(cat $VI/logs/sync_0050.status); st=$(date -u -d "${s%% *}" +%s 2>/dev/null || echo 0); C=$(ps -o pid=,args= --ppid $P 2>/dev/null | awk '{print $1, $2}')
  if [ $st -gt $done_t ] && [[ "$s" == *ok* ]] && [[ "$C" == *sleep* ]] && flock -n $L true; then kill $P ${C%% *}; sleep 2; echo "$(date -u +%FT%TZ) 0050 loop stopped (pid $P) after status: $s"; break; fi; sleep 30; done
ps -p $P -o pid= > /dev/null && echo "0050 loop STILL RUNNING (pid $P) - not stopped"
st=$(git -C $VI/wt/wt0050 status --short | wc -l); u="$(pgrep -u chenyuma -f "$VI/wt/wt005[0]/" | wc -l)/$(ssh -n hpcgpu07 "pgrep -u chenyuma -f '$VI/wt/wt005[0]/' | wc -l")/$(ssh -n hpcgpu06 "pgrep -u chenyuma -f '$VI/wt/wt005[0]/' | wc -l")"
echo "wt0050 status_lines=$st users=$u"; [ "$st" = 0 ] && [ "$u" = "0/0/0" ] && git -C $VI/repo worktree remove $VI/wt/wt0050 && echo "removed wt0050"
echo "quota delta KB: $(( $(timeout 60 lfs quota -u chenyuma /home | awk 'NR==3{print $2}') - 7059403936 ))"
echo "## squeue $(date -u +%FT%TZ) [0051 all after]"; squeue -h -j 157296,157297 -o "%i %T %M %N"; echo "== done"; } > $O 2>&1
