#!/bin/bash
# RUN-0049 (06) -> RUN-0052 launch on hpcgpu06 + v3 sync 0052 -> stop 0045 loop, remove wt0045, wt0049
VI=/home/mail/chenyuma/visint; O=$VI/logs/0052_launch06.txt; L=$VI/logs/git.lock; V3=$VI/wt/sync3/scripts/server/sync_push.sh; W=$VI/wt/wt0052
SQ() { echo "## squeue $(date -u +%FT%TZ) [$1]"; squeue -h -j 157296,157297 -o "%i %T %M %N"; [ $(squeue -h -j 157296,157297 -t RUNNING | wc -l) -eq 2 ] || { echo "NOT BOTH RUNNING - STOP"; exit 9; }; }
{ SQ "06 before"
echo "0045-06: $(cat $VI/logs/train_0045-hpcgpu06.status); done file: $(ls $VI/logs/train_0045-hpcgpu06.done)"
echo "06 card apps: [$(ssh -n hpcgpu06 "nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv,noheader | grep 65:00")]"
echo "=========== RUN-0049 on hpcgpu06"
LG=$VI/logs/0049_$(date -u +%H%M)_06.log; t0=$(date +%s)
timeout 1800 ssh -n hpcgpu06 "srun --jobid=157296 --overlap -n1 bash $VI/probe/0049/r49_node.sh < /dev/null" > $LG 2>&1; echo "0049-06 rc=$? $(( $(date +%s)-t0 ))s log=$LG"
grep -E "diag_rc|pytest_rc|verdict|passed|failed|split_run0044\.across_merge\.(n_disagreements|max_abs_logit)|^split\.across_merge\.(n_disagreements|max_abs_logit)|control\.check\.(exit_code|across_merge\.n_disagreements)" $LG
SQ "after 0049-06"
echo "=========== RUN-0052 checks"
for v in c3 e3b_c3; do for s in 0 1 2; do d=$VI/ckpt/hpcgpu06/0045/$v/qwen3.5-4b/seed$s/ckpt
  r=$(cd $d/final && find . -type f | LC_ALL=C sort | while IFS= read -r f; do sha256sum "$f"; done | diff - $d/final.sha256.txt > /dev/null && echo OK || echo MISMATCH); echo "$v seed$s: $(wc -l < $d/final.sha256.txt) files $r"; [ $r = OK ] || { echo "ckpt mismatch - STOP"; exit 7; }; done; done
for f in $VI/logs/train.pause.hpcgpu06 $VI/logs/HOLD $VI/logs/train.pause; do [ -e $f ] && { echo "EXISTS $f - STOP"; exit 6; } || echo "absent $f"; done
echo "wt0052 HEAD $(git -C $W rev-parse HEAD) status_lines=$(git -C $W status --short | wc -l)"
chk() { a=$(sha256sum "$1" | cut -d' ' -f1); [ "$a" = "$2" ] && echo "OK   $(basename $1)" || { echo "DIFF $1 - STOP"; exit 5; }; }
chk $W/prompts/server_hpc/0052-hpc-06-4b/run.sh a4ae09a0117790047babd15ba5fd21e6a6368b58c4f678d4452c12a661919ffc
chk $W/prompts/server_hpc/0052-hpc-06-4b/eval.sh 51b0db5a39854d731831dafb4c5e40b6768c52fea0f365082e2202c055ff37bc
chk $W/prompts/server_hpc/0052-hpc-06-4b/queue.sh 5daa6013b7b97022e8fcb144a2d68a02a6deb1fe2936f7aaf91a0f62a7e3e8b4
chk $W/scripts/server/eval_lib.sh 2380464beeabe300f2cfdd17ab1d79921b8d978b454b2d9f16ee42e4ca074a59
chk $W/scripts/server/train_queue.sh fcb10f47c2fb6cf068b29f849230c82611bbcc018eb550116867b17337e0dbaa
SQ "0052 launch before"
timeout 60 ssh -n -o BatchMode=yes hpcgpu06 "bash $VI/probe/0044/launch.sh 157296 $VI/logs/run_0052.driver.log env PATH=$VI/env/bin:/usr/bin:/bin EV_CAP_A_H=80 QUEUE_CAP_H=46 EST_E3B_H=13.6 EV_CAP_B_H=20 bash $W/prompts/server_hpc/0052-hpc-06-4b/run.sh > /dev/null 2>&1 < /dev/null; tail -1 $VI/logs/0044_launched.tsv"
cd $VI
setsid nohup bash $V3 $VI/repo $L 1800 0052 experiments/E3/0052_hpc_eval_4b_hpcgpu06 experiments/E3/0052_hpc_e3b_hpcgpu06 > $VI/logs/sync_0052.log 2>&1 < /dev/null &
echo "sync_pid=$!" > $VI/logs/sync_0052.pid; echo "v3 0052 started $(cat $VI/logs/sync_0052.pid) ($(ps -o args= -p $! | cut -c1-60))"
sleep 60; squeue -s -j 157296 -o "%i %j %M"; echo "run_0052.log: $(cat $VI/logs/run_0052.log 2>/dev/null)"; echo "eval a: $(cat $VI/logs/eval_0052-a.status 2>/dev/null)"; echo "sync_0052: $(cat $VI/logs/sync_0052.status 2>/dev/null)"
SQ "0052 launch after"
echo "=========== stop 0045 loop after a round newer than 06 done"
dt=$(date -u -d "$(cut -d' ' -f1 $VI/logs/train_0045-hpcgpu06.status)" +%s); P=$(cut -d= -f2 $VI/logs/sync_0045.pid); echo "0045 loop pid $P: $(ps -o args= -p $P | cut -c1-70)"
for i in $(seq 1 150); do s=$(cat $VI/logs/sync_0045.status); st=$(date -u -d "${s%% *}" +%s 2>/dev/null || echo 0); C=$(ps -o pid=,args= --ppid $P 2>/dev/null | awk '{print $1, $2}')
  if [ $st -gt $dt ] && [[ "$s" == *ok* ]] && [[ "$C" == *sleep* ]] && flock -n $L true; then kill $P ${C%% *}; sleep 2; echo "$(date -u +%FT%TZ) 0045 loop stopped (pid $P) after status: $s"; echo "$(date -u +%FT%TZ) stopped by server CC after final push (RUN-0045 done)" > $VI/logs/sync_0045.status; break; fi; sleep 30; done
ps -p $P -o pid= > /dev/null && echo "0045 loop STILL RUNNING (pid $P)"
for w in wt0045 wt0049; do st=$(git -C $VI/wt/$w status --short | wc -l); pat="$VI/wt/${w:0:5}[${w:5:1}]${w:6}/"
  u="$(pgrep -u chenyuma -f "$pat" | wc -l)/$(ssh -n hpcgpu06 "pgrep -u chenyuma -f '$pat' | wc -l")/$(ssh -n hpcgpu07 "pgrep -u chenyuma -f '$pat' | wc -l")"
  echo "$w status_lines=$st users=$u"; [ "$st" = 0 ] && [ "$u" = "0/0/0" ] && git -C $VI/repo worktree remove $VI/wt/$w && echo "removed $w"; done
git -C $VI/repo worktree list
echo "quota delta KB: $(( $(timeout 60 lfs quota -u chenyuma /home | awk 'NR==3{print $2}') - 7059403936 ))"
SQ "all after"; echo "== done"; } > $O 2>&1
