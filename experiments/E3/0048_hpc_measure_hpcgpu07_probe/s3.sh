set -u
VI=/home/mail/chenyuma/visint
echo "## squeue $(date -u +%FT%TZ) [0048 launch before]"; squeue -h -j 157296,157297 -o "%i %T %M %N"
[ $(squeue -h -j 157296,157297 -t RUNNING | wc -l) -eq 2 ] || { echo "NOT BOTH RUNNING - STOP"; exit 9; }
timeout 60 ssh -n -o BatchMode=yes hpcgpu07 "bash $VI/probe/0044/launch.sh 157297 $VI/logs/eval_0048.driver.log env PATH=$VI/env/bin:/usr/bin:/bin bash $VI/wt/wt0048/prompts/server_hpc/0048-hpc-measure-07/measure.sh > /dev/null 2>&1 < /dev/null; tail -1 $VI/logs/0044_launched.tsv"
sleep 45; squeue -s -j 157297 -o "%i %j %M"; head -20 $VI/logs/eval_0048.driver.log; echo "--- status"; cat $VI/logs/eval_0048.status 2>/dev/null
echo "## squeue $(date -u +%FT%TZ) [0048 launch after]"; squeue -h -j 157296,157297 -o "%i %T %M %N"
