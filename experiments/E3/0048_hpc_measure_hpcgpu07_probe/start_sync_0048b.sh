#!/bin/bash
# RUN-0048 addendum-1 step 5: restart the 0048 v2 push loop with the extra path
VI=/home/mail/chenyuma/visint
cd $VI || exit 1
SYNC_WT=$VI/wt/push setsid nohup bash $VI/wt/sync/scripts/server/sync_push.sh \
  $VI/repo $VI/logs/git.lock 600 0048 \
  experiments/E3/0048_hpc_measure_hpcgpu07 experiments/E5/0048_hpc_measure_hpcgpu07 experiments/E3/0048_hpc_measure_e1_hpcgpu07 \
  >> $VI/logs/sync_0048.log 2>&1 < /dev/null &
echo "sync_pid=$!" | tee $VI/logs/sync_0048.pid
