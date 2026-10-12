#!/bin/bash
# RUN-0048 §3.5: separate v2 push loop (TAG 0048, 600 s), shares $VI/wt/push and git.lock with RUN-0045's loop
VI=/home/mail/chenyuma/visint
cd $VI || exit 1
SYNC_WT=$VI/wt/push setsid nohup bash $VI/wt/sync/scripts/server/sync_push.sh \
  $VI/repo $VI/logs/git.lock 600 0048 \
  experiments/E3/0048_hpc_measure_hpcgpu07 experiments/E5/0048_hpc_measure_hpcgpu07 \
  > $VI/logs/sync_0048.log 2>&1 < /dev/null &
echo "sync_pid=$!" | tee $VI/logs/sync_0048.pid
