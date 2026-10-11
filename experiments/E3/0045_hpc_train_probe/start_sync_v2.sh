#!/bin/bash
# RUN-0045 addendum-1 step 5: start the v2 push loop (command as in the addendum), detached
VI=/home/mail/chenyuma/visint
rm -f $VI/logs/git.lock.stop
cd $VI || exit 1
SYNC_WT=$VI/wt/push setsid nohup bash $VI/wt/sync/scripts/server/sync_push.sh \
  $VI/repo $VI/logs/git.lock 1800 0045 \
  experiments/E3/0045_hpc_train_hpcgpu06 experiments/E3/0045_hpc_train_hpcgpu07 \
  > $VI/logs/sync_0045.log 2>&1 < /dev/null &
echo "sync_pid=$!" | tee $VI/logs/sync_0045.pid
