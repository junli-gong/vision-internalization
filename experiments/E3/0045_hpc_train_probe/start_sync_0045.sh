#!/bin/bash
# RUN-0045 §5: start the hpcmgt push loop (command as in the task book), detached
VI=/home/mail/chenyuma/visint
cd $VI || exit 1
setsid nohup bash $VI/wt/wt0045/scripts/server/sync_push.sh \
  $VI/repo $VI/logs/git.lock 1800 0045 \
  experiments/E3/0045_hpc_train_hpcgpu06 experiments/E3/0045_hpc_train_hpcgpu07 \
  > $VI/logs/sync_0045.log 2>&1 < /dev/null &
echo "sync_pid=$!" | tee $VI/logs/sync_0045.pid
