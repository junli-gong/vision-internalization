#!/bin/bash
# RUN-0044 launcher (method B, D89-6): run on a compute node; starts ONE detached `srun --overlap` step inside our own job and returns.
# usage: bash launch.sh <jobid 157296|157297> <log file> <command...>; prints the srun client PID (also appended to $VI/logs/0044_launched.tsv).
set -u
J=$1 LOG=$2; shift 2
case $J in 157296|157297) ;; *) echo "refusing job $J"; exit 64 ;; esac
cd /home/mail/chenyuma/visint || exit 1
setsid nohup srun --jobid=$J --overlap -n1 "$@" > "$LOG" 2>&1 < /dev/null &
P=$!; echo "$(date -u +%FT%TZ) $(hostname -s) job=$J pid=$P log=$LOG cmd=$*" >> logs/0044_launched.tsv; echo "srun_pid=$P"
