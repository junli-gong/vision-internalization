#!/bin/bash
# RUN-0044 §4 relay, round 2 (after lab5 rebooted at ~07:27Z): resend only files missing or short on Lustre (list built from B200 vs HPC sizes).
D=$(cd "$(dirname "$0")" && pwd); L=$D/relay_logs; DST=/home/mail/chenyuma/visint/data/train_data
stream() { local n=$1 t0=$(date +%s); echo "$(date -u +%FT%TZ) start $n $(wc -l < $D/$n.txt) files" >> $L/relay.log
  ssh -o ServerAliveInterval=30 b200 "cd /dev/shm/visint/train_data && tar -cf - -T -" < $D/$n.txt \
    | ssh -o ServerAliveInterval=30 a100 "ssh -o ServerAliveInterval=30 gpu 'tar -xf - -C $DST'" > $L/$n.err 2>&1
  echo "$(date -u +%FT%TZ) end $n rc=${PIPESTATUS[*]} secs=$(( $(date +%s) - t0 ))" >> $L/relay.log; }
stream need_r2_a & stream need_r2_b & stream need_r2_c & wait
echo "$(date -u +%FT%TZ) round2 done" >> $L/relay.log
