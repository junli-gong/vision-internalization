#!/bin/bash
# RUN-0044 §4 relay: B200 /dev/shm/visint/train_data -> Mac -> lab5 -> hpcmgt -> Lustre. Read-only on B200, nothing stored on Mac/lab5.
# 3 parallel streams: refcoco_train even files, refcoco_train odd files, ade20k+nyu_normals+dedupe. Each logs start/end/rc/bytes.
D=$(cd "$(dirname "$0")" && pwd); L=$D/relay_logs; mkdir -p $L
DST=/home/mail/chenyuma/visint/data/train_data
stream() { # name, remote selection command
  local n=$1 sel=$2 t0=$(date +%s)
  echo "$(date -u +%FT%TZ) start $n" >> $L/relay.log
  ssh -o ServerAliveInterval=30 b200 "cd /dev/shm/visint/train_data && $sel" \
    | tee >(wc -c > $L/$n.bytes) \
    | ssh -o ServerAliveInterval=30 a100 "ssh -o ServerAliveInterval=30 gpu 'tar -xf - -C $DST'" > $L/$n.err 2>&1
  local rc=("${PIPESTATUS[@]}")
  echo "$(date -u +%FT%TZ) end $n rc=${rc[*]} secs=$(( $(date +%s) - t0 ))" >> $L/relay.log
}
stream refcoco_even "find refcoco_train -type f | LC_ALL=C sort | awk 'NR%2==0' | tar -cf - -T -" &
stream refcoco_odd  "find refcoco_train -type f | LC_ALL=C sort | awk 'NR%2==1' | tar -cf - -T -" &
stream ade_normals_dedupe "tar -cf - ade20k nyu_normals dedupe" &
wait
echo "$(date -u +%FT%TZ) all done" >> $L/relay.log
