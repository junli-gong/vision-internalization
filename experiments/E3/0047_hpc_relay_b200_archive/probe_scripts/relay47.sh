#!/bin/bash
# RUN-0047 relay: dev machine archive -> lab5 -> hpcmgt -> Lustre $VI/b200 (read-only on source; nothing stored on lab5). COPYFILE_DISABLE=1.
D=$(cd "$(dirname "$0")" && pwd); SRC=$HOME/Research/vision-internalization-archive/b200_20261011; DST=/home/mail/chenyuma/visint/b200; L=$D/logs
export COPYFILE_DISABLE=1
stream() { local n=$1 t0=$(date +%s); echo "$(date -u +%FT%TZ) start $n $(wc -l < $D/$n.txt) files" >> $L/relay.log
  ( cd "$SRC" && tar -cf - -T "$D/$n.txt" ) | tee >(wc -c > $L/$n.bytes) \
    | ssh -o ServerAliveInterval=30 a100 "ssh -o ServerAliveInterval=30 gpu 'tar -xf - -C $DST'" > $L/$n.err 2>&1
  echo "$(date -u +%FT%TZ) end $n rc=${PIPESTATUS[*]} secs=$(( $(date +%s) - t0 ))" >> $L/relay.log; }
for n in "$@"; do stream $n & done; wait
echo "$(date -u +%FT%TZ) done $*" >> $L/relay.log
