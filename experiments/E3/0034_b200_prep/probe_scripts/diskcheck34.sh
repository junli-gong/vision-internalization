#!/bin/bash
# usage: diskcheck34.sh <label>; appends to ledger (time, label, / avail B, /dev/shm/visint B, $PROJ B, /dev/shm/visint/train_data B, MemAvailable B);
# exit 3 if / avail < 3 GB, exit 4 if MemAvailable < 60 GB
LEDGER=/home/ubuntu/workspace/wzg/gjl/vision-internalization/logs/0034/disk_ledger.tsv
avail=$(df -B1 --output=avail / | tail -1 | tr -d ' ')
shm=$(du -sb /dev/shm/visint 2>/dev/null | cut -f1)
proj=$(du -sb /home/ubuntu/workspace/wzg/gjl/vision-internalization 2>/dev/null | cut -f1)
td=$(du -sb /dev/shm/visint/train_data 2>/dev/null | cut -f1)
mem=$(awk '/MemAvailable/{print $2*1024}' /proc/meminfo)
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Is)" "$1" "$avail" "${shm:-0}" "${proj:-0}" "${td:-0}" "$mem" >> $LEDGER
awk -v l="$1" -v a="$avail" -v s="${shm:-0}" -v p="${proj:-0}" -v t="${td:-0}" -v m="$mem" 'BEGIN{printf "[disk] %s  /avail=%.3fGB  shm_visint=%.3fGB  proj=%.3fGB  train_data=%.3fGB  mem_avail=%.1fGB\n", l, a/1e9, s/1e9, p/1e9, t/1e9, m/1e9}'
if [ "$avail" -lt 3000000000 ]; then echo "!!! DISK RED LINE: / avail < 3 GB — STOP"; exit 3; fi
if [ "$mem" -lt 60000000000 ]; then echo "!!! MEMORY RED LINE: MemAvailable < 60 GB — STOP"; exit 4; fi
