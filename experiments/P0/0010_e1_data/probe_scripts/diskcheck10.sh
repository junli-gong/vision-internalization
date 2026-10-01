#!/bin/bash
# usage: diskcheck.sh <label>; appends to ledger, exits 3 if / avail < 3 GB
LEDGER=/home/ubuntu/workspace/wzg/gjl/vision-internalization/logs/0010/disk_ledger.tsv
avail=$(df -B1 --output=avail / | tail -1 | tr -d ' ')
shm=$(du -sb /dev/shm/visint 2>/dev/null | cut -f1)
proj=$(du -sb /home/ubuntu/workspace/wzg/gjl/vision-internalization 2>/dev/null | cut -f1)
printf "%s\t%s\t%s\t%s\t%s\n" "$(date -Is)" "$1" "$avail" "${shm:-0}" "${proj:-0}" >> $LEDGER
awk -v l="$1" -v a="$avail" -v s="${shm:-0}" -v p="${proj:-0}" 'BEGIN{printf "[disk] %s  /avail=%.3fGB  shm_visint=%.3fGB  proj=%.3fGB\n", l, a/1e9, s/1e9, p/1e9}'
if [ "$avail" -lt 3000000000 ]; then echo "!!! DISK RED LINE: / avail < 3 GB — STOP"; exit 3; fi
# RUN-0010 extra: $PROJ/data size and /dev/shm/visint/tmp
printf "        data=%s  shm_tmp=%s\n" "$(du -sh /home/ubuntu/workspace/wzg/gjl/vision-internalization/data 2>/dev/null | cut -f1)" "$(du -sh /dev/shm/visint/tmp 2>/dev/null | cut -f1)"
