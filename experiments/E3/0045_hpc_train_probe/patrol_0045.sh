#!/bin/bash
# RUN-0045 §6 patrol (read-only): squeue, driver status/process, training RSS, our card, sync loop, quota
VI=/home/mail/chenyuma/visint
echo "## patrol $(date -u +%FT%TZ)"; squeue -h -j 157296,157297 -o "%i %T %M %N"; squeue -s -j 157296,157297 -o "%i %j %M" | grep -v -E "batch|extern|\.34 "
for h in hpcgpu06 hpcgpu07; do
  echo "--- $h status: $(cat $VI/logs/train_0045-$h.status 2>/dev/null)"
  tail -3 $VI/logs/train_0045-$h.driver.log
  ssh -n -o BatchMode=yes $h "ps -u chenyuma -o pid,rss,etimes,args | egrep '[t]rain_e3b|[q]ueue.sh' | awk '{printf \"  pid=%s rss=%.1fGB et=%ss %s %s %s %s\\n\", \$1, \$2/1048576, \$3, \$5, \$6, \$7, \$8}'; echo \"  memavail=\$(awk '/MemAvailable/{printf \"%.0f\", \$2/1048576}' /proc/meminfo)GB load=\$(cut -d' ' -f1-3 /proc/loadavg)\"; nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_memory --format=csv,noheader | grep 65:00 | sed 's/^/  card: /'"
done
echo "--- sync: $(cat $VI/logs/sync_0045.status 2>/dev/null); pid alive: $(kill -0 $(cut -d= -f2 $VI/logs/sync_0045.pid) 2>/dev/null && echo yes || echo NO)"; tail -2 $VI/logs/sync_0045.log
echo "--- quota KB: $(timeout 60 lfs quota -u chenyuma /home | awk 'NR==3{print $2}') (baseline 7059403936); ckpt: $(du -sh $VI/ckpt 2>/dev/null | cut -f1)"
