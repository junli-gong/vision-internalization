h=$(hostname -s); echo "===== $h"
pp=$(pgrep -u chenyuma -f '^/bin/bash /var/spool/slurm/d/job' | head -5); echo "slurm_script pids: $pp"
ps -u chenyuma -o pid,ppid,pgid,sid,etimes,rss,args --forest | cut -c1-220 | grep -v "ps -u\|cut -c\|bash -s\|grep -v" | head -40
B=/home/mail/chenyuma/minimax-h3-fp8
echo "--- api log tail ($h)"; ls -la --time-style=full-iso $B/logs/api-a100-1gpu-$h.log $B/logs/comfy-a100-1gpu-$h.log; tail -n 15 $B/logs/api-a100-1gpu-$h.log | cut -c1-200
echo "--- api POST counts by date"; grep -oE '^[0-9-]{10}|"(POST|GET) [^ ]+' $B/logs/api-a100-1gpu-$h.log | grep -c POST; grep -E 'POST' $B/logs/api-a100-1gpu-$h.log | tail -5 | cut -c1-200
echo "--- comfy log tail"; tail -n 8 $B/logs/comfy-a100-1gpu-$h.log | cut -c1-200; grep -c "Prompt executed" $B/logs/comfy-a100-1gpu-$h.log
echo "--- established conns on api port"; ss -tnp 2>/dev/null | egrep ':840[67]' | head
echo "--- /tmp /dev/shm writable?"; ls -ld /tmp /dev/shm; df -h /tmp | tail -1
