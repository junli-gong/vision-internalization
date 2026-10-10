echo "--- self cgroup"; cat /proc/self/cgroup | head
for d in $(awk -F: '/memory/{print $3}' /proc/self/cgroup); do for p in /sys/fs/cgroup/memory$d /sys/fs/cgroup/memory${d%/step_*} /sys/fs/cgroup/memory${d%/step_*/*}; do [ -f $p/memory.limit_in_bytes ] && echo "$p limit=$(cat $p/memory.limit_in_bytes)"; done; done
echo "--- cpuset"; cat /sys/fs/cgroup/cpuset$(awk -F: '/cpuset/{print $3}' /proc/self/cgroup)/cpuset.cpus 2>/dev/null; taskset -pc $$
echo "--- devices allowed"; nvidia-smi -L
if [ "$(hostname -s)" = hpcgpu07 ]; then P=3102924; echo "--- pid $P"; ps -o pid,ppid,etimes,rss,args -p $P; tr '\0' '\n' < /proc/$P/environ 2>/dev/null | egrep '^(CUDA_VISIBLE_DEVICES|SLURM_STEP_ID|SLURM_JOB_ID|PWD)='; cat /proc/$P/cgroup | egrep 'devices|memory'; ls -l /proc/$P/fd 2>/dev/null | grep -c nvidia; fi
