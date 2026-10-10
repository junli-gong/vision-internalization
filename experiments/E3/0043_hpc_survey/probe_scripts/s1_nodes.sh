set -u
SQ() { echo "## squeue $(date -u +%FT%TZ) [$1]"; squeue -h -j 157296,157297 -o "%i %T %M %N"; }
SQ before
for J in 157296 157297; do
  echo "########## job $J"
  timeout 120 srun --jobid=$J --overlap -n1 bash -c '
    echo "host=$(hostname -s) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_STEP_ID=${SLURM_STEP_ID:-}"
    echo "--- nvidia-smi (in-allocation)"; nvidia-smi
    echo "--- query"; nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,memory.used,utilization.gpu,driver_version,compute_cap --format=csv
    echo "--- compute apps"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    echo "--- nproc=$(nproc)"; grep -c ^processor /proc/cpuinfo; lscpu | egrep "Model name|^CPU\(s\)|Socket|NUMA node\(s\)"
    echo "--- cgroup mem"; cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null; free -g
    echo "--- df"; df -h / /tmp /dev/shm /home 2>/dev/null; df -h | egrep -v "tmpfs|overlay" | head -20
    echo "--- os"; cat /etc/os-release | head -3; uname -r; ldd --version | head -1
    echo "--- uptime"; uptime
  ' < /dev/null 2>&1
  echo "rc=$?"
done
SQ after
