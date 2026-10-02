#!/bin/bash
# usage: gpu_sampler.sh <pid to follow> <out prefix>; samples nvidia-smi every 5 s while <pid> is alive.
while kill -0 $1 2>/dev/null; do
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader >> $2_gpu.csv
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | sed "s/^/$(date +%H:%M:%S), /" >> $2_apps.csv
  sleep 5
done
