#!/bin/bash
# RUN-0012: waits for the first acceptance driver (pid $1), then (each step only if cumulative time <= 6 h):
# acceptance on the other 4 models (task book §3, triggered by a failure), then the estimate on all 8 models (§4).
P=/home/ubuntu/workspace/wzg/gjl/vision-internalization; L=$P/logs/0012; R=$P/repo
while kill -0 $1 2>/dev/null; do sleep 15; done
el=$(( $(date +%s) - $(cat $L/start_epoch.txt) ))
if [ $el -le 21600 ] && ! grep -q "6h budget exceeded" $L/driver.log; then
  echo "=== chain: acceptance extension start (elapsed ${el}s)" >> $L/driver.log
  bash $R/experiments/E1/0012_acceptance/probe_scripts/run_accept.sh qwen3.5-2b qwen3.5-4b gemma-4-e4b-it internvl3.5-4b-hf > $L/driver_accept_ext.out 2>&1
fi
touch $L/acceptance_phase_done
el=$(( $(date +%s) - $(cat $L/start_epoch.txt) ))
if [ $el -le 21600 ] && ! grep -q "6h budget exceeded" $L/driver.log; then
  echo "=== chain: estimate start (elapsed ${el}s)" >> $L/driver.log
  bash $R/experiments/E1/0012_estimate/probe_scripts/run_estimate.sh qwen3.5-0.8b qwen3.5-2b gemma-4-e2b-it internvl3.5-2b-hf qwen3.5-4b gemma-4-e4b-it internvl3.5-4b-hf qwen3.5-9b > $L/driver_estimate.out 2>&1
fi
echo "=== chain done $(date -Is)" >> $L/driver.log
