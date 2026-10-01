#!/bin/bash
# RUN-0013: acceptance on all 8 models (§3), then the estimate on all 8 models (§4); each driver checks the 8 h cap before every model.
P=/home/ubuntu/workspace/wzg/gjl/vision-internalization; L=$P/logs/0013; R=$P/repo
MODELS="qwen3.5-0.8b qwen3.5-2b gemma-4-e2b-it internvl3.5-2b-hf qwen3.5-4b gemma-4-e4b-it internvl3.5-4b-hf qwen3.5-9b"
bash $R/experiments/E1/0013_acceptance/probe_scripts/run_accept13.sh $MODELS > $L/driver_accept.out 2>&1
touch $L/acceptance_phase_done
[ -f $L/cap_reached ] || bash $R/experiments/E1/0013_estimate/probe_scripts/run_estimate13.sh $MODELS > $L/driver_estimate.out 2>&1
echo "=== chain done $(date -Is)" >> $L/driver.log
