#!/bin/bash
# RUN-0001 §3.4: vLLM nightly / SGLang main dry-run, no install. TMPDIR on /dev/shm so resolver downloads never hit /.
source /home/ubuntu/workspace/wzg/gjl/vision-internalization/scripts/env.sh
L=$PROJ/logs/0001
which=$1
if [ "$which" = vllm ]; then
  export TMPDIR=/dev/shm/visint/tmp-pip-vllm
  cd $TMPDIR && (time pip install --dry-run --no-cache-dir --report $L/pip_dryrun_vllm_nightly.json vllm --pre --extra-index-url https://wheels.vllm.ai/nightly) > $L/pip_dryrun_vllm_nightly.log 2>&1
else
  export TMPDIR=/dev/shm/visint/tmp-pip-sgl
  cd $TMPDIR && (time pip install --dry-run --no-cache-dir --report $L/pip_dryrun_sglang_main.json 'sglang[all] @ git+https://github.com/sgl-project/sglang.git#subdirectory=python') > $L/pip_dryrun_sglang_main.log 2>&1
fi
echo "exit=$?" >> $L/pip_dryrun_${which}.done
