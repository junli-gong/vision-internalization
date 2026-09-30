#!/bin/bash
source /home/ubuntu/workspace/wzg/gjl/vision-internalization/scripts/env.sh
cd $PROJ/tmp && (time python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('Qwen/Qwen3.5-9B', revision='c202236235'))
") > $PROJ/logs/0001/dl_9b.log 2>&1
echo "exit=$?" > $PROJ/logs/0001/dl_9b.done
