# RUN-0001 env; source me. One assignment per line on purpose:
# `export A=x B=$A` expands $A before assigning, giving B=/... on the root disk.
export ROOT=/home/ubuntu/workspace/wzg/gjl
export PROJ=$ROOT/vision-internalization
export REPO=$PROJ/repo
export HF_HOME=$ROOT/.hf
export HF_HUB_CACHE=/dev/shm/visint/hf-hub
export TMPDIR=$PROJ/tmp
export PIP_NO_CACHE_DIR=1
export PIP_CACHE_DIR=$PROJ/tmp/pip-nocache
export XDG_CACHE_HOME=$PROJ/tmp/xdg-cache
export TRITON_CACHE_DIR=$PROJ/tmp/triton-cache
export TORCHINDUCTOR_CACHE_DIR=$PROJ/tmp/inductor-cache
export HF_DATASETS_CACHE=$PROJ/tmp/hf-datasets
export HF_HUB_DISABLE_TELEMETRY=1
case ":$PATH:" in *":$PROJ/venv/bin:"*) ;; *) [ -d $PROJ/venv/bin ] && export PATH=$PROJ/venv/bin:$PATH ;; esac
# hf-xet chunk cache + logs default to $HF_HOME/xet (root disk); keep them on the RAM disk, no chunk cache
export HF_XET_CACHE=/dev/shm/visint/xet
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0
