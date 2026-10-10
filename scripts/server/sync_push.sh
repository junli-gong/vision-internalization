#!/usr/bin/env bash
# 登录节点上的结果同步循环（ITSC HPC，D89）：计算节点上的训练驱动用 TQ_NO_GIT=1，不碰 git；这里每 INTERVAL 秒把给定的结果目录提交、
# rebase、推送一次。只提交这些目录下的改动（git commit -- <路径>），检查点不在仓库里。git 操作都拿 $LOCK（与驱动、CC 共用）。
# 用法：sync_push.sh <仓库> <锁文件> <间隔秒> <说明前缀> <相对路径>…   例：sync_push.sh $VI/repo $VI/logs/git.lock 1800 "0045" experiments/E3/0045_hpc_train_06
# 一轮之后若存在 <锁文件>.stop 就退出；单轮：ONCE=1。心跳写 <锁文件 所在目录>/sync_<说明前缀>.status。
set -uo pipefail
REPO=${1:?repo}; LOCK=${2:?lock}; INTERVAL=${3:?interval}; TAG=${4:?tag}; shift 4
[ $# -ge 1 ] || { echo "至少给一个相对路径" >&2; exit 2; }
ST="$(dirname "$LOCK")/sync_${TAG}.status"
beat() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" > "$ST.tmp" && mv -f "$ST.tmp" "$ST"; }
one_round() {
  (
    if command -v flock >/dev/null 2>&1; then flock 9 || exit 1; fi  # 开发机（macOS）没有 flock，彩排时不加锁
    cd "$REPO" || exit 1
    git add -A -- "$@" || exit 1
    if git diff --cached --quiet -- "$@"; then exit 0; fi
    git commit -q -m "[results] $TAG: sync $(date -u +%Y-%m-%dT%H:%M:%SZ)" -- "$@" || exit 1
    for i in 1 2 3; do
      git pull -q --rebase --autostash origin main && git push -q origin HEAD:main && exit 0
      sleep 20
    done
    exit 1
  ) 9>"$LOCK"
}
while :; do
  if one_round "$@"; then beat "ok"; else beat "push failed (will retry)"; fi
  [ "${ONCE:-0}" = 1 ] && break
  [ -e "$LOCK.stop" ] && { beat "stopped ($LOCK.stop)"; break; }
  sleep "$INTERVAL"
done
