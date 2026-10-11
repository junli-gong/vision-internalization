#!/usr/bin/env bash
# 登录节点上的结果同步循环（ITSC HPC，D89；v2，RUN-0045 的事故之后）：计算节点上的驱动用 TQ_NO_GIT=1，结果写进<源仓库>的工作区，
# 这里每 INTERVAL 秒把给定的结果目录提交并推送一次。
# v1 的事故（RUN-0045，10-11）：v1 在源仓库里 commit，再 pull --rebase --autostash。autostash 把训练期间改动过的已跟踪文件暂存后写回成
#   新文件（新 inode）；驱动整段开着写的 driver/logs/<运行>.cur 被换掉，段结束时读到截断的内容，正常停下的段被判成"没有完成也没有按段停下"。
# v2 不碰源仓库的工作区与索引（只 fetch）：每轮把这些目录 rsync 到一个专用的推送工作树（SYNC_WT，detached，只有本脚本写），
#   在那里提交、推送；推送被拒就 fetch 后 rebase 再推。*.cur（驱动的单段输出，段结束就删）不同步，推送工作树里已有的一并删掉。
#   源仓库的工作区从此落后于 origin：不要在源仓库里 pull 或 stash；新任务的工作树用 git fetch 后 worktree add --detach origin/main 建。
# 用法：SYNC_WT=<推送工作树> sync_push.sh <源仓库> <锁文件> <间隔秒> <说明前缀> <相对路径>…
#   例：SYNC_WT=$VI/wt/push sync_push.sh $VI/repo $VI/logs/git.lock 1800 0045 experiments/E3/0045_hpc_train_hpcgpu06 …
#   推送工作树不存在就自动建（git -C <源仓库> worktree add --detach <SYNC_WT> origin/main）。git 操作都拿 $LOCK（与 CC 共用）。
# 一轮之后若存在 <锁文件>.stop 就退出；单轮：ONCE=1。心跳写 <锁文件所在目录>/sync_<说明前缀>.status。
set -uo pipefail
SRC=${1:?源仓库}; LOCK=${2:?锁文件}; INTERVAL=${3:?间隔秒}; TAG=${4:?说明前缀}; shift 4
[ $# -ge 1 ] || { echo "至少给一个相对路径" >&2; exit 2; }
: "${SYNC_WT:?SYNC_WT=<推送工作树>（v2 不在源仓库里提交，见文件头）}"
case "$SYNC_WT" in "$SRC"|"$SRC"/*) echo "SYNC_WT 不能是源仓库或在它里面：$SYNC_WT" >&2; exit 2;; esac
ST="$(dirname "$LOCK")/sync_${TAG}.status"
beat() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" > "$ST.tmp" && mv -f "$ST.tmp" "$ST"; }
one_round() {
  (
    if command -v flock >/dev/null 2>&1; then flock 9 || exit 1; fi  # 开发机（macOS）没有 flock，彩排时不加锁
    git -C "$SRC" fetch -q origin main || exit 1
    if [ ! -e "$SYNC_WT/.git" ]; then git -C "$SRC" worktree add -q --detach "$SYNC_WT" origin/main || exit 1; fi
    cd "$SYNC_WT" || exit 1
    git reset -q --hard origin/main || exit 1  # 推送工作树只有本脚本写：上一轮没推上去的提交丢掉也无妨，这一轮从源目录重新同步
    local p paths=()
    for p in "$@"; do
      [ -d "$SRC/$p" ] || continue
      mkdir -p "$p"
      rsync -a --delete --delete-excluded --exclude '*.cur' "$SRC/$p/" "$p/" || exit 1
      paths+=("$p")
    done
    [ ${#paths[@]} -gt 0 ] || exit 0
    git add -A -- "${paths[@]}" || exit 1
    if git diff --cached --quiet -- "${paths[@]}"; then exit 0; fi
    git commit -q -m "[results] $TAG: sync $(date -u +%Y-%m-%dT%H:%M:%SZ)" -- "${paths[@]}" || exit 1
    for i in 1 2 3; do
      git push -q origin HEAD:main && exit 0
      sleep 20
      git fetch -q origin main || continue
      git rebase -q origin/main || { git rebase --abort; exit 1; }  # 这些目录只有本脚本提交，rebase 不会冲突；冲突了就停下等人看
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
