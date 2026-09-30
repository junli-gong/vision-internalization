#!/usr/bin/env bash
# 服务器 CC 值守（模式 B）：等 origin 上出现未执行的新任务书；找到就把路径打印到 stdout 并退出 0。
# 用法: server_standby_wait.sh <repo_dir> <status_file> <done_file> [poll_s=120] [max_s=21600] [branch=main]
#   新任务 = origin/<branch> 上 prompts/server/ 下形如 NNNN-*.md 的文件（含 NNNN-addendum-K.md），
#            且文件名不在 <done_file> 里（每行一个文件名）；多个时按文件名排序取第一个。
#   origin/<branch> 的仓库根有 HOLD 文件时只写心跳、不接单。
#   每轮把 "<UTC时间> <状态> <说明>" 覆写进 <status_file>，管理员据此判断值守是否还活着。
#   只做 git fetch 与 ls-tree，不动工作区；拉取由 CC 拿到任务后自己做。
# 退出码: 0 找到新任务；3 等满 max_s（请重挂）；2 参数错误。
set -u
REPO="${1:?usage: server_standby_wait.sh <repo> <status_file> <done_file> [poll_s] [max_s] [branch]}"
STATUS="${2:?status file}"; DONE="${3:?done file}"
POLL="${4:-120}"; MAX="${5:-21600}"; BRANCH="${6:-main}"
git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1 || { echo "not a git repo: $REPO" >&2; exit 2; }
mkdir -p "$(dirname "$STATUS")" "$(dirname "$DONE")"
touch "$DONE"
beat() { printf '%s %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" > "$STATUS.tmp" && mv -f "$STATUS.tmp" "$STATUS"; }
start=$(date +%s)
state=waiting
while :; do
  if git -C "$REPO" fetch -q origin "$BRANCH" 2>/dev/null; then
    if git -C "$REPO" cat-file -e "origin/$BRANCH:HOLD" 2>/dev/null; then
      state=hold; beat hold "HOLD on origin/$BRANCH"
    else
      next=$(git -C "$REPO" ls-tree --name-only "origin/$BRANCH" -- prompts/server/ \
               | sed 's#^prompts/server/##' | grep -E '^[0-9]{4}-.+\.md$' | LC_ALL=C sort \
               | grep -vxF -f "$DONE" | head -n 1)
      if [ -n "$next" ]; then
        beat found "$next"
        echo "prompts/server/$next"
        exit 0
      fi
      state=waiting; beat waiting "no new task"
    fi
  else
    beat "$state" "fetch failed, retrying"
  fi
  if [ $(( $(date +%s) - start )) -ge "$MAX" ]; then beat "$state" "max wait reached, re-arm"; exit 3; fi
  sleep "$POLL"
done
