#!/usr/bin/env bash
# 用法: init_project.sh <repo_dir> [SPEC_FILE_NAME] [MODE=A|B]
# 在 repo_dir 下建三角色实验流程的骨架（不覆盖已存在的文件），不做 git 操作。
set -euo pipefail
DIR="${1:?repo dir}"; SPEC="${2:-SPEC.md}"; MODE="${3:-A}"
ASSETS="$(cd "$(dirname "$0")/../assets" && pwd)"
mkdir -p "$DIR"/{prompts/dev,prompts/server,reports,results,tests}
copy() { [ -e "$DIR/$2" ] && echo "skip (exists): $2" || { sed "s/{{SPEC_FILE}}/$SPEC/g" "$ASSETS/$1" > "$DIR/$2"; echo "created: $2"; }; }
copy CLAUDE.md.template CLAUDE.md
copy DECISIONS.md.template DECISIONS.md
copy EXPERIMENTS.md.template EXPERIMENTS.md
[ -e "$DIR/$SPEC" ] || { printf '# %s — 实验规格\n\n（规格正文；硬约束与预注册判据必须在实验开始前写死）\n' "$SPEC" > "$DIR/$SPEC"; echo "created: $SPEC (empty stub)"; }
[ -e "$DIR/.gitignore" ] || printf '__pycache__/\n*.py[cod]\n.venv/\n.DS_Store\ndatasets/\ncheckpoints/\n*.pt\n*.pth\n*.log\ncache/\n.claude/worktrees/\n' > "$DIR/.gitignore"
touch "$DIR/prompts/dev/.keep" "$DIR/prompts/server/.keep" "$DIR/reports/.keep" "$DIR/results/.keep"
# 服务器上没有全局 skill：把整个 skill 复制进仓库（全局版改动后记得同步）
SKILL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
[ -e "$DIR/.claude/skills/exp-triad-flow" ] || { mkdir -p "$DIR/.claude/skills" && cp -R "$SKILL_DIR" "$DIR/.claude/skills/exp-triad-flow" && echo "copied skill into .claude/skills/"; }
if [ "$MODE" = B ]; then copy server-standby.template.md prompts/server/STANDBY.md; fi
echo "done (mode $MODE). 下一步：填 CLAUDE.md 的 {{占位}}（含模式）、模式 B 还要填 prompts/server/STANDBY.md 的路径，写规格，git init + 首 commit（见 references/bootstrap.md）"
