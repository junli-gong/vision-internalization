#!/usr/bin/env bash
# 用法: init_project.sh <repo_dir> [SPEC_FILE_NAME]
# 在 repo_dir 下建三角色实验流程的骨架（不覆盖已存在的文件），不做 git 操作。
set -euo pipefail
DIR="${1:?repo dir}"; SPEC="${2:-SPEC.md}"
ASSETS="$(cd "$(dirname "$0")/../assets" && pwd)"
mkdir -p "$DIR"/{prompts/dev,prompts/server,reports,results,tests}
copy() { [ -e "$DIR/$2" ] && echo "skip (exists): $2" || { sed "s/{{SPEC_FILE}}/$SPEC/g" "$ASSETS/$1" > "$DIR/$2"; echo "created: $2"; }; }
copy CLAUDE.md.template CLAUDE.md
copy DECISIONS.md.template DECISIONS.md
copy EXPERIMENTS.md.template EXPERIMENTS.md
[ -e "$DIR/$SPEC" ] || { printf '# %s — 实验规格\n\n（规格正文；硬约束与预注册判据必须在实验开始前写死）\n' "$SPEC" > "$DIR/$SPEC"; echo "created: $SPEC (empty stub)"; }
[ -e "$DIR/.gitignore" ] || printf '__pycache__/\n*.py[cod]\n.venv/\n.DS_Store\ndatasets/\ncheckpoints/\n*.pt\n*.pth\n*.log\ncache/\n' > "$DIR/.gitignore"
touch "$DIR/prompts/dev/.keep" "$DIR/prompts/server/.keep" "$DIR/reports/.keep" "$DIR/results/.keep"
echo "done. 下一步：填 CLAUDE.md 的 {{占位}}，写规格，git init + 首 commit（见 references/bootstrap.md）"
