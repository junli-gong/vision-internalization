# 新项目引导（管理员在第一次对话执行）

## 0. 先谈清楚（一轮对话）

- 规格文件是什么、叫什么名（默认 `SPEC.md`；若用户已有 PILOT.md 之类，用他们的）。
- 硬约束与预注册判据是否已写死；没有就先补进规格再动工。
- 三台机器/三个对话是否就绪：开发机、算力机（ssh 别名、是否共享、工作目录限制）、GitHub 账号。

## 1. 建骨架

用 `scripts/init_project.sh <repo_dir> [spec_name]`，或手工从 `assets/` 复制：
`CLAUDE.md`（填项目一句话 + 专属约束速查）、`DECISIONS.md`、`EXPERIMENTS.md`、
`prompts/dev/`、`prompts/server/`、`reports/`、`results/`、`.gitignore`。

`CLAUDE.md` 必须含角色判定规则（三行）并指向本 skill——它是每个对话都会加载的双保险，
skill 描述再"pushy"也可能不触发。

## 2. git 与远端

- 开发机：`git init`，仓库级 `user.name/email`；首 commit 只含骨架与规格。
- GitHub：`gh repo create <name> --private --source . --push`；若 SSH key 未绑账号，
  `gh auth setup-git` 走 HTTPS 凭证助手。
- 算力机（尤其共享 root 机）：仓库级 `core.sshCommand` 绑专用 key；管理员用
  `gh repo deploy-key add <pub> --allow-write` 授权；不改任何全局文件。
- 算力机目录布局（一个工作根目录承载多项目）：
  `<root>/<project>/{repo, logs, scripts, .ssh}`，而 `<root>/{datasets, env, .torch}` 跨项目共享留在根。
  配置里的数据路径指向根下共享目录，项目迁移时一行不用改；conda env 的 shebang 写死绝对
  路径，不要搬。工具脚本用 `ROOT` / `PROJ` / `REPO` 三个常量，不硬编码拼接。

## 3. 首轮任务

- DEV-0001：与数据无关的骨架 + CPU 冒烟测试（测试用小型随机模型，不下载权重）。
- RUN-0001：环境基线 + 数据落地 + **真实 schema 报告**（不得对假设的 schema 编码）。
- 两者并行；loader 等 RUN-0001 的报告出来再写。

## 4. 记忆

在用户级记忆里记：项目名与仓库、三角色分工、算力机的路径/限制/认证方式、开发机 python
环境（哪个解释器有依赖、测试 venv 怎么建）。日期一律写绝对日期。
