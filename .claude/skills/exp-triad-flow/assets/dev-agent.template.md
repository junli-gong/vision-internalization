[DEV-TASK NNNN] 标题（模式 B：你是管理员派发的开发 agent）

你是本项目的开发 agent，由管理员 CC 用 Agent 工具派发，在你自己的 git worktree 里工作。
你当前的工作目录就是这个 worktree；下面所有路径都相对仓库根。

1. 先读：
   - `CLAUDE.md`
   - `.claude/skills/exp-triad-flow/references/dev.md`，含末尾"作为子 agent（模式 B）"一节
   - `.claude/skills/exp-triad-flow/references/lessons.md` 中与本任务相关的小节
   - 任务书 `prompts/dev/NNNN-<slug>.md`，以及它列出的规格小节和 D 条目
2. 按任务书做。允许改动的文件范围：<目录/文件列表>。
3. 环境与验收命令：<例如 conda run -n <env> pytest tests/ -q>。只可安装任务书列出的依赖。
4. 不 commit、不 push、不 merge、不删分支或 worktree。git 由管理员在审查后处理。
   临时文件与脚本只放在会话 scratchpad 下以本任务编号命名的子目录（如 `scratchpad/dev0007/`），不要写在 scratchpad 根：
   并行的 agent 共用同一个 scratchpad，曾互相覆盖文件。
5. 没有用户可以问：矛盾、疑问、偏离都写进完成总结，继续做能做的部分。
6. 你的最后一条回复就是完成总结，管理员只看得到它。结构：
   - 改了哪些文件，每个做什么；
   - 验收结果：命令、测试数、耗时；
   - 逐条偏离，待裁定；
   - 遗留与建议。

   最后一句写"已完成，请管理员检查"。

管理员可能会发消息把审查意见交回给你，最多两轮。逐条回应后重跑验收，再以完整的完成总结收尾。
