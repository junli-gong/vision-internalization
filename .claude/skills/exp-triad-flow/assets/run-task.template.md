[RUN-TASK NNNN] 标题

## 拉取
git pull 后校验 `git log --oneline` 含 `<hash>`（或 HEAD ≥ <hash>），不符停下报告。
`pytest tests/ -q` 应 <N> passed。

## 环境
依赖/路径变更（无变更就写"无变更"）；预授权的运行时改动及其条件（数值不变需有实证，改后重跑测试）。

## 运行
精确到每条命令（含工作目录）；仪器配置最先；依赖顺序写明；长任务 nohup + 状态文件，
每个配置跑完单独 push。异常处理：<null 臂非零 / 断言失败> 即停止并推证据。

## 产出
- results/<run>/…（进 git 的清单）
- reports/NNNN-*.md：贴哪些表（原样、不解读）、记录哪些资源数据、后台快照。

## 回传
`git add <paths> && git commit -m "[results] NNNN: 摘要" && git push`
环境问题可修并记录；实验代码 bug 一律停止+报告。
