---
name: exp-triad-flow
description: 三角色（管理员 CC / 开发 CC / 服务器 CC）科研实验项目流程——用户在三个 Claude Code 对话之间手动转交任务书，管理员规划与审查、开发写代码不提交、服务器拉取运行回传结果。凡是在一个采用此流程的实验仓库里工作，或用户提到：新实验项目起步、初始化实验仓库、三角色、开发CC、服务器CC、写任务书/prompt、审查合入、跑实验、结果回传、跑完了、开发完成、更新台账、裁定、决策记录、Layer 1/下一阶段规划，都必须先用这个 skill 判定自己的角色再动手。收到以 [DEV-TASK 或 [RUN-TASK 开头的消息时同样必须使用。Also triggers on: three-role experiment workflow, dev task, run task, experiment ledger, decision log, bootstrap a new research repo.
---

# 三角色实验流程（通用版）

一个实验仓库由三个 Claude Code 实例协作，用户手动在三者之间转交任务书。
角色不同，职责和禁区完全不同，所以**第一步永远是判定自己是谁**。

## 角色判定

看本对话的**第一条用户消息**：

| 开头标记 | 角色 | 所在机器 | 详细规则 |
|---|---|---|---|
| `[DEV-TASK NNNN]` | 开发 CC | 开发机 | `references/dev.md` |
| `[RUN-TASK NNNN]` | 服务器 CC | 算力服务器 | `references/server.md` |
| 其他 | 管理员 CC（默认） | 开发机 | `references/manager.md` |

判定后读对应 reference 再行动。标记与内容明显矛盾时先问用户，不要猜。
新项目尚未初始化时（仓库里没有 `DECISIONS.md`/`EXPERIMENTS.md`），管理员先读
`references/bootstrap.md` 建骨架。

## 仓库里的固定文件（所有角色通用）

| 文件 | 作用 | 谁写 |
|---|---|---|
| 规格文件（`SPEC.md` 或项目自定名，在 CLAUDE.md 里指明） | 实验规格，single source of truth，含硬约束与预注册判据 | 用户/管理员，实验开始后不改 |
| `DECISIONS.md` | 编号裁定 D1、D2…：规格留白处的裁决，**只追加不插入、不改旧条** | 管理员 |
| `EXPERIMENTS.md` | 台账：RUN/DEV 记录（倒序）+ 里程碑 | 管理员 |
| `CLAUDE.md` | 项目一句话 + 角色判定规则 + 项目专属约束速查 | 管理员（模板见 assets） |
| `prompts/dev/`、`prompts/server/` | 每份任务书存档（NNNN-slug.md；补充用 NNNN-addendum-K.md） | 管理员 |
| `reports/NNNN-*.md` | 服务器每次运行的报告，只贴数字不解读 | 服务器 |
| `results/<run>/` + 顶层汇总 | 实验产出；顶层汇总由管理员在每次拉取结果后重新生成 | 服务器 / 管理员 |
| `FINDINGS.md` | 结论文档，结论性文字由管理员写，脚本只供数 | 管理员 |

规格冲突时优先级：用户当场指示 > DECISIONS.md > 规格文件。发现冲突本身要报告。

## 主循环（管理员视角）

```
规划 → 写 dev 任务书(存档) → 用户转交开发CC → 开发改代码不提交 →
用户说"开发完成" → 管理员审查(清单) → 逐文件 add + commit + push →
写 run 任务书(存档, 含 hash) → 用户转交服务器CC → 服务器拉取校验 hash → 跑 →
results + report 推回 → 管理员 pull → 重建顶层汇总 → 解读 → 台账/裁定 → 下一步
```

三条铁律，任何角色不得违反：

1. **实验逻辑只能由开发 CC 写、管理员审查后合入**。服务器绝不改实验逻辑；管理员只做
   裁定范围内的常量级修改并在 commit message 注明。
2. **仪器先于效应**：null 臂/对照配置先跑，其结果必须**精确**为零而非约等于零；非零
   是 bug 不是发现，修好前不报告任何效应。
3. **判据与预测先于数据**：go/no-go 判据写进规格；每个诊断性运行发出前，管理员把预期
   结果写进台账或任务书，事后对照。

## 来自实战的规则

`references/lessons.md` 收录了一次五天 pilot 中每一条由事故转化的规则（共享工作区的
git 纪律、约定单一来源、台账编辑防静默失败、hash 变基同步、步数混淆……）。管理员在
项目启动时通读一遍；其他角色读与自己相关的小节。

## 何时升级流程

- 同一类事故第二次出现 → 立即把规则写进项目 CLAUDE.md（项目级）或本 skill 的
  lessons.md（通用级），不要只在对话里提醒。
- 规格与现实冲突 → 新增 D 条目裁定，不静默简化（"若配置无法在合理时间内实现，说出来并
  提议最便宜的替代"）。
