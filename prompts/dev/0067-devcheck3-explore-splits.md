[DEV-TASK 0067] 密封的 devcheck3、探索用的 explore_val，以及探索候选的判定脚本（D126-4、D126-5、D128）

## 背景

- 负责人授权自主探索（D126）。探索出来的方法要进论文主结果，只有一条通道：在**密封的 devcheck3** 上与 W2-c 配对判定，最多 2 个候选，11-02 判定。
- devcheck3 要"现在就密封"：清单按哈希固定，写进预注册；判定之前，任何人不看任何系统在它上面的结果。
- 探索另需一份从 val 取的句子 explore_val，与 devcheck3 按图不相交。
- 判定规则也要在看到任何数据之前写成代码，与预注册一起固定。

## 先读

- `CLAUDE.md`、`.claude/skills/exp-triad-flow/references/dev.md`（含"作为子 agent"一节）
- DECISIONS.md 的 D126（第 4、5 条）、D128
- `src/w2/data.py`：devcheck2 怎样构造、开发机上的骨架与服务器上生成的流程、`needs_frozen`、`w2_items`、核对；本任务照它的样子写。
- `src/data/subsets.py`（`hash_rank`、`population_digest`）、`src/data/loaders.load_refcoco`
- `configs/w2_splits.json`、`configs/w1_splits.json`、`configs/e1_eval_subsets.json`
- `scripts/run_w2.py`（`--split` 怎样取样本）、`scripts/w2_gate.py`（配对 bootstrap、ratio-of-sums）
- `scripts/health_check.py`（标红规则与 `--compare` 的输出）

## 任务

1. **`src/data/explore_splits.py`**，清单写 `configs/explore_splits.json`。`configs/w2_splits.json` 不动，它的 sha256 是 W2 冻结配置身份的一部分。
   - **总体**：E1 的三个 val 评测总体 `refcoco_unc_val`、`refcoco+_unc_val`、`refcocog_umd_val`，即 `load_refcoco(根, 数据集, 'val', subset='eval')`。条数与 sha256 必须等于 `configs/e1_eval_subsets.json` 的 populations。样本 id 带数据集前缀才唯一，与 W2-c 训练池的约定相同。
   - **排除**（D126-5，D128 修正）：
     - 按图（COCO 图像 id）排除 W1、W2 用过的全部划分，即 devtune800 与 devcheck2 所在的图。refcoco 与 refcoco+ 共用图，所以按图排除跨数据集生效。
     - 按句排除 `configs/e1_eval_subsets.json` 的 subsets 里落在这三个总体上的全部句子：直接引用这三个总体的子集，以及经 `think:refexp_poly` 这类链间接引用的。
   - **devcheck3**：分三层：
     - refcoco 200 句、refcoco+ 200 句、refcocog 100 句；
     - 每层按 `hash_rank(salt='devcheck3_v1', purpose='devcheck3')` 升序取前 n 句；
     - 不够 n 句就报错。
   - **explore_val**：同样的总体与排除，再按图排除 devcheck3 的全部图。
     - 分三层：400、400、200 句，`salt='explore_val_v1'`、`purpose='explore_val'`。
     - 不够就取全部，在清单里记实数。
   - **划分的形式**：每层一个划分，例如 `devcheck3_refcoco_unc_val`、`devcheck3_refcoco+_unc_val`、`devcheck3_refcocog_umd_val`，explore_val 同理。每个划分单数据集，与 W2 的 test 划分同一种条目格式，`w2_items` 能直接取样本。
     - 另给两个组名：`devcheck3` = 三个 devcheck3 划分，`explore_val` = 三个 explore_val 划分。
     - 判定与报告都按组合并：500 句一起算 ratio-of-sums。
   - **清单内容**：
     - 每个划分的 ids（自然顺序）、`image_of`、n 与 sha256；
     - 构造的每一步：总体、排除、剩余的句数与图数，以及各自的 sha256；
     - 组的 sha256。
   - **核对**（`--check`，不要数据也能核对的部分照 devcheck2）：
     - devcheck3 与 explore_val 按图不相交；
     - 两者与 devtune800、devcheck2 按图不相交，与 5 个 test 划分按图不相交；
     - 与 E1 评测子集按句不相交；
     - 与 W2-c 训练池按图不相交：训练池的加载器已按图剔除 `refcoco/eval`，即三个数据集 val/test 共用的图，核对时引用同一份剔除清单。有数据时核对，并记数。
   - **生成**：开发机上没有 RefCOCO 的数据。开发机写骨架，划分为待生成（status = pending）；服务器用 `python -m src.data.explore_splits --build --data-root $PROJ/data --out <路径>` 在 CPU 上生成，交回后由管理员放进 configs。与 devcheck2 的流程相同。生成命令与预计用时写进模块说明。
2. **密封守卫**：
   - 一个公共函数，例如 `require_unsealed(split, unseal)`：
     - 划分属于 devcheck3（任何一层或组名）而 `unseal` 不是 `G-EX` 时拒绝，报错说明 D126；
     - 放行时把门名与时间记进运行身份。
   - `scripts/run_w2.py` 加 `--splits-file`，默认 `configs/w2_splits.json`，默认时行为与输出逐字节不变，测试钉住；再加 `--unseal-devcheck3 <门名>`。
   - devcheck3 只能用冻结的配置跑，同 `needs_frozen`。
   - 以后的探索运行器都调同一个守卫。在模块说明里写清：新运行器取 devcheck3 必须经过它。
   - 生成清单本身只写 id，不跑任何系统，不受守卫限制。
3. **判定脚本 `scripts/gate_explore.py`**（判定在 11-02，脚本现在写好、测好，与预注册一起固定）：
   - 输入：
     - 参照 W2-c 与候选在 devcheck3 三个划分上的结果目录，均为 W2 格式的 `predictions.jsonl`，每句的交与并。
     - 可选：候选与基座同机配对的健康检查结果，即 `health_check.json` 或 `--compare` 的输出。
   - 先核对：两边的句子集合都恰是 devcheck3 的 500 句，id 与清单的 sha256 一致；任何一边缺句就拒绝判定。
   - 读数：500 句合并的 cIoU，即 ratio-of-sums，两边各一个；差值 = 候选 − W2-c。
   - CI：配对 bootstrap，按句重抽 10,000 次，97.5% 的双侧 CI。种子固定。与 `scripts/w2_gate.py` 同一套函数，复用，不复制。
   - 规则（D126-4），满足其一即通过：
     - (i) 差值 CI 下界 > 0；
     - (ii) 差值 CI 下界 > −0.02，并且健康检查没有任何一项标红。标红口径与 `scripts/health_check.py` 相同，复用它的判定函数。没给健康检查时 (ii) 记为"无法判定"，不算通过。
   - 输出 `gate_explore.json` 与一段 markdown：两边的 cIoU、差值与 CI、(i) 与 (ii) 各自的结果、结论；另附 gIoU 作诊断，不入规则。
   - 最多 2 个候选：脚本一次判一个候选，多个就分别调用；不做多重比较校正，负责人的规则里没有。
4. **测试**（`HF_HUB_OFFLINE=1 conda run -n visint pytest tests/test_explore_splits.py tests/test_gate_explore.py tests/test_w2_*.py -q`）：
   - 构造规则在 fixtures 上：分层、排除、不相交、哈希顺序；不够时报错（devcheck3）与取全部（explore_val）；
   - 骨架与 `--check`；
   - 守卫：没有 `--unseal-devcheck3` 或门名不对时拒绝；放行时运行身份里有记录；
   - `run_w2.py` 默认时逐字节不变；
   - 判定脚本：
     - 合成数据上 (i)、(ii) 的四种组合；
     - 缺句时拒绝；
     - 与 `w2_gate.py` 在同一份数据上的 CI 相同；
     - 健康检查标红的复用。

## 不做

- 不改 W2 已冻结的配置、划分与默认行为。
- 不在开发机上下载 RefCOCO。
- 不 commit、不 push。

## 完成总结要写

- 改了哪些文件；
- 构造规则；骨架的 sha256；服务器上的生成命令与预计用时；
- 守卫的用法；
- 判定脚本的输入、输出与规则；
- 测试结果原文；
- 与任务书的任何出入。
