[DEV-TASK 0062] W2 测试：不再把"仓库里的配置是未冻结的草稿"写死，冻结前后都能通过（D111 的经验；DEV-0056 的同类问题）

## 背景

- `configs/w2.json` 会在 W2-a 调参之后冻结，冻结由 `scripts/mgr/w2_tune.py freeze` 完成。冻结的内容是调参选中的那个臂：
  - 布局可能是 grid 或 separate；
  - 挑选时的思考可能开或关；
  - K 可能是 4、6、8；
  - 提出可能是 box 或 box_points。
- 冻结同时把 `configs/w2c_train.json` 的 `w2_config.sha256` 填上。
- 管理员在临时工作树里试过冻结：
  - `tests/test_w2_config.py::test_unfrozen_draft_and_d113_numbers` 失败；
  - `tests/test_w2c_train.py` 里也有一项失败，它依赖 `w2_config.sha256` 是 null。
  - 其他测试也可能写死了草稿的取值（思考开、box、K = 8 等），冻结成别的臂就会失败。
- W1 冻结时出过同样的问题（DEV-0056，D112）。

## 任务

1. **找出所有依赖草稿的测试**：W2 与 W2-c 测试里，凡是读 `configs/w2.json` 或 `configs/w2c_train.json`，并断言了某个调参项的取值、`frozen` 为 false、`sha256` 为 null 的，都要找出来。
   - 调参项包括：`frozen`、`propose.mode`、`select.thinking`、`render.layout`、`candidates.generators` 与 K、`style`。
2. **草稿夹具**：
   - 把当前的 `configs/w2.json`（sha256 `a4c013c3…`）与 `configs/w2c_train.json` 原样存为夹具 `tests/fixtures/w2/w2_draft.json` 与 `tests/fixtures/w2/w2c_train_draft.json`，测试里钉住两者的 sha256；
   - 依赖草稿取值的测试改读夹具。
3. **与冻结无关的检查**：给仓库里的配置写测试，冻结前后都成立。
   - 冻结前（`frozen: false`）：仓库配置与草稿夹具逐字节相同。
   - 冻结后（`frozen: true`）：
     - 1 号生成器是 `grabcut_default`，K 在 2–8；
     - `select.thinking_budget ≤ 2048`；
     - `status` 写明了 D 编号与臂的配置 sha256；
     - `configs/w2c_train.json` 的 `w2_config.sha256` 等于 `configs/w2.json` 的 sha256；
     - 非 tiny 的 `run_w2.py` 参数检查对 devcheck2 放行，即冻结保护通过。
4. **验证**：
   - 现在（未冻结）跑 `HF_HUB_OFFLINE=1 conda run -n visint pytest tests/test_w2_*.py tests/test_w2c_*.py -q`，全部通过；
   - 在临时副本里用 `scripts/mgr/w2_tune.py arm` 生成两种不同的臂（例如 `separate`、关思考、K = 4；以及 `box_points`），分别 `freeze` 之后再跑同样的测试，都全部通过。
   - 不要提交冻结后的配置，副本用完删掉。

## 不做

- 不改 `src/`、`scripts/`、`configs/`，只改 `tests/`。
- 不 commit、不 push。

## 完成总结要写

- 改了哪些测试；
- 三种情形（未冻结、两种冻结）的测试结果原文。
