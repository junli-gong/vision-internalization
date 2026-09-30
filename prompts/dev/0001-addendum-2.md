[DEV-TASK 0001 addendum-2] 按复核修掉 addendum-1 新引入的问题（第 2 轮返工，也是最后一轮）

## 背景

addendum-1 已合入。独立审查复核确认 S1–S3、M1–M4 全部修好，核心数值对照仍然逐位一致或只有浮点级误差。复核同时查出修复新引入的问题，编号沿用复核报告（N1–N9）。

裁定见 `DECISIONS.md` 的 D12。复核脚本在 `/private/tmp/claude-501/-Users-tim-Research-CVPR2027/66adb8f0-dadd-4263-acc8-6bc423ab4433/scratchpad/review0001/`（`recheck_*.py`、`spot_mutants.py`）。

这是第 2 轮返工，按流程是最后一轮。做不到的，写进偏离说明原因，不要硬凑。

## 必须修

1. **N1 — `intersect_and_union` 的 `reduce_zero` 改为必填的关键字参数，不给默认值。**
   - 原因：按旧契约（先 reduce 再调用）会减两次，完美预测的 mIoU 算成 0.0；两类用法忘传参数时 mIoU 为 0.042，应为 0.920。
   - 同步改 docstring 与所有测试。
   - 加测试：漏传时报 TypeError。

2. **N2 — 多选题规范化的"其他选项标记"收窄**。前缀形式后面的余下文本里，只把下面三种算作其他选项：
   - 括号形式 `(X)` 或 `[X]`，不分大小写，X 不等于所选字母；
   - 列表式 `X)`：独立字母后接右括号，前面不是字母；
   - 明确给出另一个答案的说法：不分大小写地匹配 `answer is X`、`answer: X`，X 不等于所选字母。

   **`X.` 和 `X:` 不再算标记**：它们和缩写（U.S.）、句末提到其他选项的写法无法区分。
   - 测试：复核里 9 个正确答案全部判对；M2 的三个含糊例子仍返回 None：
     - `"(A) or (B)"` 命中括号形式；
     - `"A. No, the answer is B."` 命中 answer is；
     - `"A) and C) are both right"` 命中列表式。
   - 另外，`multiple_choice_accuracy` 同时返回 None 率（规范化失败的比例），让报告能分系统列出。

3. **N3 — `detection_bootstrap_statistic` 开头就检查 image_id**：预测的 image_id 必须都在 GT 里，否则与 `detection_map` 报同样的错。用复核的例子写测试：image_id 是字符串 '1'，GT 里是整数 1。

4. **N4 — 网格解析的两条规则**：
   - **污染行**：0/1 数字至少 ⌈N/2⌉ 个，**并且**占该行非分隔字符的一半或以上，同时还含别的字符。
     - 这样，坐标说明行和图例行不再是污染行；
     - "0110… <- top row"、行内字母笔误、`[0,1,…]` 仍是污染行。
   - **候选块**：含至少一条纯网格行，或含至少 ⌈N/2⌉ 条污染行。只含一两条污染行的段不是候选块。
   - 测试：
     - 复核 N4 表里的三种输出，网格本身完美，都能解析成功；
     - M1 的几个例子仍判 ILLEGAL_CHAR；
     - 每行都写成 `[0,1,…]` 的网格仍判 ILLEGAL_CHAR；
     - 原 `tests/test_grid_codec.py:241` 按新规则改写。

5. **N7 — `is_significant` 加防转置守卫**：n_seeds 大于 n_samples 时报错，提示很可能传反了。写测试。

6. **N8 — 检测的逐图计数与配对路径**：
   - `operating_point` 另外返回逐图计数：每张图的 (image_id, tp, fp, n_gt)，按 image_id 排序。
   - 新增 `f1_statistic`：输入形状 (n_images, 3) 的 (tp, fp, n_gt)，算 2Σtp / (Σ(tp+fp) + Σn_gt)，可以直接交给 `paired_bootstrap_ci` 或 `is_significant`。
   - 测试：
     - 由逐图计数算出的 F1 与 `operating_point` 的 F1_50 逐位相同；
     - 两个系统在同一批图上的配对 CI 例子。

7. **N5、N6 — 只改文档，写进 `detection.py` 的 docstring**：
   - bootstrap 的点估计用的是 `n_tiebreaks`（默认 1），它的 estimate 不是标题数字。标题 AP 取 `detection_map`（K=20），CI 取 bootstrap，这样的 CI 偏宽，偏保守。
   - AP 忽略全集里没有 GT 的类别，F1 把这类"幻觉类别"的框算作 FP。
   - 预测列表要按规范顺序传入：先按 image_id，再按模型的输出顺序，稳定排序。加一个 `canonical_order(preds)` 辅助函数，并写测试。

## 允许改动的文件

- `src/metrics/seg.py`、`src/metrics/misc.py`、`src/metrics/detection.py`、`src/grid/codec.py`、`src/stats.py`
- `tests/` 下相应的测试文件

**不许碰**：`src/constants.py`、`src/adapters/`、`src/executors/`、`requirements.txt`。

## 验收

- `conda run -n visint pytest tests/ -q` 全绿，报告测试数与耗时。
- 复核里的每个复现都给出修正后的行为。
- 为 N1、N2、N4、N7 各补一个会被抓住的变异，写进变异对照表。
- 逐条列出偏离。

完成后不要 commit、不要 push。用 SendMessage 把完成总结发给管理员会话 `cvpr2027-4f`。
