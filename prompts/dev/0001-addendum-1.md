[DEV-TASK 0001 addendum-1] 按独立审查修复 DEV-0001（第 1 轮返工）

## 背景

DEV-0001 按现状已合入 main（带已知问题），见 D10。管理员另派独立审查 agent 审了过门仪器代码，报告要点如下：
- 核心数值与官方实现逐位一致：Boundary IoU、mIoU、cIoU/gIoU、深度、法向、框、网格互换。
- 有 3 处会直接改变门判定的缺陷（S1–S3）、4 处中等问题（M1–M4）、若干低风险问题。

复现脚本与官方代码原文在 `/private/tmp/claude-501/-Users-tim-Research-CVPR2027/66adb8f0-dadd-4263-acc8-6bc423ab4433/scratchpad/review0001/`，包括 `check_stats.py`、`check_detection*.py`、`check_grid_parse.py`、`mutate.sh`。可以直接改写成回归测试。

先读 `DECISIONS.md` 的 D10（对你 9 条偏离的裁定），以及 `DECISIONS_NEEDED.md` 的 N9。

## 必须修（编号对应审查报告）

1. **S1 — NaN 使 CI 失效**
   - `bootstrap_ci`、`paired_bootstrap_ci`、`is_significant` 的逐样本输入含 NaN 或 inf 时，抛 `ValueError` 并报出位置。
   - 不静默丢弃：对齐与过滤是调用方的事。
   - 用审查的复现（在 a[17] 置 NaN）写测试。

2. **S2 — 显著性判定重写**，要求"两条规则检验同一个统计量"，并且分方向：
   - `paired_bootstrap_ci(a, b, *, statistic=None, ...)`：a、b 按样本对齐，形状 (n, …)；同一组重采样下标同时作用于两者，返回 statistic(A*) − statistic(B*) 的 CI。statistic 为 None 时是一维均值。
   - `is_significant(per_sample_a, per_sample_b, *, statistic, higher_is_better, ...)`：输入形状 (n_samples, n_seeds, …)。
     - seed 级指标在函数内部用同一个 statistic 对每个 seed 的全集计算，不再由调用方另传。
     - 规则 1：|seed 均值差| > 2 × 合并标准差（ddof=1，每方至少 2 个 seed）。
     - 规则 2：按样本配对重采样，每次对各 seed 分别算 statistic 再取 seed 平均，差值的 95% CI 不含 0。
     - 规则 3：规则 1 的差值与规则 2 的 CI 同号。
     - 返回里要有 `significantly_better`（A 显著优于 B，方向按 `higher_is_better`）和 `significantly_worse`。三条都满足才算显著。
   - 提供统计量辅助函数：
     - 均值：gIoU、逐图 AbsRel、准确率；
     - `ratio_of_sums`：cIoU；
     - 由逐图逐类 (交, 并) 算 mIoU：nanmean，对齐现有的 mean_iou；
     - 逐像素汇总的法向平均误差：逐图给 (误差和, 像素数)。
   - 测试：
     - 审查的 S2 复现：用 `statistic=ratio_of_sums`，两条规则方向一致；
     - 符号相反的构造：判不显著；
     - `higher_is_better=False`（AbsRel）的构造。

3. **S3 — 无分数检测的 AP 与图像编号无关**
   - 预测全部没有分数时，AP 与 AP50 取 K 次随机打破平局的平均：图内与跨图一起随机置换，固定种子，K 放进 constants。同时报告 K 次的最小值和最大值。
   - 另输出单一工作点在 IoU=0.5 下的 P、R、F1。
   - 测试：
     - 审查的复现 1：两种编号给出相同的 AP；
     - 复现 2：编号置换下 AP 不再变化。
   - 给 AP 提供按图像重采样的 bootstrap 路径（statistic 回调，重采样次数可调）。
   - 标题指标用 mAP 还是 F1，由项目负责人在 N9 定；两者都实现。

4. **M1 — 网格解析**
   - 含至少 ⌈N/2⌉ 个 0/1 数字、同时还含其他任何字符（字母、别的数字、标点）的行，记为污染行：归入相邻网格块，不作为分段符，结果为 `ILLEGAL_CHAR`。
   - 测试：
     - 行末带注释 "<- top row"、多出一行、行内有字母笔误，都应失败，且失败原因正确；
     - 网格紧跟在 "Here is the 16x16 grid:" 这样的说明行之后，仍能正常解析；
     - 思考段里的草稿网格不归解析器管（DEV-0003 让适配器先剥离思考段）。

5. **M2 — 多选题规范化**：前缀形式只在余下文本里不再出现其他选项标记（`(B)`、`B.`、`B)`、`B:`）时才接受，否则返回 None。审查给的三个例子写成测试。

6. **M3 — 非二值输入**
   - 分割指标和网格编解码只接受 bool，或取值只在 {0, 1} 的整数数组，否则抛 `ValueError`。
   - 预测为 None 时按解析失败处理：该样本 IoU 记 0，cIoU 累计 (0, |GT|)；GT 为空时也记 0。
   - 写测试。

7. **M4 — 测试缺口**。为审查列出的每个"没被抓住的变异"补测试，然后用 `mutate.sh`（或等价做法）确认每个变异都被抓住。完成总结里附一张变异 → 被哪条测试抓住的表。
   - mIoU：含一个 IoU 为 0 的类（并 > 0，交 = 0）。
   - 检测：AP ≠ AP50，并能区分 AP75。
   - Boundary IoU：取一个 round 与 floor 结果不同的尺寸；预测不包含 GT 的例子。
   - 深度尺度-平移：
     - 预测不是 GT 的精确仿射；
     - 无效 GT 和 crop 外的像素带极端值；
     - 对齐后越界、需要裁剪；
     - 拟合必须用未裁剪的预测。
   - ratio bootstrap：CI 宽度 > 0，并与一个逐次循环的参考实现在同一种子下逐位相同。

8. **计算记账**：
   - `call_time_s`：逐次调用耗时之和。
   - `wall_time_s`：实际墙钟，只在样本级由调用方用 `as_sample(wall_time_s=...)` 给出。
   - `COMPUTE_SCHEMA_VERSION` 升到 2。
   - 测试并发情形：4 个块各 2 s，墙钟约 2 s，调用耗时 8 s。

9. **法向**：`valid` 改为必填参数，不再隐式推断。原有的零向量和非有限值掩码照旧叠加。

10. **ADE 常量**：`ADE_NUM_CLASSES`、`ADE_REDUCE_ZERO_LABEL` 接到 mIoU 相关函数的默认值上，不能是没人用的死常量。

11. **深度分档边界**：`depth_to_bins` 改用 `np.searchsorted(depth_bin_edges(), …)`，保证每条边界的归档一致。测试 65 条边界全部正确归档。

12. **NYU 上端开区间（D10）**：`NYU_DEPTH_MAX_INCLUSIVE = False`，与 BTS/AdaBins 一致。加测试钉住：GT 恰为 10.0 的像素被排除。

## 允许改动的文件

- `src/stats.py`、`src/metrics/*.py`、`src/grid/codec.py`、`src/accounting.py`
- `tests/` 下已有的测试文件，也可以新增测试文件
- `src/constants.py`：只改 `NYU_DEPTH_MAX_INCLUSIVE`，另可新增检测平局打破所需的常量。新常量加在"分割指标"一节的末尾（ADE 常量之后），**不要追加到文件末尾**：并行的 DEV-0002 会在文件末尾追加。

**不许碰**：
- `src/adapters/`：层号约定、视觉特征接口、token 预算累计、原始文本字段这些接口补充，已划给 DEV-0003；
- `src/executors/`；
- `requirements.txt`。

## 验收

- `conda run -n visint pytest tests/ -q` 全绿，报告测试数与耗时。
- 审查的复现脚本改写成测试后，都给出修正后的行为。
- 附变异 → 测试对照表。
- 逐条列出偏离。

完成后不要 commit、不要 push。用 SendMessage 把完成总结发给管理员会话 `cvpr2027-4f`。
