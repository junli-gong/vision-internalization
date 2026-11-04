# X-M 的去留判定 G-XM（D165 第五节，规则写死）

**结论：通过**

- X-M 检查点 manifest 29e26667c7810ef2…（提示 xm-prompts-v1）；X-2 seed 0 检查点 a49cf4e5a81fd8f7…；模型 qwen3.5-9b
- 判定集 xm_decision：500 张图、1437 对（96a12dc2f902…）
- 节点：{'decision': {'xm_main': 'hpcgpu07', 'xm_box_grabcut': 'hpcgpu07', 'base_det': 'hpcgpu07'}, 'refseg_xm': ['hpcgpu07'], 'refseg_x2': ['hpcgpu07'], 'lb_xm': 'hpcgpu07', 'lb_base': 'hpcgpu07', 'lb_x2': 'hpcgpu07'}

| 条 | 读数 | X-M | 对照 | 差 | 95% CI | 阈值（下界 >） | 结果 |
|---|---|---|---|---|---|---|---|
| 1 | mask_f1_rexomni（X-M xm_spline_trimap − X-M xm_box_grabcut） | 0.6076 | 0.4136 | +0.1941 | [+0.1648, +0.2181] | 0 | 满足 |
| 2 | box_f1_rexomni（X-M（调用 1 的框） − 基座零样本 base_det） | 0.7266 | 0.6748 | +0.0519 | [+0.0263, +0.0628] | 0 | 满足 |
| 3 | ciou（X-M − X-2 seed 0） | 0.7357 | 0.7303 | +0.0054 | [-0.0017, +0.0123] | -0.02 | 满足 |
| 4（d170） | mmlu_pro_letter.accuracy（X-M − 同节点基座，百分点，n = 2000） | 52.85 | 54.15 | -1.300 | [-2.800, +0.150] | -5.7 | 满足 |
| 4（d170） | ifeval.prompt_level_strict（X-M − 同节点基座，百分点，n = 541） | 80.59 | 83.73 | -3.142 | [-6.100, -0.370] | -8.1 | 满足 |

四条（第 1–3 条为 D165 第五节原文；第 4 条为 XM-5a 的改写，D170）：
1. 描轮廓仍然关键：实例分割 F1，X-M 显著高于"X-M 的框直接交给 GrabCut"，即差值的 CI 下界 > 0。 → **满足**
2. 成为检测器：检测 F1，X-M 显著高于基座零样本，即差值的 CI 下界 > 0。 → **满足**
3. 指代分割不退步：在 2,300 句开发数据上，X-M 的 cIoU 不劣于 X-2，即差值的 CI 下界 > −0.02。 → **满足**
4. 大题量 MMLU-Pro（字母准确率，2,000 题）与 IFEval（prompt-level strict，541 条），X-M 相对同节点基座的差值，95% CI 下界都不低于"X-2 配方三个种子相对各自同节点基座的 CI 下界中最低者，再减 2 个百分点"：MMLU-Pro：CI 下界 > −5.7；IFEval：CI 下界 > −8.1。 → **满足**

**第 4 条，d170（进结论）**：X-M − 同节点基座 → **满足**（MMLU-Pro 下界 -2.800 > -5.7？IFEval 下界 -6.100 > -8.1？）
**第 4 条原文，d165（诊断，不入结论）**：X-M − X-2 seed 0：mmlu_pro_letter -1.150 [-2.400, +0.100]；ifeval +0.370 [-2.218, +2.957]；若计入（下界都 > -2）：不满足；两次运行同节点：True

(1)(2) 按图像重抽、(3) 逐句配对、(4) 逐题配对：95%、10,000 次、种子 20260929，不取整；第 4 条的写法 d170。

判定点核对：decision_n_images_is_500 = True；decision_splits_is_repo_config = True；decision_unsealed_with_G-XM = True；decision_same_node = True；refseg_same_node = True；lb_same_node = True；prompt_version_matches = True；decision_prompts_frozen = True；not_tiny = True；no_limit = True；same_model = True；refseg_is_dev2300 = True；lb_full = True；candidate_is_xm_train_seed0 = True；reference_is_x2_train_seed0 = True
