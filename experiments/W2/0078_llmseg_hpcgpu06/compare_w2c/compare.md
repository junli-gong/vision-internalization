# LLMSeg 与 W2-c 的配对比较

LLMSeg − W2-c 的配对差（a = LLMSeg，b = W2-c）；rec = LLMSeg 修正后（标题），seg = LLMSeg 修正前；W2-c 是它的标题读数

| 划分 | LLMSeg 读数 | 指标 | n | LLMSeg | W2-c | 差 | 95% CI |
|---|---|---|---|---|---|---|---|
| test_refcoco_unc_testA | rec | cIoU | 300 | 0.2463 | 0.6645 | -0.4182 | [-0.4590, -0.3779] |
| test_refcoco_unc_testA | rec | gIoU | 300 | 0.2354 | 0.6633 | -0.4279 | [-0.4580, -0.3990] |
| test_refcoco_unc_testA | seg | cIoU | 300 | 0.3871 | 0.6645 | -0.2774 | [-0.3087, -0.2446] |
| test_refcoco_unc_testA | seg | gIoU | 300 | 0.3817 | 0.6633 | -0.2816 | [-0.3085, -0.2543] |
| test_refcoco_unc_testB | rec | cIoU | 300 | 0.3201 | 0.6918 | -0.3718 | [-0.4166, -0.3246] |
| test_refcoco_unc_testB | rec | gIoU | 300 | 0.2940 | 0.6986 | -0.4046 | [-0.4392, -0.3728] |
| test_refcoco_unc_testB | seg | cIoU | 300 | 0.4528 | 0.6918 | -0.2390 | [-0.2752, -0.2013] |
| test_refcoco_unc_testB | seg | gIoU | 300 | 0.4489 | 0.6986 | -0.2497 | [-0.2786, -0.2200] |
| test_refcoco+_unc_testA | rec | cIoU | 300 | 0.2340 | 0.6126 | -0.3786 | [-0.4196, -0.3345] |
| test_refcoco+_unc_testA | rec | gIoU | 300 | 0.2173 | 0.6326 | -0.4153 | [-0.4468, -0.3852] |
| test_refcoco+_unc_testA | seg | cIoU | 300 | 0.3664 | 0.6126 | -0.2462 | [-0.2872, -0.2031] |
| test_refcoco+_unc_testA | seg | gIoU | 300 | 0.3520 | 0.6326 | -0.2806 | [-0.3077, -0.2524] |
| test_refcoco+_unc_testB | rec | cIoU | 300 | 0.2942 | 0.6071 | -0.3129 | [-0.3633, -0.2657] |
| test_refcoco+_unc_testB | rec | gIoU | 300 | 0.2696 | 0.6323 | -0.3628 | [-0.3985, -0.3270] |
| test_refcoco+_unc_testB | seg | cIoU | 300 | 0.3926 | 0.6071 | -0.2145 | [-0.2586, -0.1749] |
| test_refcoco+_unc_testB | seg | gIoU | 300 | 0.3992 | 0.6323 | -0.2332 | [-0.2679, -0.2000] |
| test_refcocog_umd_test | rec | cIoU | 300 | 0.2745 | 0.6483 | -0.3739 | [-0.4181, -0.3302] |
| test_refcocog_umd_test | rec | gIoU | 300 | 0.2317 | 0.6364 | -0.4047 | [-0.4366, -0.3757] |
| test_refcocog_umd_test | seg | cIoU | 300 | 0.3954 | 0.6483 | -0.2529 | [-0.2856, -0.2179] |
| test_refcocog_umd_test | seg | gIoU | 300 | 0.3755 | 0.6364 | -0.2609 | [-0.2878, -0.2324] |
| 合并 | rec | cIoU | 1500 | 0.2749 | 0.6436 | -0.3687 | [-0.3895, -0.3472] |
| 合并 | rec | gIoU | 1500 | 0.2496 | 0.6526 | -0.4030 | [-0.4176, -0.3882] |
| 合并 | seg | cIoU | 1500 | 0.3983 | 0.6436 | -0.2453 | [-0.2626, -0.2277] |
| 合并 | seg | gIoU | 1500 | 0.3914 | 0.6526 | -0.2612 | [-0.2741, -0.2479] |

- {'rule': 'D125-4：LLMSeg（9B）不低于 W2-c 就立即报告；这里给点估计与 CI 两种口径，不作判定', 'point_estimate_ge_0': False, 'ci_high_ge_0': False}
