# X-1 上限（devcheck2，格点 noise）

- 500 句；框 IoU ≥ 0.9 记好框；CI 为逐句 bootstrap 95%（1000 次，种子 20260929）。
- token：Qwen/Qwen3.5-9B@c20223623576，格式 x1-text-v1（只数这个通道的输出，框不计）。

## 要点：每种框上的参照与各通道、各噪声档里 cIoU 最高的格点

| 框 | 行 | 格点 | cIoU [95% CI] | 好框 cIoU | 坏框 cIoU | token 均值 |
|---|---|---|---|---|---|---|
| gt | 参照 | `box` | 0.533 [0.515, 0.552] | 0.533 | — | 0.0 |
| gt | 参照 | `gen.gc_default` | 0.607 [0.579, 0.634] | 0.607 | — | 0.0 |
| gt | 参照 | `genK.4` | 0.693 [0.674, 0.713] | 0.693 | — | 1.0 |
| gt | spl 最高（none） | `spl.n24.raster.none` | 0.930 [0.917, 0.940] | 0.930 | — | 237.0 |
| gt | spl 最高（light） | `spl.n24.raster.light` | 0.919 [0.906, 0.929] | 0.919 | — | 237.0 |
| gt | spl 最高（medium） | `spl.n24.gc_trimap.medium` | 0.890 [0.878, 0.901] | 0.890 | — | 237.1 |
| gt | spl 最高（heavy） | `spl.n24.gc_trimap.heavy` | 0.853 [0.840, 0.865] | 0.853 | — | 237.3 |
| gt | cor 最高（light） | `cor.gc_default.m2.nolock.grabcut.r3.light` | 0.739 [0.715, 0.762] | 0.739 | — | 165.9 |
| gt | cor 最高（medium） | `cor.gc_default.m2.nolock.grabcut.r3.medium` | 0.680 [0.654, 0.705] | 0.680 | — | 165.3 |
| gt | cor 最高（heavy） | `cor.gc_default.m2.lock.grabcut.r3.heavy` | 0.641 [0.613, 0.666] | 0.641 | — | 164.8 |
| base | 参照 | `box` | 0.498 [0.477, 0.520] | 0.533 | 0.421 | 0.0 |
| base | 参照 | `gen.gc_default` | 0.578 [0.549, 0.607] | 0.641 | 0.446 | 0.0 |
| base | 参照 | `genK.4` | 0.644 [0.619, 0.670] | 0.709 | 0.508 | 1.0 |
| base | spl 最高（none） | `spl.n24.raster.none` | 0.876 [0.852, 0.899] | 0.932 | 0.750 | 237.0 |
| base | spl 最高（light） | `spl.n24.raster.light` | 0.867 [0.843, 0.889] | 0.922 | 0.742 | 237.0 |
| base | spl 最高（medium） | `spl.n24.gc_trimap.medium` | 0.839 [0.815, 0.860] | 0.898 | 0.705 | 237.2 |
| base | spl 最高（heavy） | `spl.n24.gc_trimap.heavy` | 0.807 [0.783, 0.828] | 0.862 | 0.682 | 237.3 |
| base | cor 最高（light） | `cor.gc_default.m2.nolock.grabcut.r3.light` | 0.685 [0.655, 0.713] | 0.758 | 0.524 | 165.3 |
| base | cor 最高（medium） | `cor.gc_default.m2.nolock.grabcut.r3.medium` | 0.636 [0.607, 0.665] | 0.709 | 0.480 | 164.7 |
| base | cor 最高（heavy） | `cor.gc_default.m2.nolock.grabcut.r3.heavy` | 0.592 [0.561, 0.624] | 0.658 | 0.450 | 164.5 |
| w2c | 参照 | `box` | 0.508 [0.487, 0.530] | 0.538 | 0.434 | 0.0 |
| w2c | 参照 | `gen.gc_default` | 0.585 [0.553, 0.617] | 0.635 | 0.467 | 0.0 |
| w2c | 参照 | `genK.4` | 0.658 [0.631, 0.685] | 0.711 | 0.531 | 1.0 |
| w2c | 参照 | `ref.w2c_final` | 0.642 [0.616, 0.670] | 0.701 | 0.505 | 0.0 |
| w2c | spl 最高（none） | `spl.n24.raster.none` | 0.899 [0.877, 0.917] | 0.931 | 0.811 | 237.0 |
| w2c | spl 最高（light） | `spl.n24.raster.light` | 0.889 [0.868, 0.907] | 0.921 | 0.802 | 237.0 |
| w2c | spl 最高（medium） | `spl.n24.gc_trimap.medium` | 0.860 [0.838, 0.877] | 0.896 | 0.762 | 237.1 |
| w2c | spl 最高（heavy） | `spl.n24.gc_trimap.heavy` | 0.827 [0.807, 0.845] | 0.860 | 0.741 | 237.3 |
| w2c | cor 最高（light） | `cor.w2c_final.m2.nolock.grabcut.r3.light` | 0.750 [0.727, 0.774] | 0.809 | 0.605 | 167.9 |
| w2c | cor 最高（medium） | `cor.w2c_final.m2.nolock.grabcut.r3.medium` | 0.705 [0.682, 0.730] | 0.754 | 0.582 | 168.4 |
| w2c | cor 最高（heavy） | `cor.w2c_final.m2.nolock.grabcut.r3.heavy` | 0.663 [0.634, 0.693] | 0.726 | 0.512 | 167.6 |

## 框：gt

n = 500；好框 500、坏框 0（其中解析失败 0）；框 Acc@0.5 = 1.000

| 格点 | cIoU [95% CI] | 好框 cIoU | 坏框 cIoU | gIoU | token 均值 |
|---|---|---|---|---|---|
| `box` | 0.533 [0.515, 0.552] | 0.533 | — | 0.572 | 0.0 |
| `gen.gc_default` | 0.607 [0.579, 0.634] | 0.607 | — | 0.598 | 0.0 |
| `gen.gc_soft_box_minus10` | 0.602 [0.579, 0.628] | 0.602 | — | 0.616 | 0.0 |
| `gen.gc_default+lcc+fill` | 0.607 [0.579, 0.634] | 0.607 | — | 0.597 | 0.0 |
| `gen.ws_pts` | 0.571 [0.548, 0.598] | 0.571 | — | 0.617 | 0.0 |
| `genK.4` | 0.693 [0.674, 0.713] | 0.693 | — | 0.717 | 1.0 |
| `spl.n12.raster.none` | 0.855 [0.840, 0.867] | 0.855 | — | 0.861 | 118.7 |
| `spl.n12.gc_trimap.none` | 0.865 [0.852, 0.876] | 0.865 | — | 0.865 | 118.7 |
| `spl.n16.raster.none` | 0.894 [0.881, 0.905] | 0.894 | — | 0.898 | 158.1 |
| `spl.n16.gc_trimap.none` | 0.887 [0.874, 0.897] | 0.887 | — | 0.885 | 158.1 |
| `spl.n24.raster.none` | 0.930 [0.917, 0.940] | 0.930 | — | 0.932 | 237.0 |
| `spl.n24.gc_trimap.none` | 0.898 [0.886, 0.908] | 0.898 | — | 0.896 | 237.0 |
| `spl.n12.raster.light` | 0.851 [0.836, 0.863] | 0.851 | — | 0.857 | 118.7 |
| `spl.n12.gc_trimap.light` | 0.864 [0.851, 0.875] | 0.864 | — | 0.864 | 118.7 |
| `spl.n16.raster.light` | 0.888 [0.875, 0.899] | 0.888 | — | 0.892 | 158.1 |
| `spl.n16.gc_trimap.light` | 0.885 [0.873, 0.896] | 0.885 | — | 0.884 | 158.1 |
| `spl.n24.raster.light` | 0.919 [0.906, 0.929] | 0.919 | — | 0.921 | 237.0 |
| `spl.n24.gc_trimap.light` | 0.897 [0.885, 0.907] | 0.897 | — | 0.895 | 237.0 |
| `spl.n12.raster.medium` | 0.830 [0.816, 0.842] | 0.830 | — | 0.836 | 118.7 |
| `spl.n12.gc_trimap.medium` | 0.854 [0.841, 0.866] | 0.854 | — | 0.856 | 118.7 |
| `spl.n16.raster.medium` | 0.856 [0.843, 0.868] | 0.856 | — | 0.860 | 158.2 |
| `spl.n16.gc_trimap.medium` | 0.875 [0.862, 0.885] | 0.875 | — | 0.874 | 158.2 |
| `spl.n24.raster.medium` | 0.878 [0.865, 0.888] | 0.878 | — | 0.881 | 237.1 |
| `spl.n24.gc_trimap.medium` | 0.890 [0.878, 0.901] | 0.890 | — | 0.888 | 237.1 |
| `spl.n12.raster.heavy` | 0.786 [0.771, 0.799] | 0.786 | — | 0.791 | 118.7 |
| `spl.n12.gc_trimap.heavy` | 0.822 [0.808, 0.835] | 0.822 | — | 0.823 | 118.7 |
| `spl.n16.raster.heavy` | 0.802 [0.788, 0.814] | 0.802 | — | 0.806 | 158.2 |
| `spl.n16.gc_trimap.heavy` | 0.841 [0.827, 0.853] | 0.841 | — | 0.839 | 158.2 |
| `spl.n24.raster.heavy` | 0.815 [0.801, 0.828] | 0.815 | — | 0.818 | 237.3 |
| `spl.n24.gc_trimap.heavy` | 0.853 [0.840, 0.865] | 0.853 | — | 0.852 | 237.3 |
| `cor.gc_default.m2.lock.grabcut.r1.light` | 0.677 [0.652, 0.701] | 0.677 | — | 0.657 | 54.8 |
| `cor.gc_default.m2.lock.grabcut.r2.light` | 0.714 [0.690, 0.737] | 0.714 | — | 0.687 | 110.2 |
| `cor.gc_default.m2.lock.grabcut.r3.light` | 0.737 [0.712, 0.759] | 0.737 | — | 0.708 | 165.9 |
| `cor.gc_default.m2.nolock.grabcut.r1.light` | 0.677 [0.652, 0.701] | 0.677 | — | 0.657 | 54.8 |
| `cor.gc_default.m2.nolock.grabcut.r2.light` | 0.715 [0.691, 0.737] | 0.715 | — | 0.687 | 110.2 |
| `cor.gc_default.m2.nolock.grabcut.r3.light` | 0.739 [0.715, 0.762] | 0.739 | — | 0.709 | 165.9 |
| `cor.gc_default.m2.lock.grabcut.r1.medium` | 0.632 [0.603, 0.659] | 0.632 | — | 0.618 | 54.8 |
| `cor.gc_default.m2.lock.grabcut.r2.medium` | 0.659 [0.633, 0.685] | 0.659 | — | 0.636 | 110.0 |
| `cor.gc_default.m2.lock.grabcut.r3.medium` | 0.680 [0.654, 0.705] | 0.680 | — | 0.656 | 165.3 |
| `cor.gc_default.m2.nolock.grabcut.r1.medium` | 0.632 [0.603, 0.659] | 0.632 | — | 0.618 | 54.8 |
| `cor.gc_default.m2.nolock.grabcut.r2.medium` | 0.660 [0.634, 0.685] | 0.660 | — | 0.637 | 110.0 |
| `cor.gc_default.m2.nolock.grabcut.r3.medium` | 0.680 [0.654, 0.705] | 0.680 | — | 0.656 | 165.3 |
| `cor.gc_default.m2.lock.grabcut.r1.heavy` | 0.625 [0.596, 0.651] | 0.625 | — | 0.613 | 54.8 |
| `cor.gc_default.m2.lock.grabcut.r2.heavy` | 0.636 [0.610, 0.663] | 0.636 | — | 0.621 | 109.8 |
| `cor.gc_default.m2.lock.grabcut.r3.heavy` | 0.641 [0.613, 0.666] | 0.641 | — | 0.629 | 164.8 |
| `cor.gc_default.m2.nolock.grabcut.r1.heavy` | 0.625 [0.596, 0.651] | 0.625 | — | 0.613 | 54.8 |
| `cor.gc_default.m2.nolock.grabcut.r2.heavy` | 0.636 [0.610, 0.663] | 0.636 | — | 0.621 | 109.8 |
| `cor.gc_default.m2.nolock.grabcut.r3.heavy` | 0.640 [0.612, 0.666] | 0.640 | — | 0.628 | 164.8 |

## 框：base

n = 500；好框 327、坏框 173（其中解析失败 4）；框 Acc@0.5 = 0.918

| 格点 | cIoU [95% CI] | 好框 cIoU | 坏框 cIoU | gIoU | token 均值 |
|---|---|---|---|---|---|
| `box` | 0.498 [0.477, 0.520] | 0.533 | 0.421 | 0.526 | 0.0 |
| `gen.gc_default` | 0.578 [0.549, 0.607] | 0.641 | 0.446 | 0.569 | 0.0 |
| `gen.gc_soft_box_minus10` | 0.562 [0.536, 0.590] | 0.623 | 0.436 | 0.576 | 0.0 |
| `gen.gc_default+lcc+fill` | 0.577 [0.549, 0.608] | 0.640 | 0.447 | 0.568 | 0.0 |
| `gen.ws_pts` | 0.541 [0.519, 0.565] | 0.595 | 0.426 | 0.575 | 0.0 |
| `genK.4` | 0.644 [0.619, 0.670] | 0.709 | 0.508 | 0.661 | 1.0 |
| `spl.n12.raster.none` | 0.808 [0.784, 0.830] | 0.855 | 0.701 | 0.804 | 118.7 |
| `spl.n12.gc_trimap.none` | 0.815 [0.791, 0.836] | 0.869 | 0.693 | 0.805 | 118.7 |
| `spl.n16.raster.none` | 0.844 [0.820, 0.866] | 0.895 | 0.728 | 0.837 | 158.1 |
| `spl.n16.gc_trimap.none` | 0.834 [0.811, 0.855] | 0.892 | 0.706 | 0.822 | 158.1 |
| `spl.n24.raster.none` | 0.876 [0.852, 0.899] | 0.932 | 0.750 | 0.868 | 237.0 |
| `spl.n24.gc_trimap.none` | 0.844 [0.820, 0.865] | 0.903 | 0.711 | 0.832 | 237.0 |
| `spl.n12.raster.light` | 0.804 [0.781, 0.827] | 0.851 | 0.699 | 0.800 | 118.7 |
| `spl.n12.gc_trimap.light` | 0.814 [0.791, 0.835] | 0.868 | 0.694 | 0.804 | 118.7 |
| `spl.n16.raster.light` | 0.838 [0.814, 0.860] | 0.889 | 0.723 | 0.831 | 158.1 |
| `spl.n16.gc_trimap.light` | 0.834 [0.810, 0.855] | 0.891 | 0.705 | 0.822 | 158.1 |
| `spl.n24.raster.light` | 0.867 [0.843, 0.889] | 0.922 | 0.742 | 0.858 | 237.0 |
| `spl.n24.gc_trimap.light` | 0.844 [0.820, 0.865] | 0.903 | 0.711 | 0.832 | 237.0 |
| `spl.n12.raster.medium` | 0.787 [0.764, 0.808] | 0.832 | 0.684 | 0.781 | 118.6 |
| `spl.n12.gc_trimap.medium` | 0.806 [0.783, 0.827] | 0.859 | 0.687 | 0.796 | 118.6 |
| `spl.n16.raster.medium` | 0.811 [0.788, 0.832] | 0.859 | 0.703 | 0.804 | 158.2 |
| `spl.n16.gc_trimap.medium` | 0.827 [0.804, 0.848] | 0.883 | 0.700 | 0.816 | 158.2 |
| `spl.n24.raster.medium` | 0.830 [0.807, 0.851] | 0.881 | 0.713 | 0.822 | 237.2 |
| `spl.n24.gc_trimap.medium` | 0.839 [0.815, 0.860] | 0.898 | 0.705 | 0.826 | 237.2 |
| `spl.n12.raster.heavy` | 0.742 [0.720, 0.762] | 0.783 | 0.647 | 0.737 | 118.6 |
| `spl.n12.gc_trimap.heavy` | 0.775 [0.751, 0.795] | 0.826 | 0.660 | 0.766 | 118.6 |
| `spl.n16.raster.heavy` | 0.759 [0.737, 0.779] | 0.800 | 0.664 | 0.752 | 158.2 |
| `spl.n16.gc_trimap.heavy` | 0.791 [0.768, 0.811] | 0.841 | 0.677 | 0.780 | 158.2 |
| `spl.n24.raster.heavy` | 0.773 [0.750, 0.793] | 0.817 | 0.670 | 0.763 | 237.3 |
| `spl.n24.gc_trimap.heavy` | 0.807 [0.783, 0.828] | 0.862 | 0.682 | 0.793 | 237.3 |
| `cor.gc_default.m2.lock.grabcut.r1.light` | 0.616 [0.585, 0.648] | 0.693 | 0.450 | 0.606 | 54.6 |
| `cor.gc_default.m2.lock.grabcut.r2.light` | 0.655 [0.624, 0.686] | 0.726 | 0.501 | 0.636 | 109.9 |
| `cor.gc_default.m2.lock.grabcut.r3.light` | 0.681 [0.650, 0.711] | 0.755 | 0.517 | 0.656 | 165.3 |
| `cor.gc_default.m2.nolock.grabcut.r1.light` | 0.616 [0.585, 0.648] | 0.693 | 0.450 | 0.606 | 54.6 |
| `cor.gc_default.m2.nolock.grabcut.r2.light` | 0.660 [0.629, 0.690] | 0.731 | 0.504 | 0.639 | 109.9 |
| `cor.gc_default.m2.nolock.grabcut.r3.light` | 0.685 [0.655, 0.713] | 0.758 | 0.524 | 0.657 | 165.3 |
| `cor.gc_default.m2.lock.grabcut.r1.medium` | 0.588 [0.559, 0.618] | 0.651 | 0.454 | 0.577 | 54.6 |
| `cor.gc_default.m2.lock.grabcut.r2.medium` | 0.609 [0.579, 0.638] | 0.672 | 0.473 | 0.593 | 109.6 |
| `cor.gc_default.m2.lock.grabcut.r3.medium` | 0.633 [0.603, 0.661] | 0.705 | 0.477 | 0.614 | 164.7 |
| `cor.gc_default.m2.nolock.grabcut.r1.medium` | 0.588 [0.559, 0.618] | 0.651 | 0.454 | 0.577 | 54.6 |
| `cor.gc_default.m2.nolock.grabcut.r2.medium` | 0.610 [0.581, 0.638] | 0.674 | 0.473 | 0.594 | 109.6 |
| `cor.gc_default.m2.nolock.grabcut.r3.medium` | 0.636 [0.607, 0.665] | 0.709 | 0.480 | 0.616 | 164.7 |
| `cor.gc_default.m2.lock.grabcut.r1.heavy` | 0.580 [0.548, 0.610] | 0.642 | 0.442 | 0.565 | 54.7 |
| `cor.gc_default.m2.lock.grabcut.r2.heavy` | 0.585 [0.552, 0.617] | 0.652 | 0.438 | 0.571 | 109.5 |
| `cor.gc_default.m2.lock.grabcut.r3.heavy` | 0.590 [0.559, 0.622] | 0.656 | 0.446 | 0.577 | 164.5 |
| `cor.gc_default.m2.nolock.grabcut.r1.heavy` | 0.580 [0.548, 0.610] | 0.642 | 0.442 | 0.565 | 54.7 |
| `cor.gc_default.m2.nolock.grabcut.r2.heavy` | 0.585 [0.552, 0.617] | 0.652 | 0.437 | 0.571 | 109.5 |
| `cor.gc_default.m2.nolock.grabcut.r3.heavy` | 0.592 [0.561, 0.624] | 0.658 | 0.450 | 0.578 | 164.5 |

## 框：w2c

n = 500；好框 347、坏框 153（其中解析失败 0）；框 Acc@0.5 = 0.934

| 格点 | cIoU [95% CI] | 好框 cIoU | 坏框 cIoU | gIoU | token 均值 |
|---|---|---|---|---|---|
| `box` | 0.508 [0.487, 0.530] | 0.538 | 0.434 | 0.542 | 0.0 |
| `gen.gc_default` | 0.585 [0.553, 0.617] | 0.635 | 0.467 | 0.580 | 0.0 |
| `gen.gc_soft_box_minus10` | 0.571 [0.543, 0.600] | 0.619 | 0.456 | 0.587 | 0.0 |
| `gen.gc_default+lcc+fill` | 0.584 [0.552, 0.616] | 0.635 | 0.464 | 0.578 | 0.0 |
| `gen.ws_pts` | 0.552 [0.531, 0.579] | 0.593 | 0.455 | 0.591 | 0.0 |
| `genK.4` | 0.658 [0.631, 0.685] | 0.711 | 0.531 | 0.682 | 1.0 |
| `ref.w2c_final` | 0.642 [0.616, 0.670] | 0.701 | 0.505 | 0.669 | 0.0 |
| `spl.n12.raster.none` | 0.829 [0.808, 0.847] | 0.855 | 0.758 | 0.824 | 118.7 |
| `spl.n12.gc_trimap.none` | 0.836 [0.816, 0.854] | 0.869 | 0.750 | 0.825 | 118.7 |
| `spl.n16.raster.none` | 0.865 [0.844, 0.883] | 0.895 | 0.787 | 0.858 | 158.1 |
| `spl.n16.gc_trimap.none` | 0.856 [0.835, 0.873] | 0.891 | 0.763 | 0.843 | 158.1 |
| `spl.n24.raster.none` | 0.899 [0.877, 0.917] | 0.931 | 0.811 | 0.889 | 237.0 |
| `spl.n24.gc_trimap.none` | 0.867 [0.846, 0.884] | 0.904 | 0.769 | 0.853 | 237.0 |
| `spl.n12.raster.light` | 0.825 [0.804, 0.842] | 0.851 | 0.755 | 0.821 | 118.7 |
| `spl.n12.gc_trimap.light` | 0.835 [0.816, 0.853] | 0.868 | 0.749 | 0.824 | 118.7 |
| `spl.n16.raster.light` | 0.860 [0.839, 0.878] | 0.889 | 0.782 | 0.852 | 158.1 |
| `spl.n16.gc_trimap.light` | 0.855 [0.834, 0.872] | 0.890 | 0.762 | 0.842 | 158.1 |
| `spl.n24.raster.light` | 0.889 [0.868, 0.907] | 0.921 | 0.802 | 0.879 | 237.0 |
| `spl.n24.gc_trimap.light` | 0.867 [0.845, 0.884] | 0.904 | 0.768 | 0.853 | 237.0 |
| `spl.n12.raster.medium` | 0.803 [0.783, 0.820] | 0.829 | 0.735 | 0.799 | 118.7 |
| `spl.n12.gc_trimap.medium` | 0.825 [0.804, 0.842] | 0.858 | 0.738 | 0.815 | 118.7 |
| `spl.n16.raster.medium` | 0.831 [0.811, 0.848] | 0.859 | 0.757 | 0.824 | 158.1 |
| `spl.n16.gc_trimap.medium` | 0.846 [0.825, 0.863] | 0.881 | 0.755 | 0.834 | 158.1 |
| `spl.n24.raster.medium` | 0.849 [0.828, 0.866] | 0.879 | 0.768 | 0.841 | 237.1 |
| `spl.n24.gc_trimap.medium` | 0.860 [0.838, 0.877] | 0.896 | 0.762 | 0.847 | 237.1 |
| `spl.n12.raster.heavy` | 0.760 [0.739, 0.777] | 0.783 | 0.697 | 0.756 | 118.6 |
| `spl.n12.gc_trimap.heavy` | 0.796 [0.775, 0.813] | 0.826 | 0.715 | 0.786 | 118.6 |
| `spl.n16.raster.heavy` | 0.779 [0.760, 0.795] | 0.803 | 0.712 | 0.773 | 158.1 |
| `spl.n16.gc_trimap.heavy` | 0.815 [0.794, 0.832] | 0.846 | 0.730 | 0.803 | 158.1 |
| `spl.n24.raster.heavy` | 0.792 [0.773, 0.809] | 0.817 | 0.725 | 0.784 | 237.3 |
| `spl.n24.gc_trimap.heavy` | 0.827 [0.807, 0.845] | 0.860 | 0.741 | 0.814 | 237.3 |
| `cor.gc_default.m2.lock.grabcut.r1.light` | 0.629 [0.596, 0.660] | 0.686 | 0.484 | 0.619 | 54.9 |
| `cor.gc_default.m2.lock.grabcut.r2.light` | 0.671 [0.641, 0.699] | 0.720 | 0.547 | 0.653 | 110.4 |
| `cor.gc_default.m2.lock.grabcut.r3.light` | 0.691 [0.661, 0.718] | 0.745 | 0.554 | 0.669 | 166.2 |
| `cor.gc_default.m2.nolock.grabcut.r1.light` | 0.629 [0.596, 0.660] | 0.686 | 0.484 | 0.619 | 54.9 |
| `cor.gc_default.m2.nolock.grabcut.r2.light` | 0.674 [0.644, 0.702] | 0.724 | 0.548 | 0.656 | 110.4 |
| `cor.gc_default.m2.nolock.grabcut.r3.light` | 0.695 [0.665, 0.721] | 0.750 | 0.555 | 0.671 | 166.2 |
| `cor.gc_default.m2.lock.grabcut.r1.medium` | 0.607 [0.574, 0.638] | 0.667 | 0.461 | 0.597 | 55.0 |
| `cor.gc_default.m2.lock.grabcut.r2.medium` | 0.629 [0.597, 0.660] | 0.689 | 0.482 | 0.610 | 110.2 |
| `cor.gc_default.m2.lock.grabcut.r3.medium` | 0.646 [0.615, 0.676] | 0.703 | 0.501 | 0.624 | 165.3 |
| `cor.gc_default.m2.nolock.grabcut.r1.medium` | 0.607 [0.574, 0.638] | 0.667 | 0.461 | 0.597 | 55.0 |
| `cor.gc_default.m2.nolock.grabcut.r2.medium` | 0.628 [0.595, 0.659] | 0.688 | 0.481 | 0.610 | 110.2 |
| `cor.gc_default.m2.nolock.grabcut.r3.medium` | 0.646 [0.616, 0.675] | 0.703 | 0.501 | 0.625 | 165.3 |
| `cor.gc_default.m2.lock.grabcut.r1.heavy` | 0.596 [0.565, 0.626] | 0.648 | 0.468 | 0.582 | 54.9 |
| `cor.gc_default.m2.lock.grabcut.r2.heavy` | 0.603 [0.571, 0.633] | 0.653 | 0.480 | 0.587 | 110.1 |
| `cor.gc_default.m2.lock.grabcut.r3.heavy` | 0.604 [0.573, 0.635] | 0.658 | 0.471 | 0.589 | 165.3 |
| `cor.gc_default.m2.nolock.grabcut.r1.heavy` | 0.596 [0.565, 0.626] | 0.648 | 0.468 | 0.582 | 54.9 |
| `cor.gc_default.m2.nolock.grabcut.r2.heavy` | 0.603 [0.572, 0.633] | 0.654 | 0.481 | 0.588 | 110.1 |
| `cor.gc_default.m2.nolock.grabcut.r3.heavy` | 0.604 [0.573, 0.635] | 0.661 | 0.462 | 0.588 | 165.3 |
| `cor.w2c_final.m2.lock.grabcut.r1.light` | 0.698 [0.675, 0.721] | 0.752 | 0.564 | 0.707 | 55.8 |
| `cor.w2c_final.m2.lock.grabcut.r2.light` | 0.733 [0.710, 0.756] | 0.790 | 0.588 | 0.732 | 111.8 |
| `cor.w2c_final.m2.lock.grabcut.r3.light` | 0.750 [0.727, 0.773] | 0.810 | 0.602 | 0.747 | 167.9 |
| `cor.w2c_final.m2.nolock.grabcut.r1.light` | 0.698 [0.675, 0.721] | 0.752 | 0.564 | 0.707 | 55.8 |
| `cor.w2c_final.m2.nolock.grabcut.r2.light` | 0.736 [0.714, 0.758] | 0.791 | 0.598 | 0.733 | 111.8 |
| `cor.w2c_final.m2.nolock.grabcut.r3.light` | 0.750 [0.727, 0.774] | 0.809 | 0.605 | 0.749 | 167.9 |
| `cor.w2c_final.m2.lock.grabcut.r1.medium` | 0.664 [0.640, 0.691] | 0.720 | 0.531 | 0.680 | 55.9 |
| `cor.w2c_final.m2.lock.grabcut.r2.medium` | 0.684 [0.659, 0.710] | 0.736 | 0.552 | 0.690 | 112.0 |
| `cor.w2c_final.m2.lock.grabcut.r3.medium` | 0.703 [0.679, 0.728] | 0.754 | 0.574 | 0.706 | 168.4 |
| `cor.w2c_final.m2.nolock.grabcut.r1.medium` | 0.664 [0.640, 0.691] | 0.720 | 0.531 | 0.680 | 55.9 |
| `cor.w2c_final.m2.nolock.grabcut.r2.medium` | 0.684 [0.659, 0.710] | 0.737 | 0.553 | 0.690 | 112.0 |
| `cor.w2c_final.m2.nolock.grabcut.r3.medium` | 0.705 [0.682, 0.730] | 0.754 | 0.582 | 0.707 | 168.4 |
| `cor.w2c_final.m2.lock.grabcut.r1.heavy` | 0.647 [0.620, 0.677] | 0.706 | 0.508 | 0.672 | 55.8 |
| `cor.w2c_final.m2.lock.grabcut.r2.heavy` | 0.654 [0.625, 0.683] | 0.715 | 0.508 | 0.675 | 111.8 |
| `cor.w2c_final.m2.lock.grabcut.r3.heavy` | 0.661 [0.632, 0.691] | 0.723 | 0.513 | 0.680 | 167.6 |
| `cor.w2c_final.m2.nolock.grabcut.r1.heavy` | 0.647 [0.620, 0.677] | 0.706 | 0.508 | 0.672 | 55.8 |
| `cor.w2c_final.m2.nolock.grabcut.r2.heavy` | 0.654 [0.626, 0.683] | 0.716 | 0.508 | 0.675 | 111.8 |
| `cor.w2c_final.m2.nolock.grabcut.r3.heavy` | 0.663 [0.634, 0.693] | 0.726 | 0.512 | 0.681 | 167.6 |
