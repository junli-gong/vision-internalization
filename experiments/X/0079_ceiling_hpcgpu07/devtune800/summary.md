# X-1 上限（devtune800，格点 noise）

- 800 句；框 IoU ≥ 0.9 记好框；CI 为逐句 bootstrap 95%（1000 次，种子 20260929）。
- token：Qwen/Qwen3.5-9B@c20223623576，格式 x1-text-v1（只数这个通道的输出，框不计）。

## 要点：每种框上的参照与各通道、各噪声档里 cIoU 最高的格点

| 框 | 行 | 格点 | cIoU [95% CI] | 好框 cIoU | 坏框 cIoU | token 均值 |
|---|---|---|---|---|---|---|
| gt | 参照 | `box` | 0.550 [0.534, 0.566] | 0.550 | — | 0.0 |
| gt | 参照 | `gen.gc_default` | 0.621 [0.599, 0.644] | 0.621 | — | 0.0 |
| gt | 参照 | `genK.4` | 0.700 [0.682, 0.718] | 0.700 | — | 1.0 |
| gt | spl 最高（none） | `spl.n24.raster.none` | 0.943 [0.938, 0.947] | 0.943 | — | 236.5 |
| gt | spl 最高（light） | `spl.n24.raster.light` | 0.932 [0.928, 0.937] | 0.932 | — | 236.5 |
| gt | spl 最高（medium） | `spl.n24.gc_trimap.medium` | 0.903 [0.898, 0.909] | 0.903 | — | 236.6 |
| gt | spl 最高（heavy） | `spl.n24.gc_trimap.heavy` | 0.866 [0.860, 0.873] | 0.866 | — | 236.7 |
| gt | cor 最高（light） | `cor.gc_default.m2.nolock.grabcut.r3.light` | 0.734 [0.712, 0.754] | 0.734 | — | 166.3 |
| gt | cor 最高（medium） | `cor.gc_default.m2.nolock.grabcut.r3.medium` | 0.691 [0.670, 0.712] | 0.691 | — | 165.7 |
| gt | cor 最高（heavy） | `cor.gc_default.m2.nolock.grabcut.r3.heavy` | 0.651 [0.629, 0.675] | 0.651 | — | 165.4 |
| base | 参照 | `box` | 0.503 [0.485, 0.521] | 0.557 | 0.390 | 0.0 |
| base | 参照 | `gen.gc_default` | 0.572 [0.547, 0.598] | 0.662 | 0.392 | 0.0 |
| base | 参照 | `genK.4` | 0.638 [0.615, 0.662] | 0.727 | 0.462 | 1.0 |
| base | spl 最高（none） | `spl.n24.raster.none` | 0.860 [0.836, 0.882] | 0.943 | 0.683 | 236.4 |
| base | spl 最高（light） | `spl.n24.raster.light` | 0.851 [0.828, 0.873] | 0.933 | 0.677 | 236.5 |
| base | spl 最高（medium） | `spl.n24.gc_trimap.medium` | 0.824 [0.800, 0.846] | 0.907 | 0.647 | 236.6 |
| base | spl 最高（heavy） | `spl.n24.gc_trimap.heavy` | 0.790 [0.767, 0.811] | 0.869 | 0.620 | 236.8 |
| base | cor 最高（light） | `cor.gc_default.m2.nolock.grabcut.r3.light` | 0.665 [0.638, 0.691] | 0.760 | 0.468 | 164.3 |
| base | cor 最高（medium） | `cor.gc_default.m2.nolock.grabcut.r3.medium` | 0.620 [0.592, 0.646] | 0.711 | 0.434 | 163.5 |
| base | cor 最高（heavy） | `cor.gc_default.m2.nolock.grabcut.r3.heavy` | 0.593 [0.566, 0.620] | 0.683 | 0.412 | 163.2 |

## 框：gt

n = 800；好框 800、坏框 0（其中解析失败 0）；框 Acc@0.5 = 1.000

| 格点 | cIoU [95% CI] | 好框 cIoU | 坏框 cIoU | gIoU | token 均值 |
|---|---|---|---|---|---|
| `box` | 0.550 [0.534, 0.566] | 0.550 | — | 0.566 | 0.0 |
| `gen.gc_default` | 0.621 [0.599, 0.644] | 0.621 | — | 0.594 | 0.0 |
| `gen.gc_soft_box_minus10` | 0.612 [0.592, 0.632] | 0.612 | — | 0.605 | 0.0 |
| `gen.gc_default+lcc+fill` | 0.621 [0.599, 0.645] | 0.621 | — | 0.591 | 0.0 |
| `gen.ws_pts` | 0.583 [0.567, 0.600] | 0.583 | — | 0.604 | 0.0 |
| `genK.4` | 0.700 [0.682, 0.718] | 0.700 | — | 0.705 | 1.0 |
| `spl.n12.raster.none` | 0.872 [0.865, 0.879] | 0.872 | — | 0.866 | 118.4 |
| `spl.n12.gc_trimap.none` | 0.881 [0.874, 0.888] | 0.881 | — | 0.869 | 118.4 |
| `spl.n16.raster.none` | 0.907 [0.902, 0.913] | 0.907 | — | 0.901 | 157.7 |
| `spl.n16.gc_trimap.none` | 0.900 [0.893, 0.906] | 0.900 | — | 0.887 | 157.7 |
| `spl.n24.raster.none` | 0.943 [0.938, 0.947] | 0.943 | — | 0.936 | 236.5 |
| `spl.n24.gc_trimap.none` | 0.911 [0.905, 0.916] | 0.911 | — | 0.898 | 236.5 |
| `spl.n12.raster.light` | 0.868 [0.860, 0.875] | 0.868 | — | 0.862 | 118.4 |
| `spl.n12.gc_trimap.light` | 0.880 [0.873, 0.888] | 0.880 | — | 0.868 | 118.4 |
| `spl.n16.raster.light` | 0.901 [0.895, 0.907] | 0.901 | — | 0.894 | 157.8 |
| `spl.n16.gc_trimap.light` | 0.899 [0.893, 0.905] | 0.899 | — | 0.886 | 157.8 |
| `spl.n24.raster.light` | 0.932 [0.928, 0.937] | 0.932 | — | 0.925 | 236.5 |
| `spl.n24.gc_trimap.light` | 0.910 [0.905, 0.916] | 0.910 | — | 0.898 | 236.5 |
| `spl.n12.raster.medium` | 0.847 [0.840, 0.855] | 0.847 | — | 0.840 | 118.4 |
| `spl.n12.gc_trimap.medium` | 0.870 [0.863, 0.878] | 0.870 | — | 0.858 | 118.4 |
| `spl.n16.raster.medium` | 0.873 [0.867, 0.880] | 0.873 | — | 0.864 | 157.8 |
| `spl.n16.gc_trimap.medium` | 0.891 [0.885, 0.898] | 0.891 | — | 0.878 | 157.8 |
| `spl.n24.raster.medium` | 0.892 [0.887, 0.898] | 0.892 | — | 0.884 | 236.6 |
| `spl.n24.gc_trimap.medium` | 0.903 [0.898, 0.909] | 0.903 | — | 0.891 | 236.6 |
| `spl.n12.raster.heavy` | 0.805 [0.797, 0.813] | 0.805 | — | 0.795 | 118.4 |
| `spl.n12.gc_trimap.heavy` | 0.838 [0.830, 0.846] | 0.838 | — | 0.824 | 118.4 |
| `spl.n16.raster.heavy` | 0.819 [0.811, 0.826] | 0.819 | — | 0.810 | 157.9 |
| `spl.n16.gc_trimap.heavy` | 0.854 [0.847, 0.862] | 0.854 | — | 0.842 | 157.9 |
| `spl.n24.raster.heavy` | 0.829 [0.823, 0.836] | 0.829 | — | 0.820 | 236.7 |
| `spl.n24.gc_trimap.heavy` | 0.866 [0.860, 0.873] | 0.866 | — | 0.852 | 236.7 |
| `cor.gc_default.m2.lock.grabcut.r1.light` | 0.680 [0.657, 0.702] | 0.680 | — | 0.653 | 55.1 |
| `cor.gc_default.m2.lock.grabcut.r2.light` | 0.711 [0.688, 0.732] | 0.711 | — | 0.680 | 110.6 |
| `cor.gc_default.m2.lock.grabcut.r3.light` | 0.732 [0.710, 0.753] | 0.732 | — | 0.697 | 166.3 |
| `cor.gc_default.m2.nolock.grabcut.r1.light` | 0.680 [0.657, 0.702] | 0.680 | — | 0.653 | 55.1 |
| `cor.gc_default.m2.nolock.grabcut.r2.light` | 0.714 [0.694, 0.735] | 0.714 | — | 0.681 | 110.6 |
| `cor.gc_default.m2.nolock.grabcut.r3.light` | 0.734 [0.712, 0.754] | 0.734 | — | 0.698 | 166.3 |
| `cor.gc_default.m2.lock.grabcut.r1.medium` | 0.643 [0.619, 0.666] | 0.643 | — | 0.614 | 55.0 |
| `cor.gc_default.m2.lock.grabcut.r2.medium` | 0.673 [0.651, 0.694] | 0.673 | — | 0.640 | 110.3 |
| `cor.gc_default.m2.lock.grabcut.r3.medium` | 0.690 [0.669, 0.710] | 0.690 | — | 0.656 | 165.7 |
| `cor.gc_default.m2.nolock.grabcut.r1.medium` | 0.643 [0.619, 0.666] | 0.643 | — | 0.614 | 55.0 |
| `cor.gc_default.m2.nolock.grabcut.r2.medium` | 0.673 [0.651, 0.694] | 0.673 | — | 0.640 | 110.3 |
| `cor.gc_default.m2.nolock.grabcut.r3.medium` | 0.691 [0.670, 0.712] | 0.691 | — | 0.657 | 165.7 |
| `cor.gc_default.m2.lock.grabcut.r1.heavy` | 0.631 [0.609, 0.654] | 0.631 | — | 0.600 | 55.0 |
| `cor.gc_default.m2.lock.grabcut.r2.heavy` | 0.640 [0.618, 0.663] | 0.640 | — | 0.611 | 110.0 |
| `cor.gc_default.m2.lock.grabcut.r3.heavy` | 0.650 [0.628, 0.673] | 0.650 | — | 0.620 | 165.4 |
| `cor.gc_default.m2.nolock.grabcut.r1.heavy` | 0.631 [0.609, 0.654] | 0.631 | — | 0.600 | 55.0 |
| `cor.gc_default.m2.nolock.grabcut.r2.heavy` | 0.641 [0.618, 0.663] | 0.641 | — | 0.612 | 110.0 |
| `cor.gc_default.m2.nolock.grabcut.r3.heavy` | 0.651 [0.629, 0.675] | 0.651 | — | 0.622 | 165.4 |

## 框：base

n = 800；好框 516、坏框 284（其中解析失败 14）；框 Acc@0.5 = 0.895

| 格点 | cIoU [95% CI] | 好框 cIoU | 坏框 cIoU | gIoU | token 均值 |
|---|---|---|---|---|---|
| `box` | 0.503 [0.485, 0.521] | 0.557 | 0.390 | 0.514 | 0.0 |
| `gen.gc_default` | 0.572 [0.547, 0.598] | 0.662 | 0.392 | 0.550 | 0.0 |
| `gen.gc_soft_box_minus10` | 0.559 [0.536, 0.583] | 0.650 | 0.379 | 0.554 | 0.0 |
| `gen.gc_default+lcc+fill` | 0.573 [0.547, 0.598] | 0.662 | 0.393 | 0.548 | 0.0 |
| `gen.ws_pts` | 0.532 [0.513, 0.552] | 0.604 | 0.389 | 0.556 | 0.0 |
| `genK.4` | 0.638 [0.615, 0.662] | 0.727 | 0.462 | 0.645 | 1.0 |
| `spl.n12.raster.none` | 0.798 [0.774, 0.819] | 0.873 | 0.637 | 0.794 | 118.4 |
| `spl.n12.gc_trimap.none` | 0.803 [0.779, 0.825] | 0.884 | 0.630 | 0.795 | 118.4 |
| `spl.n16.raster.none` | 0.830 [0.806, 0.851] | 0.909 | 0.660 | 0.826 | 157.7 |
| `spl.n16.gc_trimap.none` | 0.820 [0.796, 0.841] | 0.903 | 0.643 | 0.812 | 157.7 |
| `spl.n24.raster.none` | 0.860 [0.836, 0.882] | 0.943 | 0.683 | 0.857 | 236.4 |
| `spl.n24.gc_trimap.none` | 0.829 [0.806, 0.851] | 0.913 | 0.651 | 0.821 | 236.4 |
| `spl.n12.raster.light` | 0.795 [0.771, 0.816] | 0.869 | 0.635 | 0.791 | 118.4 |
| `spl.n12.gc_trimap.light` | 0.802 [0.778, 0.823] | 0.883 | 0.629 | 0.794 | 118.4 |
| `spl.n16.raster.light` | 0.824 [0.800, 0.846] | 0.903 | 0.656 | 0.820 | 157.7 |
| `spl.n16.gc_trimap.light` | 0.819 [0.795, 0.840] | 0.902 | 0.643 | 0.811 | 157.7 |
| `spl.n24.raster.light` | 0.851 [0.828, 0.873] | 0.933 | 0.677 | 0.848 | 236.5 |
| `spl.n24.gc_trimap.light` | 0.829 [0.805, 0.850] | 0.912 | 0.651 | 0.821 | 236.5 |
| `spl.n12.raster.medium` | 0.776 [0.753, 0.796] | 0.848 | 0.620 | 0.771 | 118.4 |
| `spl.n12.gc_trimap.medium` | 0.794 [0.770, 0.816] | 0.874 | 0.621 | 0.786 | 118.4 |
| `spl.n16.raster.medium` | 0.799 [0.777, 0.820] | 0.874 | 0.638 | 0.793 | 157.8 |
| `spl.n16.gc_trimap.medium` | 0.811 [0.788, 0.833] | 0.894 | 0.635 | 0.802 | 157.8 |
| `spl.n24.raster.medium` | 0.816 [0.794, 0.837] | 0.893 | 0.652 | 0.811 | 236.6 |
| `spl.n24.gc_trimap.medium` | 0.824 [0.800, 0.846] | 0.907 | 0.647 | 0.816 | 236.6 |
| `spl.n12.raster.heavy` | 0.738 [0.718, 0.758] | 0.805 | 0.594 | 0.731 | 118.4 |
| `spl.n12.gc_trimap.heavy` | 0.766 [0.744, 0.787] | 0.842 | 0.602 | 0.757 | 118.4 |
| `spl.n16.raster.heavy` | 0.751 [0.730, 0.770] | 0.820 | 0.601 | 0.742 | 157.8 |
| `spl.n16.gc_trimap.heavy` | 0.779 [0.756, 0.799] | 0.857 | 0.611 | 0.769 | 157.8 |
| `spl.n24.raster.heavy` | 0.760 [0.739, 0.780] | 0.830 | 0.609 | 0.753 | 236.8 |
| `spl.n24.gc_trimap.heavy` | 0.790 [0.767, 0.811] | 0.869 | 0.620 | 0.781 | 236.8 |
| `cor.gc_default.m2.lock.grabcut.r1.light` | 0.616 [0.589, 0.646] | 0.708 | 0.430 | 0.592 | 54.2 |
| `cor.gc_default.m2.lock.grabcut.r2.light` | 0.644 [0.615, 0.672] | 0.734 | 0.458 | 0.618 | 109.2 |
| `cor.gc_default.m2.lock.grabcut.r3.light` | 0.660 [0.632, 0.688] | 0.752 | 0.469 | 0.633 | 164.3 |
| `cor.gc_default.m2.nolock.grabcut.r1.light` | 0.616 [0.589, 0.646] | 0.708 | 0.430 | 0.592 | 54.2 |
| `cor.gc_default.m2.nolock.grabcut.r2.light` | 0.645 [0.616, 0.673] | 0.736 | 0.458 | 0.619 | 109.2 |
| `cor.gc_default.m2.nolock.grabcut.r3.light` | 0.665 [0.638, 0.691] | 0.760 | 0.468 | 0.635 | 164.3 |
| `cor.gc_default.m2.lock.grabcut.r1.medium` | 0.590 [0.565, 0.615] | 0.679 | 0.410 | 0.562 | 54.3 |
| `cor.gc_default.m2.lock.grabcut.r2.medium` | 0.602 [0.576, 0.629] | 0.690 | 0.422 | 0.577 | 108.8 |
| `cor.gc_default.m2.lock.grabcut.r3.medium` | 0.618 [0.591, 0.644] | 0.708 | 0.434 | 0.590 | 163.5 |
| `cor.gc_default.m2.nolock.grabcut.r1.medium` | 0.590 [0.565, 0.615] | 0.679 | 0.410 | 0.562 | 54.3 |
| `cor.gc_default.m2.nolock.grabcut.r2.medium` | 0.603 [0.576, 0.629] | 0.692 | 0.423 | 0.578 | 108.8 |
| `cor.gc_default.m2.nolock.grabcut.r3.medium` | 0.620 [0.592, 0.646] | 0.711 | 0.434 | 0.592 | 163.5 |
| `cor.gc_default.m2.lock.grabcut.r1.heavy` | 0.566 [0.538, 0.593] | 0.653 | 0.391 | 0.546 | 54.2 |
| `cor.gc_default.m2.lock.grabcut.r2.heavy` | 0.581 [0.553, 0.607] | 0.668 | 0.404 | 0.560 | 108.6 |
| `cor.gc_default.m2.lock.grabcut.r3.heavy` | 0.592 [0.564, 0.618] | 0.681 | 0.412 | 0.572 | 163.2 |
| `cor.gc_default.m2.nolock.grabcut.r1.heavy` | 0.566 [0.538, 0.593] | 0.653 | 0.391 | 0.546 | 54.2 |
| `cor.gc_default.m2.nolock.grabcut.r2.heavy` | 0.583 [0.555, 0.609] | 0.670 | 0.406 | 0.561 | 108.6 |
| `cor.gc_default.m2.nolock.grabcut.r3.heavy` | 0.593 [0.566, 0.620] | 0.683 | 0.412 | 0.573 | 163.2 |
