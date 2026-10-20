[RUN-TASK 0058] HPC 项目环境加装 W1 用的两个不含学习的图像库：opencv-python-headless 与 scikit-image（D103）。只用 CPU；不改任何已有的包；约 10 分钟

## 铁律

`~/a100-hpc-guide.md` 第 0 节：**任何情况下都不能取消或停止 Slurm 任务 157296 和 157297。**
- 前后各跑一次 `squeue`。
- 不碰 06 上的 RUN-0056、07 上的 RUN-0054。
- 只改我们自己的项目环境 `$VI/env`。不改 chenyuma 的 conda base，不改 `~/.bashrc`。
- pip 一律 `--no-cache-dir`。

## 为什么

- W1（D103）的工作台要用边缘检测、GrabCut、分水岭、超像素等不含学习的算法。
- 项目环境里目前没有 cv2，也没有 skimage（RUN-0044 的环境报告；`tests/test_metrics_seg.py:129` 在 HPC 上一直因此跳过）。

## 做法

1. hpcmgt 上 `squeue`。记下现在的版本：

   ```bash
   VI=/home/mail/chenyuma/visint
   $VI/env/bin/python -c "import numpy, scipy, PIL, torch; print(numpy.__version__, scipy.__version__, PIL.__version__, torch.__version__)"
   $VI/env/bin/python -m pip freeze > $VI/tmp/pip_freeze_before_0058.txt
   ```

2. **先演练**，不装：

   ```bash
   $VI/env/bin/python -m pip install --no-cache-dir --dry-run "opencv-python-headless==4.10.0.84" "scikit-image==0.24.0" 2>&1 | tail -20
   ```

   - 输出里只允许出现**新装**的包：opencv-python-headless、scikit-image，以及它们的新依赖，例如 imageio、tifffile、lazy_loader、networkx。
   - **任何已有的包要升级或降级**，就停下，把演练输出原文发给管理员。尤其是 numpy、scipy、pillow、torch、packaging。
   - 如果 `$VI/env` 是继承 `$ROOT/env` 的 venv（`--system-site-packages`），注意 pip 是否会把包装进 `$ROOT/env`。只允许装进 `$VI/env`。
3. 演练没问题就照装。命令与第 2 步相同，去掉 `--dry-run`。
4. **核对**：

   ```bash
   $VI/env/bin/python -c "import cv2, skimage, numpy, scipy, PIL, torch; print(cv2.__version__, skimage.__version__, numpy.__version__, scipy.__version__, PIL.__version__, torch.__version__); print(hasattr(cv2, 'grabCut'), hasattr(cv2, 'watershed'))"
   $VI/env/bin/python -m pip freeze > $VI/tmp/pip_freeze_after_0058.txt
   diff $VI/tmp/pip_freeze_before_0058.txt $VI/tmp/pip_freeze_after_0058.txt
   ```

   - numpy、scipy、PIL、torch 的版本必须与第 1 步相同。
   - diff 里只有新增的行。
5. 再跑一次 `squeue`。

## 报告

推送 `reports/0058-hpc-env-w1-libs.md`，内容：
- 第 1 步的版本；
- 演练输出原文；
- 第 4 步的输出与 diff 原文。
