# RUN-0058 报告：HPC 项目环境加装 opencv-python-headless 与 scikit-image（W1，D103）

任务书 `prompts/server_hpc/0058-hpc-env-w1-libs.md`（origin a4de0900）。全部在 hpcmgt 上做，只用 CPU；没有碰 06 的 RUN-0056、07 的 RUN-0054。

## 第 1 步

- 10-20 21:37:55Z squeue：157296、157297 都是 RUNNING。
- `$VI/env` 是独立的 conda 环境（有 `conda-meta/`，没有 `pyvenv.cfg`）：`sys.prefix` = `sys.base_prefix` = `/home/mail/chenyuma/visint/env`，`site.getsitepackages()` 只有 `/home/mail/chenyuma/visint/env/lib/python3.12/site-packages`。不继承 `$ROOT/env`。
- 版本（numpy、scipy、PIL、torch）：`2.5.2 1.18.0 12.3.0 2.13.0+cu130`。
- 之前没有 cv2、skimage（`ModuleNotFoundError`）。
- pip 26.2.1；`pip config list`：`global.index-url='https://pypi.tuna.tsinghua.edu.cn/simple'`。
- `$VI/tmp/pip_freeze_before_0058.txt`：80 行。

## 第 2 步：演练（`--no-cache-dir --dry-run`），rc=0，输出原文

```
Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
Collecting opencv-python-headless==4.10.0.84
  Downloading opencv_python_headless-4.10.0.84-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (49.9 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 49.9/49.9 MB 3.6 MB/s  0:00:13
Collecting scikit-image==0.24.0
  Downloading scikit_image-0.24.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (15.0 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 15.0/15.0 MB 3.7 MB/s  0:00:04
Requirement already satisfied: numpy>=1.21.2 in ./env/lib/python3.12/site-packages (from opencv-python-headless==4.10.0.84) (2.5.2)
Requirement already satisfied: scipy>=1.9 in ./env/lib/python3.12/site-packages (from scikit-image==0.24.0) (1.18.0)
Requirement already satisfied: networkx>=2.8 in ./env/lib/python3.12/site-packages (from scikit-image==0.24.0) (3.6.1)
Requirement already satisfied: pillow>=9.1 in ./env/lib/python3.12/site-packages (from scikit-image==0.24.0) (12.3.0)
Collecting imageio>=2.33 (from scikit-image==0.24.0)
  Downloading imageio-2.38.1-py3-none-any.whl (328 kB)
Collecting tifffile>=2022.8.12 (from scikit-image==0.24.0)
  Downloading tifffile-2026.10.15-py3-none-any.whl (276 kB)
Requirement already satisfied: packaging>=21 in ./env/lib/python3.12/site-packages (from scikit-image==0.24.0) (26.2)
Collecting lazy-loader>=0.4 (from scikit-image==0.24.0)
  Downloading lazy_loader-0.6-py3-none-any.whl (8.8 kB)
Would install ImageIO-2.38.1 lazy-loader-0.6 opencv-python-headless-4.10.0.84 scikit-image-0.24.0 tifffile-2026.10.15
```

- 只有新装：opencv-python-headless、scikit-image、ImageIO、lazy-loader、tifffile。
- numpy、scipy、networkx、pillow、packaging 都是 `Requirement already satisfied`，没有升降级。装的位置是 `./env/lib/python3.12/site-packages`（即 `$VI/env`）。

## 第 3 步：安装（同一命令去掉 `--dry-run`），rc=0

```
Installing collected packages: tifffile, opencv-python-headless, lazy-loader, imageio, scikit-image

Successfully installed imageio-2.38.1 lazy-loader-0.6 opencv-python-headless-4.10.0.84 scikit-image-0.24.0 tifffile-2026.10.15
```

## 第 4 步：核对

```
4.10.0 0.24.0 2.5.2 1.18.0 12.3.0 2.13.0+cu130
True True
```

- cv2 4.10.0、skimage 0.24.0；numpy、scipy、PIL、torch 与第 1 步相同；`cv2.grabCut`、`cv2.watershed` 都在。
- `diff pip_freeze_before_0058.txt pip_freeze_after_0058.txt` 原文（只有新增）：

```
23a24
> ImageIO==2.38.1
28a30
> lazy-loader==0.6
50a53
> opencv-python-headless==4.10.0.84
65a69
> scikit-image==0.24.0
71a76
> tifffile==2026.10.15
```

## 第 5 步

- 21:39:45Z squeue：两条 RUNNING。
