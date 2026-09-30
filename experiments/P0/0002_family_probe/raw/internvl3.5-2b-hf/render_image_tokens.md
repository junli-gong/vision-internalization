| 输入（W×H）与参数 | 视觉 token 数 | tile 数（pixel_values 第 0 维） | pixel_values 形状 |
|---|---|---|---|
| 448x448 default | 256 | 1 | [1, 3, 448, 448] |
| 448x448 crop_to_patches=False max_patches=12 | 256 | 1 | [1, 3, 448, 448] |
| 448x448 crop_to_patches=True max_patches=1 | 256 | 1 | [1, 3, 448, 448] |
| 448x448 crop_to_patches=True max_patches=4 | 256 | 1 | [1, 3, 448, 448] |
| 448x448 crop_to_patches=True max_patches=6 | 256 | 1 | [1, 3, 448, 448] |
| 448x448 crop_to_patches=True max_patches=12 | 256 | 1 | [1, 3, 448, 448] |
| 640x480 default | 3328 | 13 | [13, 3, 448, 448] |
| 640x480 crop_to_patches=False max_patches=12 | 256 | 1 | [1, 3, 448, 448] |
| 640x480 crop_to_patches=True max_patches=1 | 256 | 1 | [1, 3, 448, 448] |
| 640x480 crop_to_patches=True max_patches=4 | 256 | 1 | [1, 3, 448, 448] |
| 640x480 crop_to_patches=True max_patches=6 | 1792 | 7 | [7, 3, 448, 448] |
| 640x480 crop_to_patches=True max_patches=12 | 3328 | 13 | [13, 3, 448, 448] |
| 1024x768 default | 3328 | 13 | [13, 3, 448, 448] |
| 1024x768 crop_to_patches=False max_patches=12 | 256 | 1 | [1, 3, 448, 448] |
| 1024x768 crop_to_patches=True max_patches=1 | 256 | 1 | [1, 3, 448, 448] |
| 1024x768 crop_to_patches=True max_patches=4 | 1280 | 5 | [5, 3, 448, 448] |
| 1024x768 crop_to_patches=True max_patches=6 | 1792 | 7 | [7, 3, 448, 448] |
| 1024x768 crop_to_patches=True max_patches=12 | 3328 | 13 | [13, 3, 448, 448] |
| 1024x1024 default | 2560 | 10 | [10, 3, 448, 448] |
| 1024x1024 crop_to_patches=False max_patches=12 | 256 | 1 | [1, 3, 448, 448] |
| 1024x1024 crop_to_patches=True max_patches=1 | 256 | 1 | [1, 3, 448, 448] |
| 1024x1024 crop_to_patches=True max_patches=4 | 1280 | 5 | [5, 3, 448, 448] |
| 1024x1024 crop_to_patches=True max_patches=6 | 1280 | 5 | [5, 3, 448, 448] |
| 1024x1024 crop_to_patches=True max_patches=12 | 2560 | 10 | [10, 3, 448, 448] |
| 500x333 default | 1792 | 7 | [7, 3, 448, 448] |
| 500x333 crop_to_patches=False max_patches=12 | 256 | 1 | [1, 3, 448, 448] |
| 500x333 crop_to_patches=True max_patches=1 | 256 | 1 | [1, 3, 448, 448] |
| 500x333 crop_to_patches=True max_patches=4 | 768 | 3 | [3, 3, 448, 448] |
| 500x333 crop_to_patches=True max_patches=6 | 1792 | 7 | [7, 3, 448, 448] |
| 500x333 crop_to_patches=True max_patches=12 | 1792 | 7 | [7, 3, 448, 448] |
| 2000x300 default | 2048 | 8 | [8, 3, 448, 448] |
| 2000x300 crop_to_patches=False max_patches=12 | 256 | 1 | [1, 3, 448, 448] |
| 2000x300 crop_to_patches=True max_patches=1 | 256 | 1 | [1, 3, 448, 448] |
| 2000x300 crop_to_patches=True max_patches=4 | 1280 | 5 | [5, 3, 448, 448] |
| 2000x300 crop_to_patches=True max_patches=6 | 1792 | 7 | [7, 3, 448, 448] |
| 2000x300 crop_to_patches=True max_patches=12 | 2048 | 8 | [8, 3, 448, 448] |