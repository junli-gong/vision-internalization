---
task_categories:
- depth-estimation
size_categories:
- 10K<n<100K
---

# Dataset Card for nyu-depthv2-wds

<!-- Provide a quick summary of the dataset. -->

This is the NYU DepthV2 dataset, converted into the webdataset format. https://huggingface.co/datasets/sayakpaul/nyu_depth_v2/

There are 47584 samples in the training split, and 654 samples in the validation split.

I shuffled both the training samples, and the validation samples.

I also cropped 16 pixels from all sides of the image, and depth image. I did this because there is a white border around all images.

This is an example of the border artifacts in the images:
![image/png](https://cdn-uploads.huggingface.co/production/uploads/6424d3394191f56e07324282/Ll0lkte1-x9ZAEwR6dMvI.png)

## Dataset Details

### Dataset Description

<!-- Provide a longer summary of what this dataset is. -->

Each sample contains a jpg image, and a numpy array of depth data. Depth maps are stored in float16.

### Dataset Sources [optional]

See https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html

## Uses

<!-- Address questions around how the dataset is intended to be used. -->

Train a depth prediction model on the train split, and test on the val split!
