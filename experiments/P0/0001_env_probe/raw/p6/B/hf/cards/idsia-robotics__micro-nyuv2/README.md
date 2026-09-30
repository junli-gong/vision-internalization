# Mini-NYUv2: a 48x48 dataset for tiny Monocular Depth Estimation

## Dataset

This dataset scales down to 48x48 resolution a custom split, containing kitchens, living rooms, bedrooms and bathrooms, of the NYUv2-Depth dataset. It contains:

- A training split of 22600 images in 48x48 resolution
- A validation split of 2726 images in 48x48 resolution
- A test split of 2948 images in 48x48 resolution

All images are annotated with 48x48 depth and disparity maps obtained directly from the Kinect sensor. For every image, also a 360x360 centered portion of the original ground truth is provided, matching the same field of view of the respective image. If a 48x48 depth prediction related to a 48x48-sized image is upscaled to 360x360 resolution, these ground truth portions can be used to compute the prediction accuracy. 

## Usage

To use the dataset, simply: 
~~~
tar -xzvf nyuv2_48x48.tar.gz -C ./
~~~

Then, you can find the dataset under "nyuv2_48x48/". 

## License

This dataset is derived from [NYU Depth Dataset V2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html), and therefore released under [MIT](https://mit-license.org/) license. 

## Citation

If you use this dataset, please cite: 

~~~
@article{nadalini2025multi,
  title={Multi-modal On-Device Learning for Monocular Depth Estimation on Ultra-low-power MCUs},
  author={Nadalini, Davide and Rusci, Manuele and Cereda, Elia and Benini, Luca and Conti, Francesco and Palossi, Daniele},
  journal={arXiv preprint arXiv:2512.00086},
  year={2025}
}
~~~
