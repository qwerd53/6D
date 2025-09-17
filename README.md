<div style="text-align: center;">
  <h1>Matching 2D Images in 3D: Metric Relative Pose from Metric Correspondences</h1>
    <p>
    <a href="https://scholar.google.com/citations?user=m_SPRGUAAAAJ&hl=en">Axel Barroso-Laguna</a>
    ·
    <a href="https://scholar.google.com/citations?user=l-zRzDEAAAAJ&hl=en">Sowmya Munukutla</a>
    ·
    <a href="https://www.robots.ox.ac.uk/~victor/">Victor Adrian Prisacariu</a>
    ·
    <a href="https://ebrach.github.io/">Eric Brachmann</a>
  </p>
  <h2 style="font-size:1.7em; margin-top: -0.5rem; margin-bottom: -0.5rem;">CVPR 2024 (Oral)</h2>  
  <h3><a href="https://nianticlabs.github.io/mickey/">Project Page</a> | <a href="https://storage.googleapis.com/niantic-lon-static/research/mickey/mickey_main_paper.pdf">Paper</a> | <a href="https://arxiv.org/abs/2404.06337">arXiv</a> | <a href="https://storage.googleapis.com/niantic-lon-static/research/mickey/mickey_supp.pdf">Supplemental</a></h3>
</div>

This is the reference implementation of the paper **"Matching 2D Images in 3D: Metric Relative Pose from Metric Correspondences"** presented at **CVPR 2024**.

The paper introduces **M**etr**ic Key**points (MicKey), a feature detection pipeline that regresses keypoint positions in camera space.
MicKey presents a differentiable approach to establish metric correspondences via descriptor matching. From the metric correspondences, MicKey recovers metric relative poses.
MicKey is trained in an end-to-end fashion using differentiable pose optimization and requires only image pairs and their ground truth relative poses for supervision.

<p align="center">
    <img src="resources/teaser_mickey.png" alt="teaser" width="90%">
</p>

## Setup

Assuming a fresh [Anaconda](https://www.anaconda.com/download/) distribution, you can install dependencies with:
```shell
conda env create -f resources/environment.yml
conda activate mickey
```
We ran our experiments with PyTorch 2.0.1, CUDA 11.6, Python 3.8.17 and Debian GNU/Linux 11.

## Evaluating MicKey
MicKey aims at addressing the problem of instant Augmented Reality (AR) introduced in the [Map-free benchmark](https://research.nianticlabs.com/mapfree-reloc-benchmark).
In the Map-free set up, instead of building 3D maps from hundreds of images and scale calibrations, they propose to use only one photo of a scene as the map.
The Map-free benchmark then evaluates how accurate is the estimated metric relative pose between the reference image (the map)
and the query image (the user).

### Download Map-free dataset
You can find the Map-free dataset in [their project page](https://research.nianticlabs.com/mapfree-reloc-benchmark/dataset).
Extract the test.zip file into `data/mapfree`. Optionally, if you want to train MicKey, also download train and val zip files. 

### Pre-trained Models
We provide two [MicKey models](https://storage.googleapis.com/niantic-lon-static/research/mickey/assets/mickey_weights.zip).
  * _mickey.ckpt_: These are the default weights for MicKey, without using the overlapping scores provides in Map-free dataset and following the curriculum learning strategy described in the paper.
  * _mickey_sc.ckpt_: These are the weights when training MicKey using the min and max overlapping scores defined in Map-free.

Extract mickey_weights.zip into `weights/`. In the zip file, we also provide the default configuration needed to run the evaluation. 

### Run the submission script
Similar to Map-free code base, we provide a [submission script](submission.py) to generate submission files:

```shell
python submission.py --config path/to/config --checkpoint path/to/checkpoint --o results/your_method
```
The resulting file `results/your_method/submission.zip` can be uploaded to the Map-free [online benchmark website](https://research.nianticlabs.com/mapfree-reloc-benchmark) and compared against existing methods in the [leaderboard](https://research.nianticlabs.com/mapfree-reloc-benchmark/leaderboard).

### Run the local evaluation
The Map-free benchmark does not provide ground-truth poses for the test set. But we can still evaluate our method locally on the validation set.
```shell
python submission.py --config path/to/config --checkpoint path/to/checkpoint --o results/your_method --split val
```
and evaluate it as:
```shell
python -m benchmark.mapfree --submission_path results/your_method/submission.zip --split val
```

### Download MicKey correspondences and depth files
We provide the depth maps and correspondences computed by MicKey.
- Download [MicKey depth maps](https://storage.googleapis.com/niantic-lon-static/research/map-free-reloc/assets/mickey_depths.tar.gz).
- Download [MicKey correspondences](https://storage.googleapis.com/niantic-lon-static/research/map-free-reloc/assets/mickey_correspondences.zip).
- Extract the contents of both files to `data/mapfree`

Refer to the [Map-free benchmark](https://github.com/nianticlabs/map-free-reloc/tree/main?tab=readme-ov-file#feature-matching--scale-from-depth-baselines) to learn how to load precomputed correspondes and depth maps in their feature matching pipeline. 
  
### Running MicKey in custom images
We provide a [demo script](demo_inference.py) to run the relative pose estimation pipeline on custom image pairs.
As an example, we store in `data/toy_example` two images with their respective intrinsics.
The script computes their metric relative pose and saves the corresponding depth and keypoint score maps in the image folder.
Run the demo script as:
```shell
python demo_inference.py --im_path_ref data/toy_example/im0.jpg \
                         --im_path_dst data/toy_example/im1.jpg \
                         --intrinsics data/toy_example/intrinsics.txt \
                         --checkpoint path/to/checkpoint \
                         --config path/to/config
```

To generate the 3D assets as in [MicKey's webpage](https://nianticlabs.github.io/mickey/), you can turn on the
`--generate_3D_vis` flag. This will generate a rendered image with the input images, their computed 3D camera positions,
and the set of 3D point inliers.

## Training MicKey
Besides the test scripts, we also provide the training code to train MicKey. 

We provide two default configurations in `config/MicKey/`:
  * _curriculum_learning.yaml_: This configuration follows the curriculum learning approach detailed in the paper. 
   It hence does not use any image overlapping information but only relative ground truth poses during training. 
  * _overlap_score.yaml_: This configuration relies on the image overlapping information to only choose solvable image pairs during training.

Besides the two default configurations, we also provide a configuration to speed up their training.
These configurations use low-resolution images and do not add the null hypothesis (refer to Section 3.1.4 for details).
We recommend initializing MicKey with these configurations and then
fine-tuning the network with the default ones (which use high-resolution images and the null hypothesis).
They can be found under `config/MicKey/`:
  * _curriculum_learning_warm_up.yaml_
  * _overlap_score_warm_up.yaml_

To train MicKey default model, use:
```shell
python train.py --config config/MicKey/curriculum_learning.yaml \
                --dataset_config config/datasets/mapfree.yaml \
                --experiment experiment_name \
                --path_weights path/to/checkpoint/folder
```
Resume training from a checkpoint by adding `--resume {path_to_checkpoint}`.

The top models, according to the validation loss, the VCRE metric, and the pose AUC score, are saved during training.
Tensorboard results and checkpoints are saved into the folder `dir/to/weights/experiment_name`.

Note that by default, the configuration is set to use 4 GPUs. 
You can reduce the expected number of GPUs in the config file (e.g., _NUM_GPUS: 1_). 

## Changelog
- 19 September 2024: Added vectorized RANSAC and warm up configurations.
- 13 August 2024: Added visualization code.
- 7 June 2024: Added precomputed depth maps and keypoint correspondences.

## BibTeX
If you use this code in your research, please consider citing our paper:

```bibtex
@inproceedings{barroso2024mickey,
  title={Matching 2D Images in 3D: Metric Relative Pose from Metric Correspondences},
  author={Barroso-Laguna, Axel and Munukutla, Sowmya and Prisacariu, Victor and Brachmann, Eric},
  booktitle={CVPR},
  year={2024}
}
```

## License
Copyright © Niantic, Inc. 2024. Patent Pending. All rights reserved. This code is for non-commercial use. Please see the [license](LICENSE) file for terms.

## Acknowledgements
We use parts of code from different repositories. We thank the authors and maintainers of the following repositories.
- [Map-free](https://research.nianticlabs.com/mapfree-reloc-benchmark)
- [ACE](https://github.com/nianticlabs/ace)
- [ACE0](https://github.com/nianticlabs/acezero)
- [DUSt3R](https://github.com/naver/dust3r)
- [RoMa](https://github.com/Parskatt/RoMa)
- [DINOv2](https://github.com/facebookresearch/dinov2)
- [LoFTR](https://github.com/zju3dv/LoFTR)
- [DPT](https://github.com/isl-org/DPT)
- [ExtremeRotation](https://github.com/RuojinCai/ExtremeRotation_code)

