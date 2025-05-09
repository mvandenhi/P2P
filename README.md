# From Pixels to Perception
This repository contains the code for the paper "*From Pixels to Perception: Interpretable Predictions via Instance-wise Grouped Feature Selection*" (P2P) [[paper](https://icml.cc/virtual/2025/poster/45880)].

**Abstract**: Understanding the decision-making process of machine learning models provides valuable insights into the task, the data, and the reasons behind a model's failures. In this work, we propose a method that performs inherently interpretable predictions through the instance-wise sparsification of input images. To align the sparsification with human perception, we learn the masking in the space of semantically meaningful pixel regions rather than on pixel-level. Additionally, we introduce an explicit way to dynamically determine the required level of sparsity for each instance. We show empirically on semi-synthetic and natural image datasets that our inherently interpretable classifier produces more meaningful, human-understandable predictions than state-of-the-art benchmarks.

## Instructions
1. Install the packages and dependencies from the `environment.yml` file. 
2. Download the datasets described in the manuscript and potentially update the `data_path` variable in `/configs/data/data_defaults.yaml`.
    1. CIFAR-10 will get downloaded automatically upon using it.
    2. ImageNet can be downloaded here: [ImageNet Download](https://image-net.org/download.php).
    3. ImageNet9 requires ImageNet. To obtain the test set data and masks, download the backgrounds_challenge_data here: [ImageNet9 Download](https://github.com/MadryLab/backgrounds_challenge).  
    For the training and validation images, the original repository's links are currently broken. As a workaround, we provide the image list in the `utils` folder. Move these files to the `imagenet9` folder in your `data_path` directory, so that the folder structure looks like:  
    ```
    imagenet9/
    ├── backgrounds_challenge_data/
    ├── imagenet9_train_orig_path.pkl
    └── imagenet9_val_orig_path.pkl
    ```
    4. For BAM Scene and Object, follow the instructions here: [BAM Download](https://github.com/google-research-datasets/bam). Please pay special attention to the [stop sign class](https://github.com/google-research-datasets/bam/issues/7).
    The expected folder structure is
    ```
    bam/
    └── data/
        ├── scene/
        └── obj/
    ```
    5. For COCO-10, first, download COCO2017 here [COCO Download](https://cocodataset.org/#download) such that the folder structure is as follows
    ```
    coco/
    ├── annotations/
    ├── images/
    └── images_val/
    ```
    where annotations correspond to `2017 Train/Val annotations [241MB]`.
    Then, run `utils/construct_coco10_dataset.py`.
3. For Weights & Biases support, set mode to 'online' and adjust entity in `configs/config.yaml`.
4. Run the script `main.py` with the desired configuration of dataset and model from the `configs/` folder. We provide a description of all arguments in the config files.  

## Running Experiments

Some example runs:
- **P2P on CIFAR-10 dataset**:
  `python main.py +model=P2P +data=cifar10`  
- **P2P on ImageNet dataset for various certainty thresholds**:
  `python main.py +model=P2P +data=imagenet model.num_epochs=20 model.certainty_threshold=[0.8,0.9,0.95,0.99]`  
- **Fixed threshold variant of P2P with 40% of pixels retained on COCO-10**:  
  `python main.py +model=P2P +data=coco10 model.use_dynamic_threshold=False model.reg_threshold=0.4`  

## Citing
To cite P2P please use the following BibTEX entry:

```
@inproceedings{
vandenhirtz2025p2p,
title={From Pixels to Perception: Interpretable Predictions via Instance-wise Grouped Feature Selection},
author={Vandenhirtz, Moritz and Vogt, Julia E},
booktitle={International Conference on Machine Learning},
year={2025}
url={https://icml.cc/virtual/2025/poster/45880}
}
```

