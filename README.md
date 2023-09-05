# Image Labels Are All You Need for Coarse Seagrass Segmentation

The official repository for the paper: "Image Labels Are All You Need for Coarse Seagrass Segmentation".
 
Accepted for presentation at the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2024.  If this repository contributes to your research, please consider citing the publication below.

```
Scarlett Raine, Ross Marchant, Brano Kusy, Frederic Maire and Tobias Fischer (2024). 
Image Labels Are All You Need for Coarse Seagrass Segmentation. ACCEPTED TO WACV 2024. 
```

### Bibtex
```
@article{raine2023image,
  title={Image Labels Are All You Need for Coarse Seagrass Segmentation},
  author={Raine, Scarlett and Marchant, Ross and Kusy, Brano and Maire, Frederic and Fischer, Tobias},
  journal={arXiv preprint arXiv:2303.00973},
  year={2023}
}
```

Full Paper (Pre-print only): \[[Paper](https://arxiv.org/abs/2303.00973)]

YouTube Video: \[[Video](https://youtu.be/TeyjcVYFaKY)]

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Acknowledgements](#acknowledgements)

<a name="installation"></a>
## Installation
We suggest using the Mamba (or Anaconda) package manager to install dependencies.

1. Download Mamba
2. Create environment from .yml provided - run the following commands:

```mamba env create -f wacv_environment.yml```

```mamba install imgaug```

<a name="quick-start"></a>

## Quick Start 

### Inference
-Models and script provided for our ensemble of SeaCLIP and SeaFeats to perform inference on a test image (provided).

-Run the inference script:

```python ensemble_whole_image_inference.py```

-This script will read in the saved models and perform inference on the test image provided.

### Calculate test results
-To calculate the test results on the DeepSeagrass test dataset, run the following script:

```test_ensemble.py```

-This code assumes that the DeepSeagrass test dataset is downloaded and saved in a folder called DeepSeagrass-Test with folder structure like so:

```
Code
|---DeepSeagrass-Test
|   |---Background
|   |---Ferny
|   |---Rounded
|   |---Strappy
```

### Trained models provided

-SeaFeats trained end-to-end: ```SeaFeats.pt```

-SeaCLIP trained end-to-end: ```SeaCLIP.pt```

## Training From Scratch

### 1. SeaFeats
### Train SeaFeats feature extractor
-We use SimCLR as a pretext task to train the SeaFeats feature extractor

-We use the implementation at \[[SimCLR](https://github.com/sthalles/SimCLR)] but increase patch size to 132x132 pixels and use batch size of 132 for 200 epochs

-The checkpoint should be saved as 'simclr_feature_extractor.pth.tar' so that it can be read by train_seafeats.py

### Train SeaFeats end to end using our Feature Similarity Loss
-Run the script provided:

```python train_seafeats.py```

### 2. SeaCLIP
#### Generating pseudo-labels with CLIP
-We use CLIP as described in the original paper:

```
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In Int. Conf. Mach. Learn., pages 8748–8763, 2021.
```

-The script provided generates binary pseudo-labels using the following prompts:

Background: ["a photo of sand", "a photo of water", "a photo of sand or water", "a blurry photo of water", "a blurry photo of sand"]

Seagrass: ["a blurry photo of seagrass", "a photo containing some seagrass", "a photo of underwater plants", "a photo of underwater grass", "a photo of green, grass-like leaves underwater", "a photo of seagrass"]

-Run the script provided:

```python create_labels_with_clip.py```

#### Train SeaCLIP model
-Run the script provided:

```python train_seaclip.py```

<a name="datasets"></a>
## Datasets

The DeepSeagrass dataset can be downloaded at: \[[Dataset](https://data.csiro.au/collection/csiro:47653)]

The Global Wetlands test patch dataset can be downloaded at: \[[Dataset](https://doi.org/10.5281/zenodo.7659203)]

The original Global Wetlands dataset can be downloaded at: \[[Dataset](https://github.com/globalwetlands/luderick-seagrass)]

<a name="acknowledgements"></a>
## Acknowledgements
This work was done in collaboration between QUT and CSIRO's Data61. 
