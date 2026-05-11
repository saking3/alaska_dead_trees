# Identifying Dead Trees in High-Resolution Aerial Imagery

Deep learning algorithms for automated detection of trees damaged by spruce bark beetles in Alaskan forests.

---

## Overview

This project benchmarks five computer vision models (two foundation models and three CNN-based architectures) for binary segmentation of dead trees in high-resolution aerial imagery, with a focus on understanding how model performance scales with training set size.

## Sample Area
Kenai Peninsula and area near Nancy Lake State Recreation Area, Alaska.
<img width="432" height="344" alt="samplearea" src="https://github.com/user-attachments/assets/aa0de040-cdd6-4a97-9ac8-a8b874dc9d35" />

KMZ files containing the flight paths and an Jupyter notebook that uses geopandas to view the flight paths against a map are contained within the flight_path_kmz folder. 

## Methodology

### Lebelling Pipeline

Labeled masks were generated using a K-means clustering procedure augmented with Gray Level Co-occurrence Matrix (GLCM) texture features on the red band. 60 clusters were identified per image, then each class was manually reclassed into binary masks using QGIS (1 = dead tree, 0 = background).

Several automated labeling approaches were evaluated:
- `segment-geospatial` Python package — insufficient precision
- SAM zero-shot auto-segmentation — failed to distinguish dead trees from background reliably

### Dataset Construction

7 aerial images were tiled into 256×256 pixel patches using the `patchify` package. Empty tiles (containing only 0 or 1 values) were removed. Final split:
- Training: 5 images
- Validation: 1 image
- Test: 1 image

Training subsets of 10, 100, 500, 1000, and 5000 tiles were used to evaluate data scaling behavior. Full dataset available on [HuggingFace](https://huggingface.co/datasets/saking3/alaska_dead_trees).

### Models

All models were trained with fully unfrozen parameters using transfer learning. Models evaluated:

| Model | Description |
|---|---|
| **DINOv2** | Vision Transformer foundation model; linear segmentation head added for binary prediction |
| **SAM** | Segment Anything Model; fine-tuned for binary segmentation task |
| **ResNet-152 (ImageNet)** | 152-layer residual network; initialized with ImageNet weights, final layers modified for binary output |
| **ResNet-152 (no pretraining)** | Same architecture; randomly initialized weights |
| **CNN-53** | Custom 53-layer CNN: initial preprocessing block (conv + batch norm + ReLU + pooling) followed by 16 blocks of 3 convolutional layers each, with final conv + interpolation layer for binary output |

### Repository Structure

**Dataset creation and preprocessing**
- `BinaryMaskNoiseRemoval.ipynb` — mask postprocessing
- `HuggingFaceDatasetCreation.ipynb` — dataset preparation and upload

**Model training**
- `DINOv2Model.ipynb`
- `SAMModel.ipynb`
- `Resnet152model.ipynb`
- `RAWResnet152model.ipynb`
- `CNN50.ipynb`

**Full-image inference**
- `DINOv2_PredictEntireImage.ipynb`
- `SAM_PredictEntireImage.ipynb`

> **Note:** The HuggingFace dataset excludes empty tiles (tiles containing only 0 or 1 values). To run predictions on a complete aerial image and stitch results back into a single output, use the full-image inference notebooks above rather than the standard model notebooks.

**Flight path data**
- `flight_path_kmz/` — KMZ files for flight paths + a geopandas visualization notebook

## Environment Setup
Can pip install packages as needed. 

---

## Results

### Total Accuracy

| Training Set Size | SAM | DINOv2 | ResNet-152 (ImageNet) | ResNet-152 (no pretrain) | CNN-53 |
|---|---|---|---|---|---|
| 10 | 0.691 | **0.965** | 0.932 | 0.704 | 0.841 |
| 100 | 0.901 | **0.974** | 0.895 | 0.959 | 0.945 |
| 500 | 0.858 | **0.974** | 0.956 | 0.967 | 0.962 |
| 1000 | 0.856 | **0.974** | 0.960 | 0.964 | 0.970 |
| 5000 | 0.849 | **0.975** | 0.900 | 0.971 | 0.970 |

### Mean IoU

| Training Set Size | SAM | DINOv2 | ResNet-152 (ImageNet) | ResNet-152 (no pretrain) | CNN-53 |
|---|---|---|---|---|---|
| 10 | 0.029 | **0.472** | 0.000 | 0.141 | 0.027 |
| 100 | 0.044 | **0.589** | 0.202 | 0.348 | 0.378 |
| 500 | 0.086 | **0.547** | 0.366 | 0.462 | 0.386 |
| 1000 | 0.143 | **0.557** | 0.422 | 0.436 | 0.448 |
| 5000 | 0.155 | **0.597** | 0.345 | 0.525 | 0.532 |

## Key Findings

- **DINOv2 is the strongest performer overall**, leading on accuracy, mean IoU, and true positive/negative rates across nearly all training set sizes.

- **DINOv2 converges early.** Performance stabilizes at ~100 training images (accuracy ~0.974, mean IoU ~0.547–0.597), with minimal gains at larger dataset sizes. This makes it  practical for remote sensing applications where labeled data is limited.

- **ImageNet pretraining hurt ResNet-152** The randomly initialized ResNet-152 outperformed the ImageNet-pretrained version across most conditions, suggesting that ImageNet weights are a poor initialization for aerial imagery of evergreen canopy.

- **SAM underperforms.** Even at 5,000 training images, SAM achieves only 0.155 mean IoU, well below all other models. Its architecture appears ill-suited to this fine-grained binary segmentation task after fine-tuning.

- **CNN-based models improve predictably with data.** ResNet-152 (w/o pretraining) and CNN-53 both trend upward with dataset size and converge near DINOv2 performance at 5,000 images.


## Future Work

- **Diversify training data geographically.** Current training images are from a limited spatial extent. Adding imagery from sites with different lighting conditions, terrain, and beetle damage stages would improve model robustness and generalizability.

- **Evaluate SAM2.** SAM2 was released after this study was conducted and would likely outperform the original SAM — worth a direct comparison.

- **Expand to multi-class damage staging.** Current task is binary (dead/not dead). A multi-class approach distinguishing early, mid, and late-stage beetle damage would be more operationally useful for forest management applications.

## General Workflow description
**Dataset creation/preprocessing:** BinaryMaskNoiseRemoval.ipynb, HuggingFaceDatasetCreation.ipynb

**Models:** Contained within DINOv2Model.ipynb, SAMModel.ipynb, Resnet152model.ipynb, RAWResnet152model.ipynb, and CNN50.ipynb.

**Running predictions on an entire image (vs a single tile):** The huggingface dataset has had all "blank" tiles (tiles with only 0 or 1 values) removed, so if you would like to tile up an whole image (including empty tiles), run predictions on it, then have it stitched back into a single image, you will need to follow the protocols in: DINOv2_PredictEntireImage.ipynb and SAM_PredictEntireImage.ipynb  

---

## References
[SAM Model Reference](https://github.com/bnsreenu/python_for_microscopists/blob/master/331_fine_tune_SAM_mito.ipynb)

[DINOv2 Model Reference](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DINOv2/Train_a_linear_classifier_on_top_of_DINOv2_for_semantic_segmentation.ipynb)

[ResNet-152 Model](https://huggingface.co/microsoft/resnet-152)
