# Impact of CT dose on AI performance

This repository contains the code used to evaluate how CT dose variations affect the stability of radiomic and deep learning features, and their impact on AI model performance. Using a standardized multi-scanner dataset from a 3D-printed anthropomorphic phantom with liver anomalies, we extracted features from six regions of interest using PyRadiomics, a shallow CNN, SwinUNETR, and a CT foundation model (CT-FM). We assessed feature robustness using intraclass correlation (ICC), visualized embeddings with UMAP, and evaluated generalization across dose levels in classification tasks.

## analyze

Contains all scripts related to downstream analysis and evaluation:

- Classification: Code for training and evaluating classification models using extracted features.

- UMAP: Scripts to generate 2D visualizations of feature distributions.

- Boxplots: Tools to analyze and visualize feature robustness (e.g., ICC).

- Metadata processing: Utilities to handle acquisition metadata and structure datasets.

## radiomics

Radiomics feature extraction using PyRadiomics.

- NIfTI mask generation from DICOM segmentations (nifti_masks_generation.py)

- Feature extraction from full ROI segmentations (pyradiomics_extraction.py, extract_pyradiomics_features.py)

- Patch-level feature extraction (10 patches per ROI)

- Extraction for CT-ORG dataset

## cnn

Deep learning–based feature extraction using a pretrained CNN model.

- Main code for patch-based feature extraction (patches of size 64×64×32, centered on ROI coordinates)

- Scripts for extracting features from CT-ORG and MNIST datasets

- Patch-level feature extraction (10 patches per ROI)

## swin

Transformer-based feature extraction using a pretrained Swin Transformer (SwinUNETR).

- Main code for patch-based feature extraction (patches of size 64×64×32, centered on ROI coordinates)

- Scripts for extracting features from CT-ORG and MNIST datasets

## ct-fm

Feature extraction using a pretrained CT-FM model.

- Main code for patch-based feature extraction (patches of size 64×64×32, centered on ROI coordinates)

- Scripts for extracting features from CT-ORG and MNIST datasets


## Phantom Scans Dataset

The dataset of phantom scans is available on TCIA here:

> *Amirian, M., Bach, M., Jimenez del Toro, O. A., Aberle, C., Schaer, R., Andrearczyk, V., Maestrati, J.-F., Flouris, K., Obmann, M., Dromain, C., Dufour, B., Poletti, P.-A., von Tengg-Kobligk, H., Alkadhi, H., Konukoglu, E., Müller, H., Stieltjes, B., & Depeursinge, A. (2025). A Multi-Centric Anthropomorphic 3D CT Phantom-Based Benchmark Dataset for Harmonization (CT4Harmonization-Multicentric) (Version 1) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/M0PB-BH69*

If you publish any work which uses this dataset, please cite the following publication:

> *Amirian, M., Bach, M., Jimenez del Toro, O. A., Aberle, C., Schaer, R., Andrearczyk, V., Maestrati, J.-F., Martin Asiain, M., Flouris, K., Obmann, M., Dromain, C., Dufour, B., Poletti, P.-A., von Tengg-Kobligk, H., Hügli, R., Kretzschmar, M., Alkadhi, H., Konukoglu, E., Müller, H., Stieltjes, B., & Depeursinge, A. (2025). A Multi-Centric Anthropomorphic 3D CT Phantom-Based Benchmark Dataset for Harmonization, under submission, 2025.*



