# harmonization

It contains the code for every harmonization experiments done during the internship.

extract is for feature extraction, it use a dataset file in a typical json format to load the volumes. The "image" key of an item in the dataset file should contain the path to the volume.

train is for training a model, it use a dataset file in a typical json format to load the volumes. The "image" key of an item in the dataset file should contain the path to the volume and the "roi_label" key should contain the label of the roi if training with fixed rois.

the "info" key of an item in the dataset file should contain the information of the volume useful for creating the feature set file. (like the manufacturer, the slice thickness, the spacing between slices, etc.)

# analyze

It contains the code for every analysis done during the internship.

analyze is the code used to produce the tsne and pca plots with the features sets.

batch_classif is used to run different classification experiments using the code from "classification"

# qa4iqi-extraction

the code is coming from the original repository of the QA4IQI project. It is used to extract the radiomics features from the phantom scans. a few lines were modified so it might still be of use for the project.

## Code for QA4IQI Radiomics Feature Extraction

1. (OPTIONAL) To build this container from source, navigate to this directory and run:

   ```
   docker build -t medgift/qa4iqi-extraction:latest .
   ```

2. Run a container using the **medgift/qa4iqi-extraction** image (built locally or pulled from Docker Hub directly)

   ```
   docker run -it --rm -v <PATH_TO_DATASET_FOLDER>:/data/ct-phantom4radiomics -v <PATH_TO_OUTPUT_FOLDER>:/data/output medgift/qa4iqi-extraction:latest
   ```

   Where:
   - ```<PATH_TO_DATASET_FOLDER>``` is an empty folder on your local hard drive where the dataset will be downloaded to. Make sure to have enough space available on your hard drive, as the full dataset is ~42GB.
   - ```<PATH_TO_OUTPUT_FOLDER>``` is an empty folder on your local hard drive where the extracted features will be saved in a file called **features.csv**.

## Phantom Scans Dataset

The dataset of phantom scans is available on TCIA here: https://doi.org/10.7937/a1v1-rc66

If you publish any work which uses this dataset, please cite the following publication :

> *Schaer, R., Bach, M., Obmann, M., Flouris, K., Konukoglu, E., Stieltjes, B., Müller, H., Aberle, C., Jimenez del Toro, O. A., & Depeursinge, A. (2023). Task-Based Anthropomorphic CT Phantom for Radiomics Stability and Discriminatory Power Analyses (CT-Phantom4Radiomics)*

## Reference images

The QA4IQI reference data (used for printing the phantom) can be downloaded here: https://www.dropbox.com/s/yf5cqprkyuxwcwv/refData.zip?dl=0