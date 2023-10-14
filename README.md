# This repository contains a Jupyter Notebook that demonstrates an anomaly detection model available in `anomalib` library. Namely: PaDiM: A Patch Distribution Modeling Framework for Anomaly Detection in MVTec dataset. This method is implemented using the anomalib library in Python.

## Image Anomaly Detection

Detecting anomalies in images is a popular application of anomaly detection.
In this tutorial, we will go over a popular dataset known as the "[MVTec Anomaly Detection](https://www.mvtec.com/company/research/datasets/mvtec-ad)" dataset. MVTec is a dataset for benchmarking anomaly detection methods with a focus on industrial inspection.

Here are some features of the dataset:

*   Contains over 5000 high-resolution images.
*   Images are divided into fifteen distinct object and texture categories such as bottle, cable, carpet, wood, leather and pill.
*   Each category consists of two sets: training images (defect-free) and test images (with defects and without defects).


Some example of the MVTec dataset:
<br>
<img width="1042" alt="Screen Shot 2023-10-14 at 1 50 42 PM" src="https://github.com/sarehsoltani/Image-Anomaly-Detection/assets/23232055/0c335e83-21be-4d6c-aaa5-799ca81c7dcb">

## **Anomalib**:
Anomalib is a deep learning library for developing and deploying state-of-the-art anomaly detection algorithms for benchmarking on both public and private datasets.


# Key features:

*   The largest public collection of ready-to-use deep learning anomaly detection algorithms and benchmark datasets.
*   Provides a set of tools that facilitate the development and implementation of Anomaly Detection models.
*   Focus on image-based anomaly detection or anomalous pixel regions within images in a dataset.

## Model

Currently, there are **13** anomaly detection models available in `anomalib` library. Namely,

*   [CFA](https://github.com/openvinotoolkit/anomalib/tree/main/src/anomalib/models/cfa)
*   [PADIM](https://github.com/openvinotoolkit/anomalib/tree/main/src/anomalib/models/padim)
*   [CFlow](https://github.com/openvinotoolkit/anomalib/tree/main/src/anomalib/models/cflow)
*   [DFKDE](https://github.com/openvinotoolkit/anomalib/tree/main/src/anomalib/models/dfkde)
*   [DFM](https://github.com/openvinotoolkit/anomalib/tree/main/src/anomalib/models/dfm)
*   [DRAEM](https://github.com/openvinotoolkit/anomalib/tree/main/src/anomalib/models/draem)
*   [EfficientAd](https://github.com/openvinotoolkit/anomalib/tree/main/src/anomalib/models/efficient_ad)
*   [FastFlow](https://github.com/openvinotoolkit/anomalib/tree/main/src/anomalib/models/fastflow)
*   [GANomaly](https://github.com/openvinotoolkit/anomalib/tree/main/src/anomalib/models/ganomaly)
*   [PatchCore](https://github.com/openvinotoolkit/anomalib/tree/main/src/anomalib/models/patchcore)
*   [Reverse Distillation](https://github.com/openvinotoolkit/anomalib/tree/main/src/anomalib/models/reverse_distillation)
*   [STFPM](https://github.com/openvinotoolkit/anomalib/tree/main/src/anomalib/models/stfpm)

### In this demo, we'll be using Padim.  

# PaDiM: A Patch Distribution Modeling Framework for Anomaly Detection


* Patch-based algorithm relying on pre-trained CNN feature extractor.
* Breaks image into patches, extracts embeddings using different layers.
* Concatenates activation vectors for diverse semantic levels.
* Encodes fine-grained and global contexts in embeddings.
* Reduces dimensions of embedding vectors to mitigate redundancy.
* Generates multivariate Gaussian distribution for each patch embedding.
* Distribution calculated across entire training batch.
* Inference uses Mahalanobis distance to score test image patches.
* Uses inverse of covariance matrix from training for Mahalanobis distance.
* Anomaly map formed from Mahalanobis distance scores.
* Higher scores in anomaly map indicate anomalous regions.


  


