# HDC-SEResNet for Gambung Tea Leaf Classification

This repository contains the code and resources for the "HDC-SEResNet for Gambung Tea Leaf Classification" project, developed as part of a B.Tech pre-final year research project. The project focuses on developing a novel Convolutional Neural Network (CNN) architecture for accurately classifying Gambung tea leaves into five distinct classes.

## Project Overview

The primary goal of this research is to develop and evaluate a robust CNN architecture tailored for tea leaf classification, particularly addressing the challenge of limited initial datasets. The proposed HDC-SEResNet model integrates Hybrid Dilated Convolutions (HDC) for multi-scale feature extraction and Squeeze-and-Excitation (SE) mechanisms for adaptive channel recalibration within a residual learning framework. This approach aims to enhance classification accuracy for practical agricultural applications like quality control and variety identification.

## Key Features

*   **Novel CNN Architecture (HDC-SEResNet):** Custom-designed for enhanced feature representation.
*   **Multi-Scale Feature Extraction:** Utilizes Hybrid Dilated Convolutions.
*   **Adaptive Channel Recalibration:** Employs Squeeze-and-Excitation blocks.
*   **Data Augmentation:** Extensive offline and online augmentation to overcome limited data.
*   **High Classification Accuracy:** Achieved **96.92%** validation accuracy on 5 tea leaf classes.
*   **Comparative Analysis:** Benchmarked against several state-of-the-art CNN models.

## Architecture: HDC-SEResNet

The HDC-SEResNet architecture is built upon a standard residual network backbone, modified with custom HDC-SE-ResBlocks.

*   **Input Layer:** Standardized RGB images (224x224x3).
*   **Stem Block:** Initial Convolution, Batch Normalization, ReLU, Max Pooling.
*   **Stacked HDC-SE-ResBlocks:**
    *   **Hybrid Dilated Convolutions (HDC):** Parallel 3x3 convolutions with different dilation rates (1, 2, 3) followed by concatenation and a 1x1 convolution for feature fusion.
    *   **Squeeze-and-Excitation (SE):** Global Average Pooling (Squeeze) followed by two Fully Connected layers (Excitation) to recalibrate channel-wise feature responses.
    *   **Residual Connection:** Standard shortcut connection to facilitate training of deep networks.
*   **Classification Head:** Global Average Pooling, Dropout, Fully Connected layer, Softmax.


## Dataset

*   **Source:** [Gambung Tea Leaf Dataset on Kaggle](https://www.kaggle.com/datasets/cendikiawan/dauntehgmb) by Cendikiawan.
*   **Classes:** 5 distinct tea leaf clones (GMB_01 to GMB_05).
*   **Initial Size:** Approximately 600 images.
*   **Augmented Size:** Approximately 6,000 images after offline augmentation (rotations, scaling, reflections, translations).
*   **Splits:** 90% Training (~5,400 images), 10% Validation (~600 images).

## Methodology

### Data Preparation & Augmentation
1.  **Offline Augmentation:** Each of the ~600 original images was transformed ~10 times using random rotations, scaling, horizontal/vertical reflections, and translations.
2.  **Online Augmentation:** During training, MATLAB's `imageDataAugmenter` was used for random reflections, rotations (±15°), scaling (0.8-1.2x), and translations (±30 pixels) on training batches.
3.  **Normalization:** Z-score normalization was applied to input images.

### Model Training
*   **Framework:** MATLAB Deep Learning Toolbox
*   **Optimizer:** Adam
*   **Initial Learning Rate:** 1e-3 (with piecewise decay)
*   **Mini-Batch Size:** 32
*   **Max Epochs:** 30 (with early stopping based on validation performance)
*   **Hardware:** GPU accelerated training.

## Results

*   **Overall Validation Accuracy:** **96.92%**
*   **ROC AUC Scores:** High AUC values ranging from 0.97 to 0.99 across all classes.
*   **Comparative Performance:** The HDC-SEResNet outperformed ResNet-50, ResNet-101, DenseNet-201, Inception V3, Inception-ResNet V2, and Xception on the augmented dataset.
  
## Technologies Used

*   **Primary Language & Framework:** MATLAB (Deep Learning Toolbox)
*   **Core Concepts:** Deep Learning, Computer Vision, Convolutional Neural Networks (CNNs), Residual Networks (ResNet), Hybrid Dilated Convolutions (HDC), Squeeze-and-Excitation (SE) Networks, Data Augmentation.
*   **Data Handling:** Image Processing

## Setup and Usage


1.  **Prerequisites:**
    *   MATLAB (Version R20XXx or later)
    *   Deep Learning Toolbox
    *   Image Processing Toolbox (if used for pre-processing)
    *   (Any other specific toolboxes or libraries)

2.  **Clone the repository:**
    ```bash
    git clone https://github.com/4xush/hdc-seresnet-tea-leaf-classification/
    ```

3.  **Dataset:**
    *   Download the [Gambung Tea Leaf Dataset](https://www.kaggle.com/datasets/cendikiawan/dauntehgmb).
    *   Place the dataset in a directory (e.g., `./dataset/`) and structure it as expected by the scripts (e.g., separate folders for each class within `train` and `validation` subdirectories if you pre-processed it this way).
    *   *Alternatively, describe if your scripts handle downloading or if the augmented dataset is provided/generated by a script.*
    *   

## Future Work

*   Investigate misclassifications using techniques like CAM/Grad-CAM.
*   Optimize HDC-SE block parameters.
*   Explore advanced augmentation (e.g., GANs).
*   Evaluate transfer learning approaches.
*   Develop a mobile/edge deployment version.

## Acknowledgements

*   Dr. Mourina Ghosh (Supervisor)
*   Indian Institute of Information Technology Guwahati
*   Cendikiawan for the [Daun Teh GMB Dataset on Kaggle](https://www.kaggle.com/datasets/cendikiawan/dauntehgmb).
*   Authors of cited papers (ResNet, SE-Nets, HDC).
