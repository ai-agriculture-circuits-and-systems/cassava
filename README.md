# Cassava Leaf Disease Classification

## Overview
This project focuses on developing a machine learning model to classify diseases in Cassava plants using leaf images. Cassava is a crucial food security crop in Africa, primarily grown by small-holder farmers. The goal is to help farmers quickly identify plant diseases to prevent crop loss and ensure food security.

## Dataset
The dataset consists of leaf images from the Cassava plant, showing both healthy leaves and four different disease conditions:
- Cassava Mosaic Disease (CMD)
- Cassava Bacterial Blight (CBB)
- Cassava Green Mite (CGM)
- Cassava Brown Streak Disease (CBSD)

### Dataset Statistics
- Total Images: 9,430 labeled images
- Training Set: 5,656 images
- Test Set: 1,885 images
- Validation Set: 1,889 images

Note: The dataset is imbalanced, with CMD and CBSD classes comprising 72% of the images.

### Data Structure
```
data/
├── train/     # Training images
└── test/      # Test images
```

## Project Structure
```
.
├── data/          # Dataset directory
├── imgs/          # Project images and visualizations
├── pdata/         # Additional data resources
└── README.md      # Project documentation
```

## Challenge Description
The primary task is a 5-class classification problem to distinguish between:
1. Healthy leaves
2. Cassava Mosaic Disease (CMD)
3. Cassava Green Mite disease (CGM)
4. Cassava Bacterial Blight (CBB)
5. Cassava Brown Streak Disease (CBSD)

### Extended Challenge
A secondary task involves predicting both disease incidence and severity levels. Each disease has severity levels scored from 1-5:
- Level 1: Healthy leaf
- Level 5: Severely infected plant

## Getting Started
1. Clone the repository
2. Download the dataset from [Kaggle](https://www.kaggle.com/c/cassava-disease/data)
3. Extract the dataset to the `data/` directory

## Dataset Source
This dataset is available through:
- [Kaggle Competition](https://www.kaggle.com/c/cassava-disease/data)
- [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/cassava)

## License
This project is licensed under the terms included in the LICENSE file.

## Acknowledgments
- Dataset provided by the Cassava Disease Classification Challenge
- TensorFlow Datasets for providing easy access to the dataset
- Kaggle for hosting the competition and dataset
