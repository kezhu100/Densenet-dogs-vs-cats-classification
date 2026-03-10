# DenseNet Dogs vs Cats Image Classification

This project implements an image classification system based on **DenseNet** using **TensorFlow/Keras**.  
The goal of the project is to build a deep learning model that can distinguish between **cat** and **dog** images.

This project was developed as part of the course:

**EE6483 Artificial Intelligence and Data Mining – NTU**

---

# Project Overview

The project implements a full deep learning pipeline including:

- Image preprocessing
- Data augmentation
- CNN model training (DenseNet)
- Model evaluation
- Visualization of training results

The model is trained to classify images into two categories:


Cat
Dog


---

# Project Structure


densenet-dog-cat-classification
│
├ README.md
├ requirements.txt
│
├ augmentation.ipynb
├ demo_cifar10.ipynb
├ demo_dogcat.ipynb
└ tool.py


File descriptions:

| File | Description |
|-----|-------------|
| `augmentation.ipynb` | Experiments for image data augmentation |
| `demo_cifar10.ipynb` | Testing the CNN architecture using the CIFAR10 dataset |
| `demo_dogcat.ipynb` | Main notebook for training the Dogs vs Cats classification model |
| `tool.py` | Utility functions for loading data, visualization, and evaluation |

---

# Machine Learning Pipeline

The project follows a typical deep learning workflow:


Dataset
↓
Image preprocessing
↓
Data augmentation
↓
DenseNet model training
↓
Model evaluation
↓
Visualization


Key evaluation outputs include:

- Training accuracy
- Validation accuracy
- Loss curve
- Confusion matrix
- Classification report

---

# Technologies Used

- Python 3.9
- TensorFlow / Keras
- NumPy
- OpenCV
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

# Installation

Install the required dependencies:


pip install -r requirements.txt


---

# How to Run

1. Install dependencies


pip install -r requirements.txt


2. Open the notebook


demo_dogcat.ipynb


3. Run all cells to train the model.

---

# Results

The trained model can effectively classify **cats and dogs** images.

Model evaluation includes:

- Accuracy curve
- Loss curve
- Confusion matrix
- Classification metrics (precision, recall, F1-score)

---

# Author

**Wuhaotian**  **Kezhu**  **Lijiaqing**

MSc Signal Processing and Machine Learning  
Nanyang Technological University