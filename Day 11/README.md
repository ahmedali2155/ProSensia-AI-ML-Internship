# Deep Learning Baseline using PyTorch

## Project Overview

This project demonstrates the implementation of a Multi-Layer Perceptron (MLP) using PyTorch for Loan Default Prediction. The objective was to understand the complete deep learning workflow, including tensor conversion, neural network construction, manual training loops, and loss visualization.

## Dataset

Loan Default Prediction Dataset (cleaned and balanced using SMOTE).

## Model Architecture

- Input Layer
- Hidden Layer (16 neurons)
- ReLU Activation
- Hidden Layer (8 neurons)
- ReLU Activation
- Output Layer (2 classes)

## Training

- Framework: PyTorch
- Loss Function: CrossEntropyLoss
- Optimizer: Adam
- Epochs: 20

## Results

- Training loss decreased consistently during training.
- Final Test Accuracy: **64.23%**

## Conclusion

The project successfully implemented a custom neural network using PyTorch without relying on Scikit-Learn's `.fit()` method. The model learned meaningful patterns as shown by the decreasing training loss.