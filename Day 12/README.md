# Day 12 - Deep Learning Baseline using PyTorch DataLoader

## Project Overview

This project upgrades the previous PyTorch MLP implementation by introducing TensorDataset, DataLoader, mini-batch training, validation, and the Adam optimizer.

## Dataset

Loan Default Prediction Dataset (cleaned, encoded, and balanced using SMOTE from the previous project).

## Model Architecture

- Input Layer
- Hidden Layer (16 neurons)
- ReLU Activation
- Hidden Layer (8 neurons)
- ReLU Activation
- Output Layer (2 classes)

## Technologies

- PyTorch
- TensorDataset
- DataLoader
- Adam Optimizer
- CrossEntropyLoss

## Training Configuration

- Batch Size: 64
- Epochs: 25
- Learning Rate: 0.001
- Optimizer: Adam

## Results

- Validation Accuracy: **41.46%**
- Training and validation loss were tracked throughout all epochs.
- The project demonstrates mini-batch deep learning using PyTorch.

## Conclusion

The project successfully upgraded a basic PyTorch model into a mini-batch training pipeline using TensorDataset and DataLoader. Validation loss tracking provided insight into the model's generalization during training.`



# Day 13 - Regularized Neural Network using PyTorch

## Objective

The objective of this assignment is to improve the baseline neural network by applying regularization techniques to achieve more stable training and better generalization.

## Dataset

- Loan Default Prediction Dataset
- File: train_ctrUa4K.csv

## Techniques Used

- PyTorch
- TensorDataset
- DataLoader
- Mini-batch Training
- Neural Network (MLP)
- Batch Normalization
- Dropout (0.3)
- Adam Optimizer
- CrossEntropyLoss

## Models

### Baseline Model

- Linear (Input → 16)
- ReLU
- Linear (16 → 8)
- ReLU
- Linear (8 → 2)

### Regularized Model

- Linear (Input → 16)
- BatchNorm1d
- ReLU
- Dropout (0.3)
- Linear (16 → 8)
- BatchNorm1d
- ReLU
- Dropout (0.3)
- Linear (8 → 2)

## Results

The baseline model showed unstable validation loss with larger fluctuations.

The regularized model produced smoother training and validation curves, demonstrating more stable learning and improved generalization.

## Learning Outcomes

- Implemented Batch Normalization
- Implemented Dropout Regularization
- Compared baseline and regularized neural networks
- Visualized training and validation losses
- Understood Gamma (γ), Beta (β), and Inverted Dropout Scaling