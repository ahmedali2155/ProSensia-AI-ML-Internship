# Deep Learning Baseline using PyTorch DataLoader

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