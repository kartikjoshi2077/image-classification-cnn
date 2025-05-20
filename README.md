# CNN Experiments on CIFAR-10 and CIFAR-100
This repository presents two convolutional neural network (CNN) experiments focused on image classification using the CIFAR-10 and CIFAR-100 datasets. The goal is to develop a robust and optimized pipeline for image recognition using residual architectures and transfer learning strategies.

Notebooks Overview
### CNN Experiment 1.ipynb
Goal:
Train a baseline CNN architecture (with batch normalization) on CIFAR-10 and fine-tune it on CIFAR-100.

Key Features:

Standard CNN with Conv → BatchNorm → ReLU → Pool → Dropout.

EarlyStopping and ModelCheckpoint callbacks.

Trains first on CIFAR-10 for 30 epochs.

Fine-tunes on CIFAR-100 for 35 epochs (if the CIFAR-10 model is found).

Performance metrics (accuracy, precision, recall, F1-score) are computed and saved to disk.

Generates classification reports using sklearn.

### CNN Experiment 2.ipynb
Goal:
Build and train an improved 5-block ResNet architecture with better generalization, advanced data augmentation, and fine-tuning on CIFAR-100.

Key Features:

Architecture
Implements a custom 5-block residual network:

Uses pre-activation (BatchNorm → ReLU → Conv) per ResNet v2 guidelines.

Includes projection shortcuts for downsampling.

Introduces progressive Dropout and L2 regularization.

Ends with a GlobalAveragePooling and dense classification layer.

## Training Improvements
Label Smoothing in categorical cross-entropy loss.

Optimizer: SGD + Nesterov momentum (better generalization vs Adam).

Learning Rate Schedule: Warm-up phase. Cosine decay.

Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler

## Data Handling
CIFAR-10 and CIFAR-100 support. Advanced normalization using dataset mean and std.

Augmentation using: Rotation, Width/height shift, Horizontal flip, Shear, zoom, and brightness jitter

## Evaluation
Final evaluation includes: Accuracy, Classification report (macro metrics), Filtered analysis for top-20 most common classes in CIFAR-100. All summaries and metrics are saved to disk.

## Transfer Learning Workflow
Train on CIFAR-10
Use full training with batch norm and data augmentation.

Fine-tune on CIFAR-100
Load best model from CIFAR-10, replace final layer, and continue training.
