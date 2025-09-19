# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The Deep Mahalanobis Detector is a PyTorch implementation for detecting out-of-distribution (OOD) samples and adversarial attacks using Mahalanobis distance-based methods. This is the code for the paper "A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks" (NeurIPS 2018).

## Environment Setup

### Virtual Environment Activation
```bash
source venv/bin/activate
```

The project uses a Python virtual environment located at `venv/`. Always activate this environment before running any commands.

### Dependencies
The project requires:
- PyTorch (with/without CUDA support)
- NumPy, SciPy, scikit-learn
- Matplotlib (for visualization)
- torchvision (for datasets and transforms)

No formal requirements.txt exists - dependencies are installed manually in the virtual environment.

## Core Architecture

### Directory Structure
- `models/` - Neural network architectures (ResNet, DenseNet, WideResNet)
- `data/` - Dataset storage (auto-downloads CIFAR-10, CIFAR-100, SVHN)
- `pre_trained/` - Pre-trained model weights (.pth files)
- `output/` - Generated features and results
- `lib/` - Utility functions (not actively used)

### Key Files and Their Purposes

**Main Detection Scripts:**
- `OOD_Baseline_and_ODIN.py` - Run baseline OOD detection methods
- `OOD_Generate_Mahalanobis.py` - Generate Mahalanobis features for OOD detection
- `OOD_Regression_Mahalanobis.py` - Train logistic regression detector on Mahalanobis features

**Adversarial Detection Scripts:**
- `ADV_Samples.py` - Generate adversarial samples (FGSM, PGD, etc.)
- `ADV_Generate_LID_Mahalanobis.py` - Generate LID and Mahalanobis features for adversarial detection
- `ADV_Regression.py` - Train detectors for adversarial samples

**Core Libraries:**
- `lib_generation.py` - Feature extraction functions (Mahalanobis distance, LID calculation, sample estimators)
- `lib_regression.py` - Data splitting and regression utilities for training detectors
- `data_loader.py` - Dataset loading utilities for CIFAR-10, CIFAR-100, SVHN
- `calculate_log.py` - Logging and evaluation utilities

**Model Training:**
- `train_resnet.py` - Train ResNet models from scratch
- `train_densenet.py` - Train DenseNet models from scratch
- `train_utils.py` - Training utilities and helper functions

### Neural Network Models
The project supports three main architectures:
1. **ResNet** (`models/resnet.py`) - Standard ResNet implementation
2. **DenseNet** (`models/densenet.py`) - DenseNet implementation
3. **WideResNet** (`models/wideresnet.py`) - Wide ResNet implementation

All models expect 32x32 RGB input images and support CIFAR-10 (10 classes), CIFAR-100 (100 classes), and SVHN (10 classes).

## Common Development Workflows

### Running OOD Detection
```bash
# Activate environment first
source venv/bin/activate

# Run baseline methods (requires pre-trained models)
python OOD_Baseline_and_ODIN.py --dataset cifar10 --net_type resnet --gpu 0

# Generate Mahalanobis features
python OOD_Generate_Mahalanobis.py --dataset cifar10 --net_type resnet --gpu 0

# Train Mahalanobis detector
python OOD_Regression_Mahalanobis.py --net_type resnet
```

### Running Adversarial Detection
```bash
# Generate adversarial samples
python ADV_Samples.py --dataset cifar10 --net_type resnet --adv_type FGSM --gpu 0

# Generate detection features
python ADV_Generate_LID_Mahalanobis.py --dataset cifar10 --net_type resnet --adv_type FGSM --gpu 0

# Train adversarial detector
python ADV_Regression.py --net_type resnet
```

### Training Models from Scratch
```bash
# Train ResNet (2-4 hours on GPU)
python train_resnet.py --dataset cifar10 --epochs 200

# Train DenseNet (4-6 hours on GPU)
python train_densenet.py --dataset cifar10 --epochs 300
```

### GPU vs CPU Usage
- Use `--gpu 0` for GPU acceleration (recommended)
- Use `--gpu -1` for CPU-only execution (slower)

## Key Implementation Details

### Feature Extraction Pipeline
1. **Sample Estimator** (`lib_generation.py:sample_estimator`) - Computes class-conditional means and precision matrices from training data
2. **Mahalanobis Distance** (`lib_generation.py:get_Mahalanobis_score`) - Calculates confidence scores using Mahalanobis distance
3. **LID Calculation** (`lib_generation.py:mle_batch`) - Computes Local Intrinsic Dimensionality for adversarial detection

### Model Loading Convention
Pre-trained models follow the naming pattern: `{net_type}_{dataset}.pth`
- Example: `resnet_cifar10.pth`, `densenet_cifar100.pth`, `wideresnet_svhn.pth`

### Dataset Handling
- CIFAR-10/100 and SVHN datasets are automatically downloaded to `data/` on first use
- Out-of-distribution test datasets (Tiny-ImageNet, LSUN) must be manually downloaded
- The codebase has been updated for PyTorch 2.x compatibility with `weights_only=False` in `torch.load()` calls

## PyTorch Compatibility Notes

The codebase has been modernized for PyTorch 2.x:
- All `torch.load()` calls include `weights_only=False` for backward compatibility
- Model architectures are compatible with modern PyTorch versions
- CUDA support works with current PyTorch installations

## No Testing Infrastructure

This research codebase does not include formal testing, linting, or build processes. Verification is done by:
1. Running help commands: `python OOD_Baseline_and_ODIN.py --help`
2. Testing basic functionality with sample runs
3. Monitoring output in `output/` directory for generated features and results