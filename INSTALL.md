# Deep Mahalanobis Detector - Windows WSL2 Installation Guide

This guide will walk you through setting up the Deep Mahalanobis Detector on Windows using WSL2 (Windows Subsystem for Linux). 

## 📋 Prerequisites

- **Windows 10** version 2004+ or **Windows 11**
- **Administrator access** to enable WSL2
- **At least 8GB RAM** (16GB recommended)
- **NVIDIA GPU** (recommended for training/faster inference)
- **10GB+ free disk space**

---

## 🔧 Part 1: WSL2 Setup

### Step 1: Enable WSL2 on Windows

1. **Open PowerShell as Administrator** and run:
   ```powershell
   # Enable WSL feature
   dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
   
   # Enable Virtual Machine Platform
   dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
   
   # Restart your computer
   ```

2. **Download and install the WSL2 Linux kernel update**:
   - Go to: https://aka.ms/wsl2kernel
   - Download and run the installer

3. **Set WSL2 as default**:
   ```powershell
   wsl --set-default-version 2
   ```

### Step 2: Install Ubuntu

1. **Install Ubuntu from Microsoft Store**:
   - Open Microsoft Store
   - Search for "Ubuntu" 
   - Install "Ubuntu" (latest LTS version)

2. **Launch Ubuntu** and complete setup:
   - Set username and password
   - Wait for initial setup to complete

3. **Verify WSL2 installation**:
   ```powershell
   # In PowerShell/Command Prompt
   wsl --list --verbose
   ```
   You should see Ubuntu running with VERSION 2.

---

## 🐍 Part 2: Python Environment Setup

### Step 1: Update Ubuntu System

```bash
# Update package lists
sudo apt update

# Upgrade system packages
sudo apt upgrade -y

# Install essential build tools
sudo apt install -y build-essential python3-pip python3-dev python3-venv git wget curl
```

### Step 2: Verify Python Installation

```bash
# Check Python version (should be 3.8+)
python3 --version

# Check pip
pip3 --version
```

---

## 📁 Part 3: Project Setup

### Step 1: Clone the Repository

```bash
# Create projects directory
mkdir -p ~/projects
cd ~/projects

# Clone the Deep Mahalanobis Detector repository
git clone [YOUR_REPO_URL] deep_mahalanobis_detector
cd deep_mahalanobis_detector
```

### Step 2: Create Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Verify activation (prompt should show (venv))
which python
```

### Step 3: Install Python Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support (if you have NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Or install CPU-only version (if no GPU)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other required packages
pip install numpy scipy scikit-learn matplotlib
```

### Step 4: Verify PyTorch Installation

```bash
# Test PyTorch installation
python -c "import torch; print(f'PyTorch {torch.__version__} - CUDA Available: {torch.cuda.is_available()}')"
```

Expected output:
- With GPU: `PyTorch 2.x.x+cu121 - CUDA Available: True`
- CPU only: `PyTorch 2.x.x+cpu - CUDA Available: False`

---

## 📊 Part 4: Data and Model Setup

### Step 1: Create Required Directories

```bash
# Create necessary directories
mkdir -p data pre_trained output checkpoints
```

### Step 2: Download Pre-trained Models

You need to obtain the pre-trained models. The original Dropbox links are no longer available, but you can:

**Option A: Find Alternative Sources**
- Check the [ODIN repository](https://github.com/facebookresearch/odin) for updated links
- Search for community-provided models on GitHub

**Option B: Use Compatible Models from Other Sources**
- Look for WideResNet models trained on CIFAR-10/100/SVHN
- Look for DenseNet models trained on CIFAR-10/100/SVHN

**Option C: Train Your Own Models (see Training section below)**

Required model files:
```
pre_trained/
├── densenet_cifar10.pth
├── densenet_cifar100.pth
├── densenet_svhn.pth
├── wideresnet_cifar10.pth (or resnet_cifar10.pth)
├── wideresnet_cifar100.pth (or resnet_cifar100.pth)
└── wideresnet_svhn.pth (or resnet_svhn.pth)
```

### Step 3: Download Datasets

The out-of-distribution datasets will be downloaded automatically, but you can pre-download them:

```bash
# Create data directory structure
mkdir -p data

# Out-of-distribution datasets (optional - will download automatically)
# These are used for testing OOD detection
```

---

## 🧪 Part 5: Testing the Installation

### Step 1: Test Basic Functionality

```bash
# Activate virtual environment (if not already active)
source venv/bin/activate

# Test help command
python OOD_Baseline_and_ODIN.py --help
```

### Step 2: Test with Sample Data (if you have models)

```bash
# Test DenseNet on CIFAR-10 (requires densenet_cifar10.pth)
python OOD_Baseline_and_ODIN.py --dataset cifar10 --net_type densenet --gpu 0

# Test WideResNet on CIFAR-10 (requires wideresnet_cifar10.pth)  
python OOD_Baseline_and_ODIN.py --dataset cifar10 --net_type wideresnet --gpu 0

# For CPU-only (no GPU)
python OOD_Baseline_and_ODIN.py --dataset cifar10 --net_type densenet --gpu -1
```

### Step 3: Test Mahalanobis Detector

```bash
# Generate Mahalanobis features
python OOD_Generate_Mahalanobis.py --dataset cifar10 --net_type densenet --gpu 0

# Train Mahalanobis detector
python OOD_Regression_Mahalanobis.py --net_type densenet
```

---

## 🏋️ Part 6: Training Your Own Models (Optional)

If you need to train models yourself:

### Step 1: Use Provided Training Scripts

```bash
# Train ResNet on CIFAR-10 (example)
python train_resnet.py --dataset cifar10 --epochs 200 --gpu 0

# Train DenseNet on CIFAR-10 (example)
python train_densenet.py --dataset cifar10 --epochs 300 --gpu 0
```

### Step 2: Use External WideResNet Training

```bash
# Clone WideResNet training repository
git clone https://github.com/AlexandrosFerles/Wide-Residual-Networks-PyTorch
cd Wide-Residual-Networks-PyTorch

# Follow their README for training on CIFAR-10, CIFAR-100, SVHN
```

---

## 🚀 Part 7: Usage Examples

### Basic Out-of-Distribution Detection

```bash
# Activate environment
source venv/bin/activate

# Run baseline OOD detection methods
python OOD_Baseline_and_ODIN.py --dataset cifar10 --net_type densenet --gpu 0

# Generate Mahalanobis features
python OOD_Generate_Mahalanobis.py --dataset cifar10 --net_type densenet --gpu 0

# Train Mahalanobis detector
python OOD_Regression_Mahalanobis.py --net_type densenet
```

### Adversarial Detection

```bash
# Generate adversarial samples
python ADV_Samples.py --dataset cifar10 --net_type densenet --adv_type FGSM --gpu 0

# Generate features for adversarial detection
python ADV_Generate_LID_Mahalanobis.py --dataset cifar10 --net_type densenet --adv_type FGSM --gpu 0

# Train adversarial detector
python ADV_Regression.py --net_type densenet
```

---

## 🛠️ Troubleshooting

### WSL2 Issues

**WSL2 not starting:**
```powershell
# In PowerShell as Administrator
wsl --shutdown
wsl --set-default-version 2
wsl --set-version Ubuntu 2
```

**Access Windows files from WSL:**
```bash
# Windows C: drive is mounted at /mnt/c/
cd /mnt/c/Users/YourUsername/
```

### Python/PyTorch Issues

**CUDA not detected:**
```bash
# Check NVIDIA drivers in Windows
# Reinstall PyTorch with CUDA support
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Module import errors:**
```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Reinstall problematic packages
pip install --upgrade numpy scipy scikit-learn
```

**Memory issues:**
```bash
# Use smaller batch sizes
python OOD_Baseline_and_ODIN.py --dataset cifar10 --net_type densenet --batch_size 32 --gpu 0

# Use CPU if GPU memory is insufficient
python OOD_Baseline_and_ODIN.py --dataset cifar10 --net_type densenet --gpu -1
```

### Model Loading Issues

**"FileNotFoundError" for models:**
```bash
# Check if model files exist
ls -la pre_trained/

# Verify file naming convention
# Files should be named: {net_type}_{dataset}.pth
# e.g., densenet_cifar10.pth, wideresnet_cifar10.pth
```

**PyTorch loading errors:**
- The codebase has been updated for PyTorch 2.x compatibility
- All `torch.load` calls include `weights_only=False` for compatibility

---

## 📂 Project Structure

After complete setup, your directory should look like:

```
deep_mahalanobis_detector/
├── data/                          # Datasets (auto-downloaded)
├── pre_trained/                   # Pre-trained models
│   ├── densenet_cifar10.pth
│   ├── densenet_cifar100.pth
│   ├── wideresnet_cifar10.pth
│   └── ...
├── models/                        # Model architectures
├── output/                        # Results and outputs
├── checkpoints/                   # Training checkpoints
├── venv/                         # Python virtual environment
├── *.py                          # Main scripts
├── INSTALL.md                    # This file
└── README.md                     # Project README
```

---

## 🎯 Quick Start Checklist

- [ ] ✅ WSL2 installed and Ubuntu running
- [ ] ✅ Python 3.8+ with pip installed
- [ ] ✅ Virtual environment created and activated
- [ ] ✅ PyTorch installed (with/without CUDA)
- [ ] ✅ Required Python packages installed
- [ ] ✅ Project repository cloned
- [ ] ✅ Directory structure created
- [ ] ✅ Pre-trained models obtained
- [ ] ✅ Help command works: `python OOD_Baseline_and_ODIN.py --help`
- [ ] ✅ Basic test runs successfully

---

## 📚 Additional Resources

### Documentation
- [WSL2 Official Documentation](https://docs.microsoft.com/en-us/windows/wsl/)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [Original Mahalanobis Paper](https://arxiv.org/abs/1807.03888)

### Related Repositories
- [ODIN Repository](https://github.com/facebookresearch/odin)
- [Wide-ResNet Training](https://github.com/AlexandrosFerles/Wide-Residual-Networks-PyTorch)
- [LID Detection](https://github.com/xingjunm/lid_adversarial_subspace_detection)

### Hardware Recommendations
- **Minimum**: 8GB RAM, CPU-only training
- **Recommended**: 16GB+ RAM, NVIDIA GPU (GTX 1660+)
- **Optimal**: 32GB RAM, RTX 3070+ or better

---

## 🤝 Contributing

If you encounter issues or have improvements:

1. **Check existing issues** in the repository
2. **Create detailed bug reports** with system information
3. **Include error logs** and reproduction steps
4. **Suggest improvements** to this installation guide

---

## 📜 License and Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{lee2018simple,
  title={A simple unified framework for detecting out-of-distribution samples and adversarial attacks},
  author={Lee, Kimin and Lee, Kibok and Lee, Honglak and Shin, Jinwoo},
  booktitle={Advances in neural information processing systems},
  pages={7167--7177},
  year={2018}
}
```

---

**Happy detecting! 🎉**

*Last updated: September 2025*
*Compatible with: Windows 10/11 + WSL2 + Ubuntu + PyTorch 2.x*
