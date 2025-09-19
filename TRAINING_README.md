# Training Your Own Models

## Quick Start Commands

### ResNet Models
```bash
source venv/bin/activate

# Train ResNet models (2-4 hours each on GPU)
python train_resnet.py --dataset cifar10 --epochs 200    # Creates: pre_trained/resnet_cifar10.pth
python train_resnet.py --dataset cifar100 --epochs 200   # Creates: pre_trained/resnet_cifar100.pth  
python train_resnet.py --dataset svhn --epochs 160       # Creates: pre_trained/resnet_svhn.pth
```

### DenseNet Models  
```bash
source venv/bin/activate

# Train DenseNet models (4-6 hours each on GPU)
python train_densenet.py --dataset cifar10 --epochs 300   # Creates: pre_trained/densenet_cifar10.pth
python train_densenet.py --dataset cifar100 --epochs 300  # Creates: pre_trained/densenet_cifar100.pth
python train_densenet.py --dataset svhn --epochs 200      # Creates: pre_trained/densenet_svhn.pth
```

## Expected Results
- CIFAR-10: ResNet ~95%, DenseNet ~96%  
- CIFAR-100: ResNet ~77%, DenseNet ~80%
- SVHN: ResNet ~96%, DenseNet ~97%

## Hardware Requirements
- **GPU**: Highly recommended (10-20x faster than CPU)
- **Time**: ~16-24 hours total for all 6 models
- **Storage**: ~12GB for models and checkpoints

## After Training
Once complete, you'll have all 6 required model files:
- `pre_trained/resnet_cifar10.pth`
- `pre_trained/resnet_cifar100.pth`
- `pre_trained/resnet_svhn.pth`
- `pre_trained/densenet_cifar10.pth`
- `pre_trained/densenet_cifar100.pth`
- `pre_trained/densenet_svhn.pth`

Then you can use the original Mahalanobis detector scripts!
