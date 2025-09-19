# Today's To-Do for Deep Mahalanobis Detector Research Project

Generated on: 2025-09-19

## Project Status
Working on undergraduate engineering research project to understand and adapt the Deep Mahalanobis Detector codebase for detecting out-of-distribution samples and adversarial attacks.

## Key Questions Answered

### 1. What does "inducing" a generative classifier mean?
The paper converts a discriminative neural network (e.g., ResNet, DenseNet) into a generative model by:
- Extracting intermediate layer features from training data
- Computing class-conditional mean vectors for each class
- Estimating class-conditional covariance matrices (precision matrices)
- This creates Gaussian distributions in feature space for each class

**Implementation:** See `lib_generation.py:sample_estimator()` function (lines 45-124)

### 2. How is Mahalanobis distance used for detection?
For a test sample:
- Extract features from the same intermediate layers
- Compute Mahalanobis distance to each class distribution
- Take the minimum distance across all classes
- If distance > threshold → sample is OOD or adversarial

**Implementation:** See `lib_generation.py:get_Mahalanobis_score()` function (lines 126-200)

### 3. CPU Compatibility Status
- **Training scripts**: Already support CPU fallback (`train_resnet.py`, `train_densenet.py`)
- **Detection scripts**: Currently CUDA-only, need modification
- **Core issue**: Hardcoded `.cuda()` calls and device selection

## Phase 1: Understanding the Framework ✅

### Completed Today:
- [x] Read and understand project plan
- [x] Analyzed codebase structure and key files
- [x] Understood the "generative classifier induction" process
- [x] Mapped out Mahalanobis distance implementation
- [x] Identified CPU compatibility issues

## Phase 2: CPU Compatibility Adaptation (Next Priority)

### Immediate Tasks:
1. **Modify OOD_Baseline_and_ODIN.py for CPU support**
   - Replace hardcoded `torch.cuda.set_device(args.gpu)`
   - Add device selection logic: `device = torch.device('cuda:X' if cuda_available else 'cpu')`
   - Replace `model.cuda()` with `model.to(device)`

2. **Update OOD_Generate_Mahalanobis.py**
   - Same device handling modifications
   - Update data loader device placement

3. **Modify lib_generation.py for device flexibility**
   - Make `sample_estimator()` device-agnostic
   - Update `get_Mahalanobis_score()` tensor operations
   - Replace hardcoded `.cuda()` calls with device parameter

4. **Update adversarial detection scripts**
   - Modify `ADV_Samples.py` for CPU support
   - Update `ADV_Generate_LID_Mahalanobis.py`

### Implementation Strategy:
```python
# Device selection pattern to implement:
if torch.cuda.is_available() and args.gpu >= 0:
    device = torch.device(f'cuda:{args.gpu}')
else:
    device = torch.device('cpu')
    args.gpu = -1  # Signal CPU mode to other functions
```

## Phase 3: Testing and Validation

### Test Plan:
1. **Baseline test with GPU** (if available)
   - Run existing scripts to ensure no regression

2. **CPU functionality test**
   - Test with `--gpu -1` flag
   - Use smaller batch sizes for CPU efficiency
   - Verify output file generation

3. **Performance comparison**
   - Document execution time differences
   - Note memory usage patterns

## Phase 4: Custom Adaptation

### Future Tasks:
- Train models on custom datasets
- Adapt data loaders for specific use cases
- Implement domain-specific preprocessing
- Create comprehensive documentation

## Technical Notes

### Key Files and Functions:
- `lib_generation.py:sample_estimator()` - Computes class statistics
- `lib_generation.py:get_Mahalanobis_score()` - Core detection algorithm
- `OOD_*.py` - Main detection pipelines
- `train_*.py` - Model training (already CPU-compatible)

### CPU Optimization Tips:
- Use smaller batch sizes (32-64 vs 200)
- Consider reducing feature extraction layers
- Monitor memory usage during covariance estimation

### Dependencies for CPU Mode:
- PyTorch (CPU version)
- scikit-learn (for covariance estimation)
- NumPy, SciPy

## Next Session Goals:
1. Implement CPU support in OOD detection scripts
2. Test basic functionality on CPU
3. Document any performance considerations
4. Prepare for custom model training