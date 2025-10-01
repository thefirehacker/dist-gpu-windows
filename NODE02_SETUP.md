# Node 02 Setup Guide

## 🎯 Goal
Replicate Node 01's environment on Node 02

## 📋 What's on Node 01
- **OS:** Windows (no WSL)
- **GPU:** NVIDIA GeForce RTX 2050
- **CUDA:** 11.8
- **PyTorch:** 2.7.1+cu118

## 🔧 Setup Steps for Node 02

### 1. Install CUDA Toolkit 11.8

Download and install from:
https://developer.nvidia.com/cuda-11-8-0-download-archive

Choose: **Windows → x86_64 → 10/11 → exe (local)**

### 2. Install Python 3.11+

If not already installed:
https://www.python.org/downloads/

### 3. Create Virtual Environment

```bash
cd D:\04Code\dist-gpu-windows  # Or your project path
python -m venv .venv
source .venv/Scripts/activate  # In Git Bash
```

### 4. Install PyTorch and Dependencies

```bash
# Install PyTorch with CUDA 11.8
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install nbdistributed numpy==1.26.4
```

Or use requirements.txt:
```bash
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu118
```

### 5. Verify Installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0))"
```

Expected output:
```
PyTorch: 2.7.1+cu118
CUDA: 11.8
GPU: NVIDIA GeForce RTX 2050
```

### 6. Copy Project Files

Copy these files from Node 01 to Node 02:
- `train_multinode.py`
- `run_node1.sh`
- `NODE02_WORKER.md`

### 7. Configure and Run

Follow **NODE02_WORKER.md** for running instructions.

## ✅ Verification

After setup, test single-node first:
```bash
python train_standalone.py
```

Should show:
```
🎉 Success! Distributed training works on your system!
```
