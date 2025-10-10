# Node 02 - Worker (Rank 1) - WSL Setup

## Prerequisites

1. **Install WSL Ubuntu** on Windows
2. **Install NVIDIA CUDA on WSL driver** on Windows host
3. **Clone the repo** in WSL: `git clone <repo-url> ~/dist-gpu-windows`

## Setup (Run in WSL Ubuntu)

### 1. Update System & Install Tools
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential python3 python3-pip python3-venv git
```

### 2. Create Virtual Environment
```bash
cd ~/dist-gpu-windows
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install PyTorch with CUDA
```bash
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 4. Verify Installation
```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('NCCL:', torch.distributed.is_nccl_available())"
```

**Expected output:**
```
PyTorch: 2.x.x
CUDA: True
NCCL: True
```

### 5. Get Node 01's Windows IP Address
Ask Node 01 operator for their **Windows IP** (not WSL IP).
Example: `192.168.29.67`

### 6. Test Connection (Optional)
```bash
ping 192.168.29.67  # Use Node 01's Windows IP
```

## Run

**Start this SECOND (after Node 01 is running):**

In WSL, with virtual environment activated:

```bash
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=192.168.29.67:29400 \
  train_torchrun.py
```

**Replace `192.168.29.67` with Node 01's actual Windows IP address.**

## Expected Output

```
Initializing with backend: nccl
[rank 1] world_size=2 device=cuda hostname=YOUR-HOSTNAME
[rank 1] gathered=[0, 1]
[rank 1] barrier OK; shutting down
```

## Troubleshooting

### Connection refused or timeout
- **Cause**: Node 01 not started yet, or firewall blocking, or port forwarding not set up
- **Fix**: 
  1. Ensure Node 01 is running first
  2. Verify Node 01 has port forwarding set up (see Node 01 setup)
  3. Check if you can ping Node 01's Windows IP

### Wrong IP address
- **Cause**: Using WSL IP instead of Windows IP
- **Fix**: Use Node 01's **Windows IP** (from `ipconfig`), NOT the WSL IP (from `hostname -I`)

### NCCL not available
- **Cause**: PyTorch installed without CUDA support
- **Fix**: 
  ```bash
  pip uninstall torch torchvision torchaudio -y
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
  ```

### CUDA not available
- **Cause**: NVIDIA driver not installed on Windows host
- **Fix**: Install NVIDIA CUDA on WSL driver from https://developer.nvidia.com/cuda/wsl
