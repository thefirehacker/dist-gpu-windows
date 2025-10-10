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
# Verify PyTorch and CUDA
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('NCCL:', torch.distributed.is_nccl_available())"

# Verify torchrun is available
which torchrun
torchrun --version
```

**Expected output:**
```
PyTorch: 2.x.x
CUDA: True
NCCL: True
/home/youruser/dist-gpu-windows/.venv/bin/torchrun
torchrun 2.x.x
```

### 5. Setup Firewall for NCCL (PowerShell as Admin on Windows)

**Open PowerShell as Administrator on the Windows host** and run:

```powershell
# Open rendezvous port (29400)
New-NetFirewallRule -DisplayName "PyTorch Distributed 29400" -Direction Inbound -LocalPort 29400 -Protocol TCP -Action Allow

# Open NCCL communication port range (for GPU-to-GPU communication)
New-NetFirewallRule -DisplayName "NCCL Communication Ports" -Direction Inbound -LocalPort 20000-40000 -Protocol TCP -Action Allow
```

**Note:** Node 02 does NOT need port forwarding (no incoming connections from other nodes).

### 6. Get Node 01's Windows IP Address
Ask Node 01 operator for their **Windows IP** (not WSL IP).
Example: `192.168.29.67`

### 7. Test Connection (Optional)
```bash
ping 192.168.29.67  # Use Node 01's Windows IP
```

## Run

**Start this SECOND (after Node 01 is running):**

In WSL, with virtual environment activated:

### Standard Run:
```bash
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0

torchrun --nnodes=2 --node_rank=1 --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=192.168.29.67:29400 \
  train_torchrun.py
```

### With Debug Logging (recommended for first run):
```bash
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

torchrun --nnodes=2 --node_rank=1 --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=192.168.29.67:29400 \
  train_torchrun.py
```

**Replace `192.168.29.67` with Node 01's actual Windows IP address.**

**NCCL Environment Variables Explained:**
- `NCCL_IB_DISABLE=1` - Disables InfiniBand (not available on consumer hardware)
- `NCCL_P2P_DISABLE=1` - Disables peer-to-peer GPU access (not needed for multi-node)
- `NCCL_SOCKET_IFNAME=eth0` - Use eth0 network interface (WSL's default)
- `NCCL_DEBUG=INFO` - Show detailed NCCL logs (connection setup, ports used)

## Expected Output

### With `NCCL_DEBUG=INFO` enabled:

You'll see detailed NCCL logs:

```
Initializing with backend: nccl
NCCL version 2.x.x+cuda12.4
NCCL INFO Bootstrap : Using eth0:172.x.x.x<0>
NCCL INFO NET/Plugin : No plugin found (libnccl-net.so)
NCCL INFO NCCL_IB_DISABLE set by environment to 1
NCCL INFO NCCL_P2P_DISABLE set by environment to 1
NCCL INFO Channel 00/02 : 1 -> 0 via NET/Socket/0
NCCL INFO Connected all rings
[rank 1] world_size=2 device=cuda hostname=TUF-Node02
[rank 1] gathered=[0, 1]
[rank 1] barrier OK; shutting down
```

### Without debug (standard):
```
Initializing with backend: nccl
[rank 1] world_size=2 device=cuda hostname=TUF-Node02
[rank 1] gathered=[0, 1]
[rank 1] barrier OK; shutting down
```

## Troubleshooting

### Connection refused or timeout (rendezvous)
- **Cause**: Node 01 not started yet, or firewall blocking port 29400
- **Fix**: 
  1. Ensure Node 01 is running first
  2. Verify Node 01 has port forwarding set up (see Node 01 setup)
  3. Check if you can ping Node 01's Windows IP
  4. Verify firewall allows port 29400 on Node 01

### NCCL timeout - "server socket has timed out"
- **Cause**: NCCL communication ports (20000-40000) blocked by firewall
- **Fix**: 
  1. Ensure **BOTH** nodes have NCCL firewall rules (see step 5 above)
  2. On Node 01, verify port forwarding includes NCCL ports
  3. Run with `NCCL_DEBUG=INFO` to see which port is failing

### Wrong IP address
- **Cause**: Using WSL IP instead of Windows IP
- **Fix**: Use Node 01's **Windows IP** (from `ipconfig`), NOT the WSL IP (from `hostname -I`)

### torchrun not found
- **Cause**: Virtual environment not activated
- **Fix**: 
  ```bash
  cd ~/dist-gpu-windows
  source .venv/bin/activate
  which torchrun  # Should show path in .venv
  ```

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

### NCCL error: "Invalid argument" or "Invalid device pointer"
- **Cause**: Tensors not on correct device
- **Fix**: The script handles this automatically, but ensure CUDA is available
