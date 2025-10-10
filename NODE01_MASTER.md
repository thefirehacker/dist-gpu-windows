# Node 01 - Master (Rank 0) - WSL Setup

## Prerequisites

1. **Install WSL Ubuntu** on Windows
2. **Install NVIDIA CUDA on WSL driver** on Windows host
3. **Clone the repo** in WSL: `git clone <repo-url> ~/dist-gpu-windows`

## Setup

### 1. WSL Environment Setup (Run in WSL Ubuntu)

```bash
# Update system and install tools
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential python3 python3-pip python3-venv git

# Create virtual environment
cd ~/dist-gpu-windows
python3 -m venv .venv
source .venv/bin/activate

# Install PyTorch with CUDA
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 2. Verify Installation (In WSL)
```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('NCCL:', torch.distributed.is_nccl_available())"
```

**Expected output:**
```
PyTorch: 2.x.x
CUDA: True
NCCL: True
```

### 3. Get Your IP Addresses

**In WSL:**
```bash
hostname -I
```
Example output: `172.20.5.204` (your WSL IP)

**In Windows PowerShell:**
```powershell
ipconfig | findstr IPv4
```
Example output: `192.168.29.67` (your Windows IP - share this with Node 02)

### 4. Setup Port Forwarding (PowerShell as Admin)

**Open PowerShell as Administrator** and run:

```powershell
# Forward port 29400 from Windows to WSL
netsh interface portproxy add v4tov4 listenport=29400 listenaddress=0.0.0.0 connectport=29400 connectaddress=172.20.5.204

# Open Windows Firewall
New-NetFirewallRule -DisplayName "PyTorch Distributed 29400" -Direction Inbound -LocalPort 29400 -Protocol TCP -Action Allow
```

**Replace `172.20.5.204` with your actual WSL IP from step 3.**

### 5. Verify Port Forwarding (PowerShell)
```powershell
netsh interface portproxy show all
```

Expected output:
```
Listen on ipv4:             Connect to ipv4:
Address         Port        Address         Port
--------------- ----------  --------------- ----------
0.0.0.0         29400       172.20.5.204    29400
```

## Run

**Start this FIRST (before Node 02):**

In WSL, with virtual environment activated:

```bash
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=127.0.0.1:29400 \
  train_torchrun.py
```

## Expected Output

The master will wait for Node 02 to connect. You'll see:

```
Initializing with backend: nccl
[rank 0] world_size=2 device=cuda hostname=AIEDX-AsusTUF
[rank 0] gathered=[0, 1]
[rank 0] barrier OK; shutting down
```

## Troubleshooting

### Connection timeout or "failed to connect to 127.0.0.1:29400"
- **Cause**: Port forwarding not set up correctly or port already in use
- **Fix**: 
  1. Verify port forwarding: `netsh interface portproxy show all` (in PowerShell)
  2. Check if port is in use: `netstat -an | findstr 29400` (in PowerShell)
  3. Remove old port forwarding and re-add it

### Node 02 can't connect
- **Cause**: Firewall blocking or wrong IP shared
- **Fix**: 
  1. Share your **Windows IP** (from `ipconfig`), NOT WSL IP
  2. Verify firewall rule exists: `Get-NetFirewallRule -DisplayName "PyTorch Distributed 29400"` (PowerShell)
  3. Test from Node 02: `telnet 192.168.29.67 29400` (if telnet installed)

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

## Quick Command Reference

**Get WSL IP:**
```bash
hostname -I
```

**Get Windows IP:**
```powershell
ipconfig | findstr IPv4
```

**Check port forwarding:**
```powershell
netsh interface portproxy show all
```

**Remove port forwarding (if needed):**
```powershell
netsh interface portproxy delete v4tov4 listenport=29400 listenaddress=0.0.0.0
```

**Test if port is listening (in WSL after starting master):**
```bash
ss -tulpn | grep 29400
```
