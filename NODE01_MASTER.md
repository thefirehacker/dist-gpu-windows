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

### 4. Setup Firewall for NCCL (PowerShell as Admin)

**Open PowerShell as Administrator** and run:

```powershell
# Open rendezvous port (29400)
New-NetFirewallRule -DisplayName "PyTorch Distributed 29400" -Direction Inbound -LocalPort 29400 -Protocol TCP -Action Allow

# Open NCCL communication port range (for GPU-to-GPU communication)
New-NetFirewallRule -DisplayName "NCCL Communication Ports" -Direction Inbound -LocalPort 20000-40000 -Protocol TCP -Action Allow
```

### 5. Setup Port Forwarding (PowerShell as Admin)

**Open PowerShell as Administrator** and run:

```powershell
# Get your WSL IP (run: wsl hostname -I)
$WSLIP = "172.20.5.204"  # Replace with your actual WSL IP from step 3

# Forward rendezvous port
netsh interface portproxy add v4tov4 listenport=29400 listenaddress=0.0.0.0 connectport=29400 connectaddress=$WSLIP

# Forward NCCL communication ports (subset for efficiency)
29400..29410 | ForEach-Object {
    netsh interface portproxy add v4tov4 listenport=$_ listenaddress=0.0.0.0 connectport=$_ connectaddress=$WSLIP
}
```

**Replace `172.20.5.204` with your actual WSL IP from step 3.**

### 6. Verify Port Forwarding (PowerShell)
```powershell
netsh interface portproxy show all
```

Expected output:
```
Listen on ipv4:             Connect to ipv4:
Address         Port        Address         Port
--------------- ----------  --------------- ----------
0.0.0.0         29400       172.20.5.204    29400
0.0.0.0         29401       172.20.5.204    29401
0.0.0.0         29402       172.20.5.204    29402
...
```

## Run

**Start this FIRST (before Node 02):**

In WSL, with virtual environment activated:

### Standard Run:
```bash
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0

torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=127.0.0.1:29400 \
  train_torchrun.py
```

### With Debug Logging (recommended for first run):
```bash
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=127.0.0.1:29400 \
  train_torchrun.py
```

**NCCL Environment Variables Explained:**
- `NCCL_IB_DISABLE=1` - Disables InfiniBand (not available on consumer hardware)
- `NCCL_P2P_DISABLE=1` - Disables peer-to-peer GPU access (not needed for multi-node)
- `NCCL_SOCKET_IFNAME=eth0` - Use eth0 network interface (WSL's default)
- `NCCL_DEBUG=INFO` - Show detailed NCCL logs (connection setup, ports used)

## Expected Output

### With `NCCL_DEBUG=INFO` enabled:

The master will wait for Node 02 to connect. You'll see detailed NCCL logs:

```
Initializing with backend: nccl
NCCL version 2.x.x+cuda12.4
NCCL INFO Bootstrap : Using eth0:172.20.5.204<0>
NCCL INFO NET/Plugin : No plugin found (libnccl-net.so)
NCCL INFO NCCL_IB_DISABLE set by environment to 1
NCCL INFO NCCL_P2P_DISABLE set by environment to 1
NCCL INFO Channel 00/02 : 0 -> 1 via NET/Socket/0
NCCL INFO Connected all rings
[rank 0] world_size=2 device=cuda hostname=AIEDX-AsusTUF
[rank 0] gathered=[0, 1]
[rank 0] barrier OK; shutting down
```

### Without debug (standard):
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
