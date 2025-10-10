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
# OPTION A: Disable Windows Firewall for Private Network (Recommended for initial testing)
Set-NetFirewallProfile -Profile Private -Enabled False

# Verify firewall is disabled
Get-NetFirewallProfile | Select-Object Name, Enabled

# OPTION B: Keep firewall enabled and add specific rules
New-NetFirewallRule -DisplayName "PyTorch Distributed 29500" -Direction Inbound -LocalPort 29500 -Protocol TCP -Action Allow
New-NetFirewallRule -DisplayName "NCCL Communication Ports" -Direction Inbound -LocalPort 20000-40000 -Protocol TCP -Action Allow

# Also allow ICMP (ping) for testing connectivity
New-NetFirewallRule -DisplayName "Allow ICMPv4-In" -Protocol ICMPv4 -IcmpType 8 -Enabled True -Direction Inbound -Action Allow
```

**Note:** 
- If using Option A, remember to re-enable firewall after testing: `Set-NetFirewallProfile -Profile Private -Enabled True`
- Node 02 does NOT need port forwarding (no incoming connections from other nodes)

### 6. Get Node 01's Windows IP Address
Ask Node 01 operator for their **Windows IP** (not WSL IP).
Example: `192.168.29.67`

### 7. Test Connection (Optional)
```bash
ping 192.168.29.67  # Use Node 01's Windows IP
```

## Run - Choose Your Backend

**Start this SECOND (after Node 01 is running):**

In WSL, with virtual environment activated.

---

## Option A: Gloo Backend (Recommended for WSL Multi-Node)

### Why Gloo?
WSL2's NAT networking has fundamental limitations with NCCL across physical machines. Gloo works reliably with port forwarding and is the recommended choice for WSL multi-node setups.

### Run Command:
```bash
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=192.168.29.67:29400 \
  train_torchrun_wsl_gloo.py
```

**Replace `192.168.29.67` with Node 01's actual Windows IP address.**

### Expected Output:
```
Initializing with backend: gloo
Note: Using Gloo for WSL multi-node compatibility
[rank 1] world_size=2 device=cuda hostname=TUF-Node02
[rank 1] gathered=[0, 1]
[rank 1] barrier OK; shutting down
```

### Performance:
- **Bandwidth**: 1-10 GB/s (CPU-mediated)
- **Use case**: Development, testing, small models
- **Reliability**: Excellent with WSL

---

## Option B: NCCL Backend (Requires WSL Mirrored Networking)

### Prerequisites: Enable WSL2 Mirrored Networking

**IMPORTANT:** NCCL requires WSL2 mirrored networking mode. You must complete this setup first.

**Requirements:**
- Windows 11 version 22H2 or later (Build 22621+)
- Check version in PowerShell: `[System.Environment]::OSVersion.Version`

**Setup Steps (PowerShell as Administrator):**

1. **Create/Edit `.wslconfig` file:**
```powershell
# Create .wslconfig with mirrored networking
@"
[wsl2]
networkingMode=mirrored
"@ | Out-File -FilePath "$env:USERPROFILE\.wslconfig" -Encoding ASCII -Force

# Verify it was created
Get-Content $env:USERPROFILE\.wslconfig
```

2. **Shutdown and restart WSL:**
```powershell
wsl --shutdown
Start-Sleep -Seconds 30
wsl
```

3. **Verify mirrored networking is working (in WSL):**
```bash
ip addr show eth0 | grep "inet "
```

You should see your **Windows IP** (e.g., `192.168.29.197`), NOT a `172.x.x.x` IP.

If you still see `172.x.x.x`, mirrored mode is not working. Check your Windows version or use Option A (Gloo) instead.

### Setup Environment Variables:
```bash
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
```

### Run Command:
```bash
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=1 \
  --master_addr=192.168.29.67 \
  --master_port=29500 \
  train_torchrun.py
```

**Replace `192.168.29.67` with Node 01's actual Windows IP address.**

### NCCL Environment Variables Explained:
- `NCCL_IB_DISABLE=1` - Disables InfiniBand (not available on consumer hardware)
- `NCCL_P2P_DISABLE=1` - Disables peer-to-peer GPU access (not needed for multi-node)
- `NCCL_SOCKET_IFNAME=eth0` - Use eth0 network interface (WSL's default)
- `NCCL_DEBUG=INFO` - Show detailed NCCL logs (connection setup, ports used)

### Expected Output (If Successful):
```
Initializing with backend: nccl
[rank 1] world_size=2 device=cuda hostname=TUF-Node02
TUF-Node02:556:556 [0] NCCL INFO NCCL_SOCKET_IFNAME set by environment to eth0
TUF-Node02:556:556 [0] NCCL INFO Bootstrap : Using eth0:192.168.29.197<0>
TUF-Node02:556:556 [0] NCCL INFO NET/Plugin: No plugin found (libnccl-net.so)
TUF-Node02:556:556 [0] NCCL INFO NET/Plugin: Using internal network plugin.
NCCL version 2.21.5+cuda12.4
TUF-Node02:556:574 [0] NCCL INFO NCCL_IB_DISABLE set by environment to 1.
TUF-Node02:556:574 [0] NCCL INFO NET/Socket : Using [0]eth0:192.168.29.197<0>
TUF-Node02:556:574 [0] NCCL INFO Using network Socket
[rank 1] gathered=[0, 1]
[rank 1] barrier OK; shutting down
```

### Performance (If Working):
- **Bandwidth**: 10-50 GB/s (GPU-direct)
- **Use case**: Production, large models
- **Reliability**: Poor on WSL (use native Linux instead)

## Troubleshooting

### Connection refused or timeout (rendezvous)
- **Cause**: Node 01 not started yet, or firewall blocking port 29400
- **Fix**: 
  1. Ensure Node 01 is running first
  2. Verify Node 01 has port forwarding set up (see Node 01 setup)
  3. Check if you can ping Node 01's Windows IP
  4. Verify firewall allows port 29400 on Node 01

### NCCL timeout - "server socket has timed out" or "connection reset"
- **Cause**: WSL2 NAT networking limitation - NCCL cannot work reliably across WSL instances on different machines
- **Fix**: 
  1. **Use Option 1 (Gloo backend)** - this is the recommended workaround for WSL
  2. Alternative: Move to native Linux on both machines for full NCCL support
  3. Advanced: Try WSL2 with mirrored networking (Windows 11 22H2+) - experimental

### NCCL error: "socketStartConnect: Connect to IP<port> failed : Software caused connection abort"
- **Cause**: Firewall or antivirus (e.g., Norton, Avira) blocking NCCL GPU-to-GPU communication
- **Fix**:
  1. **Disable antivirus firewall temporarily** (e.g., Norton Security/Firewall, Avira)
  2. **Disable Windows Firewall for Private network** (see Step 5 - Option A)
  3. Add firewall exceptions for Python and the NCCL port range (20000-40000)
  4. Check if both nodes have WSL mirrored networking enabled
  5. Verify both nodes can ping each other: `ping 192.168.29.67`

### Ping fails to Node 01 / Router AP Isolation
- **Cause**: Router has AP Isolation enabled, blocking WiFi clients from communicating
- **Fix**:
  1. **Access router admin** (usually `http://192.168.29.1` for Jio Fiber)
  2. Find and **disable "AP Isolation"** or "Client Isolation" in WiFi settings
  3. **Reboot router** after changing settings
  4. Alternative: **Use direct Ethernet cable** between laptops (recommended for best performance):
     - Connect both laptops with an Ethernet cable
     - On Node 01: Set static IP `10.0.0.1` on Ethernet adapter
     - On Node 02: Set static IP `10.0.0.2` on Ethernet adapter
     - Use `--master_addr=10.0.0.1` in training commands
     - Benefits: No router issues, lower latency, higher bandwidth

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
