# Node 02 - Worker (Rank 1) - WSL Setup

## Prerequisites

1. **Install WSL Ubuntu** on Windows
2. **Install NVIDIA CUDA on WSL driver** on Windows host
3. **Clone the repo** in WSL: `git clone <repo-url> ~/dist-gpu-windows`

## Choose Your Network Setup

You have two options for connecting nodes:

### Option 1: Direct Ethernet Cable (Recommended)
- **Best performance**: 1 Gbps dedicated bandwidth, lowest latency
- **Most reliable**: No router, no AP isolation issues
- **Setup**: See [Direct Ethernet Setup](#direct-ethernet-setup) below

### Option 2: WiFi/Router Network
- **Easier setup**: Use existing WiFi network
- **Shared bandwidth**: May be slower if network is busy
- **Setup**: Follow standard setup instructions

---

## Direct Ethernet Setup (No Router)

### Prerequisites
- One Ethernet cable connecting both nodes directly
- Both nodes must configure static IPs
- Node 01 must be configured first

### Step 1: Configure Static IP on Ethernet Adapter (Windows)

**On Node 02 (this machine):**

1. Open **Settings** → **Network & Internet** → **Ethernet**
2. Click on your Ethernet adapter (e.g., "Ethernet" or "Realtek PCIe GbE")
3. Click **Edit** next to "IP assignment"
4. Select **Manual**, turn on **IPv4**, and set:
   - **IP address**: `192.168.100.2`
   - **Subnet prefix length**: `24`
   - **Gateway**: (leave empty)
   - **Preferred DNS**: (leave empty)
5. Click **Save**

**Note:** Node 01 should already have IP `192.168.100.1` configured.

### Step 2: Set Network to Private Profile (PowerShell as Admin)

```powershell
# Set ethernet adapter to Private network profile
Set-NetConnectionProfile -InterfaceAlias "Ethernet" -NetworkCategory Private

# Or if your adapter has a different name:
Get-NetAdapter | Where-Object {$_.Status -eq "Up"}  # Find your adapter name
Set-NetConnectionProfile -InterfaceAlias "YOUR_ADAPTER_NAME" -NetworkCategory Private

# Verify it's set to Private
Get-NetConnectionProfile
```

### Step 3: Enable ICMP (Ping) for Testing (PowerShell as Admin)

```powershell
# Allow ping through firewall
New-NetFirewallRule -DisplayName "Allow ICMPv4-In" -Protocol ICMPv4 -IcmpType 8 -Direction Inbound -Action Allow

# Or disable firewall for private network (for initial testing)
Set-NetFirewallProfile -Profile Private -Enabled False

# Verify firewall status
Get-NetFirewallProfile | Select-Object Name, Enabled
```

### Step 4: Test Connectivity (Both Nodes)

**From Node 02 (Windows PowerShell):**
```powershell
ping 192.168.100.1
```

**From Node 01 (Windows PowerShell):**
```powershell
ping 192.168.100.2
```

**Expected output:**
```
Pinging 192.168.100.1 with 32 bytes of data:
Reply from 192.168.100.1: bytes=32 time<1ms TTL=128
```

### Step 5: Verify Link Speed

```powershell
Get-NetAdapter | Where-Object {$_.Status -eq "Up"} | Select-Object Name, Status, LinkSpeed
```

**Expected:** `LinkSpeed: 1 Gbps` (or 100 Mbps)

### Step 6: Configure Firewall for PyTorch (PowerShell as Admin)

```powershell
# Allow PyTorch distributed training ports
New-NetFirewallRule -DisplayName "PyTorch Distributed 29500" -Direction Inbound -LocalPort 29500 -Protocol TCP -Action Allow
New-NetFirewallRule -DisplayName "NCCL Communication Ports" -Direction Inbound -LocalPort 20000-40000 -Protocol TCP -Action Allow
```

**Note:** Node 02 worker doesn't need port forwarding (only Node 01 master does).

---

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

---

## Option C: Direct Ethernet with NCCL (Recommended for Best Performance)

### Prerequisites
- Completed [Direct Ethernet Setup](#direct-ethernet-setup) above
- Both nodes connected via ethernet cable with static IPs (192.168.100.1 and 192.168.100.2)
- WSL mirrored networking enabled (see Option B prerequisites)
- Ping successful between nodes
- **Node 01 must be running first**

### Why Direct Ethernet?
- **Dedicated bandwidth**: Full 1 Gbps, no interference from other devices
- **Lower latency**: Direct connection, no router hops
- **No router issues**: Bypasses AP isolation, firewall, and DHCP problems
- **Best for development**: Consistent performance, easy troubleshooting

### Verify WSL Can Access Ethernet Network

**In WSL, check if you can see the ethernet IP:**
```bash
ip addr show | grep -E "192\.168\.100"
```

You should see `192.168.100.2` listed (same as your Windows ethernet IP with mirrored networking).

**Test connectivity from WSL:**
```bash
ping 192.168.100.1  # Should reach Node 01
```

### Setup Environment Variables

```bash
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
```

### Run Command

**IMPORTANT: Start Node 01 first, then run this on Node 02:**

```bash
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=1 \
  --master_addr=192.168.100.1 \
  --master_port=29500 \
  train_torchrun.py
```

### Expected Output

```
Initializing with backend: nccl
[rank 1] world_size=2 device=cuda hostname=TUF-Node02
TUF-Node02:556:556 [0] NCCL INFO NCCL_SOCKET_IFNAME set by environment to eth0
TUF-Node02:556:556 [0] NCCL INFO Bootstrap : Using eth0:192.168.100.2<0>
TUF-Node02:556:556 [0] NCCL INFO NET/Socket : Using [0]eth0:192.168.100.2<0>
TUF-Node02:556:556 [0] NCCL INFO Using network Socket
[rank 1] gathered=[0, 1]
[rank 1] barrier OK; shutting down
```

### Performance
- **Bandwidth**: 10-50 GB/s (GPU-direct with NCCL)
- **Latency**: <1ms between nodes
- **Use case**: Best option for WSL multi-node training
- **Reliability**: Excellent (no router issues)

### Quick Launch Script

Create `run_ethernet_worker.sh`:
```bash
#!/bin/bash
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

torchrun --nnodes=2 --node_rank=1 --nproc_per_node=1 \
  --master_addr=192.168.100.1 \
  --master_port=29500 \
  train_torchrun.py
```

Make it executable: `chmod +x run_ethernet_worker.sh`

Run: `./run_ethernet_worker.sh`

---

## Troubleshooting

### Direct Ethernet Connection Issues

#### Ping fails to Node 01 over ethernet
- **Cause**: Network profile is Public, or firewall blocking ICMP
- **Fix**:
  1. Set network to Private: `Set-NetConnectionProfile -InterfaceAlias "Ethernet" -NetworkCategory Private`
  2. Enable ICMP: `New-NetFirewallRule -DisplayName "Allow ICMPv4-In" -Protocol ICMPv4 -IcmpType 8 -Direction Inbound -Action Allow`
  3. Or disable firewall temporarily: `Set-NetFirewallProfile -Profile Private -Enabled False`
  4. Verify Node 02 has correct static IP (192.168.100.2)
  5. Verify Node 01 has correct static IP (192.168.100.1)

#### WSL cannot see ethernet IP (192.168.100.x)
- **Cause**: WSL mirrored networking not enabled
- **Fix**: Enable mirrored networking (see Option B prerequisites)
- **Verify**: `ip addr show` should show 192.168.100.2, NOT 172.x.x.x
- **Alternative**: Use WiFi network (Option A or Option B with WiFi IPs)

#### Ethernet link not detected or "Media disconnected"
- **Cause**: Cable not properly connected or faulty
- **Fix**:
  1. Verify cable is securely plugged into both nodes
  2. Check link status: `Get-NetAdapter | Select-Object Name, Status, LinkSpeed`
  3. Should show `Status: Up, LinkSpeed: 1 Gbps`
  4. Try different cable if still not working

#### Cannot connect to Node 01 master (connection refused)
- **Cause**: Node 01 not started yet, or firewall blocking
- **Fix**:
  1. Ensure Node 01 is running first (master must start before worker)
  2. Verify firewall allows port 29500 on Node 01
  3. Test from WSL: `ping 192.168.100.1`
  4. Check Node 01 is listening: On Node 01, run `ss -tulpn | grep 29500` after starting master

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
