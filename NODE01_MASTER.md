# Node 01 - Master (Rank 0) - WSL Setup

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

### Step 1: Configure Static IP on Ethernet Adapter (Windows)

**On Node 01 (this machine):**

1. Open **Settings** → **Network & Internet** → **Ethernet**
2. Click on your Ethernet adapter (e.g., "Ethernet" or "Realtek PCIe GbE")
3. Click **Edit** next to "IP assignment"
4. Select **Manual**, turn on **IPv4**, and set:
   - **IP address**: `192.168.100.1`
   - **Subnet prefix length**: `24`
   - **Gateway**: (leave empty)
   - **Preferred DNS**: (leave empty)
5. Click **Save**

**On Node 02:**
- Same steps, but use IP: `192.168.100.2`

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

**From Node 01 (Windows PowerShell):**
```powershell
ping 192.168.100.2
```

**From Node 02 (Windows PowerShell):**
```powershell
ping 192.168.100.1
```

**Expected output:**
```
Pinging 192.168.100.2 with 32 bytes of data:
Reply from 192.168.100.2: bytes=32 time<1ms TTL=128
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

---

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

**Note:** If using Option A, remember to re-enable firewall after testing:
```powershell
Set-NetFirewallProfile -Profile Private -Enabled True
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

## Run - Choose Your Backend

**Start this FIRST (before Node 02):**

In WSL, with virtual environment activated.

---

## Option A: Gloo Backend (Recommended for WSL Multi-Node)

### Why Gloo?
WSL2's NAT networking has fundamental limitations with NCCL across physical machines. Gloo works reliably with port forwarding and is the recommended choice for WSL multi-node setups.

### Run Command:
```bash
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=172.20.5.204:29400 \
  train_torchrun_wsl_gloo.py
```

**Replace `172.20.5.204` with your actual WSL IP from step 3.**

### Expected Output:
```
Initializing with backend: gloo
Note: Using Gloo for WSL multi-node compatibility
[rank 0] world_size=2 device=cuda hostname=AIEDX-AsusTUF
[rank 0] gathered=[0, 1]
[rank 0] barrier OK; shutting down
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

You should see your **Windows IP** (e.g., `192.168.29.67`), NOT a `172.x.x.x` IP.

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
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 \
  --master_addr=192.168.29.67 \
  --master_port=29500 \
  train_torchrun.py
```

**Replace `192.168.29.67` with your actual Windows IP (should be same as WSL IP with mirrored mode).**

**Note:** With mirrored networking, your WSL IP = Windows IP, so no port forwarding is needed!

### NCCL Environment Variables Explained:
- `NCCL_IB_DISABLE=1` - Disables InfiniBand (not available on consumer hardware)
- `NCCL_P2P_DISABLE=1` - Disables peer-to-peer GPU access (not needed for multi-node)
- `NCCL_SOCKET_IFNAME=eth0` - Use eth0 network interface (WSL's default)
- `NCCL_DEBUG=INFO` - Show detailed NCCL logs (connection setup, ports used)

### Expected Output (If Successful):
```
Initializing with backend: nccl
[rank 0] world_size=2 device=cuda hostname=AIEDX-AsusTUF
AIEDX-AsusTUF:706:706 [0] NCCL INFO NCCL_SOCKET_IFNAME set by environment to eth0
AIEDX-AsusTUF:706:706 [0] NCCL INFO Bootstrap : Using eth0:192.168.29.67<0>
AIEDX-AsusTUF:706:706 [0] NCCL INFO NET/Plugin: No plugin found (libnccl-net.so)
AIEDX-AsusTUF:706:706 [0] NCCL INFO NET/Plugin: Using internal network plugin.
AIEDX-AsusTUF:706:706 [0] NCCL INFO cudaDriverVersion 12030
NCCL version 2.21.5+cuda12.4
AIEDX-AsusTUF:706:724 [0] NCCL INFO NCCL_IB_DISABLE set by environment to 1.
AIEDX-AsusTUF:706:724 [0] NCCL INFO NET/Socket : Using [0]eth0:192.168.29.67<0>
AIEDX-AsusTUF:706:724 [0] NCCL INFO Using network Socket
[rank 0] gathered=[0, 1]
[rank 0] barrier OK; shutting down
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

You should see `192.168.100.1` listed (same as your Windows ethernet IP with mirrored networking).

**Test connectivity from WSL:**
```bash
ping 192.168.100.2  # Should reach Node 02
```

### Setup Environment Variables

```bash
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
```

### Run Command

```bash
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 \
  --master_addr=192.168.100.1 \
  --master_port=29500 \
  train_torchrun.py
```

### Expected Output

```
Initializing with backend: nccl
[rank 0] world_size=2 device=cuda hostname=AIEDX-AsusTUF
AIEDX-AsusTUF:706:706 [0] NCCL INFO NCCL_SOCKET_IFNAME set by environment to eth0
AIEDX-AsusTUF:706:706 [0] NCCL INFO Bootstrap : Using eth0:192.168.100.1<0>
AIEDX-AsusTUF:706:706 [0] NCCL INFO NET/Socket : Using [0]eth0:192.168.100.1<0>
AIEDX-AsusTUF:706:706 [0] NCCL INFO Using network Socket
[rank 0] gathered=[0, 1]
[rank 0] barrier OK; shutting down
```

### Performance
- **Bandwidth**: 10-50 GB/s (GPU-direct with NCCL)
- **Latency**: <1ms between nodes
- **Use case**: Best option for WSL multi-node training
- **Reliability**: Excellent (no router issues)

### Quick Launch Script

Create `run_ethernet_master.sh`:
```bash
#!/bin/bash
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 \
  --master_addr=192.168.100.1 \
  --master_port=29500 \
  train_torchrun.py
```

Make it executable: `chmod +x run_ethernet_master.sh`

Run: `./run_ethernet_master.sh`

---

## Troubleshooting

### Direct Ethernet Connection Issues

#### Ping fails between ethernet nodes
- **Cause**: Network profile is Public, or firewall blocking ICMP
- **Fix**:
  1. Set network to Private: `Set-NetConnectionProfile -InterfaceAlias "Ethernet" -NetworkCategory Private`
  2. Enable ICMP: `New-NetFirewallRule -DisplayName "Allow ICMPv4-In" -Protocol ICMPv4 -IcmpType 8 -Direction Inbound -Action Allow`
  3. Or disable firewall temporarily: `Set-NetFirewallProfile -Profile Private -Enabled False`
  4. Verify both nodes have correct static IPs (192.168.100.1 and 192.168.100.2)

#### WSL cannot see ethernet IP (192.168.100.x)
- **Cause**: WSL mirrored networking not enabled, OR WiFi has higher priority than Ethernet
- **Fix**: 
  1. Enable mirrored networking (see Option B prerequisites)
  2. If both WiFi and Ethernet are connected, set Ethernet to higher priority (see below)
- **Verify**: `ip addr show` should show 192.168.100.1, NOT 172.x.x.x
- **Alternative**: Use WiFi network (Option A or Option B with WiFi IPs)

#### WSL picks WiFi IP instead of Ethernet IP (both adapters active)
- **Cause**: Windows network adapter priority - WiFi has lower metric (higher priority) than Ethernet
- **Symptoms**: 
  - Windows shows both IPs (WiFi: 192.168.29.x, Ethernet: 192.168.100.x)
  - WSL only shows WiFi IP (192.168.29.x)
  - Node 02 can ping both IPs from Windows, but NCCL fails
- **Fix - Set Ethernet to Higher Priority (PowerShell as Admin):**
  ```powershell
  # First, find your adapter names
  Get-NetAdapter | Where-Object {$_.Status -eq "Up"}
  
  # Set Ethernet adapter to higher priority (lower metric = higher priority)
  # Replace "Ethernet" with your actual adapter name (e.g., "Pytorch-Dist-Training")
  Set-NetIPInterface -InterfaceAlias "Ethernet" -InterfaceMetric 10
  
  # Set WiFi to lower priority (higher metric = lower priority)
  Set-NetIPInterface -InterfaceAlias "Wi-Fi" -InterfaceMetric 100
  
  # Verify the changes
  Get-NetIPInterface | Where-Object {$_.ConnectionState -eq "Connected"} | Select-Object InterfaceAlias, InterfaceMetric, AddressFamily | Sort-Object InterfaceMetric
  ```
  
  **Expected output:**
  ```
  InterfaceAlias        InterfaceMetric AddressFamily
  --------------        --------------- -------------
  Ethernet                           10 IPv4
  Ethernet                           10 IPv6
  Wi-Fi                             100 IPv4
  Wi-Fi                             100 IPv6
  ```
  
  **Then restart WSL:**
  ```powershell
  wsl --shutdown
  # Wait 15-20 seconds
  wsl
  ```
  
  **Verify in WSL:**
  ```bash
  ip addr show eth0 | grep "inet "
  # Should now show: inet 192.168.100.1/24 (Ethernet IP, not WiFi)
  ```
  
- **Alternative**: Disable WiFi adapter completely:
  ```powershell
  Disable-NetAdapter -Name "Wi-Fi" -Confirm:$false
  ```

#### Ethernet link not detected or "Media disconnected"
- **Cause**: Cable not properly connected or faulty
- **Fix**:
  1. Verify cable is securely plugged into both nodes
  2. Check link status: `Get-NetAdapter | Select-Object Name, Status, LinkSpeed`
  3. Should show `Status: Up, LinkSpeed: 1 Gbps`
  4. Try different cable if still not working

#### NCCL timeout over ethernet
- **Cause**: Firewall blocking PyTorch ports
- **Fix**:
  1. Ensure firewall rules are added: See [Step 6](#step-6-configure-firewall-for-pytorch-powershell-as-admin)
  2. Or disable firewall: `Set-NetFirewallProfile -Profile Private -Enabled False`
  3. Verify port 29500 is open on both nodes

### Connection timeout with Gloo backend
- **Cause**: Port forwarding not set up correctly or port already in use
- **Fix**: 
  1. Verify port forwarding: `netsh interface portproxy show all` (in PowerShell)
  2. Check if port is in use: `ss -tulpn | grep 29400` (in WSL)
  3. Ensure you're using WSL IP for endpoint: `hostname -I` (should match endpoint IP)

### Node 02 can't connect
- **Cause**: Firewall blocking or wrong IP shared
- **Fix**: 
  1. Share your **Windows IP** (from `ipconfig`), NOT WSL IP
  2. Verify firewall rule exists: `Get-NetFirewallRule -DisplayName "PyTorch Distributed 29400"` (PowerShell)
  3. Test connectivity: `ping 192.168.29.67` from Node 02

### NCCL timeout or connection reset (Option B)
- **Cause**: WSL2 NAT networking limitation - NCCL cannot work reliably across WSL instances on different machines
- **Fix**: Use Option A (Gloo backend) instead, OR move to native Linux for full NCCL support

### NCCL error: "socketStartConnect: Connect to IP<port> failed : Software caused connection abort"
- **Cause**: Firewall or antivirus (e.g., Norton) blocking NCCL GPU-to-GPU communication
- **Fix**:
  1. **Disable antivirus firewall temporarily** (e.g., Norton Security/Firewall, Avira)
  2. **Disable Windows Firewall for Private network** (see Step 4 - Option A)
  3. Add firewall exceptions for Python and the NCCL port range (20000-40000)
  4. Ensure WSL mirrored networking is enabled (see Option B prerequisites)
  5. Verify both nodes can ping each other: `ping 192.168.29.197` (Node 02's IP)

### Ping fails between nodes / Router AP Isolation
- **Cause**: Router has AP Isolation enabled, blocking WiFi clients from communicating
- **Fix**:
  1. **Access router admin** (usually `http://192.168.29.1` for Jio Fiber)
  2. Find and **disable "AP Isolation"** or "Client Isolation" in WiFi settings
  3. **Reboot router** after changing settings
  4. Alternative: **Use direct Ethernet cable** between laptops (recommended for best performance):
     - Connect laptops with Ethernet cable
     - Set static IPs: Node 01 = `10.0.0.1`, Node 02 = `10.0.0.2`
     - Use `--master_addr=10.0.0.1` in training commands
     - Benefits: No router issues, lower latency, higher bandwidth

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
