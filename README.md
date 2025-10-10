# 🚀 Distributed PyTorch Multi-Node GPU Training

**Production-ready multi-node distributed PyTorch training setup using WSL2, NCCL, and torchrun.**

Scale your deep learning workloads across multiple Windows machines with GPUs, leveraging NVIDIA's NCCL for high-performance GPU-to-GPU communication.

## 🎯 Overview

This project provides a complete, battle-tested solution for distributed PyTorch training across multiple physical machines:

- **🖥️ Platform**: Windows 11 + WSL2 (Ubuntu)
- **⚡ Backend**: NCCL (GPU-accelerated) + Gloo (CPU fallback)
- **🔗 Networking**: WSL2 Mirrored Networking Mode
- **🎮 GPUs**: NVIDIA CUDA-enabled GPUs (RTX/GTX series)
- **📡 Rendezvous**: PyTorch's native distributed launcher (`torchrun`)

### Architecture

```
┌─────────────────────────────────┐      ┌─────────────────────────────────┐
│   Node 01 (Master - Rank 0)     │      │   Node 02 (Worker - Rank 1)     │
│  ┌──────────────────────────┐   │      │  ┌──────────────────────────┐   │
│  │  Windows 11 + WSL2       │   │      │  │  Windows 11 + WSL2       │   │
│  │  ├─ Ubuntu 22.04         │   │      │  │  ├─ Ubuntu 22.04         │   │
│  │  ├─ PyTorch 2.x + CUDA   │   │      │  │  ├─ PyTorch 2.x + CUDA   │   │
│  │  ├─ NCCL 2.21.5          │   │      │  │  ├─ NCCL 2.21.5          │   │
│  │  └─ NVIDIA GPU (CUDA)    │   │      │  │  └─ NVIDIA GPU (CUDA)    │   │
│  └──────────────────────────┘   │      │  └──────────────────────────┘   │
│         192.168.29.67            │      │         192.168.29.197          │
└─────────────────────────────────┘      └─────────────────────────────────┘
                    │                                    │
                    └────────────────┬───────────────────┘
                                     │
                              Local Network / Direct Ethernet
                              (10-50 GB/s with NCCL)
```

### Key Features

✅ **Multi-node NCCL**: True GPU-to-GPU communication across physical machines  
✅ **WSL2 Native**: No Docker, no VM overhead - direct CUDA access  
✅ **Mirrored Networking**: WSL2's latest networking mode for seamless connectivity  
✅ **Dual Backend**: NCCL for performance, Gloo for compatibility  
✅ **Production Ready**: Comprehensive troubleshooting and firewall configuration  
✅ **Easy Setup**: Step-by-step guides for both master and worker nodes

## 📁 Project Structure

```
dist-gpu-windows/
├── 📄 train_torchrun.py           # Main training script (NCCL backend)
├── 📄 worker.py                   # Standalone worker script
├── 📄 requirements.txt            # Python dependencies
│
├── 📚 Documentation
│   ├── NODE01_MASTER.md           # Complete Node 01 setup guide
│   ├── NODE02_WORKER.md           # Complete Node 02 setup guide
│   ├── QUICK_START.md             # Quick reference guide
│   ├── SOLUTION.md                # Architecture & design decisions
│   ├── WSL_MIRRORED_NETWORKING.md # Mirrored networking setup
│   └── WINDOWS_SETUP.md           # Windows-specific instructions
│
├── 🚀 Launch Scripts
│   ├── run_node0.sh               # Launch master node
│   ├── run_node1.sh               # Launch worker node
│   ├── run_single_node.sh         # Single-node testing
│   └── run_train.sh               # Training launcher
│
├── 📦 Archive & Tests
│   ├── archive/                   # Previous implementations
│   ├── Test/                      # Test scripts and utilities
│   └── issues/                    # Issue tracking and solutions
│
└── 🔬 Experimental
    └── L1-nbdistributed/          # Jupyter notebook experiments
```

### Key Files

| File | Purpose | When to Use |
|------|---------|-------------|
| `train_torchrun.py` | Main distributed training script with NCCL | Multi-node GPU training (recommended) |
| `NODE01_MASTER.md` | Master node setup instructions | Setting up Node 01 |
| `NODE02_WORKER.md` | Worker node setup instructions | Setting up Node 02 |
| `QUICK_START.md` | Quick reference and troubleshooting | Fast setup & debugging |
| `WSL_MIRRORED_NETWORKING.md` | WSL2 networking configuration | Enabling mirrored mode |

## 🚀 Quick Start

### Prerequisites

**System Requirements (Both Nodes):**
- 💻 Windows 11 (Build 22621+ for mirrored networking)
- 🐧 WSL2 with Ubuntu 22.04
- 🎮 NVIDIA GPU (RTX/GTX series)
- 🔧 NVIDIA CUDA on WSL driver
- 🌐 Same local network or direct Ethernet connection

**Software Requirements:**
- Python 3.10+
- PyTorch 2.x with CUDA 12.4 support
- Git

---

### 🎯 30-Second Setup

**On Both Node 01 and Node 02:**

#### 1️⃣ Install WSL2 (Windows PowerShell as Admin)
```powershell
wsl --install -d Ubuntu-22.04
```

#### 2️⃣ Clone Repository (Inside WSL)
```bash
git clone <your-repo-url> ~/dist-gpu-windows
cd ~/dist-gpu-windows
```

#### 3️⃣ Setup Python Environment (WSL)
```bash
# Install dependencies
sudo apt update && sudo apt install -y build-essential python3 python3-pip python3-venv git

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install PyTorch with CUDA
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

#### 4️⃣ Enable WSL2 Mirrored Networking (Windows PowerShell as Admin)
```powershell
# Create .wslconfig
@"
[wsl2]
networkingMode=mirrored
"@ | Out-File -FilePath "$env:USERPROFILE\.wslconfig" -Encoding ASCII -Force

# Restart WSL
wsl --shutdown
Start-Sleep -Seconds 30
wsl
```

#### 5️⃣ Configure Firewall (Windows PowerShell as Admin)
```powershell
# Disable Windows Firewall for Private network (recommended for testing)
Set-NetFirewallProfile -Profile Private -Enabled False

# OR add specific rules
New-NetFirewallRule -DisplayName "PyTorch Distributed 29500" -Direction Inbound -LocalPort 29500 -Protocol TCP -Action Allow
New-NetFirewallRule -DisplayName "NCCL Communication Ports" -Direction Inbound -LocalPort 20000-40000 -Protocol TCP -Action Allow
```

#### 6️⃣ Disable Antivirus (Temporarily)
- **Norton/Avira**: Disable firewall during testing
- This prevents blocking of NCCL GPU communication

---

### 🏃 Run Multi-Node Training

**Step 1: Get Node 01's IP (WSL on Node 01)**
```bash
ip addr show eth0 | grep "inet "
# Example: 192.168.29.67
```

**Step 2: Start Master Node (WSL on Node 01)**
```bash
cd ~/dist-gpu-windows
source .venv/bin/activate

# Set NCCL environment variables
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

# Launch master (replace IP with your Node 01 IP)
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 \
  --master_addr=192.168.29.67 \
  --master_port=29500 \
  train_torchrun.py
```

**Step 3: Start Worker Node (WSL on Node 02)**
```bash
cd ~/dist-gpu-windows
source .venv/bin/activate

# Set NCCL environment variables
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

# Launch worker (use Node 01's IP)
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=1 \
  --master_addr=192.168.29.67 \
  --master_port=29500 \
  train_torchrun.py
```

### ✅ Expected Output

**Both nodes should show:**
```
Initializing with backend: nccl
[rank X] world_size=2 device=cuda hostname=NODE-NAME
NCCL INFO Bootstrap : Using eth0:192.168.29.XX<0>
NCCL INFO Connected all rings
NCCL INFO Connected all trees
[rank X] gathered=[0, 1]
[rank X] barrier OK; shutting down
```

---

### 📚 Detailed Guides

For complete setup instructions and troubleshooting:

| Guide | Description |
|-------|-------------|
| 📖 [NODE01_MASTER.md](NODE01_MASTER.md) | Complete setup for master node (Rank 0) |
| 📖 [NODE02_WORKER.md](NODE02_WORKER.md) | Complete setup for worker node (Rank 1) |
| 📖 [QUICK_START.md](QUICK_START.md) | Quick reference and common commands |
| 📖 [WSL_MIRRORED_NETWORKING.md](WSL_MIRRORED_NETWORKING.md) | WSL2 networking configuration |
| 📖 [SOLUTION.md](SOLUTION.md) | Architecture and design decisions |

## 🔧 Configuration

### Network Settings

| Setting | Value | Description |
|---------|-------|-------------|
| **Port** | 29500 | Master rendezvous port |
| **Backend** | NCCL / Gloo | NCCL for GPU, Gloo for CPU fallback |
| **Rendezvous** | Native `torchrun` | Uses `--master_addr` and `--master_port` |
| **Network Mode** | WSL2 Mirrored | Direct host network access |
| **Communication** | TCP/IP | Over Ethernet (eth0) |

### NCCL Environment Variables

```bash
export NCCL_IB_DISABLE=1         # Disable InfiniBand (not available on consumer hardware)
export NCCL_P2P_DISABLE=1        # Disable peer-to-peer GPU access (for multi-node)
export NCCL_SOCKET_IFNAME=eth0   # Use Ethernet interface
export NCCL_DEBUG=INFO           # Enable debug logging
```

### Environment Variables (Auto-set by torchrun)

```bash
MASTER_ADDR=192.168.29.67   # Set via --master_addr
MASTER_PORT=29500           # Set via --master_port
WORLD_SIZE=2                # Total number of processes
RANK=0/1                    # Process rank (0=master, 1=worker)
LOCAL_RANK=0                # GPU index on local machine
```

## 🧪 Testing & Validation

### Verify CUDA and NCCL (WSL)
```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('NCCL:', torch.distributed.is_nccl_available())"
```

### Test Network Connectivity
```bash
# From Node 02 to Node 01
ping 192.168.29.67

# Check if port is listening (on Node 01 after starting master)
ss -tulpn | grep 29500
```

### Verify WSL Mirrored Networking
```bash
# Your WSL IP should match Windows IP
ip addr show eth0 | grep "inet "
```

### Single-Node Test (Before Multi-Node)
```bash
# Test on one machine with 1 GPU
torchrun --nproc_per_node=1 --nnodes=1 train_torchrun.py
```

## 📊 Performance

### NCCL Backend (Recommended)
- **Bandwidth**: 10-50 GB/s (GPU-to-GPU direct)
- **Latency**: < 10μs (local network)
- **Use Case**: Production training, large models
- **Requirements**: WSL2 mirrored networking

### Gloo Backend (Fallback)
- **Bandwidth**: 1-10 GB/s (CPU-mediated)
- **Latency**: ~100μs
- **Use Case**: Development, compatibility testing
- **Requirements**: Standard WSL2 NAT networking

### Network Recommendations
- ✅ **Best**: Direct Ethernet cable (10 Gbps)
- ✅ **Good**: WiFi 6 on same router (1-2 Gbps)
- ⚠️ **Avoid**: WiFi with AP isolation enabled

## 🐛 Troubleshooting

### Common Issues & Solutions

#### ❌ Ping Fails Between Nodes
**Symptoms:** `ping 192.168.29.67` times out

**Causes & Fixes:**
1. **Router AP Isolation**: 
   - Access router admin (`http://192.168.29.1` for Jio Fiber)
   - Disable "AP Isolation" or "Client Isolation"
   - Reboot router

2. **Windows Firewall**:
   ```powershell
   Set-NetFirewallProfile -Profile Private -Enabled False
   ```

3. **Antivirus Blocking** (Norton/Avira):
   - Temporarily disable antivirus firewall
   - Add exceptions for Python and ports 20000-40000

4. **Alternative**: Use direct Ethernet cable between laptops

#### ❌ NCCL Connection Timeout
**Error:** `The client socket has timed out after 60000ms`

**Solutions:**
1. Ensure both nodes have WSL2 mirrored networking enabled
2. Verify firewall is disabled: `Get-NetFirewallProfile`
3. Check NCCL environment variables are set on both nodes
4. Use `--master_addr` and `--master_port` (not `--rdzv_endpoint`)

#### ❌ NCCL Connection Reset / Socket Error
**Error:** `socketStartConnect: Connect to IP<port> failed : Software caused connection abort`

**Solutions:**
1. Disable Windows Firewall on both nodes (see Step 5)
2. Disable antivirus (Norton, Avira, etc.)
3. Verify mirrored networking: `ip addr show eth0 | grep "inet "`
4. Add firewall rules for ports 20000-40000

#### ❌ CUDA Not Available
**Error:** `CUDA: False` when checking PyTorch

**Solutions:**
1. Install NVIDIA CUDA on WSL driver from [nvidia.com/cuda/wsl](https://developer.nvidia.com/cuda/wsl)
2. Verify with: `nvidia-smi` (should show GPU)
3. Reinstall PyTorch with CUDA:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

#### ❌ NCCL Not Available
**Error:** `NCCL: False` when checking PyTorch

**Solution:** Install PyTorch with CUDA support (NCCL is included)

#### ❌ WSL Mirrored Networking Not Working
**Error:** WSL IP still shows `172.x.x.x` instead of Windows IP

**Solutions:**
1. Verify Windows version: Build 22621+ required
   ```powershell
   [System.Environment]::OSVersion.Version
   ```
2. Check `.wslconfig` location: `C:\Users\<YourUser>\.wslconfig`
3. Ensure file has correct format (no BOM, ASCII encoding)
4. Full WSL restart:
   ```powershell
   wsl --shutdown
   Start-Sleep -Seconds 30
   wsl
   ```

#### ❌ Port Already in Use
**Error:** `OSError: [Errno 98] Address already in use`

**Solutions:**
1. Find process using port: `sudo lsof -i :29500`
2. Kill the process: `sudo kill -9 <PID>`
3. Or use a different port (e.g., 29501)

### Network Requirements Checklist

- ✅ Both machines on same WiFi network or direct Ethernet
- ✅ Windows Firewall disabled for Private network
- ✅ Antivirus firewall disabled (Norton, Avira, etc.)
- ✅ WSL2 mirrored networking enabled (Build 22621+)
- ✅ Port 29500 accessible (rendezvous)
- ✅ Ports 20000-40000 accessible (NCCL communication)
- ✅ No VPN or proxy interference
- ✅ Router AP Isolation disabled

## 🔄 Development Workflow

1. **Start Simple**: Test single-node first, then multi-node
2. **Enable Mirrored Networking**: Critical for NCCL to work
3. **Disable Firewalls**: Start with all firewalls off, add rules later
4. **Check Connectivity**: Ensure nodes can ping each other
5. **Monitor Logs**: Use `NCCL_DEBUG=INFO` to diagnose issues
6. **Scale Gradually**: 2 nodes → 3 nodes → N nodes

## 📈 Performance Tips

### Optimize Network
- ✅ Use direct Ethernet connection for lowest latency
- ✅ Disable power-saving on network adapters
- ✅ Use dedicated network interface for distributed training
- ✅ Monitor bandwidth: `iperf3` between nodes

### Optimize Training
- 🎯 Batch size: Larger batches better utilize multi-GPU
- 🎯 Gradient accumulation: Simulate larger batches
- 🎯 Mixed precision: Use FP16 to reduce communication overhead
- 🎯 Efficient collectives: Use `all_reduce` over `all_gather` when possible

### Monitor Performance
```bash
# GPU utilization
nvidia-smi -l 1

# Network traffic
iftop -i eth0

# NCCL performance test
nccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 1
```

## 🤝 Contributing

Contributions are welcome! This project is the result of extensive troubleshooting and experimentation with WSL2 + NCCL multi-node setups.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-improvement`)
3. Test on both master and worker nodes
4. Commit your changes (`git commit -m 'Add amazing improvement'`)
5. Push to the branch (`git push origin feature/amazing-improvement`)
6. Open a Pull Request

## 📝 License

MIT License - feel free to use this in your projects!

## 🙏 Acknowledgments

- [PyTorch Distributed](https://pytorch.org/tutorials/beginner/dist_overview.html) - Official documentation
- [NVIDIA NCCL](https://developer.nvidia.com/nccl) - High-performance GPU communication
- [WSL2 Mirrored Networking](https://learn.microsoft.com/en-us/windows/wsl/networking) - Microsoft WSL docs
- Community contributions and testing

---

## 💝 Made with Love

**Created by [Amardeep Singh Sidhu](https://github.com/thefirehacker) ([@thefirehacker](https://github.com/thefirehacker))**

*Building AI solutions at [@AIEdX](https://github.com/AIEdX) & [@bubblspace](https://github.com/bubblspace)*

🌐 [bubblspace.com](https://bubblspace.com) | 🐦 [@thefirehacker](https://twitter.com/thefirehacker) | 💼 [LinkedIn](https://linkedin.com/in/thefirehacker)

---

⭐ If this project helped you, consider giving it a star on GitHub!

📧 Questions? Open an issue or reach out at contact@aiedx.com
