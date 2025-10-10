# Distributed PyTorch Setup: Multi-GPU Training

A distributed PyTorch setup for running computations across multiple GPU nodes using torchrun + c10d rendezvous.

## 🎯 Overview

This project demonstrates how to set up distributed PyTorch training/inference across Windows GPU nodes:
- **Windows GPU (Primary/Rank 0)**: Acts as master node and primary worker
- **Windows GPU (Worker/Rank 1)**: Joins as worker node
- **Architecture**: Both nodes run on Windows with CUDA support

## 📁 Project Structure

```
.
├── L1-nbdistributed/
│   ├── Ops.ipynb              # DEPRECATED - Old Mac coordinator (TCPStore approach - doesn't work)
│   └── Ops_02.ipynb           # nbdistributed interactive notebook (run on GPU node)
├── train_torchrun.py          # Training script for torchrun (RECOMMENDED)
├── worker.py                   # DEPRECATED - Old Windows worker script
├── test_connection.py          # Network connectivity test
├── archive/                    # Previous versions and backups
├── WINDOWS_SETUP.md           # Detailed Windows setup instructions
└── README.md                  # This file
```

### Which Files to Use:

**✅ For Multi-Node Training:**
- `train_multinode.py` + manual TCP initialization (see Quick Start below)

**✅ For Single-Node Testing:**
- `train_standalone.py` - Test distributed training on single machine

**✅ For Interactive Development:**
- `L1-nbdistributed/Ops.ipynb` (nbdistributed) - run on a GPU node in Jupyter

**❌ Deprecated (Don't Use):**
- `train_torchrun.py` - Old torchrun approach with libuv issues
- `worker.py` - Old worker script, replaced by train_multinode.py

## 🚀 Quick Start

### Prerequisites

**Both Windows Machines:**
- Python 3.11+
- PyTorch with CUDA support
- NVIDIA GPU with drivers
- CUDA toolkit 11.8
- Virtual environment

### Setup Steps

#### 1. Install Dependencies

On both machines:
```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install PyTorch and dependencies
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu118
```

#### 2. Single Node Testing

Test on one machine first:
```bash
python train_standalone.py
```

#### 3. Multi-Node Setup

**Node 01 (Master - Rank 0):**
1. Find your IP: `ipconfig | findstr IPv4`
2. Open firewall: `New-NetFirewallRule -DisplayName "PyTorch Distributed" -Direction Inbound -LocalPort 29500 -Protocol TCP -Action Allow`
3. Start master: `python train_multinode.py --rank 0 --world-size 2 --master-addr YOUR_IP`

**Node 02 (Worker - Rank 1):**
1. Get Node 01's IP address
2. Start worker: `python train_multinode.py --rank 1 --world-size 2 --master-addr NODE01_IP`

### Quick Reference

See the detailed guides:
- **NODE01_MASTER.md** - For Node 01 (Rank 0) setup
- **NODE02_WORKER.md** - For Node 02 (Rank 1) setup
- **QUICK_START.md** - Overview and troubleshooting

---

## 🔧 Old Multi-GPU Setup (torchrun - may not work)

**On Windows GPU #1 (Primary - rank 0 + rendezvous host):**

1. Find its LAN IP:
   ```powershell
   ipconfig | findstr IPv4
   # Example output: IPv4 Address. . . . . . . . . . . : 192.168.29.67
   ```

2. Open firewall (run PowerShell as Administrator):
   ```powershell
   New-NetFirewallRule -DisplayName "Torch c10d 29400" -Direction Inbound -LocalPort 29400 -Protocol TCP -Action Allow
   ```

3. Start as rank 0:
   ```powershell
   # Use 0.0.0.0 to listen on all interfaces
   torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=0.0.0.0:29400 train_torchrun.py
   ```

**On Windows/Linux GPU #2 (Worker - rank 1):**

```powershell
# Connect to primary GPU's IP (e.g., 192.168.29.67)
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=192.168.29.67:29400 train_torchrun.py
```

**Expected output (each rank):**
```text
[rank 0] world_size=2 device=cuda hostname=PRIMARY-GPU
[rank 0] gathered=[0, 1]
[rank 0] barrier OK; shutting down

[rank 1] world_size=2 device=cuda hostname=WORKER-GPU
[rank 1] gathered=[0, 1]
[rank 1] barrier OK; shutting down
```

**Alternative: etcd rendezvous (advanced - requires Linux/x86 server)**

Note: etcd has compatibility issues on Apple M1/M2 Macs. Use a Linux server if you need etcd:

```bash
# On Linux server:
# Install etcd and start it
etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://0.0.0.0:2379

# On each Windows GPU:
pip install python-etcd==0.4.5
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 --rdzv_backend=etcd-v2 --rdzv_endpoint=LINUX_IP:2379 train_torchrun.py
```

**Recommendation:** Stick with c10d for simplicity.

#### 4. Legacy path (TCPStore notebook + worker.py)
If you still want to try notebook store + `worker.py`, run the TCPStore cell in `L1-nbdistributed/Ops.ipynb` and then execute `worker.py` on Windows with the Mac IP. The torchrun path above is recommended.

## 🔧 Configuration

### Network Settings (c10d rendezvous)
- **Port**: 29400 (rendezvous on Windows rank 0)
- **Backend**: Gloo (GPU-GPU setup)
- **Rendezvous**: c10d (no extra dependencies)

### Network Settings (legacy TCPStore)
- **Port**: 12355 (configurable)
- **Backend**: Gloo (CPU-GPU mixed setup)
- **IP Detection**: Automatic (no manual configuration needed)

### Environment Variables (set by torchrun automatically)
```bash
MASTER_ADDR=<set-by-torchrun>
MASTER_PORT=<set-by-torchrun>
WORLD_SIZE=<set-by-torchrun>
RANK=<set-by-torchrun>
LOCAL_RANK=<set-by-torchrun>
```

## 🧪 Testing

### Basic Connection Test
```python
# In Jupyter notebook after connection
result = test_all_gather_basic()
```

### Network Connectivity Test
```bash
# On Windows, test connection to Mac
python test_connection.py
```

## 📊 Features

- **Heterogeneous Setup**: Mac CPU/MPS + Windows GPU
- **Automatic IP Detection**: No manual IP configuration
- **IP-only Connections**: Avoids hostname resolution issues
- **Connection Retry Logic**: Robust connection handling
- **Device-aware Operations**: Automatic device allocation based on rank

## 🐛 Troubleshooting

### Common Issues

#### Connection Timeout
```
Error: The client socket has failed to connect
```
**Solution**: Ensure Mac coordinator is running first, then start Windows worker

#### Hostname Resolution Error
```
Error: failed to connect to AMARDEEPS-IMAC
```
**Solution**: Use the updated scripts that force IP-only connections

#### Port Already in Use
```
Error: Port 12355 not available
```
**Solution**: 
- Check if another process is using the port: `lsof -i :12355`
- Restart both coordinator and worker

#### CUDA Not Available on Windows
```
Warning: CUDA not available! Running on CPU
```
**Solution**: 
- Install CUDA toolkit
- Install PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

#### NumPy ABI Compatibility Error (Windows)
```
Error: A module that was compiled using NumPy 1.x cannot be run in NumPy 2.1.2
```
**Solution**: 
- Downgrade NumPy to 1.26.4: `pip install "numpy==1.26.4"`
- This is already included in the c10d setup steps above

### Network Requirements

- Both machines must be on the same network
- **Port 29400** must be open on Windows rank 0 (c10d rendezvous)
- **Port 12355** for legacy TCPStore setup
- **Port 2379** for etcd rendezvous (alternative)
- Firewall configuration required (see setup steps)
- No VPN interference

## 🔄 Development Workflow

1. **Make changes** to coordinator notebook or worker script
2. **Test locally** with connection verification
3. **Commit changes** (Git automatically ignores temporary files)
4. **Deploy** to both machines

## 📈 Performance Tips

- Use appropriate tensor sizes for network transfer
- Leverage device-specific optimizations (MPS on Mac, CUDA on Windows)
- Monitor network latency between machines
- Consider batch sizes for distributed operations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Test changes on both Mac and Windows
4. Submit a pull request

## 📝 License

[Add your license information here]

## 🙏 Acknowledgments

- PyTorch distributed documentation
- Gloo backend for cross-platform compatibility
- Community contributions and testing

---

For detailed Windows setup instructions, see [`WINDOWS_SETUP.md`](WINDOWS_SETUP.md).

For issues and questions, please open a GitHub issue.
