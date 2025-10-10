# Quick Start Guide - Windows Multi-Node Distributed Training

## 🎯 Overview

This guide helps you set up distributed PyTorch training across two Windows machines:
- **Node 01**: Master (Rank 0) - Your current machine
- **Node 02**: Worker (Rank 1) - Another Windows machine

## 📋 Prerequisites

Both machines need:
- Windows 10/11
- Python 3.11+
- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.8
- Same network connection

## 🚀 Step-by-Step Setup

### Step 1: Prepare Node 01 (Master - This Machine)

#### 1.1 Activate Virtual Environment
```bash
.venv\Scripts\activate
```

#### 1.2 Find Your IP Address
```bash
ipconfig | findstr IPv4
```
Note down your IP (e.g., `192.168.29.67`)

#### 1.3 Open Firewall Port
Run PowerShell as Administrator:
```powershell
New-NetFirewallRule -DisplayName "PyTorch Distributed" -Direction Inbound -LocalPort 29500 -Protocol TCP -Action Allow
```

#### 1.4 Start Master Node
```bash
python train_multinode.py --rank 0 --world-size 2 --master-addr 192.168.29.67
```

**Expected Output:**
```
============================================================
Multi-Node Distributed Training Setup
============================================================
Local IP:     192.168.29.67
Master Addr:  192.168.29.67
Master Port:  29500
Rank:         0
World Size:   2
Backend:      gloo
============================================================

[INFO] Initializing process group...
   Init method: tcp://192.168.29.67:29500
[SUCCESS] Successfully initialized!
   Rank: 0/2
   Device: cuda
   Hostname: YOUR-PC-NAME

[TEST] Test 1: All-gather operation...
   [Rank 0] Gathered ranks: [0, 1]
   
[TEST] Test 2: Broadcast from rank 0...
   [Rank 0] Broadcasting: [42.0, 100.0, 256.0]
   
[TEST] Test 3: Barrier synchronization...
   [Rank 0] [SUCCESS] Barrier passed!
   
[TEST] Test 4: All-reduce (sum)...
   [Rank 0] After reduce (sum): 3.0
   
============================================================
[SUCCESS] [Rank 0] All tests passed! Shutting down...
============================================================
```

### Step 2: Prepare Node 02 (Worker - Other Machine)

#### 2.1 Copy Project Files
Copy these files to the other Windows machine:
- `train_multinode.py`
- `requirements.txt`
- `run_node1.sh`

#### 2.2 Setup Environment
```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu118

# Verify installation
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

#### 2.3 Update Configuration
Edit `run_node1.sh` and set the master IP:
```bash
NODE0_IP="192.168.29.67"  # Use Node 01's actual IP
```

#### 2.4 Start Worker Node
```bash
python train_multinode.py --rank 1 --world-size 2 --master-addr 192.168.29.67
```

**Expected Output:**
```
============================================================
Multi-Node Distributed Training Setup
============================================================
Local IP:     [WORKER_IP]
Master Addr:  192.168.29.67
Master Port:  29500
Rank:         1
World Size:   2
Backend:      gloo
============================================================

[INFO] Initializing process group...
   Init method: tcp://192.168.29.67:29500
[SUCCESS] Successfully initialized!
   Rank: 1/2
   Device: cuda
   Hostname: [WORKER-PC-NAME]

[TEST] Test 1: All-gather operation...
   [Rank 1] Gathered ranks: [0, 1]
   
[TEST] Test 2: Broadcast from rank 0...
   [Rank 1] Received: [42.0, 100.0, 256.0]
   
[TEST] Test 3: Barrier synchronization...
   [Rank 1] [SUCCESS] Barrier passed!
   
[TEST] Test 4: All-reduce (sum)...
   [Rank 1] After reduce (sum): 3.0
   
============================================================
[SUCCESS] [Rank 1] All tests passed! Shutting down...
============================================================
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Connection Refused
**Error:** `[ERROR] Failed to initialize: Connection refused`

**Solutions:**
- Ensure Node 01 (master) is running first
- Check firewall: port 29500 must be open on Node 01
- Verify IP address is correct

#### 2. Timeout Error
**Error:** `[ERROR] Failed to initialize: Timeout`

**Solutions:**
- Test network connectivity: `ping NODE01_IP`
- Check if both machines are on same network
- Disable VPN if active

#### 3. PyTorch Not Found
**Error:** `ModuleNotFoundError: No module named 'torch'`

**Solutions:**
- Activate virtual environment: `.venv\Scripts\activate`
- Install PyTorch: `pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu118`

#### 4. CUDA Not Available
**Error:** `Device: cpu` (should be `cuda`)

**Solutions:**
- Install CUDA Toolkit 11.8
- Install PyTorch with CUDA support
- Check GPU drivers

#### 5. Unicode Error
**Error:** `UnicodeEncodeError: 'charmap' codec can't encode character`

**Solutions:**
- Use the updated `train_multinode.py` (already fixed)
- Run in PowerShell instead of Command Prompt

### Network Requirements

- Both machines must be on the same network
- Port 29500 must be open on Node 01
- No VPN interference
- Firewall configured correctly

### Verification Steps

1. **Test single-node first:**
   ```bash
   python train_standalone.py
   ```

2. **Test network connectivity:**
   ```bash
   ping NODE01_IP
   ```

3. **Check firewall:**
   ```powershell
   Get-NetFirewallRule -DisplayName "PyTorch Distributed"
   ```

## 📞 Getting Help

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all prerequisites are met
3. Test single-node setup first
4. Check network connectivity between machines

## 🎉 Success!

When both nodes show `[SUCCESS] All tests passed!`, your distributed training setup is working correctly!
