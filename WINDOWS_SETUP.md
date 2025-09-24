# Windows RTX 2080 Setup Instructions

This guide helps you set up your Windows machine with RTX 2080 GPU for distributed PyTorch training with your Mac coordinator.

## Prerequisites

- Windows 10/11 with RTX 2080 GPU
- Python 3.8+ installed
- Both machines on the same network
- Mac coordinator running first

## Step 1: Install CUDA and PyTorch

### Option A: Install CUDA Toolkit (Recommended)
1. Download CUDA Toolkit 11.8 from [NVIDIA Developer](https://developer.nvidia.com/cuda-11-8-0-download-archive)
2. Install the toolkit following the installer instructions
3. Verify installation:
   ```cmd
   nvcc --version
   ```

### Option B: Use PyTorch with CUDA (Easier)
Skip CUDA toolkit installation and use PyTorch's bundled CUDA:

```cmd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Step 2: Install Dependencies

```cmd
# Install required packages
pip install -r requirements.txt

# Or install individually:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install nbdistributed numpy
```

## Step 3: Verify GPU Setup

Run this Python script to verify your GPU setup:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

Expected output:
```
CUDA available: True
GPU count: 1
GPU name: GeForce RTX 2080
```

## Step 4: Network Configuration

### Find Your Mac's IP Address
On your Mac, run this in the notebook or terminal:
```python
import socket
hostname = socket.gethostname()
local_ip = socket.gethostbyname(hostname)
print(f"Mac IP: {local_ip}")
```

### Configure Windows Firewall
1. Open Windows Defender Firewall
2. Click "Allow an app or feature through Windows Defender Firewall"
3. Click "Change settings" → "Allow another app"
4. Add Python.exe and allow it for both Private and Public networks
5. Or temporarily disable firewall for testing

### Test Network Connectivity
On Windows, test connection to Mac:
```cmd
ping YOUR_MAC_IP_HERE
telnet YOUR_MAC_IP_HERE 12355
```

## Step 5: Run the Worker

1. **First, start the Mac coordinator** (run the notebook cells)
2. **Then, on Windows, run:**
   ```cmd
   python worker.py
   ```

The worker will:
- Check CUDA setup
- Ask for your Mac's IP address
- Connect to the distributed training session
- Run distributed operations tests

## Troubleshooting

### Common Issues

#### 1. "CUDA not available"
**Solution:**
```cmd
# Uninstall and reinstall PyTorch with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 2. "Connection refused" or "Timeout"
**Solutions:**
- Ensure Mac coordinator is running first
- Check Mac's IP address is correct
- Disable Windows Firewall temporarily
- Try different port (change MASTER_PORT in both scripts)

#### 3. "Address already in use"
**Solution:**
- Change MASTER_PORT to a different port (e.g., "12356")
- Update both Mac and Windows scripts with same port

#### 4. "Process group not initialized"
**Solution:**
- Ensure both machines are on same network
- Check firewall settings
- Verify IP addresses are correct

### Network Debugging

#### Check Network Connectivity
```cmd
# On Windows
ipconfig
ping YOUR_MAC_IP
telnet YOUR_MAC_IP 12355
```

#### Check Port Availability
```cmd
# Check if port is in use
netstat -an | findstr 12355
```

### Performance Tips

1. **Use wired connection** instead of WiFi for better stability
2. **Close unnecessary applications** to free up GPU memory
3. **Monitor GPU usage** with `nvidia-smi` command
4. **Use smaller batch sizes** if running out of GPU memory

## Expected Output

When everything works correctly, you should see:

```
🚀 Windows RTX 2080 Distributed Worker
==================================================
=== CUDA Setup Check ===
PyTorch version: 2.0.1+cu118
CUDA available: True
CUDA version: 11.8
Number of GPUs: 1
Current GPU: 0
GPU Name: GeForce RTX 2080
GPU Memory: 8.0 GB

This Windows machine IP: 192.168.1.101

==================================================
CONFIGURATION REQUIRED:
==================================================
Please enter your Mac's IP address (from the notebook output)
Example: 192.168.1.100

Enter Mac's IP address: 192.168.1.100

=== Distributed Worker Setup ===
Master Address: 192.168.1.100
Master Port: 12355
World Size: 2
This machine's rank: 1
✅ Distributed worker initialized successfully!
Rank: 1, World size: 2

=== Testing Distributed Operations ===
Rank 1 (Windows RTX 2080): Running on device cuda:0

--- Test 1: Basic All-Gather ---
Rank 1: Input tensor: tensor([100, 110, 120], device='cuda:0')
Rank 1: Gathered tensors: [tensor([0, 10, 20], device='cuda:0'), tensor([100, 110, 120], device='cuda:0')]

--- Test 2: Advanced Operations ---
Rank 1: Advanced all-gather completed in 0.0023 seconds
Rank 1: Input shape: torch.Size([2, 3])
Rank 1: Input tensor:
tensor([[10, 11, 12],
        [13, 14, 15]], device='cuda:0')

--- Test 3: Barrier Synchronization ---
Rank 1: Waiting at barrier...
Rank 1: Barrier passed! All processes synchronized.

✅ Windows worker is ready and operational!
🎯 GPU: GeForce RTX 2080
🌐 Connected to Mac coordinator at 192.168.1.100
⏳ Waiting for distributed operations...
```

## Next Steps

Once the worker is running successfully:
1. The Mac coordinator can now run distributed operations
2. Both machines will participate in all_gather operations
3. GPU computations will run on Windows RTX 2080
4. Results will be synchronized between machines

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify network connectivity between machines
3. Ensure both machines have the correct dependencies
4. Check firewall and antivirus settings
