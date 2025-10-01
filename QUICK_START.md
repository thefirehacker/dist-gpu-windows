# Quick Start - Multi-Node Training

## 📋 Files

### For Node 01 (Master - Rank 0)
- 📄 **NODE01_MASTER.md** - Read this on Node 01
- 🚀 **run_node0.sh** - Run this script

### For Node 02 (Worker - Rank 1)
- 📄 **NODE02_WORKER.md** - Read this on Node 02
- 🚀 **run_node1.sh** - Run this script

## ⚡ Quick Setup

### On Node 01 (Windows01):

1. Find your IP:
   ```bash
   ipconfig | findstr IPv4
   ```

2. Edit `run_node0.sh` with your IP

3. Open firewall (PowerShell as Admin):
   ```powershell
   New-NetFirewallRule -DisplayName "PyTorch Distributed" -Direction Inbound -LocalPort 29500 -Protocol TCP -Action Allow
   ```

4. Run:
   ```bash
   bash run_node0.sh
   ```

### On Node 02 (Windows02):

1. Get Node 01's IP address

2. Edit `run_node1.sh` with Node 01's IP

3. Run:
   ```bash
   bash run_node1.sh
   ```

## ✅ Success

Both nodes will show:
```
🎉 [Rank X] All tests passed!
```

---

For detailed instructions, see:
- **NODE01_MASTER.md** for Node 01
- **NODE02_WORKER.md** for Node 02
