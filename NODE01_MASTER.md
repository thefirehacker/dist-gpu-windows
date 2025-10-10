# Node 01 - Master (Rank 0)

## Setup

### 1. Activate Virtual Environment
```bash
.venv\Scripts\activate
```

### 2. Find Your IP Address
```bash
ipconfig | findstr IPv4
```
Example output: `192.168.29.67`

### 3. Open Firewall (PowerShell as Admin)
```powershell
New-NetFirewallRule -DisplayName "PyTorch Distributed" -Direction Inbound -LocalPort 29500 -Protocol TCP -Action Allow
```

### 4. Edit `run_node0.sh`
```bash
NODE0_IP="192.168.29.67"  # Change to your actual IP
```

## Run

**Start this FIRST (before Node 02):**

```bash
# Option 1: Using the script
bash run_node0.sh

# Option 2: Direct command
python train_multinode.py --rank 0 --world-size 2 --master-addr 192.168.29.67
```

## Expected Output

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

## Troubleshooting

- **Connection timeout**: Make sure firewall port 29500 is open
- **Address in use**: Port 29500 already taken, restart or change port
- **Node 02 can't connect**: Share your IP with Node 02 operator
