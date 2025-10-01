# Node 01 - Master (Rank 0)

## Setup

### 1. Find Your IP Address
```bash
ipconfig | findstr IPv4
```
Example output: `192.168.29.67`

### 2. Open Firewall (PowerShell as Admin)
```powershell
New-NetFirewallRule -DisplayName "PyTorch Distributed" -Direction Inbound -LocalPort 29500 -Protocol TCP -Action Allow
```

### 3. Edit `run_node0.sh`
```bash
NODE0_IP="192.168.29.67"  # Change to your actual IP
```

## Run

**Start this FIRST (before Node 02):**

```bash
bash run_node0.sh
```

## Expected Output

```
Starting Node 0 (Coordinator)
Local IP: 192.168.29.67

🔄 Initializing process group...
✅ Successfully initialized!
   Rank: 0/2
   Device: cuda
   
🧪 Test 1: All-gather operation...
   [Rank 0] Gathered ranks: [0, 1]
   
🧪 Test 2: Broadcast from rank 0...
   [Rank 0] Broadcasting: [42.0, 100.0, 256.0]
   
🧪 Test 3: Barrier synchronization...
   [Rank 0] ✅ Barrier passed!
   
🧪 Test 4: All-reduce (sum)...
   [Rank 0] After reduce (sum): 3.0
   
🎉 [Rank 0] All tests passed!
```

## Troubleshooting

- **Connection timeout**: Make sure firewall port 29500 is open
- **Address in use**: Port 29500 already taken, restart or change port
- **Node 02 can't connect**: Share your IP with Node 02 operator
