# Node 02 - Worker (Rank 1)

## Setup

### 1. Create Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu118
```

### 3. Verify Installation
```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

### 4. Get Node 01's IP Address
Ask Node 01 operator for their IP address.
Example: `192.168.29.67`

### 5. Edit `run_node1.sh`
```bash
NODE0_IP="192.168.29.67"  # Change to Node 01's IP
```

### 6. Test Connection (Optional)
```bash
ping 192.168.29.67  # Use Node 01's actual IP
```

## Run

**Start this SECOND (after Node 01 is running):**

```bash
# Option 1: Using the script
bash run_node1.sh

# Option 2: Direct command
python train_multinode.py --rank 1 --world-size 2 --master-addr 192.168.29.67
```

## Expected Output

```
Starting Node 1 (Worker)
Connecting to Node 0 at: 192.168.29.67

============================================================
Multi-Node Distributed Training Setup
============================================================
Local IP:     [YOUR_IP]
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
   Hostname: [YOUR-PC-NAME]

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

## Troubleshooting

- **Connection refused**: Node 01 not started yet or firewall blocking
- **Timeout**: Check if you can ping Node 01's IP
- **Wrong IP**: Verify Node 01's IP address with operator
