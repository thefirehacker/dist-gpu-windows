# Node 02 - Worker (Rank 1)

## Setup

### 1. Get Node 01's IP Address
Ask Node 01 operator for their IP address.
Example: `192.168.29.67`

### 2. Edit `run_node1.sh`
```bash
NODE0_IP="192.168.29.67"  # Change to Node 01's IP
```

### 3. Test Connection (Optional)
```bash
ping 192.168.29.67  # Use Node 01's actual IP
```

## Run

**Start this SECOND (after Node 01 is running):**

```bash
bash run_node1.sh
```

## Expected Output

```
Starting Node 1 (Worker)
Connecting to Node 0 at: 192.168.29.67

🔄 Initializing process group...
✅ Successfully initialized!
   Rank: 1/2
   Device: cuda
   
🧪 Test 1: All-gather operation...
   [Rank 1] Gathered ranks: [0, 1]
   
🧪 Test 2: Broadcast from rank 0...
   [Rank 1] Received: [42.0, 100.0, 256.0]
   
🧪 Test 3: Barrier synchronization...
   [Rank 1] ✅ Barrier passed!
   
🧪 Test 4: All-reduce (sum)...
   [Rank 1] After reduce (sum): 3.0
   
🎉 [Rank 1] All tests passed!
```

## Troubleshooting

- **Connection refused**: Node 01 not started yet or firewall blocking
- **Timeout**: Check if you can ping Node 01's IP
- **Wrong IP**: Verify Node 01's IP address with operator
