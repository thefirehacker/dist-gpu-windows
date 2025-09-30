# Issue #01: Gloo Backend Hostname Resolution & Coordinator-Only Pattern

## 📝 **Issue Summary**
Attempting to set up distributed PyTorch training with Mac as coordinator (Rank 0) and Windows RTX 2080 as worker (Rank 1) results in multiple interconnected issues.

## 🔍 **Root Cause Analysis**

### **Primary Issue: Hostname Resolution Failure**
- **Error**: `makeDeviceForHostname(): unsupported gloo device`
- **Network Error**: `The client socket has failed to connect to [AMARDEEPS-IMAC]:12355`
- **DNS Failure**: Windows cannot resolve `Amardeeps-iMac.local` (mDNS/.local domain)

**Diagnostic Results:**
```cmd
# Windows Command Prompt
> nslookup Amardeeps-iMac.local
reliance.reliance can't find Amardeep-iMac.local : non existent domain
```

### **Secondary Issue: Architecture Pattern Mismatch**
- **Desired**: Mac as pure coordinator (no training computation)
- **Current Implementation**: Uses `all_gather` operations requiring ALL ranks to participate
- **Conflict**: PyTorch DDP `all_gather` mandates every rank performs computation

## 🛠 **Technical Details**

### **Environment**
- **Mac (Coordinator)**: macOS, PyTorch 2.8.0, IP: 192.168.29.234
- **Windows (Worker)**: Windows 10, PyTorch with CUDA, RTX 2080
- **Backend**: Gloo (only available backend)
- **Network**: Same LAN, port 12355

### **PyTorch Backend Availability**
```python
# Mac Results
Gloo available: True
NCCL available: False  
MPI available: False
```

### **Gloo Backend Behavior**
The Gloo backend's `makeDeviceForHostname()` function:
1. Takes IP address (192.168.29.234) from `MASTER_ADDR`
2. Performs reverse DNS lookup: IP → hostname
3. Gets `Amardeeps-iMac.local` from system
4. Attempts forward DNS lookup: hostname → IP
5. **Fails** because Windows cannot resolve `.local` domains (requires mDNS/Bonjour)

## 🚨 **Failed Solutions - TCPStore Approach**

### **Attempted Fix 1: TCPStore Rendezvous with use_libuv=False**
```python
# Mac (Controller - TCPStore master)
from torch.distributed import TCPStore
LOCAL_IP = "192.168.29.234"
store = TCPStore(host_name=LOCAL_IP, port=12355, world_size=2, is_master=True, use_libuv=False)

# Windows (Worker - TCPStore client)
store = TCPStore(host_name="192.168.29.234", port=12355, world_size=2, is_master=False, use_libuv=False)
dist.init_process_group(backend="gloo", store=store, rank=1, world_size=2)
```
**Error**: `The client socket has failed to connect to [AMARDEEPS-IMAC]:12355 (system error: 10049)`
**Root Cause**: Even with `use_libuv=False`, Gloo's internal `makeDeviceForHostname()` still performs reverse DNS lookup, converting IP to hostname

### **Attempted Fix 2: Force IPv4 with Environment Variables**
```python
# Windows worker.py
os.environ['GLOO_SOCKET_FAMILY'] = 'AF_INET'
os.environ['GLOO_USE_IPV6'] = '0'
os.environ['GLOO_SOCKET_IFNAME'] = 'Wi-Fi'  # Explicit network adapter
os.environ['TP_SOCKET_IFNAME'] = 'Wi-Fi'
```
**Result**: Same 10049 error - Gloo still converts IP → hostname internally

### **Attempted Fix 3: Windows hosts File Entry**
```bash
# C:\Windows\System32\drivers\etc\hosts
192.168.29.234    AMARDEEPS-IMAC Amardeeps-iMac.local
```
**Result**: Partial success - hostname resolves, but Gloo's cross-OS device binding still fails

### **Attempted Fix 4: Clear Hostname Environment Variables**
```python
# Tried clearing hostname environment variables
hostname_vars = ['HOSTNAME', 'HOST', 'COMPUTERNAME']
for var in hostname_vars:
    if var in os.environ:
        del os.environ[var]

os.environ['MASTER_ADDR'] = "192.168.29.234"  # Force IP
```
**Result**: Gloo ignores this and performs internal hostname resolution anyway

### **Attempted Fix 5: TCP Init Method**
```python
init_method = f'tcp://192.168.29.234:12355'
dist.init_process_group(backend='gloo', init_method=init_method, rank=1, world_size=2)
```
**Result**: Same `makeDeviceForHostname()` error

### **Attempted Fix 6: TCPStore with Mac IP as host_name**
```python
# Mac
store = TCPStore(host_name="192.168.29.234", port=12355, world_size=2, is_master=True, use_libuv=False)
```
**Result**: Still fails - Gloo's reverse DNS happens AFTER TCPStore connection

## 🔬 **Deep Dive: Gloo Cross-OS Issue**

### **The Gloo makeDeviceForHostname() Problem**
Gloo's internal code path (in C++):
1. Receives IP address `192.168.29.234` from TCPStore or init_method
2. Calls `gethostbyaddr()` for reverse DNS: `192.168.29.234` → `AMARDEEPS-IMAC` (or `Amardeeps-iMac.local`)
3. Attempts to create socket device using hostname
4. On Windows, tries to resolve `AMARDEEPS-IMAC` back to IP
5. **Fails with error 10049**: "The requested address is not valid in its context"

**Why It Fails Cross-OS:**
- macOS returns `.local` mDNS hostnames (e.g., `Amardeeps-iMac.local`)
- Windows cannot resolve `.local` domains without Bonjour/mDNS services
- Even when Windows hosts file has the entry, Gloo's socket binding fails due to OS-level differences
- The reverse DNS → forward DNS → socket bind cycle is deeply embedded in Gloo's C++ code

### **Error Code 10049 Analysis**
**Windows Socket Error 10049** = `WSAEADDRNOTAVAIL`
- "Cannot assign requested address"
- Occurs when Gloo tries to bind a socket to a hostname that Windows doesn't recognize as valid
- Not a network/firewall issue - it's an address validity issue at the OS level

## 🔄 **Current Code Pattern Issue**

### **Existing Implementation**
```python
# Current approach uses all_gather (WRONG for coordinator pattern)
def test_all_gather_basic():
    if rank == 0:  # Mac coordinator
        device = torch.device("mps")
        tensor_list = [torch.zeros(3, dtype=torch.int64) for _ in range(world_size)]
        input_tensor = torch.tensor([rank*100, rank*100+10, rank*100+20])
        dist.all_gather(tensor_list, input_tensor)  # ❌ Requires Mac to compute
    
    elif rank == 1:  # Windows worker  
        device = torch.device("cuda:0")
        # Same all_gather call - both ranks must participate
```

### **Architecture Mismatch**
- **Problem**: `all_gather` is a collective operation requiring ALL ranks to participate in computation
- **Desired**: Mac as pure coordinator without training computation
- **Reality**: PyTorch DDP assumes homogeneous participation

## 🎯 **Viable Solutions**

### **Solution 1: Parameter Server Pattern (Recommended)**
```python
# Use PyTorch RPC framework instead of DDP
import torch.distributed.rpc as rpc

# Mac - Parameter Server
def parameter_server():
    # Store model parameters
    # Receive gradients from workers via RPC
    # Send updated parameters back
    # No collective operations needed

# Windows - Worker
def worker():
    # Fetch parameters from server via RPC
    # Perform training computation on GPU
    # Send gradients back to server
```

### **Solution 2: Point-to-Point Communication**
```python
# Replace all_gather with send/recv
if rank == 0:  # Mac coordinator
    result = torch.empty(tensor_size)
    dist.recv(result, src=1)  # Receive from Windows
    # Process and coordinate
    
elif rank == 1:  # Windows worker
    result = model(batch)  # GPU computation
    dist.send(result, dst=0)  # Send to Mac
```

### **Solution 3: Minimal Participation Pattern**
```python
# Mac participates minimally in collective operations
if rank == 0:  # Mac coordinator
    dummy_tensor = torch.zeros(1, device="cpu")  # Minimal computation
    dist.all_gather(tensor_list, dummy_tensor)  # Must participate
    # Handle coordination tasks
    
elif rank == 1:  # Windows worker
    gpu_result = model(batch).to("cuda")  # Heavy computation
    dist.all_gather(tensor_list, gpu_result)  # Real work
```

### **Solution 4: Network Configuration Fix**
```bash
# Windows hosts file: C:\Windows\System32\drivers\etc\hosts
192.168.29.234    Amardeeps-iMac.local
```
**Note**: Addresses hostname resolution but doesn't solve architecture pattern issue

## 🚧 **Blockers**

1. **Gloo Backend Limitation**: Cannot bypass hostname resolution in cross-platform setup
2. **DDP Framework Assumption**: Designed for homogeneous participation, not coordinator-worker pattern
3. **Network Configuration**: Windows mDNS/.local domain resolution issues
4. **Backend Options**: Limited to Gloo (NCCL unavailable on macOS)

## ✅ **Final Solution: torchrun + etcd v3 Rendezvous**

### **Why We Switched to torchrun**
The TCPStore approach was fundamentally flawed because:
1. **Gloo's cross-OS hostname resolution is unavoidable** when Mac and Windows form a process group
2. **Architectural mismatch**: Mac cannot be rank 0 if it doesn't participate in collective operations
3. **No workaround exists** for Gloo's internal reverse DNS behavior

### **torchrun + External Rendezvous Architecture**
**Key Insight**: Separate rendezvous from training process group

**Architecture:**
```
┌─────────────────────────────────────────────┐
│  Mac (M1, macOS)                            │
│  ┌─────────────────────────────────────┐    │
│  │  etcd v3 Server (Port 2379)         │    │
│  │  - HTTP/gRPC rendezvous only        │    │
│  │  - NOT part of training group       │    │
│  └─────────────────────────────────────┘    │
└─────────────────────────────────────────────┘
                    ▲
                    │ HTTP rendezvous
                    │
        ┌───────────┴───────────┐
        │                       │
┌───────▼──────────┐    ┌──────▼───────────┐
│ Windows Rank 0   │◄──►│ Windows Rank 1   │
│ (RTX 2080)       │    │ (RTX 2080)       │
│                  │    │                  │
│ Gloo sockets     │    │ Gloo sockets     │
│ (Windows-to-     │    │ (Windows-to-     │
│  Windows only)   │    │  Windows only)   │
└──────────────────┘    └──────────────────┘
```

### **Implementation: etcd v3 with etcd-v2 Backend**

**Critical Discovery**: PyTorch has TWO etcd backends:
- `--rdzv_backend=etcd` → Uses etcd v2 API (legacy, doesn't work with etcd v3.x)
- `--rdzv_backend=etcd-v2` → Uses etcd v3 API (confusing name, but correct for modern etcd!)

**Mac Setup:**
```bash
# Install etcd v3 (already done via brew)
brew install etcd

# Start etcd v3 server
etcd --listen-client-urls http://0.0.0.0:2379 \
     --advertise-client-urls http://0.0.0.0:2379
```

**Windows Worker Setup:**
```powershell
# Install NumPy fix
pip install "numpy==1.26.4"

# Run with etcd-v2 backend (uses etcd v3 API!)
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 \
  --rdzv_backend=etcd-v2 \
  --rdzv_endpoint=192.168.29.234:2379 \
  train_torchrun.py
```

### **Why This Works**
1. **Mac never joins training group** - only hosts etcd rendezvous
2. **Windows workers connect via HTTP** - no Gloo sockets to Mac
3. **Windows-to-Windows Gloo** - same OS, no hostname issues
4. **No reverse DNS problems** - etcd uses IP addresses only
5. **Achieves coordinator pattern** - Mac coordinates without participating in training

### **Alternative: c10d Rendezvous (No etcd needed)**
For simpler setups without Mac involvement:

```powershell
# Windows Rank 0 (rendezvous host)
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=0.0.0.0:29400 \
  train_torchrun.py

# Windows Rank 1
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=RANK0_IP:29400 \
  train_torchrun.py
```

**c10d vs etcd:**
- **c10d**: Simpler, no dependencies, Windows-only rendezvous
- **etcd**: Production-ready, fault-tolerant, Mac can monitor/control

## 📊 **Impact Assessment**

- **Code Changes**: Significant rewrite required
- **Learning Value**: Better understanding of distributed ML architectures
- **Performance**: More suitable for heterogeneous setups
- **Scalability**: Easier to add multiple workers later

## 🔗 **References**

- [PyTorch Distributed Documentation](https://pytorch.org/docs/stable/distributed.html)
- [PyTorch RPC Framework](https://pytorch.org/docs/stable/rpc.html)
- [Parameter Server Pattern in ML](https://www.cs.cmu.edu/~muli/file/parameter_server_osdi14.pdf)

---
**Status**: Open  
**Priority**: High  
**Assigned**: Research team  
**Created**: 2025-01-26  
**Updated**: 2025-01-26
