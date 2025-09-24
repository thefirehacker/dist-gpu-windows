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

## 🚨 **Failed Solutions**

### **Attempted Fix 1: Force IP-Only Mode**
```python
# Tried clearing hostname environment variables
hostname_vars = ['HOSTNAME', 'HOST', 'COMPUTERNAME']
for var in hostname_vars:
    if var in os.environ:
        del os.environ[var]

# Tried explicit IP settings
os.environ['MASTER_ADDR'] = "192.168.29.234"  # Direct IP
```
**Result**: Gloo still performs internal hostname resolution

### **Attempted Fix 2: TCP Init Method**
```python
init_method = f'tcp://{master_ip}:12355'
dist.init_process_group(backend='gloo', init_method=init_method, rank=1, world_size=2)
```
**Result**: Same hostname resolution error

### **Attempted Fix 3: Environment Variable Method**
```python
dist.init_process_group(backend='gloo')  # Using env vars only
```
**Result**: Same hostname resolution error

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

## 🎯 **Recommended Resolution**

**Implement Solution 1 (Parameter Server Pattern)**:
1. Abandon current DDP `all_gather` approach
2. Implement PyTorch RPC-based parameter server
3. Mac acts as true coordinator using `torch.distributed.rpc`
4. Windows worker uses RPC calls for parameter exchange
5. Eliminates hostname resolution issues
6. Achieves desired coordinator-only pattern

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
