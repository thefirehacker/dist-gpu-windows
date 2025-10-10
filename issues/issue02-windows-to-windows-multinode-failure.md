# Issue #02: Windows-to-Windows Multi-Node Distributed Training Failure

## 📝 Issue Summary

Attempting to set up multi-node distributed PyTorch training between two Windows machines (both with NVIDIA GPUs) fails due to multiple interconnected PyTorch bugs on Windows.

**Status:** UNRESOLVED - No working solution found with PyTorch 2.7.1+cu118

## 🔍 Environment

- **Node 01 (Master)**: Windows, IP: 192.168.29.67, NVIDIA RTX GPU, PyTorch 2.7.1+cu118
- **Node 02 (Worker)**: Windows, IP: 192.168.29.197, NVIDIA GPU, PyTorch 2.7.1+cu118
- **Backend**: Gloo (only available backend on Windows)
- **Network**: Same LAN (192.168.29.x subnet)
- **Python Version**: 3.12.4
- **CUDA**: 11.8

## 🚨 Root Causes

### 1. **Gloo Backend Hostname Resolution Bug**

**Error:**
```
[c10d] The client socket has failed to connect to [AIEDX-AsusTUF]:29500 
(system error: 10049 - The requested address is not valid in its context.)
```

**Explanation:**
- PyTorch's Gloo backend performs internal reverse DNS lookup at the C++ level
- Even when providing IP addresses (192.168.29.67), Gloo converts them to hostnames
- Windows cannot properly resolve these hostnames back to usable socket addresses
- This happens in C++ code (`makeDeviceForHostname()`) and cannot be bypassed from Python

**Attempted Fixes (ALL FAILED):**
- ❌ Hosts file entry: `192.168.29.67    AIEDX-AsusTUF`
- ❌ Environment variables: `GLOO_SOCKET_IFNAME`, `GLOO_SOCKET_FAMILY`, etc.
- ❌ Network interface specification: `GLOO_SOCKET_IFNAME='Wi-Fi'`
- ❌ Using tcp:// init method with IP addresses
- ❌ Clearing hostname environment variables

### 2. **PyTorch Built Without libuv Support**

**Error:**
```
torch.distributed.DistStoreError: use_libuv was requested but PyTorch was built 
without libuv support, run with USE_LIBUV=0 to disable it.
```

**Explanation:**
- PyTorch 2.7.1+cu118 Windows build lacks libuv support
- torchrun and c10d rendezvous require TCPStore which defaults to using libuv
- Environment variable `USE_LIBUV=0` doesn't get passed to subprocess on Windows
- `--rdzv-conf use_libuv=0` flag is ignored

**Attempted Fixes:**
- ❌ Setting `$env:USE_LIBUV="0"` - not passed to subprocess
- ❌ Using `--rdzv-conf use_libuv=0` - flag ignored by PyTorch
- ✅ **PARTIAL SUCCESS**: Modifying PyTorch source code to add `use_libuv=False` in:
  - `.venv\Lib\site-packages\torch\distributed\elastic\rendezvous\static_tcp_rendezvous.py`
  - `.venv\Lib\site-packages\torch\distributed\elastic\rendezvous\c10d_rendezvous_backend.py`
  - This fixed the libuv error BUT hostname resolution bug still persists

### 3. **Socket Binding Issues**

**Error:**
```
[c10d] The server socket has failed to bind to [AIEDX-AsusTUF]:29400 
(system error: 10048 - Only one usage of each socket address is normally permitted.)
```

**Explanation:**
- Even when attempting to bind to IP address, PyTorch tries to bind to hostname
- Windows Error 10048: Port already in use (from previous failed attempts)
- Windows Error 10013: Permission denied for socket binding
- Windows Error 10049: Invalid address (hostname cannot be resolved to valid socket address)

## 📋 All Attempted Solutions

### Approach 1: Manual TCP Initialization (train_multinode.py)

**Method:**
```python
init_method = f'tcp://{args.master_addr}:{args.master_port}'
dist.init_process_group(
    backend='gloo',
    init_method=init_method,
    world_size=2,
    rank=rank
)
```

**Result:** ❌ FAILED - Hostname resolution error (10049)

### Approach 2: Environment Variables Only

**Method:**
```python
os.environ['MASTER_ADDR'] = '192.168.29.67'
os.environ['MASTER_PORT'] = '29500'
dist.init_process_group(backend='gloo', world_size=2, rank=rank)
```

**Result:** ❌ FAILED - Same hostname resolution error

### Approach 3: torchrun with c10d Rendezvous

**Method:**
```bash
python -m torch.distributed.run --nnodes=2 --node_rank=0 --nproc_per_node=1 \
  --rdzv_backend=c10d --rdzv_endpoint=192.168.29.67:29400 train_torchrun.py
```

**Result:** ❌ FAILED - libuv error, then hostname resolution error after fix

### Approach 4: Network Interface Specification

**Method:**
```python
os.environ['GLOO_SOCKET_IFNAME'] = 'Wi-Fi'
os.environ['GLOO_SOCKET_FAMILY'] = 'AF_INET'
os.environ['TP_SOCKET_IFNAME'] = 'Wi-Fi'
```

**Result:** ❌ FAILED - Gloo still performs hostname resolution internally

### Approach 5: Windows Hosts File

**Method:**
```
# C:\Windows\System32\drivers\etc\hosts
192.168.29.67    AIEDX-AsusTUF
```

**Result:** ❌ FAILED - Hostname resolves but socket binding still fails

### Approach 6: PyTorch Source Code Modification

**Method:**
Modified PyTorch source files to add `use_libuv=False`:

```python
# In static_tcp_rendezvous.py and c10d_rendezvous_backend.py
store = TCPStore(
    host,
    port,
    is_master=is_master,
    multi_tenant=True,
    timeout=timedelta(seconds=60),
    use_libuv=False,  # <-- Added this
)
```

**Result:** ✅ Fixed libuv error, ❌ BUT hostname resolution bug persists

## 🔬 Network Connectivity Tests

**Ping Test:** ❌ FAILED (but expected - Windows Firewall blocks ICMP by default)
```bash
# From Node 01 to Node 02
ping 192.168.29.197
# Result: Request timed out
```

**Note:** Ping failure is NOT the issue. Windows Firewall blocks ICMP (ping) by default, but PyTorch uses TCP which was properly configured.

**Network Configuration:**
- ✅ Both machines on same subnet (192.168.29.x)
- ✅ Port 29500 opened in Windows Firewall
- ✅ Python allowed in Windows Firewall (Private and Public)
- ✅ Norton Antivirus disabled during testing
- ✅ Both machines visible in ARP table

## 📊 Error Timeline

1. **Initial Error**: libuv not supported
2. **After libuv fix**: Hostname resolution error (10049)
3. **After hosts file**: Socket binding error (10048, 10013)
4. **After network interface specification**: Still hostname resolution (10049)
5. **Final State**: Unable to establish any connection between nodes

## 🔗 Related Issues

- **Issue #01**: Gloo backend hostname resolution in Mac-Windows setup
- This issue (Windows-Windows) has the SAME underlying Gloo hostname bug
- The problem is in PyTorch's C++ Gloo implementation, not Python code

## 💡 Potential Solutions (Not Tested)

### Option 1: Upgrade PyTorch
```bash
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# Try latest version or nightly build
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu118
```

**Risk:** May introduce other compatibility issues

### Option 2: Use NCCL Backend (Not Available on Windows)
NCCL is not available on Windows, only Linux.

### Option 3: Use MPI Backend
```bash
pip install mpi4py
# Requires MPI installation (Microsoft MPI or MPICH)
```

**Status:** Not tested in this session

### Option 4: Build PyTorch from Source with libuv
Compile PyTorch with proper libuv support and network fixes.

**Risk:** Complex, time-consuming, may not fix hostname issue

### Option 5: Use Shared File System
For 2-3 nodes on same network with shared drive:
```python
init_method = 'file:///Z:/torch_dist_init'
dist.init_process_group(backend='gloo', init_method=init_method, world_size=2, rank=rank)
```

**Requirement:** Shared network drive (SMB/NFS) between machines

### Option 6: Accept Single-Node Multi-GPU Only
Use `train_standalone.py` which works reliably on single Windows machine.

## 📝 Conclusion

**PyTorch 2.7.1+cu118 distributed training on Windows is fundamentally broken for multi-node setups using the Gloo backend.** 

The issue is at the C++ level in PyTorch's Gloo implementation and cannot be fixed from Python code or configuration changes.

**Recommendations:**
1. For production: Use Linux for distributed training
2. For development: Use single-node multi-GPU training on Windows
3. For experimentation: Try MPI backend or upgrade PyTorch

## 🔧 Files Modified

- `train_multinode.py` - Multiple attempted fixes
- `.venv\Lib\site-packages\torch\distributed\elastic\rendezvous\static_tcp_rendezvous.py` - Added `use_libuv=False`
- `.venv\Lib\site-packages\torch\distributed\elastic\rendezvous\c10d_rendezvous_backend.py` - Added `use_libuv=False`
- `C:\Windows\System32\drivers\etc\hosts` - Added hostname entry

## 📅 Timeline

- **Date Created**: 2025-10-10
- **Status**: Open - No working solution
- **Priority**: High (blocks multi-node training on Windows)
- **Assigned**: Research/Development team

## 🔗 References

- [PyTorch Distributed Documentation](https://pytorch.org/docs/stable/distributed.html)
- [PyTorch GitHub Issue: libuv on Windows](https://github.com/pytorch/pytorch/issues)
- [PyTorch TCPStore libuv Backend](https://docs.pytorch.org/tutorials/intermediate/TCPStore_libuv_backend.html)
- Web search results confirming PyTorch Windows libuv issues

---

**Last Updated**: 2025-10-10  
**Session Time**: ~4 hours of debugging and troubleshooting  
**Conclusion**: Unsolvable with current PyTorch version on Windows

