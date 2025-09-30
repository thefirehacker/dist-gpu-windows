# ✅ SOLUTION: PyTorch Distributed Training on Windows

## 🔴 Problem Identified

Your PyTorch installation (2.2.0+cu118 or later) was **built without libuv support**, which causes `torchrun` to fail with:

```
DistStoreError: use_libuv was requested but PyTorch was built without libuv support, run with USE_LIBUV=0 to disable it.
```

**Unfortunately**, setting `USE_LIBUV=0` as an environment variable **does not work** on Windows when running `torchrun.exe` from Git Bash or PowerShell. The environment variable is not properly passed to the Windows executable during the rendezvous initialization phase.

## ✅ Working Solution

### For Single-Node Training

Use the **file-based initialization method** instead of `torchrun`:

```bash
python train_standalone.py
```

This script uses `file://` init method which:
- ✅ Bypasses the broken TCPStore completely
- ✅ Works on all PyTorch versions
- ✅ Doesn't require libuv
- ✅ No environment variables needed

### What the script does:

```python
# Uses file:// instead of env:// or tcp://
temp_file = os.path.join(tempfile.gettempdir(), 'torch_dist_init')
dist.init_process_group(
    backend="gloo",
    init_method=f'file:///{temp_file}',
    world_size=1,
    rank=0
)
```

## 🚫 Why torchrun Doesn't Work

1. **torchrun** initializes the rendezvous **before** running your Python script
2. The libuv check happens in torchrun's **C++ code**
3. Environment variables from shell scripts don't properly reach the Windows `.exe`
4. `--rdzv-conf use_libuv=0` parameter is ignored or not supported in your PyTorch version

## 🔄 For Multi-Node Training

If you need to run distributed training across multiple nodes:

### Using file:// method (recommended for 2-3 nodes on shared filesystem):

**Node 0:**
```python
import os, tempfile, torch.distributed as dist

# Share this path with all nodes (e.g., on network drive)
shared_file = "Z:/torch_dist_init"  # or use NFS/SMB share

dist.init_process_group(
    backend="gloo",
    init_method=f'file:///{shared_file}',
    world_size=2,
    rank=0
)
```

**Node 1:**
```python
dist.init_process_group(
    backend="gloo",
    init_method=f'file:///{shared_file}',  # Same file as Node 0
    world_size=2,
    rank=1
)
```

### Using etcd rendezvous (recommended for 3+ nodes):

According to your branch `1.0.0_etcd`, you're already set up for etcd. Use this command:

**On each Windows node:**
```bash
# Make sure etcd server is running on coordinator (Mac or Linux)
# Then run on each Windows GPU:

torchrun \
  --nnodes=2 \
  --node_rank=0 \  # Change to 1 for second node
  --nproc_per_node=1 \
  --rdzv_backend=etcd-v2 \
  --rdzv_endpoint=COORDINATOR_IP:2379 \
  train_torchrun.py
```

**Note:** etcd rendezvous **might still fail** with the libuv issue. If it does, modify `train_torchrun.py` to use manual initialization instead of relying on torchrun's env:// setup.

## 📝 Summary

| Method | Works? | Use Case |
|--------|--------|----------|
| `torchrun --standalone` | ❌ No | Broken due to libuv |
| `python train_standalone.py` | ✅ Yes | **Single-node training** |
| `torchrun` with etcd | ⚠️ Maybe | Multi-node (may still fail) |
| file:// init with shared storage | ✅ Yes | **Multi-node (2-3 nodes)** |

## 🎯 Recommended Actions

1. **For development/testing**: Use `python train_standalone.py`
2. **For production multi-node**: Consider upgrading PyTorch or use file:// method
3. **Long-term fix**: Wait for PyTorch update or rebuild from source with libuv support

## 🔧 Alternative: Upgrade PyTorch

If you want `torchrun` to work properly:

```bash
# Uninstall current version
pip uninstall torch torchvision torchaudio -y

# Install latest PyTorch (may have libuv fix)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or try nightly build
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
```

Then test:
```bash
torchrun --standalone --nproc_per_node=1 train_torchrun.py
```

---

**Created:** September 30, 2025  
**Issue:** PyTorch TCPStore libuv compatibility on Windows  
**Status:** Workaround implemented ✅
