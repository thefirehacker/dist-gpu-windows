# Fix for PyTorch Distributed Training on Windows

## Problem
PyTorch 2.2.0+cu118 has a broken TCPStore on your Windows system, causing all distributed training to fail with "DistNetworkError: Unknown error".

## Solutions

### Option 1: Upgrade PyTorch to Latest Version (RECOMMENDED)

```bash
# Uninstall current PyTorch
pip uninstall torch torchvision torchaudio

# Install latest PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or install with CUDA 12.1 if your driver supports it
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

After upgrading, test with:
```bash
torchrun --standalone --nproc_per_node=1 train_torchrun.py
```

### Option 2: Use File-Based Initialization (WORKAROUND)

If you can't upgrade PyTorch, use `train_file_init.py` which bypasses TCPStore:

```bash
python train_file_init.py
```

This uses `file://` init method instead of `env://` which avoids the broken TCPStore.

### Option 3: Multi-Node Training Without torchrun

For multi-node setups, modify `train_torchrun.py` to use file-based init:

```python
import tempfile
import os

# Instead of:
# dist.init_process_group(backend="gloo")

# Use:
temp_file = os.path.join(tempfile.gettempdir(), 'torch_dist_init')
dist.init_process_group(
    backend="gloo",
    init_method=f'file:///{temp_file}',
    world_size=YOUR_WORLD_SIZE,
    rank=YOUR_RANK
)
```

## Verification

After applying a fix, verify it works:

```bash
# Test basic distributed training
python train_file_init.py

# Test torchrun (after upgrade)
torchrun --standalone --nproc_per_node=1 train_torchrun.py
```

## Why This Happened

PyTorch 2.2.0 had known issues with TCPStore on Windows, particularly with:
- Windows Firewall interference
- Network adapter configuration
- libuv implementation bugs

These were fixed in later PyTorch versions (2.3+).

## Next Steps

1. **Recommended**: Upgrade to PyTorch 2.4+ or 2.5+
2. Test with the standalone command
3. If upgrading isn't possible, use the file-based workaround
