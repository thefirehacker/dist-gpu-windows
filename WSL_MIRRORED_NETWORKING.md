# WSL2 Mirrored Networking for NCCL Multi-Node Training

## Overview

WSL2's default NAT networking causes issues with PyTorch NCCL multi-node training. **Mirrored networking mode** resolves this by making WSL's network stack behave like native Linux, allowing direct connections between nodes.

**Source:** [Discord discussion](https://discordapp.com/channels/1334797380058611746/1334851200834605136/1426200116502462475)

---

## Requirements

- **Windows 11 version 22H2 or later** (released September 2022)
- WSL2 installed
- Both machines must be configured with mirrored networking

---

## Setup Instructions

### 1. Check Windows Version

Open PowerShell and run:

```powershell
winver
```

You should see **Windows 11 22H2** or later (Build 22621 or higher).

If you're on an older version, update Windows first.

### 2. Create/Edit `.wslconfig` File

**On BOTH Node 01 and Node 02:**

Open PowerShell and run:

```powershell
notepad $env:USERPROFILE\.wslconfig
```

Add or update the file with:

```ini
[wsl2]
networkingMode=mirrored
```

**Complete example `.wslconfig`:**

```ini
[wsl2]
# Enable mirrored networking for direct network access
networkingMode=mirrored

# Optional: Reduce memory usage
memory=8GB
processors=4

# Optional: Disable automatic localhost port forwarding (mirrored mode handles this)
localhostForwarding=false
```

Save and close the file.

### 3. Shutdown WSL Completely

In PowerShell:

```powershell
wsl --shutdown
```

Wait 10 seconds for WSL to fully shut down.

### 4. Restart WSL

Open a new WSL terminal or run:

```powershell
wsl
```

### 5. Verify Mirrored Networking

In WSL, check your IP:

```bash
hostname -I
```

You should now see your **Windows host IP** (e.g., `192.168.29.67`) instead of an internal WSL IP (like `172.x.x.x`).

---

## Updated Multi-Node Setup

With mirrored networking, the setup becomes much simpler!

### Node 01 - Master Setup

**No port forwarding needed!** WSL uses the same network as Windows.

**In WSL:**

```bash
cd ~/dist-gpu-windows
source .venv/bin/activate

# Get your IP (should now match Windows IP)
hostname -I

# Run master with NCCL
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=192.168.29.67:29400 \
  train_torchrun.py
```

Use your actual IP from `hostname -I`.

### Node 02 - Worker Setup

**In WSL:**

```bash
cd ~/dist-gpu-windows
source .venv/bin/activate

# Run worker with NCCL
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

torchrun --nnodes=2 --node_rank=1 --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=192.168.29.67:29400 \
  train_torchrun.py
```

Replace `192.168.29.67` with Node 01's IP.

### Firewall Configuration

You still need to open the firewall on **both** nodes:

**PowerShell as Administrator:**

```powershell
# Open rendezvous port
New-NetFirewallRule -DisplayName "PyTorch Distributed 29400" -Direction Inbound -LocalPort 29400 -Protocol TCP -Action Allow

# Open NCCL communication ports
New-NetFirewallRule -DisplayName "NCCL Communication Ports" -Direction Inbound -LocalPort 20000-40000 -Protocol TCP -Action Allow
```

---

## Benefits of Mirrored Networking

✅ **No port forwarding needed** - WSL uses the host's network directly

✅ **NCCL works properly** - Direct GPU-to-GPU communication

✅ **Same IP as Windows** - Simpler configuration

✅ **Better performance** - No NAT overhead

✅ **Cleaner networking** - WSL behaves like native Linux

---

## Troubleshooting

### `.wslconfig` not working

- Ensure the file is in `C:\Users\YourUsername\.wslconfig` (not in WSL)
- Check file encoding is UTF-8 or ASCII (not UTF-16)
- Verify Windows 11 version is 22H2 or later
- Try `wsl --shutdown` then wait 30 seconds before restarting

### Still seeing 172.x.x.x IP

- Mirrored mode might not be enabled properly
- Run `wsl --shutdown` and wait longer (1 minute)
- Check `dmesg | grep -i mirror` in WSL for mirrored networking logs
- Verify `.wslconfig` syntax is correct

### Connection still failing

- Ensure **both** machines have mirrored networking enabled
- Verify firewall rules on both machines
- Check both machines can ping each other
- Ensure both are on the same network/subnet

### Performance issues

Mirrored networking can have higher CPU usage. If needed, you can optimize:

```ini
[wsl2]
networkingMode=mirrored
dnsTunneling=true
firewall=true
autoProxy=true
```

---

## Comparison: NAT vs Mirrored Networking

| Feature | NAT Mode (Default) | Mirrored Mode |
|---------|-------------------|---------------|
| WSL IP | Different (172.x.x.x) | Same as Windows |
| Port forwarding | Required | Not needed |
| NCCL multi-node | Broken | Works |
| Gloo multi-node | Works (slow) | Works (faster) |
| Network complexity | High | Low |
| Windows version | Any | 11 22H2+ |

---

## Expected Output

With mirrored networking and NCCL, you should see:

**Node 01:**
```
Initializing with backend: nccl
NCCL version 2.x.x+cuda12.4
NCCL INFO Bootstrap : Using eth0:192.168.29.67<0>
NCCL INFO NET/Plugin : No plugin found (libnccl-net.so)
NCCL INFO NCCL_IB_DISABLE set by environment to 1
NCCL INFO NCCL_P2P_DISABLE set by environment to 1
NCCL INFO Channel 00/02 : 0 -> 1 via NET/Socket/0
NCCL INFO Connected all rings
[rank 0] world_size=2 device=cuda hostname=AIEDX-AsusTUF
[rank 0] gathered=[0, 1]
[rank 0] barrier OK; shutting down
```

**Node 02:**
```
Initializing with backend: nccl
NCCL version 2.x.x+cuda12.4
NCCL INFO Bootstrap : Using eth0:192.168.29.197<0>
NCCL INFO Connected all rings
[rank 1] world_size=2 device=cuda hostname=TUF-Node02
[rank 1] gathered=[0, 1]
[rank 1] barrier OK; shutting down
```

---

## References

- [WSL mirrored networking documentation](https://learn.microsoft.com/en-us/windows/wsl/networking#mirrored-mode-networking)
- [Discord discussion on WSL NCCL issue](https://discordapp.com/channels/1334797380058611746/1334851200834605136/1426200116502462475)
- [Windows 11 22H2 release notes](https://support.microsoft.com/en-us/windows/windows-11-version-22h2-update)

---

**Last Updated:** 2025-10-10  
**Tested on:** Windows 11 22H2, WSL2 Ubuntu 22.04, PyTorch 2.6.0+cu124

