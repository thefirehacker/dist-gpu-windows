# Solution 01 — Mac as Pure Controller (Rendezvous Only)

Design a setup where the Mac acts strictly as a controller/orchestrator and does NOT participate in the distributed training group. Only the Windows GPU workers join the PyTorch process group and run collective ops (`all_gather`, `all_reduce`, barriers, etc.). This mirrors how labs coordinate training from a laptop while training runs on remote/rack GPUs.

---

## Architecture

```
                 ┌──────────────────────────────────────────────────┐
                 │                    Mac (Controller)              │
                 │  • Hosts rendezvous key–value store (TCPStore)   │
                 │  • Orchestrates, logs, checkpoints               │
                 │  • Does NOT join the process group               │
                 └───────────────┬──────────────────────────────────┘
                                 │   TCP (rendezvous only)
                                 │   MASTER_ADDR=Mac_IP, PORT=12355
                ┌────────────────┴────────────────┐
                │                                 │
   ┌────────────▼────────────┐        ┌───────────▼────────────┐
   │  Windows Worker #1      │        │  Windows Worker #2      │
   │  Rank = 0               │        │  Rank = 1               │
   │  Joins process group    │◄──────►│  Joins process group    │
   │  Runs GPU training      │  GLOO  │  Runs GPU training      │
   │  Collectives between    │  data  │  Collectives between    │
   │  Windows workers only   │  plane │  Windows workers only   │
   └──────────────────────────┘        └─────────────────────────┘
```

Key points:
- The Mac is NOT part of the process group. It only hosts the rendezvous (`TCPStore`).
- Rank 0 lives on Windows Worker #1 (not on the Mac).
- All collectives happen strictly between Windows workers, eliminating the Mac hostname/device issues.

---

## Why this solves the problem
- Avoids `makeDeviceForHostname()` path on the Mac; the Mac is only a TCP key–value store.
- Gloo connections form between Windows ranks only; the Mac’s hostname never enters the data plane.
- Matches the "laptop as controller, racks as trainers" pattern used in practice.

---

## Requirements
- Same LAN; open TCP port (default 12355) on the Mac.
- PyTorch 2.x on all machines. Backend: `gloo` on Windows.
- One process per Windows worker for a simple start (scale to multiple as needed).

---

## Step 1 — Start rendezvous store on the Mac (controller)
Run this on the Mac (from a terminal, or a dedicated cell in `ops.ipynb`). This does NOT start training; it only brings up the store and keeps it alive.

```bash
python3 - << 'PY'
from torch.distributed import TCPStore
import time
# world_size := number of Windows worker processes that will join the group
WORLD_SIZE = 2
store = TCPStore(hostname="0.0.0.0", port=12355, world_size=WORLD_SIZE, is_master=True)
print(f"✅ TCPStore up on 0.0.0.0:12355, expecting {WORLD_SIZE} workers")
# keep the store alive
try:
    while True:
        time.sleep(3600)
except KeyboardInterrupt:
    print("Store shutting down…")
PY
```

Notes:
- Use your Mac's IP (e.g., `192.168.29.234`) for workers to reach this store.
- `WORLD_SIZE` = total number of Windows worker processes (e.g., 2 for two laptops, or 2 procs across machines).

---

## Step 2 — Initialize workers on Windows (process group members only)
Modify `worker.py` to use `TCPStore` and explicit `rank/world_size`. Example skeleton for each Windows worker:

```python
import torch
import torch.distributed as dist

MAC_IP = "192.168.29.234"   # <- set to your Mac's IP
PORT = 12355
WORLD_SIZE = 2               # total number of Windows worker processes
RANK = 0                     # unique per worker: 0 for first worker, 1 for second, …

# Rendezvous only — Mac runs is_master=True; workers always is_master=False
store = dist.TCPStore(MAC_IP, PORT, WORLD_SIZE, is_master=False)

dist.init_process_group(
    backend="gloo",
    store=store,
    rank=RANK,
    world_size=WORLD_SIZE,
    timeout=torch.distributed.default_pg_timeout,
)

print(f"Joined process group as rank {RANK}/{WORLD_SIZE}")
# From here, all collectives (all_gather/all_reduce/barrier) involve ONLY Windows ranks
```

Assign ranks:
- Windows Worker #1 → `RANK = 0` (this becomes the "rank 0" node for collectives)
- Windows Worker #2 → `RANK = 1`

---

## Step 3 — Training loop (Windows only)
Continue using your existing collective code on Windows. For example:

```python
rank = dist.get_rank()
world_size = dist.get_world_size()

# Device selection on Windows workers
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Example: all_gather across Windows ranks
tensor_list = [torch.zeros(3, dtype=torch.int64, device=device) for _ in range(world_size)]
input_tensor = torch.tensor([rank*100, rank*100+10, rank*100+20], dtype=torch.int64, device=device)

dist.all_gather(tensor_list, input_tensor)
print(f"Rank {rank}: gathered = {tensor_list}")
```

The Mac does not call `init_process_group` and does not perform collectives.

---

## Scaling to N workers
- Set `WORLD_SIZE = N` on the Mac store and on each Windows worker.
- Launch N Windows worker processes; give each a unique `RANK ∈ [0, N-1]`.
- If a single Windows host runs multiple processes/GPUs, assign distinct ranks per process.

---

## Troubleshooting
- Port blocked: ensure macOS firewall allows inbound TCP on the chosen port (12355).
- Interface binding: on Windows hosts with multiple NICs, you can pin Gloo to a specific interface: `os.environ['GLOO_SOCKET_IFNAME'] = '<adapter_name>'` (optional; only if you see binding issues).
- Hostname/mDNS: this design avoids hostname resolution in the data plane; you do not need `.local` resolution on Windows.
- Timeouts: increase `timeout=` in `init_process_group` if startup is slow.

---

## Why not put rank 0 on the Mac?
- In DDP/collectives, rank 0 participates in all ops. Placing rank 0 on the Mac would force the Mac into the training group (and reintroduce the Gloo/hostname/device issues).
- Hosting a standalone rendezvous (`TCPStore`) on the Mac cleanly separates coordination from training.

---

## Summary
- Mac = pure controller (runs TCPStore only), not a member of the training group.
- Windows workers = all members of the process group; rank 0 is on Windows.
- All collectives run strictly between Windows ranks.
- Mirrors real-world "laptop orchestrates, racks train" workflows.
