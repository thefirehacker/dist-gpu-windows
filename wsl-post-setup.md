# WSL Post-Setup Guide - Multi-Node Distributed Training

Complete setup guide for WSL Ubuntu after Ubuntu is installed and git repo is cloned.

---

## 🧩 Step-by-Step Setup

Run all commands inside WSL Ubuntu as your normal user (e.g., `firehacker`), inside the repo folder
(e.g., `~/dist-gpu-windows` or `/mnt/d/04Code/dist-gpu-windows`).

---

## 🧰 1️⃣ Update packages and install Python tools

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential python3 python3-pip python3-venv git
```

Check Python version:
```bash
python3 --version
```

**Expected:** Python 3.10.x or 3.11.x

---

## 🧱 2️⃣ Create and activate a virtual environment

```bash
cd ~/dist-gpu-windows     # adjust path to where you cloned
python3 -m venv .venv
source .venv/bin/activate
```

You should now see `(.venv)` prefix in your prompt.

---

## ⚙️ 3️⃣ Upgrade pip + wheel + setuptools

```bash
pip install --upgrade pip setuptools wheel
```

---

## ⚡ 4️⃣ Verify NVIDIA driver / CUDA access

Make sure your Windows host has the NVIDIA CUDA on WSL driver installed.

Check in WSL:
```bash
nvidia-smi
```

✅ You should see your GPU listed (RTX / GTX model).

**If not** → update the Windows NVIDIA driver from:
🔗 https://developer.nvidia.com/cuda/wsl

---

## 🧠 5️⃣ Install PyTorch + CUDA support

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

*(If you have a different CUDA runtime, e.g., 12.1 → replace `cu124` with `cu121`)*

**Verify installation:**
```bash
python -c "import torch; print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPUs:', torch.cuda.device_count())"
```

**Expected output:**
```
Torch: 2.x.x
CUDA available: True
GPUs: 1
```

**Verify NCCL availability:**
```bash
python -c "import torch; print('NCCL available:', torch.distributed.is_nccl_available())"
```

**Expected:** `NCCL available: True`

---

## 📦 6️⃣ Install repo dependencies

If your repo includes a `requirements.txt`:
```bash
pip install -r requirements.txt
```

If not, install common helpers manually:
```bash
pip install tqdm accelerate psutil
```

---

## 🔍 7️⃣ Quick sanity test

```bash
python - <<'EOF'
import torch, platform, os
print("System:", platform.platform())
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("NCCL available:", torch.distributed.is_nccl_available())
EOF
```

✅ If you see your GPU info and NCCL available — environment is ready.

---

## 🚀 8️⃣ Run a local single-node test

```bash
torchrun --standalone --nproc_per_node=1 train_torchrun.py
```

**Expected output:**
```
Initializing with backend: nccl
[rank 0] world_size=1 device=cuda hostname=AIEDX-AsusTUF
[rank 0] gathered=[0]
[rank 0] barrier OK; shutting down
```

---

## 🌐 9️⃣ Setup Port Forwarding (Windows → WSL)

### Why Port Forwarding?

WSL uses NAT networking. Your WSL has an internal IP (e.g., `172.20.5.204`) that's NOT accessible from other physical computers. Other machines can only reach your **Windows IP** (e.g., `192.168.29.67`).

**Solution:** Forward traffic from Windows IP to WSL IP.

### Get your WSL IP:

In WSL:
```bash
hostname -I
```

Example output: `172.20.5.204`

### Get your Windows IP:

In PowerShell:
```powershell
ipconfig | findstr IPv4
```

Example output: `192.168.29.67` (the one on your WiFi/Ethernet network, NOT 172.x.x.x)

### Setup Port Forwarding on Windows:

**Open PowerShell as Administrator** (Win + X → "Windows PowerShell (Admin)")

```powershell
# Forward port 29400 (rendezvous) from Windows to WSL
netsh interface portproxy add v4tov4 listenport=29400 listenaddress=0.0.0.0 connectport=29400 connectaddress=172.20.5.204

# Open Windows Firewall for port 29400
New-NetFirewallRule -DisplayName "PyTorch Distributed 29400" -Direction Inbound -LocalPort 29400 -Protocol TCP -Action Allow
```

**Replace `172.20.5.204` with your actual WSL IP from `hostname -I`**

### Verify port forwarding:

```powershell
netsh interface portproxy show all
```

**Expected output:**
```
Listen on ipv4:             Connect to ipv4:
Address         Port        Address         Port
--------------- ----------  --------------- ----------
0.0.0.0         29400       172.20.5.204    29400
```

### To remove port forwarding (if needed):

```powershell
netsh interface portproxy delete v4tov4 listenport=29400 listenaddress=0.0.0.0
```

---

## 🧱 10️⃣ Multi-Node Launch Commands

### Get Node IPs:

**On Node 01 (Master):**

In WSL:
```bash
hostname -I
```
Note the WSL IP (e.g., `172.20.5.204`)

In PowerShell:
```powershell
ipconfig | findstr IPv4
```
Note the Windows IP (e.g., `192.168.29.67`)

**On Node 02 (Worker):**

Same process - get both WSL IP and Windows IP.

---

### Launch Training:

**IMPORTANT: Run the master node from BOTH WSL nodes, NOT from Windows!**

**On Node 01 (Master) - In WSL:**
```bash
# Use localhost/127.0.0.1 - the master creates the store locally
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=127.0.0.1:29400 \
  train_torchrun.py
```

**On Node 02 (Worker) - In WSL:**
```bash
# Use Node 01's WINDOWS IP (not WSL IP)
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=192.168.29.67:29400 \
  train_torchrun.py
```

**Key Points:**
- **Node 01** uses `127.0.0.1` to bind locally, then port forwarding exposes it
- **Node 02** uses Node 01's **Windows IP** (`192.168.29.67`)
- Port forwarding bridges Windows → WSL automatically!

---

## 📊 Expected Multi-Node Output

**On Node 01 (Rank 0):**
```
Initializing with backend: nccl
[rank 0] world_size=2 device=cuda hostname=AIEDX-AsusTUF
[rank 0] gathered=[0, 1]
[rank 0] barrier OK; shutting down
```

**On Node 02 (Rank 1):**
```
Initializing with backend: nccl
[rank 1] world_size=2 device=cuda hostname=OTHER-HOSTNAME
[rank 1] gathered=[0, 1]
[rank 1] barrier OK; shutting down
```

✅ Both nodes should show `gathered=[0, 1]` indicating successful communication!

---

## 🔧 Troubleshooting

### Issue: "Connection refused" or timeout

**Check:**
1. Port forwarding is set up on Node 01 (Windows side)
2. Windows Firewall allows port 29400
3. Both nodes can ping each other's Windows IPs:
   ```bash
   ping 192.168.29.67  # From Node 02 to Node 01
   ```

### Issue: "NCCL not available"

**Solution:**
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Issue: "CUDA not available"

**Check:**
1. Run `nvidia-smi` in WSL - should show your GPU
2. Update Windows NVIDIA driver: https://developer.nvidia.com/cuda/wsl
3. Restart WSL: `wsl --shutdown` (from PowerShell), then reopen

---

## 🧹 Optional Cleanup Commands

**Deactivate virtual environment:**
```bash
deactivate
```

**Shut down WSL after training:**
```powershell
# From Windows PowerShell
wsl --shutdown
```

**Remove all port forwarding rules:**
```powershell
# From PowerShell as Admin
netsh interface portproxy reset
```

---

## ✅ Quick Reference Summary

| Step | Command |
|------|---------|
| Update packages | `sudo apt update && sudo apt install -y build-essential python3-venv python3-pip git` |
| Create venv | `python3 -m venv .venv && source .venv/bin/activate` |
| Install PyTorch | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124` |
| Verify CUDA | `python -c "import torch; print(torch.cuda.is_available())"` |
| Verify NCCL | `python -c "import torch; print(torch.distributed.is_nccl_available())"` |
| Single-node test | `torchrun --standalone --nproc_per_node=1 train_torchrun.py` |
| Get WSL IP | `hostname -I` |
| Get Windows IP | `ipconfig \| findstr IPv4` (in PowerShell) |
| Setup port forward | `netsh interface portproxy add v4tov4 listenport=29400 ...` (PowerShell Admin) |
| Multi-node master | `torchrun --nnodes=2 --node_rank=0 --rdzv_endpoint=127.0.0.1:29400 train_torchrun.py` |
| Multi-node worker | `torchrun --nnodes=2 --node_rank=1 --rdzv_endpoint=WINDOWS_IP:29400 train_torchrun.py` |

---

## 📝 Important Notes

1. **Port forwarding must be set up on BOTH nodes** if you want bidirectional master/worker switching
2. **Always use Windows IP** when connecting FROM another physical computer
3. **Always use WSL IP** when running locally on the same machine
4. **NCCL backend** is much faster than Gloo for GPU-to-GPU communication
5. **Firewall rules** must allow the port on Windows (not WSL)

---

**Last Updated:** 2025-10-10  
**Tested on:** WSL2 Ubuntu 22.04, PyTorch 2.6.0+cu124, NVIDIA RTX GPUs

