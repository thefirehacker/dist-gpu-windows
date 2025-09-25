#!/usr/bin/env python3
"""
Windows Worker for Distributed PyTorch — TCPStore Rendezvous (Mac as Pure Controller)
- Connects to a TCPStore hosted on the Mac (controller only)
- Joins the PyTorch process group with explicit rank/world_size
- All collectives occur strictly between Windows workers
"""

import torch
import torch.distributed as dist
import os
import sys
import time
import socket

os.environ['GLOO_SOCKET_FAMILY'] = 'AF_INET'
os.environ['GLOO_USE_IPV6'] = '0'
os.environ['GLOO_DEVICE_TRANSPORT'] = 'TCP'

def get_local_ip() -> str:
    """Get the local IP address of this Windows machine."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def check_cuda_setup() -> bool:
    """Print CUDA info; continue even if CUDA is unavailable."""
    print("=== CUDA Setup Check ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )
    else:
        print("⚠️  CUDA not available! Training will run on CPU if used.")
    return True


def initialize_distributed_via_store(mac_ip: str, rank: int, world_size: int, port: int = 12355) -> bool:
    """Join the process group using a TCPStore running on the Mac controller.

    The Mac hosts TCPStore(is_master=True). Windows workers connect with is_master=False.
    The Mac is NOT a group member; only Windows workers call init_process_group.
    """
    print("\n=== Windows Worker — Join via TCPStore ===")
    print(f"Controller (Mac) IP: {mac_ip}")
    print(f"Rank: {rank} / World Size: {world_size}")

    try:
        store = dist.TCPStore(mac_ip, port, world_size, is_master=False, use_libuv=False)
        dist.init_process_group(
            backend="gloo",
            store=store,
            rank=rank,
            world_size=world_size,
            timeout=torch.distributed.default_pg_timeout,
        )
        print("✅ Joined process group successfully")
        print(f"Rank confirmed: {dist.get_rank()}  World size: {dist.get_world_size()}")
        return True
    except Exception as e:
        print(f"❌ Failed to join process group: {e}")
        print("\nTroubleshooting:")
        print("1) Ensure the Mac TCPStore is running (see notebook controller cell)")
        print("2) Verify IP and port are correct and reachable")
        print("3) Confirm world_size matches number of worker processes")
        print("4) Check firewall on both machines for TCP port 12355")
        return False


def test_connection_collectives() -> bool:
    """Run a minimal all_gather across Windows workers only."""
    if not dist.is_initialized():
        print("❌ Distributed not initialized")
        return False

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("\n=== Collective Test (Windows ranks only) ===")
    print(f"Rank {rank}: device = {device}")

    tensor_list = [torch.zeros(2, dtype=torch.int64, device=device) for _ in range(world_size)]
    input_tensor = torch.tensor([rank * 100, rank * 100 + 1], dtype=torch.int64, device=device)

    dist.all_gather(tensor_list, input_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()

    print(f"Rank {rank}: gathered = {tensor_list}")
    return True


def prompt_value(prompt: str, validator) -> str:
    while True:
        val = input(prompt).strip()
        try:
            if validator(val):
                return val
        except Exception:
            pass
        print("❌ Invalid input. Please try again.")


def main() -> None:
    print("🚀 Windows Distributed Worker — TCPStore Rendezvous")
    print("=" * 60)
    check_cuda_setup()

    local_ip = get_local_ip()
    print(f"\n📍 This Windows machine IP: {local_ip}")

    # Read from env first to enable non-interactive starts
    mac_ip = os.environ.get("MAC_IP") or prompt_value(
        "\n🖥️  Enter Mac (controller) IP address: ",
        lambda s: s and s != "127.0.0.1" and all(p.isdigit() and 0 <= int(p) <= 255 for p in s.split(".") if len(s.split(".")) == 4),
    )

    world_size_str = os.environ.get("WORLD_SIZE") or prompt_value(
        "Workers world_size (e.g., 2): ", lambda s: s.isdigit() and int(s) >= 1
    )
    rank_str = os.environ.get("RANK") or prompt_value(
        "This worker's rank [0..world_size-1]: ",
        lambda s: s.isdigit(),
    )

    WORLD_SIZE = int(world_size_str)
    RANK = int(rank_str)

    if RANK < 0 or RANK >= WORLD_SIZE:
        print("❌ Rank must be in [0..world_size-1]")
        sys.exit(1)

    if not initialize_distributed_via_store(mac_ip=mac_ip, rank=RANK, world_size=WORLD_SIZE):
        sys.exit(1)

    if not test_connection_collectives():
        print("❌ Collective test failed")
        sys.exit(1)

    try:
        print("\n🎉 Worker ready — waiting for training workload … (Ctrl+C to exit)")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down worker …")
    finally:
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
                print("✅ Process group destroyed")
        except Exception:
            pass


if __name__ == "__main__":
    main()
