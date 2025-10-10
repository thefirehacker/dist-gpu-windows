import os
import socket
import tempfile
import torch
import torch.distributed as dist


def infer_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    # Use file:// init method which doesn't require TCPStore
    # This works better on Windows with some PyTorch versions
    temp_file = os.path.join(tempfile.gettempdir(), 'torch_dist_init')
    
    # Clean up any previous file
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    # Initialize process group with file:// method
    dist.init_process_group(
        backend="gloo",
        init_method=f'file:///{temp_file}',
        world_size=1,
        rank=0
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = infer_device()

    print(
        f"[SUCCESS] [rank {rank}] world_size={world_size} device={device} "
        f"hostname={socket.gethostname()}"
    )

    # Simple cross-rank check: gather all ranks
    gathered_ranks = [torch.zeros(1, dtype=torch.int64) for _ in range(world_size)]
    my_rank_tensor = torch.tensor([rank], dtype=torch.int64)
    dist.all_gather(gathered_ranks, my_rank_tensor)
    print(f"[SUCCESS] [rank {rank}] gathered={ [int(t.item()) for t in gathered_ranks] }")

    dist.barrier()
    print(f"[SUCCESS] [rank {rank}] barrier OK; shutting down")
    dist.destroy_process_group()
    
    # Clean up
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    print("\n[SUCCESS] Distributed training works on your system!")


if __name__ == "__main__":
    main()
