import socket
import torch
import torch.distributed as dist


def infer_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    # torchrun provides env:// rendezvous; do not pass store/init_method here
    dist.init_process_group(backend="gloo")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = infer_device()

    print(
        f"[rank {rank}] world_size={world_size} device={device} "
        f"hostname={socket.gethostname()}"
    )

    # Simple cross-rank check: gather all ranks
    gathered_ranks = [torch.zeros(1, dtype=torch.int64) for _ in range(world_size)]
    my_rank_tensor = torch.tensor([rank], dtype=torch.int64)
    dist.all_gather(gathered_ranks, my_rank_tensor)
    print(f"[rank {rank}] gathered={ [int(t.item()) for t in gathered_ranks] }")

    dist.barrier()
    print(f"[rank {rank}] barrier OK; shutting down")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()


