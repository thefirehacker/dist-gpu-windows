"""
Fixed multi-node distributed training script using torchrun
This version works around Windows networking issues
"""
import os
import torch
import torch.distributed as dist
from datetime import timedelta


def infer_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    # Get rank and world size from environment variables set by torchrun
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    print(f"\n{'='*60}")
    print(f"Multi-Node Distributed Training Setup")
    print(f"{'='*60}")
    print(f"Rank:         {rank}")
    print(f"World Size:   {world_size}")
    print(f"Local Rank:   {local_rank}")
    print(f"{'='*60}\n")
    
    # Initialize process group
    print(f"[INFO] Initializing process group...")
    
    try:
        dist.init_process_group(backend='gloo')
    except Exception as e:
        print(f"[ERROR] Failed to initialize: {e}")
        return

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = infer_device()

    print(f"[SUCCESS] Successfully initialized!")
    print(f"   Rank: {rank}/{world_size}")
    print(f"   Device: {device}")
    print(f"   Hostname: {os.environ.get('HOSTNAME', 'unknown')}\n")

    # Test 1: All-gather ranks
    print(f"[TEST] Test 1: All-gather operation...")
    gathered_ranks = [torch.zeros(1, dtype=torch.int64) for _ in range(world_size)]
    my_rank_tensor = torch.tensor([rank], dtype=torch.int64)
    dist.all_gather(gathered_ranks, my_rank_tensor)
    gathered_values = [int(t.item()) for t in gathered_ranks]
    print(f"   [Rank {rank}] Gathered ranks: {gathered_values}")
    
    # Test 2: Broadcast from rank 0
    print(f"\n[TEST] Test 2: Broadcast from rank 0...")
    if rank == 0:
        broadcast_tensor = torch.tensor([42.0, 100.0, 256.0])
        print(f"   [Rank 0] Broadcasting: {broadcast_tensor.tolist()}")
    else:
        broadcast_tensor = torch.zeros(3)
    
    dist.broadcast(broadcast_tensor, src=0)
    print(f"   [Rank {rank}] Received: {broadcast_tensor.tolist()}")
    
    # Test 3: Barrier synchronization
    print(f"\n[TEST] Test 3: Barrier synchronization...")
    print(f"   [Rank {rank}] Waiting at barrier...")
    dist.barrier()
    print(f"   [Rank {rank}] [SUCCESS] Barrier passed!")
    
    # Test 4: All-reduce sum
    print(f"\n[TEST] Test 4: All-reduce (sum)...")
    reduce_tensor = torch.tensor([rank + 1.0])
    print(f"   [Rank {rank}] Before reduce: {reduce_tensor.item()}")
    dist.all_reduce(reduce_tensor, op=dist.ReduceOp.SUM)
    print(f"   [Rank {rank}] After reduce (sum): {reduce_tensor.item()}")
    
    print(f"\n{'='*60}")
    print(f"[SUCCESS] [Rank {rank}] All tests passed! Shutting down...")
    print(f"{'='*60}\n")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
