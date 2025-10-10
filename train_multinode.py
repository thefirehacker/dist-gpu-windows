"""
Multi-node distributed training script
Works around PyTorch libuv issues by using manual TCP initialization

Usage:
  Node 0 (coordinator): python train_multinode.py --rank 0 --world-size 2 --master-addr <THIS_NODE_IP>
  Node 1 (worker):      python train_multinode.py --rank 1 --world-size 2 --master-addr <NODE_0_IP>
"""
import argparse
import os
import socket
import torch
import torch.distributed as dist


def infer_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_local_ip():
    """Get this machine's local IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return socket.gethostbyname(socket.gethostname())


def main():
    parser = argparse.ArgumentParser(description='Multi-node distributed training')
    parser.add_argument('--rank', type=int, required=True, help='Rank of this node (0, 1, 2, ...)')
    parser.add_argument('--world-size', type=int, required=True, help='Total number of nodes')
    parser.add_argument('--master-addr', type=str, required=True, help='IP address of rank 0 node')
    parser.add_argument('--master-port', type=str, default='29500', help='Port for communication (default: 29500)')
    parser.add_argument('--backend', type=str, default='gloo', choices=['gloo', 'nccl'], help='Backend to use')
    
    args = parser.parse_args()
    
    # Set environment variables for distributed training
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    os.environ['WORLD_SIZE'] = str(args.world_size)
    os.environ['RANK'] = str(args.rank)
    
    local_ip = get_local_ip()
    
    print(f"\n{'='*60}")
    print(f"Multi-Node Distributed Training Setup")
    print(f"{'='*60}")
    print(f"Local IP:     {local_ip}")
    print(f"Master Addr:  {args.master_addr}")
    print(f"Master Port:  {args.master_port}")
    print(f"Rank:         {args.rank}")
    print(f"World Size:   {args.world_size}")
    print(f"Backend:      {args.backend}")
    print(f"{'='*60}\n")
    
    # Initialize process group using tcp:// init method
    # This avoids the broken TCPStore that torchrun uses
    init_method = f'tcp://{args.master_addr}:{args.master_port}'
    
    print(f"[INFO] Initializing process group...")
    print(f"   Init method: {init_method}")
    
    try:
        dist.init_process_group(
            backend=args.backend,
            init_method=init_method,
            world_size=args.world_size,
            rank=args.rank,
            timeout=torch.distributed.timedelta(seconds=30)
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize: {e}")
        print(f"\nTroubleshooting:")
        print(f"1. Ensure rank 0 node ({args.master_addr}) is started FIRST")
        print(f"2. Check firewall: port {args.master_port} must be open on rank 0")
        print(f"3. Verify network connectivity: ping {args.master_addr}")
        return

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = infer_device()

    print(f"[SUCCESS] Successfully initialized!")
    print(f"   Rank: {rank}/{world_size}")
    print(f"   Device: {device}")
    print(f"   Hostname: {socket.gethostname()}\n")

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
