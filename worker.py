#!/usr/bin/env python3
"""
Windows RTX 2050 Worker for Distributed PyTorch Training
This script runs on the Windows machine with RTX 2050 GPU
"""

import torch
import torch.distributed as dist
import os
import sys
import time
import socket

def get_local_ip():
    """Get the local IP address of this Windows machine"""
    try:
        # Connect to a remote address to determine local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
        return local_ip
    except Exception:
        return "127.0.0.1"

def check_cuda_setup():
    """Check CUDA installation and GPU availability"""
    print("=== CUDA Setup Check ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("❌ CUDA not available! Please install PyTorch with CUDA support:")
        print("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        return False
    
    return True

def initialize_distributed_worker(master_addr, master_port="12355"):
    """Initialize this Windows machine as a distributed worker"""
    
    # Configuration
    WORLD_SIZE = 2  # Mac (rank 0) + Windows (rank 1)
    RANK = 1       # This Windows machine is rank 1
    
    # Set environment variables
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    os.environ['WORLD_SIZE'] = str(WORLD_SIZE)
    os.environ['RANK'] = str(RANK)
    
    print(f"\n=== Distributed Worker Setup ===")
    print(f"Master Address: {master_addr}")
    print(f"Master Port: {master_port}")
    print(f"World Size: {WORLD_SIZE}")
    print(f"This machine's rank: {RANK}")
    
    try:
        # Initialize process group with 'gloo' backend for CPU-GPU mixed setup
        dist.init_process_group(backend='gloo', timeout=torch.distributed.constants.default_pg_timeout)
        
        print(f"✅ Distributed worker initialized successfully!")
        print(f"Rank: {dist.get_rank()}, World size: {dist.get_world_size()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize distributed worker: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure both machines are on the same network")
        print("2. Check that the Mac's IP address is correct")
        print("3. Verify port 12355 is not blocked by firewall")
        print("4. Make sure the Mac coordinator is running first")
        return False

def test_distributed_operations():
    """Test distributed operations on this Windows worker"""
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda:0")  # Use RTX 2050
    
    print(f"\n=== Testing Distributed Operations ===")
    print(f"Rank {rank} (Windows RTX 2050): Running on device {device}")
    
    # Test 1: Basic all_gather
    print("\n--- Test 1: Basic All-Gather ---")
    tensor_list = [torch.zeros(3, dtype=torch.int64).to(device) for _ in range(world_size)]
    input_tensor = torch.tensor([rank*100, rank*100+10, rank*100+20], dtype=torch.int64).to(device)
    
    print(f"Rank {rank}: Input tensor: {input_tensor}")
    
    dist.all_gather(tensor_list, input_tensor)
    torch.cuda.synchronize()  # Ensure GPU operations complete
    
    print(f"Rank {rank}: Gathered tensors: {tensor_list}")
    
    # Test 2: Advanced operations with timing
    print("\n--- Test 2: Advanced Operations ---")
    tensor_size = (2, 3)
    tensor_list_adv = [torch.zeros(tensor_size, dtype=torch.int64).to(device) for _ in range(world_size)]
    input_tensor_adv = torch.tensor([[rank*10, rank*10+1, rank*10+2], 
                                   [rank*10+3, rank*10+4, rank*10+5]], 
                                  dtype=torch.int64).to(device)
    
    start_time = time.time()
    dist.all_gather(tensor_list_adv, input_tensor_adv)
    torch.cuda.synchronize()
    end_time = time.time()
    
    print(f"Rank {rank}: Advanced all-gather completed in {end_time - start_time:.4f} seconds")
    print(f"Rank {rank}: Input shape: {input_tensor_adv.shape}")
    print(f"Rank {rank}: Input tensor:\n{input_tensor_adv}")
    
    # Test 3: Barrier synchronization
    print("\n--- Test 3: Barrier Synchronization ---")
    print(f"Rank {rank}: Waiting at barrier...")
    dist.barrier()
    print(f"Rank {rank}: Barrier passed! All processes synchronized.")
    
    return True

def main():
    """Main function to run the Windows worker"""
    
    print("🚀 Windows RTX 2050 Distributed Worker")
    print("=" * 50)
    
    # Check CUDA setup
    if not check_cuda_setup():
        sys.exit(1)
    
    # Get local IP for reference
    local_ip = get_local_ip()
    print(f"\nThis Windows machine IP: {local_ip}")
    
    # Get Mac's IP address
    print("\n" + "=" * 50)
    print("CONFIGURATION REQUIRED:")
    print("=" * 50)
    print("Please enter your Mac's IP address (from the notebook output)")
    print("Example: 192.168.1.100")
    
    while True:
        master_addr = input("\nEnter Mac's IP address: ").strip()
        if master_addr and master_addr != "127.0.0.1":
            break
        print("Please enter a valid IP address (not localhost)")
    
    # Initialize distributed worker
    if not initialize_distributed_worker(master_addr):
        sys.exit(1)
    
    try:
        # Test distributed operations
        test_distributed_operations()
        
        print(f"\n✅ Windows worker is ready and operational!")
        print(f"🎯 GPU: {torch.cuda.get_device_name(0)}")
        print(f"🌐 Connected to Mac coordinator at {master_addr}")
        print(f"⏳ Waiting for distributed operations...")
        
        # Keep worker alive for operations
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print(f"\n🛑 Worker shutting down...")
    except Exception as e:
        print(f"\n❌ Error during operations: {e}")
    finally:
        try:
            dist.destroy_process_group()
            print("✅ Distributed process group destroyed")
        except:
            pass

if __name__ == "__main__":
    main()
