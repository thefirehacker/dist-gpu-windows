#!/usr/bin/env python3
"""
Windows RTX 2050 Worker for Distributed PyTorch Training (Fixed libuv issues)
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
    """Initialize this Windows machine as a distributed worker with fixed libuv issues"""
    
    # Configuration
    WORLD_SIZE = 2  # Mac (rank 0) + Windows (rank 1)
    RANK = 1       # This Windows machine is rank 1
    
    print(f"\n=== Distributed Worker Setup ===")
    print(f"Master Address: {master_addr}")
    print(f"Master Port: {master_port}")
    print(f"World Size: {WORLD_SIZE}")
    print(f"This machine's rank: {RANK}")
    
    # Clear environment variables that might cause libuv issues
    env_vars_to_clear = [
        'GLOO_SOCKET_IFNAME',
        'GLOO_DEVICE_TRANSPORT',
        'NCCL_SOCKET_IFNAME',
        'NCCL_IB_DISABLE',
        'NCCL_P2P_DISABLE'
    ]
    
    for var in env_vars_to_clear:
        if var in os.environ:
            del os.environ[var]
    
    # Set required environment variables
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    os.environ['WORLD_SIZE'] = str(WORLD_SIZE)
    os.environ['RANK'] = str(RANK)
    
    # Force specific settings to avoid libuv
    os.environ['GLOO_SOCKET_IFNAME'] = ''
    os.environ['NCCL_SOCKET_IFNAME'] = ''
    os.environ['GLOO_DEVICE_TRANSPORT'] = 'TCP'
    os.environ['GLOO_SOCKET_FAMILY'] = 'AF_INET'
    os.environ['TP_SOCKET_IFNAME'] = ''
    
    try:
        # Try different initialization methods with anti-libuv settings
        
        # Method 1: Force TCP transport without libuv
        try:
            print("Trying TCP transport initialization (no libuv)...")
            
            # Additional environment settings to force TCP
            os.environ['GLOO_SOCKET_IFNAME'] = ''
            os.environ['NCCL_SOCKET_IFNAME'] = ''
            
            # Initialize with explicit TCP backend
            init_method = f'tcp://{master_addr}:{master_port}'
            dist.init_process_group(
                backend='gloo', 
                init_method=init_method,
                rank=RANK,
                world_size=WORLD_SIZE,
                timeout=torch.distributed.constants.default_pg_timeout
            )
            print(f"✅ TCP transport successful!")
            return True
        except Exception as e1:
            print(f"TCP transport failed: {e1}")
        
        # Method 2: Try with MPI backend if available
        try:
            if dist.is_mpi_available():
                print("Trying MPI backend initialization...")
                dist.init_process_group(backend='mpi', rank=RANK, world_size=WORLD_SIZE)
                print(f"✅ MPI backend successful!")
                return True
            else:
                print("MPI backend not available, skipping...")
        except Exception as e2:
            print(f"MPI backend failed: {e2}")
        
        # Method 3: Environment variable method as fallback
        try:
            print("Trying environment variable initialization...")
            dist.init_process_group(backend='gloo')
            print(f"✅ Environment method successful!")
            return True
        except Exception as e3:
            print(f"Environment method failed: {e3}")
        
        # If all methods fail
        print(f"❌ All initialization methods failed")
        print("This appears to be a PyTorch build issue with libuv support")
        return False
        
    except Exception as e:
        print(f"❌ Failed to initialize distributed worker: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure both machines are on the same network")
        print("2. Check that the Mac's IP address is correct")
        print("3. Verify port 12355 is not blocked by firewall")
        print("4. Make sure the Mac coordinator is running first")
        print("5. Try restarting both Python processes")
        return False

def test_simple_operations():
    """Test simple distributed operations"""
    
    if not dist.is_initialized():
        print("❌ Distributed not initialized")
        return False
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print(f"\n=== Simple Distributed Test ===")
    print(f"Rank {rank} (Windows RTX 2050): Running on device {device}")
    
    try:
        # Test 1: Simple tensor creation and barrier
        print("\n--- Test 1: Barrier Synchronization ---")
        print(f"Rank {rank}: Waiting at barrier...")
        dist.barrier()
        print(f"Rank {rank}: Barrier passed!")
        
        # Test 2: Simple all_gather
        print("\n--- Test 2: Simple All-Gather ---")
        tensor_list = [torch.zeros(2, dtype=torch.int64) for _ in range(world_size)]
        input_tensor = torch.tensor([rank * 10, rank * 10 + 1], dtype=torch.int64)
        
        # Move to device if CUDA available
        if device.type == 'cuda':
            tensor_list = [t.to(device) for t in tensor_list]
            input_tensor = input_tensor.to(device)
        
        print(f"Rank {rank}: Input tensor: {input_tensor}")
        
        dist.all_gather(tensor_list, input_tensor)
        
        # Synchronize GPU if needed
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        print(f"Rank {rank}: Gathered tensors: {tensor_list}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main function to run the Windows worker"""
    
    print("🚀 Windows RTX 2050 Distributed Worker (Fixed)")
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
    print("Please enter your Mac's IP address")
    
    while True:
        master_addr = input("\nEnter Mac's IP address: ").strip()
        if master_addr and master_addr != "127.0.0.1":
            break
        print("Please enter a valid IP address (not localhost)")
    
    # Initialize distributed worker
    if not initialize_distributed_worker(master_addr):
        print("\n💡 Try these solutions:")
        print("1. Restart the Mac coordinator")
        print("2. Check Mac firewall settings")
        print("3. Try a different port (change MASTER_PORT on both sides)")
        print("4. Ensure PyTorch versions match on both machines")
        print("5. Install PyTorch with different build (pip install torch --force-reinstall --no-cache-dir)")
        print("6. This might be a PyTorch libuv build issue - consider using a different PyTorch version")
        sys.exit(1)
    
    try:
        print(f"\n✅ Distributed initialization successful!")
        print(f"Rank: {dist.get_rank()}, World size: {dist.get_world_size()}")
        
        # Run simple tests
        if test_simple_operations():
            print(f"\n🎉 Windows worker is operational!")
            print(f"🎯 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
            print(f"🌐 Connected to Mac coordinator at {master_addr}")
            print(f"⏳ Ready for distributed operations...")
            
            # Keep worker alive
            print("\nPress Ctrl+C to stop the worker...")
            while True:
                time.sleep(1)
        else:
            print("❌ Tests failed")
            
    except KeyboardInterrupt:
        print(f"\n🛑 Worker shutting down...")
    except Exception as e:
        print(f"\n❌ Error during operations: {e}")
    finally:
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
                print("✅ Distributed process group destroyed")
        except:
            pass

if __name__ == "__main__":
    main()
