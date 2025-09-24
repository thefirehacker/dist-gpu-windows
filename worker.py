#!/usr/bin/env python3
"""
Windows Worker for Distributed PyTorch - IP ONLY (No Hostname Resolution)
This script forces IP-only connections to avoid hostname resolution issues.
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
        print("⚠️  CUDA not available! Running on CPU")
    
    return True

def initialize_distributed_worker_ip_only(master_ip):
    """Initialize Windows worker with FORCED IP-only connection"""
    
    print(f"\n=== IP-ONLY Distributed Worker Setup ===")
    print(f"🌐 Master IP (Mac): {master_ip}")
    print(f"🚫 NO hostname resolution - IP ONLY!")
    
    # COMPLETELY clear hostname-related environment variables
    hostname_vars = ['HOSTNAME', 'HOST', 'COMPUTERNAME', 'USERDOMAIN', 'USERDNSDOMAIN']
    for var in hostname_vars:
        if var in os.environ:
            del os.environ[var]
            print(f"🗑️  Cleared {var}")
    
    # Set FORCED IP-only environment
    os.environ['MASTER_ADDR'] = master_ip  # MUST be IP, not hostname
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = '2'
    os.environ['RANK'] = '1'  # Windows is rank 1
    
    # Anti-hostname resolution settings
    os.environ['GLOO_SOCKET_IFNAME'] = ''
    os.environ['NCCL_SOCKET_IFNAME'] = ''
    os.environ['GLOO_DEVICE_TRANSPORT'] = 'TCP'
    os.environ['GLOO_SOCKET_FAMILY'] = 'AF_INET'
    os.environ['TP_SOCKET_IFNAME'] = ''
    
    print(f"✅ Environment configured:")
    print(f"   MASTER_ADDR = {os.environ['MASTER_ADDR']}")
    print(f"   MASTER_PORT = {os.environ['MASTER_PORT']}")
    print(f"   WORLD_SIZE = {os.environ['WORLD_SIZE']}")
    print(f"   RANK = {os.environ['RANK']}")
    
    # First wait for Mac coordinator to start listening
    print(f"\n🔄 Waiting for Mac coordinator to start listening on port 12355...")
    
    # Test connection first
    max_retries = 30  # 30 seconds
    for attempt in range(max_retries):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex((master_ip, 12355))
                if result == 0:
                    print(f"✅ Port 12355 is reachable on {master_ip}")
                    break
        except Exception:
            pass
        
        print(f"⏳ Attempt {attempt + 1}/{max_retries}: Port not ready, waiting...")
        time.sleep(1)
    else:
        print(f"❌ Port 12355 never became available on {master_ip}")
        print(f"💡 Make sure to run the Mac coordinator first!")
        return False
    
    try:
        # Method 1: Explicit TCP init (most reliable for IP-only)
        print(f"\n🔄 Attempting connection to {master_ip}:12355...")
        init_method = f'tcp://{master_ip}:12355'
        print(f"🌐 TCP Init Method: {init_method}")
        
        dist.init_process_group(
            backend='gloo',
            init_method=init_method,
            rank=1,
            world_size=2,
            timeout=torch.distributed.default_pg_timeout
        )
        
        print(f"✅ Connection successful!")
        print(f"Rank: {dist.get_rank()}, World size: {dist.get_world_size()}")
        return True
        
    except Exception as e:
        print(f"❌ TCP connection failed: {e}")
        
        # Method 2: Environment variable fallback
        try:
            print(f"🔄 Trying environment variable method...")
            dist.init_process_group(backend='gloo')
            print(f"✅ Environment method successful!")
            return True
        except Exception as e2:
            print(f"❌ Environment method also failed: {e2}")
            return False

def test_connection():
    """Test the distributed connection"""
    if not dist.is_initialized():
        print("❌ Distributed not initialized")
        return False
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print(f"\n=== Connection Test ===")
    print(f"✅ Rank {rank} (Windows): Connected!")
    print(f"🖥️  Device: {device}")
    print(f"🌐 World size: {world_size}")
    
    try:
        # Test barrier synchronization
        print(f"🔄 Testing barrier synchronization...")
        dist.barrier()
        print(f"✅ Barrier test successful!")
        
        # Test simple tensor operations
        print(f"🔄 Testing tensor operations...")
        input_tensor = torch.tensor([rank * 100, rank * 100 + 1], dtype=torch.int64)
        if device.type == 'cuda':
            input_tensor = input_tensor.to(device)
        
        tensor_list = [torch.zeros(2, dtype=torch.int64) for _ in range(world_size)]
        if device.type == 'cuda':
            tensor_list = [t.to(device) for t in tensor_list]
        
        print(f"📤 Sending: {input_tensor}")
        dist.all_gather(tensor_list, input_tensor)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        print(f"📥 Received: {tensor_list}")
        print(f"✅ Tensor operations successful!")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

def main():
    """Main function"""
    
    print("🚀 Windows Distributed Worker - IP ONLY VERSION")
    print("=" * 60)
    print("🎯 This version FORCES IP-only connections (no hostname resolution)")
    print("=" * 60)
    
    # Check setup
    if not check_cuda_setup():
        print("⚠️  Continuing without CUDA...")
    
    # Show local IP for reference
    local_ip = get_local_ip()
    print(f"\n📍 This Windows machine IP: {local_ip}")
    
    # Get Mac IP
    print(f"\n{'='*50}")
    print("REQUIRED: Mac Coordinator IP Address")
    print("='*50")
    print("Enter the IP address of your Mac (NOT hostname!)")
    print("Example: 192.168.1.100")
    
    while True:
        mac_ip = input("\n🖥️  Enter Mac IP address: ").strip()
        
        # Validate it's an IP address, not hostname
        if not mac_ip:
            print("❌ Please enter an IP address")
            continue
        
        if mac_ip == "127.0.0.1" or mac_ip == "localhost":
            print("❌ Cannot use localhost - enter the actual network IP")
            continue
        
        # Basic IP validation
        parts = mac_ip.split('.')
        if len(parts) == 4:
            try:
                for part in parts:
                    if not (0 <= int(part) <= 255):
                        raise ValueError
                break
            except ValueError:
                print("❌ Invalid IP format")
                continue
        else:
            print("❌ Invalid IP format (must be x.x.x.x)")
            continue
    
    # Initialize worker
    print(f"\n🔄 Connecting to Mac at {mac_ip}...")
    if not initialize_distributed_worker_ip_only(mac_ip):
        print(f"\n❌ Connection failed!")
        print(f"💡 Troubleshooting:")
        print(f"1. Ensure Mac coordinator is running first")
        print(f"2. Verify IP address {mac_ip} is correct")
        print(f"3. Check firewall settings on both machines")
        print(f"4. Ensure both machines are on same network")
        sys.exit(1)
    
    # Test connection
    if not test_connection():
        print(f"❌ Connection tests failed")
        sys.exit(1)
    
    # Keep worker alive
    try:
        print(f"\n🎉 Windows worker successfully connected!")
        print(f"🎯 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        print(f"🌐 Connected to Mac at {mac_ip}")
        print(f"🔄 Ready for distributed operations...")
        print(f"\n⌨️  Press Ctrl+C to stop the worker")
        
        # Keep alive and wait for operations
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print(f"\n🛑 Shutting down worker...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
                print("✅ Cleanup completed")
        except:
            pass

if __name__ == "__main__":
    main()
