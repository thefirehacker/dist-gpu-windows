#!/usr/bin/env python3
"""
Simple connection test to Mac coordinator
"""

import torch
import torch.distributed as dist
import os
import socket
import time

def test_network_connection(mac_ip, port=12355):
    """Test basic network connectivity"""
    print(f"Testing network connection to {mac_ip}:{port}")
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((mac_ip, port))
        sock.close()
        
        if result == 0:
            print(f"✅ Network connection to {mac_ip}:{port} successful")
            return True
        else:
            print(f"❌ Cannot connect to {mac_ip}:{port}")
            return False
    except Exception as e:
        print(f"❌ Network test failed: {e}")
        return False

def test_simple_distributed_init(mac_ip, port="12355"):
    """Test simple distributed initialization"""
    print(f"\nTesting distributed initialization to {mac_ip}:{port}")
    
    # Set environment variables
    os.environ['MASTER_ADDR'] = mac_ip
    os.environ['MASTER_PORT'] = port
    os.environ['WORLD_SIZE'] = '2'
    os.environ['RANK'] = '1'
    
    # Clear any problematic environment variables
    if 'GLOO_SOCKET_IFNAME' in os.environ:
        del os.environ['GLOO_SOCKET_IFNAME']
    
    try:
        # Try different initialization approaches
        
        # Approach 1: Basic init_process_group
        print("Trying basic initialization...")
        dist.init_process_group(
            backend='gloo',
            init_method=f'tcp://{mac_ip}:{port}',
            rank=1,
            world_size=2,
            timeout=torch.distributed.constants.default_pg_timeout
        )
        
        print(f"✅ Distributed initialization successful!")
        print(f"Rank: {dist.get_rank()}, World size: {dist.get_world_size()}")
        
        # Test a simple operation
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        tensor = torch.tensor([1, 2, 3], dtype=torch.int64).to(device)
        print(f"Test tensor created on {device}: {tensor}")
        
        return True
        
    except Exception as e:
        print(f"❌ Distributed initialization failed: {e}")
        return False
    finally:
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except:
            pass

def main():
    print("🔍 Connection Test to Mac Coordinator")
    print("=" * 50)
    
    # Get Mac IP from user
    while True:
        mac_ip = input("Enter Mac's IP address: ").strip()
        if mac_ip and mac_ip != "127.0.0.1":
            break
        print("Please enter a valid IP address (not localhost)")
    
    port = 12355
    
    # Test 1: Network connectivity
    if not test_network_connection(mac_ip, port):
        print("\n💡 Troubleshooting suggestions:")
        print("1. Ensure Mac coordinator is running and listening")
        print("2. Check Windows firewall settings")
        print("3. Check Mac firewall settings")
        print("4. Verify both machines are on same network")
        return
    
    # Test 2: Distributed initialization
    if test_simple_distributed_init(mac_ip, str(port)):
        print("\n🎉 Connection test successful!")
        print("You can now run: python worker.py")
    else:
        print("\n❌ Connection test failed")
        print("There might be a PyTorch distributed configuration issue")

if __name__ == "__main__":
    main()
