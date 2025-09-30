"""
Diagnose PyTorch distributed networking issues on Windows
"""
import socket
import torch

print("=== PyTorch Distributed Diagnostics ===\n")

# 1. PyTorch version info
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
print()

# 2. Check network connectivity
print("=== Network Connectivity ===")
hostname = socket.gethostname()
try:
    local_ip = socket.gethostbyname(hostname)
    print(f"Hostname: {hostname}")
    print(f"Local IP: {local_ip}")
except Exception as e:
    print(f"❌ Error getting local IP: {e}")
print()

# 3. Test localhost connection
print("=== Testing Localhost ===")
try:
    test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    test_socket.settimeout(2)
    result = test_socket.connect_ex(('127.0.0.1', 29500))
    if result == 0:
        print("❌ Port 29500 is already in use")
    else:
        print("✅ Port 29500 is available")
    test_socket.close()
except Exception as e:
    print(f"⚠️  Socket test error: {e}")
print()

# 4. Test TCPStore creation
print("=== Testing TCPStore ===")
try:
    from torch.distributed import TCPStore
    # Try to create a TCPStore
    print("Attempting to create TCPStore on localhost:29501...")
    store = TCPStore(
        host_name="127.0.0.1",
        port=29501,
        world_size=1,
        is_master=True,
        timeout=5,
        use_libuv=False  # Try without libuv
    )
    print("✅ TCPStore created successfully with use_libuv=False")
    store.set("test_key", "test_value")
    value = store.get("test_key")
    print(f"✅ TCPStore set/get works: {value}")
    del store
except Exception as e:
    print(f"❌ TCPStore failed: {type(e).__name__}: {e}")
    print("\nTrying with use_libuv=True...")
    try:
        store = TCPStore(
            host_name="127.0.0.1",
            port=29502,
            world_size=1,
            is_master=True,
            timeout=5,
            use_libuv=True  # Try with libuv
        )
        print("✅ TCPStore created successfully with use_libuv=True")
        del store
    except Exception as e2:
        print(f"❌ TCPStore also failed with libuv: {type(e2).__name__}: {e2}")

print()
print("=== Diagnosis Complete ===")
print("\nRecommendations:")
print("1. If TCPStore fails, check Windows Firewall settings")
print("2. Try running Command Prompt as Administrator")
print("3. Consider updating PyTorch: pip install --upgrade torch")
print("4. For multi-node: use file:// init_method instead of env://")
