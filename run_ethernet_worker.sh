#!/bin/bash
# Launch script for Node 02 (Worker) using direct ethernet connection
# Prerequisites:
# - Ethernet cable connecting both nodes
# - Static IPs configured: Node 01 = 192.168.100.1, Node 02 = 192.168.100.2
# - WSL mirrored networking enabled
# - Firewall configured to allow ports 29500 and 20000-40000
# - Node 01 (Master) must be started FIRST

echo "=========================================="
echo "Node 02 (Worker) - Direct Ethernet Launch"
echo "=========================================="
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "ERROR: Virtual environment not activated!"
    echo "Please run: source .venv/bin/activate"
    exit 1
fi

# Check connectivity to Node 01 (Master)
echo "Testing connectivity to Node 01 Master (192.168.100.1)..."
if ping -c 1 -W 2 192.168.100.1 > /dev/null 2>&1; then
    echo "✓ Node 01 (Master) is reachable"
else
    echo "✗ Cannot reach Node 01 Master at 192.168.100.1"
    echo "  Please verify:"
    echo "  1. Ethernet cable is connected"
    echo "  2. Both nodes have static IPs configured"
    echo "  3. Firewall allows ICMP (ping)"
    exit 1
fi

echo ""
echo "Network Configuration:"
echo "  Master Address: 192.168.100.1"
echo "  Master Port: 29500"
echo "  World Size: 2 nodes"
echo "  Node Rank: 1 (Worker)"
echo ""

# Configure NCCL for ethernet
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

echo "NCCL Configuration:"
echo "  NCCL_SOCKET_IFNAME=eth0"
echo "  NCCL_IB_DISABLE=1"
echo "  NCCL_P2P_DISABLE=1"
echo "  NCCL_DEBUG=INFO"
echo ""

echo "IMPORTANT: Make sure Node 01 (Master) is already running!"
echo "Press Ctrl+C to cancel, or wait 3 seconds to continue..."
sleep 3

echo ""
echo "Connecting to Node 01 (Master)..."
echo ""

# Run torchrun with ethernet configuration
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=1 \
  --master_addr=192.168.100.1 \
  --master_port=29500 \
  train_torchrun.py

echo ""
echo "Training completed on Node 02 (Worker)"
