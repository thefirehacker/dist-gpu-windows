#!/bin/bash
# Launch script for Node 01 (Master) using direct ethernet connection
# Prerequisites:
# - Ethernet cable connecting both nodes
# - Static IPs configured: Node 01 = 192.168.100.1, Node 02 = 192.168.100.2
# - WSL mirrored networking enabled
# - Firewall configured to allow ports 29500 and 20000-40000

echo "=========================================="
echo "Node 01 (Master) - Direct Ethernet Launch"
echo "=========================================="
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "ERROR: Virtual environment not activated!"
    echo "Please run: source .venv/bin/activate"
    exit 1
fi

# Check connectivity to Node 02
echo "Testing connectivity to Node 02 (192.168.100.2)..."
if ping -c 1 -W 2 192.168.100.2 > /dev/null 2>&1; then
    echo "✓ Node 02 is reachable"
else
    echo "✗ Cannot reach Node 02 at 192.168.100.2"
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
echo "  Node Rank: 0 (Master)"
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

echo "Starting distributed training on Node 01 (Master)..."
echo "Waiting for Node 02 to connect..."
echo ""

# Run torchrun with ethernet configuration
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 \
  --master_addr=192.168.100.1 \
  --master_port=29500 \
  train_torchrun.py

echo ""
echo "Training completed on Node 01 (Master)"
