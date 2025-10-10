#!/bin/bash
# Node 01 - Master (Rank 0)
# Run this on Windows01 using Git Bash

# Set your IP address here
NODE0_IP="192.168.29.67"  # Change to your actual IP

echo "Starting Node 0 (Coordinator)"
echo "Local IP: $NODE0_IP"
echo ""
echo "Waiting for Node 1 to connect..."
echo ""

python train_multinode.py --rank 0 --world-size 2 --master-addr $NODE0_IP
