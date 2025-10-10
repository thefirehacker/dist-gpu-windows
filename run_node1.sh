#!/bin/bash
# Node 02 - Worker (Rank 1)
# Run this on Windows02 using Git Bash

# Set Node 01's IP address here
NODE0_IP="192.168.29.67"  # Change to Node 01's IP

echo "Starting Node 1 (Worker)"
echo "Connecting to Node 0 at: $NODE0_IP"
echo ""

python train_multinode.py --rank 1 --world-size 2 --master-addr $NODE0_IP
