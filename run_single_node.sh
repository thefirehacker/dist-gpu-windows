#!/bin/bash
# Single-node training script for Windows (Git Bash)

# Activate virtual environment
source .venv/Scripts/activate

# Run torchrun with standalone mode (no rendezvous needed for single node)
torchrun \
    --standalone \
    --nproc_per_node=1 \
    train_torchrun.py
