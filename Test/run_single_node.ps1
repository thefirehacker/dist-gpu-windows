# Single-node training script for Windows (PowerShell)

# Activate virtual environment
& .\.venv\Scripts\Activate.ps1

# Run torchrun with standalone mode (no rendezvous needed for single node)
& .\.venv\Scripts\torchrun.exe `
    --standalone `
    --nproc_per_node=1 `
    train_torchrun.py
