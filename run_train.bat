@echo off
REM Run single-node distributed training on Windows

REM Pass use_libuv=0 directly to torchrun via rdzv-conf
torchrun --standalone --nproc_per_node=1 --rdzv-conf use_libuv=0 train_torchrun.py
