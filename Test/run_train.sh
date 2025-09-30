#!/bin/bash
# Run torchrun with USE_LIBUV=0 to fix the libuv error

export USE_LIBUV=0
torchrun --standalone --nproc_per_node=1 train_torchrun.py
