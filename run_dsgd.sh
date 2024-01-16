#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
bflrun -np 4 python dsgd.py \
    --batch_size 128 \
    --epochs 5 \
    --lr 0.1 \