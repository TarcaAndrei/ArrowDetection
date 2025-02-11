#!/bin/bash

cd /home/user/repos/ArrowDetection/

PYTHONPATH=$(pwd) torchrun \
    --standalone \
    --nproc-per-node=4 \
    pipelines/engine_vit.py
