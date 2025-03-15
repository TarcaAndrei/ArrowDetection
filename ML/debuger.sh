#!/bin/bash


export PYTHONPATH=$PYTHONPATH:$(pwd)


python \
    -m debugpy \
    --listen 0.0.0.0:8765 \
    --wait-for-client \
    inference/inference_pipeline.py \

