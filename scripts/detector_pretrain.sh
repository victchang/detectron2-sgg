#!/bin/bash
export master_port="tcp://127.0.0.1:3407"
export gpu_num=2
export CUDA_VISIBLE_DEVICES="2,3"
export output_dir="./checkpoints/detection/X-101-FPN-0628"

python tools/train_net.py --dist-url $master_port --num-gpus $gpu_num --config-file configs/detection/X-101-FPN.yaml --resume \
    OUTPUT_DIR $output_dir