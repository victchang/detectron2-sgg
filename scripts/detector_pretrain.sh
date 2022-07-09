#!/bin/bash
export master_port="tcp://127.0.0.1:10001"
export gpu_num=4
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export output_dir="./checkpoints/detection/X-101-FPN"

python tools/train_net.py --dist-url $master_port --num-gpus $gpu_num --config-file configs/detection/X-101-FPN.yaml \
    OUTPUT_DIR $output_dir
