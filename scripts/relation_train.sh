#!/bin/bash
export master_port="tcp://127.0.0.1:10001"
export gpu_num=4
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export output_dir="./checkpoints/sgdet/X-101-FPN-MSDN"

python tools/train_net.py --dist-url $master_port --num-gpus $gpu_num --config-file configs/relation/X-101-FPN-MSDN.yaml \
    OUTPUT_DIR $output_dir
