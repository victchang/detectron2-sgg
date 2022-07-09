#!/bin/bash
export master_port="tcp://127.0.0.1:10028"
export gpu_num=1
export CUDA_VISIBLE_DEVICES="2,3"
export output_dir="./checkpoints/sgdet/X-101-FPN-MSDN-0709"

python tools/train_net.py --dist-url $master_port --num-gpus $gpu_num --config-file configs/relation/X-101-FPN-MSDN.yaml \
    MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING True \
    OUTPUT_DIR $output_dir