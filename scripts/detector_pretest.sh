#!/bin/bash
export master_port="tcp://127.0.0.1:10001"
export gpu_num=4
export CUDA_VISIBLE_DEVICES="0,1,2,3"

export archive="/home/dcs405a/victchang/sgg/checkpoints/detection/X-101-FPN/"
export model="model_final.pth"

python tools/train_net.py --dist-url $master_port --num-gpus $gpu_num --eval-only --config-file $archive/config.yaml \
    MODEL.WEIGHTS $archive/$model \
    OUTPUT_DIR $archive/testing
