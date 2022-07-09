#!/bin/bash
export master_port="tcp://127.0.0.1:10001"
export gpu_num=1
export CUDA_VISIBLE_DEVICES="0,1"

export archive="/home/dcs405a/victchang/sgg/checkpoints/sgdet/X-101-FPN-MSDN-0708"
export model="model_0019999.pth"

python tools/train_net.py --dist-url $master_port --num-gpus $gpu_num --eval-only --config-file $archive/config.yaml \
    MODEL.WEIGHTS $archive/$model \
    TEST.DETECTIONS_PER_IMAGE 60 \
    OUTPUT_DIR $archive/testing