# detectron2-sgg (Will be ready soon)
A scene graph generation codebase attached to detectron2

## TODO
- [ ] initial release
- [ ] add predcls and sgcls
- [ ] add some common benchmarks
- [ ] add some data resampling methods

## Overview
This work provides a clean and general codebase for scene graph generation that is attached to [detectron2](https://github.com/facebookresearch/detectron2). While there already exists [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch), it is built on top of deprecated [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). Being the successor of maskrcnn-benchmark, detectron2 has higher throughputs in both training and inference. As a result, this work is built in order to take such advantage, and deliver an easy-to-use sgg framework.

This work is modified from [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) and [fcsgg](https://github.com/liuhengyue/fcsgg).


## Installation
### Requirements
According to detectron2 v0.5, the requirements are
- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.7 and torchvision that matches the PyTorch installation
- OpenCV is optional but needed by demo and visualization

### Installing steps
```
# install pre-built detectron2 at https://detectron2.readthedocs.io/en/v0.5/tutorials/install.html
git clone https://github.com/victchang/detectron2-sgg.git
cd detectron2-sgg
pip install -r requirements.txt
```

## Dataset Preparation (by [fcsgg](https://github.com/liuhengyue/fcsgg/blob/master/README.md#dataset-preparation))
1. Download the VG images [part1 (9 Gb)](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) [part2 (5 Gb)](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip). Extract these images to the file `datasets/vg/VG_100K`. 

```
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip -P datasets/vg/
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip -P datasets/vg/
unzip -j datasets/vg/images.zip -d datasets/vg/VG_100K
unzip -j datasets/vg/images2.zip -d datasets/vg/VG_100K
```
Optionally, remove the .zip files.
```
rm datasets/vg/images.zip
rm datasets/vg/images2.zip
```   
  
2. Download the [scene graphs](https://onedrive.live.com/embed?cid=22376FFAD72C4B64&resid=22376FFAD72C4B64%21779871&authkey=AA33n7BRpB1xa3I) and extract them to `datasets/vg/VG-SGG-with-attri.h5`.

If use other paths, one may need to modify the related paths in file [visual_genome.py](fcsgg/data/datasets/visual_genome.py).

The correct structure of files should be

```
fcsgg/
  |-- datasets/
     |-- vg/
        |-- VG-SGG-with-attri.h5         # `roidb_file`, HDF5 containing the GT boxes, classes, and relationships
        |-- VG-SGG-dicts-with-attri.json # `dict_file`, JSON Contains mapping of classes/relationships to words
        |-- image_data.json              # `image_file`, HDF5 containing image filenames
        |-- VG_100K                      # `img_dir`, contains all the images
           |-- 1.jpg
           |-- 2.jpg
           |-- ...

```

## Faster-RCNN Pre-Training
You can download the pre-trained [Faster-RCNN (X-101-FPN)](). Place the checkpoint wherever you like, just to remember to modify the path of `MODEL.WEIGHTS`
The pretrained detector achieves 14.92mAP/28.27mAP50 on VG150 testing set.
###### NOTE: the pre-trained Faster-RCNN is not the same as [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch), since detectron2 and maskrcnn-benchmark are implemented [differently](https://detectron2.readthedocs.io/en/v0.5/notes/compatibility.html).

You can also modify the config file to build and train the desired detector:
```
./scripts/detector_pretrain.sh  # for training
./scripts/detector_pretest.sh   # for testing
```
See [here](https://github.com/facebookresearch/detectron2/blob/main/detectron2/config/defaults.py) for detailed configuraitons.

## Scene Graph Generation as a ROI head
Following the definition in [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/master/README.md#scene-graph-generation-as-roi_head), sgg models are designed as a roi head, and placed under ```d2sgg/modeling/roi_heads/relation_head```

### Building sgg models
See [building sgg models]()

### SGG Training
You can launch the sgg training with the provided scripts:
```
./scripts/relation_train.sh  # for training
```
The settings for PredCls, SGCls, and SGDet are similar to [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/master/README.md#perform-training-on-scene-graph-generation)

For **Predicate Classification (PredCls)**:
```
MODEL.RELATION.USE_GT_LABEL True MODEL.RELATION.USE_GT_BOX True
```
For **Scene Graph Classificaiton (SGCls)**:
```
MODEL.RELATION.USE_GT_LABEL False MODEL.RELATION.USE_GT_BOX True
```
For **Scene Graph Detection (SGDet)**:
```
MODEL.RELATION.USE_GT_LABEL False MODEL.RELATION.USE_GT_BOX False
```

### SGG Evaluation
You can launch the sgg testing with the provided scripts:
```
./scripts/relation_test.sh  # for training
```

## Citations
If you find this work helpful, please use the following BibTeX entry.
```
@misc{chang2022d2sgg,
  author =       {Chun-Wei Chang},
  title =        {Detectron2-sgg},
  howpublished = {\url{https://github.com/victchang/detectron2-sgg}},
  year =         {2022}
}
```
