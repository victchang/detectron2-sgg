# detectron2-sgg (Will be ready soon)
A scene graph generation codebase attached to detectron2

## TODO
- [ ] add predcls and sgcls
- [ ] add some common benchmarks
- [ ] add some data resampling methods

## Overview
This work provides a clean and general codebase for scene graph generation that is attached to [detectron2](https://github.com/facebookresearch/detectron2). While there already exists [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch), it is built on top of deprecated maskrcnn-benchmark. Being the successor of maskrcnn benchmark, detectron2 has faster speed in both training and inference. As a result, this work is built in order to take such advantage, and deliver an easy-to-use framework like Scene-Graph-Benchmark.pytorch.
This work is modified from [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch) and [fcsgg](https://github.com/liuhengyue/fcsgg).


## Installation
### Requirements
Following detectron2 v0.5, the requirements are
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
## Dataset Preparation (by [fcsgg](https://github.com/liuhengyue/fcsgg))
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
