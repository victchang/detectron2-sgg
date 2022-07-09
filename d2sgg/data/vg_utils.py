"""
Core modules for generating FCSGG ground-truth data.
"""
__author__ = "Hengyue Liu"
__copyright__ = "Copyright (c) 2021 Futurewei Inc."
__credits__ = ["Detectron2"]
__license__ = "MIT License"
__version__ = "0.1"
__maintainer__ = "Hengyue Liu"
__email__ = "onehothenry@gmail.com"

import logging
from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
from detectron2.data import transforms as T
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import (
    Boxes,
    BoxMode,
)

from d2sgg.structures import SGInstances

__all__ = [
    "build_augmentation",
    "transform_instance_annotations",
    "annotations_to_scene_graph",
]

DEFAULT_FIELDS = ["gt_classes", "gt_ct_maps", "gt_wh", "gt_reg", "gt_centers_int",
    "gt_relations", "gt_relations_weights"]

SIZE_RANGE = {4: (0., 1/16), 8: (1/16, 1/8), 16: (1/8, 1/4), 32: (1/4, 1.)}

# SIZE_RANGE = {8: (0., 1/16), 16: (1/16, 1/8), 32: (1/8, 1/4), 64: (1/4, 1/2), 128: (1/2, 1.)}

dataset_meta = MetadataCatalog.get("vg_train")

def build_augmentation(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    augmentation = [T.ResizeShortestEdge(min_size, max_size, sample_style)]
    if is_train and cfg.INPUT.FLIP:
        augmentation.append(T.RandomFlip())
    return augmentation

def transform_instance_annotations(
    annotation, transforms, image_size, *, keypoint_hflip_indices=None
):
    """
    Apply transforms to box, segmentation and keypoints annotations of a single instance.
    It will use `transforms.apply_box` for the box, and
    `transforms.apply_coords` for segmentation polygons & keypoints.
    If you need anything more specially designed for each data structure,
    you'll need to implement your own version of this function or the transforms.
    Args:
        annotation (dict): dict of instance annotations for a single instance.
            It will be modified in-place.
        transforms (TransformList or list[Transform]):
        image_size (tuple): the height, width of the transformed image
    Returns:
        dict:
            the same input dict with fields "bbox", "segmentation", "keypoints"
            transformed according to `transforms`.
            The "bbox_mode" field will be set to XYXY_ABS.
    """
    if isinstance(transforms, (tuple, list)):
        transforms = T.TransformList(transforms)
    # bbox is 1d (per-instance bounding box)
    bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    # clip transformed bbox to image size
    bbox = transforms.apply_box(np.array([bbox]))[0].clip(min=0)
    annotation["bbox"] = np.minimum(bbox, list(image_size + image_size)[::-1])
    annotation["bbox_mode"] = BoxMode.XYXY_ABS

    return annotation


def annotations_to_instances(annos, relations, image_size):
    """
        Create an :class:`Instances` object used by the models,
        from instance annotations in the dataset dict.
        Args:
            annos (list[dict]): a list of instance annotations in one image, each
                element for one instance.
            relations (list[list]): (N, 3) triplet for N relations <s, o, predicate label>
            image_size (tuple): height, width
        Returns:
            Instances:
                It will contain fields "gt_boxes", "gt_classes",
                "gt_masks", "gt_keypoints", if they can be obtained from `annos`.
                This is the format that builtin models expect.
        """
    boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
    scene_graph = SGInstances(image_size)
    gt_boxes = Boxes(boxes)

    gt_classes = [obj["category_id"] for obj in annos]
    gt_classes = torch.tensor(gt_classes, dtype=torch.int64)

    # we filter empty box here, as well as relations
    valid = gt_boxes.nonempty()
    gt_boxes = gt_boxes[valid]
    gt_classes = gt_classes[valid]
    if relations and (not valid.all()):
        valid_box_inds = valid.nonzero(as_tuple=True)[0]
        valid_box_inds = valid_box_inds.numpy()
        old2new = {ind: i for i, ind in enumerate(valid_box_inds)}
        filtered_relations = []
        for i in range(len(relations)):
            s_ind, o_ind, r = relations[i]
            if (s_ind not in valid_box_inds) or (o_ind not in valid_box_inds):
                continue
            # map to the new index
            filtered_relations.append([old2new[s_ind], old2new[o_ind], r])
        relations = filtered_relations
    # common fields
    scene_graph.gt_boxes = gt_boxes
    scene_graph.gt_classes = gt_classes
    relations = torch.as_tensor(relations, dtype=torch.long)
    # in case we got empty tensor
    if relations.numel() == 0:
        relations = relations.reshape((0, 3))
    scene_graph.set_rel_fields("gt_boxes", gt_boxes)
    scene_graph.set_rel_fields("gt_classes", gt_classes)
    scene_graph.set_rel_fields("gt_relations", relations)
    return scene_graph