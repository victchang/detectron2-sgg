"""
Building relation predictor

"""
__author__ = "Victor Chang"
__credits__ = ["Detectron2"]
__license__ = "MIT License"
__version__ = "0.1"
__maintainer__ = "Victor Chang"

import torch
import torch.nn as nn

from detectron2.config import configurable

from .feature_extractor import build_box_feature_extractor, build_union_feature_extractor
from .sampling import build_relation_samp_processor
from .relation_predictor import build_relation_predictor
from .inference import build_relation_post_processor
from .loss import object_loss, relation_loss

class RelationHead(nn.Module):
    """
    Generic Relation Head class.
    """

    @configurable
    def __init__(
        self,
        input_shape,
        samp_processor,
        box_feature_extractor,
        union_feature_extractor,
        predictor,
        post_processor,
        mode,
        num_classes,
        num_predicates,
        use_union_features,
    ):
        super().__init__()

        self.input_shape = input_shape
        self.samp_processor = samp_processor
        self.box_feature_extractor = box_feature_extractor
        self.union_feature_extractor = union_feature_extractor
        self.predictor = predictor
        self.post_processor = post_processor
        self.mode = mode
        self.num_classes = num_classes,
        self.num_predicates = num_predicates
        self.use_union_features = use_union_features

        self.predictor_latency = 0

    @classmethod
    def from_config(cls, cfg, input_shape):
        mode = ("predcls" if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_LABEL else "sgcls") if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX else "sgdet"
        return {
            "input_shape": input_shape,
            "samp_processor": build_relation_samp_processor(cfg),
            "box_feature_extractor": build_box_feature_extractor(cfg, input_shape),
            "union_feature_extractor": build_union_feature_extractor(cfg, input_shape),
            "predictor": build_relation_predictor(cfg, mode),
            "post_processor": build_relation_post_processor(cfg),
            "mode": mode,
            "num_classes": cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES,
            "num_predicates": cfg.MODEL.ROI_RELATION_HEAD.NUM_PREDICATES,
            "use_union_features": cfg.MODEL.ROI_RELATION_HEAD.USE_UNION_FEATURES,
        }

    def forward(self, features, proposals, targets=None, logger=None):
        """
        Arguments:
            features (dict[str, Tensor]): feature-maps from possibly several levels
            proposals (list[Instances]): proposal boxes. Note: it has been post-processed (regression, nms) in sgdet mode
            targets (list[SGInstances], optional): the ground-truth targets.
        """

        if self.training:
            # TODO: relation subsamples and assign ground truth label during training
            with torch.no_grad():
                if self.mode == "predcls" or self.mode == "sgcls":
                    proposals, rel_labels, rel_pair_idxs, rel_binarys = self.samp_processor.gtbox_relsample(proposals, targets)
                else:
                    proposals, rel_labels, rel_pair_idxs, rel_binarys = self.samp_processor.detect_relsample(proposals, targets)
        else:
            rel_labels, rel_binarys = None, None
            rel_pair_idxs = self.samp_processor.prepare_test_pairs(list(features.values())[0].device, proposals)

        # use box_head to extract features that will be fed to the later predictor processing
        box_features = self.box_feature_extractor(features, proposals)

        if self.use_union_features:
            union_features = self.union_feature_extractor(features, proposals, rel_pair_idxs)
        else:
            union_features = None
        
        import time; start = time.perf_counter()
        obj_pred_logits, rel_pred_logits = self.predictor(
            proposals, box_features, union_features, rel_pair_idxs, rel_binarys, logger
        )
        self.predictor_latency += (time.perf_counter() - start)
        
        # for inference
        if not self.training:
            return self.post_processor(proposals, obj_pred_logits, rel_pred_logits, rel_pair_idxs)

        loss_objects = object_loss(obj_pred_logits, [p.gt_classes for p in proposals])
        loss_relations = relation_loss(rel_pred_logits, rel_labels)
        losses = {"loss_obj": loss_objects, "loss_rel": loss_relations}
        return losses

def build_relation_head(cfg, input_shape):
    """
    Constructs a new relation head (RelationHead).
    """
    return RelationHead(cfg, input_shape)