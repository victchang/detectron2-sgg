# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn
from detectron2.structures import Boxes
from detectron2.config import configurable

from .utils import obj_prediction_nms

class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    @configurable
    def __init__(
        self,
        num_classes,
        num_predicates,
        use_gt_box=False,
        later_nms_pred_thres=0.3,
    ):
        """
        Arguments:
        """
        super(PostProcessor, self).__init__()
        self.num_classes = num_classes
        self.num_predicates = num_predicates
        self.use_gt_box = use_gt_box
        self.later_nms_pred_thres = later_nms_pred_thres
    
    @classmethod
    def from_config(cls, cfg):
        return {
            "num_classes": cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES,
            "num_predicates": cfg.MODEL.ROI_RELATION_HEAD.NUM_PREDICATES,
            "use_gt_box": cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX,
            "later_nms_pred_thres": cfg.MODEL.ROI_RELATION_HEAD.LATER_NMS_PRED_THRES,
        }

    def forward(self, proposals, obj_pred_logits, rel_pred_logits, rel_pair_idxs):
        """
        Modified from Kaihua's work
        """

        for i, (proposal, obj_logits, rel_logits, rel_pair_idx) in enumerate(
            zip(proposals, obj_pred_logits, rel_pred_logits, rel_pair_idxs)
        ):
            
            obj_class_probs = F.softmax(obj_logits, dim=-1)
            obj_class_probs[:, self.num_classes] = 0  # set background score to 0

            # TODO:
            # if self.use_gt_box:
            #     obj_scores, obj_classes = obj_class_probs.max(dim=1)
            # else:
            #     # NOTE: by kaihua, apply late nms for object prediction
            #     obj_classes = obj_prediction_nms(proposal.pred_boxes_per_cls, obj_logits, self.num_classes, self.later_nms_pred_thres)
            #     obj_score_inds = torch.arange(len(proposal), device=obj_logits.device) * self.num_classes + obj_classes
            #     obj_scores = obj_class_probs.view(-1)[obj_score_inds]

            #     # # TODO
            #     # obj_scores = obj_class_probs[torch.arange(len(proposal)), obj_classes]
            
            # assert obj_scores.shape[0] == len(proposal)

            # if self.use_gt_box:
            #     pred_boxes = proposal.pred_boxes
            # else:
            #     pred_boxes = Boxes(proposal.pred_boxes_per_cls[torch.arange(len(proposal)), obj_classes])

            pred_boxes = proposals[i].pred_boxes
            obj_scores, obj_classes = obj_class_probs.max(dim=-1)
            proposals[i].set_rel_fields("pred_boxes", pred_boxes)
            proposals[i].set_rel_fields("pred_classes", obj_classes)
            proposals[i].set_rel_fields("scores", obj_scores)
            
            # sorting triples according to score production
            rel_class_probs = F.softmax(rel_logits, dim=-1)
            rel_class_probs[:, self.num_predicates] = 0  # set background score to 0

            # TODO Kaihua: how about using weighted sum here?  e.g. rel*1 + obj *0.8 + obj*0.8
            # obj_scores0 = obj_scores[rel_pair_idx[:, 0]]
            # obj_scores1 = obj_scores[rel_pair_idx[:, 1]]
            # rel_scores, _ = rel_class_probs.max(dim=-1)

            # triplet_scores = rel_scores * obj_scores0 * obj_scores1
            # _, sorting_idx = torch.sort(triplet_scores.view(-1), dim=0, descending=True)
            # rel_pair_idx = rel_pair_idx[sorting_idx]
            # rel_class_probs = rel_class_probs[sorting_idx]
            
            # should have fields : rel_inds, rel_scores for relation evaluation
            proposals[i].set_rel_fields("rel_scores", rel_class_probs) # (#rel, #rel_class)
            proposals[i].set_rel_fields("rel_inds", rel_pair_idx) # (#rel, 2)
        return proposals


def build_relation_post_processor(cfg):
    return PostProcessor(cfg)