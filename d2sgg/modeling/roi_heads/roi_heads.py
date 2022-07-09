# Copyright (c) Facebook, Inc. and its affiliates.
import time
import inspect
import logging
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.modeling import (
    ROI_HEADS_REGISTRY,
    StandardROIHeads,
)

from d2sgg.structures import SGInstances
from .relation_head import build_relation_head
from .box_head import FastRCNNOutputLayers

logger = logging.getLogger(__name__)

@ROI_HEADS_REGISTRY.register()
class RelationalROIHeads(StandardROIHeads):
    """
    Including box head and relation head for scene graph generation
    """

    @configurable
    def __init__(
        self,
        *,
        relation_on: bool = False,
        relation_head: nn.Module = None,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            relation_on (bool): whether to do relation prediction.
            relation_head (nn.Module): transform features to make relation prediction.
        """
        super().__init__(**kwargs)
        self.relation_on = relation_on
        self.relation_head = relation_head

        self.relation_head_latency = 0

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret["relation_on"] = cfg.MODEL.RELATION_ON

        if inspect.ismethod(cls._init_box_head):
            ret["box_predictor"] = cls._init_box_predictor(cfg, ret["box_head"].output_shape)
        if inspect.ismethod(cls._init_relation_head):
            ret.update(cls._init_relation_head(cfg, input_shape))
        return ret

    @classmethod
    def _init_box_predictor(cls, cfg, input_shape):
        return FastRCNNOutputLayers(cfg, input_shape)

    @classmethod
    def _init_relation_head(cls, cfg, input_shape):
        if not cfg.MODEL.RELATION_ON:
            return {}
        return {"relation_head": build_relation_head(cfg, input_shape)}

    @torch.no_grad()
    def label_proposals(
        self, proposals: List[Instances], targets: List[Instances]
    ) -> List[Instances]:
        """
        Keep all the proposals and add labels to them.
        Modified from `label_and_sample_proposals`
        """
        proposals_with_gt = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)

            if has_gt > 0:
                gt_classes = targets_per_image.gt_classes[matched_idxs]
                # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
                gt_classes[matched_labels == 0] = self.num_classes
                # Label ignore proposals (-1 label)
                gt_classes[matched_labels == -1] = -1
            else:
                gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

            # Set target attributes of proposals:
            proposals_per_image.gt_classes = gt_classes
            proposals_with_gt.append(proposals_per_image)
        return proposals_with_gt

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images

        if self.training:
            assert targets, "'targets' argument is required during training"
            if self.relation_on:
                proposals = self.label_proposals(proposals, targets)  # for sgg training
                with torch.no_grad():
                    pred_instances = self._forward_box(features, proposals)
                losses = self._forward_relation(features, pred_instances, targets)
            else:
                proposals = self.label_and_sample_proposals(proposals, targets)
                del targets
                losses = self._forward_box(features, proposals)
            return proposals, losses

        else:
            pred_instances = self._forward_box(features, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.
        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.
        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.
        Returns:
            list[Instances]:
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        start = time.perf_counter()
        instances = self._forward_relation(features, instances)
        self.relation_head_latency += (time.perf_counter() - start)
        return instances

    def _forward_box(self, features: Dict[str, torch.Tensor], proposals: List[Instances]):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.
        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".
        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features

        if self.training and not self.relation_on:
            losses = self.box_predictor.losses(predictions, proposals)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances

    def _forward_relation(self, features: Dict[str, torch.Tensor], instances: List[Instances], targets=None):
        """
        Forward logic of the relation prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "rel_inds" and "rel_scores" and return it.
        """
        if not self.relation_on:
            return {} if self.training else instances

        if self.training:
            assert targets is not None, "No targets for training relation head"
            assert len(instances) == len(targets)

        # Convert instances into SGInstances for later relation assignments
        for i in range(len(instances)):
            instances[i] = SGInstances(instances[i].image_size, **instances[i].get_fields())

        return self.relation_head(features, instances, targets)