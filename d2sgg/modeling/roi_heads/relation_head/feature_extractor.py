from typing import Dict, List
import torch
import torch.nn as nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, cat
from detectron2.modeling import build_box_head
from detectron2.modeling.poolers import ROIPooler

from d2sgg.structures import SGInstances

from .utils import (
    pairwise_union,
    resize_boxes,
)

__all__ = ["BoxFeatureExtractor", "UnionFeatureExtractor", "build_box_feature_extractor", "build_union_feature_extractor"]

class BoxFeatureExtractor(nn.Module):

    @configurable
    def __init__(
        self,
        in_features,
        box_pooler,
        box_head,
    ):
        super().__init__()
        self.in_features = in_features
        self.box_pooler = box_pooler
        self.box_head = box_head

    @classmethod
    def from_config(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        return {
            "in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
        }

    def forward(self, features: Dict[str, torch.Tensor], proposals: List[SGInstances]):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[SGInstances]): the per-image object proposals with
                their matching ground truth.
                Each has field "pred_boxes"

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.in_features]
        box_features = self.box_pooler(features, [x.pred_boxes for x in proposals])
        box_features = self.box_head(box_features)
        return box_features

class UnionFeatureExtractor(nn.Module):

    @configurable
    def __init__(
        self,
        in_features,
        box_pooler,
        box_head,
        reduce_channel=None,
        rect_size=None,
        rect_conv=None,
    ):
        super().__init__()

        self.in_features = in_features
        self.box_pooler = box_pooler
        self.box_head = box_head
        self.rect_size = rect_size
        self.rect_conv = rect_conv

    @classmethod
    def from_config(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )

        # union rectangle size
        rect_size = pooler_resolution * 4 - 1
        rect_conv = nn.Sequential(*[
            nn.Conv2d(2, in_channels // 2, kernel_size=7, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels // 2, momentum=0.01),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels, momentum=0.01),
        ])

        return {
            "in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "rect_size": rect_size,
            "rect_conv": rect_conv,
        }

    def forward(self, features: Dict[str, torch.Tensor], proposals: List[SGInstances], rel_pair_idxs=None):
        device = features[self.in_features[0]].device
        union_proposals = []
        rect_inputs = []
        for proposal, rel_pair_idx in zip(proposals, rel_pair_idxs):
            head_proposal = proposal[rel_pair_idx[:, 0]]
            tail_proposal = proposal[rel_pair_idx[:, 1]]
            union_proposals.append(pairwise_union(head_proposal, tail_proposal))

            # if self.rect_conv is not None:
                # use range to construct rectangle, sized (rect_size, rect_size)
            num_rel = len(rel_pair_idx)
            dummy_x_range = torch.arange(self.rect_size, device=device).view(1, 1, -1).expand(num_rel, self.rect_size, self.rect_size)
            dummy_y_range = torch.arange(self.rect_size, device=device).view(1, -1, 1).expand(num_rel, self.rect_size, self.rect_size)
            
            # resize bbox to the scale rect_size
            head_boxes = resize_boxes(head_proposal.pred_boxes, head_proposal.image_size, (self.rect_size, self.rect_size))
            tail_boxes = resize_boxes(tail_proposal.pred_boxes, tail_proposal.image_size, (self.rect_size, self.rect_size))

            head_boxes = head_boxes.tensor
            tail_boxes = tail_boxes.tensor

            head_rect = ( (dummy_x_range >= head_boxes[:, 0].floor().view(-1, 1, 1).long())
                        & (dummy_x_range <= head_boxes[:, 2].ceil().view(-1, 1, 1).long())
                        & (dummy_y_range >= head_boxes[:, 1].floor().view(-1, 1, 1).long())
                        & (dummy_y_range <= head_boxes[:, 3].ceil().view(-1, 1, 1).long())).float()
            tail_rect = ( (dummy_x_range >= tail_boxes[:, 0].floor().view(-1, 1, 1).long())
                        & (dummy_x_range <= tail_boxes[:, 2].ceil().view(-1, 1, 1).long())
                        & (dummy_y_range >= tail_boxes[:, 1].floor().view(-1, 1, 1).long())
                        & (dummy_y_range <= tail_boxes[:, 3].ceil().view(-1, 1, 1).long())).float()

            # (num_rel, 4, rect_size, rect_size)
            rect_input = torch.stack((head_rect, tail_rect), dim=1)
            rect_inputs.append(rect_input)

        # union visual feature. size (total_num_rel, in_channels, POOLER_RESOLUTION, POOLER_RESOLUTION)
        features = [features[f] for f in self.in_features]
        union_vis_features = self.box_pooler(features, [x.union_boxes for x in union_proposals])

        # rectangle feature. size (total_num_rel, in_channels, POOLER_RESOLUTION, POOLER_RESOLUTION)
        rect_inputs = cat(rect_inputs, dim=0)
        rect_features = self.rect_conv(rect_inputs)

        # merge two parts
        union_features = union_vis_features + rect_features
        union_features = self.box_head(union_features)  # (total_num_rel, out_channels)
        return union_features

def build_box_feature_extractor(cfg, input_shape):
    return BoxFeatureExtractor(cfg, input_shape)

def build_union_feature_extractor(cfg, input_shape):
    return UnionFeatureExtractor(cfg, input_shape)