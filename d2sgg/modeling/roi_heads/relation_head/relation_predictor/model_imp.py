# modified from https://github.com/rowanz/neural-motifs
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.config import configurable
from detectron2.layers import (
    Linear,
)

from ..utils import (
    to_onehot,
    squeeze_tensor,
)

from .classifier import (
    build_classifier,
)

from .build import REL_PREDICTOR_REGISTRY
from detectron2.data import MetadataCatalog, DatasetCatalog


class IMPContext(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, num_iters=3):
        super(IMPContext, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_iters = num_iters

        self.obj_unary = Linear(in_dim, hidden_dim)
        self.edge_unary = Linear(in_dim, hidden_dim)

        self.edge_gru = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)
        self.node_gru = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)

        self.sub_vert_w_fc = nn.Sequential(nn.Linear(hidden_dim*2, 1), nn.Sigmoid())
        self.obj_vert_w_fc = nn.Sequential(nn.Linear(hidden_dim*2, 1), nn.Sigmoid())
        self.out_edge_w_fc = nn.Sequential(nn.Linear(hidden_dim*2, 1), nn.Sigmoid())
        self.in_edge_w_fc = nn.Sequential(nn.Linear(hidden_dim*2, 1), nn.Sigmoid())
    
    def forward(self, proposals, box_features, union_features, rel_pair_idxs, logger=None):
        num_objs = [len(b) for b in proposals]

        obj_rep = self.obj_unary(box_features)
        rel_rep = F.relu(self.edge_unary(union_features))

        obj_count = obj_rep.shape[0]
        rel_count = rel_rep.shape[0]

        # generate sub-rel-obj mapping
        sub2rel = torch.zeros(obj_count, rel_count).to(obj_rep.device).float()
        obj2rel = torch.zeros(obj_count, rel_count).to(obj_rep.device).float()
        obj_offset = 0
        rel_offset = 0
        sub_global_inds = []
        obj_global_inds = []
        for pair_idx, num_obj in zip(rel_pair_idxs, num_objs):
            num_rel = pair_idx.shape[0]
            sub_idx = pair_idx[:,0].contiguous().long().view(-1) + obj_offset
            obj_idx = pair_idx[:,1].contiguous().long().view(-1) + obj_offset
            rel_idx = torch.arange(num_rel).to(obj_rep.device).long().view(-1) + rel_offset

            sub_global_inds.append(sub_idx)
            obj_global_inds.append(obj_idx)

            sub2rel[sub_idx, rel_idx] = 1.0
            obj2rel[obj_idx, rel_idx] = 1.0

            obj_offset += num_obj
            rel_offset += num_rel

        sub_global_inds = torch.cat(sub_global_inds, dim=0)
        obj_global_inds = torch.cat(obj_global_inds, dim=0)

        # iterative message passing
        hx_obj = torch.zeros(obj_count, self.hidden_dim, requires_grad=False).to(obj_rep.device).float()
        hx_rel = torch.zeros(rel_count, self.hidden_dim, requires_grad=False).to(obj_rep.device).float()

        vert_factor = [self.node_gru(obj_rep, hx_obj)]
        edge_factor = [self.edge_gru(rel_rep, hx_rel)]

        for i in range(self.num_iters):
            # compute edge context
            sub_vert = vert_factor[i][sub_global_inds]
            obj_vert = vert_factor[i][obj_global_inds]
            weighted_sub = self.sub_vert_w_fc(
                torch.cat((sub_vert, edge_factor[i]), 1)) * sub_vert
            weighted_obj = self.obj_vert_w_fc(
                torch.cat((obj_vert, edge_factor[i]), 1)) * obj_vert

            edge_factor.append(self.edge_gru(weighted_sub + weighted_obj, edge_factor[i]))

            # Compute vertex context
            pre_out = self.out_edge_w_fc(torch.cat((sub_vert, edge_factor[i]), 1)) * edge_factor[i]
            pre_in = self.in_edge_w_fc(torch.cat((obj_vert, edge_factor[i]), 1)) * edge_factor[i]
            vert_ctx = sub2rel @ pre_out + obj2rel @ pre_in
            vert_factor.append(self.node_gru(vert_ctx, vert_factor[i]))

        obj_feats = vert_factor[-1]
        rel_feats = edge_factor[-1]

        return obj_feats, rel_feats

@REL_PREDICTOR_REGISTRY.register()
class IMPPredictor(nn.Module):

    @configurable
    def __init__(
        self,
        mode,
        context_layer,
        obj_classifier,
        rel_classifier,
    ):

        super().__init__()

        self.mode = mode
        self.context_layer = context_layer
        self.obj_classifier = obj_classifier
        self.rel_classifier = rel_classifier

    @classmethod
    def from_config(cls, cfg, mode):
        in_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        hidden_dim = cfg.MODEL.RELATION.IMP_MODULE.HIDDEN_DIM
        num_iters = cfg.MODEL.RELATION.IMP_MODULE.NUM_ITERS
        return {
            "mode": mode,
            "context_layer": IMPContext(cfg, mode, in_dim, hidden_dim, num_iters),
            "obj_classifier": build_classifier(cfg, hidden_dim, cfg.MODEL.RELATION.NUM_CLASSES+1),
            "rel_classifier": build_classifier(cfg, hidden_dim, cfg.MODEL.RELATION.NUM_PREDICATES+1),
        }

    def forward(
        self,
        proposals,
        box_features,
        union_features,
        rel_pair_idxs
    ):
        obj_feats, rel_feats = self.context_layer(
            proposals, box_features, union_features, rel_pair_idxs
        )

        obj_pred_logits = self.obj_classifier(obj_feats)
        rel_pred_logits = self.rel_classifier(rel_feats)

        num_objs = [len(b) for b in proposals]
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        assert len(num_rels) == len(num_objs)

        obj_pred_logits = obj_pred_logits.split(num_objs, dim=0)
        rel_pred_logits = rel_pred_logits.split(num_rels, dim=0)

        return obj_pred_logits, rel_pred_logits
