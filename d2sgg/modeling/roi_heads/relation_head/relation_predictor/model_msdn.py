
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.config import configurable
from detectron2.layers import (
    Linear,
    cat,
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

class MessagePassingUnit_v1(nn.Module):
    def __init__(self, input_dim, filter_dim=64):
        """
        Args:
            input_dim:
            filter_dim: the channel number of attention between the nodes
        """
        super(MessagePassingUnit_v1, self).__init__()
        self.w = nn.Sequential(
            nn.LayerNorm(input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, filter_dim, bias=True),
        )

        self.fea_size = input_dim
        self.filter_size = filter_dim

    def forward(self, unary_term, pair_term, aux_gate=None):

        if unary_term.size()[0] == 1 and pair_term.size()[0] > 1:
            unary_term = unary_term.expand(pair_term.size()[0], unary_term.size()[1])
        if unary_term.size()[0] > 1 and pair_term.size()[0] == 1:
            pair_term = pair_term.expand(unary_term.size()[0], pair_term.size()[1])
        
        paired_feats = cat([unary_term, pair_term], 1)

        gate = torch.sigmoid(self.w(paired_feats))
        if gate.shape[1] > 1:
            gate = gate.mean(1)  # average the nodes attention between the nodes

        output = pair_term * gate.view(-1, 1).expand(gate.size()[0], pair_term.size()[1])

        return output, gate

class MessageFusion(nn.Module):
    def __init__(self, input_dim, dropout=False):
        super(MessageFusion, self).__init__()
        self.wih = nn.Linear(input_dim, input_dim, bias=True)
        self.whh = nn.Linear(input_dim, input_dim, bias=True)
        self.dropout = dropout

    def forward(self, input, hidden):
        output = self.wih(F.relu(input)) + self.whh(F.relu(hidden))
        if self.dropout:
            output = F.dropout(output, training=self.training)
        return output

class MSDNContext(nn.Module):

    def __init__(
        self,
        in_dim,
        hidden_dim,
        gate_width,
        num_iters,
        share_params_each_iter,
        merge_predicate_nodes,
    ):
        super().__init__()

        self.num_iters = num_iters
        self.share_params_each_iter = share_params_each_iter
        self.merge_predicate_nodes = merge_predicate_nodes

        self.obj_downdim_fc = nn.Sequential(
            Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.rel_downdim_fc = nn.Sequential(
            Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        if share_params_each_iter:
            num_param_sets = 1
        else:
            num_param_sets = num_iters
        
        self.gate_sub2pred = nn.Sequential(*[MessagePassingUnit_v1(hidden_dim, gate_width) for _ in range(num_param_sets)])
        self.gate_obj2pred = nn.Sequential(*[MessagePassingUnit_v1(hidden_dim, gate_width) for _ in range(num_param_sets)])
        self.gate_pred2sub = nn.Sequential(*[MessagePassingUnit_v1(hidden_dim, gate_width) for _ in range(num_param_sets)])
        self.gate_pred2obj = nn.Sequential(*[MessagePassingUnit_v1(hidden_dim, gate_width) for _ in range(num_param_sets)])

        self.object_msg_fusion = nn.Sequential(*[MessageFusion(hidden_dim) for _ in range(num_param_sets)])
        self.pred_msg_fusion = nn.Sequential(*[MessageFusion(hidden_dim) for _ in range(num_param_sets)])
        
    def _prepare_adjacency_matrix(self, proposals, rel_pair_idxs):
        """
        prepare the index of how subject and object related to the union boxes
        :param num_proposals:
        :param rel_pair_idxs:
        :return:
            ALL RETURN THINGS ARE BATCH-WISE CONCATENATED

            rel_inds,
                extent the instances pairing matrix to the batch wised (num_rel, 2)
            subj_pred_map,
                how the instances related to the relation predicates as the subject (num_inst, rel_pair_num)
            obj_pred_map
                how the instances related to the relation predicates as the object (num_inst, rel_pair_num)
            selected_relness,
                the relatness score for selected relationship proposal that send message to adjency nodes (val_rel_pair_num, 1)
            selected_rel_prop_pairs_idx
                the relationship proposal id that selected relationship proposal that send message to adjency nodes (val_rel_pair_num, 1)
        """
        rel_inds_batch_cat = []
        offset = 0
        num_proposals = [len(props) for props in proposals]

        for prop, rel_ind_i in zip(proposals, rel_pair_idxs):
            rel_ind_i = copy.deepcopy(rel_ind_i)
            rel_ind_i += offset
            offset += len(prop)
            rel_inds_batch_cat.append(rel_ind_i)

        rel_inds_batch_cat = cat(rel_inds_batch_cat, dim=0)

        sub_pred_map = rel_inds_batch_cat.new(sum(num_proposals), rel_inds_batch_cat.shape[0]).fill_(0).float().detach()
        obj_pred_map = rel_inds_batch_cat.new(sum(num_proposals), rel_inds_batch_cat.shape[0]).fill_(0).float().detach()

        # or all relationship pairs
        selected_rel_prop_pairs_idx = torch.arange(len(rel_inds_batch_cat[:, 0]), device=rel_inds_batch_cat.device)
        sub_pred_map.scatter_(0, (rel_inds_batch_cat[:, 0].contiguous().view(1, -1)), 1)
        obj_pred_map.scatter_(0, (rel_inds_batch_cat[:, 1].contiguous().view(1, -1)), 1)

        return (
            rel_inds_batch_cat,
            sub_pred_map,
            obj_pred_map,
            selected_rel_prop_pairs_idx,
        )

    # Here, we do all the operations out of loop, the loop is just to combine the features
    # Less kernel evoke frequency improve the speed of the model
    def prepare_message(
        self,
        target_features,
        source_features,
        select_mat,
        gate_module,
    ):
        """
        generate the message from the source nodes for the following merge operations.

        Then the message passing process can be
        :param target_features: (num_inst, dim)
        :param source_features: (num_rel, dim)
        :param select_mat:  (num_inst, rel_pair_num)
        :param gate_module:
        :param relness_scores: (num_rel, )
        :param relness_logit (num_rel, num_rel_category)

        :return: messages representation: (num_inst, dim)
        """
        feature_data = []
        if select_mat.sum() == 0:
            temp = torch.zeros(
                (target_features.size()[1:]),
                requires_grad=True,
                dtype=target_features.dtype,
                device=target_features.dtype,
            )
            feature_data = torch.stack(temp, 0)
        else:
            transfer_list = (select_mat > 0).nonzero()
            source_indices = transfer_list[:, 1]
            target_indices = transfer_list[:, 0]
            source_f = torch.index_select(source_features, 0, source_indices)
            target_f = torch.index_select(target_features, 0, target_indices)

            transferred_features, weighting_gate = gate_module(target_f, source_f)
            aggregator_matrix = torch.zeros(
                (target_features.shape[0], transferred_features.shape[0]),
                dtype=weighting_gate.dtype,
                device=weighting_gate.device,
            )

            for f_id in range(target_features.shape[0]):
                if select_mat[f_id, :].data.sum() > 0:
                    # average from the multiple sources
                    feature_indices = squeeze_tensor(
                        (transfer_list[:, 0] == f_id).nonzero()
                    )  # obtain source_relevant_idx
                    # (target, source_relevant_idx)
                    aggregator_matrix[f_id, feature_indices] = 1
            # (target, source_relevant_idx) @ (source_relevant_idx, feat-dim) => (target, feat-dim)
            aggregate_feat = torch.matmul(aggregator_matrix, transferred_features)
            avg_factor = aggregator_matrix.sum(dim=1)
            vaild_aggregate_idx = avg_factor != 0
            avg_factor = avg_factor.unsqueeze(1).expand(
                avg_factor.shape[0], aggregate_feat.shape[1]
            )
            aggregate_feat[vaild_aggregate_idx] /= avg_factor[vaild_aggregate_idx]

            feature_data = aggregate_feat
        return feature_data

    def forward(
        self,
        proposals,
        box_features,
        union_features,
        rel_pair_idxs,
        rel_gt_binarys=None,
        logger=None,
    ):
        """
        :param proposals: instance proposals
        :param box_features: obj_num, pooling_dim
        :param union_features:  rel_num, pooling_dim
        :param rel_pair_inds: relaion pair indices list(tensor)
        :param rel_binarys: [num_prop, num_prop] the relatedness of each pair of boxes
        :return:
        """

        # build up list for massage passing process
        inst_feature4iter = [self.obj_downdim_fc(box_features)]
        rel_feature4iter = [self.rel_downdim_fc(union_features)]

        (
            batchwise_rel_pair_inds,
            subj_pred_map,
            obj_pred_map,
            selected_rel_prop_pairs_idx,
        ) = self._prepare_adjacency_matrix(proposals, rel_pair_idxs)

        # graph module
        for t in range(self.num_iters):
            param_idx = 0
            if not self.share_params_each_iter:
                param_idx = t
            """update object features pass message from the predicates to instances"""
            object_sub = self.prepare_message(
                inst_feature4iter[t], rel_feature4iter[t], subj_pred_map, self.gate_pred2sub[param_idx],
            )
            object_obj = self.prepare_message(
                inst_feature4iter[t], rel_feature4iter[t], obj_pred_map, self.gate_pred2obj[param_idx],
            )

            GRU_input_feature_object = (object_sub + object_obj) / 2.0
            inst_feature4iter.append(
                inst_feature4iter[t] + self.object_msg_fusion[param_idx](GRU_input_feature_object, inst_feature4iter[t])
            )

            """update predicate features from entities features"""
            indices_sub = batchwise_rel_pair_inds[:, 0]
            indices_obj = batchwise_rel_pair_inds[:, 1]  # num_rel, 1

            # obj to pred on all pairs
            feat_sub2pred = torch.index_select(inst_feature4iter[t], 0, indices_sub)
            feat_obj2pred = torch.index_select(inst_feature4iter[t], 0, indices_obj)
            phrase_sub, sub2pred_gate_weight = self.gate_sub2pred[param_idx](
                rel_feature4iter[t], feat_sub2pred
            )
            phrase_obj, obj2pred_gate_weight = self.gate_obj2pred[param_idx](
                rel_feature4iter[t], feat_obj2pred
            )
            GRU_input_feature_phrase = (phrase_sub + phrase_obj) / 2.0
            rel_feature4iter.append(
                rel_feature4iter[t] + self.pred_msg_fusion[param_idx](GRU_input_feature_phrase, rel_feature4iter[t])
            )
            
        refined_inst_features = inst_feature4iter[-1]
        refined_rel_features = rel_feature4iter[-1]

        return refined_inst_features, refined_rel_features

@REL_PREDICTOR_REGISTRY.register()
class MSDNPredictor(nn.Module):

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
        in_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM,
        hidden_dim = cfg.MODEL.ROI_RELATION_HEAD.MSDN_MODULE.HIDDEN_DIM
        gate_width = cfg.MODEL.ROI_RELATION_HEAD.MSDN_MODULE.GATE_WIDTH
        num_iters = cfg.MODEL.ROI_RELATION_HEAD.MSDN_MODULE.NUM_ITERS
        share_params_each_iter = cfg.MODEL.ROI_RELATION_HEAD.MSDN_MODULE.SHARE_PARAMS_EACH_ITER
        merge_predicate_nodes = cfg.MODEL.ROI_RELATION_HEAD.MERGE_PREDICATE_NODES
        return {
            "mode": mode,
            "context_layer": MSDNContext(
                in_dim[0], hidden_dim, gate_width, num_iters, share_params_each_iter, merge_predicate_nodes
            ),
            "obj_classifier": build_classifier(hidden_dim, cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES+1),
            "rel_classifier": build_classifier(hidden_dim, cfg.MODEL.ROI_RELATION_HEAD.NUM_PREDICATES+1),
        }

    def forward(
        self,
        proposals,
        box_features,
        union_features,
        rel_pair_idxs,
        rel_binarys,
        logger=None,
    ):
        """
        :param proposals:
        :param box_features:
        :param union_features:
        :param rel_pair_idxs:
        :param rel_labels:
        :param rel_binarys:
            the box pairs with that match the ground truth [num_prp, num_prp]
        :param logger:

        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        
        obj_feats, rel_feats = self.context_layer(
            proposals, box_features, union_features, rel_pair_idxs, rel_binarys, logger
        )

        if self.mode == "predcls":
            obj_labels = cat([proposal.gt_classes for proposal in proposals], dim=0)
            obj_pred_logits = to_onehot(obj_labels, self.num_obj)
        else:
            obj_pred_logits = self.obj_classifier(obj_feats)
        
        rel_pred_logits = self.rel_classifier(rel_feats)

        num_objs = [len(b) for b in proposals]
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        assert len(num_rels) == len(num_objs)

        obj_pred_logits = obj_pred_logits.split(num_objs, dim=0)
        rel_pred_logits = rel_pred_logits.split(num_rels, dim=0)

        return obj_pred_logits, rel_pred_logits

