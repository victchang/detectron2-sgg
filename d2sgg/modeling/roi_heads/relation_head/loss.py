# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from detectron2.layers import cat, cross_entropy

def object_loss(obj_pred_logits, targets):
    obj_pred_logits = cat(obj_pred_logits, dim=0)
    targets = cat(targets, dim=0)
    loss = cross_entropy(obj_pred_logits, targets.long())
    return loss

def relation_loss(rel_pred_logits, targets):
    rel_pred_logits = cat(rel_pred_logits, dim=0)
    targets = cat(targets, dim=0)
    loss = cross_entropy(rel_pred_logits, targets.long())
    return loss
