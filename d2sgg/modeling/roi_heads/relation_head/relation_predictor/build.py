# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.utils.registry import Registry

REL_PREDICTOR_REGISTRY = Registry("REL_PREDICTOR")  # noqa F401 isort:skip
REL_PREDICTOR_REGISTRY.__doc__ = """
Registry for predictors.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_relation_predictor(cfg, input_dim):
    """
    Build the relation predictor, defined by `cfg.ROI_RELATION_HEAD.PREDICTOR`.
    Note that it does not load any weights from `cfg`.
    """
    predictor = cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR
    predictor = REL_PREDICTOR_REGISTRY.get(predictor)(cfg, input_dim)
    return predictor