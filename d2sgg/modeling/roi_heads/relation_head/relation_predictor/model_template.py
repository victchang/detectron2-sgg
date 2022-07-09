
import copy
import torch
import torch.nn.functional as F
from torch import nn

from detectron2.config import configurable

from .classifier import (
    build_classifier,
)

from .build import REL_PREDICTOR_REGISTRY
from detectron2.data import MetadataCatalog, DatasetCatalog

@REL_PREDICTOR_REGISTRY.register()
class TemplatePredictor(nn.Module):  # the name of the predictor

    @configurable
    def __init__(
        self,
        mode,
        # Any argument you would like to add
    ):
        super().__init__()
        self.mode = mode  # to indicate which mode it is (one of predcls, sgcls, and sgdet)

        """
        Other actions
        """
        

    @classmethod
    def from_config(cls, cfg, mode):
        """
        Build the predictor from config
        Returns a dict that will be passed to __init__() to construct the predictor
        """
        return {
            "mode": mode,
        }

    def forward(self, x):
        raise NotImplementedError