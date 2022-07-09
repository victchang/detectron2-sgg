import math
import torch
from torch import nn
from torch.nn import functional as F, init

class DotProductClassifier(nn.Module):

    def __init__(
        self,
        in_dim,
        num_classes,
        bias=True,
        learnable_scale=False
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(num_classes, in_dim))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_classes))
        self.scales = None
        if learnable_scale:
            self.scales = nn.Parameter(torch.ones(num_classes))

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def fix_weights(self, requires_grad=False):
        self.weight.requires_grad = requires_grad
        if self.bias is not None:
            self.bias.requires_grad = requires_grad

    def forward(self, input):
        output = F.linear(input, self.weight, self.bias)
        if self.scales is not None:
            output *= self.scales
        return output

def build_classifier(in_dim, num_classes, bias=True):
    return DotProductClassifier(in_dim, num_classes, bias)