import torch
from torch import nn
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from torch import Tensor

model = resnet50(weights=ResNet50_Weights.DEFAULT)

class BackboneBase(nn.Module):
    def __init__(self, backbone, 
            train_backbone : bool, 
            num_channels : int, 
            return_interm_layers : bool,
        ):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone:
                parameter.requires_grad_(False)
        self.num_channels = num_channels
        if return_interm_layers:
            return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
        else:
            return_layers = {'layer4': '0'}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, input):
        xs = self.body(input)
        out : Dict[str, Tensor] = {}
        for name, x in xs.items():
            out[name] = x
        return out
    
class Backbone(BackboneBase):
    def __init__(
        self,
        name : str, 
        train_backbone : bool,
        return_interm_layers : bool,
    ):
        backbone = getattr(torchvision.models, name)(
            weights="DEFAULT",
            norm_layer=torch.nn.BatchNorm2d,
        )
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)

if __name__ == "__main__":
    resnet50 = Backbone('resnet50', False, True)
    input = torch.randn(1, 3, 1024, 1024)
    output = resnet50(input)
    for key, value in output.items():
        print(key, value.shape)


        