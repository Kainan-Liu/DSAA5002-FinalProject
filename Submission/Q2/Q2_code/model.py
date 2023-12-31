import torch
import torch.nn as nn
import torchvision.models as pretrain_model
from typing import Optional
from collections import OrderedDict


class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features, act: Optional[bool] = True, last: bool = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.act = act
        self.last = last

        self.linear = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features, bias=True if not self.last else False),
            nn.BatchNorm1d(out_features) if not last else nn.Identity(),
            nn.ReLU() if act else nn.Identity()
        )

    def forward(self, x):
        return self.linear(x)


class MyResNet(nn.Module):
    def __init__(self,
                 num_classes: int,
                 in_features: int = 1000, 
                 hidden_features: Optional[list] = [512, 256, 64],
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.hidden_features = hidden_features

        self.backbone = pretrain_model.resnet34(weights=pretrain_model.ResNet34_Weights.DEFAULT)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        self.classifer = nn.Sequential(
            LinearBlock(in_features=in_features, out_features=hidden_features[0]),
            LinearBlock(in_features=hidden_features[0], out_features=hidden_features[1]),
            LinearBlock(in_features=hidden_features[1], out_features=hidden_features[2]),
            LinearBlock(in_features=hidden_features[2], out_features=self.num_classes, act=False, last=True)
        )
        
        layers = OrderedDict([
            ("backbone", self.backbone),
            ("classifier", self.classifer)
        ])

        self.model = nn.Sequential(layers)

    def forward(self, x):
        return self.model(x)