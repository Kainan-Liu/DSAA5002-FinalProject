import torch
import torch.nn as nn
from typing import Union, Optional

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
            nn.Tanh() if act else nn.Identity()
        )

    def forward(self, x):
        return self.linear(x)
    

class ANNet(nn.Module):
    def __init__(self, in_features, out_features, *args, **kwargs) -> None:
        super(ANNet, self).__init__(*args, **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        
        hidden_features = [16, 32, 64]
        self.model = nn.Sequential(
            LinearBlock(in_features=in_features, out_features=hidden_features[0]),
            LinearBlock(in_features=hidden_features[0], out_features=hidden_features[1]),
            LinearBlock(in_features=hidden_features[1], out_features=hidden_features[2]),
            LinearBlock(in_features=hidden_features[2], out_features=out_features, act=False, last=True)
        )

    def forward(self, x):
        return self.model(x)
    
class NNNet(nn.Module):
    def __init__(self, in_features, out_features, *args, **kwargs) -> None:
        super(NNNet, self).__init__(*args, **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        
        hidden_features = [16, 32]
        self.model = nn.Sequential(
            LinearBlock(in_features=in_features, out_features=hidden_features[0]),
            LinearBlock(in_features=hidden_features[0], out_features=hidden_features[1]),
            LinearBlock(in_features=hidden_features[1], out_features=out_features, act=False, last=True)
        )

    def forward(self, x):
        return self.model(x)