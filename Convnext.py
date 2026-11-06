import torch
import torch.nn as nn
import math


def DropPath(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class LayerNorm2d(nn.Module):
    
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=(7,1), padding=(3,0), groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(3 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = nn.Identity() if drop_path <= 0. else lambda x: DropPath(x, drop_path, self.training)

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = shortcut + self.drop_path(x)
        return x

class HeightWiseDownsample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.norm = LayerNorm2d(in_dim)
        self.reduction = nn.Conv2d(in_dim, out_dim, kernel_size=(2, 1), stride=(2, 1))
    def forward(self, x):
        x = self.norm(x)
        x = self.reduction(x)
        return x

class ConvNeXt(nn.Module):
 
    def __init__(self, input_channels=1, num_classes=1000,
                 depths=[2, 2, 4, 2], dims=[16, 32, 64, 128],
                 drop_path_rate=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.num_classes = num_classes
        
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, dims[0], kernel_size=(2, 1), stride=(2, 1)),
            LayerNorm2d(dims[0])
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        cur = 0
        for i in range(4):
            stage = nn.Sequential(*[
                ConvNeXtBlock(dim=dims[i], drop_path=dpr[cur + j], layer_scale_init_value=layer_scale_init_value)
                for j in range(depths[i])
            ])
            self.stages.append(stage)
            cur += depths[i]
            if i < 3:
                self.downsamples.append(HeightWiseDownsample(dims[i], dims[i+1]))

        
        self.norm = LayerNorm2d(dims[-1])
        
        self.head = nn.Linear(dims[-1] * 1, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.stem(x)                     
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < 3:
                x = self.downsamples[i](x)   
        
        
        x = x.mean(dim=2)                    
        
        x = x.flatten(1)                     
        
        x = self.head(x)
        return x
