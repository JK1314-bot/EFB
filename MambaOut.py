






from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath





class StemLayerStrip(nn.Module):
    
    def __init__(self, in_channels=3, out_channels=96, act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.norm = norm_layer(out_channels)
        self.act = act_layer()

    def forward(self, x):
        
        x = self.conv(x)                      
        x = x.permute(0, 2, 3, 1)              
        x = self.norm(x)
        x = self.act(x)
        return x


class DownsampleLayerStrip(nn.Module):
    
    def __init__(self, in_channels, out_channels, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.norm = norm_layer(out_channels)

    def forward(self, x):
        
        x = x.permute(0, 3, 1, 2)              
        x = self.conv(x)                       
        x = x.permute(0, 2, 3, 1)              
        x = self.norm(x)
        return x


class GatedCNNBlockStrip(nn.Module):
    


    def __init__(self, dim, expansion_ratio=8/3, kernel_size=7, conv_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 act_layer=nn.GELU, drop_path=0., **kwargs):
        super().__init__()
        self.norm = norm_layer(dim)
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.act = act_layer()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)
        
        self.conv = nn.Conv2d(conv_channels, conv_channels,
                              kernel_size=(kernel_size, 1),
                              padding=(kernel_size // 2, 0),
                              groups=conv_channels)
        self.fc2 = nn.Linear(hidden + 1, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x                            
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=-1)
        
        c = c.permute(0, 3, 1, 2)
        c = self.conv(c)                        
        c = c.permute(0, 2, 3, 1)               
        x = self.fc2(self.act(g) * torch.cat((i, c), dim=-1))
        x = self.drop_path(x)
        return x + shortcut






class MambaOut(nn.Module):

    def __init__(self, input_channels=1, num_classes=10,
                 depths=[2, 2, 3, 2],
                 dims=[16, 32, 64, 128],
                 kernel_size=7,
                 conv_ratio=1.0,
                 drop_path_rate=0.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 act_layer=nn.GELU,
                 head_dropout=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.num_stage = len(depths)

        
        
        self.downsample_layers = nn.ModuleList()
        
        self.downsample_layers.append(
            StemLayerStrip(in_channels=input_channels, out_channels=dims[0],
                           act_layer=act_layer, norm_layer=norm_layer)
        )
        
        for i in range(1, self.num_stage):
            self.downsample_layers.append(
                DownsampleLayerStrip(in_channels=dims[i-1], out_channels=dims[i],
                                     norm_layer=norm_layer)
            )

        
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.stages = nn.ModuleList()
        cur = 0
        for i in range(self.num_stage):
            stage = nn.Sequential(*[
                GatedCNNBlockStrip(
                    dim=dims[i],
                    expansion_ratio=8/3,
                    kernel_size=kernel_size,
                    conv_ratio=conv_ratio,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    drop_path=dp_rates[cur + j]
                ) for j in range(depths[i])
            ])
            self.stages.append(stage)
            cur += depths[i]

        
        self.norm = norm_layer(dims[-1] * 1)   
        self.head = nn.Linear(dims[-1] * 1, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        
        for i in range(self.num_stage):
            x = self.downsample_layers[i](x)      
            x = self.stages[i](x)                 
        
        
        x = x.mean(dim=1)                         
        x = x.flatten(1)                          
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.norm(x)
        x = self.head(x)
        return x






def mambaout_strip_tiny(num_classes=10, in_chans=1, **kwargs):
    
    model = MambaOut(
        in_chans=in_chans,
        num_classes=num_classes,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 576],
        kernel_size=7,
        conv_ratio=1.0,
        drop_path_rate=0.1,
        **kwargs
    )
    return model


def mambaout_strip_small(num_classes=10, in_chans=1, **kwargs):
    
    model = MambaOut(
        in_chans=in_chans,
        num_classes=num_classes,
        depths=[3, 4, 27, 3],
        dims=[96, 192, 384, 576],
        kernel_size=7,
        conv_ratio=1.0,
        drop_path_rate=0.1,
        **kwargs
    )
    return model
