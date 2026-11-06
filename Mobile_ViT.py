import torch
import torch.nn as nn
import torch.nn.functional as F

def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, groups=1):
        super().__init__()
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, 1)
            padding = (padding, 0)
            stride = (stride, 1)

        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size,
                                         stride, padding, groups=groups, bias=False))
        self.add_module('bn', nn.BatchNorm2d(out_channels))
        self.add_module('relu', nn.ReLU6(inplace=True))

class InvertedResidual(nn.Module):
    
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))

        
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_h, mlp_dim, heads=8, dropout=0.):
        super().__init__()
        self.ph = patch_h
        
        self.local_rep = nn.Sequential(
            ConvBNReLU(channel, channel, kernel_size=kernel_size, padding=kernel_size//2),
            nn.Conv2d(channel, dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(dim),
        )

        
        self.transformer = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)
        ])

        
        self.conv_proj = ConvBNReLU(dim, channel, kernel_size=1)
        self.fusion = ConvBNReLU(2 * channel, channel, kernel_size=kernel_size, padding=kernel_size//2)

    def forward(self, x):
        shortcut = x
        
        res = self.local_rep(x)
        B, C, H, W = res.shape

        
        pad_h = (self.ph - H % self.ph) % self.ph
        if pad_h > 0:
            
            res = F.pad(res, (0, 0, 0, pad_h))

            
        H_padded = H + pad_h
        new_h = H_padded // self.ph
        

        
        
        x = res.view(B, C, new_h, self.ph, W).permute(0, 3, 4, 2, 1).reshape(-1, new_h, C)

        for blk in self.transformer:
            x = blk(x)

        
        x = x.view(B, self.ph, W, new_h, C).permute(0, 4, 3, 1, 2).reshape(B, C, H_padded, W)

        
        if pad_h > 0:
            x = x[:, :, :H, :]
        

        x = self.conv_proj(x)

        
        x = self.fusion(torch.cat((shortcut, x), dim=1))
        return x


class MobileViT(nn.Module):

    def __init__(self, num_classes=6, input_channels=1, patch_h=4,
                 depths=[1, 1, 1, 1],
                 channels=[16, 32, 64, 128]):
        super().__init__()

        
        self.conv1 = ConvBNReLU(input_channels, channels[0], kernel_size=3, stride=2, padding=1)

        layers1 = [InvertedResidual(channels[0], channels[0], stride=1, expand_ratio=2)]
        
        for _ in range(depths[0]):
            layers1.append(MobileViTBlock(dim=channels[0], depth=1, channel=channels[0],
                                          kernel_size=3, patch_h=patch_h, mlp_dim=channels[0] * 2))
        self.stage1 = nn.Sequential(*layers1)

        
        
        
        layers2 = [InvertedResidual(channels[0], channels[1], stride=2, expand_ratio=2)]
        for _ in range(depths[1]):
            layers2.append(MobileViTBlock(dim=channels[1], depth=1, channel=channels[1],
                                          kernel_size=3, patch_h=patch_h, mlp_dim=channels[1] * 2))
        self.stage2 = nn.Sequential(*layers2)

        
        
        
        layers3 = [InvertedResidual(channels[1], channels[2], stride=2, expand_ratio=2)]
        for _ in range(depths[2]):
            layers3.append(MobileViTBlock(dim=channels[2], depth=1, channel=channels[2],
                                          kernel_size=3, patch_h=max(1, patch_h // 2), mlp_dim=channels[2] * 2))
        self.stage3 = nn.Sequential(*layers3)

        
        
        
        layers4 = [InvertedResidual(channels[2], channels[3], stride=2, expand_ratio=2)]
        for _ in range(depths[3]):
            layers4.append(MobileViTBlock(dim=channels[3], depth=1, channel=channels[3],
                                          kernel_size=3, patch_h=1, mlp_dim=channels[3] * 2))
        self.stage4 = nn.Sequential(*layers4)

        
        
        
        self.conv_last = ConvBNReLU(channels[3], 128, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(128*1, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.conv_last(x)
        x = self.pool(x).flatten(1)
        x = self.classifier(x)
        return x
