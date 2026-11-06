







import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import DropPath, trunc_normal_
import math




class StripConv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, 1)
        if isinstance(padding, int):
            padding = (padding, 0)
        if isinstance(stride, int):
            stride = (stride, 1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        return self.conv(x)

class StripDepthwiseConv(nn.Module):
    
    def __init__(self, dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, 1)
        if isinstance(padding, int):
            padding = (padding, 0)
        if isinstance(stride, int):
            stride = (stride, 1)
        self.dw = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=stride,
                            padding=padding, groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim + 1)

    def forward(self, x):
        return self.bn(self.dw(x))




class RepMixer(nn.Module):
    



    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.dwconv = StripDepthwiseConv(dim, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.act = nn.GELU()

    def forward(self, x):
        x = x + self.act(self.dwconv(x))
        return x

class LargeKernelConv(nn.Module):
    
    def __init__(self, dim, kernel_size=7, expand_ratio=2.0, drop_path=0.):
        super().__init__()
        hidden_dim = int(dim * expand_ratio)
        self.dw = StripDepthwiseConv(dim, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.pw1 = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False)
        self.pw2 = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.bn1 = nn.BatchNorm2d(dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        identity = x
        x = self.dw(x)
        x = self.act(self.bn2(self.pw1(x)))
        x = self.bn1(self.pw2(x))
        return identity + self.drop_path(x)

class Attention(nn.Module):
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class AttentionBlock(nn.Module):
    
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x




class RepMixerStage(nn.Module):
    
    def __init__(self, dim, depth, kernel_size=3, large_kernel_size=7, expand_ratio=2.0, drop_path=0.):
        super().__init__()
        
        if isinstance(drop_path, (list, tuple)):
            dpr = drop_path
            assert len(dpr) == depth
        else:
            dpr = [drop_path] * depth

        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(nn.ModuleList([
                RepMixer(dim, kernel_size=kernel_size),
                LargeKernelConv(dim, kernel_size=large_kernel_size, expand_ratio=expand_ratio, drop_path=dpr[i])
            ]))

    def forward(self, x):
        for mixer, lk in self.blocks:
            x = mixer(x)
            x = lk(x)   
        return x

class AttentionStage(nn.Module):
    
    def __init__(self, dim, depth, num_heads=8, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        if isinstance(drop_path, (list, tuple)):
            dpr = drop_path
            assert len(dpr) == depth
        else:
            dpr = [drop_path] * depth

        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(AttentionBlock(dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                               qkv_bias=True, drop=drop, attn_drop=attn_drop,
                                               drop_path=dpr[i]))

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  
        for blk in self.blocks:
            x = blk(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x




class HeightDownsample(nn.Module):
    
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=(2, 1), stride=(2, 1), bias=False)
        self.bn = nn.BatchNorm2d(out_dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))




class FastViT_HOnly(nn.Module):
    



    def __init__(self, input_channels=1, num_classes=10, img_height=128, img_width=9,
                 depths=[2, 2, 6, 2], dims=[16, 32, 64, 128],
                 kernel_sizes=[3, 3, 3], large_kernel_sizes=[7, 7, 7], expand_ratio=2.5,
                 num_heads=4, mlp_ratio=2., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.dims = dims

        
        self.stem = nn.Sequential(
            StripConv2d(input_channels, dims[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.GELU(),
        )

        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        in_ch = dims[0]  
        idx = 0

        for i in range(len(depths)):
            depth = depths[i]
            

            if i < 3:  
                stage = RepMixerStage(in_ch, depth, kernel_size=kernel_sizes[i],
                                      large_kernel_size=large_kernel_sizes[i],
                                      expand_ratio=expand_ratio, drop_path=dpr[idx:idx + depth])
            else:  
                stage = AttentionStage(in_ch, depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                       drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[idx:idx + depth])
            self.stages.append(stage)
            idx += depth

            
            if i < len(depths) - 1:
                next_ch = dims[i + 1]
                self.downsamples.append(HeightDownsample(in_ch, next_ch))
                in_ch = next_ch  
            else:
                self.downsamples.append(nn.Identity())

        
        self.norm = nn.BatchNorm2d(in_ch)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(in_ch * 1, num_classes) if num_classes > 0 else nn.Identity()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.stem(x)                     

        for i, stage in enumerate(self.stages):
            x = stage(x)
            x = self.downsamples[i](x)       

        x = self.norm(x)
        x = self.avgpool(x)                  
        x = x.flatten(1)                     
        x = self.head(x)
        return x
