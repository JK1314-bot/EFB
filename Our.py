from functools import partial
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from collections import OrderedDict

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, dropout=0, norm=nn.BatchNorm2d, act_func=nn.ReLU):
        super().__init__()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        )
        self.norm = norm(out_channels) if norm else None
        self.act = act_func() if act_func else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x

class BilinearMapping(nn.Module):
    def __init__(self, dim, sequence_length, sensor_count, low_rank_dim=16):
        super().__init__()
        self.dim = dim
        self.sequence_length = sequence_length
        self.sensor_count = sensor_count
        self.low_rank_dim = low_rank_dim
        self.proj = nn.Linear(dim, 2 * dim)
        self.act = nn.SiLU()
        
        self.proj_k = nn.Sequential(
            nn.Linear(sequence_length, low_rank_dim),
            nn.SiLU(),
            nn.Linear(low_rank_dim, sequence_length),
        )
        self.proj_v = nn.Linear(dim, dim)
        self.mixer = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim + 1)

    def forward(self, x):
        batch_size, _, feature_dim = x.shape
        x = x.view(batch_size, self.sequence_length, self.sensor_count, feature_dim)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size * self.sensor_count, self.sequence_length, feature_dim)
        shortcut = x

        
        x = self.proj(x)
        k, v = x.chunk(2, dim=-1)

        
        k = k.transpose(-2, -1)
        k_proj = self.proj_k(k)
        qk = k_proj.transpose(-2, -1)

        qk = self.mixer(qk)
        v = self.proj_v(v)

        x = self.act(qk) * v                 
        x = self.norm(x)
        x = x + shortcut                      

        x = x.view(batch_size, self.sensor_count, self.sequence_length, feature_dim)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, self.sequence_length * self.sensor_count, feature_dim)
        return x

class EFBMBlock(nn.Module):
    def __init__(self, dim, sensor_resolution, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.sensor_resolution = sensor_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        
        self.cpe1 = nn.Conv2d(dim, dim, kernel_size=(5,1), padding=(5//2,0), groups=dim)
        self.norm1 = norm_layer(dim)
        self.in_proj = nn.Linear(dim, dim)
        self.act_proj = nn.Linear(dim, dim)
        self.dwc = nn.Conv2d(dim, dim, kernel_size=(5,1), padding=(5//2,0), groups=dim)
        self.act = nn.SiLU()
        sequence_length, sensor_count = sensor_resolution
        self.efbm = BilinearMapping(dim, sequence_length, sensor_count, low_rank_dim=16)
        self.out_proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        
        self.cross_sensor = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=(1,5), padding=(0,5//2)),
            nn.SiLU()
        )

        self.cpe2 = nn.Conv2d(dim, dim, kernel_size=(5,1), padding=(5//2,0), groups=dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        sequence_length, sensor_count = self.sensor_resolution
        B, L, C = x.shape
        assert L == sequence_length * sensor_count, "Input feature has incorrect size"

        
        x = x + self.cpe1(x.reshape(B, sequence_length, sensor_count, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        shortcut = x

        x = self.norm1(x)
        act_res = self.act(self.act_proj(x))
        x = self.in_proj(x).view(B, sequence_length, sensor_count, C)
        x = self.act(self.dwc(x.permute(0, 3, 1, 2))).permute(0, 2, 3, 1).view(B, L, C)

        x = self.efbm(x)   

        
        x = x.transpose(1, 2).contiguous().view(B, C, sequence_length, sensor_count)
        x = self.cross_sensor(x)                
        x = x.flatten(2).transpose(1, 2)                       
        

        x = self.out_proj(x * act_res)   

        x = shortcut + self.drop_path(x)
        x = x + self.cpe2(x.reshape(B, sequence_length, sensor_count, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, sensor_resolution, depth, num_heads, mlp_ratio=4., num_classes=1000,
                 qkv_bias=True, drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.sensor_resolution = sensor_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            EFBMBlock(dim=dim, sensor_resolution=sensor_resolution, num_heads=num_heads,
                     mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                     norm_layer=norm_layer)
            for i in range(depth)
        ])

        self.downsample = downsample(sensor_resolution, dim=dim) if downsample else None

    def forward(self, x):
        for blk in self.blocks:
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)
        if self.downsample:
            x = self.downsample(x)
        return x

class TemporalDownsampling(nn.Module):
    def __init__(self, sensor_resolution, dim, ratio=2.0):
        super().__init__()
        self.sensor_resolution = sensor_resolution
        self.dim = dim
        in_channels = dim
        out_channels = 2 * dim
        
        self.conv = nn.Sequential(
            ConvLayer(in_channels, int(out_channels * ratio), kernel_size=1, norm=None),
            ConvLayer(int(out_channels * ratio), int(out_channels * ratio),
                      kernel_size=(3,1), stride=(2,1), padding=(1,0),
                      groups=int(out_channels * ratio), norm=None),
            ConvLayer(int(out_channels * ratio), out_channels, kernel_size=1, act_func=None)
        )

    def forward(self, x):
        sequence_length, sensor_count = self.sensor_resolution
        B, L, C = x.shape
        assert L == sequence_length * sensor_count, "Incorrect input feature size"
        x = self.conv(x.reshape(B, sequence_length, sensor_count, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        return x

class Stem(nn.Module):
    def __init__(self, sensor_size=(128, 9), input_channels=3, embed_dim=96):
        super().__init__()
        sensor_size = to_2tuple(sensor_size)
        
        sequence_length, sensor_count = sensor_size
        
        sequence_length = (sequence_length + 2*1 - 3) // 2 + 1
        
        
        sequence_length = (sequence_length + 2*1 - 3) // 2 + 1
        
        self.sensor_resolution = (sequence_length, sensor_count)
        self.num_tokens = sequence_length * sensor_count

        self.conv1 = ConvLayer(input_channels, embed_dim // 2,
                               kernel_size=(3,1), stride=(2,1), padding=(1,0), bias=False)
        self.conv2 = nn.Sequential(
            ConvLayer(embed_dim // 2, embed_dim // 2,
                      kernel_size=(3,1), stride=1, padding=(1,0), bias=False),
            ConvLayer(embed_dim // 2, embed_dim // 2,
                      kernel_size=(3,1), stride=1, padding=(1,0), bias=False, act_func=None)
        )
        self.conv3 = nn.Sequential(
            ConvLayer(embed_dim // 2, embed_dim * 4,
                      kernel_size=(3,1), stride=(2,1), padding=(1,0), bias=False),
            ConvLayer(embed_dim * 4, embed_dim, kernel_size=1, bias=False, act_func=None)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x) + x
        x = self.conv3(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class EFBMNet(nn.Module):
    def __init__(self, sensor_size=(128, 9), input_channels=1, num_classes=1000,
                 embed_dim=16, depths=[1, 1, 1, 1], num_heads=[1, 2, 4, 8],
                 mlp_ratio=2., qkv_bias=True, drop_rate=0., drop_path_rate=0.0,
                 norm_layer=nn.LayerNorm, ape=False, use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        
        self.sensor_embed = Stem(sensor_size, input_channels, embed_dim)
        sensor_resolution = self.sensor_embed.sensor_resolution
        self.sensor_resolution = sensor_resolution

        
        final_sequence_length, final_sensor_count = sensor_resolution
        for _ in range(self.num_layers - 1):
            final_sequence_length = (final_sequence_length + 1) // 2
        self.final_sensor_resolution = (final_sequence_length, final_sensor_count)

        
        if self.ape:
            self.temporal_pos_embed = nn.Parameter(torch.zeros(1, sensor_resolution[0], embed_dim))
            trunc_normal_(self.temporal_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        
        self.layers = nn.ModuleList()
        current_sequence_length, current_sensor_count = sensor_resolution
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                sensor_resolution=(current_sequence_length, current_sensor_count),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                mlp_ratio=self.mlp_ratio,
                num_classes=num_classes,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer+1])],
                norm_layer=norm_layer,
                downsample=TemporalDownsampling if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint
            )
            self.layers.append(layer)
            if i_layer < self.num_layers - 1:
                current_sequence_length = (current_sequence_length + 1) // 2
                

        self.norm = norm_layer(self.num_features)

        
        self.classifier = nn.Sequential(OrderedDict([
            ("avgpool", nn.AdaptiveAvgPool2d((1, final_sensor_count))),
            ("flatten", nn.Flatten(1)),
            ("head", nn.Linear(self.num_features * final_sensor_count, num_classes) if num_classes > 0 else nn.Identity()),
        ]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.sensor_embed(x)  
        B, L, C = x.shape
        sequence_length, sensor_count = self.sensor_resolution
        if self.ape:
            x = x.view(B, sequence_length, sensor_count, C)
            temporal_position = self.temporal_pos_embed.unsqueeze(2)
            x = x + temporal_position
            x = x.view(B, L, C)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)                     
        B, L, C = x.shape
        sequence_length, sensor_count = self.final_sensor_resolution
        assert L == sequence_length * sensor_count, (
            f"Shape mismatch: tokens={L}, sequence_length={sequence_length}, "
            f"sensor_count={sensor_count}"
        )
        x = x.transpose(1, 2).contiguous().view(B, C, sequence_length, sensor_count)
        x = self.classifier(x)
        return x