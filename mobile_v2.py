import torch
import torch.nn as nn

def make_divisible(v, divisor=8, min_value=None):
    
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNReLU(nn.Sequential):
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, use_relu=True):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False))
        self.add_module('bn', nn.BatchNorm2d(out_channels))
        if use_relu:
            self.add_module('relu', nn.ReLU6(inplace=True))

class DepthwiseStripConv(nn.Module):
    
    def __init__(self, in_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = (kernel_size // 2, 0)  
        self.dw = nn.Conv2d(in_channels, in_channels, kernel_size=(kernel_size, 1),
                            stride=(stride, 1), padding=padding, groups=in_channels, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.dw(x)))

class InvertedResidualBlock(nn.Module):
    



    def __init__(self, in_channels, out_channels, stride, expand_ratio, kernel_size=3):
        super().__init__()
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_residual = (stride == 1) and (in_channels == out_channels)

        layers = []
        
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, use_relu=True))
        
        layers.append(DepthwiseStripConv(hidden_dim, kernel_size=kernel_size, stride=stride))
        
        layers.append(ConvBNReLU(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, use_relu=False))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileStripNet(nn.Module):
 
    def __init__(self, num_classes=6, input_channels=1, stem_channels=16,
                 stage_out_channels=[16, 32, 64, 128],
                 stage_depths=[4, 4, 6, 4],
                 stage_kernel_sizes=[3, 3, 5, 5],
                 expand_ratio=2.5, width_mult=1.0):
        super().__init__()

        
        stem_channels = make_divisible(stem_channels * width_mult)
        stage_out_channels = [make_divisible(c * width_mult) for c in stage_out_channels]

        
        self.stem = ConvBNReLU(input_channels, stem_channels, kernel_size=(3,1), stride=(2,1), padding=(1,0), use_relu=True)

        
        self.stages = nn.ModuleList()
        in_ch = stem_channels
        for i, out_ch in enumerate(stage_out_channels):
            depth = stage_depths[i]
            kernel_size = stage_kernel_sizes[i]
            for j in range(depth):
                stride = 2 if j == 0 else 1   
                block = InvertedResidualBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    stride=stride,
                    expand_ratio=expand_ratio,
                    kernel_size=kernel_size
                )
                self.stages.append(block)
                in_ch = out_ch   

        
        self.feature_dim = in_ch

        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(self.feature_dim*1, num_classes)
        self.flatten = nn.Flatten(1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)                     
        for stage in self.stages:
            x = stage(x)
        x = self.avgpool(x)                  
        x = self.flatten(x)             
        x = self.classifier(x)               
        return x

