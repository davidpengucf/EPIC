import torch.nn as nn
import copy
from resnet import _resnet, Bottleneck
from einops.layers.torch import Rearrange


class Upsampling(nn.Sequential):
   
    def __init__(self, in_channel=2048, hidden_dims=(256, 256, 256), kernel_sizes=(4, 4, 4), bias=False):
        assert len(hidden_dims) == len(kernel_sizes), \
            'ERROR: len(hidden_dims) is different len(kernel_sizes)'

        layers = []
        for hidden_dim, kernel_size in zip(hidden_dims, kernel_sizes):
            if kernel_size == 4:
                padding = 1
                output_padding = 0
            elif kernel_size == 3:
                padding = 1
                output_padding = 1
            elif kernel_size == 2:
                padding = 0
                output_padding = 0
            else:
                raise NotImplementedError("kernel_size is {}".format(kernel_size))

            layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_channel,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=bias))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_channel = hidden_dim

        super(Upsampling, self).__init__(*layers)

        # init following Simple Baseline
        for name, m in self.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                if bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class PoseResNet(nn.Module):
    
    def __init__(self, backbone, upsampling, feature_dim, num_keypoints, finetune=False):
        super(PoseResNet, self).__init__()
        self.backbone = backbone
        self.upsampling = upsampling
        self.head = nn.Conv2d(in_channels=feature_dim, out_channels=num_keypoints, kernel_size=1, stride=1, padding=0)
        self.finetune = finetune
        for m in self.head.modules():
            nn.init.normal_(m.weight, std=0.001)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.backbone(x)
        #print(x.shape)
        x = self.upsampling(x)
        x = self.head(x)
        #print(x.shape)

        return x

    def get_parameters(self, lr=1.):
        return [
            {'params': self.backbone.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.upsampling.parameters(), 'lr': lr},
            {'params': self.head.parameters(), 'lr': lr},
        ]
        
class PoseResNet_GVB(nn.Module):
    
    def __init__(self, backbone, upsampling1, upsampling2, feature_dim, num_keypoints, finetune=False):
        super(PoseResNet_GVB, self).__init__()
        self.backbone = backbone
        self.upsampling1 = upsampling1
        self.head1 = nn.Conv2d(in_channels=feature_dim, out_channels=num_keypoints, kernel_size=1, stride=1, padding=0)
        self.upsampling2 = upsampling2
        self.head2 = nn.Conv2d(in_channels=feature_dim, out_channels=num_keypoints, kernel_size=1, stride=1, padding=0)
        self.finetune = finetune
        for m in self.head1.modules():
            nn.init.normal_(m.weight, std=0.001)
            nn.init.constant_(m.bias, 0)
        for m in self.head2.modules():
            nn.init.normal_(m.weight, std=0.001)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.backbone(x)
        #print(x.shape)
        x = self.upsampling1(x)
        x = self.head1(x)
        #print(x.shape)

        return x

    def get_parameters(self, lr=1.):
        return [
            {'params': self.backbone.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.upsampling1.parameters(), 'lr': lr},
            {'params': self.head1.parameters(), 'lr': lr},
        ]
    
class PoseResNetRLE(nn.Module):
    
    def __init__(self, backbone, upsampling, feature_dim, num_keypoints, finetune=False):
        super(PoseResNetRLE, self).__init__()
        self.backbone = backbone
        self.upsampling = upsampling
        self.head = nn.Conv2d(in_channels=feature_dim, out_channels=num_keypoints, kernel_size=1, stride=1, padding=0)
        self.finetune = finetune
        self.regr = nn.Sequential(
            # break-down the image in s1 x s2 patches and flat them
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(num_keypoints * 64 * 64, num_keypoints * 2),
            Rearrange('b (c d) -> b c d',d=2),    
        )
        for m in self.head.modules():
            nn.init.normal_(m.weight, std=0.001)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.backbone(x)
        #print(x.shape)
        x = self.upsampling(x)
        y = self.head(x)
        y_sig = self.regr(y)
        #print(x.shape)
        
        if self.training:
            return y, y_sig
        else: 
            return y

    def get_parameters(self, lr=1.):
        return [
            {'params': self.backbone.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.upsampling.parameters(), 'lr': lr},
            {'params': self.head.parameters(), 'lr': lr},
            {'params': self.regr.parameters(), 'lr': lr},
        ]


def _pose_resnet(arch, num_keypoints, block, layers, pretrained_backbone, deconv_with_bias, finetune=False, progress=True, **kwargs):
    backbone = _resnet(arch, block, layers, pretrained_backbone, progress, **kwargs)
    upsampling = Upsampling(backbone.out_features, bias=deconv_with_bias)
    model = PoseResNet(backbone, upsampling, 256, num_keypoints, finetune)
    return model


def pose_resnet101(num_keypoints, pretrained_backbone=True, deconv_with_bias=False, finetune=False, progress=True, **kwargs):
    
    return _pose_resnet('resnet101', num_keypoints, Bottleneck, [3, 4, 23, 3], pretrained_backbone, deconv_with_bias, finetune, progress, **kwargs)
