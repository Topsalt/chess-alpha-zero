import torch
import torch.nn as nn
import numpy as np
import pdb
from mmdet.registry import MODELS, TASK_UTILS

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from .cnsn import CrossNorm, SelfNorm, CNSN

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicBlockCustom(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, pos, beta, crop, cnsn_type, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlockCustom, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        assert cnsn_type in ['sn', 'cn', 'cnsn']

        if 'cn' in cnsn_type:
            crossnorm = CrossNorm(beta=beta, crop=crop)
        else:
            crossnorm = None

        if 'sn' in cnsn_type:
            print('using SelfNorm module')
            if pos == 'pre' and not self.is_in_equal_out:
                selfnorm = SelfNorm(inplanes)
            else:
                selfnorm = SelfNorm(planes)
        else:
            selfnorm = None

        self.cnsn = CNSN(selfnorm=selfnorm, crossnorm=crossnorm)

        self.pos = pos
        if pos is not None:
            print('{} in residual module: {}'.format(cnsn_type, pos))
            assert pos in ['residual', 'identity', 'pre', 'post']

    def forward(self, x):
        identity = x

        if self.pos == 'pre':
            out = self.cnsn(x)
        else:
            out = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.pos == 'residual':
            out = self.cnsn(out)
        elif self.pos == 'identity':
            identity = self.cnsn(identity)

        out += identity
        out = self.relu(out)

        if self.pos == 'post':
            return self.cnsn(out)
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleneckCustom(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, pos, cn_pos, beta, crop, cnsn_type, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, custom=False):
        super(BottleneckCustom, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.custom = custom

        if self.custom:
            assert cnsn_type in ['sn', 'cn', 'cnsn']

            if 'cn' in cnsn_type and cn_pos is None:
                crossnorm = CrossNorm(crop=crop, beta=beta)
            else:
                crossnorm = None

            if 'sn' in cnsn_type:
                print('using SelfNorm module')
                if pos == 'pre' and self.downsample is None:
                    selfnorm = SelfNorm(inplanes)
                else:
                    selfnorm = SelfNorm(planes * self.expansion)
            else:
                selfnorm = None

            self.cnsn = CNSN(selfnorm=selfnorm, crossnorm=crossnorm)
            if 'cn' in cnsn_type and cn_pos is not None:
                self.real_cn = CrossNorm(beta=beta, crop=crop)
            self.cn_pos = cn_pos

            self.pos = pos
            if pos is not None:
                print('{} in residual module: {}'.format(cnsn_type, pos))
                assert pos in ['residual', 'identity', 'pre', 'post']

    def forward(self, x):
        identity = x

        out = x
        if self.custom:
            if self.pos == 'pre':
                out = self.cnsn(out)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.custom:
            if self.pos == 'residual':
                out = self.cnsn(out)
            elif self.pos == 'identity':
                identity = self.cnsn(out)

        out += identity
        out = self.relu(out)

        if self.custom:
            if self.pos == 'post':
                out = self.cnsn(out)
            if self.cn_pos == 'post':
                out = self.real_cn(out)
        return out

@MODELS.register_module()
class ResNet_CNSN(nn.Module):

    def __init__(self, block=BottleneckCustom, layers=[3,4,6,3], init_cfg=None, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, block_idxs='1_2_3_4', active_num=None, pos=None, beta=None, crop=None, cnsn_type=None,
                 cn_pos=None):
        super(ResNet_CNSN, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        if block_idxs:
            block_idxs = block_idxs.split('_')
            block_idxs = list(map(int, block_idxs))
        print('block_idxs: {}'.format(block_idxs))
        if beta is not None:
            print('beta: {}'.format(beta))

        if crop is not None:
            print('crop in 2 instance mode: {}'.format(crop))

        self.block_idxs = block_idxs
        if block_idxs and 0 in block_idxs:
            self.img_cn = CrossNorm(crop=crop, beta=beta)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if block_idxs and 1 in block_idxs:
            self.layer1 = self._make_layer(block, 64, layers[0], pos=pos, beta=beta, crop=crop, cnsn_type=cnsn_type,
                                           cn_pos=cn_pos, custom=True)
        else:
            self.layer1 = self._make_layer(block, 64, layers[0], custom=False)

        if block_idxs and 2 in block_idxs:
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0],
                                           pos=pos, beta=beta, crop=crop, cnsn_type=cnsn_type, cn_pos=cn_pos,
                                           custom=True)
        else:
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                           dilate=replace_stride_with_dilation[0], custom=False)

        if block_idxs and 3 in block_idxs:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1],
                                           pos=pos, beta=beta, crop=crop, cnsn_type=cnsn_type, cn_pos=cn_pos,
                                           custom=True)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           dilate=replace_stride_with_dilation[1], custom=False)

        if block_idxs and 4 in block_idxs:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2],
                                           pos=pos, beta=beta, crop=crop, cnsn_type=cnsn_type, cn_pos=cn_pos,
                                           custom=True)
        else:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                           dilate=replace_stride_with_dilation[2], custom=False)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.cn_modules = []

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()
            elif isinstance(m, CrossNorm):
                self.cn_modules.append(m)

        if 'cn' in cnsn_type or cn_pos is not None:
            self.cn_num = len(self.cn_modules)
            assert self.cn_num > 0
            print('cn_num: {}'.format(self.cn_num))
            self.active_num = active_num
            assert self.active_num > 0
            print('active_num: {}'.format(self.active_num))

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def load_pretrained_weights(self, init_cfg):
        checkpoint_path = init_cfg.get('checkpoint', None)
        if checkpoint_path:
            # 设置 weights_only=False 以加载所有内容
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            modified_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.load_state_dict(modified_state_dict, strict=False)

    def _make_layer(self, block, planes, blocks, pos=None, beta=None, crop=None, cnsn_type=None, cn_pos=None, stride=1,
                    dilate=False, custom=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, pos=pos, beta=beta, crop=crop, cnsn_type=cnsn_type, cn_pos=cn_pos,
                            stride=stride, downsample=downsample,
                            groups=self.groups, base_width=self.base_width, dilation=previous_dilation,
                            norm_layer=norm_layer, custom=custom))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, pos=pos, cn_pos=cn_pos, beta=beta, crop=crop, cnsn_type=cnsn_type,
                      groups=self.groups,
                      base_width=self.base_width, dilation=self.dilation,
                      norm_layer=norm_layer, custom=custom))

        return nn.Sequential(*layers)

    def _forward_impl(self, x, aug=False):
        if self.block_idxs and 0 in self.block_idxs:
            x = self.img_cn(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        outs = []
        x = self.layer1(x)
        outs.append(x)
        x = self.layer2(x)
        outs.append(x)
        x = self.layer3(x)
        outs.append(x)
        aux = x
        x = self.layer4(x)
        outs.append(x)

        return tuple(outs)

    def forward(self, x, aug=False):
        return self._forward_impl(x, aug=aug)


def _resnet(arch, block, layers, pretrained, progress, SN=False, **kwargs):
    model = ResNet_CNSN(block, layers, **kwargs)
    if pretrained:
        if SN:
            state_dict = torch.load('/PATH/TO/PRETRAINED/MODEL.pth')
            print('model loaded from initmodel')
        else:
            state_dict = load_state_dict_from_url(model_urls[arch],
                                                  progress=progress)
        model.load_state_dict(state_dict, strict=True)
    return model


def resnet50(pretrained=False, progress=True, SN=False, **kwargs):
    return _resnet('resnet50', BottleneckCustom, [3, 4, 6, 3], pretrained, progress, SN=SN,
                   **kwargs)