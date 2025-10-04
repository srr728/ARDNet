import e2cnn.nn as enn
import math
import os
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from e2cnn import gspaces
from mmcv.cnn import (constant_init, kaiming_init)
from torch.nn.modules.batchnorm import _BatchNorm

# cp from mmcls
import logging
from abc import ABCMeta, abstractmethod

import torch.nn as nn
from mmcv.runner import load_checkpoint

from modeling.eq_resnet import *
# from modify_for_export.eq_resnet import *
import torch.nn.functional as F

class _ASPPModule(enn.EquivariantModule):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.in_type = FIELD_TYPE['regular'](gspace, inplanes)
        self.out_type = FIELD_TYPE['regular'](gspace, planes)
        self.atrous_conv = convnxn(inplanes, planes, kernel_size, padding=padding, dilation=dilation, is_backbone=False)
        self.bn = enn.InnerBatchNorm(self.out_type)
        self.relu = ennReLU(planes)

    def forward(self, x):

        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def evaluate_output_shape(self, input_shape):
        assert len(input_shape) == 4
        # assert input_shape[1] == self.in_type.size
        if self.downsample is not None:
            return self.downsample.evaluate_output_shape(input_shape)
        else:
            return input_shape

    def export(self):
        for name, module in self._modules.items():
            if isinstance(module, enn.EquivariantModule):
                # print(name, "--->", module)
                module = module.export()
                setattr(self, name, module)
        return self

class _GAPModule(enn.EquivariantModule):
    def __init__(self, inplanes, planes):
        super(_GAPModule, self).__init__()
        self.in_type = FIELD_TYPE['regular'](gspace, inplanes)
        self.out_type = FIELD_TYPE['regular'](gspace, planes)
        self.global_avg_pool = enn.PointwiseAdaptiveAvgPool(self.in_type, (1, 1))
        self.conv = convnxn(inplanes, planes, kernel_size=1, is_backbone=False)
        self.bn = enn.InnerBatchNorm(self.out_type)
        self.relu = ennReLU(planes)

    def forward(self, x):

        x = self.global_avg_pool(x)
        x = self.conv(x)
        x = self.bn(x)

        return self.relu(x)

    def evaluate_output_shape(self, input_shape):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        if self.downsample is not None:
            return self.downsample.evaluate_output_shape(input_shape)
        else:
            return input_shape

    def export(self):
        for name, module in self._modules.items():
            if isinstance(module, enn.EquivariantModule):
                # print(name, "--->", module)
                module = module.export()
                setattr(self, name, module)
        return self

class ASPP(nn.Module):
    def __init__(self, output_stride, in_channels):
        super(ASPP, self).__init__()
        inplanes = in_channels
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = _GAPModule(inplanes, 256)

        out_type = FIELD_TYPE['regular'](gspace, 256)
        self.conv1 = convnxn(256 * 5, 256, kernel_size=1, is_backbone=False)
        self.bn1 = enn.InnerBatchNorm(out_type)
        self.relu = ennReLU(256)
        self.dropout = enn.PointwiseDropout(out_type, 0.5)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        scale_factor = (x4.shape[-2] / x5.shape[-2], x4.shape[-1] / x5.shape[-1])
        if torch.is_tensor(x5):
            upsample =  torch.nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        else: 
            upsample = enn.R2Upsampling(x5.type, scale_factor, mode='bilinear', align_corners=True)
        x5 = upsample(x5)
        if torch.is_tensor(x5):
            x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        else:
            x = enn.tensor_directsum((x1, x2, x3, x4, x5))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x #self.dropout(x)

    def export(self):
        for name, module in self._modules.items():
            if isinstance(module, enn.EquivariantModule):
                # print(name, "--->", module)
                module = module.export()
                setattr(self, name, module)
        return self

class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm, last_conv_only, last_convin = None, last_convout = None, use_bn=True, interpolate=0,
                    restrict=False):
        super(Decoder, self).__init__()
        if backbone in ['resnet', 'resnet101', 'resnet50', 'resnet18']:
            low_level_inplanes = 256
        else:
            raise NotImplementedError

        self.last_conv_only = last_conv_only # always True
        self.interpolate = interpolate

        self.in_type = FIELD_TYPE['regular'](gspace, last_convin)
        self.conv1 = convnxn(last_convin, last_convout, kernel_size=3, stride=1, padding=1, is_backbone=False)
        self.out_type = FIELD_TYPE['regular'](gspace, last_convout)
        self.bn1 = enn.InnerBatchNorm(self.out_type)
        self.relu1 = ennReLU(last_convout)
        self.conv2 = convnxn(last_convout, last_convout, kernel_size=3, stride=1, padding=1, is_backbone=False)
        self.bn2 = enn.InnerBatchNorm(self.out_type)
        self.relu2 = ennReLU(last_convout)
        if restrict:
            self.restrict2 = self._restrict_layer()
            gs = self._in_type.gspace # C8
            out_type = enn.FieldType(gs, [gs.trivial_repr] + [gs.regular_repr]) # (bg, fg*8)
            self.conv3 = enn.SequentialModule(
                self.restrict2,
                enn.R2Conv(self._in_type, out_type, kernel_size=1, stride=1, bias=False, sigma=None, frequencies_cutoff=lambda r: 3 * r, initialize=False)
            )
        else:
            self.conv3 = convnxn(last_convout, num_classes, kernel_size=1, is_backbone=False, to_trivial=True)

    def _restrict_layer(self):
        # hardcode for now (D8 -> C8)
        subgroup_id = (None, 8)
        layers = list()
        layers.append(enn.RestrictionModule(self.out_type, subgroup_id))
        layers.append(enn.DisentangleModule(layers[-1].out_type))
        self._in_type = layers[-1].out_type
        self.gspace = self._in_type.gspace

        restrict_layer = enn.SequentialModule(*layers)
        return restrict_layer

    def forward(self, x, feat):
        if x is None:
            out = self.conv1(feat)
            out = self.bn1(out)
            out = self.relu1(out)
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu2(out)
            out = self.conv3(out)
            return out

        else:
            if not torch.is_tensor(x):
                x = x.tensor
            if not torch.is_tensor(feat):
                feat = feat.tensor
            out = torch.cat([x, feat], dim=1)
            out = enn.GeometricTensor(out, self.in_type)
            out = self.conv1(out)
            out = self.bn1(out)
            out = self.relu1(out)
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu2(out)
            out = self.conv3(out)
            return out

    def export(self):
        for name, module in self._modules.items():
            if isinstance(module, enn.EquivariantModule):
                # print(name, "--->", module)
                module = module.export()
                setattr(self, name, module)
        return self

class SingleDecoder(nn.Module):
    def __init__(self, in_type, out_channels):
        super(SingleDecoder, self).__init__()
        self.in_type = in_type
        gspace = in_type.gspace
        self.out_type = FIELD_TYPE['trivial'](gspace, out_channels, fixparams=False)
        self.conv1 = enn.R2Conv(self.in_type, self.out_type, kernel_size=1, \
        stride=1, bias=False, sigma=None, frequencies_cutoff=lambda r: 3 * r, initialize=False)

    def forward(self, x, feat):
        if x is None:
            out = self.conv1(feat)
            return out

        else:
            x_type = FIELD_TYPE['trivial'](gspace, 1, fixparams=False)
            if isinstance(self.conv1, enn.R2Conv):
                x = enn.GeometricTensor(x, x_type)
                out = enn.tensor_directsum([feat, x])
            else:
                out = torch.cat((feat, x), dim=1)
            out = self.conv1(out)
            return out

    def export(self):
        for name, module in self._modules.items():
            if isinstance(module, enn.EquivariantModule):
                # print(name, "--->", module)
                module = module.export()
                setattr(self, name, module)
        return self


class UnetDecoder(nn.Module):
    def __init__(self, backbone, BatchNorm, use_bn=True,):
        super(UnetDecoder, self).__init__()
        if backbone in ['resnet', 'resnet101', 'resnet50', 'resnet18']:
            low_level_inplanes = 256
        else:
            raise NotImplementedError


        # self.last_convin = last_convin
        # self.in_type = FIELD_TYPE['regular'](gspace, last_convin) # 我改过的 把regular改成trivial
        # self.conv1 = convnxn(last_convin, last_convout, kernel_size=3, stride=1, padding=1, is_backbone=False)
        # self.out_type = FIELD_TYPE['regular'](gspace, last_convout)
        # self.bn1 = enn.InnerBatchNorm(self.out_type)
        # self.relu1 = ennReLU(last_convout)
        # self.conv2 = convnxn(last_convout, last_convout, kernel_size=3, stride=1, padding=1, is_backbone=False)
        # self.bn2 = enn.InnerBatchNorm(self.out_type)
        # self.relu2 = ennReLU(last_convout)

        self.up3 = convnxn(2048+1024, 1024, kernel_size=3, stride=1, padding=1, is_backbone=False)
        self.up2 = convnxn(1024+512, 512, kernel_size=3, stride=1, padding=1, is_backbone=False)
        self.up1 = convnxn(512+256, 256, kernel_size=3, stride=1, padding=1, is_backbone=False)

        self.bn3 = enn.InnerBatchNorm(FIELD_TYPE['regular'](gspace, 1024))
        self.relu3 = ennReLU(1024)

        self.bn2 = enn.InnerBatchNorm(FIELD_TYPE['regular'](gspace, 512))
        self.relu2 = ennReLU(512)

        self.bn1 = enn.InnerBatchNorm(FIELD_TYPE['regular'](gspace, 256))
        self.relu1 = ennReLU(256)
        
        self.ins = [2048+1024, 1024+512, 512+256]


    def forward(self, feats):
        f3 = F.interpolate(feats[3].tensor, size=(feats[2].shape[-2], feats[2].shape[-1]), mode='bilinear', align_corners=True)
        out = torch.cat([f3, feats[2].tensor], dim=1)
        # print("1", f3.shape, feats[2].tensor.shape, out.shape)
        out = enn.GeometricTensor(out, FIELD_TYPE['regular'](gspace, self.ins[0]))
        out = self.up3(out)
        out = self.bn3(out)
        out = self.relu3(out)


        f2 = F.interpolate(feats[2].tensor, size=(feats[1].shape[-2], feats[1].shape[-1]), mode='bilinear', align_corners=True)
        out = torch.cat([f2, feats[1].tensor], dim=1)
        # print("2", f2.shape, feats[1].tensor.shape, out.shape)
        out = enn.GeometricTensor(out, FIELD_TYPE['regular'](gspace, self.ins[1]))
        out = self.up2(out)
        out = self.bn2(out)
        out = self.relu2(out)


        f1 = F.interpolate(feats[1].tensor, size=(feats[0].shape[-2], feats[0].shape[-1]), mode='bilinear', align_corners=True)
        out = torch.cat([f1, feats[0].tensor], dim=1)
        # print("3", f1.shape, feats[0].tensor.shape, out.shape)
        out = enn.GeometricTensor(out, FIELD_TYPE['regular'](gspace, self.ins[2]))
        out = self.up1(out)
        out = self.bn1(out)
        out = self.relu1(out)

        
        # print(out.shape)

        # out = self.bn1(out)
        # out = self.relu1(out)
        # out = self.conv2(out)
        # out = self.bn2(out)
        # out = self.relu2(out)
        
        return out
    
class MaskDecoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm, last_conv_only, last_convin = None, last_convout = None, use_bn=True, interpolate=0,
                    restrict=False):
        super(MaskDecoder, self).__init__()
        if backbone in ['resnet', 'resnet101', 'resnet50', 'resnet18']:
            low_level_inplanes = 256
        else:
            raise NotImplementedError

        self.last_conv_only = last_conv_only # always True
        self.interpolate = interpolate
        self.last_convout = last_convout

        self.in_type = FIELD_TYPE['regular'](gspace, last_convin)
        self.conv1 = convnxn(last_convin, last_convout, kernel_size=3, stride=1, padding=1, is_backbone=False)
        self.out_type = FIELD_TYPE['regular'](gspace, last_convout)
        self.bn1 = enn.InnerBatchNorm(self.out_type)
        self.relu1 = ennReLU(last_convout)
        self.conv2 = convnxn(last_convout, last_convout, kernel_size=3, stride=1, padding=1, is_backbone=False)
        self.bn2 = enn.InnerBatchNorm(self.out_type)
        self.relu2 = ennReLU(last_convout)
        if restrict:
            self.restrict2 = self._restrict_layer()
            gs = self._in_type.gspace # C8
            out_type = enn.FieldType(gs, [gs.trivial_repr] + [gs.regular_repr]) # (bg, fg*8)
            self.conv3 = enn.SequentialModule(
                self.restrict2,
                enn.R2Conv(self._in_type, out_type, kernel_size=1, stride=1, bias=False, sigma=None, frequencies_cutoff=lambda r: 3 * r, initialize=False)
            )
        else:
            self.conv3 = convnxn(last_convout, num_classes, kernel_size=1, is_backbone=False, to_trivial=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, feat):
        if not torch.is_tensor(x):
            x_ = x.tensor
        else:
            x_ = x
        if not torch.is_tensor(feat):
            feat_ = feat.tensor
        else:
            feat_ = feat
        out = torch.cat([x_, feat_], dim=1)
        out = enn.GeometricTensor(out, self.in_type)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = out.tensor * feat.tensor
        out = self.sigmoid(out)
        out = enn.GeometricTensor(out, FIELD_TYPE['regular'](gspace, self.last_convout))
        
        out = self.conv3(out)
        return out
