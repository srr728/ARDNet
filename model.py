import os
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import utils
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.module import build_decoder, build_aspp, build_backbone

import pdb
from e2cnn import gspaces
from e2cnn import nn as enn
    
class SymmetryDetectionNetwork(nn.Module):
    def __init__(self, args=None):
        super(SymmetryDetectionNetwork, self).__init__()
        self.args = args
        self.sync_bn = args.sync_bn
        self.backbone_model = args.backbone
        self.output_stride = 8
        self.freeze_bn = False
        self.num_classes = 1
        self.n_angle = args.n_angle
        self.angle_interval = 360 / self.n_angle
        self.eq_cnn = args.eq_cnn
        self.get_theta = args.get_theta

        if self.sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        res_featdim = {18: 512, 34: 512, 50:2048, 101: 2048}
        featdim = 256
        out_indices = (3,)
        last_convout = 256
        last_conv_only = True
        score_dim = 0
        last_convin = featdim + score_dim
        
        if self.eq_cnn:
            global gspace_orientation, gspace_flip, cutoff_dilation
            gspace_orientation = 8 #self.n_angle
            gspace_flip = 1
            cutoff_dilation = 0
            from modeling.eq_resnet import ReResNet

            self.backbone = ReResNet(depth=self.args.depth, strides=[1, 2, 1, 1], dilations=[1, 1, 2, 4], out_indices=out_indices)

            if args.load_eq_pretrained:
                print('loading pretrained e2cnn backbone')
                # hardcode for now
                eq_ckpt = torch.load(args.eq_model_dir)['state_dict']
                eq_ckpt = {k:v for k, v in eq_ckpt.items() if 'head' not in k}
                temp = self.backbone.state_dict()
                for k, v in temp.items():
                    # init -> eval mode
                    if 'filter' in k:
                        eq_ckpt[k] = v
                self.backbone.load_state_dict(eq_ckpt)
                print('done')
                
        else:
            self.backbone_model = 'resnet%d' % args.depth
            self.backbone = build_backbone(self.backbone_model, self.output_stride, BatchNorm, False)

        if self.get_theta in [10]:
            theta_convin = featdim + score_dim

            if args.rot_data:
                n_angle = args.n_rot
            else:
                n_angle = self.n_angle

            if self.eq_cnn:
                from modeling.eq_module import Decoder
                self.decoder_theta = Decoder(n_angle + 1, self.backbone_model, BatchNorm, last_conv_only, theta_convin, last_convout, use_bn=True, restrict=False)
            else:
                self.decoder_theta = build_decoder(n_angle + 1, self.backbone_model, BatchNorm, last_conv_only, theta_convin, last_convout, use_bn=True)

        if self.eq_cnn:
            from modeling.eq_module import ASPP
            self.aspp = ASPP(self.output_stride, in_channels=res_featdim[args.depth])
        else:
            self.aspp = build_aspp(self.backbone_model, self.output_stride, BatchNorm, inchannels=res_featdim[args.depth])

        if self.eq_cnn:
            from modeling.eq_module import Decoder
            self.sync_bn = args.sync_bn
            if self.sync_bn == True:
                BatchNorm = SynchronizedBatchNorm2d
            else:
                BatchNorm = nn.BatchNorm2d
            last_conv_only = True
            last_convin = 256*2
            last_convout = 256
            self.num_classes = 1
            self.backbone_model = args.backbone
            from modeling.eq_module import Decoder, MaskDecoder
            self.decoder_axis = Decoder(self.num_classes, self.backbone_model, BatchNorm, last_conv_only,
                                                last_convin, last_convout, use_bn=True)
            self.decoder_final = MaskDecoder(self.num_classes, self.backbone_model, BatchNorm, last_conv_only,
                                                last_convin, last_convout, use_bn=True)
            '''
            if self.get_theta in [10]:
                from modeling.eq_module import SingleDecoder
                from modeling.eq_resnet import trivial_feature_type
                in_type = self.decoder_theta.conv3.out_type
                in_type = in_type + trivial_feature_type(in_type.gspace, 1, fixparams=False)
                self.decoder_axis = SingleDecoder(in_type, self.num_classes)
            else:
                from modeling.eq_module import Decoder
                self.decoder_axis = Decoder(self.num_classes, self.backbone_model, BatchNorm, last_conv_only, last_convin, last_convout, use_bn=True)
                '''
        else:
            self.decoder_axis = build_decoder(self.num_classes, self.backbone_model, BatchNorm, last_conv_only, last_convin, last_convout, use_bn=True)
        self.sigmoid = nn.Sigmoid()
        
        
        from modeling.eq_module import UnetDecoder
        self.sync_bn = args.sync_bn
        if self.sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d
        self.backbone_model = args.backbone
        self.unet_decoder = UnetDecoder(self.backbone_model, BatchNorm, use_bn=True)
        

    def forward(self, img, lbl, mask, is_syn, a_lbl=None, sym_type='reflection', vis_only=False, angle=None):
        feat,f1 = self.backbone(img[0])
        feat = self.aspp(feat)
        
        f1 = self.unet_decoder(f1)
        
        feat2,f2 = self.backbone(img[1])
        feat2 = self.aspp(feat2)
        
        f2 = self.unet_decoder(f2)

        if self.get_theta in [10]:
            theta_out = self.decoder_theta(None, feat)
            theta_out2 = self.decoder_theta(None, feat2)
            if not torch.is_tensor(theta_out):
                _theta_out = theta_out.tensor
                theta_sum = torch.softmax(_theta_out, dim=1)[:, 1:, :, :].sum(dim=1, keepdim=True)
                _theta_out2 = theta_out2.tensor
                theta_sum2 = torch.softmax(_theta_out2, dim=1)[:, 1:, :, :].sum(dim=1, keepdim=True)
            else:
                theta_sum = torch.softmax(theta_out, dim=1)[:, 1:, :, :].sum(dim=1, keepdim=True)
                theta_sum2 = torch.softmax(theta_out2, dim=1)[:, 1:, :, :].sum(dim=1, keepdim=True)
            theta_sum = torch.clamp(theta_sum, min=0+1e-5, max=1-1e-5)
            theta_sum2 = torch.clamp(theta_sum2, min=0+1e-5, max=1-1e-5)
            if not torch.is_tensor(f1):
                f1 = f1.tensor
            if not torch.is_tensor(f2):
                f2 = f2.tensor
            f1 = F.interpolate(f1, size=(theta_out.shape[-2], theta_out.shape[-1]), mode='bilinear', align_corners=True)
            f2 = F.interpolate(f2, size=(theta_out2.shape[-2], theta_out2.shape[-1]), mode='bilinear', align_corners=True)
            out = self.decoder_axis(f1, feat)
            out_ = self.decoder_final(f1, feat)

            out2 = self.decoder_axis(f2, feat2)
            if not torch.is_tensor(theta_out):
                theta_out = theta_out.tensor
                out = out.tensor
                out_ = out_.tensor
            if not torch.is_tensor(theta_out2):
                theta_out2 = theta_out2.tensor
                out2 = out2.tensor
        else:
            if self.eq_cnn:
                out = self.decoder_axis(None, feat)
                if not torch.is_tensor(out):
                    out = out.tensor
            else:
                out = self.decoder_axis(None, feat)

        axis_out = F.interpolate(out, size=lbl[0].size()[2:], mode='bilinear', align_corners=True)
        axis_out_ = F.interpolate(out_, size=lbl[0].size()[2:], mode='bilinear', align_corners=True)
        if vis_only:
            return self.sigmoid(axis_out_) # vis purpose
        axis_loss = utils.sigmoid_focal_loss(axis_out, lbl[0], alpha=0.95)
        axis_out = self.sigmoid(axis_out_) # vis purpose
        loss = axis_loss
        loss_ = utils.sigmoid_focal_loss(axis_out_, lbl[0], alpha=0.95)
        losses = (axis_loss, axis_loss)
        
        axis_out2 = F.interpolate(out2, size=lbl[1].size()[2:], mode='bilinear', align_corners=True)
        loss1 = utils.sigmoid_focal_loss(axis_out2, lbl[1], alpha=0.95)
        axis_out2 = self.sigmoid(axis_out2)
        loss2 = utils.sigmoid_focal_loss_2(axis_out, axis_out2, alpha=0.95, angle=angle)
        loss += (loss1 + loss2 + loss_)

        if self.get_theta:
            a_lbl = F.max_pool2d(a_lbl, kernel_size=5, stride=1, padding=2)
            theta_out = F.interpolate(theta_out, size=a_lbl.size()[2:], mode='bilinear', align_corners=True)

            weight = torch.ones(theta_out.shape[1])
            weight[0] = self.args.theta_loss_weight
            weight = weight.view(1, -1, 1, 1).to(a_lbl.device)

            fg_mask = (a_lbl.sum(dim=1) > 0).long()
            a_lbl = fg_mask * (torch.argmax(a_lbl, dim=1)+1)
            theta_loss = F.cross_entropy(theta_out, a_lbl, weight=weight, reduction='none') / weight.sum(dim=1, keepdim=True)
            theta_loss = theta_loss.mean(dim=1, keepdim=True)

            losses = (axis_loss, theta_loss)

            return axis_out, theta_out, loss, losses

        return axis_out, feat, loss, losses

    def export(self):
        ### For fast inference (use this only at the test time)
        ### For more information about 'export' ,
        ### refer to https://quva-lab.github.io/e2cnn/api/e2cnn.nn.html?highlight=export#e2cnn.nn.EquivariantModule
        self.backbone = self.backbone.export()
        self.decoder_theta = self.decoder_theta.export()
        self.decoder_axis = self.decoder_axis.export()
        self.aspp = self.aspp.export()