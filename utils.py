import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw


def draw_axis(lines, size):
    axis = Image.new('L', size)
    # w, h = img.size
    draw = ImageDraw.Draw(axis)
    length = np.array([size[0], size[1], size[0], size[1]])

    # x1, y1, x2, y2
    line_coords = []

    for idx, coords in enumerate(lines):
        if coords[0] > coords[2]:
            coords = np.roll(coords, -2)
        draw.line(list(coords), fill=(idx + 1))
        coords = np.array(coords).astype(np.float32)
        _line_coords = coords / length
        line_coords.append(_line_coords)
    axis = np.asarray(axis).astype(np.float32)
    return axis, line_coords


def match_input_type(img):
    img = np.asarray(img)
    if img.shape[-1] != 3:
        img = np.stack((img, img, img), axis=-1)
    return img


def norm(img):
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    img = (img - mean) / std
    return img


def unnorm(img):
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    img = img * std + mean
    return img


##########################
### Train ################
##########################

def sigmoid_focal_loss(
        source: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2,
        reduction: str = "none",
        is_logits=True
):
    """
    Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default = 0.25
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    if is_logits:
        p = nn.Sigmoid()(source)
        ce_loss = F.binary_cross_entropy_with_logits(
            source, targets, reduction="none"
        )
    else:
        p = source
        ce_loss = F.binary_cross_entropy(source, targets, reduction="none")

    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def sigmoid_focal_loss_2(
        source: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2,
        reduction: str = "none",
        is_logits=True,
        angle=None
):
    """
    Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default = 0.25
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    from albumentations import Compose, Rotate
    from albumentations.pytorch import ToTensorV2
    # ������ǿ����
    source_ = source.clone().detach().cpu().numpy()
    ss = source.clone().detach()
    h, w = source.size(2), source.size(3)
    for i in range(source.size(0)):
        augmentation = Compose([
            Rotate((float(angle[i]), float(angle[i])), p=1.0),
            ToTensorV2(),
        ])
        augmented = augmentation(image=source_[i].reshape(h, w, -1))
        ss[i] = augmented['image']

    if is_logits:
        p = nn.Sigmoid()(ss)
        ce_loss = F.binary_cross_entropy_with_logits(
            ss, targets, reduction="none"
        )
    else:
        p = ss
        ce_loss = F.binary_cross_entropy(ss, targets, reduction="none")

    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // (args.num_epochs * 0.5))) * (0.1 ** (epoch // (args.num_epochs * 0.75)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


##########################
### Evaluation ###########
##########################

import torch
import torch.nn.functional as F


class PointEvaluation(object):
    def __init__(self, n_thresh=100, max_dist=5, blur_pred=False, device=None):
        self.n_thresh = n_thresh
        self.max_dist = max_dist
        self.thresholds = torch.linspace(1.0 / (n_thresh + 1),
                                         1.0 - 1.0 / (n_thresh + 1), n_thresh)
        if device is not None:
            self.filters = self.make_gt_filter(max_dist).to(device)
        else:
            self.filters = self.make_gt_filter(max_dist)

        # 初始化统计量
        self.tp = torch.zeros((n_thresh,))  # 真正例
        self.fp = torch.zeros((n_thresh,))  # 假正例
        self.fn = torch.zeros((n_thresh,))  # 假反例
        self.tn = torch.zeros((n_thresh,))  # 真反例
        self.total_pixels = 0  # 总像素数
        self.num_samples = 0  # 样本数
        self.blur_pred = blur_pred

        self.tp2 = torch.zeros((n_thresh,))  # 真正例
        self.pos_label2 = torch.zeros((n_thresh,))
        self.pos_pred2 = torch.zeros((n_thresh,))

    def make_gt_filter(self, max_dist):
        ks = max_dist * 2 + 1
        filters = torch.zeros(1, 1, ks, ks)
        for i in range(ks):
            for j in range(ks):
                dist = (i - max_dist) ** 2 + (j - max_dist) ** 2
                if dist <= max_dist ** 2:
                    filters[0, 0, i, j] = 1
        return filters

    def f1_score(self):
        precision = torch.where(self.pos_pred2 > 0, self.tp2 / self.pos_pred2, torch.zeros(1))
        recall = self.tp2 / self.pos_label2
        numer = precision + recall
        f1 = torch.where(numer > 0, 2 * precision * recall / numer, torch.zeros(1))
        return precision, recall, f1

    def calculate_metrics(self):
        """计算所有评估指标"""
        # 确保分母不为零
        eps = 1e-7

        # 1. 计算基本指标
        self.pos_pred = self.tp + self.fp  # 预测正例总数
        self.pos_label = self.tp + self.fn  # 真实正例总数
        neg_pred = self.fn + self.tn  # 预测反例总数
        neg_label = self.fp + self.tn  # 真实反例总数

        # 2. 总体错误 (Overall Error)
        oe = (self.fp + self.fn) / (self.total_pixels + eps)

        # 3. 正确分类百分比 (Percentage of Correct Classification)
        pcc = (self.tp + self.tn) / (self.total_pixels + eps)

        # 4. Kappa系数 (Kappa Coefficient)
        # 计算随机一致性概率
        p_e = ((self.pos_label * self.pos_pred) +
               (neg_label * neg_pred)) / (self.total_pixels ** 2 + eps)

        # 计算Kappa系数
        kc = (pcc - p_e) / (1 - p_e + eps)

        # 5. 返回所有指标
        metrics = {
            'tp': self.tp,
            'fp': self.fp,
            'fn': self.fn,
            'tn': self.tn,
            'oe': oe,
            'pcc': pcc,
            'kappa': kc,
            'precision': self.tp / (self.tp + self.fp + eps),
            'recall': self.tp / (self.tp + self.fn + eps),
            'f1': 2 * self.tp / (2 * self.tp + self.fp + self.fn + eps)
        }

        return metrics

    def __call__(self, pred, gt):
        pred = pred.detach()
        gt = gt.to(pred.device)

        # 扩展真实标注区域
        gt_expanded = F.conv2d(gt, self.filters, padding=self.max_dist)
        gt_expanded = (gt_expanded > 0).float()

        pos_label2 = gt_expanded.float().sum(dim=(2, 3)).cpu()

        # 获取输入形状信息
        B, C, H, W = pred.shape
        pixels_per_image = H * W
        batch_pixels = B * pixels_per_image

        # 更新总像素数
        self.total_pixels += batch_pixels
        self.num_samples += B

        # 遍历所有阈值进行评估
        for idx, th in enumerate(self.thresholds):
            # 二值化预测图
            _pred = (pred > th).float()

            # 可选：对预测结果进行模糊扩展
            if self.blur_pred:
                _pred = F.conv2d(_pred, self.filters, padding=self.max_dist)
                _pred = (_pred > 0).float()

            # 计算混淆矩阵
            tp = ((gt_expanded * _pred) > 0).float().sum()  # 真正例
            fp = (_pred * (1 - gt_expanded)).sum()  # 假正例
            fn = (gt_expanded * (1 - _pred)).sum()  # 假反例
            tn = ((1 - gt_expanded) * (1 - _pred)).sum()  # 真反例

            # 累积统计量
            self.tp[idx] += tp.cpu()
            self.fp[idx] += fp.cpu()
            self.fn[idx] += fn.cpu()
            self.tn[idx] += tn.cpu()

            pos_pred2 = _pred.sum(dim=(2, 3)).cpu()

            self.tp2[idx] += tp.sum().cpu()
            self.pos_pred2[idx] += pos_pred2.sum()
            self.pos_label2[idx] += pos_label2.sum()
