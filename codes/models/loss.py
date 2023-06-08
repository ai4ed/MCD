import torch
import torch.nn as nn
import torch.nn.functional as F


def loss_reduction(loss, target, reduction):
    # Loss reduction
    if reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'mean':
        loss = loss.mean()
    else:
        # if no reduction, reshape tensor like target
        loss = loss.view(*target.shape)
    return loss


def get_pt(x, target):
    pt = F.softmax(x, dim=-1)[range(x.shape[0]), target]
    return pt


def poly1_cross_entropy(x, target, epsilon=1.0, reduction='mean'):
    # poly1 cross-entropy loss from PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions
    pt = get_pt(x, target)
    ce = F.cross_entropy(x, target, reduce=False)
    loss = ce + epsilon * (1-pt)
    return loss_reduction(loss, target, reduction)


def focal_loss(x, target, gamma=2.0, reduction='mean'):
    # from Focal Loss for Dense Object Detection: https://arxiv.org/pdf/1708.02002v2.pdf
    pt = get_pt(x, target)
    ce = F.cross_entropy(x, target, reduce=False)
    loss = (1-pt)**gamma * ce
    return loss_reduction(loss, target, reduction)


def polyl_focal_loss(x, target, epsilon=1.0, gamma=2.0, reduction='mean'):
    # PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions
    fl = focal_loss(x, target, gamma=gamma, reduction=reduction)
    pt = get_pt(x, target)
    loss = fl + epsilon * (1-pt)**(gamma+1)
    return loss_reduction(loss, target, reduction)


class Loss():
    def __init__(self, loss_type='ce', epsilon=1.0, gamma=2.0, reduction='mean'):
        self.loss_type = loss_type
        self.epsilon = epsilon
        self.gamma = gamma
        self.reduction = reduction

    def get_loss(self, x, target):
        if self.loss_type == 'focal':
            loss = focal_loss(x, target, self.gamma, reduction=self.reduction)
        elif self.loss_type == 'poly1':
            loss = poly1_cross_entropy(
                x, target, epsilon=self.epsilon, reduction=self.reduction)
        elif self.loss_type == "poly1_focal":
            loss = polyl_focal_loss(
                x, target, epsilon=self.epsilon, gamma=self.gamma, reduction=self.reduction)
        else:
            reduce = False if self.reduction is None else True
            loss = F.cross_entropy(x, target, reduction=self.reduction)
        return loss