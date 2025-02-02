#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Filename:    criteria.py
# @Project:     GuideNet
# @Author:      jie
# @Time:        2021/3/14 7:51 PM

import torch
import torch.nn as nn

__all__ = [
    'RMSE',
    'MSE',
]


class RMSE(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, *args):
        # Ignore that padded part of the images
        # Output returned by model is tuple with 1 entry which corresponds to output tensor
        outputs = outputs[0][:,:,14:-14,8:-8]
        target = target[:,:,14:-14,8:-8]
        val_pixels = (target > 1e-3).float().cuda()
        err = (target * val_pixels - outputs * val_pixels) ** 2
        loss = torch.sum(err.view(err.size(0), 1, -1), -1, keepdim=True)
        cnt = torch.sum(val_pixels.view(val_pixels.size(0), 1, -1), -1, keepdim=True)
        return torch.sqrt(loss / cnt)


class MSE(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, *args):
        # Ignore that padded part of the images
        outputs = outputs[0][:,:,14:-14,8:-8]
        target = target[:,:,14:-14,8:-8]
        val_pixels = (target > 1e-3).float().cuda()
        loss = target * val_pixels - outputs * val_pixels
        return loss ** 2
