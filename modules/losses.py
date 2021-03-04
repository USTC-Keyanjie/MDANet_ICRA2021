"""
This script is modified from the work of Abdelrahman Eldesokey.
Find more details from https://github.com/abdo-eldesokey/nconv
"""

########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################

import torch
import torch.nn as nn

class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, outputs, target, *args):
        val_pixels = torch.ne(target, 0).float().cuda()
        loss = target * val_pixels - outputs * val_pixels
        return torch.sum(torch.abs(loss)) / torch.sum(val_pixels)


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, outputs, target, *args):
        val_pixels = torch.ne(target, 0).float().cuda()
        loss = target * val_pixels - outputs * val_pixels
        return torch.sum(loss ** 2) / torch.sum(val_pixels)


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, outputs, target, *args):
        return self.loss(outputs, target)


class BINLoss(nn.Module):
    def __init__(self):
        super(BINLoss, self).__init__()
        self.CrossEntropyLoss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, outputs, target_bin_cls, target_res_reg, *args):
        # outputs_bin_cls, outputs_res_reg = outputs[:, :10, :, :], outputs[:, 10:, :, :]
        outputs_bin_cls, outputs_res_reg = torch.chunk(outputs, 2, 1)
        bin_cls_loss = self.CrossEntropyLoss(outputs_bin_cls, target_bin_cls)

        val_pixels = torch.ge(target_bin_cls.unsqueeze(1), 0).float()
        outputs_res_reg = torch.gather(outputs_res_reg, 1, target_bin_cls.unsqueeze(1) * val_pixels.long())
        # val_pixels = val_pixels.float()
        res_reg_loss = outputs_res_reg * val_pixels - target_res_reg * val_pixels
        res_reg_loss = torch.sum(res_reg_loss ** 2) / torch.sum(val_pixels)
        # print(bin_cls_loss.item(), res_reg_loss.item())
        return 5 * (bin_cls_loss + 10 * res_reg_loss)

        # bin_onehot = torch.zeros_like(outputs_bin_cls).scatter_(1, target_bin_cls * val_pixels, 1)
        # val_pixels = val_pixels.float()
        # res_reg_loss = (bin_onehot * outputs_res_reg).sum(1) * val_pixels - target_res_reg * val_pixels
        # res_reg_loss = torch.sum(res_reg_loss ** 2) / torch.sum(val_pixels)
        # return bin_cls_loss + res_reg_loss


if __name__ == '__main__':
    input = torch.randn([2, 20, 384, 1248])
    object = BINLoss()
    target_bin_cls = torch.randint(low=-1, high=10, size=[2, 384, 1248])
    target_res_reg = torch.randn([2, 1, 384, 1248])

    loss = object(input, target_bin_cls, target_res_reg)
