import torch
from torch import nn as nn
from torch.nn import functional as F
from basicsr.utils.registry import LOSS_REGISTRY

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM


@LOSS_REGISTRY.register()
class KDLoss(nn.Module):
    """
    Args:
        loss_weight (float): Loss weight for KD loss. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, temperature = 0.15):
        super(KDLoss, self).__init__()
    
        self.loss_weight = loss_weight
        self.temperature = temperature

    def forward(self, S1_fea, S2_fea):
        """
        Args:
            S1_fea (List): contain shape (N, L) vector. 
            S2_fea (List): contain shape (N, L) vector.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        loss_KD_dis = 0
        loss_KD_abs = 0
        for i in range(len(S1_fea)):
            S2_distance = F.log_softmax(S2_fea[i] / self.temperature, dim=1)
            S1_distance = F.softmax(S1_fea[i].detach()/ self.temperature, dim=1)
            loss_KD_dis += F.kl_div(
                        S2_distance, S1_distance, reduction='batchmean')
            loss_KD_abs += nn.L1Loss()(S2_fea[i], S1_fea[i].detach())
        return self.loss_weight * loss_KD_dis, self.loss_weight * loss_KD_abs

@LOSS_REGISTRY.register()
class SSIMLoss(nn.Module):
    """
    Args:
        loss_weight (float): Loss weight for KD loss. Default: 1.0.
        ms: multi-scale ssim 还是 ssim
    """

    def __init__(self, loss_weight=1.0, ms=False, win_size=11, win_sigma=1.5, data_range=1.0, size_average=True, channel=3):
        super(SSIMLoss, self).__init__()

        self.loss_weight = loss_weight

        if ms:
            self.ssim_module = MS_SSIM(win_size, win_sigma, data_range, size_average, channel)
        else:
            self.ssim_module = SSIM(win_size, win_sigma, data_range, size_average, channel)

    def forward(self, pred, target):
        """
        Args:
            输入图像的range需要在0-1之间
        """

        return self.loss_weight * (1-self.ssim_module(pred, target))


