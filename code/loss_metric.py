import torch
import torch.nn as nn
import torch.nn.functional as F


# No need to do any activation in train pipeline, model output activation will be add it here.

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, need_act = True):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.need_act = need_act

    def forward(self, inputs, targets):
        """
        inputs: (N, 1, H, W) - predicted probabilities (after sigmoid)
        targets: (N, 1, H, W) - ground truth (0 or 1)
        """
        if self.need_act:
            inputs = nn.Sigmoid()(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        return 1 - dice  # because we want to minimize the loss


def IOU_score(preds, targets, threshold=0.5, eps=1e-6, need_act = True):
    """
    preds: (N, 1, H, W) - predicted probabilities
    targets: (N, 1, H, W) - ground truth binary mask (0 or 1)
    threshold: threshold to binarize predictions
    eps: small value to avoid division by zero
    """
    if need_act:
        preds = nn.Sigmoid()(preds)
    preds = (preds > threshold).float()
    targets = targets.float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection

    iou = (intersection + eps) / (union + eps)
    return iou.mean()  # return average IoU across batch



class StructureLoss(nn.Module):
    def __init__(self):
        super(StructureLoss, self).__init__()

    def forward(self, pred, mask):
        """
        inputs: (N, 1, H, W) - predicted probabilities (after sigmoid)
        targets: (N, 1, H, W) - ground truth (0 or 1)
        """
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)

        weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    
        pred = torch.sigmoid(pred)
        inter = ((pred * mask)*weit).sum(dim=(2, 3))
        union = ((pred + mask)*weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1)/(union - inter+1)

        return (wbce + wiou).mean()  # because we want to minimize the loss

import torch
import torch.nn as nn

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6):
        """
        Tversky Loss function

        Args:
            alpha: weight for false positives
            beta: weight for false negatives
            smooth: smoothing constant to avoid division by zero
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Args:
            inputs: predicted tensor with shape [B, 1, H, W] or [B, H, W]
            targets: ground truth tensor with same shape as inputs
        """
        inputs = torch.sigmoid(inputs)  # if not already probabilities

        # Flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        tversky_index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return 1 - tversky_index

