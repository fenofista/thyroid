import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        inputs: (N, 1, H, W) - predicted probabilities (after sigmoid)
        targets: (N, 1, H, W) - ground truth (0 or 1)
        """
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        return 1 - dice  # because we want to minimize the loss


def IOU_score(preds, targets, threshold=0.5, eps=1e-6):
    """
    preds: (N, 1, H, W) - predicted probabilities
    targets: (N, 1, H, W) - ground truth binary mask (0 or 1)
    threshold: threshold to binarize predictions
    eps: small value to avoid division by zero
    """
    preds = (preds > threshold).float()
    targets = targets.float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection

    iou = (intersection + eps) / (union + eps)
    return iou.mean()  # return average IoU across batch

