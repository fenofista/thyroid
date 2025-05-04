import torch
import torch.nn as nn
import torch.nn.functional as F
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

