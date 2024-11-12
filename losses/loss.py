import torch
import torch.nn as nn
import torch.nn.functional as F
#from lovasz_losses import lovasz_hinge

# class CombinedLovaszBCELoss(nn.Module):
#     def __init__(self, weight_bce=0.5, weight_lovasz=0.5):
#         super(CombinedLovaszBCELoss, self).__init__()
#         self.weight_bce = weight_bce
#         self.weight_lovasz = weight_lovasz
#         self.bce_loss = nn.BCEWithLogitsLoss()

#     def forward(self, inputs, targets):
#         # BCE Loss
#         bce = self.bce_loss(inputs, targets)

#         # Lovász Hinge Loss
#         # La Lovász Hinge Loss richiede i logits non attivati (senza sigmoid)
#         lovasz = lovasz_hinge(inputs, targets)

#         # Combinazione delle perdite
#         loss = self.weight_bce * bce + self.weight_lovasz * lovasz
#         return loss
    
# Dice Loss for segmentation tasks (for roots)
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        true_pos = torch.sum(targets * inputs)
        false_neg = torch.sum(targets * (1 - inputs))
        false_pos = torch.sum((1 - targets) * inputs)
        tversky_coeff = (true_pos + self.smooth) / \
                        (true_pos + self.alpha * false_neg + self.beta * false_pos + self.smooth)
        return 1 - tversky_coeff
    
class DiceLoss(nn.Module):
    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs).to('cuda:0')  # Convert logits to probabilities
        targets = targets.to('cuda:0')
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice
class StructureLoss(nn.Module):
    def __init__(self):
        super(StructureLoss, self).__init__()
        self.bce_function = F.binary_cross_entropy_with_logits
    def forward(self, pred, mask):
        weit = 1 + 5 * \
            torch.abs(F.avg_pool2d(mask, kernel_size=31,
                                   stride=1, padding=15) - mask)
        wbce = self.bce_function(pred, mask, reduction='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

 
        pred = torch.sigmoid(pred)
        inter = ((pred * mask)*weit).sum(dim=(2, 3))
        union = ((pred + mask)*weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1)/(union - inter+1)

 
        return (wbce + wiou).mean()
# Focal Loss for imbalanced classes (for tips and sources)
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce_loss(inputs, targets)
        pt = torch.sigmoid(inputs)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        focal_loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Weighted BCE Loss for class imbalance (for tips and sources)
class WeightedBCELoss(nn.Module):
    def __init__(self, weight):
        super(WeightedBCELoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=weight)

    def forward(self, inputs, targets):
        return self.bce_loss(inputs, targets)
    
