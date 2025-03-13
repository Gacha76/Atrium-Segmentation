import torch

SMOOTH = 1e-8
THRESHOLD = 0.5


class DiceScore(torch.nn.Module):
    """
    class to compute the Dice Score
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, mask, is_val=False):
        if is_val:
            pred = (pred > THRESHOLD).float()

        # flatten label and prediction tensors
        pred = torch.flatten(pred)
        mask = torch.flatten(mask)

        counter = (pred * mask).sum()  # Counter
        denominator = pred.sum() + mask.sum() + SMOOTH  # denominator
        dice_score = (2 * counter + SMOOTH) / denominator

        return dice_score
    

class Precision(torch.nn.Module):
    """
    class to compute the precision
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, mask, is_val=False):
        if is_val:
            pred = (pred > THRESHOLD).float()

        # flatten label and prediction tensors
        pred = torch.flatten(pred)
        mask = torch.flatten(mask)

        counter = (pred * mask).sum() + SMOOTH  # Counter
        denominator = pred.sum() + SMOOTH  # denominator
        precision = counter / denominator

        return precision
    

class Recall(torch.nn.Module):
    """
    class to compute the recall
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, mask, is_val=False):
        if is_val:
            pred = (pred > THRESHOLD).float()

        # flatten label and prediction tensors
        pred = torch.flatten(pred)
        mask = torch.flatten(mask)

        counter = (pred * mask).sum() + SMOOTH  # Counter
        denominator = mask.sum() + SMOOTH  # denominator
        recall = counter / denominator

        return recall
    

class IoU(torch.nn.Module):
    """
    class to compute the Intersection over Union
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, mask, is_val=False):
        if is_val:
            pred = (pred > THRESHOLD).float()

        # flatten label and prediction tensors
        pred = torch.flatten(pred)
        mask = torch.flatten(mask)

        counter = (pred * mask).sum()  # Counter
        denominator = pred.sum() + mask.sum() - counter + SMOOTH  # denominator

        iou = (counter + SMOOTH) / denominator

        return iou
