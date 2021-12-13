def get_ious(pred, gt):
    """Caculate intersection over union between predcition and ground truth

    Parameters
    ----------
        pred: 
            predictions from the model
        gt: 
            ground truth labels
    """

    smooth = 1
    # flatten label and prediction tensors
    inputs = pred.view(-1)
    targets = gt.view(-1)

    # intersection is equivalent to True Positive count
    # union is the mutually inclusive area of all labels & predictions
    intersection = (inputs * targets).sum()
    total = (inputs + targets).sum()
    union = total - intersection

    IoU = (intersection + smooth) / (union + smooth)

    return IoU


def get_dice_metric(pred, gt):
    smooth = 1
    # flatten label and prediction tensors
    inputs = gt.view(-1)
    targets = pred.view(-1)

    intersection = (inputs * targets).sum()
    total = inputs.sum() + targets.sum()
    dice = (2. * intersection + smooth) / (total + smooth)
    return dice
