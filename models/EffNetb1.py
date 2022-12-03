import torchvision

def MainModel(nOut=256, **kwargs):
    return torchvision.models.efficientnet_b1(num_classes=nOut)