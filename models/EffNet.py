
import torchvision

def MainModel(nOut=256, **kwargs):
    efficientnet = torchvision.models.efficientnet_b3(num_classes=nOut)
    return efficientnet