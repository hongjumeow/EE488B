
import torch

def MainModel(nOut=256, **kwargs):
    efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
    return efficientnet(num_classes=nOut)