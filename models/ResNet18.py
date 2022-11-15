#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torchvision

def Renet18(nOut=256, **kwargs):
    
    return torchvision.models.resnet18(num_classes=nOut)
