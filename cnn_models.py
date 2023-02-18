import torch
import torchvision
from torch import nn
from torchvision import models



def get_resnet50(class_num:int = 32):
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights, num_classes=class_num)
    transforms = weights.transforms()

    return model, transforms


def get_resnet18(class_num:int = 32):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights, num_classes=class_num)
    transforms = weights.transforms()

    return model, transforms


def get_vgg13(class_num:int = 32):
    weights = models.VGG13_Weights.DEFAULT
    model = models.vgg13(weights=weights, num_classes=class_num)
    transforms = weights.transforms()

    return model, transforms

def get_inceptionV3(class_num:int = 32):
    weights = models.Inception_V3_Weights.DEFAULT
    model = models.inception_v3(weights=weights, num_classes=class_num)
    transforms = weights.transforms()

    return model, transforms