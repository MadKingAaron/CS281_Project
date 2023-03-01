#from transformers import AutoImageProcessor, Swinv2ForImageClassification, ViTForImageClassification
import torch
import torchvision
from torch import nn
from torchvision import models


"""def load_swinV2_huggingFace(num_classes:int=32):
    image_processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
    model = Swinv2ForImageClassification.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")

    model.classifier = torch.nn.Linear(in_features=768, out_features=num_classes, bias=True)
    
    return model, image_processor

def load_vit_huggingFace(num_classes:int=32):
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

    model.classifier = torch.nn.Linear(in_features=768, out_features=num_classes, bias=True)

    return model, image_processor"""


def get_vitB32(class_num:int = 32):
    weights = models.ViT_B_32_Weights.DEFAULT
    model = models.vit_b_32(weights=weights, num_classes=class_num)
    transforms = weights.transforms()

    return model, transforms

def get_swinTiny(class_num:int = 32):
    weights = models.Swin_T_Weights.DEFAULT
    model = models.swin_t(weights=weights, num_classes=class_num)
    transforms = weights.transforms()

    return model, transforms