from transformers import AutoImageProcessor, Swinv2ForImageClassification, ViTForImageClassification
import torch


def load_swinV2(num_classes:int=32):
    image_processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
    model = Swinv2ForImageClassification.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")

    model.classifier = torch.nn.Linear(in_features=768, out_features=num_classes, bias=True)
    
    return model, image_processor

def load_vit(num_classes:int=32):
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

    model.classifier = torch.nn.Linear(in_features=768, out_features=num_classes, bias=True)

    return model, image_processor