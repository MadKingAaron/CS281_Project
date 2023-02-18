import torch
import torchvision
import argparse
import get_simclr_model
import cnn_models
import transformer_models
import train_test_model




model_dict = {'simclr': lambda x: get_simclr_model.get_simclr_pretrained(class_num=x),
    'resnet18':cnn_models.get_resnet18, 'resnet50':cnn_models.get_resnet50, 
    'vgg':cnn_models.get_vgg13, 'inception':cnn_models.get_inceptionV3,
    'vit':transformer_models.get_vitB32, 'swin':transformer_models.get_swinTiny}

def parse_args():
    '''Parse input arguments'''
    
    parser = argparse.ArgumentParser()
    model_choices = ['simclr','resnet18', 'resnet50', 'vgg', 'inception', 'vit', 'swin']
    parser.add_argument(
        "--model_type", type=str, default="simclr", choices=model_choices, help="Model type for training - {}".format(str(model_choices))
    )
    parser.add_argument(
        "--classes", type=int, default=32, help="Number of classfication classes"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs for training"
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="Training Learning Rate"
    )
    parser.add_argument(
        '--model_save', default='./data/saved_model.pth', type=str, help="File path to trained model to"
    )
    parser.add_argument("--download_ds", action='store_true')
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    
    args = parser.parse_args()
    return args

def get_model(model_type:str, num_classes:int):
    getter_func = model_dict[model_type]
    return getter_func(num_classes)

def get_loss_optimizer_sched(lr:float, model:torch.nn.Module):
    return train_test_model.get_optimzer_loss_func(lr, model)

def get_loaders(download:bool = False):
    train, val, test = train_test_model.get_oxford_pets(download=download)
    #return train, val, test
    train, val, test = train_test_model.get_dataloaders(train, test, val)

    return train, val, test

def main():
    args = parse_args()
    model, transforms = get_model(args.model_type, args.classes)
    loss_func, optimizer, scheduler = get_loss_optimizer_sched(args.lr, model)
    train, val, test = get_loaders(args.download_ds)

