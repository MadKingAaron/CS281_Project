import torch
import sys
import numpy as np
import os
import torchvision
import zipfile
import gdown

from torch import nn



def get_file_id_by_model(folder_name):
  file_id = {'resnet18_100-epochs_stl10': '14_nH2FkyKbt61cieQDiSbBVNP8-gtwgF',
             'resnet18_100-epochs_cifar10': '1lc2aoVtrAetGn0PnTkOyFzPCIucOJq7C',
             'resnet50_50-epochs_stl10': '1ByTKAUsdm_X7tLcii6oAEl5qFRqRMZSu'}
  return file_id.get(folder_name, "Model not found.")


def download_file(file_url:str, output_location:str):
    gdown.download(file_url, output_location)

def unzip(file:str, tgt_dir:str):
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(tgt_dir)

def get_model_checkpoint(model_name:str = 'resnet50_50-epochs_stl10') -> str:
    download_folder = './simclr_model'
    folder_name = 'resnet50_50-epochs_stl10'
    folder_location = os.path.join(download_folder, folder_name)
    
    file_id = get_file_id_by_model(folder_name)
    url = "https://drive.google.com/uc?id={}".format(file_id)

    download_file(url, folder_location+'.zip')
    unzip(folder_location+'.zip', download_folder)


    checkpt_path = os.path.join(download_folder, 'checkpoint_0040.pth.tar')

    return checkpt_path
    #os.system('gdown https://drive.google.com/uc?id={} --folder {}'.format(file_id, download_folder))
    #os.system('unzip {}'.format(folder_name))

def get_model(checkpt_path:str, model_arch:str = 'resnet50', classes:int = 10):
    if 'resnet50' in model_arch:
        model = torchvision.models.resnet50(pretrained=False, num_classes=classes)
        transforms = torchvision.models.ResNet50_Weights.DEFAULT.transforms()
    elif 'resnet18' in model_arch:
        model = torchvision.models.resnet18(pretrained=False, num_classes=classes)
        transforms = torchvision.models.ResNet18_Weights.DEFAULT.transforms()
    else:
        raise Exception('Wrong model arch selected')
    
    checkpoint = torch.load(checkpt_path, map_location=torch.device('cpu'))

    state_dict = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        if k.startswith('backbone.'):
            if k.startswith('backbone') and not k.startswith('backbone.fc'):
                # remove prefix
                state_dict[k[len("backbone."):]] = state_dict[k]
        del state_dict[k]

    log = model.load_state_dict(state_dict, strict=False)
    assert log.missing_keys == ['fc.weight', 'fc.bias']

    return model, transforms

def get_simclr_pretrained(pretrained_model:str = 'resnet50_50-epochs_stl10', class_num:int = 10):
    checkpt_path = get_model_checkpoint(model_name=pretrained_model)
    model, transforms = get_model(checkpt_path=checkpt_path, model_arch=pretrained_model, classes=class_num)
    return model, transforms

if __name__ == '__main__':
    checkpt_path = get_model_checkpoint()
    model, transforms = get_model(checkpt_path=checkpt_path, classes=30)
    print(model)