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
        "--classes", type=int, default=37, help="Number of classfication classes"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs for training"
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="Training Learning Rate"
    )
    parser.add_argument(
        '--model_save', default='./data/saved_model.pth', type=str, help="File path to save trained model to"
    )
    parser.add_argument("--download_ds", action='store_true')
    parser.add_argument("--device", type=str, default="cpu", choices=['cpu', 'cuda', 'mps'], help="Device to use")
    
    args = parser.parse_args()
    return args

class Model_Wrapper(torch.nn.Module):
    def __init__(self, backbone_model:torch.nn.Module, backbone_output_dim:int = 1000, num_class:int = 32) -> None:
        super().__init__()
        self.backbone = backbone_model
        self.projection_head = torch.nn.Linear(in_features=backbone_output_dim, out_features=num_class, bias=True)
    
    def forward(self, x):
        outputs = self.backbone(x)
        outputs = self.projection_head(outputs)

        return outputs


def get_model(model_type:str, num_classes:int):
    getter_func = model_dict[model_type]
    return getter_func(num_classes)

def get_loss_optimizer_sched(lr:float, model:torch.nn.Module):
    return train_test_model.get_optimzer_loss_func(lr, model)

def get_loaders(transforms, download:bool = False):
    train, val, test = train_test_model.get_oxford_pets(transforms, download=download)
    #return train, val, test
    train, val, test = train_test_model.get_dataloaders(train, test, val, batch_size=64)

    return train, val, test

def main():
    args = parse_args()
    device = torch.device(args.device)
    
    backbone_model, transforms = get_model(args.model_type, 1000)
    model = Model_Wrapper(backbone_model=backbone_model, num_class=args.classes).to(device)
    
    loss_func, optimizer, scheduler = get_loss_optimizer_sched(args.lr, model)
    train, val, test = get_loaders(transforms, args.download_ds)

    print('Training')
    model = train_test_model.train_model(optimizer=optimizer, loss_func=loss_func, trainloader=train,
                                        valloader=val, model=model, epochs=args.epochs, scheduler=scheduler,
                                        transforms=transforms, device=device)
    
    print('\nTesting')
    # train_test_model.validate_model(model, train, loss_func, device=device)
    train_test_model.test_model(model, test, loss_func, device=device)

    torch.save(model.state_dict(), args.model_save)

if __name__ == '__main__':
    main()