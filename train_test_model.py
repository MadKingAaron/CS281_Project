import torch
import torch.optim as optim
from torch import nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter

import torchvision.datasets as datasets
from torchvision import transforms

from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report

from tqdm import tqdm


def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)

    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def get_oxford_pets(transform:any, root:str = './oxford_pets', val_split:float = 0.25, download:bool = False):
    #transform = transforms.Compose([
    #    transforms.ToTensor()
    #])
    train_val = datasets.OxfordIIITPet(root=root, split='trainval', download=download, transform=transform)

    train, val = train_val_dataset(dataset=train_val, val_split=val_split)

    test = datasets.OxfordIIITPet(root=root, split='test', download=download, transform=transform)

    return train, val, test

def get_dataloaders(train, val, test, batch_size:int = 256):
    
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                          shuffle=True, num_workers=2, )
    val_loader = torch.utils.data.DataLoader(val, batch_size=64,
                                          shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test, batch_size=64,
                                          shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader


def get_optimzer_loss_func(initial_lr:float, model:nn.Module):
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer)

    return loss_func, optimizer, scheduler

def apply_transforms(transforms, imgs:torch.Tensor, transformer_model:bool) -> torch.Tensor:
    if transformer_model:
        imgs = transforms([x for x in imgs], return_tensors="pt")['pixel_values']
    else:
        imgs = transforms(imgs)
    
    return imgs



def train_model(optimizer, loss_func, trainloader, valloader, model:nn.Module, epochs:int = 50, transformer_model:bool = False, scheduler = None, transforms = None,
                device = 'cpu'):
    running_loss = 0.0
    writer = SummaryWriter()
    for epoch in range(epochs):
        model.train()
        print('Epoch %d' %(epoch+1))
        for i, data in enumerate(tqdm(trainloader), 0):
            # Get inputs and labels
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero grad
            optimizer.zero_grad()

            # Apply image transformations
            # inputs = apply_transforms(transforms, inputs, transformer_model)

            # Forward + Backprop
            outputs = model(inputs)
            #print('Shape:', outputs.shape)

            if transformer_model:
                outputs = outputs.logits
            
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()


            # Print current stats
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
        
        # Get validation accuracy and loss
        #print(labels.shape)
        val_accuracy, val_loss = validate_model(model, valloader, loss_func, transformer_model, transforms, device)
        writer.add_scalar("Val/AvgLoss", val_loss, epoch)
        writer.add_scalar("Train/AvgLoss", running_loss, epoch)

        if scheduler is not None:
            scheduler.step(val_loss)
        running_loss = 0.0
    writer.close()
    return model

def validate_model(model, valloader, loss_func, transformer_model:bool = False, transforms = None, device = 'cpu'):
    correct = 0
    total = 0
    total_loss = 0.0
    i = 0
    model.eval()

    for inputs, labels in valloader:
        i += 1   
        # Apply image transformations
        # inputs = apply_transforms(transforms, inputs, transformer_model)
        inputs, labels = inputs.to(device), labels.to(device)
        #print(labels.shape)
        outputs = model(inputs)

        if transformer_model:
            outputs = outputs.logits

        #print(outputs)
        # Calc loss
        loss = loss_func(outputs, labels)
        total_loss += loss.item()

        # Predict classes
        _, preds = torch.max(outputs, 1)
            
        # Check predictions
        for label, pred, in zip(labels, preds):
            if label == pred:
                correct += 1
            total += 1

    return (correct/total), (total_loss/i)

def test_model(model, testloader, transformer_model:bool = False,  device = 'cpu'):
    model.eval()
    total_preds = None
    total_labels = None
    i = 1
    for inputs, labels in testloader:
            
        # Apply image transformations
        # inputs = apply_transforms(transforms, inputs, transformer_model)
        inputs, labels = inputs.to(device), labels.to(device)
        #print(labels.shape)
        outputs = model(inputs)

        #if transformer_model:
        #    outputs = outputs.logits

    

        # Predict classes
        _, preds = torch.max(outputs, 1)

        if total_preds is None:
            total_preds = preds.cpu()
            total_labels = labels.cpu()
        else:
            #print('i:', i)
            total_labels = torch.cat((total_labels, labels.cpu())).cpu()
            total_preds = torch.cat((total_preds, preds.cpu())).cpu()

        i+=1
    report = classification_report(y_true=total_labels, y_pred=total_preds)
    print(report)