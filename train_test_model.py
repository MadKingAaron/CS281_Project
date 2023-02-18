import torch
import torch.optim as optim
from torch import nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

import torchvision.datasets as datasets


def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)

    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def get_oxford_pets(root:str = './oxford_pets', val_split:float = 0.25):
    train_val = datasets.OxfordIIITPet(root=root, split='trainval')

    train, val = train_val_dataset(dataset=train_val, val_split=val_split)

    test = datasets.OxfordIIITPet(root=root, split='test')

    return train, val, test

def get_dataloaders(train, val, test, batch_size:int = 256):
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val, batch_size=64,
                                          shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test, batch_size=64,
                                          shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader


def get_optimzer_loss_func(initial_lr:float, model:nn.Module):
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer)

    return loss_func, scheduler

def train_model(optimizer, loss_func, trainloader, valloader, model:nn.Module, epochs:int = 50):
    running_loss = 0.0

    for epoch in range(epochs):
        for i, data in enumerate(trainloader, 0):
            # Get inputs and labels
            inputs, labels = data

            # Zero grad
            optimizer.zero_grad()

            # Forward + Backprop
            outputs = model(inputs)
            
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()


            # Print current stats
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
        
        # Get validation accuracy
        val_accuracy = validate_model(model, valloader)

    return model

def validate_model(model, testloader):
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = model(inputs)

            # Predict classes
            _, preds = torch.max(outputs, 1)
            
            # Check predictions
            for label, pred, in zip(labels, preds):
                if label == pred:
                    correct += 1
                total += 1
    
    return correct/total


