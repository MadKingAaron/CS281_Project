import torch
import torch.optim as optim
from torch import nn



def train_model(optimizer, loss_func, trainloader, model:nn.Module, epochs:int = 50):
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


