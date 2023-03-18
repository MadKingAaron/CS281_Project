# CS281 Project

## Train and Test Model
Running ```run_training.py``` will both train a model and test it:
```
python run_training.py <program_arguments> 
```

Potential Program Arguments:

* ```--model_type <model_type>``` -> Options: ```simclr, resnet18, resnet50, vgg, incpetion, vit, swin```
* ``` --classes <number of classes> ``` -> Number of classes for Oxford-IIIT Pets Dataset is 37
* ```--epochs <training_epochs>```
* ```-lr <learning_rate>```
* ```--model_save <File path to save trained model to>```
* ```--download_ds``` -> Only run once to download the Oxford-IIIT Pets Dataset
* ```--device <device>``` -> Options: ```cuda, cpu, mps```
* ```--batch_size <training batch size>```
* ```--transfer_learning``` -> Use when you want to freeze the weights for the backbone model during training


## Before running
**MAKE SURE TO CREATE A FOLDER NAMED ```simclr_model``` IN PROJECT FOLDER BEFORE TRAINING THE SIMCLR MODEL**

## Requirements
* PyTorch, Torchvision
* Tensorboard
* Tqdm
* Scikit-Learn




---
