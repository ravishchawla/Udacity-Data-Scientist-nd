# Imports here
import torch;
import numpy as np;
from torch import nn;
from torch import optim;
import torch.functional as func;
from torchvision import datasets, transforms, models;
import time;
from os import path;
import argparse;
import utils
#python train.py --save_dir saving_log/ --arch densenet --hidden_units 256 --learning_rate 0.001 --epochs 10 --gpu
def main(save_dir, arch, hidden_units, learning_rate, epochs, gpu):
    print("Save_Dir: " , save_dir);
    print("Arch: " , arch);
    print("Hidden_Units: " , hidden_units);
    print("Learn_rate: " , learning_rate);
    print("Epochs: " , epochs);
    print("Use_GPU" , gpu);
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'

    train_transforms = transforms.Compose(
        [transforms.RandomRotation(30),
         transforms.RandomResizedCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize([.485, .456, .406], [.229, .224, .225])]);

    testn_transforms = transforms.Compose(
    [transforms.Resize(255),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([.485, .456, .406], [.229, .224, .225])]);

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms);
    valid_data = datasets.ImageFolder(valid_dir, transform=testn_transforms);

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True);
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True);

    pretrained_model = utils.obtain_pretrained_model(arch, hidden_units);
    criterion = nn.NLLLoss();
    optimizer = optim.Adam(pretrained_model.classifier.parameters(), lr=0.001);
    
    if gpu == True:
        device='cuda';
    else:
        device='cpu';

    utils.train_on_data(pretrained_model, train_loader, valid_loader, epochs, 25, criterion, optimizer, device);
    utils.save_model_checkpoint(pretrained_model, arch, hidden_units, train_data, save_dir);
    
if __name__ == '__main__':
    print("foo");
    parser = argparse.ArgumentParser(description='Image Classification Project')
    parser.add_argument('save_dir', action="store", help="location for saving checkpoints (relative)")
    parser.add_argument('--arch', action="store", dest="arch", help="densenet | vgg13 | vgg16 (default)", default="vgg16");
    parser.add_argument('--hidden_units', action="store", dest="hidden_units", help="A good value will be at least 102 and less than maximum units of the architecture (25088 for VGG and 1024 for densenet). Default is 4096 for vgg16", default=4096, type=int);    
    parser.add_argument('--learning_rate', action="store", dest="learning_rate", type=float, help="Range in (0,1), default is 0.001", default=0.001);
    parser.add_argument('--epochs', action="store", dest="epochs", type=int, help="default is 1", default=1);
    parser.add_argument('--gpu', action="store_true", dest="gpu", default=False, help="is provided CUDA gpu will be used, else CPU")
    
    parsed_args = parser.parse_args();

    main(parsed_args.save_dir, parsed_args.arch, parsed_args.hidden_units, parsed_args.learning_rate, parsed_args.epochs, parsed_args.gpu);