import torch;
import numpy as np;
from torch import nn;
from torch import optim;
import torch.functional as func;
from torchvision import datasets, transforms, models;
from collections import OrderedDict
import time;
import os;
from os import path;
import PIL;
from PIL import Image

def obtain_pretrained_model(model_name, hidden_units):
    print('bar');
    if model_name == 'vgg13':
        print('Loading pretrained vgg13 model');
        model = models.vgg13(pretrained=True);
        
        model = set_classifier(model, 25088, hidden_units, 102);
        
    elif model_name == 'densenet':
        print('Loading pretrained densenet model');
        model = models.densenet121(pretrained=True);

        model = set_classifier(model, 1024, hidden_units, 102);
    else:
        print('Loading pretrained vgg16 model');
        model = models.vgg16(pretrained=True);
        
        model = set_classifier(model, 25088, hidden_units, 102);
            
    print('Model Classifier Architecture: ');
    print(model.classifier);
    print('\n');
    return model;
def set_classifier(model, in_feats, hidden_units, out_units):
    for param in model.parameters():
        param.require_grad = False;
            
    classifier = nn.Sequential(OrderedDict([
        ('fc1' , nn.Linear(in_feats, hidden_units)),
        ('relu1' , nn.ReLU()),
        ('drop1' , nn.Dropout(0.5)),
        ('fc2' , nn.Linear(hidden_units, hidden_units // 2)),
        ('relu2' , nn.ReLU()),
        ('drop2' , nn.Dropout(0.2)),
        ('fc3' , nn.Linear(hidden_units // 2, out_units)),
        ('output' , nn.LogSoftmax(dim=1))
    ]));
    
    model.classifier = classifier;
    return model;

def train_on_data(model, train_loader, valid_loader, epochs, print_every, criterion, optimizer, device):
    start_epoch = time.time();
    
    model.to(device);
    model.train();
    
    for epoch in range(epochs):
        steps = 0;
        avg_loss = 0;
        for ite, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device);

            steps = steps + 1;

            optimizer.zero_grad();

            outputs = model.forward(inputs);
            loss = criterion(outputs, labels);
            loss.backward();
            optimizer.step();

            avg_loss = avg_loss + loss.item();

            if (steps % print_every == 0):
                print(f'Epoch : {epoch}/{steps} :: Training Loss : {(avg_loss / print_every):.4f} :: Time: {(time.time() - start_epoch):.3f}');
                avg_loss = 0;

        test_on_validation(model, criterion, valid_loader, device);

def save_model_checkpoint(model, arch, hidden_units, train_data, directory):
    try:
        os.makedirs(directory);
    except:
        print('Error creating directory, already exists. Different path required.');
            
    checkpoint = {'classifier' : model.classifier,
              'class_to_idx' : train_data.class_to_idx,
              'state_dict' : model.state_dict(),
              'arch' : arch,
              'hidden_units' : hidden_units};

    path = directory + 'checkpoint.pth';
    torch.save(checkpoint, path);
    print('Model properties saved at ', path);
        
def test_on_validation(model, criterion, valid_loader, device):
    correct = 0;
    total = 0;
    
    with torch.no_grad():
        model.eval();
        start_valid = time.time();
        
        avg_v_loss = 0;
        valid_steps = 0;
        for data in valid_loader:
            inputs, labels = data;
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model.forward(inputs);
            
            loss = criterion(outputs, labels);
            
            avg_v_loss = avg_v_loss + loss.item();
            valid_steps = valid_steps + 1;
            
            preds = torch.exp(outputs);

            correct = correct + torch.sum(labels.data == preds.max(dim=1)[1]).item()
            
            total = total + labels.size(0);

    accuracy = (correct / total);
    
    avg_v_loss = avg_v_loss / valid_steps;
    
    print(f'Validation Accuracy: {(100 * accuracy):.3f}\% :: Validation Loss: {(avg_v_loss):.3f} Time: {time.time() - start_valid}');
    
def predict(image_path, model, idx_to_class, cat_to_name, topk, device):
    model.to(device);
    with torch.no_grad():
        torch_image = torch.from_numpy(process_image(image_path)).type(torch.FloatTensor);
        
        torch_image.resize_([1, 3, 224, 224]);
        torch_image = torch_image.to(device);
        
        outputs = model.forward(torch_image);
        predictions = torch.exp(outputs);
        
        top_k = predictions.topk(topk);
        
        top_k_probs, top_k_outs = predictions.topk(topk);
        
        top_k_probs = top_k_probs.cpu().numpy().tolist()[0];
        top_k_outs = top_k_outs.cpu().numpy().tolist()[0];
        
        if cat_to_name is None:
            class_names = [idx_to_class[cla] for cla in top_k_outs];
        else:    
            class_names = [cat_to_name[idx_to_class[cla]] for cla in top_k_outs];

    return top_k_probs, top_k_outs, class_names;
    
def load_model_checkpoint(path):
    if ".pth" not in path:
        print('Invalid checkpoint file provided. File must be of format .pth');
    try:
        _checkpoint = torch.load(path);
        arch = _checkpoint['arch'];
        hidden_units = _checkpoint['hidden_units'];
        classifier = _checkpoint['classifier'];
        class_to_idx = _checkpoint['class_to_idx'];
        state_dict = _checkpoint['state_dict'];
        
        model = obtain_pretrained_model(arch, hidden_units);
        model.classifier = classifier;
        model.state_dict = state_dict;
        
        return model, class_to_idx;
    except:
        print('Unable to load checkpoint. File may not exist or be malformed');
        
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    image_transforms = transforms.Compose([transforms.Resize(255),
                         transforms.CenterCrop(224),
                         transforms.ToTensor(),
                         transforms.Normalize([.485, .456, .406], [.229, .224, .225])]);

    
    pil_image = Image.open(image);
    transformed_image = image_transforms(pil_image).float();
    np_image = np.array(transformed_image);
    
    return np_image;