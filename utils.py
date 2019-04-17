import argparse
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets,transforms,models
from workspace_utils import active_session
import json
import numpy as np
from PIL import Image

def save_checkpoint(checkpoint_path,model,optimizer,arch,hidden_unit):
    if arch=='vgg16':
         checkpoint={'input_size':25088,
                     'output_size':102,
                     'hidden_unit':hidden_unit,
                     'state_dict':model.state_dict(),
                     'optim_dict':optimizer.state_dict,
                     'dropout':0.4,
                     'architecture':arch,
                     'class_to_idx':model.class_to_idx}
    else: 
        checkpoint={'input_size':1024,
                    'output_size':102,
                    'hidden_unit':hidden_unit,
                    'state_dict':model.state_dict(),
                    'optim_dict':optimizer.state_dict,
                    'dropout':0.4,
                    'architecture':arch,
                    'class_to_idx':model.class_to_idx}
    torch.save(checkpoint,checkpoint_path)
    
def load_checkpoint(filepath):
    checkpoint=torch.load(filepath)
    input_size=checkpoint['input_size']
    output_size=checkpoint['output_size']
    hidden_unit=checkpoint['hidden_unit']
    dropout=checkpoint['dropout']
    arch=checkpoint['architecture']
    
    if arch=='vgg16':
        model=models.vgg16(pretrained=True)     
    else:
       model=models.densenet121(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad=False
        
    classifier=nn.Sequential(nn.Linear(input_size,hidden_unit),
                             nn.ReLU(),
                             nn.Dropout(dropout),
                             nn.Linear(hidden_unit,output_size),
                             nn.LogSoftmax(dim=1))
    model.classifier=classifier
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx=checkpoint['class_to_idx']
    
    return model


def cat_to_names(category_name):
    with open(category_name) as f:
        cat_to_name = json.load(f)

        return cat_to_name



    