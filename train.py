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
from utils import save_checkpoint

def parse_args():
    parser=argparse.ArgumentParser(description="Training Network")
    parser.add_argument('--data_dir',action='store')
    parser.add_argument('--learning_rate',dest='learning_rate',default='0.001')
    parser.add_argument('--epochs',dest='epochs',default='5')
    parser.add_argument('--arch',dest='arch',default='densenet121',choices=['vgg16','densenet121'])
    parser.add_argument('--hidden_unit',dest='hidden_unit',default='500')
    parser.add_argument('--gpu',action='store',default='gpu')
    parser.add_argument('--save_dir',dest='save_dir',action='store',default='checkpoint.pth')
    
    return parser.parse_args()

def train(model,train_loader,validate_loader,epochs,gpu,optimizer,criterion):
    if gpu=='gpu':
        model=model.to('cuda')
    else:
        model=model.to('cpu')
        
    epochs=epochs
    print_every=10
    running_loss=0
    steps=0

    with active_session():
        for e in range(epochs):
            for images,labels in train_loader:
                steps+=1
                if gpu=='gpu':
                    images=images.to('cuda')
                    labels=labels.to('cuda')
                    
                else:
                    images=images.to('cpu')
                    labels=labels.to('cpu')
                    
                optimizer.zero_grad()

                logits=model.forward(images)
                loss=criterion(logits,labels)
                loss.backward()
                optimizer.step()
                running_loss+=loss.item()

                if steps%print_every==0:
                    validation_loss=0
                    validation_accuracy=0
                    model.eval()
                    with torch.no_grad():
                        for images,labels in validate_loader:
                             if gpu=='gpu':
                                images=images.to('cuda')
                                labels=labels.to('cuda')
                    
                             else:
                                images=images.to('cpu')
                                labels=labels.to('cpu')
                                
                             logits=model.forward(images)
                             loss=criterion(logits,labels)
                             validation_loss+=loss.item()

                             ps=torch.exp(logits)
                             top_p,top_class=ps.topk(1,dim=1)
                             equals=top_class==labels.view(*top_class.shape)
                             validation_accuracy+=torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch: {e+1}/{epochs} "
                          f" Training Loss:{running_loss/print_every:.3f} "
                          f" Validation Loss:{validation_loss/len(validate_loader):.3f} "
                          f" Validation Accuracy:{validation_accuracy/len(validate_loader)*100:.3f} ")
                    running_loss=0
                    model.train()
                   
def main():
    args=parse_args()
    
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms=transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485,0.456,0.406],
                                                          [0.229,0.224,0.225])]) 
    test_transforms=transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485,0.456,0.406],
                                                             [0.229,0.224,0.225])])
    validate_transforms=transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485,0.456,0.406],
                                                             [0.229,0.224,0.225])])
    
    train_data=datasets.ImageFolder(train_dir,transform=train_transforms)
    test_data=datasets.ImageFolder(test_dir,transform=test_transforms)
    validate_data=datasets.ImageFolder(valid_dir,transform=validate_transforms)
    
    train_loader=torch.utils.data.DataLoader(train_data,batch_size=64,shuffle=True)
    test_loader=torch.utils.data.DataLoader(test_data,batch_size=64)
    validate_loader=torch.utils.data.DataLoader(validate_data,batch_size=64)
    
    epochs=int(args.epochs)
    lr=float(args.learning_rate)
    gpu=args.gpu
    arch=args.arch
    hidden_unit=int(args.hidden_unit)
    
    
    if arch=='vgg16':
        model=models.vgg16(pretrained=True)
        classifier=nn.Sequential(nn.Linear(25088 ,hidden_unit),
                                 nn.ReLU(),
                                 nn.Dropout(0.4),
                                 nn.Linear(hidden_unit,102),
                                 nn.LogSoftmax(dim=1))
        
    else:
        model=models.densenet121(pretrained=True)
        classifier=nn.Sequential(nn.Linear(1024,hidden_unit),
                                 nn.ReLU(),
                                 nn.Dropout(0.4),
                                 nn.Linear(hidden_unit,102),
                                 nn.LogSoftmax(dim=1))
    
    

    for param in model.parameters():
        param.requires_grad=False

    
    model.classifier=classifier

    criterion=nn.NLLLoss()
    optimizer=optim.Adam(model.classifier.parameters(),lr)
    train(model,train_loader,validate_loader,epochs,gpu,optimizer,criterion)
    print("Training Over")
    model.class_to_idx=train_data.class_to_idx
    checkpoint_path=args.save_dir
    save_checkpoint(checkpoint_path,model,optimizer,arch,hidden_unit)
    
    
if __name__=="__main__":
    main()
                     