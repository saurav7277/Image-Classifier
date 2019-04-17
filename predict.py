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
from utils import load_checkpoint,cat_to_names

def parse_arg():
    parser=argparse.ArgumentParser(description="Predicting From Network")
    parser.add_argument('--checkpoint',action='store',default='checkpoint.pth')
    parser.add_argument('--topk',dest='topk',default='1')
    parser.add_argument('--img_path',dest='img_path',default='flowers/test/1/image_06752.jpg')
    parser.add_argument('--gpu',action='store',default='gpu')
    parser.add_argument('--category_names',dest='category_names',default='cat_to_name.json')

    return parser.parse_args()

def process_image(image_path):
    img_pil = Image.open(image_path)

    transformation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    img_tensor = transformation(img_pil)

    return img_tensor

def predict(image_path,model,topk,gpu):

    if gpu=='gpu':
        model=model.to('cuda')
    else:
        model=model.to('cpu')
    image=process_image(image_path)
    image=image.unsqueeze_(0)
    image=image.float()
    if gpu=='gpu':
        image=image.to('cuda')

    else:
        image=image.to('cpu')
    model.eval()
    with torch.no_grad():
        logits=model.forward(image)
        ps=torch.exp(logits)

        return ps.topk(topk,dim=1)

def main():
    args=parse_arg()
    model=load_checkpoint(args.checkpoint)
    cat_to_name=cat_to_names(args.category_names)

    probabilities=predict(args.img_path,model,int(args.topk),args.gpu)
    top_prob=np.array(probabilities[0][0].cpu())
    top_class_name=[cat_to_name[str(index+1)] for index in np.array(probabilities[1][0].cpu())]


    print("Image You selected For Prediction: " +args.img_path)
    print(top_prob)
    print(top_class_name)


if __name__=="__main__":
    main()
