from Network import generator,Vgg19,discriminator,AnimeGenerator
from config import config
from smooth import edge_promoting
from data_loader import Dataset,photo_dataset

import argparse
from PIL import Image
import glob
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from adjust_brightness import adjust_brightness_from_src_to_dst
from utils import custom_blur_demo
from torchvision.utils import save_image
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import cv2
import numpy as np

def imsave(images, path):
    return cv2.imwrite(path, cv2.cvtColor(images, cv2.COLOR_RGB2BGR))
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='models/pytorch_generator_hayao.pt',
                    help='path to model')
parser.add_argument('--test', type=str, default='test',
                    help='path to the test folder')
args = parser.parse_args()
G=AnimeGenerator()
G.load_state_dict(torch.load(args.model,map_location='cpu'))
if not os.path.exists('result'):
    os.mkdir('result')
#img_list = list(sorted(glob.glob('dataset1/test/test_photo256' + '/*.*')))
img_list = list(sorted(glob.glob(args.test + '/*.*')))
photo_transform = transforms.Compose([
    transforms.Resize((config.img_size, config.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
init_transform=transforms.Compose([
    transforms.Resize((config.img_size, config.img_size)),
    transforms.ToTensor()
])
num=1
for i in tqdm(img_list):
    with torch.no_grad():
        img0 = Image.open(i).convert("RGB")
        img_init = init_transform(img0)
        img = photo_transform(img0)
        img = torch.unsqueeze(img, dim=0)

        G_out = G(img)
        G_out = (G_out[0] + 1) / 2
        G_out = G_out.numpy() * 255
        img_init = img_init.numpy() * 255
        G_out = G_out.astype(np.uint8)
        img_init = img_init.astype(np.uint8)
        G_out = G_out.astype(np.uint8)
        G_out = G_out.transpose(1, 2, 0)
        img_init = img_init.transpose(1, 2, 0)
        # cv2.imshow('image', img_init)
        # cv2.imshow('G',G_out)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        imsave(adjust_brightness_from_src_to_dst(G_out, img_init), 'result/' + str(num) + '.png')
        # out_img=torch.cat((img_init,(G_out[0]+1)/2),dim=2)

        # save_image((G_out+1)/2,'result/'+str(num)+'.png')
        num += 1
