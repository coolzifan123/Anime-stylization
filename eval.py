from Network import generator,Vgg19,discriminator,AnimeGenerator
from config import config
from smooth import edge_promoting
from data_loader import Dataset,photo_dataset

from PIL import Image
import glob
from tqdm import tqdm
import torch
import torchvision.transforms as transforms

from torchvision.utils import save_image
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os


G=generator()
G.load_state_dict(torch.load('models/hayaogen_epoch100.pth',map_location='cpu'))
if not os.path.exists('result'):
    os.mkdir('result')
#img_list = list(sorted(glob.glob('dataset1/test/test_photo256' + '/*.*')))
img_list = list(sorted(glob.glob('test' + '/*.*')))
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

    img0 = Image.open(i).convert("RGB")
    img_init=init_transform(img0)
    img = photo_transform(img0)
    img=torch.unsqueeze(img,dim=0)

    G_out=G(img)
    #out_img=torch.cat((img_init,(G_out[0]+1)/2),dim=2)
    out_img=(G_out[0]+1)/2
    save_image(out_img,'result/'+'c'+str(num)+'.png')
    save_image(img_init,'result/'+'img'+str(num)+'.png')
    num+=1