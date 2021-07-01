from Network import generator,Vgg19,discriminator
from config import config
from smooth import edge_promoting
from data_loader import Dataset,photo_dataset

from torchvision.utils import save_image
from PIL import Image
import glob
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os

if not os.path.exists('result'):
    os.mkdir('result')
device=torch.device('cuda')
class trainer(object):
    def __init__(self,config):
        self.batch_size=config.batch_size
        self.cartoon_path = config.cartoon_root
        self.smooth_path = config.cartoon_root+'_smooth'
        self.photo_path=config.photo_root
        self.test_path=config.test_root
        self.model_path=config.model_path
        self.img_size=config.img_size
        self.content_coe=config.content_coe
        self.save_interval=config.save_interval
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        self.BCE_loss=nn.BCELoss()
        self.L1_loss=nn.L1Loss()
        self.generator=generator().to(device)
        self.discriminator=discriminator().to(device)
        self.vgg=Vgg19().to(device)
        self.generator.train()
        self.discriminator.train()
        self.vgg.eval()
        self.train_epoch=config.train_epoch
        self.pretrain_epoch=config.pretrain_epoch
        self.G_optim=optim.Adam(self.generator.parameters(),lr=config.lrG,betas=(config.beta1,config.beta2))
        self.D_optim=optim.Adam(self.discriminator.parameters(),lr=config.lrD,betas=(config.beta1,config.beta2))
        self.G_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=self.G_optim,
                                                     milestones=[config.train_epoch // 2, config.train_epoch // 4 * 3],
                                                     gamma=0.1)
        self.D_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=self.D_optim,
                                                     milestones=[config.train_epoch // 2, config.train_epoch // 4 * 3],
                                                     gamma=0.1)


        if not config.is_smooth:
            print('smoothing images---')
            edge_promoting(self.cartoon_path, self.smooth_path)
            print('smoothing is done---')

        photo_transform = transforms.Compose([
            transforms.Resize((config.img_size, config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        cartoon_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        photo_data=photo_dataset(photo_path=self.photo_path,transform=photo_transform)
        data=Dataset(photo_path=self.photo_path,cartoon_path=self.smooth_path,transform1=photo_transform,
                     transform2=cartoon_transform)
        # photo_data=Dataset(path=self.photo_path,transform=photo_transform)
        # cartoon_data=Dataset(path=self.smooth_path,transform=cartoon_transform)
        self.data_loader=torch.utils.data.DataLoader(data,batch_size=config.batch_size,
                                                      shuffle=True,num_workers=4,drop_last=True)
        self.photo_loader=torch.utils.data.DataLoader(photo_data,batch_size=config.batch_size,
                                                      shuffle=True,num_workers=4,drop_last=False)



    def pretrain(self):
        print('Pre-training start!')
        for epoch in range(1, self.pretrain_epoch + 1):
            pretrain_loss_epoch = []
            print('pre-training %d epoch' % (epoch))
            for img in tqdm(self.photo_loader):
                img = img.to(device)
                self.G_optim.zero_grad()
                feature = self.vgg((img + 1) / 2)
                G_out = self.generator(img)
                G_feature = self.vgg((G_out + 1) / 2)
                c_loss = 10 * self.L1_loss(G_feature, feature)
                pretrain_loss_epoch.append(c_loss.item())
                c_loss.backward()
                self.G_optim.step()

            avg_loss = sum(pretrain_loss_epoch) / len(pretrain_loss_epoch)
            print('pretrain: train loss: {:.2f}'.format(avg_loss))
            if ((epoch) % config.save_interval == 0) or (epoch == self.pretrain_epoch):
                print('Saving checkpoint')
                torch.save(self.generator.state_dict(),
                           os.path.join(self.model_path, 'pretrain_' + 'epoch' + str(epoch) + '.pth'))

    def train(self):
        real = torch.ones(self.batch_size, 1, self.img_size // 4, self.img_size // 4).to(device)
        fake = torch.zeros(self.batch_size, 1, self.img_size // 4, self.img_size // 4).to(device)
        print('training start')
        for epoch in range(1,self.train_epoch+1):
            self.generator.train()
            dis_loss_epoch=[]
            gen_loss_epoch=[]
            con_loss_epoch=[]
            print('training %d epoch' % (epoch))
            for src, tgt in tqdm(self.data_loader):
                smooth_img=tgt[:,:,:,self.img_size:]
                cartoon_img=tgt[:,:,:,:self.img_size]
                smooth_img=smooth_img.to(device)
                cartoon_img=cartoon_img.to(device)
                src=src.to(device)

                # train discriminator
                self.D_optim.zero_grad()

                D_real=self.discriminator(cartoon_img)
                D_real_loss=self.BCE_loss(D_real,real)

                G_out=self.generator(src)
                D_fake=self.discriminator(G_out)
                D_fake_loss=self.BCE_loss(D_fake,fake)

                D_edge=self.discriminator(smooth_img)
                D_edge_loss=self.BCE_loss(D_edge,fake)

                dis_loss=D_real_loss + D_fake_loss + D_edge_loss
                dis_loss_epoch.append(dis_loss.item())

                dis_loss.backward()
                self.D_optim.step()

                # train generator
                self.G_optim.zero_grad()
                G_out=self.generator(src)
                D_fake=self.discriminator(G_out)
                D_fake_loss=self.BCE_loss(D_fake,real)

                src_feature=self.vgg((src+1)/2)
                G_feature=self.vgg((G_out+1)/2)
                con_loss=self.content_coe * self.L1_loss(G_feature,src_feature.detach())

                gen_loss=D_fake_loss+con_loss
                gen_loss_epoch.append(gen_loss)
                con_loss_epoch.append(con_loss)

                gen_loss.backward()
                self.G_optim.step()


            self.G_scheduler.step()
            self.D_scheduler.step()
            avg_dis_loss=sum(dis_loss_epoch)/len(dis_loss_epoch)
            avg_gen_loss=sum(gen_loss_epoch)/len(gen_loss_epoch)
            avg_con_loss=sum(con_loss_epoch)/len(con_loss_epoch)
            print(
                'epoch: %d, Discriminator loss: %.3f, Generator loss: %.3f, Content loss: %.3f' % (
                (epoch), avg_dis_loss,avg_gen_loss, avg_con_loss))
            with SummaryWriter(log_dir='logs',comment='train') as W:
                W.add_scalar('train/con_loss',avg_con_loss,epoch)
                W.add_scalar('train/dis_loss',avg_dis_loss,epoch)
                W.add_scalar('train/gen_loss',avg_gen_loss,epoch)
            if (epoch % self.save_interval == 0) or (epoch == self.train_epoch):
                print('saving checkpoint')
                torch.save(self.generator.state_dict(),
                           os.path.join(self.model_path,self.style+'gen_'+'epoch'+str(epoch)+'.pth'))
                torch.save(self.discriminator.state_dict(),
                           os.path.join(self.model_path, self.style+'dis_' + 'epoch' + str(epoch) + '.pth'))
                #with SummaryWriter(log_dir='logs',comment='eval') as W:
                with torch.no_grad():
                    self.generator.eval()
                    imgs = []
                    img_list = list(sorted(glob.glob(self.test_path + '/*.*')))
                    photo_transform = transforms.Compose([
                        transforms.Resize((config.img_size, config.img_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                    ])
                    for i in img_list:
                        img = Image.open(i).convert("RGB")
                        img = photo_transform(img)
                        img=torch.unsqueeze(img,dim=0).to(device)

                        G_out=self.generator(img)
                        out_img=torch.cat((img[0],G_out[0]),dim=2)

                        save_image(out_img,'result/'+i[-6:])




def main():
    agent=trainer(config)
    if config.generator_model == None:
        agent.pretrain()
    else:
        agent.generator.load_state_dict(torch.load(config.generator_model))
    if config.discriminator_model != None:
        agent.discriminator.load_state_dict(torch.load(config.discriminator_model))
    agent.train()
if __name__=='__main__':
    main()