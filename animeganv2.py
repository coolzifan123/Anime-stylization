from data_loader import ImageFolder,MergeDataset,MultiRandomSampler
from torch.utils.data import TensorDataset,DataLoader
from Network import AnimeGenerator,AnimeDiscriminator,Vgg19
from configv2 import config
from smooth import edge_promoting
from utils import rgb2yuv,variation_loss

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
        self.batch_size = config.batch_size
        self.cartoon_path = config.cartoon_root
        self.smooth_path = config.cartoon_root+'_smooth'
        self.photo_path = config.photo_root
        self.test_path = config.test_root
        self.model_path = config.model_path
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.img_size = config.img_size
        self.save_interval = config.save_interval

        self.con_weight=config.con_weight
        self.sty_weight=config.sty_weight
        self.color_weight=config.color_weight
        self.tv_weight=config.tv_weight
        self.g_adv_weight=config.g_adv_weight
        self.d_adv_weight=config.d_adv_weight
        self.train_epoch = config.train_epoch
        self.pretrain_epoch = config.pretrain_epoch
        data_mean=[-0.97881186, -1.9799739,   2.9587858 ]
        if not config.is_smooth:
            print('smoothing images---')
            edge_promoting(self.cartoon_path, self.smooth_path,cat=False)
            print('smoothing is done---')
        self.train_real_transform = transforms.Compose([
            transforms.Resize((config.img_size, config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        self.train_anime_transform = transforms.Compose([
            transforms.ToTensor() ,
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        self.train_gray_transform = transforms.Compose([
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
        train_real = ImageFolder(self.photo_path, transform=self.train_real_transform)
        train_anime = ImageFolder(self.cartoon_path,
                                  transform=self.train_anime_transform)
        train_anime_gray = ImageFolder(self.cartoon_path,
                                       transform=self.train_gray_transform)
        train_anime = TensorDataset(train_anime, train_anime_gray)
        train_smooth_gray = ImageFolder(self.smooth_path,
                                        transform=self.train_gray_transform)
        self.ds_train = MergeDataset(train_real, train_anime, train_smooth_gray)
        self.train_dataloader=DataLoader(self.ds_train,
                                        sampler=MultiRandomSampler(self.ds_train),
                                        batch_size=self.batch_size,
                                        num_workers=4,
                                        pin_memory=True)
        self.photo_loader=DataLoader(train_real,batch_size=config.batch_size,
                                                      shuffle=True,num_workers=4,drop_last=False)
        self.generator=AnimeGenerator().to(device)
        self.discriminator=AnimeDiscriminator().to(device)
        self.vgg=Vgg19().to(device)
        self.l1_loss=nn.L1Loss()
        self.huber_loss=nn.SmoothL1Loss()
        self.G_optim = optim.Adam(self.generator.parameters(), lr=config.lrG, betas=(config.beta1, config.beta2))
        self.D_optim = optim.Adam(self.discriminator.parameters(), lr=config.lrD, betas=(config.beta1, config.beta2))
        self.G_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=self.G_optim,
                                                          milestones=[config.train_epoch // 2,
                                                                      config.train_epoch // 4 * 3],
                                                          gamma=0.1)
        self.D_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=self.D_optim,
                                                          milestones=[config.train_epoch // 2,
                                                                      config.train_epoch // 4 * 3],
                                                          gamma=0.1)

        self.generator.train()
        self.discriminator.train()
        self.vgg.eval()

    def pretrain(self):
        print('Pre-training start!')
        pre_optim=optim.Adam(self.generator.parameters(), lr=0.0002, betas=(config.beta1, config.beta2))
        for epoch in range(1, self.pretrain_epoch + 1):
            pretrain_loss_epoch = []
            print('pre-training %d epoch' % (epoch))
            for img in tqdm(self.photo_loader):
                img = img.to(device)
                self.G_optim.zero_grad()
                feature = self.vgg((img + 1) / 2)
                G_out = self.generator(img)
                G_feature = self.vgg((G_out + 1) / 2)
                c_loss = 1.5 * self.l1_loss(G_feature, feature)
                pretrain_loss_epoch.append(c_loss.item())
                c_loss.backward()
                pre_optim.step()

            avg_loss = sum(pretrain_loss_epoch) / len(pretrain_loss_epoch)
            print('pretrain: train loss: {:.2f}'.format(avg_loss))
            if ((epoch) % config.save_interval == 0) or (epoch == self.pretrain_epoch):
                print('Saving checkpoint')
                torch.save(self.generator.state_dict(),
                           os.path.join(self.model_path, 'v2_pretrain_' + 'epoch' + str(epoch) + '.pth'))

    def train(self):
        print('training start')
        for epoch in range(1, self.train_epoch + 1):
            self.generator.train()
            dis_loss_epoch = []
            d_real_loss_epoch=[]
            d_fake_loss_epoch=[]
            d_blur_loss_epoch=[]
            gen_loss_epoch = []
            con_loss_epoch = []
            color_loss_epoch=[]
            style_loss_epoch=[]
            print('training %d epoch' % (epoch))
            for batch in tqdm(self.train_dataloader):
                input_photo, (input_cartoon, anime_gray_data), anime_smooth_gray_data = batch
                input_photo=input_photo.to(device)
                input_cartoon=input_cartoon.to(device)
                anime_gray_data=anime_gray_data.to(device)
                anime_smooth_gray_data=anime_smooth_gray_data.to(device)

                # train discriminator
                self.D_optim.zero_grad()

                generated = self.generator(input_photo)
                anime_logit = self.discriminator(input_cartoon)
                anime_gray_logit = self.discriminator(anime_gray_data)
                generated_logit = self.discriminator(generated)
                smooth_logit = self.discriminator(anime_smooth_gray_data)

                (d_real_loss, d_gray_loss, d_fake_loss, d_real_blur_loss) = self.discriminator_loss(
                    anime_logit, anime_gray_logit,
                    generated_logit, smooth_logit)
                d_real_loss=self.d_adv_weight*d_real_loss
                d_gray_loss = self.d_adv_weight * d_gray_loss
                d_fake_loss = self.d_adv_weight * d_fake_loss
                d_real_blur_loss = self.d_adv_weight * d_real_blur_loss
                d_loss_total = d_real_loss + d_fake_loss + d_gray_loss + d_real_blur_loss

                dis_loss_epoch.append(d_loss_total.item())
                d_blur_loss_epoch.append(d_real_blur_loss.item())
                d_fake_loss_epoch.append(d_fake_loss.item())
                # d_gray_loss_epoch.append(d_gray_loss.item())
                d_real_loss_epoch.append(d_real_loss.item())

                d_loss_total.backward()
                self.D_optim.step()

                # train generator
                self.G_optim.zero_grad()
                generated = self.generator(input_photo)
                generated_logit = self.discriminator(generated)
                c_loss, s_loss = self.con_sty_loss(input_photo,
                                                   anime_gray_data,
                                                   generated)
                c_loss=self.con_weight*c_loss
                s_loss=self.sty_weight*s_loss
                tv_loss=self.tv_weight*variation_loss(generated)
                col_loss = self.color_loss(input_photo, generated) * self.color_weight
                g_loss=self.g_adv_weight*self._g_loss(generated_logit)
                g_loss_total = c_loss + s_loss + col_loss + g_loss + tv_loss

                gen_loss_epoch.append(g_loss_total.item())
                con_loss_epoch.append(c_loss.item())
                color_loss_epoch.append(col_loss.item())
                style_loss_epoch.append(s_loss.item())

                g_loss_total.backward()
                self.G_optim.step()

            self.G_scheduler.step()
            self.D_scheduler.step()
            avg_dis_loss = sum(dis_loss_epoch) / len(dis_loss_epoch)
            avg_d_blur_loss=sum(d_blur_loss_epoch)/len(d_blur_loss_epoch)
            avg_d_real_loss=sum(d_real_loss_epoch)/len(d_real_loss_epoch)
            avg_d_fake_loss=sum(d_fake_loss_epoch)/len(d_fake_loss_epoch)
            avg_gen_loss = sum(gen_loss_epoch) / len(gen_loss_epoch)
            avg_con_loss = sum(con_loss_epoch) / len(con_loss_epoch)
            avg_color_loss=sum(color_loss_epoch)/len(color_loss_epoch)
            avg_sty_loss=sum(style_loss_epoch)/len(style_loss_epoch)
            print(
                'epoch: %d, Discriminator loss: %.3f, Generator loss: %.3f, Content loss: %.3f' % (
                    (epoch), avg_dis_loss, avg_gen_loss, avg_con_loss))
            with SummaryWriter(log_dir='logs',comment='train') as w:
                w.add_scalar('dis/blur_loss',avg_d_blur_loss,epoch)
                w.add_scalar('dis/real_loss',avg_d_real_loss,epoch)
                w.add_scalar('dis/fake_loss',avg_d_fake_loss,epoch)
                w.add_scalar('dis/dis_loss',avg_dis_loss,epoch)
                w.add_scalar('gen/content_loss',avg_con_loss,epoch)
                w.add_scalar('gen/style_loss',avg_sty_loss,epoch)
                w.add_scalar('gen/color_loss',avg_color_loss,epoch)
            if (epoch % self.save_interval == 0) or (epoch == self.train_epoch):
                print('saving checkpoint')
                torch.save(self.generator.state_dict(),
                           os.path.join(self.model_path,'v2'+'gen_'+'epoch'+str(epoch)+'.pt'))
                torch.save(self.discriminator.state_dict(),
                           os.path.join(self.model_path, 'v2'+'dis_' + 'epoch' + str(epoch) + '.pt'))
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
                        out_img=torch.cat((img[0],(G_out[0]+1)/2),dim=2)

                        save_image(out_img,'result/'+i[-6:])


    def discriminator_loss(self, real, gray, fake, real_blur):
        real_loss = torch.mean(torch.square(real - 1.0))
        gray_loss = torch.mean(torch.square(gray))
        fake_loss = torch.mean(torch.square(fake))
        real_blur_loss = torch.mean(torch.square(real_blur))
        return 1.2 * real_loss, 1.2 * gray_loss, 1.2 * fake_loss, 0.8 * real_blur_loss

    def gram(self, x):
        b, c, h, w = x.shape
        gram = torch.einsum('bchw,bdhw->bcd', x, x)
        return gram / (c * h * w)

    def style_loss(self, style, fake):
        return self.l1_loss(self.gram(style), self.gram(fake))

    def con_sty_loss(self, real, anime, fake):
        real_feature_map = self.vgg(real)
        fake_feature_map = self.vgg(fake)
        anime_feature_map = self.vgg(anime)

        c_loss = self.l1_loss(real_feature_map, fake_feature_map)
        s_loss = self.style_loss(anime_feature_map, fake_feature_map)

        return c_loss, s_loss

    def color_loss(self, con, fake):
        con = rgb2yuv((con+1)/2)
        fake = rgb2yuv((fake+1)/2)
        return (self.l1_loss(con[..., 0], fake[..., 0]) +
                self.huber_loss(con[..., 1], fake[..., 1]) +
                self.huber_loss(con[..., 2], fake[..., 2]))

    def _g_loss(self, fake_logit):
        # 1/2 * (fake-c)^2
        return torch.mean((fake_logit - 1) ** 2)






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
